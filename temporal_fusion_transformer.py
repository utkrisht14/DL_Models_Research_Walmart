# Load the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from graphs_plot import compare_model_graph, log_model_plot_overall_preds
from train import denormalize

import wandb
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Read the dataframe
df = pd.read_csv("walmart_dataset.csv", index_col="Date", parse_dates=True)

# First define Gated Residual Network for the internal layer
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, context_size=None, dropout_rate=0.1):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(output_size)
        self.gate = nn.Linear(output_size, output_size)

        # Transform residual to match hidden size
        self.residual_transform = nn.Linear(input_size, hidden_size)

        if context_size is not None:
            self.context_layer = nn.Linear(context_size, hidden_size)
        else:
            self.context_layer = None

    def forward(self, x, context=None):
        if self.context_layer is not None and context is not None:
            context_output = self.context_layer(context)
            x = x + context_output

        residual = self.residual_transform(x)  # Transform residual to match hidden size
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # Gated residual connection
        x = torch.sigmoid(self.gate(x)) * x + residual
        x = self.layer_norm(x)

        return x


# Next to define Variable selection network for dynamic inputs. The steps are same as in paper
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_vars):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)  # Softmax for variable selection
        self.num_vars = num_vars  # Number of time-varying features

        # Linear layer to compute selection weights for features
        self.fc_selection = nn.Linear(input_size, num_vars)

        # GRN applied after the feature selection
        self.grn = GatedResidualNetwork(num_vars, hidden_size, hidden_size)

    def forward(self, x):
        # x shape is: [batch_size, time_steps, input_size]

        # Step 1: Compute selection weights based on the input features
        selection_weights = self.fc_selection(x.mean(dim=1))  # Average over time steps to get selection weights
        selection_weights = self.softmax(selection_weights)  # Output shape: [batch_size, num_vars]

        # Step 2: Reshape selection weights to apply them to each time step
        selection_weights = selection_weights.unsqueeze(1)  # [batch_size, 1, num_vars] to broadcast over time steps

        # Step 3: Apply selection weights to the feature dimension
        selected_input = x[:, :, :self.num_vars] * selection_weights  # Apply selection to the first `num_vars` features

        # Step 4: Apply GRN to the selected inputs
        transformed_input = self.grn(selected_input)  # Output shape: [batch_size, time_steps, hidden_size]

        return transformed_input, selection_weights


# LSTM Encoder and Decoder for the sequential data
class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first= True)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)

# Temporal self-attention layer
class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)  # Multi-head attention
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # self.self_attn returns (attn_output, attn_weights)
        # We only want attention output
        attn_output, _ = self.self_attn(x, x, x)  # Key, Query, and Values are all "x"

        # Apply dropout and residual connection
        output = self.dropout(attn_output) + x

        # Apply layer normalization
        output = self.layer_norm(output)
        return output


# Static Covariate encoder for static features
class StaticCovariateEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate= 0.1):
        super().__init__()
        self.grn = GatedResidualNetwork(input_size, hidden_size, hidden_size) # Use GRN for encoding static covariates

    def forward(self, static_covariates):
        return self.grn(static_covariates) # Encode static covariates through GRN


# Single output forecasting Layer i.e. no quantile in this case.
# In paper there are quantiles for 10%, 50%, 90% etc. for multiple outputs.
class SingleOutputForecastLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc(x)


# Now define the Temporal Fusion Transformer for the Time-Series Forecasting
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_vars, num_heads, dropout_rate=0.1):
        super().__init__()

        # Static covariate enrichment using GRN
        self.static_covariate_grn = StaticCovariateEncoder(input_size, hidden_size, dropout_rate)

        # LSTM encoder for past inputs
        self.lstm_encoder = LSTMEncoderDecoder(input_size, hidden_size)

        # LSTM decoder for future inputs
        self.lstm_decoder = LSTMEncoderDecoder(input_size, hidden_size)

        # Temporal self-attention
        self.attn_layer = TemporalSelfAttention(hidden_size, num_heads, dropout_rate)

        # Variable selection network for past inputs
        self.variable_selection_network = VariableSelectionNetwork(hidden_size, hidden_size, num_vars)

        # Forecasting a single output (1 day prediction)
        self.single_output_dense = SingleOutputForecastLayer(hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, static_covariates, time_varying_past, time_varying_future):
        # Static covariates
        static_embedding = self.static_covariate_grn(static_covariates)

        # LSTM Encoder (past data)
        encoded_past, _ = self.lstm_encoder(time_varying_past)

        # Variable selection for past inputs
        selected_vars_past, _ = self.variable_selection_network(encoded_past)

        # Apply self-attention over past inputs
        attn_output = self.attn_layer(selected_vars_past)

        # LSTM Decoder (future inputs)
        decoded_future, (h_n, c_n) = self.lstm_decoder(time_varying_future)

        # Combine with static covariates and the last hidden state of the decoder
        combined_output = attn_output[:, -1, :] + static_embedding + h_n[-1]  # Get the last hidden state

        # Forecast a single output (1 day prediction)
        single_output = self.single_output_dense(combined_output)

        # Add an extra dimension to make the output shape [batch_size, 1]
        return single_output  # This will make the output [32, 1]


# Here I need to separately define the custom dataset here. Cause there are many changes from the original custom dataset.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, label_column, static_columns, time_varying_columns, input_window_size, prediction_window_size):
        super().__init__()
        self.data = data
        self.label_column = label_column
        self.static_columns = static_columns
        self.time_varying_columns = time_varying_columns
        self.input_window_size = input_window_size
        self.prediction_window_size = prediction_window_size

        # Separate features and labels
        self.features = self.data.drop(columns=[self.label_column]).values
        self.labels = self.data[self.label_column].values

        # Split static and time-varying features
        self.static_features = self.data[self.static_columns].values
        self.time_varying_features = self.data[self.time_varying_columns].values

        # Normalize the features and the labels
        self.static_features, self.static_min, self.static_max = self.minmax_normalization(self.static_features)
        self.time_varying_features, self.time_min, self.time_max = self.minmax_normalization(self.time_varying_features)
        self.labels, self.labels_min, self.labels_max = self.minmax_normalization(self.labels.reshape(-1, 1))

    def __len__(self):
        return len(self.data) - self.input_window_size - self.prediction_window_size + 1

    def __getitem__(self, idx):
        # Extract input window for time-varying features
        time_varying_past = self.time_varying_features[idx: idx + self.input_window_size]
        time_varying_future = self.time_varying_features[idx + self.input_window_size: idx + self.input_window_size + self.prediction_window_size]

        # Extract static covariates
        static_covariates = self.static_features[idx]

        # Extract the target (label)
        y = self.labels[idx + self.input_window_size: idx + self.input_window_size + self.prediction_window_size]

        # For TFT model, return static covariates, time-varying past, time-varying future, and target
        return torch.tensor(static_covariates, dtype=torch.float32), \
               torch.tensor(time_varying_past, dtype=torch.float32), \
               torch.tensor(time_varying_future, dtype=torch.float32),\
               torch.tensor(y, dtype=torch.float32)

    def minmax_normalization(self, data):
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        epsilon = 1e-8
        normalized_data = (data - min_val) / (max_val - min_val + epsilon)
        return normalized_data, min_val, max_val


# Prepare the dataloaders according to the TFT model
def prepare_dataloaders(df, label_column, input_window_size, prediction_window_size, batch_size, static_columns=None, time_varying_columns=None):
    test_size = int(0.2 * len(df))

    # Split dataset into train and test sets
    train_set = df[:-test_size]
    test_set = df[-test_size:]

    # Initialize datasets with static and time-varying columns
    train_dataset = CustomDataset(
        train_set, label_column=label_column, input_window_size=input_window_size,
        prediction_window_size=prediction_window_size, static_columns=static_columns, time_varying_columns=time_varying_columns
    )
    test_dataset = CustomDataset(
        test_set, label_column=label_column, input_window_size=input_window_size,
        prediction_window_size=prediction_window_size, static_columns=static_columns, time_varying_columns=time_varying_columns
    )

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataset, test_dataset, train_dataloader, test_dataloader



# Now prepare the data
# Define static and time-varying columns. This need to be done manually.
static_columns = ["CPI", "Unemployment_Rate", "CCI", "GPD_Growth_Rate", "day_before_weekend", "day_before_holiday", "day_after_holiday", "day_after_weekend"]
time_varying_columns = ["Open", "High", "Low", "Volume", "RSI", "VIX", "Oil_Price", "Gold_Price"]


# Define the training function separately, because there are few changes because of static and time_varying_avriables
def train_evaluate_model(model, model_name, optimizer, criterion, train_dataloader, test_dataloader,
                         train_dataset, test_dataset, epochs=50, lr=1e-3, input_window_size=20, scheduler=None,
                         device=None, model_type=None, patience=25):
    """
    Train and evaluate the model for Temporal Transformer Fusion

    Args:
    - model: The model to train.
    - model_name: Name of the model for logging in wandb.
    - optimizer: Optimizer for the model.
    - criterion: Loss function
    - train_dataloader: Dataloader for training data.
    - test_dataloader: Dataloader for testing data.
    - train_dataset: Dataset for normalization metrics.
    - test_dataset: Dataset for normalization metrics.
    - epochs: Number of training epochs.
    - lr: Learning rate.
    - input_window_size: Input sequence window size.
    - scheduler: learning rate scheduler (default is None). Could be provided from paper "Attention....."
    - device: Device to run the model on
    - model_type: The type of the model (used for input/output formatting).
    - patience: Number of epochs to wait before stopping if no improvement in test loss.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize a new W&B run for each model
    wandb.init(project="Walmart_Research", name=model_name)

    start_time = time.time()

    overall_preds = []
    overall_targets = []

    # Early Stopping variables
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss = 0
        model.train()

        # Training loop
        for batch in train_dataloader:
            # Unpack the batch to get static covariates, time-varying past, time-varying future, and target
            static_covariates, time_varying_past, time_varying_future, y_train = batch
            static_covariates, time_varying_past, time_varying_future, y_train = static_covariates.to(device), time_varying_past.to(device), time_varying_future.to(device), y_train.to(device)

            # Forward pass for TFT model
            y_pred = model(static_covariates, time_varying_past, time_varying_future)

            # Predictions and targets should have the same shape
            if y_pred.shape != y_train.shape:
                y_train = y_train.view_as(y_pred)

            # Compute the loss
            loss = criterion(y_pred, y_train)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        # Log train loss to wandb
        wandb.log({"Train Loss": train_loss})
        print(f"Epoch: {epoch + 1} | Train Loss: {train_loss}")

        # Evaluate on test data
        model.eval()
        test_loss = 0
        epoch_preds = []
        epoch_targets = []

        # Evaluation loop
        with torch.inference_mode():
            for batch in test_dataloader:
                # Unpack the batch to get static covariates, time-varying past, time-varying future, and target
                static_covariates, time_varying_past, time_varying_future, y_test = batch
                static_covariates, time_varying_past, time_varying_future, y_test = static_covariates.to(device), time_varying_past.to(device), time_varying_future.to(device), y_test.to(device)

                # Forward pass
                y_val = model(static_covariates, time_varying_past, time_varying_future)

                # Predictions and targets should have the same shape. (Added after debugging)
                if y_val.shape != y_test.shape:
                    y_test = y_test.view_as(y_val)

                # Compute the test loss
                test_loss += criterion(y_val, y_test).item()

                # Collect predictions and targets
                epoch_preds.extend(y_val.detach().cpu().numpy())
                epoch_targets.extend(y_test.detach().cpu().numpy())

        mae = mean_absolute_error(epoch_preds, epoch_targets)
        mse = mean_squared_error(epoch_preds, epoch_targets)

        wandb.log({"Test Loss": test_loss / len(test_dataloader), "MAE": mae, "MSE": mse})
        print(f"Epoch {epoch + 1} | Test Loss: {test_loss / len(test_dataloader)} | MAE: {mae} | MSE: {mse} ")

        overall_preds.extend(epoch_preds)
        overall_targets.extend(epoch_targets)

        # Early Stopping and Best Model Saving
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{model_name}_best_model.pth")  # Save best model to disk
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Load the best weights after training ends
    model.load_state_dict(torch.load(f"{model_name}_best_model.pth"))

    # Convert lists to numpy arrays
    overall_preds = np.array(overall_preds)
    overall_targets = np.array(overall_targets)

    # Metrics on normalized data
    overall_mae_norm = mean_absolute_error(overall_preds, overall_targets)
    overall_mse_norm = mean_squared_error(overall_preds, overall_targets)
    r_square_norm = r2_score(overall_targets, overall_preds)

    # Denormalize predictions and targets using test_dataset min/max values
    overall_preds_denorm = denormalize(overall_preds, test_dataset.labels_min, test_dataset.labels_max)
    overall_targets_denorm = denormalize(overall_targets, test_dataset.labels_min, test_dataset.labels_max)

    # Metrics on denormalized data
    overall_mae_denorm = mean_absolute_error(overall_preds_denorm, overall_targets_denorm)
    overall_mse_denorm = mean_squared_error(overall_preds_denorm, overall_targets_denorm)
    r_square_denorm = r2_score(overall_targets_denorm, overall_preds_denorm)

    # Log both normalized and denormalized metrics
    wandb.log({"Overall Test MAE (Norm)": overall_mae_norm,
               "Overall Test MSE (Norm)": overall_mse_norm,
               "Overall Test MAE (Denorm)": overall_mae_denorm,
               "Overall Test MSE (Denorm)": overall_mse_denorm,
               "R-square (Denorm)": r_square_denorm})

    print(f"\nOverall Test MAE (Norm): {overall_mae_norm}")
    print(f"Overall Test MSE (Norm): {overall_mse_norm}")
    print(f"Overall Test MAE (Denorm): {overall_mae_denorm}")
    print(f"Overall Test MSE (Denorm): {overall_mse_denorm}")
    print(f"R-square (Denorm): {r_square_denorm} \n")

    end_time = time.time()
    print(f"Total time taken: {(end_time - start_time) / 60} minutes.\n")
    wandb.log({"Running Time": end_time - start_time})

    return overall_preds_denorm, overall_targets_denorm, overall_preds, overall_targets



# Define some hyperparameters
input_window_size = 7  # Input window size for past time steps (7 days)
num_time_varying_features = len(time_varying_columns)  # Number of time-varying features (past + future)
num_static_covariates = len(static_columns)  # Number of static covariates
hidden_size = 64  # Hidden size
num_attention_heads = 4  # Number of attention heads in self-attention
dropout_rate = 0.2  # Dropout rate for regularization
num_epochs= 100
learning_rate = 1e-3  # Going with default learning rate

# Number of variables to select (for variable selection networks)
num_vars = num_time_varying_features

# Initialize the Temporal Fusion Transformer model
tft_model_7 = TemporalFusionTransformer(
    input_size=num_static_covariates,
    hidden_size=hidden_size,
    num_vars=num_time_varying_features,
    num_heads=num_attention_heads,
    dropout_rate=dropout_rate
)

# Move the model to GPU
tft_model_7 = tft_model_7.to(device)

# Define optimizer, loss function, and learning rate
optimizer_7 = torch.optim.Adam(tft_model_7.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.MSELoss()

# Prepare data loaders with static and time-varying columns for TFT model
train_dataset_7, test_dataset_7, train_dataloader_7, test_dataloader_7 = prepare_dataloaders(
    df, label_column="Adj Close Target", static_columns=static_columns, time_varying_columns=time_varying_columns,
    input_window_size=input_window_size, prediction_window_size=1, batch_size=32
)

# Track the log on wandb
wandb.init(project="Walmart_TFT", name="Model TFT 7 Days Window")  # Start a new W&B run

# Train and evaluate the TFT model for the widnow period of 7 days.
overall_preds_denorm_7, overall_targets_denorm_7, overall_preds_7, overall_targets_7 = train_evaluate_model(
    model=tft_model_7,
    model_name="TFT 7-Days Window",
    optimizer=optimizer_7,
    criterion=criterion,
    train_dataloader=train_dataloader_7,
    test_dataloader=test_dataloader_7,
    train_dataset=train_dataset_7,
    test_dataset=test_dataset_7,
    epochs= num_epochs,
    lr=learning_rate,
    input_window_size=input_window_size,
    device=device,
    model_type="tft"
)
wandb.finish()

# Define the new window size for 20 days
input_window_size = 20

# Now prepare data loaders with static and time-varying columns for TFT model with 20 days input window
train_dataset_20, test_dataset_20, train_dataloader_20, test_dataloader_20 = prepare_dataloaders(
    df, label_column="Adj Close Target", static_columns=static_columns, time_varying_columns=time_varying_columns,
    input_window_size=input_window_size, prediction_window_size=1, batch_size=32
)

# Define the model for the 20 days input window
tft_model_20 = TemporalFusionTransformer(
    input_size=num_static_covariates,
    hidden_size=hidden_size,
    num_vars=num_time_varying_features,
    num_heads=num_attention_heads,
    dropout_rate=dropout_rate
)

# Move the model to GPU
tft_model_20 = tft_model_20.to(device)

# Define the optimizer for 20 days
optimizer_20 = torch.optim.Adam(tft_model_20.parameters(), lr=learning_rate, weight_decay=1e-5)


# Track the log on wandb
wandb.init(project="Walmart_TFT", name="Model TFT 20 Days Window")  # Start a new W&B run

# Train the model
overall_preds_denorm_20, overall_targets_denorm_20, overall_preds_20, overall_targets_20 = train_evaluate_model(
    model=tft_model_20,
    model_name="TFT 20-Days Window",
    optimizer=optimizer_20,
    criterion=criterion,
    train_dataloader=train_dataloader_20,
    test_dataloader=test_dataloader_20,
    train_dataset=train_dataset_20,
    test_dataset=test_dataset_20,
    epochs=num_epochs,
    lr=learning_rate,
    input_window_size=input_window_size,
    device=device,
    model_type="tft"
)
wandb.finish()

# Define the new window size for 50 days
input_window_size = 50

# Now prepare data loaders with static and time-varying columns for TFT model with 50 days input window
train_dataset_50, test_dataset_50, train_dataloader_50, test_dataloader_50 = prepare_dataloaders(
    df, label_column="Adj Close Target", static_columns=static_columns, time_varying_columns=time_varying_columns,
    input_window_size=input_window_size, prediction_window_size=1, batch_size=32
)

# Define the model for the 50 days input window
tft_model_50 = TemporalFusionTransformer(
    input_size=num_static_covariates,
    hidden_size=hidden_size,
    num_vars=num_time_varying_features,
    num_heads=num_attention_heads,
    dropout_rate=dropout_rate
)

# Move the model to GPU
tft_model_50 = tft_model_50.to(device)

# Define the optimizer for 50 days
optimizer_50 = torch.optim.Adam(tft_model_50.parameters(), lr=learning_rate, weight_decay=1e-5)

# Track the log on wandb
wandb.init(project="Walmart_TFT", name="Model TFT 50 Days Window")  # Start a new W&B run

# Train the model
overall_preds_denorm_50, overall_targets_denorm_50, overall_preds_50, overall_targets_50 = train_evaluate_model(
    model=tft_model_50,
    model_name="TFT 50-Days Window",
    optimizer=optimizer_50,
    criterion=criterion,
    train_dataloader=train_dataloader_50,
    test_dataloader=test_dataloader_50,
    train_dataset=train_dataset_50,
    test_dataset=test_dataset_50,
    epochs=num_epochs,
    lr=learning_rate,
    input_window_size=input_window_size,
    device=device,
    model_type="tft"
)
wandb.finish()

# Define the new window size for 100 days
input_window_size = 100

# Now prepare data loaders with static and time-varying columns for TFT model with 100 days input window
train_dataset_100, test_dataset_100, train_dataloader_100, test_dataloader_100 = prepare_dataloaders(
    df, label_column="Adj Close Target", static_columns=static_columns, time_varying_columns=time_varying_columns,
    input_window_size=input_window_size, prediction_window_size=1, batch_size=32
)

# Define the model for the 100 days input window
tft_model_100 = TemporalFusionTransformer(
    input_size=num_static_covariates,
    hidden_size=hidden_size,
    num_vars=num_time_varying_features,
    num_heads=num_attention_heads,
    dropout_rate=dropout_rate
)

# Move the model to GPU if available
tft_model_100 = tft_model_100.to(device)

# Define the optimizer for 100 days
optimizer_100 = torch.optim.Adam(tft_model_100.parameters(), lr=learning_rate, weight_decay=1e-5)

# Track the log on wandb
wandb.init(project="Walmart_TFT", name="Model TFT 100 Days Window")  # Start a new W&B run

# Train the model
overall_preds_denorm_100, overall_targets_denorm_100, overall_preds_100, overall_targets_100 = train_evaluate_model(
    model=tft_model_100,
    model_name="TFT 100-Days Window",
    optimizer=optimizer_100,
    criterion=criterion,
    train_dataloader=train_dataloader_100,
    test_dataloader=test_dataloader_100,
    train_dataset=train_dataset_100,
    test_dataset=test_dataset_100,
    epochs=num_epochs,
    lr=learning_rate,
    input_window_size=input_window_size,
    device=device,
    model_type="tft"
)
wandb.finish()

# Next plot the graph for the various models to compare
last_lookback_days = 100 # We plot the graph to see 100 days of predictions vs 100 days of actual data

# Plot for 7 days window period
log_model_plot_overall_preds("TFT Model 7 Days", overall_targets_denorm_7, overall_preds_denorm_7,
                             last_lookback_days= last_lookback_days)

# Plot for 20 days window period
log_model_plot_overall_preds("TFT Model 20 Days", overall_targets_denorm_20, overall_preds_denorm_20,
                             last_lookback_days= last_lookback_days)

# Plot for 50 days window period
log_model_plot_overall_preds("TFT Model 50 Days", overall_targets_denorm_50, overall_preds_denorm_50,
                             last_lookback_days= last_lookback_days)

# Plot for 100 days window period
log_model_plot_overall_preds("TFT Model 100 Days", overall_targets_denorm_100, overall_preds_denorm_100,
                             last_lookback_days= last_lookback_days)

# Compare the graph of all the data
compare_model_graph(overall_targets_denorm_7, overall_preds_denorm_7, overall_preds_denorm_20, overall_preds_denorm_50, overall_preds_denorm_100, last_lookback_days=last_lookback_days)

# Save the plot locally
plt.savefig("tft_comparison_plot.png")


# Save the predictions in a DataFrame and log them
# Find the minimum length across all arrays to manage mismatch
min_length = min(len(overall_targets_denorm_7),
                 len(overall_preds_denorm_7),
                 len(overall_preds_denorm_20),
                 len(overall_preds_denorm_50),
                 len(overall_preds_denorm_100))

# Truncate all arrays to the same length
predictions_df_tft = pd.DataFrame({
    "Actual": overall_targets_denorm_7[:min_length].ravel(),
    "Transformer_7_Lookback": overall_preds_denorm_7[:min_length].ravel(),
    "Transformer_20_Lookback": overall_preds_denorm_20[:min_length].ravel(),
    "Transformer_50_Lookback": overall_preds_denorm_50[:min_length].ravel(),
    "Transformer_100_Lookback": overall_preds_denorm_100[:min_length].ravel(),
})

# Log the DataFrame as a table to W&B
wandb.init(project="Walmart_Research_Data", name="Transformer Predictions Table")
wandb.log({"TFT Predictions Table": wandb.Table(dataframe=predictions_df_tft)})

# Save the DataFrame as a CSV file for local use
predictions_df_tft.to_csv("tft_predictions.csv", index=True)

# End W&B run
wandb.finish()





