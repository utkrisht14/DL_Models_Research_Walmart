import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from custom_dataset import prepare_dataloaders
from train import train_evaluate_model
from graphs_plot import plot_graph, compare_model_graph, log_model_plot_overall_preds

import wandb


df = pd.read_csv("walmart_dataset.csv", index_col="Date", parse_dates= True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Define the Temporal Block.
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
        super().__init__()

        # Calculate padding to ensure the output sequence length is the same as the input
        padding = (kernel_size - 1) * dilation // 2

        # First dilated causal convolution with weight normalization
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                                                    padding=padding, dilation=dilation))
        self.relu = nn.ReLU()  # Activation function after the first convolution
        self.dropout = nn.Dropout(dropout)

        # Second 1D convolution layer
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))

        # Stack operations in sequence: conv1 -> ReLU -> Dropout -> conv2 -> ReLU -> Dropout
        self.net = nn.Sequential(self.conv1, self.relu, self.dropout, self.conv2, self.relu, self.dropout)

        # Downsample if the input and output channel dimensions are different for residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        # Pass input through the convolution layers
        out = self.net(x)

        # If input and output have different sizes, downsample input before adding residual
        res = x if self.downsample is None else self.downsample(x)

        # Add the residual and apply ReLU activation
        return self.relu(out + res)

# Define the TCN class
class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()

        layers = []
        num_levels = len(num_channels)  # Number of temporal blocks (levels)

        # Loop to create each temporal block with increasing dilation and stack them
        for i in range(num_levels):
            dilation_size = 2 ** i  # Dilation increases exponentially [1, 2, 4, 8,...] as mentioned in paper
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # Input channels for the first block
            out_channels = num_channels[i]  # Output channels for each block

            # Add each TemporalBlock to the layers list
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     dropout=dropout)]

        # Stack the layers into a sequential module
        self.network = nn.Sequential(*layers)

        # Linear layer that maps the output to the target value
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # Permute input shape: (batch_size, seq_len, features) -> (batch_size, features, seq_len)
        x = x.permute(0, 2, 1)

        # Pass input through the temporal blocks
        out = self.network(x)

        # Select the output from the last time step for each batch
        out = out[:, :, -1]

        return self.linear(out)

# Define the variables
input_size = df.shape[1] - 1  # Number of input features
num_channels = [128, 128, 128]  # Number of output channels for each temporal block
kernel_size = 3  # Kernel size for the 1d convolutional layer
dropout = 0.2  # Dropout rate for regularization
num_epochs = 60  # Number of training epochs
learning_rate = 1e-5  # Learning rate

# Initialize the TCN model
model_TCN_7 = TCN(num_inputs=input_size, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer_7 = torch.optim.Adam(model_TCN_7.parameters(), weight_decay=1e-5, lr=learning_rate)

# Prepare the data loaders with a window size of 7
train_dataset_7, test_dataset_7, train_dataloader_7, test_dataloader_7 = prepare_dataloaders(df, label_column="Adj Close Target", input_window_size=7, prediction_window_size=1, batch_size=32)

# Train the model for a 7-day window period using the TCN model
wandb.init(project="Walmart_TCN", name="Model TCN 7 Days Window")  # Start a new Weights & Biases run
overall_preds_denorm_TCN_7, overall_targets_denorm_TCN_7, overall_preds_TCN_7, overall_targets_TCN_7 = train_evaluate_model(
    model=model_TCN_7,
    model_name="Model TCN 7 Days Window",
    train_dataloader=train_dataloader_7,
    test_dataloader=test_dataloader_7,
    criterion=criterion,
    train_dataset=train_dataset_7,
    test_dataset=test_dataset_7,
    optimizer=optimizer_7,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=7)

wandb.finish()  # End the W&B run

# Now define for the 20 days sequence window
# Initialize the TCN model
model_TCN_20 = TCN(num_inputs=input_size, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout).to(device)

# Set the lower learning rate and numbers of epochs to avoid NaN values
learning_rate = 1e-6
num_epochs = 35

# Define the optimizer for the 20 days
optimizer_20 = torch.optim.Adam(model_TCN_20.parameters(), weight_decay=1e-5, lr=learning_rate)

# Prepare the data loaders with a window size of 20
train_dataset_20, test_dataset_20, train_dataloader_20, test_dataloader_20 = prepare_dataloaders(df, label_column="Adj Close Target", input_window_size=20, prediction_window_size=1, batch_size=32)

# Train the model for a 20-day window period using the TCN model
wandb.init(project="Walmart_TCN", name="Model TCN 20 Days Window")  # Start a new Weights & Biases run
overall_preds_denorm_TCN_20, overall_targets_denorm_TCN_20, overall_preds_TCN_20, overall_targets_TCN_20 = train_evaluate_model(
    model=model_TCN_20,
    model_name="Model TCN 20 Days Window",
    train_dataloader=train_dataloader_20,
    test_dataloader=test_dataloader_20,
    criterion=criterion,
    train_dataset=train_dataset_20,
    test_dataset=test_dataset_20,
    optimizer=optimizer_20,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=20)

wandb.finish()  # End the W&B run

# Now define for the 50 days sequence window
# Initialize the TCN model
model_TCN_50 = TCN(num_inputs=input_size, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout).to(device)

# Define the optimizer for the 50 days
optimizer_50 = torch.optim.Adam(model_TCN_50.parameters(), weight_decay=1e-5, lr=learning_rate)

# Prepare the data loaders with a window size of 50
train_dataset_50, test_dataset_50, train_dataloader_50, test_dataloader_50 = prepare_dataloaders(df, label_column="Adj Close Target", input_window_size=50, prediction_window_size=1, batch_size=32)

# Train the model for a 50-day window period using the TCN model
wandb.init(project="Walmart_TCN", name="Model TCN 50 Days Window")  # Start a new Weights & Biases run
overall_preds_denorm_TCN_50, overall_targets_denorm_TCN_50, overall_preds_TCN_50, overall_targets_TCN_50 = train_evaluate_model(
    model=model_TCN_50,
    model_name="Model TCN 50 Days Window",
    train_dataloader=train_dataloader_50,
    test_dataloader=test_dataloader_50,
    criterion=criterion,
    train_dataset=train_dataset_50,
    test_dataset=test_dataset_50,
    optimizer=optimizer_50,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=50)

wandb.finish()  # End the W&B run

# Now define for the 100 days sequence window
# Initialize the TCN model
model_TCN_100 = TCN(num_inputs=input_size, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout).to(device)

# Define the optimizer for the 100 days
optimizer_100 = torch.optim.Adam(model_TCN_100.parameters(), weight_decay=1e-5, lr=learning_rate)

# Prepare the data loaders with a window size of 100
train_dataset_100, test_dataset_100, train_dataloader_100, test_dataloader_100 = prepare_dataloaders(df, label_column="Adj Close Target", input_window_size=100, prediction_window_size=1, batch_size=32)

# Train the model for a 100-day window period using the TCN model
wandb.init(project="Walmart_TCN", name="Model TCN 100 Days Window")  # Start a new Weights & Biases run
overall_preds_denorm_TCN_100, overall_targets_denorm_TCN_100, overall_preds_TCN_100, overall_targets_TCN_100 = train_evaluate_model(
    model=model_TCN_100,
    model_name="Model TCN 100 Days Window",
    train_dataloader=train_dataloader_100,
    test_dataloader=test_dataloader_100,
    criterion=criterion,
    train_dataset=train_dataset_100,
    test_dataset=test_dataset_100,
    optimizer=optimizer_100,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=100)

wandb.finish()  # End the W&B run

# Next we plot the graph for the various models to compare
last_lookback_days = 100 # We plot the graph to see 100 days of predictions vs 100 days of actual data

# Plot for 7 days window period
log_model_plot_overall_preds("TCN Model 7 Days", overall_targets_denorm_TCN_7, overall_preds_denorm_TCN_7,
                             last_lookback_days= last_lookback_days)

# Plot for 20 days window period
log_model_plot_overall_preds("TCN Model 20 Days", overall_targets_denorm_TCN_20, overall_preds_denorm_TCN_20,
                             last_lookback_days= last_lookback_days)

# Plot for 50 days window period
log_model_plot_overall_preds("TCN Model 50 Days", overall_targets_denorm_TCN_50, overall_preds_denorm_TCN_50,
                             last_lookback_days= last_lookback_days)

# Plot for 100 days window period
log_model_plot_overall_preds("TCN Model 100 Days", overall_targets_denorm_TCN_100, overall_preds_denorm_TCN_100,
                             last_lookback_days= last_lookback_days)

# Compare the graph of all the data
compare_model_graph(overall_targets_denorm_TCN_7, overall_preds_denorm_TCN_7,overall_preds_denorm_TCN_20, overall_preds_denorm_TCN_50, overall_preds_denorm_TCN_100, last_lookback_days=last_lookback_days)

# Save the plot locally
plt.savefig("TCN_comparison_plot.png")


# Save the predictions in a DataFrame and log them
# Find the minimum length across all arrays to manage mismatch
min_length = min(len(overall_targets_denorm_TCN_7),
                 len(overall_preds_denorm_TCN_7),
                 len(overall_preds_denorm_TCN_20),
                 len(overall_preds_denorm_TCN_50),
                 len(overall_preds_denorm_TCN_100))

# Truncate all arrays to the same length
predictions_df_tcn = pd.DataFrame({
    "Actual": overall_targets_denorm_TCN_7[:min_length].ravel(),
    "Transformer_7_Lookback": overall_preds_denorm_TCN_7[:min_length].ravel(),
    "Transformer_20_Lookback": overall_preds_denorm_TCN_20[:min_length].ravel(),
    "Transformer_50_Lookback": overall_preds_denorm_TCN_50[:min_length].ravel(),
    "Transformer_100_Lookback": overall_preds_denorm_TCN_100[:min_length].ravel(),
})

# Log the DataFrame as a table to W&B
wandb.init(project="Walmart_Research_Data", name="TCN Predictions Table")
wandb.log({"TCN Predictions Table": wandb.Table(dataframe=predictions_df_tcn)})

# Save the DataFrame as a CSV file for local use
predictions_df_tcn.to_csv("tcn_predictions.csv", index=True)

# End W&B run
wandb.finish()
