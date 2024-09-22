import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from custom_dataset import prepare_dataloaders
from train import train_evaluate_model
from graphs_plot import compare_model_graph, log_model_plot_overall_preds

import wandb

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Read the dataframe
df = pd.read_csv("walmart_dataset.csv", index_col="Date", parse_dates=True)

# Polynomial trend basis for the trend block
def polynomial_trend_basis(backcast_length, forecast_length, degree=2, device="cpu"):
    # Create a time vector "t" with values evenly spaced between -1 and 1 for both backcast and forecast periods
    t = torch.linspace(-1, 1, backcast_length + forecast_length).to(device)

    # Split "t" into the backcast part (for the history data)
    t_backcast = t[:backcast_length]

    # Split "t" into the forecast part (for future predictions)
    t_forecast = t[backcast_length:]

    # Create a polynomial basis for the backcast by raising "t_backcast" to increasing powers up to "degree"
    backcast_basis = torch.stack([t_backcast ** d for d in range(degree + 1)], dim=1)

    # Similarly, create a polynomial basis for the forecast by raising "t_forecast" to increasing powers up to "degree"
    forecast_basis = torch.stack([t_forecast ** d for d in range(degree + 1)], dim=1)

    # Return the polynomial bases for both backcast and forecast
    return backcast_basis, forecast_basis


# Fourier Series for seasonality blocks
def fourierseasonality_basis(backcast_length, forecast_length, harmonics=10, device="cpu"):
    # Create a time vector "t" with values evenly spaced between 0 and 2π for both backcast and forecast periods
    t = torch.linspace(0, 2 * np.pi, backcast_length + forecast_length).to(device)

    # Split "t" into the backcast part (for the history data)
    t_backcast = t[:backcast_length]

    # Split "t" into the forecast part (for future predictions)
    t_forecast = t[backcast_length:]

    # Create sine terms for the Fourier series in the backcast window for each harmonic as defined in paper
    backcast_basis = [torch.sin(h * t_backcast) for h in range(1, harmonics + 1)]

    # Create cosine terms for the Fourier series in the backcast window for each harmonic and add to the basis
    backcast_basis += [torch.cos(h * t_backcast) for h in range(1, harmonics + 1)]

    # Create sine terms for the Fourier series in the forecast window for each harmonic
    forecast_basis = [torch.sin(h * t_forecast) for h in range(1, harmonics + 1)]

    # Create cosine terms for the Fourier series in the forecast window for each harmonic and add to the basis
    forecast_basis += [torch.cos(h * t_forecast) for h in range(1, harmonics + 1)]

    # Stack both backcast and forecast bases along a new dimension to form tensors and return them
    return torch.stack(backcast_basis, dim=1), torch.stack(forecast_basis, dim=1)


# Trend Block
class NBeatsTrendBlock(nn.Module):
    def __init__(self, input_window, num_features, hidden_size, forecast_length, num_layers, degree=2):
        super().__init__()
        self.backcast_length = input_window
        self.forecast_length = forecast_length
        self.degree = degree

        input_size = input_window * num_features
        self.fc_stack = nn.ModuleList([nn.Linear(input_size, hidden_size) if i == 0 else nn.Linear(hidden_size, hidden_size) for i in range(num_layers)])
        self.fc_output = nn.Linear(hidden_size, degree + 1)

    def forward(self, x):
        batch_size = x.size(0)

        # Flatten the input to (batch_size, input_window * num_features)
        x = x.view(batch_size, -1)

        # Pass through the fully connected layers
        for i, layer in enumerate(self.fc_stack):
            x = torch.relu(layer(x))

        trend_params = self.fc_output(x)  # Output trend params

        # Generate polynomial basis for backcast and forecast
        t_backcast, t_forecast = polynomial_trend_basis(self.backcast_length, self.forecast_length, self.degree, x.device)

        # Compute backcast and forecast using matrix multiplication
        backcast = torch.matmul(t_backcast.unsqueeze(0), trend_params.unsqueeze(-1)).squeeze(-1)
        forecast = torch.matmul(t_forecast.unsqueeze(0), trend_params.unsqueeze(-1)).squeeze(-1)

        # We need to take care of two shapes: backcast matches input window, forecast predicts one day
        backcast = backcast.unsqueeze(-1).expand(-1, self.backcast_length, 17)  # [batch_size, 7, 17]
        forecast = forecast.unsqueeze(-1).expand(-1, self.forecast_length, 1)   # [batch_size, 1, 1]

        return backcast, forecast

# Seasonality Block
class NBeatsSeasonailtyBlock(nn.Module):
    def __init__(self, input_window, num_features, hidden_size, forecast_length, num_layers, harmonics=10):
        super().__init__()
        # Define the length of the backcast (input window) and forecast (output window)
        self.backcast_length = input_window
        self.forecast_length = forecast_length
        self.harmonics = harmonics  # Number of harmonics for the Fourier series

        # Compute the input size: input window * number of features
        input_size = input_window * num_features

        # Create a list of fully connected layers for the block
        # The first layer takes the input size, the rest take the hidden size
        self.fc_stack = nn.ModuleList(
            [nn.Linear(input_size, hidden_size) if i == 0 else nn.Linear(hidden_size, hidden_size) for i in
             range(num_layers)])

        # The output layer produces seasonality parameters (2 * harmonics for sine and cosine components)
        self.fc_output = nn.Linear(hidden_size, 2 * harmonics)

    def forward(self, x):
        # Get the batch size from the input tensor
        batch_size = x.size(0)

        # Flatten the input to have shape (batch_size, input_window * num_features)
        x = x.view(batch_size, -1)

        # Pass the input through each fully connected layer with ReLU activation
        for i, layer in enumerate(self.fc_stack):
            x = torch.relu(layer(x))

        # Generate seasonality parameters using the final fully connected layer
        seasonality_params = self.fc_output(x)

        # Generate the Fourier basis for both backcast and forecast using the harmonics
        t_backcast, t_forecast = fourierseasonality_basis(self.backcast_length, self.forecast_length, self.harmonics,
                                                          x.device)

        # Compute the backcast by multiplying the Fourier basis with the seasonality parameters
        backcast = torch.matmul(t_backcast.unsqueeze(0), seasonality_params.unsqueeze(-1)).squeeze(-1)

        # Compute the forecast by multiplying the Fourier basis with the seasonality parameters
        forecast = torch.matmul(t_forecast.unsqueeze(0), seasonality_params.unsqueeze(-1)).squeeze(-1)

        # Reshape the backcast to match the input window size and number of features
        backcast = backcast.unsqueeze(-1).expand(-1, self.backcast_length, 17)  # [batch_size, 7, 17]

        # Reshape the forecast to match the forecast length and output one feature
        forecast = forecast.unsqueeze(-1).expand(-1, self.forecast_length, 1)  # [batch_size, 1, 1]

        # Return the backcast and forecast components
        return backcast, forecast


# Model
class NBeatsModel(nn.Module):
    def __init__(self, input_window, num_features, hidden_size, num_blocks, num_layers, forecast_length, harmonics=10, degree=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        # Add trend and seasonality blocks to the model
        self.blocks.append(NBeatsTrendBlock(input_window, num_features, hidden_size, forecast_length, num_layers, degree))
        self.blocks.append(NBeatsSeasonailtyBlock(input_window, num_features, hidden_size, forecast_length, num_layers, harmonics))

    def forward(self, x):
        # Initialize forecast to shape [batch_size, forecast_length, num_features]
        forecast_shape = (x.size(0), 1, 1)  # Predicting 1 day, 1 feature
        forecast = torch.zeros(forecast_shape, device=x.device)

        # Use backcast as the input
        backcast = x.clone()

        # Iterate through blocks (both trend and seasonality)
        for block in self.blocks:
            block_backcast, block_forecast = block(backcast)
            backcast = backcast - block_backcast  # Subtract block_backcast from the backcast for residual learning
            forecast = forecast + block_forecast  # Add block_forecast to the forecast

        # Forecast matches the expected shape of [batch_size, forecast_length]
        return forecast.squeeze(-1)

# Define the model parameters
input_window = 7  # Length of the backcast window
num_features = df.shape[1] - 1  # Number of features; exclude the target column
hidden_size = 256  # Size of the hidden layers
num_blocks = 4  # Number of blocks (trend, seasonality)
num_layers = 3  # Number of fully connected layers per block
forecast_length = 1  # Length of the forecast (predicting 1 day)
harmonics = 10  # Number of harmonics for the Fourier series
degree = 2  # Polynomial degree for the trend block
learning_rate = 1e-4  # Learning rate
num_epochs = 100  # Number of epochs to train


# Dataset preparation function
train_dataset_7, test_dataset_7, train_dataloader_7, test_dataloader_7 = prepare_dataloaders(df, label_column="Adj Close Target", input_window_size=input_window,
                                                                                             prediction_window_size=1, batch_size=32, model_type="nbeats")

# Initialize the model
model_7 = NBeatsModel(
    input_window=input_window,
    num_features=num_features,
    hidden_size=hidden_size,
    num_blocks=num_blocks,
    num_layers=num_layers,
    forecast_length=forecast_length,
    harmonics=harmonics,
    degree=degree
)

# Send the model to the GPU
model_7 = model_7.to(device)

# Define the optimizer and loss function
optimizer_7 = torch.optim.Adam(model_7.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.MSELoss()

wandb.init(project="Walmart_Research_Ind", name="Model N-Beats 7 Days Window") # Start a new W&B run

# Call the training and evaluation function
overall_preds_denorm_7, overall_targets_denorm_7, overall_preds_7, overall_targets_7 = train_evaluate_model(
    model=model_7,
    model_name="Model N-Beats 7-Days Window",
    optimizer=optimizer_7,
    criterion=criterion,
    train_dataloader=train_dataloader_7,
    test_dataloader=test_dataloader_7,
    train_dataset=train_dataset_7,
    test_dataset=test_dataset_7,
    epochs=num_epochs,
    input_window_size=input_window,
    model_type= "nbeats"
)

wandb.finish()  # End the W&B run for this model

# Now check it for the 20 days window
input_window = 20  # Length of the backcast window (20 days here)

# Dataset preparation function
train_dataset_20, test_dataset_20, train_dataloader_20, test_dataloader_20 = prepare_dataloaders(df, label_column="Adj Close Target", input_window_size=input_window,
                                                                                             prediction_window_size=1, batch_size=32, model_type="nbeats")


# Initialize the model
model_20 = NBeatsModel(
    input_window=input_window,
    num_features=num_features,
    hidden_size=hidden_size,
    num_blocks=num_blocks,
    num_layers=num_layers,
    forecast_length=forecast_length,
    harmonics=harmonics,
    degree=degree
)

# Send the model to the GPU
model_20 = model_20.to(device)

# Define the optimizer and loss function
optimizer_20 = torch.optim.Adam(model_20.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.MSELoss()

wandb.init(project="Walmart_Research_Ind", name="Model N-Beats 20 Days Window") # Start a new W&B run

# Call the training and evaluation function
overall_preds_denorm_20, overall_targets_denorm_20, overall_preds_20, overall_targets_20 = train_evaluate_model(
    model=model_20,
    model_name="Model N-Beats 20-Days Window",
    optimizer=optimizer_20,
    criterion=criterion,
    train_dataloader=train_dataloader_20,
    test_dataloader=test_dataloader_20,
    train_dataset=train_dataset_20,
    test_dataset=test_dataset_20,
    epochs=num_epochs,
    input_window_size=input_window,
    model_type= "nbeats"
)

wandb.finish()  # End the W&B run for this model

# Define the model parameters
input_window = 50  # Length of the backcast window (50 days here)

# Dataset preparation function
train_dataset_50, test_dataset_50, train_dataloader_50, test_dataloader_50 = prepare_dataloaders(df, label_column="Adj Close Target", input_window_size=input_window,
                                                                                             prediction_window_size=1, batch_size=32, model_type="nbeats")

# Initialize the model
model_50 = NBeatsModel(
    input_window=input_window,
    num_features=num_features,
    hidden_size=hidden_size,
    num_blocks=num_blocks,
    num_layers=num_layers,
    forecast_length=forecast_length,
    harmonics=harmonics,
    degree=degree
)

# Send the model to the GPU
model_50 = model_50.to(device)

# Define the optimizer and loss function
optimizer_50 = torch.optim.Adam(model_50.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.MSELoss()

wandb.init(project="Walmart_Research_Ind", name="Model N-Beats 50 Days Window") # Start a new W&B run

# Call the training and evaluation function for 50 days
overall_preds_denorm_50, overall_targets_denorm_50, overall_preds_50, overall_targets_50 = train_evaluate_model(
    model=model_50,
    model_name="Model N-Beats 50-Days Window",
    optimizer=optimizer_50,
    criterion=criterion,
    train_dataloader=train_dataloader_50,
    test_dataloader=test_dataloader_50,
    train_dataset=train_dataset_50,
    test_dataset=test_dataset_50,
    epochs=num_epochs,
    input_window_size=input_window,
    model_type= "nbeats"
)

wandb.finish()  # End the W&B run for this model

# Now for the 100 days window size
input_window = 100  # Length of the backcast window (100 days here)

# Dataset preparation function
train_dataset_100, test_dataset_100, train_dataloader_100, test_dataloader_100 = prepare_dataloaders(df, label_column="Adj Close Target", input_window_size=input_window,
                                                                                             prediction_window_size=1, batch_size=32, model_type="nbeats")

# Initialize the model
model_100 = NBeatsModel(
    input_window=input_window,
    num_features=num_features,
    hidden_size=hidden_size,
    num_blocks=num_blocks,
    num_layers=num_layers,
    forecast_length=forecast_length,
    harmonics=harmonics,
    degree=degree
)

# Send the model to the GPU
model_100 = model_100.to(device)

# Define the optimizer and loss function
optimizer_100 = torch.optim.Adam(model_100.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.MSELoss()

wandb.init(project="Walmart_Research_Ind", name="Model N-Beats 100 Days Window") # Start a new W&B run

# Call the training and evaluation function
overall_preds_denorm_100, overall_targets_denorm_100, overall_preds_100, overall_targets_100 = train_evaluate_model(
    model=model_100,
    model_name="Model N-Beats 100-Days Window",
    optimizer=optimizer_100,
    criterion=criterion,
    train_dataloader=train_dataloader_100,
    test_dataloader=test_dataloader_100,
    train_dataset=train_dataset_100,
    test_dataset=test_dataset_100,
    epochs=num_epochs,
    input_window_size=input_window,
    model_type= "nbeats"
)

wandb.finish()  # End the W&B run for this model

# Next we plot the graph for the various models to compare
last_lookback_days = 100 # We plot the graph to see 100 days of predictions vs 100 days of actual data

# Plot for 7 days window period
log_model_plot_overall_preds("N-Beats Model 7 Days", overall_targets_denorm_7, overall_preds_denorm_7,
                             last_lookback_days)

# Plot for 20 days window period
log_model_plot_overall_preds("N-BEATS Model 20 Days", overall_targets_denorm_20, overall_preds_denorm_20,
                             last_lookback_days)

# Plot for 50 days window period
log_model_plot_overall_preds("N-BEATS Model 50 Days", overall_targets_denorm_50, overall_preds_denorm_50,
                             last_lookback_days)

# Plot for 100 days window period
log_model_plot_overall_preds("N-BEATS Model 100 Days", overall_targets_denorm_100, overall_preds_denorm_100,
                             last_lookback_days)

# Log the combined comparison plot
wandb.init(project="Walmart_Research_Plots", name="N-BEATS Model Comparison")
compare_model_graph(overall_preds_denorm_7, overall_preds_denorm_7, overall_preds_denorm_20,
                    overall_preds_denorm_50, overall_preds_denorm_100, last_lookback_days)
plt.savefig("comparison_plot.png")  # Save the plot locally
wandb.log({"Comparison of N-BEATS Models": wandb.Image("comparison_plot.png")})  # Log the plot to W&B
wandb.finish()

# Save the predictions in a DataFrame and log them
# Find the minimum length across all arrays
min_length = min(len(overall_targets_denorm_7),
                 len(overall_preds_denorm_7),
                 len(overall_preds_denorm_20),
                 len(overall_preds_denorm_50),
                 len(overall_preds_denorm_100))

# Truncate all arrays to the same length
predictions_df_N_Beats = pd.DataFrame({
    "Actual": overall_targets_denorm_7[:min_length].ravel(),
    "N-BEATS_7_Lookback": overall_preds_denorm_7[:min_length].ravel(),
    "N-BEATS_20_Lookback": overall_preds_denorm_20[:min_length].ravel(),
    "N-BEATS_50_Lookback": overall_preds_denorm_50[:min_length].ravel(),
    "N-BEATS_100_Lookback": overall_preds_denorm_100[:min_length].ravel()
})

# Log the DataFrame as a table to W&B
wandb.init(project="Walmart_Research_Data", name="N_BEATS Predictions Table")
wandb.log({"N-BEATS Predictions Table": wandb.Table(dataframe=predictions_df_N_Beats)})

# Save the DataFrame as a CSV file
predictions_df_N_Beats.to_csv("N_Beats_predictions.csv", index=True)

# End W&B run
wandb.finish()

