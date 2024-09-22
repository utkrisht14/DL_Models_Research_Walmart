import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from custom_dataset import prepare_dataloaders
from train import train_evaluate_model
from graphs_plot import plot_graph, compare_model_graph, log_model_plot_overall_preds

import wandb
import time

df = pd.read_csv("walmart_dataset.csv", index_col="Date", parse_dates= True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# I have already tried with different configuration of GRU. So, I'll use GRU single layer with no bi-directional
# Define the GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.GRU = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through the GRU
        out, (h_n) = self.GRU(x, h0)

        # Output shape handling
        out = self.linear(out[:, -1, :])  # For prediction based on the last time step
        return out

# Define the variables
input_size = df.shape[1] - 1
output_size = 1
hidden_size = 64
batch_first = True
num_layers = 1
num_epochs = 50
learning_rate = 1e-4

model_GRU_7 = GRUModel(input_size, hidden_size, num_layers, batch_first, output_size).to(device)

# Define the loss function and the optimizers
criterion = nn.MSELoss()
optimizer_7 = torch.optim.Adam(model_GRU_7.parameters(), weight_decay=1e-5, lr= learning_rate)

# I tried four window size. 7, 20, 50, and 100
# Start with window size of 7
train_dataset_7, test_dataset_7, train_dataloader_7, test_dataloader_7 = prepare_dataloaders(df, label_column="Adj Close Target", input_window_size=7, prediction_window_size=1, batch_size=32)

# Train the model for 7 days window period
wandb.init(project="Walmart_GRU", name="Model GRU 7 Days Window")  # Start a new W&B run
overall_preds_denorm_GRU_7, overall_targets_denorm_GRU_7, overall_preds_GRU_7, overall_targets_GRU_7 = train_evaluate_model(
        model= model_GRU_7,
    model_name="Model GRU 7 Days Window",
    train_dataloader= train_dataloader_7,
    test_dataloader= test_dataloader_7,
    criterion= criterion,
    train_dataset=train_dataset_7,
    test_dataset = test_dataset_7,
    optimizer= optimizer_7,
    lr= learning_rate,
    epochs=num_epochs,
    input_window_size= 7)
wandb.finish()  # End the W&B run for this model

# Next we define the window size of 20
train_dataset_20, test_dataset_20, train_dataloader_20, test_dataloader_20 = prepare_dataloaders(df, label_column="Adj Close Target", input_window_size=20, prediction_window_size=1, batch_size=32)

# Initialize the model for the window size 20
model_GRU_20 = GRUModel(input_size, hidden_size, num_layers, batch_first, output_size).to(device)

# Define the optimizers for window size 20
optimizer_20 = torch.optim.Adam(model_GRU_20.parameters(), weight_decay=1e-5, lr= learning_rate)

# Train for 20 days window period
wandb.init(project="Walmart_GRU", name="Model GRU 20 Days Window")  # Start a new W&B run
overall_preds_denorm_GRU_20, overall_targets_denorm_GRU_20, overall_preds_GRU_20, overall_targets_GRU_20 = train_evaluate_model(
        model= model_GRU_20,
        model_name="Model GRU 20 Days Window",
        train_dataloader= train_dataloader_20,
        test_dataloader= test_dataloader_20,
        train_dataset=train_dataset_20,
        test_dataset = test_dataset_20,
        criterion= criterion,
        optimizer= optimizer_20,
        lr=learning_rate,
        epochs=num_epochs,
        input_window_size= 20)
wandb.finish()

# Next we define the window size of 50
train_dataset_50, test_dataset_50,train_dataloader_50, test_dataloader_50 = prepare_dataloaders(df, label_column="Adj Close Target", input_window_size=50, prediction_window_size=1, batch_size=32)

# Initialize the model for the window size 50
model_GRU_50 = GRUModel(input_size, hidden_size, num_layers, batch_first, output_size).to(device)

# Define the optimizers for window size 50
optimizer_50 = torch.optim.Adam(model_GRU_50.parameters(), weight_decay=1e-5, lr= learning_rate)

# Train the model for 50 days window period
wandb.init(project="Walmart_GRU", name="Model GRU 50 Days Window")  # Start a new W&B run
train_dataset_50, test_dataset_50, train_dataloader_50, test_dataloader_50 = prepare_dataloaders(df, label_column="Adj Close Target", input_window_size=50, prediction_window_size=1, batch_size=32)
overall_preds_denorm_GRU_50, overall_targets_denorm_GRU_50, overall_preds_GRU_50, overall_targets_GRU_50 = train_evaluate_model(
        model= model_GRU_50,
        model_name="Model GRU 50 Days Window",
        train_dataloader= train_dataloader_50,
        test_dataloader= test_dataloader_50,
        train_dataset=train_dataset_50,
        test_dataset = test_dataset_50,
        criterion= criterion,
        optimizer= optimizer_50,
        lr=learning_rate,
        epochs=num_epochs,
        input_window_size= 50)
wandb.finish()  # End the W&B run for this model

# Next we define the window size of 100
train_dataset_100, test_dataset_100, train_dataloader_100, test_dataloader_100 = prepare_dataloaders(df, label_column="Adj Close Target", input_window_size=100, prediction_window_size=1, batch_size=32)

# Initialize the model for the window size 100
model_GRU_100 = GRUModel(input_size, hidden_size, num_layers, batch_first, output_size).to(device)

# Define the optimizers for window size 100
optimizer_100 = torch.optim.Adam(model_GRU_100.parameters(), weight_decay=1e-5, lr= learning_rate)

# Train the model for 100 window periods
wandb.init(project="Walmart_GRU", name="Model GRU 100 Days Window")
overall_preds_denorm_GRU_100, overall_targets_denorm_GRU_100, overall_preds_GRU_100, overall_targets_GRU_100 = train_evaluate_model(
        model= model_GRU_100,
        model_name="Model GRU 100 Days Window",
        train_dataloader= train_dataloader_100,
        test_dataloader= test_dataloader_100,
        train_dataset=train_dataset_100,
        test_dataset = test_dataset_100,
        criterion= criterion,
        optimizer= optimizer_100,
        lr=learning_rate,
        epochs=num_epochs,
        input_window_size= 100)
wandb.finish()

# Next we plot the graph for the various models to compare
last_lookback_days = 50 # We plot the graph to see 50 days of predictions vs 50 days of actual data


# Plot and log each model's predictions separately in Weights & Biases

# Plot for 7 days window period
log_model_plot_overall_preds("GRU Model 7 Days", overall_targets_denorm_GRU_7, overall_preds_denorm_GRU_7,
                             last_lookback_days)

# Plot for 20 days window period
log_model_plot_overall_preds("GRU Model 20 Days", overall_targets_denorm_GRU_20, overall_preds_denorm_GRU_20,
                             last_lookback_days)

# Plot for 50 days window period
log_model_plot_overall_preds("GRU Model 50 Days", overall_targets_denorm_GRU_50, overall_preds_denorm_GRU_50,
                             last_lookback_days)

# Plot for 100 days window period
log_model_plot_overall_preds("GRU Model 100 Days", overall_targets_denorm_GRU_100, overall_preds_denorm_GRU_100,
                             last_lookback_days)

# Save the combined comparison plot
compare_model_graph(overall_preds_denorm_GRU_7, overall_preds_denorm_GRU_7, overall_preds_denorm_GRU_20,
                    overall_preds_denorm_GRU_50, overall_preds_denorm_GRU_100, last_lookback_days)
plt.savefig("comparison_plot.png")  # Save the plot locally


# Save the predictions in a DataFrame and log them
# Find the minimum length across all arrays to avoid length mismatch error
min_length = min(len(overall_targets_denorm_GRU_7),
                 len(overall_preds_denorm_GRU_7),
                 len(overall_preds_denorm_GRU_20),
                 len(overall_preds_denorm_GRU_50),
                 len(overall_preds_denorm_GRU_100))

# Truncate all arrays to the same length
predictions_df_GRU = pd.DataFrame({
    "Actual": overall_targets_denorm_GRU_7[:min_length].ravel(),
    "GRU_7_Lookback": overall_preds_denorm_GRU_7[:min_length].ravel(),
    "GRU_20_Lookback": overall_preds_denorm_GRU_20[:min_length].ravel(),
    "GRU_50_Lookback": overall_preds_denorm_GRU_50[:min_length].ravel(),
    "GRU_100_Lookback": overall_preds_denorm_GRU_100[:min_length].ravel()
})

# Log the DataFrame as a table to W&B
wandb.init(project="Walmart_Research_Data", name="GRU Predictions Table")
wandb.log({"GRU Predictions Table": wandb.Table(dataframe=predictions_df_GRU)})

# Save the DataFrame as a CSV file for local use
predictions_df_GRU.to_csv("GRU_predictions.csv", index=False)

# End W&B run
wandb.finish()








