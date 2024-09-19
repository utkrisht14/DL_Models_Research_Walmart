import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from custom_dataset import prepare_dataloaders
from train import train_evaluate_model
from graphs_plot import compare_model_graph, log_model_plot_overall_preds


import wandb

# Read the dataset
df = pd.read_csv("walmart_dataset.csv", index_col="Date", parse_dates=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# Before making transformer model, there is need to make positional encoding class.
# This class adds positional encoding to the input sequence.
# Since transformers don't inherently understand the order of the sequence.
# Because of the self-attention mechanism, positional encodings are added to provide the model with information about the relative positions of the time steps.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create a matrix to store the positional encodings
        pe = torch.zeros(max_len, d_model)

        # Create a tensor that represents each position in the sequence
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create a tensor that will be used to apply sinusoidal functions to the positions.
        # Use different frequencies for each dimension of the positional encoding. Create a common div_term.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        # Apply sine to even indices in the positional encoding according to paper
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices in the positional encoding according to paper
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension to the positional encodings so that they can be easily added to the input embeddings
        # Shape after unsqueeze: (1, max_len, d_model)
        # Shape after transpose: (max_len, 1, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register the positional encoding matrix as a buffer, meaning that it won't be updated during training
        self.register_buffer("pe", pe)

    # Forward method to add positional encodings to the input sequence
    def forward(self, x):
        # Add the positional encoding to the input embeddings.
        # x is expected to have shape (seq_len, batch_size, d_model), and we add the positional encoding based on the sequence length.
        return x + self.pe[:x.size(0), :]


# Define the Transformer Model Architecture
class TransformerModels(nn.Module):
    def __init__(self, num_inputs, d_model, nhead, num_layers, dim_feedforward, dropout, n_steps_to_predict=1):
        super().__init__()

        # Embedding for the input time series (converts input to d_model dimensions)
        self.input_embedding = nn.Linear(num_inputs, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first= True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer to map back to the desired output size. Here "n_steps_to_predict" = 1
        self.fc_out = nn.Linear(d_model, n_steps_to_predict)

    def forward(self, src):
        # shape of src: (batch_size, seq_len, num_inputs)
        src = self.input_embedding(src)  # Embed the input to d_model dimensions
        src = self.positional_encoding(src)  # Add positional encoding
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)

        # Pass through the Transformer Encoder
        transformer_output = self.transformer_encoder(src)

        # Take the output of the last time step (sequence length, batch, d_model) -> (batch, d_model)
        last_time_step_output = transformer_output[-1, :, :]  # Take the last time step for forecasting

        # Pass through a linear layer to get the prediction for n steps
        output = self.fc_out(last_time_step_output)

        return output



# Hyperparameters
num_inputs = df.shape[1] - 1 # All the columns except the target column
input_window = 7  # Use past 7 time steps as input
output_window = 1  # Predict the next time step
num_epochs = 100
learning_rate = 1e-4
d_model = 128  # Transformer embedding size
nhead = 4  # Number of attention heads
num_layers = 1 # Number of transformer encoder layers
dim_feedforward = 64  # Hidden layer size in feedforward network
dropout = 0.1  # Dropout rate

model_7_no_scheduler = TransformerModels(d_model=d_model, nhead=nhead, num_inputs=num_inputs, num_layers=num_layers,
                                         dim_feedforward=dim_feedforward, dropout=dropout, n_steps_to_predict=1).to(device)

criterion = nn.MSELoss()
optimizer_7 = torch.optim.Adam(model_7_no_scheduler.parameters(), lr=learning_rate)

# I tried four window size. 7, 20, 50, and 100
# Start with window size of 7
train_dataset_7, test_dataset_7, train_dataloader_7, test_dataloader_7 = prepare_dataloaders(df,
                                                                                             label_column="Adj Close Target",
                                                                                             input_window_size= input_window,
                                                                                             prediction_window_size=1,
                                                                                             batch_size=32)

# Train the model for 7 days window period
wandb.init(project="Walmart_Transformers", name="Model Transformer 7 Days Window- - No Scheduler")  # Start a new W&B run

overall_preds_denorm_transformer_7, overall_targets_denorm_transformer_7, overall_preds_transformer_7, overall_targets_transformer_7 = train_evaluate_model(
    model=model_7_no_scheduler,
    model_name="Model Transformer 7 Days Window - No Scheduler",
    train_dataloader=train_dataloader_7,
    test_dataloader=test_dataloader_7,
    criterion=criterion,
    train_dataset=train_dataset_7,
    test_dataset=test_dataset_7,
    optimizer=optimizer_7,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=input_window)
wandb.finish()  # End the W&B run for this model


# Now for the 20 days window period without learning rate scheduler
input_window = 20  # Use past 20 time steps as input

model_20_no_scheduler = TransformerModels(d_model=d_model, nhead=nhead, num_inputs=num_inputs, num_layers=num_layers,
                                         dim_feedforward=dim_feedforward, dropout=dropout, n_steps_to_predict=1).to(device)

criterion = nn.MSELoss()
optimizer_20 = torch.optim.Adam(model_20_no_scheduler.parameters(), lr=learning_rate)

# Now train with window size of 20
train_dataset_20, test_dataset_20, train_dataloader_20, test_dataloader_20 = prepare_dataloaders(df,
                                                                                             label_column="Adj Close Target",
                                                                                             input_window_size=20,
                                                                                             prediction_window_size=1,
                                                                                             batch_size=32)


# Train the model for 20 days window period
wandb.init(project="Walmart_Transformers", name="Model Transformer 20 Days Window- No Scheduler")  # Start a new W&B run

overall_preds_denorm_transformer_20, overall_targets_denorm_transformer_20, overall_preds_transformer_20, overall_targets_transformer_20 = train_evaluate_model(
    model=model_20_no_scheduler,
    model_name="Model Transformer 20 Days Window - No Scheduler",
    train_dataloader=train_dataloader_20,
    test_dataloader=test_dataloader_20,
    criterion=criterion,
    train_dataset=train_dataset_20,
    test_dataset=test_dataset_20,
    optimizer=optimizer_20,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=input_window)
wandb.finish()  # End the W&B run for this model


# Train the model for the 50 days window period without learning rate scheduler
input_window = 50  # Use past 50 time steps as input

model_50_no_scheduler = TransformerModels(d_model=d_model, nhead=nhead, num_inputs=num_inputs, num_layers=num_layers,
                                         dim_feedforward=dim_feedforward, dropout=dropout, n_steps_to_predict=1).to(device)

criterion = nn.MSELoss()
optimizer_50 = torch.optim.Adam(model_50_no_scheduler.parameters(), lr=learning_rate)

# Now train with window size of 50
train_dataset_50, test_dataset_50, train_dataloader_50, test_dataloader_50 = prepare_dataloaders(df,
                                                                                             label_column="Adj Close Target",
                                                                                             input_window_size=input_window,
                                                                                             prediction_window_size=1,
                                                                                             batch_size=32)


# Train the model for 50 days window period
wandb.init(project="Walmart_Transformers", name="Model Transformer 50 Days Window - No Scheduler")  # Start a new W&B run

overall_preds_denorm_transformer_50, overall_targets_denorm_transformer_50, overall_preds_transformer_50, overall_targets_transformer_50 = train_evaluate_model(
    model=model_50_no_scheduler,
    model_name="Model Transformer 50 Days Window - No Scheduler",
    train_dataloader=train_dataloader_50,
    test_dataloader=test_dataloader_50,
    criterion=criterion,
    train_dataset=train_dataset_50,
    test_dataset=test_dataset_50,
    optimizer=optimizer_50,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=input_window)
wandb.finish()  # End the W&B run for this model


# Train the model for the 100 days window period without learning rate scheduler
input_window = 100  # Use past 100 time steps as input

model_100_no_scheduler = TransformerModels(d_model=d_model, nhead=nhead, num_inputs=num_inputs, num_layers=num_layers,
                                         dim_feedforward=dim_feedforward, dropout=dropout, n_steps_to_predict=1).to(device)

criterion = nn.MSELoss()
optimizer_100 = torch.optim.Adam(model_100_no_scheduler.parameters(), lr=learning_rate)

# Now train with window size of 100
train_dataset_100, test_dataset_100, train_dataloader_100, test_dataloader_100 = prepare_dataloaders(df,
                                                                                             label_column="Adj Close Target",
                                                                                             input_window_size=input_window,
                                                                                             prediction_window_size=1,
                                                                                             batch_size=32)


# Train the model for 100 days window period
wandb.init(project="Walmart_Transformers", name="Model Transformer 100 Days Window - No Scheduler")  # Start a new W&B run

overall_preds_denorm_transformer_100, overall_targets_denorm_transformer_100, overall_preds_transformer_100, overall_targets_transformer_100 = train_evaluate_model(
    model=model_100_no_scheduler,
    model_name="Model Transformer 100 Days Window - No Scheduler",
    train_dataloader=train_dataloader_100,
    test_dataloader=test_dataloader_100,
    criterion=criterion,
    train_dataset=train_dataset_100,
    test_dataset=test_dataset_100,
    optimizer=optimizer_100,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=input_window)
wandb.finish()  # End the W&B run for this model


# Adding LR Scheduler to the models as they have defined in the paper
# Define the TransformerLRScheduler class
class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.calculate_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def calculate_lr(self):
        lr = (self.d_model ** -0.5) * min(self.current_step ** -0.5, self.current_step * (self.warmup_steps ** -1.5))
        return lr

# Learning rate schedulers
warmup_steps = 500 # In paper warmup_steps is defined as 4000. But since there is less data. So went with 500.

# Start with 7 days
input_window = 7  # Use past 7 time steps as input

# Define the model with the scheduling rate
model_7_scheduler = TransformerModels(d_model=d_model, nhead=nhead, num_inputs=num_inputs, num_layers=num_layers,
                                         dim_feedforward=dim_feedforward, dropout=dropout, n_steps_to_predict=1).to(device)

criterion = nn.MSELoss()
optimizer_7_scheduler = torch.optim.Adam(model_7_scheduler.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=learning_rate) # Setting the beta and epsilon according to the research paper
lr_scheduler_7 = TransformerLRScheduler(optimizer_7_scheduler, d_model=d_model, warmup_steps=warmup_steps)

# I tried four window size. 7, 20, 50, and 100

# Train the model for 7 days window period
wandb.init(project="Walmart_Transformers", name="Model Transformer 7 Days Window - Scheduler")  # Start a new W&B run

overall_preds_scheduler_denorm_transformer_7, overall_targets_scheduler_denorm_transformer_7, overall_preds_scheduler_transformer_7, overall_targets_scheduler_transformer_7 = train_evaluate_model(
    model=model_7_scheduler,
    model_name="Model Transformer 7 Days Window- Scheduler",
    train_dataloader=train_dataloader_7,
    test_dataloader=test_dataloader_7,
    scheduler= lr_scheduler_7,
    criterion=criterion,
    train_dataset=train_dataset_7,
    test_dataset=test_dataset_7,
    optimizer=optimizer_7_scheduler,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=input_window)
wandb.finish()  # End the W&B run for this model


# Now with 20 days with scheduler
input_window = 20  # Use past 20 time steps as input


model_20_scheduler = TransformerModels(d_model=d_model, nhead=nhead, num_inputs=num_inputs, num_layers=num_layers,
                                         dim_feedforward=dim_feedforward, dropout=dropout, n_steps_to_predict=1).to(device)

criterion = nn.MSELoss()
optimizer_20_scheduler = torch.optim.Adam(model_20_scheduler.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=learning_rate) # Setting the beta and epsilon according to the research paper
lr_scheduler_20 = TransformerLRScheduler(optimizer_20_scheduler, d_model=d_model, warmup_steps=warmup_steps)


# Train the model for 20 days window period
wandb.init(project="Walmart_Transformers", name="Model Transformer 20 Days Window- Scheduler")  # Start a new W&B run

overall_preds_scheduler_denorm_transformer_20, overall_targets_scheduler_denorm_transformer_20, overall_preds_scheduler_transformer_20, overall_targets_scheduler_transformer_20 = train_evaluate_model(
    model=model_20_scheduler,
    model_name="Model Transformer 20 Days Window- Scheduler",
    train_dataloader=train_dataloader_20,
    test_dataloader=test_dataloader_20,
    scheduler= lr_scheduler_20,
    criterion=criterion,
    train_dataset=train_dataset_20,
    test_dataset=test_dataset_20,
    optimizer=optimizer_20_scheduler,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=20)
wandb.finish()  # End the W&B run for this model


# Now for 50 days
input_window = 50  # Use past 50 time steps as input

model_50_scheduler = TransformerModels(d_model=d_model, nhead=nhead, num_inputs=num_inputs, num_layers=num_layers,
                                         dim_feedforward=dim_feedforward, dropout=dropout, n_steps_to_predict=1).to(device)

criterion = nn.MSELoss()
optimizer_50_scheduler = torch.optim.Adam(model_50_scheduler.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=learning_rate) # Setting the beta and epsilon according to the research paper
lr_scheduler_50 = TransformerLRScheduler(optimizer_50_scheduler, d_model=d_model, warmup_steps=warmup_steps)

# Train the model for 50 days window period
wandb.init(project="Walmart_Transformers", name="Model Transformer 50 Days Window- Scheduler")  # Start a new W&B run

overall_preds_scheduler_denorm_transformer_50, overall_targets_scheduler_denorm_transformer_50, overall_preds_scheduler_transformer_50, overall_targets_scheduler_transformer_50 = train_evaluate_model(
    model=model_50_scheduler,
    model_name="Model Transformer 50 Days Window- Scheduler",
    train_dataloader=train_dataloader_50,
    test_dataloader=test_dataloader_50,
    scheduler= lr_scheduler_50,
    criterion=criterion,
    train_dataset=train_dataset_50,
    test_dataset=test_dataset_50,
    optimizer=optimizer_50_scheduler,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=input_window)
wandb.finish()  # End the W&B run for this model


# Now for 100 days with scheduler
input_window = 100  # Use past 100 time steps as input

model_100_scheduler = TransformerModels(d_model=d_model, nhead=nhead, num_inputs=num_inputs, num_layers=num_layers,
                                         dim_feedforward=dim_feedforward, dropout=dropout, n_steps_to_predict=1).to(device)

criterion = nn.MSELoss()
optimizer_100_scheduler = torch.optim.Adam(model_100_scheduler.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=learning_rate) # Setting the beta and epsilon according to the research paper
lr_scheduler_100 = TransformerLRScheduler(optimizer_100_scheduler, d_model=d_model, warmup_steps=warmup_steps)

# Train the model for 100 days window period
wandb.init(project="Walmart_Transformers", name="Model Transformer 100 Days Window - Scheduler")  # Start a new W&B run

overall_preds_scheduler_denorm_transformer_100, overall_targets_scheduler_denorm_transformer_100, overall_preds_scheduler_transformer_100, overall_targets_scheduler_transformer_100 = train_evaluate_model(
    model=model_100_scheduler,
    model_name="Model Transformer 100 Days Window- Scheduler",
    train_dataloader=train_dataloader_100,
    test_dataloader=test_dataloader_100,
    scheduler= lr_scheduler_100,
    criterion=criterion,
    train_dataset=train_dataset_100,
    test_dataset=test_dataset_100,
    optimizer=optimizer_100_scheduler,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=100)
wandb.finish()  # End the W&B run for this model


# Next we plot the graph for the various models to compare
last_lookback_days = 100 # We plot the graph to see 100 days of predictions vs 100 days of actual data

# Plot for 7 days window period
log_model_plot_overall_preds("Transformer Model 7 Days- No Scheduler", overall_targets_denorm_transformer_7, overall_targets_denorm_transformer_7,
                             last_lookback_days)

# Plot for 20 days window period
log_model_plot_overall_preds("Transformer Model 20 Days- No Scheduler", overall_targets_denorm_transformer_20, overall_preds_denorm_transformer_20,
                             last_lookback_days)

# Plot for 50 days window period
log_model_plot_overall_preds("Transformer Model 50 Days- No Scheduler", overall_targets_denorm_transformer_50, overall_preds_denorm_transformer_50,
                             last_lookback_days)

# Plot for 100 days window period
log_model_plot_overall_preds("Transformer Model 100 Days- No Scheduler", overall_targets_denorm_transformer_100, overall_preds_denorm_transformer_100,
                             last_lookback_days)


# Save the combined plot
compare_model_graph(overall_targets_denorm_transformer_7, overall_preds_denorm_transformer_7, overall_preds_denorm_transformer_20, overall_preds_denorm_transformer_50,
                    overall_preds_denorm_transformer_100, last_lookback_days)
plt.savefig("comparison_plot.png")  # Save the plot locally


# Save the predictions in a DataFrame and log them
# Find the minimum length across all arrays to avoid mismatch of length
min_length = min(len(overall_targets_denorm_transformer_7),
                 len(overall_preds_denorm_transformer_7),
                 len(overall_preds_denorm_transformer_20),
                 len(overall_preds_denorm_transformer_50),
                 len(overall_preds_scheduler_denorm_transformer_100),
                 len(overall_preds_scheduler_denorm_transformer_7),
                 len(overall_preds_scheduler_denorm_transformer_20),
                 len(overall_preds_scheduler_denorm_transformer_50),
                 len(overall_preds_scheduler_denorm_transformer_100))

# Truncate all arrays to the same length
predictions_df_transformers = pd.DataFrame({
    "Actual": overall_targets_denorm_transformer_7[:min_length].ravel(),
    "Transformer_no_scheduler_7_Lookback": overall_preds_denorm_transformer_7[:min_length].ravel(),
    "Transformer_no_scheduler_20_Lookback": overall_preds_denorm_transformer_20[:min_length].ravel(),
    "Transformer_no_scheduler_50_Lookback": overall_preds_denorm_transformer_50[:min_length].ravel(),
    "Transformer_no_scheduler_100_Lookback": overall_preds_denorm_transformer_100[:min_length].ravel(),
    "Transformer_scheduler_7_Lookback": overall_preds_scheduler_denorm_transformer_7[:min_length].ravel(),
    "Transformer_scheduler_20_Lookback": overall_preds_scheduler_denorm_transformer_20[:min_length].ravel(),
    "Transformer_scheduler_50_Lookback": overall_preds_scheduler_denorm_transformer_50[:min_length].ravel(),
    "Transformer_scheduler_100_Lookback": overall_preds_scheduler_denorm_transformer_100[:min_length].ravel()
})

# Log the DataFrame as a table to W&B
wandb.init(project="Walmart_Research_Data", name="Transformer Predictions Table")
wandb.log({"Transformer Predictions Table": wandb.Table(dataframe=predictions_df_transformers)})

# Save the DataFrame as a CSV file for local use
predictions_df_transformers.to_csv("transformer_predictions.csv", index=False)

# End W&B run
wandb.finish()






