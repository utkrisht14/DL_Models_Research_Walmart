# Load the necessary libraries
import numpy as np
import pandas as pd
import math
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

# Define the positional encoding for Informer
# Adds temporal information to the input sequence. It is just as in case of Transformers
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Make a common division term.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sine encoding to even places starting from zero and at even position
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply the cosine encoding to even places starting from one and at odd position
        pe[:, 1::2] = torch.cos(position * div_term)

        # For the purpose of adding positional encoding
        if self.batch_first:
            pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        else:
            pe = pe.unsqueeze(1)  # Shape: (max_len, 1, d_model)

        # Register the positional encoding matrix as a buffer, meaning it won't be updated during training
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Positional encoding should be on the same device as the input `x`
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :].to(x.device)  # Shape: [batch_size, seq_len, d_model]
        else:
            x = x + self.pe[:x.size(0), :, :].to(x.device)  # Shape: [seq_len, batch_size, d_model]
        return self.dropout(x)

# Informer encoder layer
class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Multi-head self-attention with batch_first=True
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Feedforward network with ELU activation as in paper
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ELU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Apply self-attention with residual connection and layer normalization
        attn_output, _ = self.self_attn(src, src, src)
        src = self.layer_norm1(src + self.dropout(attn_output))

        # Apply feedforward network with residual connection and layer normalization
        ff_output = self.feed_forward(src)
        src = self.layer_norm2(src + self.dropout(ff_output))
        return src

# Informer Decoder Layer
class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Feedforward network uses ELU as per the paper
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ELU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # Self-attention -> decoder side
        attn_output, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.layer_norm1(tgt + self.dropout(attn_output))

        # Cross-attention -> with encoder memory
        attn_output, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.layer_norm2(tgt + self.dropout(attn_output))

        # Feedforward network and residual connection
        ff_output = self.feed_forward(tgt)
        tgt = self.layer_norm3(tgt + self.dropout(ff_output))

        return tgt

# Informer model-based on the encoder-decoder architecture
class Informer(nn.Module):
    def __init__(self, input_size, d_model, n_heads, d_ff, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(Informer, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)  # Project the input to the model's embedding size
        self.positional_encoding = PositionalEncoding(d_model, dropout, batch_first=True)  # Positional encoding for temporal data

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            InformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_decoder_layers)
        ])

        # Output projection layer
        self.output_proj = nn.Linear(d_model, 1)  # Final output layer to map to the target dimension

    def forward(self, src, tgt):
        # tgt should have the correct shape before projecting
        if len(tgt.shape) == 2:  # If tgt is [batch_size, features], expand to [batch_size, seq_len, features]
            tgt = tgt.unsqueeze(1).expand(-1, src.shape[1], -1)  # Expand along the sequence length

        # Project inputs to the model dimension
        src = self.input_proj(src)  # [batch_size, seq_len, d_model]
        tgt = self.input_proj(tgt)  # [batch_size, seq_len, d_model]

        # Add positional encoding to both input and target sequences
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Encoder: pass through encoder layers
        for layer in self.encoder_layers:
            src = layer(src)

        # Decoder: pass through decoder layers
        for layer in self.decoder_layers:
            tgt = layer(tgt, src)

        # Project the output to the target dimension
        output = self.output_proj(tgt[:, -1, :])  # Take the last time step of decoder output
        return output

# Variables and Hyperparameters
input_window_size = 7
input_size = df.shape[1] - 1  # Number of input features (excluding the target column)
d_model = 128  # Dimension of the model's embedding space
n_heads = 8  # Number of attention heads
d_ff = 256  # Dimension of the feed-forward network
num_encoder_layers = 1  # Number of encoder layers
num_decoder_layers = 1  # Number of decoder layers
dropout = 0.2  # Dropout rate
num_epochs = 100  # Number of training epochs
learning_rate = 1e-6  # Learning rate

# Initialize the Informer model for 7 days
model_informer_7 = Informer(input_size=input_size, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer_7 = torch.optim.Adam(model_informer_7.parameters(), weight_decay=1e-5, lr=learning_rate)

# Prepare the data loaders
train_dataset_7, test_dataset_7, train_dataloader_7, test_dataloader_7 = prepare_dataloaders(
    df, label_column="Adj Close Target", input_window_size=input_window_size, prediction_window_size=1, batch_size=32
)

# Train the model using Informer for a 7-day window forecast
wandb.init(project="Walmart_Informer", name="Model Informer 7 Days Window")
overall_preds_denorm_informer_7, overall_targets_denorm_informer_7, overall_preds_informer_7, overall_targets_informer_7 = train_evaluate_model(
    model=model_informer_7,
    model_name="Model Informer 7 Days Window",
    train_dataloader=train_dataloader_7,
    test_dataloader=test_dataloader_7,
    criterion=criterion,
    train_dataset=train_dataset_7,
    test_dataset=test_dataset_7,
    optimizer=optimizer_7,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=input_window_size,
    model_type="informer"
)

wandb.finish()  # End the W&B run

# Initialize the Informer model for 20 days
input_window_size = 20
model_informer_20 = Informer(input_size=input_size, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                             num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout).to(device)

# Loss function and optimizer
optimizer_20 = torch.optim.Adam(model_informer_20.parameters(), weight_decay=1e-5, lr=learning_rate)

# Prepare the data loaders
train_dataset_20, test_dataset_20, train_dataloader_20, test_dataloader_20 = prepare_dataloaders(
    df, label_column="Adj Close Target", input_window_size=input_window_size, prediction_window_size=1, batch_size=32
)

# Train the model using Informer for a 20-day window forecast
wandb.init(project="Walmart_Informer", name="Model Informer 20 Days Window")
overall_preds_denorm_informer_20, overall_targets_denorm_informer_20, overall_preds_informer_20, overall_targets_informer_20 = train_evaluate_model(
    model=model_informer_20,
    model_name="Model Informer 20 Days Window",
    train_dataloader=train_dataloader_20,
    test_dataloader=test_dataloader_20,
    criterion=criterion,
    train_dataset=train_dataset_20,
    test_dataset=test_dataset_20,
    optimizer=optimizer_20,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=input_window_size,
    model_type="informer"
)

wandb.finish()  # End the W&B run

# Initialize the Informer model for 50 days
input_window_size = 50
model_informer_50 = Informer(input_size=input_size, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                             num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout).to(device)

# Loss function and optimizer
optimizer_50 = torch.optim.Adam(model_informer_50.parameters(), weight_decay=1e-5, lr=learning_rate)

# Prepare the data loaders
train_dataset_50, test_dataset_50, train_dataloader_50, test_dataloader_50 = prepare_dataloaders(
    df, label_column="Adj Close Target", input_window_size=input_window_size, prediction_window_size=1, batch_size=32
)

# Train the model using Informer for a 50-day window forecast
wandb.init(project="Walmart_Informer", name="Model Informer 50 Days Window")
overall_preds_denorm_informer_50, overall_targets_denorm_informer_50, overall_preds_informer_50, overall_targets_informer_50 = train_evaluate_model(
    model=model_informer_50,
    model_name="Model Informer 50 Days Window",
    train_dataloader=train_dataloader_50,
    test_dataloader=test_dataloader_50,
    criterion=criterion,
    train_dataset=train_dataset_50,
    test_dataset=test_dataset_50,
    optimizer=optimizer_50,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=input_window_size,
    model_type="informer"
)

wandb.finish()  # End the W&B run

# Initialize the Informer model for 100 days
input_window_size = 100
model_informer_100 = Informer(input_size=input_size, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                              num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout).to(device)

# Optimizer
optimizer_100 = torch.optim.Adam(model_informer_100.parameters(), weight_decay=1e-5, lr=learning_rate)

# Prepare the data loaders
train_dataset_100, test_dataset_100, train_dataloader_100, test_dataloader_100 = prepare_dataloaders(
    df, label_column="Adj Close Target", input_window_size=input_window_size, prediction_window_size=1, batch_size=32
)

# Train the model using Informer for a 100-day window forecast
wandb.init(project="Walmart_Informer", name="Model Informer 100 Days Window")
overall_preds_denorm_informer_100, overall_targets_denorm_informer_100, overall_preds_informer_100, overall_targets_informer_100 = train_evaluate_model(
    model=model_informer_100,
    model_name="Model Informer 100 Days Window",
    train_dataloader=train_dataloader_100,
    test_dataloader=test_dataloader_100,
    criterion=criterion,
    train_dataset=train_dataset_100,
    test_dataset=test_dataset_100,
    optimizer=optimizer_100,
    lr=learning_rate,
    epochs=num_epochs,
    input_window_size=input_window_size,
    model_type="informer"
)

wandb.finish()  # End the W&B run

# Next we plot the data
last_lookback_days = 100

# Plot for 7 days window period
log_model_plot_overall_preds("Informer Model 7 Days- No Scheduler", overall_targets_denorm_informer_7, overall_preds_denorm_informer_7,
                             last_lookback_days)

# Plot for 20 days window period
log_model_plot_overall_preds("Informer Model 20 Days- No Scheduler", overall_targets_denorm_informer_20, overall_preds_denorm_informer_20,
                             last_lookback_days)

# Plot for 50 days window period
log_model_plot_overall_preds("Informer Model 50 Days- No Scheduler", overall_targets_denorm_informer_50, overall_preds_denorm_informer_50,
                             last_lookback_days)

# Plot for 100 days window period
log_model_plot_overall_preds("Informer Model 100 Days- No Scheduler", overall_targets_denorm_informer_100, overall_preds_denorm_informer_100,
                             last_lookback_days)
