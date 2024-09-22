import time
import torch
import numpy as np
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


device = "cuda" if torch.cuda.is_available() else "cpu"


def denormalize(data, min_val, max_val):
    return data * (max_val - min_val + 1e-8) + min_val


def train_evaluate_model(model, model_name, optimizer, criterion, train_dataloader, test_dataloader,
                         train_dataset, test_dataset, epochs=50, lr=1e-3, input_window_size=20, scheduler=None,
                         device=None, model_type=None, patience=25):
    """
    Train and evaluate the model for time-series forecasting with Early Stopping and Best Model Saving.

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
    - scheduler: learning rate scheduler. Default is None.
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
    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss = 0
        model.train()

        # Training loop
        for batch in train_dataloader:
            X_train, y_train = batch
            X_train, y_train = X_train.to(device), y_train.to(device)

            # Fix the shape of tgt based on X_train for Informer
            if model_type == "informer":
                tgt = X_train.clone()  # Create a target sequence based on X_train
            else:
                tgt = y_train

            # Handle reshaping for different models
            if model_type == "nbeats":
                y_train = y_train.squeeze(-1)  # Flatten the target for N-Beats 
            elif model_type == "tcn" or model_type == "informer":
                y_train = y_train.view(-1)  # Flatten target for TCN or Informer
            else:
                y_train = y_train.squeeze()  # For LSTM, Transformer, squeeze to remove extra dimensions

            # Forward pass for Informer or other models
            if model_type == "informer":
                y_pred = model(X_train, tgt)  # Informer requires both src and tgt inputs
            else:
                y_pred = model(X_train)

            # Flatten predictions if necessary
            if model_type == "nbeats":
                y_pred = y_pred.squeeze(-1)  # Flatten predictions for N-Beats 
            elif model_type == "tcn" or model_type == "informer":
                y_pred = y_pred.view(-1)  # Flatten predictions for TCN or Informer

            # Ensure predictions and targets have the same shape
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
                X_test, y_test = batch
                X_test, y_test = X_test.to(device), y_test.to(device)

                # Fix the shape of tgt based on X_test for Informer
                if model_type == "informer":
                    tgt_test = X_test.clone()  # Create a target sequence based on X_test
                else:
                    tgt_test = y_test

                # Forward pass for Informer or other models
                if model_type == "informer":
                    y_val = model(X_test, tgt_test)  # Informer requires both src and tgt inputs
                else:
                    y_val = model(X_test)

                # Flatten predictions
                if model_type == "nbeats":
                    y_val = y_val.squeeze(-1)  # Flatten predictions for N-Beats
                elif model_type == "tcn" or model_type == "informer":
                    y_val = y_val.view(-1)  # Flatten predictions for TCN or Informer

                # Ensure predictions and targets have the same shape
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
