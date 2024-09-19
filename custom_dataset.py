import torch
import numpy as np
import pandas as pd

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("walmart_dataset.csv", index_col="Date", parse_dates=True)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, label_column, input_window_size, prediction_window_size, model_type="generic"):
        super().__init__()
        self.data = data
        self.label_column = label_column
        self.input_window_size = input_window_size
        self.prediction_window_size = prediction_window_size
        self.model_type = model_type  # Specify model type to handle N-Beats-specific adjustments

        # Separate features and labels
        self.features = self.data.drop(columns=[self.label_column]).values
        self.labels = self.data[self.label_column].values

        # Normalize the features and the labels
        self.features, self.features_min, self.features_max = self.minmax_normalization(self.features)
        self.labels, self.labels_min, self.labels_max = self.minmax_normalization(self.labels.reshape(-1, 1))

    def __len__(self):
        # Adjust the length for the input window and the output window
        return len(self.data) - self.input_window_size - self.prediction_window_size + 1

    def __getitem__(self, idx):
        # Extract input window
        x = self.features[idx: idx + self.input_window_size]

        # Extract the prediction
        y = self.labels[idx + self.input_window_size: idx + self.input_window_size + self.prediction_window_size]

        # If it's for the N-Beats model, adjust the shape of the target 'y'
        if self.model_type == "nbeats":
            y = y.reshape(-1)  # "y" is flattened for N-Beats forecast (1D target)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def minmax_normalization(self, data):
        # Calculate the minmax normalization along the feature dimension
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        normalized_data = (data - min_val) / (max_val - min_val + epsilon)
        return normalized_data, min_val, max_val


# Wrap the logic inside a function so that it can be called externally
def prepare_dataloaders(df, label_column, input_window_size, prediction_window_size, batch_size, model_type="generic"):
    test_size = int(0.2 * len(df))

    # Split dataset into train and test sets. Preserve the timeseries flow
    train_set = df[:-test_size]
    test_set = df[-test_size:]

    # Initialize datasets
    train_dataset = CustomDataset(
        train_set, label_column=label_column, input_window_size=input_window_size,
        prediction_window_size=prediction_window_size, model_type=model_type
    )
    test_dataset = CustomDataset(
        test_set, label_column=label_column, input_window_size=input_window_size,
        prediction_window_size=prediction_window_size, model_type=model_type
    )

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataset, test_dataset, train_dataloader, test_dataloader



