import pandas as pd
import numpy as np


class WindowedDataset:
    def __init__(self, X_train, X_eval, X_test, y_train, y_eval, y_test, window_size, overlap=0):
        """
        Initialize the WindowedDataset with data splits and windowing parameters.

        Args:
            X_train, X_eval, X_test, y_train, y_eval, y_test (pd.DataFrame): Input and target data splits.
            window_size (int): Number of rows in each window.
            overlap (float): Fractional overlap between windows (e.g., 0.5 for 50% overlap).
        """
        # Store each data split and windowing parameters
        self.X_train = X_train
        self.X_eval = X_eval
        self.X_test = X_test
        self.y_train = y_train
        self.y_eval = y_eval
        self.y_test = y_test
        self.window_size = window_size
        # Calculate the step size between windows based on overlap
        self.step_size = max(1, int(window_size * (1 - overlap)))

    def create_windows(self, dataset):
        """
        Creates overlapping windows from a given dataset.

        Args:
            dataset (pd.DataFrame): The dataset to window (either X or y data).

        Returns:
            np.ndarray: A 3D array with shape (num_windows, window_size, num_features), where each
                        slice along the first axis represents one window.
        """
        # Estimate the number of complete windows that can be created
        num_windows = (len(dataset) - self.window_size) // self.step_size + 1
        # Check if any remaining data points exist after the last full window
        last_index = (num_windows - 1) * self.step_size + self.window_size
        if last_index < len(dataset):
            # Add one more window to include these remaining data points
            num_windows += 1

        # Initialize an empty list to store each window as a 2D array
        windows = []
        for i in range(num_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            # Extract the window from the dataset
            window = dataset.iloc[start_idx:end_idx].values
            if len(window) < self.window_size:  # Check if window is shorter than required size
                # Pad shorter windows with zeros to reach the required size
                padded_window = np.zeros((self.window_size, dataset.shape[1]))
                padded_window[:len(window)] = window
                windows.append(padded_window)
            else:
                # Append full-sized windows directly
                windows.append(window)

        # Convert the list of windows to a 3D numpy array
        return np.array(windows)

    def prepare_data_for_lstm(self):
        """
        Prepares the training, evaluation, and test splits as windowed datasets for LSTM input.

        Returns:
            tuple: Each element is a 3D array containing windowed data for each split (X and y),
                   ready for LSTM input.
        """
        # Create windowed datasets for each split of X and y data
        X_train_windows = self.create_windows(self.X_train)
        X_eval_windows = self.create_windows(self.X_eval)
        X_test_windows = self.create_windows(self.X_test)

        y_train_windows = self.create_windows(self.y_train)
        y_eval_windows = self.create_windows(self.y_eval)
        y_test_windows = self.create_windows(self.y_test)

        # Return a tuple with windowed datasets for all splits
        return (X_train_windows, y_train_windows, X_eval_windows, y_eval_windows, X_test_windows, y_test_windows)

    def reconstruct_predictions(self, predictions_3d):
        """
        Reconstructs the windowed predictions back into a 2D array with the original sequence length.

        Args:
            predictions_3d (np.ndarray): Model predictions with shape (num_windows, window_size, num_features).

        Returns:
            np.ndarray: Reconstructed predictions with shape (original_length, num_features), where
                        overlapping windows are averaged.
        """
        # Extract dimensions from the predictions
        num_windows, window_size, num_features = predictions_3d.shape
        # Calculate the total length of the reconstructed array
        original_length = (num_windows - 1) * self.step_size + window_size

        # Initialize arrays to accumulate predictions and counts for averaging
        reconstructed_preds = np.zeros((original_length, num_features))
        counts = np.zeros((original_length, num_features))

        # Iterate over each window and accumulate predictions
        for i in range(num_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + window_size
            # Handle last partial window if it extends beyond the original length
            window_pred = predictions_3d[i][:min(window_size, original_length - start_idx)]
            reconstructed_preds[start_idx:start_idx + len(window_pred)] += window_pred
            counts[start_idx:start_idx + len(window_pred)] += 1

        # Divide by counts to average overlapping window predictions
        reconstructed_preds /= counts
        # Return only up to the original target length to match y_test
        return reconstructed_preds[:len(self.y_test)]


