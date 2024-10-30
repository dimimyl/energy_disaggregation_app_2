import pandas as pd
import numpy as np


class WindowedDataset:
    def __init__(self, X_train, X_eval, X_test, y_train, y_eval, y_test, window_size, overlap=0):
        """
        Initialize with the datasets, window size, and overlap percentage.

        Args:
            X_train, X_eval, X_test, y_train, y_eval, y_test (pd.DataFrame): Input and target data splits.
            window_size (int): Number of rows in each window.
            overlap (float): Fractional overlap between windows (e.g., 0.5 for 50% overlap).
        """
        self.X_train = X_train
        self.X_eval = X_eval
        self.X_test = X_test
        self.y_train = y_train
        self.y_eval = y_eval
        self.y_test = y_test
        self.window_size = window_size
        self.step_size = max(1, int(window_size * (1 - overlap)))

    def create_windows(self, dataset):
        """
        Create overlapping windows for a given dataframe.

        Args:
            dataset (pd.DataFrame): The dataframe to window.

        Returns:
            np.ndarray: 3D array of shape (num_windows, window_size, num_features).
        """
        num_windows = (len(dataset) - self.window_size) // self.step_size + 1
        windows = np.empty((num_windows, self.window_size, dataset.shape[1]))

        for i in range(num_windows):
            start_idx = i * self.step_size
            windows[i] = dataset.iloc[start_idx:start_idx + self.window_size].values

        return windows

    def prepare_data_for_lstm(self):
        """
        Prepare each split (train, eval, test) for LSTM with TimeDistributed layer.

        Returns:
            tuple: Contains windowed data for X and y for each split, ready for LSTM.
        """
        X_train_windows = self.create_windows(self.X_train)
        X_eval_windows = self.create_windows(self.X_eval)
        X_test_windows = self.create_windows(self.X_test)

        y_train_windows = self.create_windows(self.y_train)
        y_eval_windows = self.create_windows(self.y_eval)
        y_test_windows = self.create_windows(self.y_test)

        return (X_train_windows, y_train_windows, X_eval_windows, y_eval_windows, X_test_windows, y_test_windows)

    def reconstruct_predictions(self, predictions_3d):
        """
        Reconstructs 3D predictions to match the 2D original target shape.

        Args:
            predictions_3d (np.ndarray): Model predictions with shape (num_windows, window_size, num_features).

        Returns:
            np.ndarray: Reconstructed predictions with shape (original_length, num_features).
        """
        num_windows, window_size, num_features = predictions_3d.shape
        original_length = (num_windows - 1) * self.step_size + window_size

        # Initialize an array for the reconstructed predictions and a count array for averaging
        reconstructed_preds = np.zeros((original_length, num_features))
        counts = np.zeros((original_length, num_features))

        for i in range(num_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + window_size

            # Accumulate predictions and update the count for each position
            reconstructed_preds[start_idx:end_idx] += predictions_3d[i]
            counts[start_idx:end_idx] += 1

        # Divide by counts to get the average where overlapping predictions occur
        reconstructed_preds /= counts
        return reconstructed_preds


    def save_windows(self, windowed_data, file_path):
        """
        Save the windowed data to a CSV file.

        Args:
            windowed_data (np.ndarray): 3D array of the windowed data to save.
            file_path (str): The file path to save the CSV.
        """
        if windowed_data is not None:
            # Flatten the 3D array for saving
            num_windows, window_size, num_features = windowed_data.shape
            flat_data = windowed_data.reshape(num_windows * window_size, num_features)

            # Convert to DataFrame and save
            df = pd.DataFrame(flat_data)
            df.to_csv(file_path, index=False)
            print(f"Windowed dataset saved to {file_path}")
        else:
            print("No windowed data provided. Please run `create_windows()` first.")
