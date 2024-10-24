from sklearn.model_selection import train_test_split


class DatasetSplitter:
    def __init__(self, dataframe, target_columns, agg_column='agg', timestamp_column='time', client_columns=None):
        """
        Initializes the DatasetSplitter class with a dataframe.

        Parameters:
        dataframe (pd.DataFrame): The dataframe to be split.
        target_columns (list): List of columns to be used as targets (y).
        agg_column (str): The name of the aggregated power column. Default is 'agg'.
        timestamp_column (str): The name of the timestamp column. Default is 'time'.
        client_columns (list): List of one-hot encoded client columns.
        """
        self.dataframe = dataframe
        self.target_columns = target_columns
        self.agg_column = agg_column
        self.timestamp_column = timestamp_column

        # If client columns are not provided, raise an error
        if client_columns is None or len(client_columns) == 0:
            raise ValueError("Client columns must be provided.")
        self.client_columns = client_columns

    def split(self):
        """
        Splits the dataset into training, evaluation, and test sets.

        The first three clients are used for training, with 20% of this data used for evaluation.
        The fourth client is used as the test set.
        """
        # Get unique client columns (already one-hot encoded)
        train_clients = self.client_columns[:3]  # First three clients for training
        test_client = self.client_columns[3]  # Fourth client for testing

        # Train dataset: all data from the first three clients
        train_data = self.dataframe[self.dataframe[train_clients].sum(axis=1) == 1]

        # Split the training dataset into train (80%) and evaluation (20%)
        train_data, eval_data = train_test_split(train_data, test_size=0.1, random_state=42, shuffle=False)

        # Test dataset: all data from the fourth client
        fourth_client_data = self.dataframe[self.dataframe[test_client] == 1]
        test_data = fourth_client_data  # Use all data from the fourth client for testing

        # Feature columns: time, agg, and one-hot encoded clients
        feature_columns = [self.timestamp_column, self.agg_column] + self.client_columns

        # Target columns: The five device columns (provided as target_columns)
        X_train = train_data[feature_columns]
        y_train = train_data[self.target_columns]

        X_eval = eval_data[feature_columns]
        y_eval = eval_data[self.target_columns]

        X_test = test_data[feature_columns]
        y_test = test_data[self.target_columns]

        # Store the splits as attributes to be accessed later
        self.X_train, self.y_train = X_train, y_train
        self.X_eval, self.y_eval = X_eval, y_eval
        self.X_test, self.y_test = X_test, y_test

        return X_train, y_train, X_eval, y_eval, X_test, y_test

    def save_splits_to_csv(self, output_dir='output'):
        """
        Saves the split datasets (X_train, y_train, X_eval, y_eval, X_test, y_test) as CSV files.

        Parameters:
        output_dir (str): The directory where the CSV files will be saved. Default is 'output'.
        """
        # Save X_train and y_train
        self.X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
        self.y_train.to_csv(f'{output_dir}/y_train.csv', index=False)

        # Save X_eval and y_eval
        self.X_eval.to_csv(f'{output_dir}/X_eval.csv', index=False)
        self.y_eval.to_csv(f'{output_dir}/y_eval.csv', index=False)

        # Save X_test and y_test
        self.X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
        self.y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

        print(f"Datasets saved to {output_dir}/")



