import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Preprocessor:
    def __init__(self, dataframe):
        """
        Initializes the Preprocessing class with a dataframe.

        Parameters:
        dataframe (pd.DataFrame): The dataframe to be preprocessed.
        """
        self.dataframe = dataframe

    def forward_fill(self):
        """
        Applies forward fill (ffill) to handle missing values in the dataframe.
        """
        self.dataframe.ffill(inplace=True)

    def modify_clientid(self, column_name='clientid', substring='house'):
        """
        Modifies the clientid column by removing a specific substring and converting the result to an integer.

        Parameters:
        column_name (str): The name of the column to modify. Default is 'clientid'.
        substring (str): The substring to remove from the values in the column. Default is 'house'.
        """
        # Remove the specified substring
        self.dataframe[column_name] = self.dataframe[column_name].str.replace(substring, '', regex=False)
        # Convert the column to integers
        self.dataframe[column_name] = self.dataframe[column_name].astype(int)

    def one_hot_encode_clients(self):
        """
        Creates one-hot encoded columns for the 'clientid' column and places them before the existing columns.

        Returns:
        pd.DataFrame: A new dataframe with one-hot encoded columns for each unique clientid.
        """
        # Ensure 'clientid' column exists in the dataframe
        if 'clientid' not in self.dataframe.columns:
            raise ValueError("The dataframe does not contain a 'clientid' column.")

        # Perform one-hot encoding on the 'clientid' column
        one_hot_encoded_df = pd.get_dummies(self.dataframe, columns=['clientid'], prefix='client', drop_first=False)

        # Get the list of one-hot encoded columns (i.e., columns starting with 'client_')
        one_hot_columns = [col for col in one_hot_encoded_df.columns if col.startswith('client_')]

        # Get the list of other columns excluding the newly created one-hot columns
        other_columns = [col for col in one_hot_encoded_df.columns if col not in one_hot_columns]

        # Reorder the columns so that one-hot encoded columns come first
        reordered_columns = one_hot_columns + other_columns

        # Return the dataframe with reordered columns
        self.dataframe = one_hot_encoded_df[reordered_columns]

    def modify_timestamps(self, column_name='time'):
        """
        Modifies the timestamp column to numeric format (Unix timestamps).

        Parameters:
        column_name (str): The name of the timestamp column to modify. Default is 'time'.
        """
        # Ensure the time column exists in the dataframe
        if column_name not in self.dataframe.columns:
            raise ValueError(f"The dataframe does not contain a '{column_name}' column.")

        # Convert the column to datetime
        self.dataframe[column_name] = pd.to_datetime(self.dataframe[column_name])
        # Convert to Unix timestamp (seconds since epoch)
        self.dataframe[column_name] = self.dataframe[column_name].astype(
            int) // 10 ** 9  # Divide by 10^9 to convert nanoseconds to seconds

    def normalize_data(self, columns=None, method='minmax'):
        """
        Normalizes the specified columns of the dataframe to enhance ML performance.

        Parameters:
        columns (list): The list of columns to normalize. If None, all numeric columns are normalized. Default is None.
        method (str): The normalization method. Options are 'minmax' for Min-Max scaling or 'standard' for Z-score normalization. Default is 'minmax'.
        """
        # If no specific columns are provided, normalize all numeric columns
        if columns is None:
            columns = self.dataframe.select_dtypes(include=['float64', 'int']).columns.tolist()

        # Initialize the scaler
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("Normalization method must be either 'minmax' or 'standard'.")

        # Normalize the specified columns
        self.dataframe[columns] = scaler.fit_transform(self.dataframe[columns])

    def normalize_with_mean_power(self, power_column='agg'):
        """
        Normalizes the specified columns based on the mean of the aggregated power for each client.

        Parameters:
        power_column (str): The column used for normalization. Default is 'agg'.
        """
        # Ensure the specified power column exists in the dataframe
        if power_column not in self.dataframe.columns:
            raise ValueError(f"The dataframe does not contain a '{power_column}' column.")

        # Group by 'clientid' and calculate mean of the specified power column
        mean_power_by_client = self.dataframe.groupby('clientid')[power_column].mean()

        # Normalize the specified columns based on client mean power
        columns_to_normalize = self.dataframe.select_dtypes(include=['float64', 'int']).columns.tolist()
        columns_to_normalize.remove('time')  # Exclude the time column from normalization
        columns_to_normalize.remove('clientid')  # Exclude the clientid column from normalization

        # Normalize the columns for each client
        for client_id, client_data in self.dataframe.groupby('clientid'):
            client_mean_power = mean_power_by_client[client_id]
            # Normalize using the client's mean power
            self.dataframe.loc[self.dataframe['clientid'] == client_id, columns_to_normalize] /= client_mean_power

    def get_dataframe(self):
        """
        Returns the preprocessed dataframe.
        """
        return self.dataframe