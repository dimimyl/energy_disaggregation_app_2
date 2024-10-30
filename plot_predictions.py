import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt


def plot_csv_columns(file_path):
    # Load the CSV file into a pandas DataFrame, specifying the correct delimiter
    df = pd.read_csv(file_path, delimiter=',', skipinitialspace=True)

    # Check if file has headers or not; use header=None only if no headers present
    # Assuming you want to plot the 4th column (index 3, because of 0-based indexing)

    plt.plot(df.iloc[:, 4])  # Adjust the index to match the column you want to plot
    plt.xlabel("Index")  # Label for the x-axis
    plt.ylabel("Value in 4th column")  # Label for the y-axis
    plt.title("Plot of 4th Column from CSV")  # Title of the plot
    plt.show()


def plot_client_data_from_csv(x_file, y_file):
    # Read the CSV files into DataFrames
    x = pd.read_csv(x_file)
    y = pd.read_csv(y_file)

    # Get the list of client columns (assuming they all start with 'client_')
    client_columns = [col for col in x.columns if col.startswith('client_')]

    # Loop through each client column
    for client_col in client_columns:
        # Filter rows where the client is True
        client_data = x[x[client_col] == True]

        if not client_data.empty:
            # Get the corresponding y values for the filtered time
            time_indices = client_data.index
            y_data = y.loc[time_indices]

            # Plot the data
            plt.figure(figsize=(10, 6))
            plt.plot(client_data['time'], client_data['agg'], label='agg', color='blue', linewidth=2)

            plt.plot(client_data['time'], y_data['st'], label='st', color='green', linestyle='--')
            plt.plot(client_data['time'], y_data['wh'], label='wh', color='orange', linestyle='--')
            plt.plot(client_data['time'], y_data['wm'], label='wm', color='red', linestyle='--')
            plt.plot(client_data['time'], y_data['ac_power'], label='ac_power', color='purple', linestyle='--')
            plt.plot(client_data['time'], y_data['fridge_power'], label='fridge_power', color='brown', linestyle='--')

            # Labels and title
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.title(f'Data for {client_col}')
            plt.legend()

            # Show plot
            plt.show()


plot_csv_columns('output/predictions.csv')
