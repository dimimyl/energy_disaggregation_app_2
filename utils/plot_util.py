import pandas as pd
import matplotlib.pyplot as plt

def plot_devices(dataframe):
    """
    Plots each numeric column (agg, wm, st, wh, ac_power, fridge_power) over time for each unique client ID.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing the data.
    """
    # Ensure 'time' column is a datetime type
    dataframe['time'] = pd.to_datetime(dataframe['time'])

    # Group the data by 'clientid'
    grouped = dataframe.groupby('clientid')

    # Plot for each client
    for clientid, client_data in grouped:
        # Set 'time' column as the index for better plotting
        client_data.set_index('time', inplace=True)

        # Plot the relevant columns for each client
        plt.figure(figsize=(12, 8))

        # Plot each signal (agg, wm, st, wh, ac_power, fridge_power)
        for column in ['agg', 'wm', 'st', 'wh', 'ac_power', 'fridge_power']:
            if column in client_data.columns:
                plt.plot(client_data.index, client_data[column], label=column)

        # Set plot details
        plt.title(f"Signals Over Time for Client: {clientid}")
        plt.xlabel('Time (s)')
        plt.ylabel('Apparent Power (VA)')
        plt.grid(True)
        plt.legend()
        plt.show()


def plot_preprocessed_dataset(file_path):

    """
    This function plots the signals for each house after preprocessing

    Parameters:
    filepath: The path of the csv file containing preprocessed dataset

    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, delimiter=',', skipinitialspace=True)

    # Get unique client IDs
    client_ids = df['clientid'].unique()

    # Create a separate plot for each client ID
    for client_id in client_ids:
        # Filter the data for the current client
        client_data = df[df['clientid'] == client_id]

        # Set the figure size
        plt.figure(figsize=(12, 6))

        # Plotting each variable
        plt.plot(client_data['time'], client_data['agg'], label='Aggregate', alpha=0.7)
        plt.plot(client_data['time'], client_data['wm'], label='Washing Machine', alpha=0.7)
        plt.plot(client_data['time'], client_data['st'], label='Standby', alpha=0.7)
        plt.plot(client_data['time'], client_data['wh'], label='Water Heater', alpha=0.7)
        plt.plot(client_data['time'], client_data['ac_power'], label='AC Power', alpha=0.7)
        plt.plot(client_data['time'], client_data['fridge_power'], label='Fridge Power', alpha=0.7)

        # Adding labels and title
        plt.xlabel('Time')
        plt.ylabel('Power (W)')
        plt.title(f'Energy Consumption for Client ID {client_id}')
        plt.legend()
        plt.grid()

        # Show the plot
        plt.tight_layout()
        plt.show()



