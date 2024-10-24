import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


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
        plt.xlabel('Time')
        plt.ylabel('Signal Value')
        plt.grid(True)
        plt.legend()
        plt.show()


def plot_predictions (predictions):
    plt.plot(predictions[:,4])
    plt.show()

