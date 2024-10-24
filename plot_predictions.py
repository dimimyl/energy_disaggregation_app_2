import pandas as pd
import matplotlib.pyplot as plt


def plot_csv_columns(file_path):

# Load the CSV file into a pandas DataFrame, specifying the correct delimiter
    df = pd.read_csv(file_path,header=None, delimiter=',', skipinitialspace=True)
    print(df.head(10))
    print(df.shape)

plot_csv_columns('predictions.csv')
