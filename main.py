# /main.py
from data.create_windows import WindowedDataset
from data.data_loader import DataLoader
from data.preprocessor import Preprocessor
from data.dataset_splitter import DatasetSplitter
from utils.plot_util import plot_devices, plot_preprocessed_dataset
from models.rnn_model import RNNModel
from models.lstm_model import LSTMModel
from models.lstm_model_window import LSTMModelWindow
from models.knn_model import KNNModel
from models.random_forest_model import RandomForestModel
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

# Database configuration
db_config = {
    'dbname': 'smartmeterlogs',
    'user': 'readsmartmetermech',
    'password': 'zocsyx-qogcEf-saxbe8',
    'host': 'smartmetermech.chu4s6qua02r.eu-central-1.rds.amazonaws.com',
    'port': '9001'
}

houses_config = [
    {'clientid': 'house12', 'deviceid': 'st_wh_wm', 'plugid1': 'ac', 'plugid2': 'fridge'},
    {'clientid': 'house14', 'deviceid': 'st_wh_wm', 'plugid1': 'ac', 'plugid2': 'fridge'},
    {'clientid': 'house15', 'deviceid': 'st_wh_wm', 'plugid1': 'ac', 'plugid2': 'fridge'},
    {'clientid': 'house16', 'deviceid': 'st_wh_wm', 'plugid1': 'ac1', 'plugid2': 'fridge'},
]

# Load the data
loader = DataLoader(db_config)

# Function to load data from CSV
def load_data_from_csv(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    return df

# Function to fetch data using DataLoader
def load_data_from_dataloader():
    loader = DataLoader(db_config)
    merged_df = loader.fetch_data(houses_config)
    return merged_df

"""########################################################"""
""" #####################  Load Dataset ###################"""
"""########################################################"""
# Choose data loading method
use_csv = input("Do you want to use an existing CSV file for the dataset? (yes/no): ").strip().lower()

if use_csv == 'yes':
    merged_df = load_data_from_csv('output/combined_dataset.csv')
else:
    # Fetch data from DataLoader
    merged_df = load_data_from_dataloader()

"""########################################################"""
""" Plot Aggregated and Disaggregate signal for each client"""
"""########################################################"""
#plot_devices(merged_df)

"""########################################################"""
""" ################ Preprocess Dataset ##################"""
"""########################################################"""
preprocessor = Preprocessor(merged_df)
preprocessor.forward_fill()
preprocessor.modify_clientid()
preprocessor.modify_timestamps()
#preprocessor.one_hot_encode_clients()
preprocessor.normalize_with_mean_power()
# Get the preprocessed dataframe
preprocessed_df = preprocessor.get_dataframe()
# Save preprocessed dataframe to csv
preprocessed_df.to_csv('output/preprocessed_dataset.csv', index=False)
#plot_preprocessed_dataset('output/preprocessed_dataset.csv')

"""########################################################"""
""" ################### Split Dataset #####################"""
"""########################################################"""

target_columns = ['wm', 'st', 'wh', 'ac_power', 'fridge_power']  # Example target columns
splitter = DatasetSplitter(preprocessed_df, target_columns)
X_train, y_train, X_eval, y_eval, X_test, y_test = splitter.split()
splitter.save_splits_to_csv(output_dir='output')

"""########################################################"""
""" ############# Separate Dataset in Windows #############"""
"""########################################################"""
window_size=10
overlap=0.2
# Initialize the dataset
windowed_dataset = WindowedDataset(X_train, X_eval, X_test, y_train, y_eval, y_test, window_size=window_size, overlap=overlap)

X_train_windowed, y_train_windowed, X_eval_windowed, y_eval_windowed, X_test_windowed, y_test_windowed = windowed_dataset.prepare_data_for_lstm()
print(X_train.shape, y_train.shape)

"""
# Split dataset into train evaluation and test part when hot encode clients
target_columns = ['st', 'wh','wm', 'ac_power', 'fridge_power']
client_columns = [col for col in preprocessed_df.columns if col.startswith('client_')]
splitter = DatasetSplitterHotEncoding(preprocessed_df, target_columns=target_columns, client_columns=client_columns)
X_train, y_train, X_eval, y_eval, X_test, y_test = splitter.split()
splitter.save_splits_to_csv(output_dir='output')

plot_client_data('output/X_eval.csv', 'output/y_eval.csv')
"""


# Choose model type
# Uncomment one of the models you want to use:

#model = RNNModel(input_shape=(X_train.shape[1], 1), output_units=y_train.shape[1])
#model = LSTMModel(input_shape=(X_train.shape[1], 1), output_units=y_train.shape[1])
model = LSTMModelWindow(input_shape=(window_size, X_train_windowed.shape[2]), output_units=y_train_windowed.shape[2])
#model = KNNModel(n_neighbors=500)  # Default to KNN
#model = RandomForestModel(n_estimators=200, max_depth=10, random_state=42)

# Train and evaluate model
model.train(X_train_windowed, y_train_windowed)
loss = model.evaluate(X_eval_windowed, y_eval_windowed)
predictions = model.predict(X_test_windowed)

predictions_reconstructed=windowed_dataset.reconstruct_predictions(predictions)

print('Model accuracy is:',r2_score(y_test,predictions_reconstructed))
# Save predictions to CSV
np.savetxt('output/predictions.csv', predictions, delimiter=',')
print("Predictions saved to 'predictions.csv'")






