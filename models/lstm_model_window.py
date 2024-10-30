import tensorflow as tf
from .base_model import BaseModel


class LSTMModelWindow(BaseModel):
    def __init__(self, input_shape, output_units):
        """
        Initialize the LSTMModel with input shape and output units.

        Args:
            input_shape (tuple): Shape of the input data (timesteps, features).
            output_units (int): Number of units in the output layer.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_units = output_units
        self.build_model()  # Build the model during initialization

    def build_model(self):
        """
        Build the LSTM model architecture with a TimeDistributed output layer.
        """
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(200, activation='tanh', input_shape=self.input_shape, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(100, activation='relu', return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.output_units, activation='linear'))
        ])

        # Compile the model with mean squared error loss and adam optimizer
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X_train, y_train, validation_data=None, epochs=10, batch_size=3600):
        """
        Train the LSTM model.

        Args:
            X_train (np.ndarray): Training input data.
            y_train (np.ndarray): Training target data.
            validation_data (tuple, optional): Tuple (X_val, y_val) for validation data.
            epochs (int): Number of epochs to train.
            batch_size (int): Batch size for training.

        Returns:
            History: Training history.
        """
        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.

        Args:
            X_test (np.ndarray): Test input data.
            y_test (np.ndarray): Test target data.

        Returns:
            float: Test loss.
        """
        return self.model.evaluate(X_test, y_test)

    def predict(self, X_test):
        """
        Make predictions with the trained model.

        Args:
            X_test (np.ndarray): Input data for making predictions.

        Returns:
            np.ndarray: Predicted output.
        """
        return self.model.predict(X_test)
