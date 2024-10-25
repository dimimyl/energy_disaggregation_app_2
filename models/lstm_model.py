# /models/lstm_model.py
import tensorflow as tf
from .base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, input_shape, output_units):
        super().__init__()
        self.input_shape = input_shape
        self.output_units = output_units
        self.build_model()

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(200, activation='tanh', input_shape=self.input_shape, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(100, activation='relu', return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.output_units, activation='linear')
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        return self.model

    def train(self, X_train, y_train, epochs=10, batch_size=3600):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X_test):
        return self.model.predict(X_test)
