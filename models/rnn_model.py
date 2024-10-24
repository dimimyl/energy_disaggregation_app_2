# /models/rnn_model.py
import tensorflow as tf
from .base_model import BaseModel

class RNNModel(BaseModel):
    def __init__(self, input_shape, output_units):
        super().__init__()
        self.input_shape = input_shape
        self.output_units = output_units
        self.build_model()

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.SimpleRNN(64, activation='tanh', input_shape=self.input_shape),
            tf.keras.layers.Dense(self.output_units)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=False)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X_test):
        return self.model.predict(X_test)
