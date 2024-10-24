# /models/base_model.py
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass
