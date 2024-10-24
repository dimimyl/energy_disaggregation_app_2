# /models/knn_model.py
from sklearn.neighbors import KNeighborsRegressor
from .base_model import BaseModel

class KNNModel(BaseModel):
    def __init__(self, n_neighbors=5):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.build_model()

    def build_model(self):
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def predict(self, X_test):
        return self.model.predict(X_test)
