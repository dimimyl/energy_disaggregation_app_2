from sklearn.ensemble import RandomForestRegressor
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.build_model()

    def build_model(self):
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def predict(self, X_test):
        return self.model.predict(X_test)
