from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import joblib
import pandas as pd

class DecisionTreeModel:
    def __init__(self, **kwargs):
        self.model = DecisionTreeRegressor(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return r2_score(y, preds)

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = joblib.load(file_path)