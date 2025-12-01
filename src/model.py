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
        """Return R² score"""
        preds = self.predict(X)
        return r2_score(y, preds)
    
    def get_train_test_scores(self, X_train, y_train, X_test, y_test):
        """Get both training and testing R² scores"""
        train_score = self.evaluate(X_train, y_train)
        test_score = self.evaluate(X_test, y_test)
        return train_score, test_score
    
    def get_feature_importance(self):
        """Get feature importances"""
        return self.model.feature_importances_

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = joblib.load(file_path)