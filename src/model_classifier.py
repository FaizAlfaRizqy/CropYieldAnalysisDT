from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd

class DecisionTreeClassifierModel:
    """Decision Tree Classifier for Yield Category Prediction"""
    
    def __init__(self, max_depth=10, min_samples_split=20, min_samples_leaf=10, **kwargs):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            **kwargs
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """Return accuracy score"""
        preds = self.predict(X)
        return accuracy_score(y, preds)
    
    def get_classification_report(self, X, y):
        """Get detailed classification report"""
        preds = self.predict(X)
        return classification_report(y, preds)
    
    def get_confusion_matrix(self, X, y):
        """Get confusion matrix"""
        preds = self.predict(X)
        return confusion_matrix(y, preds)
    
    def get_feature_importance(self):
        """Get feature importances"""
        return self.model.feature_importances_

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = joblib.load(file_path)