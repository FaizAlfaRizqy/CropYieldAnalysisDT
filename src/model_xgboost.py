import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import pandas as pd
import numpy as np

class XGBoostModel:
    """XGBoost Regressor for Rice Yield Prediction"""
    
    def __init__(self, 
                 n_estimators=100,
                 max_depth=6,
                 learning_rate=0.1,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 min_child_weight=1,
                 gamma=0,
                 reg_alpha=0,
                 reg_lambda=1,
                 **kwargs):
        """
        Initialize XGBoost Regressor
        
        Parameters:
        - n_estimators: Number of boosting rounds (trees)
        - max_depth: Maximum depth of each tree
        - learning_rate: Step size shrinkage (eta)
        - subsample: Subsample ratio of training instances
        - colsample_bytree: Subsample ratio of columns when constructing each tree
        - min_child_weight: Minimum sum of instance weight needed in a child
        - gamma: Minimum loss reduction required to make a split
        - reg_alpha: L1 regularization term on weights
        - reg_lambda: L2 regularization term on weights
        """
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            **kwargs
        )
        self.training_history = None

    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10):
        """
        Train XGBoost model with optional early stopping
        
        Parameters:
        - X_train: Training features
        - y_train: Training target
        - X_val: Validation features (optional)
        - y_val: Validation target (optional)
        - early_stopping_rounds: Stop if no improvement for N rounds
        """
        if X_val is not None and y_val is not None:
            # Train with early stopping
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # Get training history
            self.training_history = self.model.evals_result()
        else:
            # Train without early stopping
            self.model.fit(X_train, y_train)

    def predict(self, X):
        """Make predictions"""
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
    
    def get_all_metrics(self, X, y):
        """Get comprehensive metrics"""
        preds = self.predict(X)
        
        r2 = r2_score(y, preds)
        mse = mean_squared_error(y, preds)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mse)
        
        return {
            'R²': r2,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse
        }
    
    def get_feature_importance(self, importance_type='weight'):
        """
        Get feature importances
        
        Parameters:
        - importance_type: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
        """
        if importance_type == 'weight':
            return self.model.feature_importances_
        else:
            importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
            # Convert to array matching feature order
            return np.array([importance_dict.get(f'f{i}', 0) for i in range(len(self.model.feature_importances_))])
    
    def get_training_history(self):
        """Get training history (loss per iteration)"""
        return self.training_history

    def save_model(self, file_path):
        """Save model to file"""
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        """Load model from file"""
        self.model = joblib.load(file_path)