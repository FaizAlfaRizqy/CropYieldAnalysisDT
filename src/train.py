import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from src.model import DecisionTreeModel  # ← FIX: Pastikan ini lengkap dan benar
from src.visualize import (plot_feature_importance, plot_predictions, 
                           plot_residuals, plot_decision_tree_simple)
import argparse

def preprocess_data_with_names(data):
    """
    Preprocess data and return feature names after transformation
    """
    print("Preprocessing data...")
    
    X = data.drop(['Yield_tons_per_hectare', 'Yield_Category'], axis=1)
    y = data['Yield_tons_per_hectare']
    y_category = data['Yield_Category'] 
    
    print(f"Input features: {X.columns.tolist()}")
    print(f"Target: Yield_tons_per_hectare")
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"\nNumerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    feature_names = []
    
    feature_names.extend(numerical_cols)
    
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_['cat']
        for i, col in enumerate(categorical_cols):
            categories = cat_encoder.categories_[i]
            feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    print(f"\nTotal features after encoding: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    
    # Verify no leakage
    assert 'Yield_Category' not in ' '.join(feature_names), "Data leakage detected!"
    assert 'Yield_tons_per_hectare' not in ' '.join(feature_names), "Data leakage detected!"
    
    return X_processed, y, feature_names, preprocessor

def parse_args():
    parser = argparse.ArgumentParser(description='Train Decision Tree regressor with hyperparameters')
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--min_samples_split', type=int, default=20)
    parser.add_argument('--min_samples_leaf', type=int, default=10)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    print("="*60)
    print("Rice Yield Prediction - Model Training (REGRESSION)")
    print("="*60)
    
    print("\n1. Loading data...")
    data = pd.read_csv('data/processed/rice_yield_cleaned.csv')
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    print("\n2. Preprocessing data...")
    X, y, feature_names, preprocessor = preprocess_data_with_names(data)
   
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    print("\n4. Training model...")
    model = DecisionTreeModel(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state
    )
    
    model.train(X_train, y_train)
    
    print("\n5. Evaluating model...")
    train_score, test_score = model.get_train_test_scores(X_train, y_train, X_test, y_test)
    
    print(f'\n📊 Model R² Scores:')
    print(f'   Training R² Score:   {train_score:.4f} ({train_score*100:.2f}%)')
    print(f'   Testing R² Score:    {test_score:.4f} ({test_score*100:.2f}%)')
    
    overfitting_gap = train_score - test_score
    print(f'   Overfitting Gap:     {overfitting_gap:.4f}')
    
    if overfitting_gap > 0.1:
        print(f'   ⚠️  WARNING: Significant overfitting detected!')
        print(f'   Consider: increasing min_samples_split or decreasing max_depth')
    elif overfitting_gap > 0.05:
        print(f'   ⚠️  CAUTION: Mild overfitting detected')
    else:
        print(f'   ✅ Good generalization - minimal overfitting')
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    
    print(f'\n📈 Detailed Performance Metrics:')
    print(f'\n   TRAINING SET:')
    print(f'   - R² Score:  {train_score:.4f}')
    print(f'   - MSE:       {mse_train:.4f}')
    print(f'   - MAE:       {mae_train:.4f}')
    print(f'   - RMSE:      {rmse_train:.4f}')
    
    print(f'\n   TESTING SET:')
    print(f'   - R² Score:  {test_score:.4f}')
    print(f'   - MSE:       {mse:.4f}')
    print(f'   - MAE:       {mae:.4f}')
    print(f'   - RMSE:      {rmse:.4f}')
    
    print(f'\n📊 Model Performance Interpretation:')
    print(f'   - Test R² = {test_score:.1%} of variance explained')
    print(f'   - Average error = ±{mae:.2f} tons/hectare')
    print(f'   - RMSE = {rmse:.2f} tons/hectare')
    
    if test_score >= 0.8:
        performance = "Excellent 🎉"
    elif test_score >= 0.6:
        performance = "Good 👍" 
    elif test_score >= 0.4:
        performance = "Fair 😐"
    else:
        performance = "Poor 📉"
    
    print(f'   - Overall Performance: {performance}')
    
    metrics_df = pd.DataFrame({
        'Metric': ['Train_R²', 'Test_R²', 'Train_MSE', 'Test_MSE', 
                   'Train_MAE', 'Test_MAE', 'Train_RMSE', 'Test_RMSE', 'Overfitting_Gap'],
        'Value': [train_score, test_score, mse_train, mse, 
                  mae_train, mae, rmse_train, rmse, overfitting_gap],
        'Unit': ['ratio', 'ratio', 'tons²/ha²', 'tons²/ha²', 
                 'tons/ha', 'tons/ha', 'tons/ha', 'tons/ha', 'ratio']
    })
    
    Path('reports').mkdir(exist_ok=True)
    metrics_df.to_csv('reports/model_metrics.csv', index=False)
    print("\n✓ Model metrics saved to: reports/model_metrics.csv")
    
    y_pred = model.predict(X_test)
    
    print("\n6. Generating visualizations...")
    
    feature_importance_df = plot_feature_importance(model, feature_names, top_n=15)
    
    Path('reports').mkdir(exist_ok=True)
    feature_importance_df.to_csv('reports/feature_importance.csv', index=False)
    print("✓ Feature importance saved to: reports/feature_importance.csv")
    
    plot_predictions(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_decision_tree_simple(model, feature_names)
    
    print("\n7. Saving model and preprocessor...")
    Path('models').mkdir(exist_ok=True)
    joblib.dump(model, 'models/decision_tree_model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print("✓ Model saved to models/decision_tree_model.pkl")
    print("✓ Preprocessor saved to models/preprocessor.pkl")
    print("✓ Feature names saved to models/feature_names.pkl")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print("\nNow you can see the REAL feature importance!")
    print("Expected top features: Rainfall, Temperature, Days_to_Harvest, etc.")

if __name__ == '__main__':
    main()