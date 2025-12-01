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
import matplotlib.pyplot as plt
from src.model_xgboost import XGBoostModel
from src.visualize import (plot_feature_importance, plot_predictions, 
                           plot_residuals)

def preprocess_data_with_names(data):
    """
    Preprocess data and return feature names after transformation
    """
    print("Preprocessing data...")
    
    X = data.drop(['Yield_tons_per_hectare', 'Yield_Category'], axis=1)
    y = data['Yield_tons_per_hectare']
    
    print(f"Input features: {X.columns.tolist()}")
    print(f"Target: Yield_tons_per_hectare")
    
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object', 'bool']).columns.tolist()
    
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
    
    assert 'Yield_Category' not in ' '.join(feature_names), "Data leakage detected!"
    assert 'Yield_tons_per_hectare' not in ' '.join(feature_names), "Data leakage detected!"
    
    return X_processed, y, feature_names, preprocessor

def plot_training_history(model, save_path='reports/figures/xgboost_training_history.png'):
    """Plot XGBoost training history"""
    history = model.get_training_history()
    
    if history is None:
        print("No training history available")
        return
    
    plt.figure(figsize=(12, 5))
    
    # Plot RMSE
    plt.subplot(1, 2, 1)
    epochs = range(len(history['validation_0']['rmse']))
    plt.plot(epochs, history['validation_0']['rmse'], label='Train RMSE', linewidth=2)
    plt.plot(epochs, history['validation_1']['rmse'], label='Validation RMSE', linewidth=2)
    plt.xlabel('Boosting Round', fontsize=12, fontweight='bold')
    plt.ylabel('RMSE', fontsize=12, fontweight='bold')
    plt.title('XGBoost Training History - RMSE', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Find best iteration
    best_iteration = np.argmin(history['validation_1']['rmse'])
    best_rmse = history['validation_1']['rmse'][best_iteration]
    plt.axvline(x=best_iteration, color='red', linestyle='--', 
                label=f'Best: {best_iteration} ({best_rmse:.4f})')
    
    # Plot feature importance types comparison
    plt.subplot(1, 2, 2)
    importance_types = ['weight', 'gain', 'cover']
    importances_data = []
    
    for imp_type in importance_types:
        imp = model.get_feature_importance(importance_type=imp_type)
        importances_data.append(imp.mean())
    
    bars = plt.bar(importance_types, importances_data, color=['#4CAF50', '#2196F3', '#FF9800'])
    plt.xlabel('Importance Type', fontsize=12, fontweight='bold')
    plt.ylabel('Average Importance', fontsize=12, fontweight='bold')
    plt.title('Feature Importance by Type', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    Path('reports/figures').mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved to: {save_path}")
    plt.close()

def compare_models(metrics_dt, metrics_xgb, save_path='reports/model_comparison.csv'):
    """Compare Decision Tree vs XGBoost"""
    
    comparison_df = pd.DataFrame({
        'Model': ['Decision Tree', 'XGBoost'],
        'Train_R²': [metrics_dt['train_r2'], metrics_xgb['train_r2']],
        'Test_R²': [metrics_dt['test_r2'], metrics_xgb['test_r2']],
        'Test_MAE': [metrics_dt['test_mae'], metrics_xgb['test_mae']],
        'Test_RMSE': [metrics_dt['test_rmse'], metrics_xgb['test_rmse']],
        'Overfitting_Gap': [metrics_dt['overfitting_gap'], metrics_xgb['overfitting_gap']]
    })
    
    # Calculate improvement
    improvement = pd.DataFrame({
        'Metric': ['Test_R²', 'Test_MAE', 'Test_RMSE', 'Overfitting_Gap'],
        'Decision_Tree': [metrics_dt['test_r2'], metrics_dt['test_mae'], 
                         metrics_dt['test_rmse'], metrics_dt['overfitting_gap']],
        'XGBoost': [metrics_xgb['test_r2'], metrics_xgb['test_mae'], 
                   metrics_xgb['test_rmse'], metrics_xgb['overfitting_gap']],
        'Improvement': [
            (metrics_xgb['test_r2'] - metrics_dt['test_r2']) / metrics_dt['test_r2'] * 100,
            (metrics_dt['test_mae'] - metrics_xgb['test_mae']) / metrics_dt['test_mae'] * 100,
            (metrics_dt['test_rmse'] - metrics_xgb['test_rmse']) / metrics_dt['test_rmse'] * 100,
            (metrics_dt['overfitting_gap'] - metrics_xgb['overfitting_gap']) / abs(metrics_dt['overfitting_gap']) * 100
        ]
    })
    
    comparison_df.to_csv(save_path, index=False)
    improvement.to_csv('reports/model_improvement.csv', index=False)
    
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON: Decision Tree vs XGBoost")
    print("="*60)
    print(comparison_df.to_string(index=False))
    print("\n" + "="*60)
    print("📈 IMPROVEMENT ANALYSIS")
    print("="*60)
    print(improvement.to_string(index=False))
    
    return comparison_df, improvement

def main():
    print("="*60)
    print("Rice Yield Prediction - XGBoost Model Training")
    print("="*60)
    
    # Load the data
    print("\n1. Loading data...")
    data = pd.read_csv('data/processed/rice_yield_cleaned.csv')
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Preprocess the data
    print("\n2. Preprocessing data...")
    X, y, feature_names, preprocessor = preprocess_data_with_names(data)
    
    # Split the data (80% train, 10% validation, 10% test)
    print("\n3. Splitting data (train/val/test: 80/10/10)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111, random_state=42  # 0.111 * 0.9 ≈ 0.1
    )
    
    print(f"Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"Testing set:    {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Initialize XGBoost model with optimized parameters
    print("\n4. Training XGBoost model...")
    print("Hyperparameters:")
    print("   - n_estimators: 200 (number of trees)")
    print("   - max_depth: 6 (tree depth)")
    print("   - learning_rate: 0.1 (step size)")
    print("   - subsample: 0.8 (row sampling)")
    print("   - colsample_bytree: 0.8 (column sampling)")
    
    model = XGBoostModel(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0   # L2 regularization
    )
    
    # Train with validation set for early stopping
    model.train(X_train, y_train, X_val, y_val, early_stopping_rounds=20)
    
    # Evaluate model
    print("\n5. Evaluating model...")
    train_score, test_score = model.get_train_test_scores(X_train, y_train, X_test, y_test)
    val_score = model.evaluate(X_val, y_val)
    
    # Get all metrics
    train_metrics = model.get_all_metrics(X_train, y_train)
    val_metrics = model.get_all_metrics(X_val, y_val)
    test_metrics = model.get_all_metrics(X_test, y_test)
    
    print(f'\n📊 XGBoost Model R² Scores:')
    print(f'   Training R² Score:     {train_score:.4f} ({train_score*100:.2f}%)')
    print(f'   Validation R² Score:   {val_score:.4f} ({val_score*100:.2f}%)')
    print(f'   Testing R² Score:      {test_score:.4f} ({test_score*100:.2f}%)')
    
    overfitting_gap = train_score - test_score
    print(f'   Overfitting Gap:       {overfitting_gap:.4f}')
    
    if overfitting_gap > 0.1:
        print(f'   ⚠️  WARNING: Significant overfitting detected!')
    elif overfitting_gap > 0.05:
        print(f'   ⚠️  CAUTION: Mild overfitting detected')
    else:
        print(f'   ✅ Excellent generalization!')
    
    # Detailed metrics
    print(f'\n📈 Detailed Performance Metrics:')
    print(f'\n   TRAINING SET:')
    print(f'   - R² Score:  {train_metrics["R²"]:.4f}')
    print(f'   - MSE:       {train_metrics["MSE"]:.4f}')
    print(f'   - MAE:       {train_metrics["MAE"]:.4f}')
    print(f'   - RMSE:      {train_metrics["RMSE"]:.4f}')
    
    print(f'\n   VALIDATION SET:')
    print(f'   - R² Score:  {val_metrics["R²"]:.4f}')
    print(f'   - MSE:       {val_metrics["MSE"]:.4f}')
    print(f'   - MAE:       {val_metrics["MAE"]:.4f}')
    print(f'   - RMSE:      {val_metrics["RMSE"]:.4f}')
    
    print(f'\n   TESTING SET:')
    print(f'   - R² Score:  {test_metrics["R²"]:.4f}')
    print(f'   - MSE:       {test_metrics["MSE"]:.4f}')
    print(f'   - MAE:       {test_metrics["MAE"]:.4f}')
    print(f'   - RMSE:      {test_metrics["RMSE"]:.4f}')
    
    print(f'\n📊 Model Performance Interpretation:')
    print(f'   - Test R² = {test_score:.1%} of variance explained')
    print(f'   - Average error = ±{test_metrics["MAE"]:.2f} tons/hectare')
    print(f'   - RMSE = {test_metrics["RMSE"]:.2f} tons/hectare')
    
    if test_score >= 0.9:
        performance = "Excellent! 🎉"
    elif test_score >= 0.8:
        performance = "Very Good! ⭐"
    elif test_score >= 0.7:
        performance = "Good 👍"
    else:
        performance = "Fair 😐"
    
    print(f'   - Overall Performance: {performance}')
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Set': ['Train', 'Validation', 'Test'],
        'R²': [train_metrics['R²'], val_metrics['R²'], test_metrics['R²']],
        'MSE': [train_metrics['MSE'], val_metrics['MSE'], test_metrics['MSE']],
        'MAE': [train_metrics['MAE'], val_metrics['MAE'], test_metrics['MAE']],
        'RMSE': [train_metrics['RMSE'], val_metrics['RMSE'], test_metrics['RMSE']]
    })
    
    Path('reports').mkdir(exist_ok=True)
    metrics_df.to_csv('reports/xgboost_metrics.csv', index=False)
    print("\n✓ XGBoost metrics saved to: reports/xgboost_metrics.csv")
    
    # Predictions
    y_pred_test = model.predict(X_test)
    
    # Visualizations
    print("\n6. Generating visualizations...")
    
    # Feature importance
    feature_importance_df = plot_feature_importance(model, feature_names, top_n=15,
                                                    save_path='reports/figures/xgboost_feature_importance.png')
    feature_importance_df.to_csv('reports/xgboost_feature_importance.csv', index=False)
    print("✓ Feature importance saved")
    
    # Predictions plot
    plot_predictions(y_test, y_pred_test, save_path='reports/figures/xgboost_predictions.png')
    
    # Residuals plot
    plot_residuals(y_test, y_pred_test, save_path='reports/figures/xgboost_residuals.png')
    
    # Training history
    plot_training_history(model)
    
    # Save model
    print("\n7. Saving XGBoost model and preprocessor...")
    Path('models').mkdir(exist_ok=True)
    joblib.dump(model, 'models/xgboost_model.pkl')
    joblib.dump(preprocessor, 'models/xgboost_preprocessor.pkl')
    joblib.dump(feature_names, 'models/xgboost_feature_names.pkl')
    print("✓ XGBoost model saved to models/xgboost_model.pkl")
    print("✓ Preprocessor saved to models/xgboost_preprocessor.pkl")
    print("✓ Feature names saved to models/xgboost_feature_names.pkl")
    
    # Compare with Decision Tree (if metrics available)
    print("\n8. Comparing with Decision Tree...")
    dt_metrics_path = Path('reports/model_metrics.csv')
    if dt_metrics_path.exists():
        dt_metrics_df = pd.read_csv(dt_metrics_path)
        
        # Extract Decision Tree metrics
        metrics_dt = {
            'train_r2': dt_metrics_df[dt_metrics_df['Metric'] == 'Train_R²']['Value'].values[0],
            'test_r2': dt_metrics_df[dt_metrics_df['Metric'] == 'Test_R²']['Value'].values[0],
            'test_mae': dt_metrics_df[dt_metrics_df['Metric'] == 'Test_MAE']['Value'].values[0],
            'test_rmse': dt_metrics_df[dt_metrics_df['Metric'] == 'Test_RMSE']['Value'].values[0],
            'overfitting_gap': dt_metrics_df[dt_metrics_df['Metric'] == 'Overfitting_Gap']['Value'].values[0]
        }
        
        metrics_xgb = {
            'train_r2': train_score,
            'test_r2': test_score,
            'test_mae': test_metrics['MAE'],
            'test_rmse': test_metrics['RMSE'],
            'overfitting_gap': overfitting_gap
        }
        
        compare_models(metrics_dt, metrics_xgb)
    else:
        print("⚠️  Decision Tree metrics not found. Skipping comparison.")
        print("   Run 'python src/train.py' first to train Decision Tree model.")
    
    print("\n" + "="*60)
    print("XGBoost Training completed successfully!")
    print("="*60)
    print("\n🚀 XGBoost typically provides:")
    print("   ✅ Better accuracy than single Decision Tree")
    print("   ✅ Reduced overfitting through regularization")
    print("   ✅ More robust predictions")
    print("   ✅ Feature importance from multiple perspectives")

if __name__ == '__main__':
    main()