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
from src.data_preprocessing import load_data
from src.model import DecisionTreeModel
from src.visualize import (plot_feature_importance, plot_predictions, 
                           plot_residuals, plot_decision_tree_simple)

def preprocess_data_with_names(data):
    """
    Preprocess data and return feature names after transformation
    """
    print("Preprocessing data...")
    
    # Drop BOTH target columns (nilai dan kategori)
    X = data.drop(['Yield_tons_per_hectare', 'Yield_Category'], axis=1)
    y = data['Yield_tons_per_hectare']
    y_category = data['Yield_Category']  # Simpan untuk referensi
    
    print(f"Input features: {X.columns.tolist()}")
    print(f"Target: Yield_tons_per_hectare")
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"\nNumerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Create transformers
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after transformation
    feature_names = []
    
    # Add numerical feature names
    feature_names.extend(numerical_cols)
    
    # Add categorical feature names (after one-hot encoding)
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

def main():
    print("="*60)
    print("Rice Yield Prediction - Model Training")
    print("="*60)
    
    # Load the data
    print("\n1. Loading data...")
    data = load_data('data/processed/rice_yield_cleaned.csv')
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Preprocess the data and get feature names
    print("\n2. Preprocessing data...")
    X, y, feature_names, preprocessor = preprocess_data_with_names(data)
    
    # Split the data into training and testing sets
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Initialize the model
    print("\n4. Training model...")
    model = DecisionTreeModel(max_depth=10, min_samples_split=20, min_samples_leaf=10)
    
    # Train the model
    model.train(X_train, y_train)
    
    # Evaluate the model
    print("\n5. Evaluating model...")
    score = model.evaluate(X_test, y_test)
    
    # Generate predictions for detailed evaluation
    y_pred = model.predict(X_test)
    
    # Calculate additional metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Print all metrics
    print(f'Model R² Score: {r2:.4f}')
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}') 
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    
    # Interpretation
    print(f'\n📊 Model Performance Interpretation:')
    print(f'   - R² = {r2:.1%} of variance explained')
    print(f'   - Average error = ±{mae:.2f} tons/hectare')
    print(f'   - RMSE = {rmse:.2f} tons/hectare')
    
    # Performance category
    if r2 >= 0.8:
        performance = "Excellent"
    elif r2 >= 0.6:
        performance = "Good" 
    elif r2 >= 0.4:
        performance = "Fair"
    else:
        performance = "Poor"
    
    print(f'   - Overall Performance: {performance}')
    
    # Save metrics to file
    metrics_df = pd.DataFrame({
        'Metric': ['R²', 'MSE', 'MAE', 'RMSE'],
        'Value': [r2, mse, mae, rmse],
        'Unit': ['ratio', 'tons²/ha²', 'tons/ha', 'tons/ha']
    })
    
    Path('reports').mkdir(exist_ok=True)
    metrics_df.to_csv('reports/model_metrics.csv', index=False)
    print("✓ Model metrics saved to: reports/model_metrics.csv")
    
    # Generate predictions for visualization
    y_pred = model.predict(X_test)
    
    # Create visualizations
    print("\n6. Generating visualizations...")
    
    # Feature importance (most important!)
    feature_importance_df = plot_feature_importance(model, feature_names, top_n=15)
    
    # Save feature importance to CSV
    Path('reports').mkdir(exist_ok=True)
    feature_importance_df.to_csv('reports/feature_importance.csv', index=False)
    print("✓ Feature importance saved to: reports/feature_importance.csv")
    
    # Other visualizations
    plot_predictions(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_decision_tree_simple(model, feature_names)
    
    # Save the trained model and preprocessor
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