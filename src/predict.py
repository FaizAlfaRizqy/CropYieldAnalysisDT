import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from src.data_preprocessing import load_data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_model(model_path):
    return joblib.load(model_path)

def create_preprocessor(sample_data):
    """Recreate the preprocessor used during training"""
    X_sample = sample_data.drop('Yield_tons_per_hectare', axis=1)
    
    categorical_cols = X_sample.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_sample.select_dtypes(exclude=['object']).columns.tolist()
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Fit preprocessor with sample data
    preprocessor.fit(X_sample)
    return preprocessor

def make_prediction(model, input_data):
    return model.predict(input_data)

def predict(input_data, model_path='models/decision_tree_model.pkl'):
    model = load_model(model_path)
    
    # Load sample data untuk fit preprocessor
    sample_data = load_data('data/raw/crop_yield.csv')
    preprocessor = create_preprocessor(sample_data)
    
    # Preprocess input
    processed_data = preprocessor.transform(input_data)
    
    # Predict
    predictions = make_prediction(model, processed_data)
    return predictions

# Contoh usage
if __name__ == '__main__':
    # Contoh data input
    sample_input = pd.DataFrame({
        'Region': ['West'],
        'Soil_Type': ['Sandy'],
        'Crop': ['Rice'],
        'Rainfall_mm': [500.0],
        'Temperature_Celsius': [25.0],
        'Fertilizer_Used': [True],
        'Irrigation_Used': [True],
        'Weather_Condition': ['Cloudy'],
        'Days_to_Harvest': [140]
    })
    
    try:
        prediction = predict(sample_input)
        print(f"\n{'='*50}")
        print(f"Predicted Yield: {prediction[0]:.2f} tons/hectare")
        print(f"{'='*50}")
        
        # Tampilkan input
        print("\nInput Data:")
        for col, val in sample_input.iloc[0].items():
            print(f"  {col}: {val}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()