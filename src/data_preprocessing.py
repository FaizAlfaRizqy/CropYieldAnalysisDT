import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the data by separating features and target variable.
    
    Args:
        data (pd.DataFrame): The input data
        
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
    """
    # Debug: print nama kolom
    print("Kolom yang tersedia:", data.columns.tolist())
    print("Bentuk data:", data.shape)
    
    # Kolom target yang benar adalah 'Yield_tons_per_hectare'
    X = data.drop('Yield_tons_per_hectare', axis=1)
    y = data['Yield_tons_per_hectare']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    # Define the preprocessing steps for numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Apply the transformations
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y

def handle_missing_values(data):
    # Fill missing values with the mean for numerical columns
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        data[col].fillna(data[col].mean(), inplace=True)
    
    # Fill missing values for categorical columns with the mode
    for col in data.select_dtypes(include=['object']).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    return data