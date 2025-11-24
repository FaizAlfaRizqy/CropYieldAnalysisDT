import pandas as pd
import numpy as np
from pathlib import Path
import os

def clean_and_filter_data(input_path='data/raw/crop_yield.csv', 
                          output_path='data/processed/rice_yield_cleaned.csv'):
    """
    Clean data: fokus pada Rice crop, hapus Region dan Days_to_Harvest, tambah kategori yield
    
    Args:
        input_path: Path ke raw data
        output_path: Path untuk menyimpan cleaned data
    
    Returns:
        pd.DataFrame: Cleaned data
    """
    print("="*60)
    print("Data Cleaning Process")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    data = pd.read_csv(input_path)
    print(f"   Original data shape: {data.shape}")
    print(f"   Columns: {data.columns.tolist()}")
    
    # Filter hanya Rice crop
    print("\n2. Filtering for Rice crop only...")
    print(f"   Crop distribution before filtering:")
    print(data['Crop'].value_counts())
    
    rice_data = data[data['Crop'] == 'Rice'].copy()
    print(f"\n   Data shape after filtering: {rice_data.shape}")
    print(f"   Rice samples: {len(rice_data)}")
    
    # Drop Region column
    print("\n3. Removing Region column...")
    if 'Region' in rice_data.columns:
        rice_data = rice_data.drop('Region', axis=1)
        print(f"   ✓ Region column removed")
    
    # Drop Crop column (semua Rice sekarang)
    print("\n4. Removing Crop column (all Rice now)...")
    if 'Crop' in rice_data.columns:
        rice_data = rice_data.drop('Crop', axis=1)
        print(f"   ✓ Crop column removed")
    
    # Drop Days_to_Harvest (data leakage - info yang baru diketahui setelah panen)
    print("\n5. Removing Days_to_Harvest column...")
    if 'Days_to_Harvest' in rice_data.columns:
        rice_data = rice_data.drop('Days_to_Harvest', axis=1)
        print(f"   ✓ Days_to_Harvest removed (prevents data leakage)")
        print(f"   Reason: Harvest duration is unknown at planting time")
    
    # Remove low importance categorical features (based on feature importance analysis)
    print("\n6. Removing low importance categorical features...")
    low_importance_features = [
        'Soil_Type',      # All soil type categories have ≤ 0.01% importance
        'Weather_Condition'  # Weather condition categories have ≤ 0.01% importance
    ]
    
    removed_features = []
    for feature in low_importance_features:
        if feature in rice_data.columns:
            rice_data = rice_data.drop(feature, axis=1)
            removed_features.append(feature)
            print(f"   ✓ {feature} removed (feature importance ≤ 0.01%)")
    
    if removed_features:
        print(f"   Total low importance features removed: {len(removed_features)}")
        print(f"   Features removed: {removed_features}")
    else:
        print("   No low importance features to remove")
    
    # Analyze yield distribution
    print("\n7. Analyzing yield distribution...")
    yield_stats = rice_data['Yield_tons_per_hectare'].describe()
    print(yield_stats)
    
    # Calculate percentiles for categorization
    q33 = rice_data['Yield_tons_per_hectare'].quantile(0.33)
    q67 = rice_data['Yield_tons_per_hectare'].quantile(0.67)
    
    print(f"\n   Yield thresholds:")
    print(f"   - Low Yield: < {q33:.2f} tons/hectare")
    print(f"   - Medium Yield: {q33:.2f} - {q67:.2f} tons/hectare")
    print(f"   - High Yield: > {q67:.2f} tons/hectare")
    
    # Add yield category
    print("\n8. Adding yield category...")
    def categorize_yield(yield_value):
        if yield_value < q33:
            return 'Low Yield'
        elif yield_value < q67:
            return 'Medium Yield'
        else:
            return 'High Yield'
    
    rice_data['Yield_Category'] = rice_data['Yield_tons_per_hectare'].apply(categorize_yield)
    
    # Show category distribution
    print("\n   Yield category distribution:")
    print(rice_data['Yield_Category'].value_counts().sort_index())
    
    # Check for missing values
    print("\n9. Checking for missing values...")
    missing = rice_data.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("   ✓ No missing values found")
    
    # Reorder columns (Yield_Category di akhir)
    cols = [col for col in rice_data.columns if col not in ['Yield_tons_per_hectare', 'Yield_Category']]
    cols.extend(['Yield_tons_per_hectare', 'Yield_Category'])
    rice_data = rice_data[cols]
    
    # Save cleaned data
    print("\n10. Saving cleaned data...")
    
    # Create full absolute path
    output_path_abs = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path_abs)
    
    # Create directory dengan os.makedirs
    if not os.path.exists(output_dir):
        print(f"   Creating directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"   Saving to: {output_path_abs}")
    rice_data.to_csv(output_path_abs, index=False)
    print(f"   ✓ Cleaned data saved successfully!")
    
    # Verify file exists
    if os.path.exists(output_path_abs):
        file_size = os.path.getsize(output_path_abs) / (1024 * 1024)  # MB
        print(f"   File size: {file_size:.2f} MB")
    
    # Summary
    print("\n" + "="*60)
    print("Data Cleaning Summary")
    print("="*60)
    print(f"Original data: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Cleaned data: {rice_data.shape[0]} rows, {rice_data.shape[1]} columns")
    print(f"\n✓ Removed columns: Region, Crop, Days_to_Harvest")
    print(f"✓ Removed low importance features: {removed_features}")
    print(f"✓ Added column: Yield_Category")
    print(f"\nRemaining INPUT features: {[col for col in rice_data.columns if col not in ['Yield_tons_per_hectare', 'Yield_Category']]}")
    print(f"\nSample of cleaned data:")
    print(rice_data.head(10))
    
    return rice_data

def analyze_cleaned_data(data_path='data/processed/rice_yield_cleaned.csv'):
    """
    Analyze cleaned data
    """
    print("\n" + "="*60)
    print("Data Analysis")
    print("="*60)
    
    # Use absolute path
    data_path_abs = os.path.abspath(data_path)
    
    if not os.path.exists(data_path_abs):
        print(f"ERROR: File not found: {data_path_abs}")
        return None
    
    data = pd.read_csv(data_path_abs)
    
    print("\n1. Data Shape:", data.shape)
    print("\n2. Data Types:")
    print(data.dtypes)
    
    print("\n3. INPUT Features (what farmer knows at planting):")
    input_features = [col for col in data.columns if col not in ['Yield_tons_per_hectare', 'Yield_Category']]
    for feat in input_features:
        print(f"   - {feat}")
    
    print("\n4. Statistical Summary:")
    print(data.describe())
    
    print("\n5. Yield Category Distribution:")
    category_dist = data['Yield_Category'].value_counts()
    print(category_dist)
    print("\nPercentage:")
    print(category_dist / len(data) * 100)
    
    # Update groupby analysis to handle removed columns
    if 'Soil_Type' in data.columns:
        print("\n6. Yield by Soil Type:")
        print(data.groupby('Soil_Type')['Yield_tons_per_hectare'].agg(['mean', 'std', 'count']))
    else:
        print("\n6. Soil_Type column not available (removed due to low importance)")
    
    if 'Weather_Condition' in data.columns:
        print("\n7. Yield by Weather Condition:")
        print(data.groupby('Weather_Condition')['Yield_tons_per_hectare'].agg(['mean', 'std', 'count']))
    else:
        print("\n7. Weather_Condition column not available (removed due to low importance)")
    
    if 'Fertilizer_Used' in data.columns:
        print("\n8. Yield by Fertilizer Use:")
        print(data.groupby('Fertilizer_Used')['Yield_tons_per_hectare'].agg(['mean', 'std', 'count']))
    
    if 'Irrigation_Used' in data.columns:
        print("\n9. Yield by Irrigation Use:")
        print(data.groupby('Irrigation_Used')['Yield_tons_per_hectare'].agg(['mean', 'std', 'count']))
    
    print("\n10. Correlation with Yield:")
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    correlations = data[numerical_cols].corr()['Yield_tons_per_hectare'].sort_values(ascending=False)
    print(correlations)
    
    return data

if __name__ == '__main__':
    # Clean data
    cleaned_data = clean_and_filter_data()
    
    # Analyze cleaned data
    print("\n\n")
    analyze_cleaned_data()
    
    print("\n" + "="*60)
    print("Data cleaning completed successfully!")
    print("="*60)
    print("\n📝 Note: Days_to_Harvest removed to prevent data leakage")
    print("   Low importance features (≤ 0.01%) removed for better model efficiency")
    print("   Only features available at PLANTING TIME are used for prediction")