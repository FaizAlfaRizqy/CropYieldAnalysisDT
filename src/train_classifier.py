import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix)
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from src.model_classifier import DecisionTreeClassifierModel
from src.visualize import plot_feature_importance, plot_decision_tree_simple

def preprocess_data_for_classification(data):
    """
    Preprocess data for classification task (predicting Yield_Category)
    """
    print("Preprocessing data for classification...")
    
    # Features (X) dan target kategori (y)
    X = data.drop(['Yield_tons_per_hectare', 'Yield_Category'], axis=1)
    y = data['Yield_Category']  # ← Target sekarang kategori
    
    print(f"Input features: {X.columns.tolist()}")
    print(f"Target classes: {sorted(y.unique())}")
    print(f"\nClass distribution:")
    print(y.value_counts().sort_index())
    print(f"\nClass percentages:")
    print(y.value_counts(normalize=True).sort_index() * 100)
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object', 'bool']).columns.tolist()
    
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
    
    # Get feature names
    feature_names = []
    feature_names.extend(numerical_cols)
    
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_['cat']
        for i, col in enumerate(categorical_cols):
            categories = cat_encoder.categories_[i]
            feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    print(f"\nTotal features after encoding: {len(feature_names)}")
    
    return X_processed, y, feature_names, preprocessor

def plot_confusion_matrix(y_true, y_pred, classes, save_path='reports/figures/confusion_matrix.png'):
    """
    Plot confusion matrix with percentages
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    # Plot heatmap
    sns.heatmap(cm, 
                annot=annot, 
                fmt='', 
                cmap='Blues',
                xticklabels=classes,
                yticklabels=classes,
                cbar_kws={'label': 'Count'},
                linewidths=0.5,
                linecolor='gray')
    
    plt.title('Confusion Matrix - Rice Yield Category Prediction', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Category', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Category', fontsize=12, fontweight='bold')
    
    # Add accuracy
    accuracy = accuracy_score(y_true, y_pred)
    plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.3f}', 
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    Path('reports/figures').mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()
    
    return cm

def plot_classification_metrics(metrics_df, save_path='reports/figures/classification_metrics.png'):
    """
    Plot classification metrics by class
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx]
        bars = ax.bar(metrics_df['Class'], metrics_df[metric], color=color, alpha=0.7)
        ax.set_title(f'{metric} by Class', fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=metrics_df[metric].mean(), color='red', linestyle='--', 
                   label=f'Avg: {metrics_df[metric].mean():.3f}')
        ax.legend()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Classification metrics plot saved to: {save_path}")
    plt.close()

def main():
    print("="*60)
    print("Rice Yield Category Classification - Model Training")
    print("="*60)
    
    # Load the data
    print("\n1. Loading data...")
    data = pd.read_csv('data/processed/rice_yield_cleaned.csv')
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Preprocess for classification
    print("\n2. Preprocessing data for classification...")
    X, y, feature_names, preprocessor = preprocess_data_for_classification(data)
    
    # Split the data with stratification
    print("\n3. Splitting data with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Check class distribution in splits
    print("\nClass distribution in train set:")
    print(pd.Series(y_train).value_counts().sort_index())
    print("\nClass distribution in test set:")
    print(pd.Series(y_test).value_counts().sort_index())
    
    # Initialize and train model
    print("\n4. Training Decision Tree Classifier...")
    model = DecisionTreeClassifierModel(
        max_depth=10, 
        min_samples_split=20, 
        min_samples_leaf=10
    )
    model.train(X_train, y_train)
    
    # Evaluate model
    print("\n5. Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\n📊 Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    
    # Detailed classification report
    print("\n6. Classification Report:")
    print("="*60)
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Confusion Matrix
    print("\n7. Confusion Matrix:")
    classes = sorted(y.unique())
    cm = plot_confusion_matrix(y_test, y_pred, classes)
    print(cm)
    
    # Per-class metrics
    print("\n8. Detailed Metrics by Class:")
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=classes
    )
    
    metrics_df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    print(metrics_df.to_string(index=False))
    
    # Plot metrics
    plot_classification_metrics(metrics_df)
    
    # Save metrics
    Path('reports').mkdir(exist_ok=True)
    metrics_df.to_csv('reports/classification_metrics.csv', index=False)
    print("\n✓ Classification metrics saved to: reports/classification_metrics.csv")
    
    # Overall metrics summary
    overall_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1'],
        'Value': [
            accuracy,
            precision.mean(),
            recall.mean(),
            f1.mean()
        ]
    })
    overall_metrics.to_csv('reports/overall_classification_metrics.csv', index=False)
    print("✓ Overall metrics saved to: reports/overall_classification_metrics.csv")
    
    # Feature importance
    print("\n9. Generating visualizations...")
    feature_importance_df = plot_feature_importance(model, feature_names, top_n=15)
    feature_importance_df.to_csv('reports/feature_importance_classifier.csv', index=False)
    print("✓ Feature importance saved")
    
    # Decision tree visualization
    plot_decision_tree_simple(model, feature_names, 
                             save_path='reports/figures/decision_tree_classifier.png')
    
    # Save the trained model
    print("\n10. Saving model and preprocessor...")
    Path('models').mkdir(exist_ok=True)
    joblib.dump(model, 'models/decision_tree_classifier.pkl')
    joblib.dump(preprocessor, 'models/preprocessor_classifier.pkl')
    joblib.dump(feature_names, 'models/feature_names_classifier.pkl')
    joblib.dump(classes, 'models/class_labels.pkl')
    print("✓ Classifier model saved to models/decision_tree_classifier.pkl")
    print("✓ Preprocessor saved to models/preprocessor_classifier.pkl")
    print("✓ Feature names saved to models/feature_names_classifier.pkl")
    print("✓ Class labels saved to models/class_labels.pkl")
    
    # Model interpretation
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print("\n📈 Model Performance Summary:")
    print(f"   - Overall Accuracy: {accuracy:.1%}")
    print(f"   - Average Precision: {precision.mean():.1%}")
    print(f"   - Average Recall: {recall.mean():.1%}")
    print(f"   - Average F1-Score: {f1.mean():.1%}")
    
    # Performance interpretation
    if accuracy >= 0.9:
        performance = "Excellent! 🎉"
    elif accuracy >= 0.8:
        performance = "Very Good! ⭐"
    elif accuracy >= 0.7:
        performance = "Good 👍"
    elif accuracy >= 0.6:
        performance = "Fair 😐"
    else:
        performance = "Needs Improvement 📉"
    
    print(f"   - Overall Assessment: {performance}")
    
    print("\n🎯 Now you have a CLASSIFIER model that predicts:")
    print("   - Low Yield 🔴")
    print("   - Medium Yield 🟡")
    print("   - High Yield 🟢")
    print("\n📊 Check the confusion matrix to see classification performance!")

if __name__ == '__main__':
    main()