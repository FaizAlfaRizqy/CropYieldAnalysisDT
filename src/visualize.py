import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import plot_tree
from pathlib import Path

def plot_feature_importance(model, feature_names, top_n=15, save_path='reports/figures/feature_importance.png'):
    """
    Plot feature importance dari Decision Tree model
    
    Args:
        model: Trained model dengan attribute feature_importances_
        feature_names: List nama features
        top_n: Jumlah top features yang ditampilkan
        save_path: Path untuk save figure
    """
    # Get feature importances
    importances = model.model.feature_importances_
    
    # Create dataframe
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Importance_Percent': importances * 100
    }).sort_values('Importance', ascending=False)
    
    # Print top features
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    print(f"\nTop {top_n} Most Important Features:")
    print(feature_imp_df.head(top_n).to_string(index=False))
    
    # Calculate cumulative importance
    feature_imp_df['Cumulative_Importance'] = feature_imp_df['Importance_Percent'].cumsum()
    
    # Find features that contribute to 80% importance
    features_80_pct = feature_imp_df[feature_imp_df['Cumulative_Importance'] <= 80]
    print(f"\n{len(features_80_pct)} features contribute to 80% of total importance")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart of top features
    top_features = feature_imp_df.head(top_n)
    axes[0].barh(range(len(top_features)), top_features['Importance_Percent'])
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['Feature'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Importance (%)', fontsize=12)
    axes[0].set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        axes[0].text(row['Importance_Percent'], i, f" {row['Importance_Percent']:.2f}%", 
                     va='center', fontsize=9)
    
    # Plot 2: Cumulative importance
    axes[1].plot(range(len(feature_imp_df)), feature_imp_df['Cumulative_Importance'], 
                 linewidth=2, color='darkblue')
    axes[1].axhline(y=80, color='red', linestyle='--', label='80% threshold')
    axes[1].axhline(y=95, color='orange', linestyle='--', label='95% threshold')
    axes[1].fill_between(range(len(feature_imp_df)), 0, feature_imp_df['Cumulative_Importance'], 
                         alpha=0.3)
    axes[1].set_xlabel('Number of Features', fontsize=12)
    axes[1].set_ylabel('Cumulative Importance (%)', fontsize=12)
    axes[1].set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFeature importance plot saved to: {save_path}")
    
    return feature_imp_df

def plot_feature_importance_by_category(model, feature_names, save_path='reports/figures/feature_importance_categories.png'):
    """
    Plot feature importance grouped by category (numerical vs categorical)
    """
    importances = model.model.feature_importances_
    
    # Identify feature types (assumes encoded categorical features have specific patterns)
    numerical_features = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']
    
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Categorize features
    feature_imp_df['Category'] = feature_imp_df['Feature'].apply(
        lambda x: 'Numerical' if any(num in str(x) for num in numerical_features) else 'Categorical'
    )
    
    # Group by category
    category_importance = feature_imp_df.groupby('Category')['Importance'].sum() * 100
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Pie chart
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0.05)
    axes[0].pie(category_importance, labels=category_importance.index, autopct='%1.1f%%',
                startangle=90, colors=colors, explode=explode, shadow=True)
    axes[0].set_title('Feature Importance by Category', fontsize=14, fontweight='bold')
    
    # Plot 2: Bar chart by category
    for category in feature_imp_df['Category'].unique():
        cat_data = feature_imp_df[feature_imp_df['Category'] == category].sort_values('Importance', ascending=False).head(5)
        axes[1].barh(cat_data['Feature'], cat_data['Importance'] * 100, label=category, alpha=0.7)
    
    axes[1].set_xlabel('Importance (%)', fontsize=12)
    axes[1].set_title('Top 5 Features per Category', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance by category plot saved to: {save_path}")

def plot_decision_tree_simple(model, feature_names, save_path='reports/figures/decision_tree.png'):
    """Visualize the decision tree structure"""
    plt.figure(figsize=(25, 15))
    plot_tree(model.model, 
              feature_names=feature_names,
              filled=True, 
              rounded=True, 
              fontsize=10,
              max_depth=3)  # Limit depth for readability
    plt.title("Decision Tree Structure (max_depth=3 shown)", fontsize=16)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Decision tree visualization saved to {save_path}")

def plot_predictions(y_true, y_pred, save_path='reports/figures/predictions.png'):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Yield (tons/hectare)")
    plt.ylabel("Predicted Yield (tons/hectare)")
    plt.title("Actual vs Predicted Yield")
    
    # Add metrics to plot
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    plt.text(0.05, 0.95, f'MSE: {mse:.4f}\nMAE: {mae:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Predictions plot saved to {save_path}")

def plot_residuals(y_true, y_pred, save_path='reports/figures/residuals.png'):
    """Plot residuals"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel("Predicted Yield")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted Values")
    
    # Residuals distribution
    axes[1].hist(residuals, bins=50, edgecolor='black')
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Residuals")
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residuals plot saved to {save_path}")