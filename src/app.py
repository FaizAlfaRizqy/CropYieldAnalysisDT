import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Rice Yield Prediction", 
    page_icon="🌾", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/decision_tree_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, preprocessor, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Rice Yield Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("Predict rice yield based on environmental factors and agricultural practices")
    
    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🔮 Prediction", "📊 Model Performance", "📈 Data Analysis", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    if page == "🔮 Prediction":
        show_prediction_page()
    elif page == "📊 Model Performance":
        show_model_performance_page()
    elif page == "📈 Data Analysis":
        show_data_analysis_page()
    else:
        show_about_page()

def show_prediction_page():
    st.markdown('<h2 class="sub-header">🔮 Yield Prediction</h2>', unsafe_allow_html=True)
    
    # Load model
    model, preprocessor, feature_names = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first by running: `python src/train.py`")
        return
    
    # Info box about features used
    st.markdown("""
    <div class="info-box">
    <strong>📝 Note:</strong> This model uses only the most important features based on feature importance analysis:
    <ul>
    <li>🌧️ <strong>Rainfall</strong> - Most important factor (64% importance)</li>
    <li>🌿 <strong>Fertilizer Usage</strong> - Second most important (21% importance)</li>
    <li>💧 <strong>Irrigation Usage</strong> - Third important (14% importance)</li>
    <li>🌡️ <strong>Temperature</strong> - Fourth important (0.9% importance)</li>
    </ul>
    <em>Soil Type, Weather Condition, and Pesticide Usage removed due to low predictive power (≤0.01%)</em>
    </div>
    """, unsafe_allow_html=True)
    
    # Input form
    st.markdown("### 📋 Enter Field Conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌧️ Environmental Factors**")
        rainfall = st.slider(
            "Rainfall (mm)",
            min_value=50.0,
            max_value=1500.0,
            value=600.0,
            step=10.0,
            help="Total rainfall during growing season (Most Important Factor - 64% importance)"
        )
        
        temperature = st.slider(
            "Temperature (°C)",
            min_value=15.0,
            max_value=40.0,
            value=27.0,
            step=0.5,
            help="Average temperature during growing season (0.9% importance)"
        )
        
        # Example values info
        st.info(f"""
        📊 **Data Range Reference:**
        - Rainfall: 100 - 1000 mm
        - Temperature: 15 - 40°C
        """)
    
    with col2:
        st.markdown("**🌾 Agricultural Practices**")
        
        # Fertilizer Usage - Boolean
        fertilizer_used = st.selectbox(
            "🌿 Fertilizer Usage",
            options=[True, False],
            format_func=lambda x: "✅ Used" if x else "❌ Not Used",
            index=0,  # Default to True (Used)
            help="Whether fertilizer is applied or not (21% importance - Very Important!)"
        )
        
        # Irrigation Usage - Boolean  
        irrigation_used = st.selectbox(
            "💧 Irrigation System",
            options=[True, False],
            format_func=lambda x: "✅ Used" if x else "❌ Not Used",
            index=0,  # Default to True (Used)
            help="Whether irrigation system is used or not (14% importance - Important!)"
        )
        
        # Add some spacing
        st.markdown("")
        
        # Show feature importance reminder
        st.success("""
        💡 **Key Success Factors:**
        1. 🌧️ **Adequate Rainfall** (600-900mm ideal)
        2. 🌿 **Fertilizer Application** (significantly boosts yield)
        3. 💧 **Irrigation System** (ensures consistent water supply)
        4. 🌡️ **Optimal Temperature** (25-30°C range)
        """)
    
    # Predict button
    if st.button("🎯 Predict Rice Yield", type="primary", use_container_width=True):
        # Create input dataframe (matching the cleaned dataset structure)
        input_data = pd.DataFrame({
            'Rainfall_mm': [rainfall],
            'Temperature_Celsius': [temperature],
            'Fertilizer_Used': [fertilizer_used],
            'Irrigation_Used': [irrigation_used]
        })
        
        try:
            # Preprocess and predict
            X_processed = preprocessor.transform(input_data)
            prediction = model.predict(X_processed)[0]
            
            # Load cleaned data to get actual thresholds
            data_path = Path('data/processed/rice_yield_cleaned.csv')
            if data_path.exists():
                data = pd.read_csv(data_path)
                q33 = data['Yield_tons_per_hectare'].quantile(0.33)
                q67 = data['Yield_tons_per_hectare'].quantile(0.67)
            else:
                # Fallback thresholds based on the actual data
                q33, q67 = 3.85, 5.45
            
            # Categorize yield using actual thresholds
            if prediction < q33:
                category = "Low Yield 🔴"
                color = "#F44336"
                performance = "Below Average"
            elif prediction < q67:
                category = "Medium Yield 🟡"
                color = "#FF9800"
                performance = "Average"
            else:
                category = "High Yield 🟢"
                color = "#4CAF50"
                performance = "Above Average"
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    label="🌾 Predicted Yield",
                    value=f"{prediction:.2f}",
                    delta="tons/hectare"
                )
            
            with col_b:
                st.markdown(f'<div class="metric-card"><h3>Category</h3><h2 style="color:{color}">{category}</h2><p>{performance}</p></div>', unsafe_allow_html=True)
            
            with col_c:
                # Calculate expected revenue (example: $500/ton)
                revenue = prediction * 500
                st.metric(
                    label="💰 Estimated Revenue",
                    value=f"${revenue:,.0f}",
                    delta="@ $500/ton"
                )
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Yield (tons/hectare)"},
                delta={'reference': (q33 + q67) / 2},
                gauge={
                    'axis': {'range': [None, 10]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, q33], 'color': "#FFCDD2"},
                        {'range': [q33, q67], 'color': "#FFE082"},
                        {'range': [q67, 10], 'color': "#C8E6C9"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': (q33 + q67) / 2
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Input summary
            with st.expander("📝 Input Summary & Feature Importance"):
                col_sum1, col_sum2 = st.columns(2)
                
                with col_sum1:
                    st.markdown("**Your Input:**")
                    summary_df = pd.DataFrame({
                        'Parameter': ['Rainfall', 'Temperature', 'Fertilizer', 'Irrigation'],
                        'Value': [
                            f"{rainfall} mm",
                            f"{temperature} °C",
                            "✅ Used" if fertilizer_used else "❌ Not Used",
                            "✅ Used" if irrigation_used else "❌ Not Used"
                        ]
                    })
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                with col_sum2:
                    st.markdown("**Feature Importance:**")
                    importance_df = pd.DataFrame({
                        'Feature': ['Rainfall', 'Fertilizer', 'Irrigation', 'Temperature'],
                        'Importance': ['64.0%', '21.2%', '13.8%', '0.9%'],
                        'Impact': ['🔥 Critical', '⭐ High', '📈 Medium', '🌡️ Low']
                    })
                    st.dataframe(importance_df, use_container_width=True, hide_index=True)
            
            # Enhanced Recommendations based on prediction
            st.markdown("### 💡 Recommendations")
            
            # Create recommendation columns
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                if prediction < q33:
                    st.warning(f"""
                    **Low Yield Predicted ({prediction:.2f} tons/ha)**
                    
                    🔧 **Priority Actions:**
                    - 🌧️ **Rainfall Management**: Current {rainfall}mm may be insufficient
                    - 🌿 **Fertilizer**: {"Continue using" if fertilizer_used else "⚠️ APPLY fertilizer - critical for yield"}
                    - 💧 **Irrigation**: {"Optimize timing" if irrigation_used else "⚠️ INSTALL irrigation system"}
                    """)
                elif prediction < q67:
                    st.info(f"""
                    **Medium Yield Predicted ({prediction:.2f} tons/ha)**
                    
                    ✨ **Enhancement Opportunities:**
                    - 🌧️ **Rainfall**: {rainfall}mm is adequate, monitor closely
                    - 🌿 **Fertilizer**: {"Good practice" if fertilizer_used else "Consider adding fertilizer"}
                    - 💧 **Irrigation**: {"Maintain system" if irrigation_used else "Consider irrigation for consistency"}
                    """)
                else:
                    st.success(f"""
                    **High Yield Predicted ({prediction:.2f} tons/ha)**
                    
                    🎉 **Excellent Conditions!**
                    - 🌧️ **Rainfall**: {rainfall}mm is optimal
                    - 🌿 **Fertilizer**: {"Perfect application" if fertilizer_used else "Surprisingly good without fertilizer"}
                    - 💧 **Irrigation**: {"Well managed" if irrigation_used else "Good natural conditions"}
                    """)
            
            with rec_col2:
                # Comparison with optimal conditions
                st.markdown("**🎯 Optimal Conditions Guide:**")
                
                optimal_conditions = pd.DataFrame({
                    'Factor': ['Rainfall', 'Temperature', 'Fertilizer', 'Irrigation'],
                    'Your Input': [
                        f"{rainfall} mm",
                        f"{temperature} °C", 
                        "✅" if fertilizer_used else "❌",
                        "✅" if irrigation_used else "❌"
                    ],
                    'Optimal Range': [
                        "600-900 mm",
                        "25-30 °C",
                        "✅ Used",
                        "✅ Used"
                    ],
                    'Status': [
                        "✅ Good" if 600 <= rainfall <= 900 else "⚠️ Adjust",
                        "✅ Good" if 25 <= temperature <= 30 else "⚠️ Monitor",
                        "✅ Good" if fertilizer_used else "❌ Missing",
                        "✅ Good" if irrigation_used else "❌ Missing"
                    ]
                })
                
                st.dataframe(optimal_conditions, use_container_width=True, hide_index=True)
                
                # Quick action summary
                missing_factors = []
                if not fertilizer_used:
                    missing_factors.append("Fertilizer")
                if not irrigation_used:
                    missing_factors.append("Irrigation")
                if not (600 <= rainfall <= 900):
                    missing_factors.append("Optimal Rainfall")
                if not (25 <= temperature <= 30):
                    missing_factors.append("Optimal Temperature")
                
                if missing_factors:
                    st.warning(f"⚠️ **Areas for Improvement**: {', '.join(missing_factors)}")
                else:
                    st.success("🎯 **All conditions are optimal!**")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.info("Make sure the model is trained with the correct features. Run `python src/train.py` after data cleaning.")

def show_model_performance_page():
    st.markdown('<h2 class="sub-header">📊 Model Performance</h2>', unsafe_allow_html=True)
    
    # Model metrics
    metrics_path = Path('reports/model_metrics.csv')
    if metrics_path.exists():
        st.markdown("### 📈 Model Metrics")
        metrics_df = pd.read_csv(metrics_path)
        
        col1, col2, col3, col4 = st.columns(4)
        
        for i, row in metrics_df.iterrows():
            col = [col1, col2, col3, col4][i]
            with col:
                st.metric(
                    label=f"{row['Metric']} ({row['Unit']})",
                    value=f"{row['Value']:.4f}"
                )
    
    # Check if visualizations exist
    figures_path = Path('reports/figures')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Feature Importance")
        feature_imp_path = figures_path / 'feature_importance.png'
        if feature_imp_path.exists():
            st.image(str(feature_imp_path), use_container_width=True)
        else:
            st.warning("Feature importance plot not found. Run `python src/train.py` first.")
    
    with col2:
        st.markdown("### 🎯 Predictions vs Actual")
        pred_path = figures_path / 'predictions.png'
        if pred_path.exists():
            st.image(str(pred_path), use_container_width=True)
        else:
            st.warning("Predictions plot not found.")
    
    st.markdown("### 📉 Residuals Analysis")
    resid_path = figures_path / 'residuals.png'
    if resid_path.exists():
        st.image(str(resid_path), use_container_width=True)
    else:
        st.warning("Residuals plot not found.")
    
    # Feature importance table
    st.markdown("### 📋 Feature Importance Rankings")
    feature_csv = Path('reports/feature_importance.csv')
    if feature_csv.exists():
        df = pd.read_csv(feature_csv)
        st.dataframe(
            df.head(15).style.background_gradient(subset=['Importance_Percent'], cmap='Greens'),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Feature importance data not available. Train the model first.")

def show_data_analysis_page():
    st.markdown('<h2 class="sub-header">📈 Data Analysis</h2>', unsafe_allow_html=True)
    
    # Load cleaned data
    data_path = Path('data/processed/rice_yield_cleaned.csv')
    
    if not data_path.exists():
        st.error("Cleaned data not found. Run `python src/data_cleaning.py` first.")
        return
    
    data = pd.read_csv(data_path)
    
    # Dataset overview
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{len(data):,}")
    with col2:
        st.metric("Features", len(data.columns) - 2)  # Exclude target columns
    with col3:
        st.metric("Avg Yield", f"{data['Yield_tons_per_hectare'].mean():.2f}")
    with col4:
        st.metric("Std Dev", f"{data['Yield_tons_per_hectare'].std():.2f}")
    
    # Yield distribution
    st.markdown("### 📊 Yield Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            data,
            x='Yield_tons_per_hectare',
            nbins=50,
            title='Yield Distribution',
            labels={'Yield_tons_per_hectare': 'Yield (tons/hectare)'},
            color_discrete_sequence=['#4CAF50']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            data,
            names='Yield_Category',
            title='Yield Categories',
            color_discrete_map={
                'Low Yield': '#F44336',
                'Medium Yield': '#FF9800',
                'High Yield': '#4CAF50'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Factor analysis (only for available columns)
    st.markdown("### 🔍 Yield by Factors")
    
    available_factors = [col for col in ['Fertilizer_Used', 'Irrigation_Used'] if col in data.columns]
    
    if available_factors:
        factor = st.selectbox("Select Factor", available_factors)
        
        fig = px.box(
            data,
            x=factor,
            y='Yield_tons_per_hectare',
            color=factor,
            title=f'Yield Distribution by {factor}',
            labels={'Yield_tons_per_hectare': 'Yield (tons/hectare)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical factors available for analysis.")
    
    # Correlation heatmap
    st.markdown("### 🔥 Correlation Analysis")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numerical columns for correlation analysis.")

def show_about_page():
    st.markdown('<h2 class="sub-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🌾 Rice Yield Prediction System
    
    This application uses **Machine Learning** to predict rice crop yields based on 
    the most important environmental factors and agricultural practices.
    
    #### 🎯 Key Features
    - **Real-time Predictions**: Get instant yield predictions
    - **Decision Tree Model**: Interpretable AI model (CART algorithm)
    - **Feature Importance**: Focus on factors that matter most
    - **Optimized Features**: Uses only high-impact variables (>0.01% importance)
    
    #### 📊 Model Details
    - **Algorithm**: Decision Tree Regressor (CART)
    - **Input Features**: 5 optimized variables
      - 🌧️ **Rainfall** (64% importance - Most Critical)
      - 🌿 **Fertilizer Usage** (21% importance - Very Important)  
      - 💧 **Irrigation System** (14% importance - Important)
      - 🌡️ **Temperature** (0.9% importance - Minor)
      - 🐛 **Pesticide Usage** (Variable importance)
    - **Removed Features**: Soil Type, Weather Condition (≤0.01% importance)
    - **Output**: Yield prediction (tons/hectare) + Category
    
    #### 🎨 Yield Categories (Dynamic Thresholds)
    - 🔴 **Low Yield**: < 33rd percentile
    - 🟡 **Medium Yield**: 33rd - 67th percentile  
    - 🟢 **High Yield**: > 67th percentile
    
    #### 📈 Dataset (After Optimization)
    - **Crop**: Rice only (filtered from multi-crop dataset)
    - **Total Samples**: ~166K+ records
    - **Features**: Reduced from 9+ to 5 most important
    - **Data Quality**: Cleaned, no missing values, no data leakage
    
    #### 🔬 Model Optimization
    - **Feature Selection**: Removed features with ≤0.01% importance
    - **Data Leakage Prevention**: Removed Days_to_Harvest
    - **Improved Efficiency**: 50%+ reduction in feature dimensions
    - **Better Performance**: Focus on predictive factors only
    
    #### 🛠️ Technology Stack
    - **Frontend**: Streamlit
    - **ML Framework**: scikit-learn (CART implementation)
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Data Processing**: Pandas, NumPy
    
    #### 📚 How to Use
    1. Navigate to **🔮 Prediction** page
    2. Enter your field conditions (5 key factors)
    3. Click **🎯 Predict Yield**
    4. View results with yield estimate and category
    5. Follow recommendations for improvement
    6. Check **📊 Model Performance** for insights
    
    #### 🔄 Model Training Pipeline
    ```bash
    # 1. Clean and optimize data
    python src/data_cleaning.py
    
    # 2. Train optimized model  
    python src/train.py
    
    # 3. Run web application
    streamlit run src/app.py
    ```
    
    #### 🎓 Why This Approach?
    - **Focused Prediction**: Uses only factors that truly matter
    - **Practical**: Farmers can control these 5 key variables
    - **Efficient**: Faster predictions, less complexity
    - **Interpretable**: Clear understanding of what drives yield
    - **Actionable**: Recommendations based on most important factors
    
    ---
    
    **Version**: 2.0.0 (Optimized)  
    **Last Updated**: November 2025  
    **Model Type**: Decision Tree Regression (Optimized CART)  
    **Key Improvement**: Feature importance-based optimization
    """)
    
    # System status
    st.markdown("### ⚙️ System Status")
    
    model_exists = Path('models/decision_tree_model.pkl').exists()
    data_exists = Path('data/processed/rice_yield_cleaned.csv').exists()
    
    col1, col2 = st.columns(2)
    with col1:
        if model_exists:
            st.success("✅ Model: Ready")
        else:
            st.error("❌ Model: Not Found")
    
    with col2:
        if data_exists:
            st.success("✅ Data: Available")
        else:
            st.error("❌ Data: Not Found")
    
    # Feature comparison
    st.markdown("### 🔄 Before vs After Optimization")
    
    comparison_df = pd.DataFrame({
        'Aspect': ['Features Used', 'Model Input Dimension', 'Focus', 'Efficiency'],
        'Before': ['9+ features', '~20 dimensions', 'All variables', 'Standard'],
        'After': ['5 key features', '~10 dimensions', 'High-impact only', '50%+ faster']
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

if __name__ == '__main__':
    main()