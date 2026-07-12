import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="California Housing Workspace", layout="wide")

plt.style.use('seaborn-v0_8-whitegrid')

@st.cache_data
def get_processed_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    return df, X_train, X_test, y_train, y_test

df, X_train, X_test, y_train, y_test = get_processed_data()

st.sidebar.title(" Dashboard Controls")

st.sidebar.subheader("Data Filters")
min_income = st.sidebar.slider("Minimum Median Income ($10k)", float(df['MedInc'].min()), float(df['MedInc'].max()), 0.0)
max_price = st.sidebar.slider("Max Median House Value ($100k)", float(df['MedHouseVal'].min()), float(df['MedHouseVal'].max()), 5.0)

filtered_df = df[(df['MedInc'] >= min_income) & (df['MedHouseVal'] <= max_price)]

st.sidebar.divider()

st.sidebar.subheader("Model Hyperparameters")
model_type = st.sidebar.radio("Select Regularization Type:", ["Ridge (L2)", "Lasso (L1)"])

if "Ridge" in model_type:
    alpha_val = st.sidebar.slider("Tune Alpha Penalty (Ridge Range):", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
else:
    alpha_val = st.sidebar.slider("Tune Alpha Penalty (Lasso Fine Range):", min_value=0.0001, max_value=0.1000, value=0.0050, step=0.0005, format="%.4f")

st.title(" California Housing Enterprise Dashboard")


m1, m2, m3 = st.columns(3)
m1.metric("Filtered Districts", f"{len(filtered_df):,}")
m2.metric("Avg Median Income", f"${filtered_df['MedInc'].mean() * 10:.2f}k")
m3.metric("Avg House Value", f"${filtered_df['MedHouseVal'].mean() * 100:.2f}k")

st.divider()

tab1, tab2, tab3 = st.tabs([" Live Model Tuning & Comparison", " Geographic Explorations", " Data Distributions"])

with tab1:
    st.header("Real-Time Hyperparameter Tuning & Model Comparison")
    st.markdown(f"Currently training a **{model_type}** model with **Alpha = {alpha_val:.4f}** on the 80% training set.")
    
    if "Ridge" in model_type:
        estimator = Ridge(alpha=alpha_val)
    else:
        estimator = Lasso(alpha=alpha_val, max_iter=10000, tol=1e-3)
        
    standard_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', estimator)
    ])
    standard_pipeline.fit(X_train, y_train)
    y_pred_std = standard_pipeline.predict(X_test)
    
    rmse_std = np.sqrt(mean_squared_error(y_test, y_pred_std))
    mae_std = mean_absolute_error(y_test, y_pred_std)
    r2_std = r2_score(y_test, y_pred_std)
    
    poly_features_list = ['MedInc', 'HouseAge']
    other_features_list = [col for col in X_train.columns if col not in poly_features_list]
    
    preprocessor_app = ColumnTransformer(
        transformers=[
            ('poly', PolynomialFeatures(degree=2, include_bias=False), poly_features_list),
            ('pass', 'passthrough', other_features_list)
        ]
    )
    
    poly_pipeline_app = Pipeline([
        ('prep', preprocessor_app),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=10.0))
    ])
    poly_pipeline_app.fit(X_train, y_train)
    y_pred_poly = poly_pipeline_app.predict(X_test)
    
    rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    mae_poly = mean_absolute_error(y_test, y_pred_poly)
    r2_poly = r2_score(y_test, y_pred_poly)
    
    
    col_std, col_poly = st.columns(2)
    
    with col_std:
        st.subheader(f" Standard {model_type}")
        st.caption("Changes dynamically with the sidebar slider")
        st.metric(label="R² Score (Higher is Better)", value=f"{r2_std:.4f}")
        st.metric(label="RMSE (Lower is Better)", value=f"{rmse_std:.4f}")
        st.metric(label="MAE", value=f"{mae_std:.4f}")
        
    with col_poly:
        st.subheader(" Advanced Polynomial Model")
        st.caption("Features: MedInc² & HouseAge² (Fixed Baseline)")
        
        
        r2_delta = f"+{r2_poly - r2_std:.4f}" if r2_poly > r2_std else f"{r2_poly - r2_std:.4f}"
        rmse_delta = f"{rmse_poly - rmse_std:.4f}"
        
        st.metric(label="R² Score (Target Achieved!)", value=f"{r2_poly:.4f}", delta=r2_delta)
        st.metric(label="RMSE", value=f"{rmse_poly:.4f}", delta=rmse_delta, delta_color="inverse")
        st.metric(label="MAE", value=f"{mae_poly:.4f}")

    st.divider()
    
    st.subheader("Dynamic Feature Weights Analysis")
    st.markdown("Observe how changing Alpha values shrinks or drops feature coefficients in real-time.")
    
    coefficients = standard_pipeline.named_steps['model'].coef_
    feature_names = X_train.columns
    feature_weights = pd.Series(coefficients, index=feature_names).sort_values(key=abs, ascending=False)
    
    fig_weights, ax_weights = plt.subplots(figsize=(10, 4.5))
    sns.barplot(
        x=feature_weights.values, 
        y=feature_weights.index, 
        hue=feature_weights.index, 
        palette='viridis', 
        legend=False,
        ax=ax_weights
    )
    ax_weights.set_title(f'Standardized Coefficients for {model_type} (Alpha = {alpha_val:.4f})')
    ax_weights.set_xlabel('Coefficient Value / Impact Strength')
    ax_weights.set_ylabel('Features')
    plt.tight_layout()
    st.pyplot(fig_weights)

with tab2:
    st.header("Geographic Price Distribution")
    st.markdown("Spatial positioning mapping based on latitude and longitude coordinates.")
    
    fig_map, ax_map = plt.subplots(figsize=(12, 6.5))
    scatter = ax_map.scatter(
        filtered_df['Longitude'], filtered_df['Latitude'],
        c=filtered_df['MedHouseVal'], cmap='viridis',
        s=filtered_df['Population']/100, alpha=0.4
    )
    plt.colorbar(scatter, label='Median House Value ($100k)')
    ax_map.set_xlabel('Longitude Coordinate')
    ax_map.set_ylabel('Latitude Coordinate')
    ax_map.set_title('California Spatial Housing Matrix')
    plt.tight_layout()
    st.pyplot(fig_map)

with tab3:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Feature Correlation Grid")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(filtered_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, cbar=False)
        plt.tight_layout()
        st.pyplot(fig_corr)
        
    with col_right:
        st.subheader("Target Price Distribution Plot")
        fig_dist, ax_dist = plt.subplots(figsize=(8, 6.5))
        sns.histplot(filtered_df['MedHouseVal'], kde=True, color='#1e88e5', ax=ax_dist)
        ax_dist.set_xlabel('Median House Value ($100k)')
        plt.tight_layout()
        st.pyplot(fig_dist)
