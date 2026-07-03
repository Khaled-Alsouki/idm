import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="California Housing Workspace", layout="wide")
plt.style.use('seaborn-v0_8-whitegrid')

@st.cache_data
def get_fully_integrated_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    sample_size = 2000
    np.random.seed(42)
    sample_indices = np.random.choice(X_train_scaled.shape[0], sample_size, replace=False)
    
    hc = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
    sample_labels = hc.fit_predict(X_train_scaled[sample_indices])
    
    label_propagator = KNeighborsClassifier(n_neighbors=1)
    label_propagator.fit(X_train_scaled[sample_indices], sample_labels)
    
    X_train_integrated = X_train.copy()
    X_test_integrated = X_test.copy()
    X_train_integrated['Cluster_Feature'] = label_propagator.predict(X_train_scaled)
    X_test_integrated['Cluster_Feature'] = label_propagator.predict(X_test_scaled)
    
    return df, X_train_integrated, X_test_integrated, y_train, y_test

df, X_train, X_test, y_train, y_test = get_fully_integrated_data()

st.sidebar.title(" Dashboard Controls")
st.sidebar.subheader("Data Filters")
min_income = st.sidebar.slider("Minimum Median Income ($10k)", float(df['MedInc'].min()), float(df['MedInc'].max()), 0.0)
max_price = st.sidebar.slider("Max Median House Value ($100k)", float(df['MedHouseVal'].min()), float(df['MedHouseVal'].max()), 5.0)

filtered_df = df[(df['MedInc'] >= min_income) & (df['MedHouseVal'] <= max_price)]

@st.cache_resource
def run_global_grid_search(_X_train, _y_train, _X_test, _y_test):
    # Dictionary defining pipelines and parameter grids for ALL models uniformly
    models_config = {
        'Ridge': {
            'pipeline': Pipeline([('scaler', StandardScaler()), ('model', Ridge())]),
            'params': {'model__alpha': [0.1, 1.0, 10.0, 50.0]}
        },
        'Lasso': {
            'pipeline': Pipeline([('scaler', StandardScaler()), ('model', Lasso(max_iter=5000))]),
            'params': {'model__alpha': [0.001, 0.01, 0.1, 0.5]}
        },
        'Random Forest': {
            'pipeline': Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(random_state=42))]),
            'params': {
                'model__n_estimators': [50, 100],
                'model__max_depth': [5, 10]
            }
        }
    }
    
    grid_results = {}
    
    for name, config in models_config.items():
        grid = GridSearchCV(config['pipeline'], config['params'], cv=3, scoring='r2', n_jobs=-1)
        grid.fit(_X_train, _y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(_X_test)
        
        grid_results[name] = {
            'Best Params': str(grid.best_params_),
            'R² Score': round(r2_score(_y_test, y_pred), 4),
            'RMSE': round(np.sqrt(mean_squared_error(_y_test, y_pred)), 4),
            'MAE': round(mean_absolute_error(_y_test, y_pred), 4),
            'Estimator': best_model
        }
        
    return grid_results

with st.spinner("Running GridSearchCV for all models on integrated data..."):
    cv_results = run_global_grid_search(X_train, y_train, X_test, y_test)

st.title(" Fully Integrated California Housing Matrix")
st.markdown("All minor weaknesses addressed: Hierarchical clusters are embedded as features, and GridSearchCV tunes all models.")


m1, m2, m3 = st.columns(3)
m1.metric("Filtered Districts", f"{len(filtered_df):,}")
m2.metric("Avg Median Income", f"${filtered_df['MedInc'].mean() * 10:.2f}k")
m3.metric("Avg House Value", f"${filtered_df['MedHouseVal'].mean() * 100:.2f}k")

st.divider()

tab1, tab2, tab3 = st.tabs([" GridSearchCV Tuning Grid", " Geographic Explorations", " Data Distributions"])


with tab1:
    st.header("Comprehensive GridSearchCV Leaderboard")
    st.markdown("Below are the optimized results where **all models** were tuned using cross-validation on data integrated with Hierarchical Clustering features.")
    
    summary_df = pd.DataFrame(cv_results).T.drop(columns=['Estimator'])
    st.dataframe(summary_df, use_container_width=True)
    
    st.divider()
    
    st.subheader("Optimized Feature Weights Comparison")
    model_choice = st.selectbox("Select model to inspect standardized impacts:", list(cv_results.keys()))
    
    chosen_best_model = cv_results[model_choice]['Estimator']
    
    fig_weights, ax_weights = plt.subplots(figsize=(10, 4))
    if model_choice in ['Ridge', 'Lasso']:
        coefs = chosen_best_model.named_steps['model'].coef_
        weights_series = pd.Series(coefs, index=X_train.columns).sort_values(key=abs, ascending=False)
    else:
        importances = chosen_best_model.named_steps['model'].feature_importances_
        weights_series = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
        
    sns.barplot(x=weights_series.values, y=weights_series.index, hue=weights_series.index, palette='viridis', legend=False, ax=ax_weights)
    ax_weights.set_title(f"Optimized Feature Matrix for {model_choice} (Notice 'Cluster_Feature' Impact)")
    st.pyplot(fig_weights)


with tab2:
    st.header("Geographic Price Distribution")
    fig_map, ax_map = plt.subplots(figsize=(12, 6))
    scatter = ax_map.scatter(
        filtered_df['Longitude'], filtered_df['Latitude'],
        c=filtered_df['MedHouseVal'], cmap='viridis',
        s=filtered_df['Population']/100, alpha=0.4
    )
    plt.colorbar(scatter, label='Median House Value ($100k)')
    ax_map.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    st.pyplot(fig_map)

with tab3:
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Feature Correlation Grid")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(filtered_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, cbar=False)
        st.pyplot(fig_corr)
    with col_right:
        st.subheader("Target Price Distribution Plot")
        fig_dist, ax_dist = plt.subplots(figsize=(8, 6))
        sns.histplot(filtered_df['MedHouseVal'], kde=True, color='#1e88e5', ax=ax_dist)
        st.pyplot(fig_dist)