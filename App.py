import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Page setup
st.set_page_config(page_title="Clustering Comparison", layout="wide")
st.title("Clustering Comparison: K-Means vs DBSCAN")

# --- 1. DATA LOADING & SCALING ---
# We define X_scaled at the very beginning to prevent NameErrors
try:
    df = pd.read_csv("Mall_Customers.csv")
    # Features for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Scaling the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.sidebar.success("✅ Data and X_scaled ready")
except Exception as e:
    st.error(f"Data Error: {e}. Check if Mall_Customers.csv is in your repository.")
    st.stop()

# --- 2. MODEL LOADING ---
try:
    # Using joblib for scikit-learn compatibility
    kmeans = joblib.load("kmeans_model.pkl")
    dbscan = joblib.load("dbscan_model.pkl")
    st.sidebar.success("✅ Models loaded")
except Exception as e:
    st.error("Model Error: .pkl files not found. Ensure filenames match exactly.")
    st.stop()

# --- 3. PREDICTIONS ---
# Now X_scaled is guaranteed to be defined
df['KMeans Cluster'] = kmeans.predict(X_scaled)
df['DBSCAN Cluster'] = dbscan.fit_predict(X_scaled)

# --- 4. VISUALIZATION ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("K-Means Clustering")
    fig1, ax1 = plt.subplots()
    ax1.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['KMeans Cluster'], cmap='viridis')
    ax1.set_xlabel("Annual Income")
    ax1.set_ylabel("Spending Score")
    st.pyplot(fig1)

with col2:
    st.subheader("DBSCAN Clustering")
    fig2, ax2 = plt.subplots()
    # DBSCAN often identifies noise as -1
    ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['DBSCAN Cluster'], cmap='plasma')
    ax2.set_xlabel("Annual Income")
    ax2.set_ylabel("Spending Score")
    st.pyplot(fig2)

st.divider()
st.info("Results compared: K-Means partitions data, while DBSCAN identifies density.")
