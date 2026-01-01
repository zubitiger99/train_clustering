import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="Clustering App", layout="wide")
st.title("Clustering Comparison: K-Means vs DBSCAN")

# --- STEP 1: LOAD DATA AND SCALE (Must be first) ---
try:
    df = pd.read_csv("Mall_Customers.csv")
    # Extract features
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Define X_scaled here so it is available for all following lines
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.sidebar.success("Data loaded and scaled successfully.")
except FileNotFoundError:
    st.error("Missing 'Mall_Customers.csv'. Please upload it to your GitHub repo.")
    st.stop()

# --- STEP 2: LOAD MODELS ---
try:
    # We use joblib because it's more stable for scikit-learn models
    kmeans = joblib.load("kmeans_model.pkl")
    dbscan = joblib.load("dbscan_model.pkl")
except FileNotFoundError:
    st.error("Model files (.pkl) not found. Check names in your repository.")
    st.stop()

# --- STEP 3: PREDICTIONS (Now X_scaled definitely exists) ---
df['KMeans Cluster'] = kmeans.predict(X_scaled)
df['DBSCAN Cluster'] = dbscan.fit_predict(X_scaled)

# --- STEP 4: VISUALIZATION ---
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
    ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['DBSCAN Cluster'], cmap='plasma')
    ax2.set_xlabel("Annual Income")
    ax2.set_ylabel("Spending Score")
    st.pyplot(fig2)

st.divider()
st.info("The app is now running with X_scaled properly defined.")
