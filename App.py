import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Clustering Comparison", layout="wide")
st.title("Clustering Comparison: K-Means vs DBSCAN")

# --- 1. DATA LOADING & PREPROCESSING ---
# This must happen first so 'X_scaled' is defined for the models
try:
    df = pd.read_csv("Mall_Customers.csv")
    # Features used for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Define and create X_scaled
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
except FileNotFoundError:
    st.error("Dataset 'Mall_Customers.csv' not found. Please ensure it is in your GitHub repo.")
    st.stop()

# --- 2. MODEL LOADING ---
try:
    # Use joblib as it is more reliable for sklearn models than pickle
    kmeans = joblib.load("kmeans_model.pkl")
    dbscan = joblib.load("dbscan_model.pkl")
except FileNotFoundError:
    st.error("Model files (.pkl) not found! Check your file names in the repository.")
    st.stop()

# --- 3. PREDICTIONS ---
# Now that X_scaled is defined and models are loaded, we can predict
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
    # DBSCAN identifies noise as -1
    ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['DBSCAN Cluster'], cmap='plasma')
    ax2.set_xlabel("Annual Income")
    ax2.set_ylabel("Spending Score")
    st.pyplot(fig2)

# --- 5. SUMMARY ---
st.divider()
st.subheader("Comparison Summary")
st.write(f"**Total Points:** {len(df)}")
st.write(f"**K-Means Clusters:** {len(df['KMeans Cluster'].unique())}")
st.write(f"**DBSCAN Clusters:** {len(df['DBSCAN Cluster'].unique())} (including noise if any)")
