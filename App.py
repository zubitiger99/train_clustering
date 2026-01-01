import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Clustering Comparison", layout="wide")
st.title("Clustering Comparison: K-Means vs DBSCAN")

# --- STEP 1: LOAD DATA ---
try:
    # Load the Kaggle dataset
    df = pd.read_csv("Mall_Customers.csv")
    # Define features (ensure these match your CSV column names exactly)
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # --- STEP 2: PREPROCESS / SCALE (Must happen before model prediction) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
except FileNotFoundError:
    st.error("Dataset 'Mall_Customers.csv' not found. Please ensure it is in your GitHub repository.")
    st.stop()
except KeyError:
    st.error("Column names don't match. Please check 'Annual Income (k$)' and 'Spending Score (1-100)'.")
    st.stop()

# --- STEP 3: LOAD MODELS ---
try:
    # Using joblib for better compatibility with scikit-learn objects
    kmeans = joblib.load("kmeans_model.pkl")
    dbscan = joblib.load("dbscan_model.pkl")
except FileNotFoundError:
    st.error("Model files (.pkl) not found! Ensure 'kmeans_model.pkl' and 'dbscan_model.pkl' are in the repo.")
    st.stop()

# --- STEP 4: PREDICTIONS ---
# Now that X_scaled and models are loaded, we can run predictions
df['KMeans Cluster'] = kmeans.predict(X_scaled)
df['DBSCAN Cluster'] = dbscan.fit_predict(X_scaled)

# --- STEP 5: VISUALIZATION ---
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
    # DBSCAN labels noise as -1
    ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['DBSCAN Cluster'], cmap='plasma')
    ax2.set_xlabel("Annual Income")
    ax2.set_ylabel("Spending Score")
    st.pyplot(fig2)

# --- STEP 6: COMPARISON SUMMARY ---
st.divider()
st.subheader("Comparison Summary")
st.write(f"**K-Means Clusters:** {len(df['KMeans Cluster'].unique())}")
st.write(f"**DBSCAN Clusters:** {len(df['DBSCAN Cluster'].unique())} (including noise if present)")

st.info("""
- **K-Means**: Assigned every point to a cluster based on centroids.
- **DBSCAN**: Grouped points based on density. Any points colored as 'noise' (usually -1) didn't fit into a dense cluster.
""")
