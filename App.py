import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.title("Clustering Comparison: K-Means vs DBSCAN")

# 1. Load Data First
try:
    df = pd.read_csv("Mall_Customers.csv")
    # Identify the correct column names from your CSV
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
except FileNotFoundError:
    st.error("Dataset 'Mall_Customers.csv' not found. Please upload it to your GitHub repo.")
    st.stop()

# 2. Scale Data (Models need scaled data to make accurate predictions)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Load Models (Using joblib as it is more robust for Scikit-Learn)
try:
    kmeans = joblib.load("kmeans_model.pkl")
    dbscan = joblib.load("dbscan_model.pkl") # Standardized to lowercase
except FileNotFoundError:
    st.error("Model files (.pkl) not found! Ensure they are in the same folder as app.py.")
    st.stop()

# 4. Predictions
# K-Means uses .predict
df['KMeans Cluster'] = kmeans.predict(X_scaled)

# DBSCAN: Note that saved DBSCAN models usually need to be re-fit on the data
# if they were not saved with the core_sample_indices_
df['DBSCAN Cluster'] = dbscan.fit_predict(X_scaled)

# 5. Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("K-Means Clustering")
    fig1, ax1 = plt.subplots()
    scatter1 = ax1.scatter(X.iloc[:,0], X.iloc[:,1], c=df['KMeans Cluster'], cmap='viridis')
    ax1.set_xlabel("Annual Income")
    ax1.set_ylabel("Spending Score")
    st.pyplot(fig1)

with col2:
    st.subheader("DBSCAN Clustering")
    fig2, ax2 = plt.subplots()
    # DBSCAN often results in -1 for outliers (noise)
    scatter2 = ax2.scatter(X.iloc[:,0], X.iloc[:,1], c=df['DBSCAN Cluster'], cmap='plasma')
    ax2.set_xlabel("Annual Income")
    ax2.set_ylabel("Spending Score")
    st.pyplot(fig2)

# 6. Comparison Summary
st.divider()
st.subheader("Comparison Summary")
st.write(f"**K-Means Clusters Found:** {len(df['KMeans Cluster'].unique())}")
st.write(f"**DBSCAN Clusters Found:** {len(df['DBSCAN Cluster'].unique())} (including noise)")

st.info("""
- **K-Means**: Forces every point into a cluster. Works best for circular, balanced data.
- **DBSCAN**: Identifies high-density areas. Points in low-density areas are labeled as **-1** (Noise).
""")
