import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Clustering App", layout="wide")
st.title("Clustering Comparison: K-Means vs DBSCAN")

# 1. LOAD DATA & SCALE FIRST (To define X_scaled)
try:
    df = pd.read_csv("Mall_Customers.csv")
    # Using specific column names from the Mall Customers dataset
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# 2. LOAD MODELS
try:
    kmeans = joblib.load("kmeans_model.pkl")
    dbscan = joblib.load("dbscan_model.pkl")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# 3. PREDICTIONS
# Now that X_scaled and models are both ready, we can predict
df['KMeans Cluster'] = kmeans.predict(X_scaled)
df['DBSCAN Cluster'] = dbscan.fit_predict(X_scaled)

# 4. VISUALIZATION
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
    # DBSCAN identifies outliers/noise as -1
    ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['DBSCAN Cluster'], cmap='plasma')
    ax2.set_xlabel("Annual Income")
    ax2.set_ylabel("Spending Score")
    st.pyplot(fig2)

st.divider()
st.write("Models successfully loaded and applied to the dataset.")
