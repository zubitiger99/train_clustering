import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib



# 1. Load the models at the start
try:
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("dbscan_model.pkl", "rb") as f:
        dbscan = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found! Please check if .pkl files are in the repository.")
    st.stop() # Stops the app from running the rest of the code

# ... your data processing code (X_scaled definition) ...

# 2. Now use the models
if 'dbscan' in locals():
    df['DBSCAN Cluster'] = dbscan.fit_predict(X_scaled)

# Try loading with joblib if pickle continues to fail
kmeans = joblib.load("kmeans_model.pkl")
dbScan = joblib.load("dbscan_model.pkl")
st.title("Clustering Comparison: K-Means vs DBSCAN")

# Load data
df = pd.read_csv("Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Predictions
df['KMeans Cluster'] = kmeans.predict(X_scaled)
df['DBSCAN Cluster'] = dbscan.fit_predict(X_scaled)

# Plot K-Means
st.subheader("K-Means Clustering")
fig1, ax1 = plt.subplots()
ax1.scatter(X.iloc[:,0], X.iloc[:,1], c=df['KMeans Cluster'])
ax1.set_xlabel("Annual Income")
ax1.set_ylabel("Spending Score")
st.pyplot(fig1)

# Plot DBSCAN
st.subheader("DBSCAN Clustering")
fig2, ax2 = plt.subplots()
ax2.scatter(X.iloc[:,0], X.iloc[:,1], c=df['DBSCAN Cluster'])
ax2.set_xlabel("Annual Income")
ax2.set_ylabel("Spending Score")
st.pyplot(fig2)

# Comparison
st.subheader("Comparison Summary")
st.write("""
- **K-Means** requires predefined clusters and works well for spherical data.
- **DBSCAN** detects noise and finds arbitrary-shaped clusters.
- DBSCAN labels noise as **-1**.
""")





