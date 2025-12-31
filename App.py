import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.title("Clustering Comparison: K-Means vs DBSCAN")

# Load data
df = pd.read_csv("Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load models
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
dbscan = pickle.load(open("dbscan_model.pkl", "rb"))

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


