import pandas as pd
import pickle
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
optimal_k = 5
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_model.fit_predict(X_scaled)
df['KMeans_Cluster'] = kmeans_labels

# DBSCAN
dbscan_model = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan_model.fit_predict(X_scaled)
df['DBSCAN_Cluster'] = dbscan_labels

# Metrics
metrics = {
    'kmeans': {
        'silhouette': silhouette_score(X_scaled, kmeans_labels),
        'davies_bouldin': davies_bouldin_score(X_scaled, kmeans_labels),
        'calinski_harabasz': calinski_harabasz_score(X_scaled, kmeans_labels),
        'wcss_values': [],
        'silhouette_scores': []
    },
    'dbscan': {
        'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
        'n_noise': list(dbscan_labels).count(-1),
        'silhouette': None,
        'davies_bouldin': None,
        'calinski_harabasz': None
    }
}

best_params = {'eps': 0.5, 'min_samples': 5}

# Save EVERYTHING into one PKL
models_data = {
    'data': df,
    'X': X,
    'X_scaled': X_scaled,
    'kmeans_model': kmeans_model,
    'dbscan_model': dbscan_model,
    'scaler': scaler,
    'metrics': metrics,
    'optimal_k': optimal_k,
    'best_params': best_params
}

with open("clustering_models.pkl", "wb") as f:
    pickle.dump(models_data, f)

print("clustering_models.pkl created successfully")
