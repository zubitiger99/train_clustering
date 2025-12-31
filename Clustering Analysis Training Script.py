"""
Mall Customer Segmentation - Clustering Analysis
Dataset: Mall Customer Segmentation Data from Kaggle
This script trains K-Means and DBSCAN models and saves them as .pkl files
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset
# Download from: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
print("Loading dataset...")
df = pd.read_csv('Mall_Customers.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Feature selection
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
print(f"\nFeatures selected: Annual Income, Spending Score")
print(f"Feature matrix shape: {X.shape}")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "="*60)
print("K-MEANS CLUSTERING")
print("="*60)

# Elbow Method - Find optimal K
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    print(f"K={k}: WCSS={kmeans.inertia_:.2f}, Silhouette={silhouette_score(X_scaled, labels):.4f}")

# Select optimal K (highest silhouette score)
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"\nOptimal K selected: {optimal_k}")

# Train final K-Means model
kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
kmeans_labels = kmeans_model.fit_predict(X_scaled)

# Calculate metrics
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_db = davies_bouldin_score(X_scaled, kmeans_labels)
kmeans_ch = calinski_harabasz_score(X_scaled, kmeans_labels)

print(f"\nK-Means Results:")
print(f"  Clusters: {optimal_k}")
print(f"  Silhouette Score: {kmeans_silhouette:.4f}")
print(f"  Davies-Bouldin Index: {kmeans_db:.4f}")
print(f"  Calinski-Harabasz Score: {kmeans_ch:.2f}")
print(f"  Cluster distribution: {np.bincount(kmeans_labels)}")

print("\n" + "="*60)
print("DBSCAN CLUSTERING")
print("="*60)

# Find optimal DBSCAN parameters
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, -1], axis=0)

# Suggest eps value (you may need to adjust based on k-distance plot)
suggested_eps = np.percentile(distances, 90)
print(f"Suggested eps value: {suggested_eps:.4f}")

# Try different parameter combinations
best_score = -1
best_params = {}
eps_values = [0.3, 0.4, 0.5, 0.6, 0.7]
min_samples_values = [4, 5, 6, 7]

print("\nTesting DBSCAN parameters...")
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Only consider valid clusterings
        if n_clusters > 1 and n_noise < len(labels) * 0.5:
            try:
                score = silhouette_score(X_scaled[labels != -1], labels[labels != -1])
                print(f"  eps={eps}, min_samples={min_samples}: Clusters={n_clusters}, Noise={n_noise}, Silhouette={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}
            except:
                pass

print(f"\nBest DBSCAN parameters: eps={best_params['eps']}, min_samples={best_params['min_samples']}")

# Train final DBSCAN model
dbscan_model = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
dbscan_labels = dbscan_model.fit_predict(X_scaled)

# Calculate metrics
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
mask = dbscan_labels != -1

if n_clusters_dbscan > 1 and mask.sum() > 0:
    dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
    dbscan_db = davies_bouldin_score(X_scaled[mask], dbscan_labels[mask])
    dbscan_ch = calinski_harabasz_score(X_scaled[mask], dbscan_labels[mask])
else:
    dbscan_silhouette = dbscan_db = dbscan_ch = None

print(f"\nDBSCAN Results:")
print(f"  Clusters: {n_clusters_dbscan}")
print(f"  Noise points: {n_noise} ({n_noise/len(dbscan_labels)*100:.1f}%)")
if dbscan_silhouette:
    print(f"  Silhouette Score: {dbscan_silhouette:.4f}")
    print(f"  Davies-Bouldin Index: {dbscan_db:.4f}")
    print(f"  Calinski-Harabasz Score: {dbscan_ch:.2f}")

# Add cluster labels to dataframe
df['KMeans_Cluster'] = kmeans_labels
df['DBSCAN_Cluster'] = dbscan_labels

print("\n" + "="*60)
print("SAVING MODELS AND DATA")
print("="*60)

# Create models dictionary
models_data = {
    'kmeans_model': kmeans_model,
    'dbscan_model': dbscan_model,
    'scaler': scaler,
    'data': df,
    'X': X,
    'X_scaled': X_scaled,
    'kmeans_labels': kmeans_labels,
    'dbscan_labels': dbscan_labels,
    'optimal_k': optimal_k,
    'best_params': best_params,
    'metrics': {
        'kmeans': {
            'silhouette': kmeans_silhouette,
            'davies_bouldin': kmeans_db,
            'calinski_harabasz': kmeans_ch,
            'wcss_values': wcss,
            'silhouette_scores': silhouette_scores
        },
        'dbscan': {
            'silhouette': dbscan_silhouette,
            'davies_bouldin': dbscan_db,
            'calinski_harabasz': dbscan_ch,
            'n_clusters': n_clusters_dbscan,
            'n_noise': n_noise
        }
    }
}

# Save as pickle file
with open('clustering_models.pkl', 'wb') as f:
    pickle.dump(models_data, f)

print("✓ Models saved to 'clustering_models.pkl'")

# Also save the original dataset for reference
df.to_csv('clustered_data.csv', index=False)
print("✓ Clustered data saved to 'clustered_data.csv'")

print("\n" + "="*60)
print("CLUSTER ANALYSIS")
print("="*60)

print("\nK-Means Cluster Profiles:")
for i in range(optimal_k):
    cluster_data = df[df['KMeans_Cluster'] == i]
    print(f"\nCluster {i} (n={len(cluster_data)}):")
    print(f"  Age: {cluster_data['Age'].mean():.1f} ± {cluster_data['Age'].std():.1f}")
    print(f"  Income: ${cluster_data['Annual Income (k$)'].mean():.1f}k ± ${cluster_data['Annual Income (k$)'].std():.1f}k")
    print(f"  Spending: {cluster_data['Spending Score (1-100)'].mean():.1f} ± {cluster_data['Spending Score (1-100)'].std():.1f}")

print("\n" + "="*60)
print("FILES READY FOR GITHUB AND STREAMLIT!")
print("="*60)
print("\nGenerated files:")
print("1. clustering_models.pkl - Contains trained models and data")
print("2. clustered_data.csv - Dataset with cluster labels")
print("\nNext steps:")
print("1. Upload clustering_models.pkl to GitHub")
print("2. Create Streamlit app (see streamlit_app.py)")
print("3. Deploy on Streamlit Cloud")