"""
Mall Customer Segmentation - Streamlit Dashboard
Interactive visualization and comparison of K-Means and DBSCAN clustering
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    try:
        with open('clustering_models.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'clustering_models.pkl' is in the same directory.")
        st.stop()

# Load data
models_data = load_models()
df = models_data['data']
X = models_data['X']
X_scaled = models_data['X_scaled']
kmeans_model = models_data['kmeans_model']
dbscan_model = models_data['dbscan_model']
scaler = models_data['scaler']
metrics = models_data['metrics']
optimal_k = models_data['optimal_k']
best_params = models_data['best_params']

# Title and description
st.title("🛍️ Mall Customer Segmentation Dashboard")
st.markdown("### K-Means vs DBSCAN Clustering Analysis")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/shopping-mall.png", width=100)
    st.header("Navigation")
    page = st.radio(
        "Select Page:",
        ["📊 Overview", "🎯 K-Means Analysis", "🔍 DBSCAN Analysis", 
         "⚖️ Comparison", "🎮 Interactive Clustering"]
    )
    
    st.markdown("---")
    st.markdown("### Dataset Info")
    st.metric("Total Customers", len(df))
    st.metric("Features", "2 (Income, Spending)")
    st.metric("K-Means Clusters", optimal_k)
    st.metric("DBSCAN Clusters", metrics['dbscan']['n_clusters'])

# Page: Overview
if page == "📊 Overview":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Customers", len(df))
    with col2:
        st.metric("Avg Age", f"{df['Age'].mean():.1f} years")
    with col3:
        st.metric("Avg Income", f"${df['Annual Income (k$)'].mean():.1f}k")
    with col4:
        st.metric("Avg Spending", f"{df['Spending Score (1-100)'].mean():.1f}")
    
    st.markdown("### Dataset Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig_age = px.histogram(df, x='Age', nbins=20, 
                               title='Age Distribution',
                               color_discrete_sequence=['#636EFA'])
        fig_age.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Gender distribution
        gender_counts = df['Gender'].value_counts()
        fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index,
                           title='Gender Distribution',
                           color_discrete_sequence=['#636EFA', '#EF553B'])
        fig_gender.update_layout(height=300)
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        # Income distribution
        fig_income = px.histogram(df, x='Annual Income (k$)', nbins=20,
                                 title='Annual Income Distribution',
                                 color_discrete_sequence=['#00CC96'])
        fig_income.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_income, use_container_width=True)
        
        # Spending distribution
        fig_spending = px.histogram(df, x='Spending Score (1-100)', nbins=20,
                                   title='Spending Score Distribution',
                                   color_discrete_sequence=['#AB63FA'])
        fig_spending.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_spending, use_container_width=True)
    
    # Scatter plot
    st.markdown("### Income vs Spending Score")
    fig_scatter = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)',
                            color='Age', size='Age',
                            title='Customer Income vs Spending Score (sized and colored by Age)',
                            color_continuous_scale='Viridis')
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

# Page: K-Means Analysis
elif page == "🎯 K-Means Analysis":
    st.header("K-Means Clustering Analysis")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Optimal K", optimal_k)
    with col2:
        st.metric("Silhouette Score", f"{metrics['kmeans']['silhouette']:.4f}")
    with col3:
        st.metric("Davies-Bouldin Index", f"{metrics['kmeans']['davies_bouldin']:.4f}")
    with col4:
        st.metric("Calinski-Harabasz", f"{metrics['kmeans']['calinski_harabasz']:.2f}")
    
    # Elbow curve and Silhouette scores
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Elbow Method")
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(range(2, 11)),
            y=metrics['kmeans']['wcss_values'],
            mode='lines+markers',
            name='WCSS',
            line=dict(color='#636EFA', width=3),
            marker=dict(size=10)
        ))
        fig_elbow.add_vline(x=optimal_k, line_dash="dash", line_color="red",
                           annotation_text=f"Optimal K={optimal_k}")
        fig_elbow.update_layout(
            xaxis_title='Number of Clusters (K)',
            yaxis_title='WCSS',
            height=400
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    with col2:
        st.markdown("### Silhouette Analysis")
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=list(range(2, 11)),
            y=metrics['kmeans']['silhouette_scores'],
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='#00CC96', width=3),
            marker=dict(size=10)
        ))
        fig_sil.add_vline(x=optimal_k, line_dash="dash", line_color="red",
                         annotation_text=f"Optimal K={optimal_k}")
        fig_sil.update_layout(
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Silhouette Score',
            height=400
        )
        st.plotly_chart(fig_sil, use_container_width=True)
    
    # Clustering visualization
    st.markdown("### K-Means Clustering Results")
    
    df_kmeans = df.copy()
    df_kmeans['Cluster'] = df_kmeans['KMeans_Cluster'].astype(str)
    
    fig_kmeans = px.scatter(df_kmeans, 
                           x='Annual Income (k$)', 
                           y='Spending Score (1-100)',
                           color='Cluster',
                           title=f'K-Means Clustering (K={optimal_k})',
                           hover_data=['Age', 'Gender'])
    
    # Add centroids
    centroids = scaler.inverse_transform(kmeans_model.cluster_centers_)
    fig_kmeans.add_trace(go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode='markers',
        marker=dict(symbol='x', size=20, color='red', line=dict(width=2, color='black')),
        name='Centroids'
    ))
    
    fig_kmeans.update_layout(height=600)
    st.plotly_chart(fig_kmeans, use_container_width=True)
    
    # Cluster profiles
    st.markdown("### Cluster Profiles")
    
    cluster_profiles = []
    for i in range(optimal_k):
        cluster_data = df[df['KMeans_Cluster'] == i]
        profile = {
            'Cluster': i,
            'Size': len(cluster_data),
            'Avg Age': f"{cluster_data['Age'].mean():.1f}",
            'Avg Income': f"${cluster_data['Annual Income (k$)'].mean():.1f}k",
            'Avg Spending': f"{cluster_data['Spending Score (1-100)'].mean():.1f}",
            'Male': len(cluster_data[cluster_data['Gender'] == 'Male']),
            'Female': len(cluster_data[cluster_data['Gender'] == 'Female'])
        }
        cluster_profiles.append(profile)
    
    cluster_df = pd.DataFrame(cluster_profiles)
    st.dataframe(cluster_df, use_container_width=True)
    
    # Cluster interpretations
    st.markdown("### Business Insights")
    cluster_names = {
        0: "💰 **Careful Spenders** - Low income, low spending. Target with discounts and value bundles.",
        1: "👑 **High Rollers** - High income, high spending. VIP treatment and premium offerings.",
        2: "🏦 **Affluent Savers** - High income, low spending. Upselling opportunity with quality emphasis.",
        3: "👥 **Average Customers** - Medium income, medium spending. Standard marketing approach.",
        4: "🛒 **Aspirational Buyers** - Low income, high spending. Financing options and trendy products."
    }
    
    for cluster_id in range(min(optimal_k, 5)):
        if cluster_id in cluster_names:
            st.markdown(f"**Cluster {cluster_id}:** {cluster_names[cluster_id]}")

# Page: DBSCAN Analysis
elif page == "🔍 DBSCAN Analysis":
    st.header("DBSCAN Clustering Analysis")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Clusters Found", metrics['dbscan']['n_clusters'])
    with col2:
        st.metric("Noise Points", f"{metrics['dbscan']['n_noise']} ({metrics['dbscan']['n_noise']/len(df)*100:.1f}%)")
    with col3:
        if metrics['dbscan']['silhouette']:
            st.metric("Silhouette Score", f"{metrics['dbscan']['silhouette']:.4f}")
        else:
            st.metric("Silhouette Score", "N/A")
    with col4:
        st.metric("Eps", f"{best_params['eps']}")
    
    st.markdown(f"**Parameters Used:** eps={best_params['eps']}, min_samples={best_params['min_samples']}")
    
    # Clustering visualization
    st.markdown("### DBSCAN Clustering Results")
    
    df_dbscan = df.copy()
    df_dbscan['Cluster'] = df_dbscan['DBSCAN_Cluster'].apply(lambda x: 'Noise' if x == -1 else str(x))
    
    fig_dbscan = px.scatter(df_dbscan, 
                           x='Annual Income (k$)', 
                           y='Spending Score (1-100)',
                           color='Cluster',
                           title=f'DBSCAN Clustering (eps={best_params["eps"]}, min_samples={best_params["min_samples"]})',
                           hover_data=['Age', 'Gender'],
                           color_discrete_map={'Noise': 'lightgray'})
    
    fig_dbscan.update_layout(height=600)
    st.plotly_chart(fig_dbscan, use_container_width=True)
    
    # Cluster profiles
    st.markdown("### Cluster Profiles")
    
    cluster_profiles = []
    unique_clusters = sorted([c for c in df['DBSCAN_Cluster'].unique() if c != -1])
    
    for i in unique_clusters:
        cluster_data = df[df['DBSCAN_Cluster'] == i]
        profile = {
            'Cluster': i,
            'Size': len(cluster_data),
            'Avg Age': f"{cluster_data['Age'].mean():.1f}",
            'Avg Income': f"${cluster_data['Annual Income (k$)'].mean():.1f}k",
            'Avg Spending': f"{cluster_data['Spending Score (1-100)'].mean():.1f}",
            'Male': len(cluster_data[cluster_data['Gender'] == 'Male']),
            'Female': len(cluster_data[cluster_data['Gender'] == 'Female'])
        }
        cluster_profiles.append(profile)
    
    # Add noise profile
    if metrics['dbscan']['n_noise'] > 0:
        noise_data = df[df['DBSCAN_Cluster'] == -1]
        profile = {
            'Cluster': 'Noise',
            'Size': len(noise_data),
            'Avg Age': f"{noise_data['Age'].mean():.1f}",
            'Avg Income': f"${noise_data['Annual Income (k$)'].mean():.1f}k",
            'Avg Spending': f"{noise_data['Spending Score (1-100)'].mean():.1f}",
            'Male': len(noise_data[noise_data['Gender'] == 'Male']),
            'Female': len(noise_data[noise_data['Gender'] == 'Female'])
        }
        cluster_profiles.append(profile)
    
    cluster_df = pd.DataFrame(cluster_profiles)
    st.dataframe(cluster_df, use_container_width=True)
    
    # DBSCAN advantages
    st.markdown("### DBSCAN Advantages")
    st.info("""
    - ✅ Discovers clusters of arbitrary shapes (not limited to spherical)
    - ✅ Automatically identifies outliers/noise points
    - ✅ Does not require specifying the number of clusters
    - ✅ Robust to outliers and works well with spatial data
    """)

# Page: Comparison
elif page == "⚖️ Comparison":
    st.header("Algorithm Comparison")
    
    # Side-by-side metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 K-Means")
        st.metric("Clusters", optimal_k)
        st.metric("Silhouette Score", f"{metrics['kmeans']['silhouette']:.4f}")
        st.metric("Davies-Bouldin Index", f"{metrics['kmeans']['davies_bouldin']:.4f}")
        st.metric("Calinski-Harabasz", f"{metrics['kmeans']['calinski_harabasz']:.2f}")
        st.metric("Noise Detection", "No")
    
    with col2:
        st.subheader("🔍 DBSCAN")
        st.metric("Clusters", metrics['dbscan']['n_clusters'])
        if metrics['dbscan']['silhouette']:
            st.metric("Silhouette Score", f"{metrics['dbscan']['silhouette']:.4f}")
            st.metric("Davies-Bouldin Index", f"{metrics['dbscan']['davies_bouldin']:.4f}")
            st.metric("Calinski-Harabasz", f"{metrics['dbscan']['calinski_harabasz']:.2f}")
        else:
            st.metric("Silhouette Score", "N/A")
            st.metric("Davies-Bouldin Index", "N/A")
            st.metric("Calinski-Harabasz", "N/A")
        st.metric("Noise Detection", f"Yes ({metrics['dbscan']['n_noise']} points)")
    
    # Side-by-side visualization
    st.markdown("### Visual Comparison")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'K-Means (K={optimal_k})', 
                       f'DBSCAN (Clusters={metrics["dbscan"]["n_clusters"]})']
    )
    
    # K-Means plot
    for i in range(optimal_k):
        cluster_data = df[df['KMeans_Cluster'] == i]
        fig.add_trace(
            go.Scatter(
                x=cluster_data['Annual Income (k$)'],
                y=cluster_data['Spending Score (1-100)'],
                mode='markers',
                name=f'K-Means Cluster {i}',
                legendgroup='kmeans',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add K-Means centroids
    centroids = scaler.inverse_transform(kmeans_model.cluster_centers_)
    fig.add_trace(
        go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode='markers',
            marker=dict(symbol='x', size=15, color='red', line=dict(width=2, color='black')),
            name='Centroids',
            legendgroup='kmeans',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # DBSCAN plot
    unique_clusters = sorted(df['DBSCAN_Cluster'].unique())
    for i in unique_clusters:
        cluster_data = df[df['DBSCAN_Cluster'] == i]
        fig.add_trace(
            go.Scatter(
                x=cluster_data['Annual Income (k$)'],
                y=cluster_data['Spending Score (1-100)'],
                mode='markers',
                name=f'DBSCAN {"Noise" if i == -1 else f"Cluster {i}"}',
                legendgroup='dbscan',
                showlegend=True,
                marker=dict(color='lightgray' if i == -1 else None, symbol='x' if i == -1 else 'circle')
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Annual Income (k$)", row=1, col=1)
    fig.update_xaxes(title_text="Annual Income (k$)", row=1, col=2)
    fig.update_yaxes(title_text="Spending Score", row=1, col=1)
    fig.update_yaxes(title_text="Spending Score", row=1, col=2)
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison table
    st.markdown("### Detailed Comparison")
    
    comparison_data = {
        'Metric': ['Number of Clusters', 'Silhouette Score', 'Davies-Bouldin Index', 
                   'Calinski-Harabasz Score', 'Noise Points', 'Cluster Shape', 
                   'Requires K', 'Best For'],
        'K-Means': [
            optimal_k,
            f"{metrics['kmeans']['silhouette']:.4f}",
            f"{metrics['kmeans']['davies_bouldin']:.4f}",
            f"{metrics['kmeans']['calinski_harabasz']:.2f}",
            'None',
            'Spherical',
            'Yes',
            'Well-separated, spherical clusters'
        ],
        'DBSCAN': [
            metrics['dbscan']['n_clusters'],
            f"{metrics['dbscan']['silhouette']:.4f}" if metrics['dbscan']['silhouette'] else 'N/A',
            f"{metrics['dbscan']['davies_bouldin']:.4f}" if metrics['dbscan']['davies_bouldin'] else 'N/A',
            f"{metrics['dbscan']['calinski_harabasz']:.2f}" if metrics['dbscan']['calinski_harabasz'] else 'N/A',
            f"{metrics['dbscan']['n_noise']} points",
            'Arbitrary',
            'No',
            'Irregular shapes, noise detection'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Recommendation
    st.markdown("### 💡 Recommendation")
    if metrics['kmeans']['silhouette'] > (metrics['dbscan']['silhouette'] if metrics['dbscan']['silhouette'] else 0):
        st.success(f"""
        **K-Means is recommended** for this dataset because:
        - Higher silhouette score ({metrics['kmeans']['silhouette']:.4f} vs {metrics['dbscan']['silhouette']:.4f if metrics['dbscan']['silhouette'] else 'N/A'})
        - Clusters are relatively well-separated and spherical
        - All customers are assigned to actionable groups
        - Easier interpretation for business stakeholders
        """)
    else:
        st.success(f"""
        **DBSCAN is recommended** for this dataset because:
        - Better at identifying irregular cluster shapes
        - Automatically detects outliers ({metrics['dbscan']['n_noise']} noise points)
        - No need to pre-specify number of clusters
        - More robust to unusual customer behaviors
        """)

# Page: Interactive Clustering
elif page == "🎮 Interactive Clustering":
    st.header("Interactive Clustering")
    st.markdown("Adjust parameters and see how clustering changes in real-time!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("K-Means Parameters")
        interactive_k = st.slider("Number of Clusters (K)", 2, 10, optimal_k)
        
        if st.button("Apply K-Means", type="primary"):
            kmeans_interactive = KMeans(n_clusters=interactive_k, init='k-means++', 
                                       random_state=42, n_init=10)
            labels_interactive = kmeans_interactive.fit_predict(X_scaled)
            
            df_interactive = df.copy()
            df_interactive['Cluster'] = labels_interactive.astype(str)
            
            fig = px.scatter(df_interactive, 
                           x='Annual Income (k$)', 
                           y='Spending Score (1-100)',
                           color='Cluster',
                           title=f'K-Means with K={interactive_k}',
                           hover_data=['Age', 'Gender'])
            
            centroids = scaler.inverse_transform(kmeans_interactive.cluster_centers_)
            fig.add_trace(go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode='markers',
                marker=dict(symbol='x', size=20, color='red', 
                          line=dict(width=2, color='black')),
                name='Centroids'
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            if len(set(labels_interactive)) > 1:
                sil_score = silhouette_score(X_scaled, labels_interactive)
                st.metric("Silhouette Score", f"{sil_score:.4f}")
    
    with col2:
        st.subheader("DBSCAN Parameters")
        interactive_eps = st.slider("Epsilon (eps)", 0.1, 1.0, best_params['eps'], 0.1)
        interactive_min_samples = st.slider("Min Samples", 2, 15, best_params['min_samples'])
        
        if st.button("Apply DBSCAN", type="primary"):
            dbscan_interactive = DBSCAN(eps=interactive_eps, min_samples=interactive_min_samples)
            labels_interactive = dbscan_interactive.fit_predict(X_scaled)
            
            df_interactive = df.copy()
            df_interactive['Cluster'] = [f'Noise' if x == -1 else str(x) for x in labels_interactive]
            
            fig = px.scatter(df_interactive, 
                           x='Annual Income (k$)', 
                           y='Spending Score (1-100)',
                           color='Cluster',
                           title=f'DBSCAN with eps={interactive_eps}, min_samples={interactive_min_samples}',
                           hover_data=['Age', 'Gender'],
                           color_discrete_map={'Noise': 'lightgray'})
            
            st.plotly_chart(fig, use_container_width=True)
            
            n_clusters = len(set(labels_interactive)) - (1 if -1 in labels_interactive else 0)
            n_noise = list(labels_interactive).count(-1)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Clusters", n_clusters)
            with col_b:
                st.metric("Noise Points", f"{n_noise} ({n_noise/len(labels_interactive)*100:.1f}%)")
            
            if n_clusters > 1 and n_noise < len(labels_interactive):
                mask = labels_interactive != -1
                if mask.sum() > 0:
                    sil_score = silhouette_score(X_scaled[mask], labels_interactive[mask])
                    st.metric("Silhouette Score", f"{sil_score:.4f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Mall Customer Segmentation Dashboard | K-Means & DBSCAN Clustering</p>
    <p>Dataset: <a href='https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python' target='_blank'>Kaggle Mall Customers</a></p>
</div>
""", unsafe_allow_html=True)