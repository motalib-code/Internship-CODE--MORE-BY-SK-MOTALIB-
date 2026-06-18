"""
Week 4 - Task 1: Clustering Algorithms Comparison
===================================================
Compares K-Means, Agglomerative Hierarchical, and DBSCAN clustering
algorithms on the same customer dataset. Includes:
  - Algorithm implementations
  - Evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
  - Visual comparison of clustering results
  - Summary comparison table
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Generate Same Dataset as K-Means Script
# ============================================================
np.random.seed(42)

n_customers = 300
X, y_true = make_blobs(
    n_samples=n_customers,
    n_features=4,
    centers=4,
    cluster_std=[1.5, 1.8, 1.3, 2.0],
    random_state=42
)

feature_names = ['Annual Income', 'Spending Score', 'Age', 'Customer Tenure']
df = pd.DataFrame(X, columns=feature_names)
df['Annual Income'] = df['Annual Income'] * 15 + 60
df['Spending Score'] = df['Spending Score'] * 12 + 50
df['Age'] = np.clip(df['Age'] * 10 + 40, 18, 70).astype(int)
df['Customer Tenure'] = np.clip(df['Customer Tenure'] * 3 + 5, 1, 15).round(1)

X_features = df[feature_names].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

print("=" * 70)
print("  CLUSTERING ALGORITHMS COMPARISON")
print("=" * 70)
print(f"\nDataset: {n_customers} samples, {len(feature_names)} features")
print(f"Features: {feature_names}")

# ============================================================
# 2. K-Means Clustering
# ============================================================
print("\n" + "-" * 70)
print("  1. K-MEANS CLUSTERING")
print("-" * 70)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['KMeans'] = kmeans.fit_predict(X_scaled)

print(f"  Parameters: n_clusters=4, n_init=10")
print(f"  Inertia: {kmeans.inertia_:.2f}")
print(f"  Converged in: {kmeans.n_iter_} iterations")
print(f"  Cluster sizes: {dict(zip(*np.unique(df['KMeans'], return_counts=True)))}")

# ============================================================
# 3. Agglomerative (Hierarchical) Clustering
# ============================================================
print("\n" + "-" * 70)
print("  2. AGGLOMERATIVE HIERARCHICAL CLUSTERING")
print("-" * 70)

# Dendrogram to determine optimal linkage
linked = linkage(X_scaled, method='ward')

fig, ax = plt.subplots(figsize=(14, 7))
dendrogram(
    linked,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True,
    ax=ax
)
ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=14, fontweight='bold')
ax.set_xlabel('Cluster Size', fontsize=12)
ax.set_ylabel('Distance', fontsize=12)
ax.axhline(y=15, color='r', linestyle='--', alpha=0.7, label='Cut threshold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: dendrogram.png")

# Fit with 4 clusters
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
df['Hierarchical'] = hierarchical.fit_predict(X_scaled)

print(f"  Parameters: n_clusters=4, linkage=ward")
print(f"  Cluster sizes: {dict(zip(*np.unique(df['Hierarchical'], return_counts=True)))}")

# ============================================================
# 4. DBSCAN Clustering
# ============================================================
print("\n" + "-" * 70)
print("  3. DBSCAN CLUSTERING")
print("-" * 70)

# Find good eps using k-distance graph
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, 4], axis=0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(distances, linewidth=2)
ax.set_xlabel('Points (sorted by distance)', fontsize=12)
ax.set_ylabel('5th Nearest Neighbor Distance', fontsize=12)
ax.set_title('K-Distance Graph for DBSCAN eps Selection', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kdistance_graph.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: kdistance_graph.png")

# Fit DBSCAN with selected parameters
dbscan = DBSCAN(eps=0.8, min_samples=5)
df['DBSCAN'] = dbscan.fit_predict(X_scaled)

n_clusters_dbscan = len(set(df['DBSCAN'])) - (1 if -1 in df['DBSCAN'].values else 0)
n_noise = list(df['DBSCAN']).count(-1)
print(f"  Parameters: eps=0.8, min_samples=5")
print(f"  Clusters found: {n_clusters_dbscan}")
print(f"  Noise points: {n_noise} ({n_noise/len(df)*100:.1f}%)")
print(f"  Cluster sizes: {dict(zip(*np.unique(df['DBSCAN'], return_counts=True)))}")

# ============================================================
# 5. Evaluation Metrics Comparison
# ============================================================
print("\n" + "=" * 70)
print("  EVALUATION METRICS COMPARISON")
print("=" * 70)

algorithms = {
    'K-Means': df['KMeans'].values,
    'Hierarchical': df['Hierarchical'].values,
    'DBSCAN': df['DBSCAN'].values,
}

results = []
for name, labels in algorithms.items():
    # Filter noise points for DBSCAN
    if name == 'DBSCAN':
        mask = labels != -1
        if mask.sum() < 2:
            continue
        X_eval = X_scaled[mask]
        labels_eval = labels[mask]
        n_clusters = len(set(labels_eval))
    else:
        X_eval = X_scaled
        labels_eval = labels
        n_clusters = len(set(labels_eval))

    sil = silhouette_score(X_eval, labels_eval)
    db = davies_bouldin_score(X_eval, labels_eval)
    ch = calinski_harabasz_score(X_eval, labels_eval)

    results.append({
        'Algorithm': name,
        'Clusters': n_clusters,
        'Silhouette (higher=better)': round(sil, 4),
        'Davies-Bouldin (lower=better)': round(db, 4),
        'Calinski-Harabasz (higher=better)': round(ch, 2),
        'Noise Points': n_noise if name == 'DBSCAN' else 0,
    })

results_df = pd.DataFrame(results)
print()
print(results_df.to_string(index=False))

# ============================================================
# 6. Visual Comparison
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: K-Means
ax = axes[0, 0]
scatter = ax.scatter(df['Annual Income'], df['Spending Score'],
                     c=df['KMeans'], cmap='viridis', alpha=0.6,
                     edgecolors='w', linewidth=0.5, s=60)
centroids_scaled = kmeans.cluster_centers_
centroids_orig = scaler.inverse_transform(centroids_scaled)
ax.scatter(centroids_orig[:, 0], centroids_orig[:, 1],
           c='red', marker='X', s=200, edgecolors='black', linewidths=2, zorder=5)
ax.set_title(f'K-Means (K=4)\nSilhouette: {results_df[results_df["Algorithm"]=="K-Means"]["Silhouette (higher=better)"].values[0]:.4f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Annual Income', fontsize=11)
ax.set_ylabel('Spending Score', fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: Hierarchical
ax = axes[0, 1]
ax.scatter(df['Annual Income'], df['Spending Score'],
           c=df['Hierarchical'], cmap='viridis', alpha=0.6,
           edgecolors='w', linewidth=0.5, s=60)
ax.set_title(f'Hierarchical (Ward, K=4)\nSilhouette: {results_df[results_df["Algorithm"]=="Hierarchical"]["Silhouette (higher=better)"].values[0]:.4f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Annual Income', fontsize=11)
ax.set_ylabel('Spending Score', fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 3: DBSCAN
ax = axes[1, 0]
colors = df['DBSCAN'].copy()
colors[colors == -1] = -1  # Keep noise as -1
ax.scatter(df['Annual Income'], df['Spending Score'],
           c=colors, cmap='viridis', alpha=0.6,
           edgecolors='w', linewidth=0.5, s=60)
# Mark noise points
noise_mask = df['DBSCAN'] == -1
ax.scatter(df.loc[noise_mask, 'Annual Income'], df.loc[noise_mask, 'Spending Score'],
           c='red', marker='x', s=40, alpha=0.5, label='Noise')
ax.set_title(f'DBSCAN (eps=0.8, min=5)\nClusters: {n_clusters_dbscan}, Noise: {n_noise}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Annual Income', fontsize=11)
ax.set_ylabel('Spending Score', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Metrics comparison bar chart
ax = axes[1, 1]
x_pos = np.arange(len(results_df))
width = 0.25

bars1 = ax.bar(x_pos - width, results_df['Silhouette (higher=better)'], width, label='Silhouette', color='steelblue')
ax2 = ax.twinx()
bars2 = ax2.bar(x_pos, results_df['Davies-Bouldin (lower=better)'], width, label='Davies-Bouldin', color='coral')
bars3 = ax2.bar(x_pos + width, results_df['Calinski-Harabasz (higher=better)'] / 1000, width,
                label='Calinski-Harabasz (÷1000)', color='seagreen')

ax.set_xlabel('Algorithm', fontsize=12)
ax.set_ylabel('Silhouette Score', fontsize=12, color='steelblue')
ax2.set_ylabel('DB (÷1000) / CH (÷1000)', fontsize=12, color='gray')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['Algorithm'], fontsize=11)
ax.set_title('Clustering Metrics Comparison', fontsize=13, fontweight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Clustering Algorithms Comparison\nAnnual Income vs Spending Score',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: clustering_comparison.png")

# ============================================================
# 7. Feature Distribution by Cluster (Box Plots)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, feat in enumerate(feature_names):
    ax = axes[idx // 2, idx % 2]
    df_melted = df.melt(id_vars=['KMeans'], value_vars=[feat], var_name='Feature', value_name='Value')
    sns.boxplot(x='KMeans', y='Value', data=df_melted, ax=ax, palette='viridis')
    ax.set_title(f'{feat} by Cluster', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Feature Distributions by K-Means Cluster', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: feature_distributions.png")

# ============================================================
# 8. Summary and Conclusions
# ============================================================
print("\n" + "=" * 70)
print("  SUMMARY & CONCLUSIONS")
print("=" * 70)

best_sil = results_df.loc[results_df['Silhouette (higher=better)'].idxmax()]
best_db = results_df.loc[results_df['Davies-Bouldin (lower=better)'].idxmin()]
best_ch = results_df.loc[results_df['Calinski-Harabasz (higher=better)'].idxmax()]

print(f"""
  Algorithm Strengths:
  --------------------
  K-Means:
    - Fast, scalable, works well with spherical clusters
    - Requires pre-specifying K
    - Sensitive to initial centroids ( mitigated by n_init)
    - Best for: Well-separated, equally-sized clusters

  Hierarchical (Agglomerative):
    - No need to pre-specify K (can cut dendrogram)
    - Produces a dendrogram for visual analysis
    - Computationally expensive for large datasets
    - Best for: When cluster hierarchy matters

  DBSCAN:
    - Automatically finds number of clusters
    - Handles noise/outliers gracefully
    - Can find arbitrarily shaped clusters
    - Sensitive to eps and min_samples parameters
    - Best for: Non-spherical clusters, noisy data

  Best Performers (by metric):
  ---------------------------
  Highest Silhouette Score:    {best_sil['Algorithm']} ({best_sil['Silhouette (higher=better)']:.4f})
  Lowest Davies-Bouldin Index: {best_db['Algorithm']} ({best_db['Davies-Bouldin (lower=better)']:.4f})
  Highest Calinski-Harabasz:   {best_ch['Algorithm']} ({best_ch['Calinski-Harabasz (higher=better)']:.2f})
""")

# Save comparison results
results_df.to_csv('clustering_comparison_results.csv', index=False)
df.to_csv('clustering_comparison_data.csv', index=False)
print("  Saved: clustering_comparison_results.csv")
print("  Saved: clustering_comparison_data.csv")

print("\n" + "=" * 70)
print("  ALL OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 70)
print("""
  Generated Files:
  - kmeans_clustering.py          (K-Means implementation)
  - clustering_comparison.py      (Algorithm comparison)
  - elbow_silhouette.png          (Optimal K selection)
  - kmeans_clusters.png           (K-Means scatter plots)
  - silhouette_analysis.png       (Silhouette per cluster)
  - dendrogram.png                (Hierarchical dendrogram)
  - kdistance_graph.png           (DBSCAN eps selection)
  - clustering_comparison.png     (Visual comparison)
  - feature_distributions.png     (Box plots by cluster)
  - kmeans_results.csv            (K-Means output data)
  - cluster_centroids.csv         (Cluster centroids)
  - clustering_comparison_results.csv  (Metrics comparison)
  - clustering_comparison_data.csv     (Full labeled data)
""")
