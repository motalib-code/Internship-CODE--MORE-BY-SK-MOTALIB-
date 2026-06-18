"""
Week 4 - Task 1: K-Means Clustering Implementation
============================================
This script demonstrates K-Means clustering on a synthetic customer dataset.
It covers:
  - Data generation and exploration
  - Optimal cluster selection (Elbow Method + Silhouette Score)
  - K-Means model fitting and evaluation
  - Visualization of clusters and centroids
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Generate Synthetic Customer Dataset
# ============================================================
np.random.seed(42)

n_customers = 300
n_features = 4  # Annual Income, Spending Score, Age, Customer Tenure

# Create 4 distinct customer segments
X, y_true = make_blobs(
    n_samples=n_customers,
    n_features=n_features,
    centers=4,
    cluster_std=[1.5, 1.8, 1.3, 2.0],
    random_state=42
)

feature_names = ['Annual Income', 'Spending Score', 'Age', 'Customer Tenure (years)']
df = pd.DataFrame(X, columns=feature_names)
df['True_Cluster'] = y_true

# Add realistic scaling to features
df['Annual Income'] = df['Annual Income'] * 15 + 60  # Range ~20-100k
df['Spending Score'] = df['Spending Score'] * 12 + 50  # Range ~10-90
df['Age'] = np.clip(df['Age'] * 10 + 40, 18, 70).astype(int)  # Range 18-70
df['Customer Tenure (years)'] = np.clip(df['Customer Tenure (years)'] * 3 + 5, 1, 15).round(1)

print("=" * 60)
print("  K-MEANS CLUSTERING ON CUSTOMER DATASET")
print("=" * 60)

# ============================================================
# 2. Data Exploration
# ============================================================
print("\n--- Dataset Overview ---")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())

print(f"\nDescriptive Statistics:")
print(df.describe().round(2).to_string())

# ============================================================
# 3. Feature Scaling
# ============================================================
X_features = df[feature_names].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# ============================================================
# 4. Optimal Number of Clusters - Elbow Method
# ============================================================
print("\n--- Finding Optimal K (Elbow Method + Silhouette) ---")
K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(sil_score)
    print(f"  K={k:2d}  |  Inertia: {kmeans.inertia_:10.2f}  |  Silhouette: {sil_score:.4f}")

optimal_k = list(K_range)[np.argmax(silhouette_scores)]
print(f"\n  >> Best K by Silhouette Score: {optimal_k}")

# Plot Elbow and Silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'Optimal K={optimal_k}')
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
axes[0].set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Silhouette plot
axes[1].plot(K_range, silhouette_scores, 'gs-', linewidth=2, markersize=8)
axes[1].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'Optimal K={optimal_k}')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score for Optimal K', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: elbow_silhouette.png")

# ============================================================
# 5. Fit K-Means with Optimal K
# ============================================================
print(f"\n--- Fitting K-Means with K={optimal_k} ---")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

centroids = scaler.inverse_transform(kmeans_final.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=feature_names)
centroid_df.index.name = 'Cluster'
print("\nCluster Centroids (Original Scale):")
print(centroid_df.round(2).to_string())

# ============================================================
# 6. Cluster Profiling
# ============================================================
print("\n--- Cluster Profiles ---")
cluster_profiles = df.groupby('Cluster')[feature_names].agg(['mean', 'std', 'count'])
for cluster_id in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")
    for feat in feature_names:
        mean_val = cluster_data[feat].mean()
        print(f"  {feat:25s}: {mean_val:8.2f}")

# ============================================================
# 7. Visualization - 2D Scatter Plots
# ============================================================
# Pairwise scatter plots of top features
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
palette = sns.color_palette('viridis', n_colors=optimal_k)

pairs = [
    ('Annual Income', 'Spending Score'),
    ('Annual Income', 'Age'),
    ('Annual Income', 'Customer Tenure (years)'),
    ('Spending Score', 'Age'),
    ('Spending Score', 'Customer Tenure (years)'),
    ('Age', 'Customer Tenure (years)'),
]

for idx, (feat_x, feat_y) in enumerate(pairs):
    ax = axes[idx // 3, idx % 3]
    scatter = ax.scatter(
        df[feat_x], df[feat_y],
        c=df['Cluster'], cmap='viridis', alpha=0.6,
        edgecolors='w', linewidth=0.5, s=60
    )
    # Plot centroids
    for c in range(optimal_k):
        ax.scatter(centroids[c][feature_names.index(feat_x)],
                   centroids[c][feature_names.index(feat_y)],
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2, zorder=5)
    ax.set_xlabel(feat_x, fontsize=11)
    ax.set_ylabel(feat_y, fontsize=11)
    ax.set_title(f'{feat_x} vs {feat_y}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle(f'K-Means Clustering Results (K={optimal_k})', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: kmeans_clusters.png")

# ============================================================
# 8. Silhouette Analysis for Final Model
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sample_silhouette_values = silhouette_samples(X_scaled, df['Cluster'].values)
y_lower = 10

for i in range(optimal_k):
    ith_cluster_values = sample_silhouette_values[df['Cluster'] == i]
    ith_cluster_values.sort()
    size_cluster_i = ith_cluster_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = plt.cm.viridis(i / optimal_k)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=12, fontweight='bold')
    y_lower = y_upper + 10

avg_silhouette = silhouette_score(X_scaled, df['Cluster'].values)
ax.axvline(x=avg_silhouette, color='red', linestyle='--', linewidth=2,
           label=f'Average Silhouette: {avg_silhouette:.3f}')
ax.set_xlabel('Silhouette Coefficient', fontsize=12)
ax.set_ylabel('Cluster', fontsize=12)
ax.set_title('Silhouette Analysis for K-Means Clusters', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('silhouette_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: silhouette_analysis.png")

# ============================================================
# 9. Save Results
# ============================================================
df.to_csv('kmeans_results.csv', index=False)
centroid_df.to_csv('cluster_centroids.csv')
print("\n  Saved: kmeans_results.csv")
print("  Saved: cluster_centroids.csv")

print("\n" + "=" * 60)
print("  K-MEANS CLUSTERING COMPLETE")
print("=" * 60)
print(f"\nFinal Model: K-Means with K={optimal_k}")
print(f"Silhouette Score: {avg_silhouette:.4f}")
print(f"Inertia: {kmeans_final.inertia_:.2f}")
print(f"Cluster Distribution:")
for c in sorted(df['Cluster'].unique()):
    print(f"  Cluster {c}: {len(df[df['Cluster']==c])} customers ({len(df[df['Cluster']==c])/len(df)*100:.1f}%)")
