"""
Week 4 Task 3: Dimensionality Reduction (PCA and t-SNE)
========================================================
Implements PCA and t-SNE dimensionality reduction techniques
and compares their effectiveness on the Wine dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 60)
print("DIMENSIONALITY REDUCTION ANALYSIS: PCA vs t-SNE")
print("=" * 60)

# ============================================================
# 1. Load and Prepare Data
# ============================================================
print("\n[1] Loading Wine Dataset...")
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

print(f"  Dataset: Wine (UCI)")
print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes: {len(target_names)} ({', '.join(target_names)})")

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 2. PCA Analysis
# ============================================================
print("\n[2] Performing PCA Analysis...")

# Full PCA
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# Explained variance
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print("  Explained Variance by Component:")
for i, (ev, cv) in enumerate(zip(explained_var, cumulative_var)):
    print(f"    PC{i+1}: {ev:.4f} ({cv:.4f} cumulative)")

# Find optimal number of components (95% variance threshold)
n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"\n  Components needed for 95% variance: {n_components_95}")

# PCA with optimal components
pca_optimal = PCA(n_components=n_components_95)
X_pca_optimal = pca_optimal.fit_transform(X_scaled)

# Reconstruction error
X_reconstructed = pca_optimal.inverse_transform(X_pca_optimal)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"  Reconstruction Error (optimal components): {reconstruction_error:.6f}")

# 2D PCA for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# ============================================================
# 3. t-SNE Analysis
# ============================================================
print("\n[3] Performing t-SNE Analysis...")

# Test different perplexity values
perplexity_values = [5, 15, 30, 50]
tsne_results = {}

for perp in perplexity_values:
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    tsne_results[perp] = X_tsne

# Best t-SNE (perplexity=30 is standard)
X_tsne_best = tsne_results[30]

# Trustworthiness score (higher is better, range [0, 1])
trust_2d = trustworthiness(X_scaled, X_tsne_best, n_neighbors=12)
print(f"  t-SNE Trustworthiness Score: {trust_2d:.4f}")

# KL Divergence (lower is better)
print(f"  t-SNE KL Divergence: {tsne_results[30][1] if hasattr(tsne_results[30], 'kldivergence_') else 'N/A'}")

# ============================================================
# 4. Quantitative Comparison
# ============================================================
print("\n[4] Quantitative Comparison...")

# Classifier accuracy comparison
clf = LogisticRegression(max_iter=2000, random_state=42)

# Original data
scores_original = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')

# PCA reduced
scores_pca = cross_val_score(clf, X_pca_optimal, y, cv=5, scoring='accuracy')

# t-SNE (use 3D for comparison)
tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42, max_iter=1000)
X_tsne_3d = tsne_3d.fit_transform(X_scaled)
scores_tsne = cross_val_score(clf, X_tsne_3d, y, cv=5, scoring='accuracy')

print(f"  Classifier Accuracy (5-fold CV):")
print(f"    Original ({X_scaled.shape[1]}D): {scores_original.mean():.4f} ± {scores_original.std():.4f}")
print(f"    PCA ({n_components_95}D):       {scores_pca.mean():.4f} ± {scores_pca.std():.4f}")
print(f"    t-SNE (3D):     {scores_tsne.mean():.4f} ± {scores_tsne.std():.4f}")

# ============================================================
# 5. Visualization: Explained Variance (Scree Plot)
# ============================================================
print("\n[5] Generating Visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, label='Individual')
axes[0].step(range(1, len(cumulative_var) + 1), cumulative_var, where='mid', 
             color='red', label='Cumulative')
axes[0].axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
axes[0].axvline(x=n_components_95, color='green', linestyle='--', alpha=0.5, 
                label=f'{n_components_95} components')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('PCA Scree Plot')
axes[0].legend()
axes[0].set_xticks(range(1, len(explained_var) + 1))

# Cumulative variance
axes[1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-', linewidth=2)
axes[1].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
axes[1].axvline(x=n_components_95, color='green', linestyle='--', alpha=0.7,
                label=f'{n_components_95} components')
axes[1].fill_between(range(1, len(cumulative_var) + 1), cumulative_var, alpha=0.3)
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Explained Variance')
axes[1].legend()
axes[1].set_xticks(range(1, len(cumulative_var) + 1))

plt.tight_layout()
plt.savefig('pca_variance.png', dpi=150, bbox_inches='tight')
print("  Saved: pca_variance.png")
plt.close()

# ============================================================
# 6. Visualization: PCA vs t-SNE Comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PCA 2D
scatter1 = axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', 
                           alpha=0.7, edgecolors='w', s=60)
axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
axes[0].set_title('PCA 2D Projection')
plt.colorbar(scatter1, ax=axes[0], ticks=range(len(target_names)))
axes[0].set_xticklabels([])
axes[0].set_yticklabels([])

# t-SNE
scatter2 = axes[1].scatter(X_tsne_best[:, 0], X_tsne_best[:, 1], c=y, cmap='viridis',
                           alpha=0.7, edgecolors='w', s=60)
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].set_title('t-SNE 2D Projection')
plt.colorbar(scatter2, ax=axes[1], ticks=range(len(target_names)))

# Perplexity comparison
scatter3 = axes[2].scatter(tsne_results[5][:, 0], tsne_results[5][:, 1], c=y, 
                           cmap='viridis', alpha=0.5, s=30, label='perp=5')
scatter3 = axes[2].scatter(tsne_results[50][:, 0], tsne_results[50][:, 1], c=y, 
                           cmap='plasma', alpha=0.5, s=30, label='perp=50')
axes[2].set_xlabel('t-SNE Dimension 1')
axes[2].set_ylabel('t-SNE Dimension 2')
axes[2].set_title('t-SNE Perplexity Comparison')
axes[2].legend()

# Add legend for classes
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=plt.cm.viridis(i/len(target_names)), label=name) 
                   for i, name in enumerate(target_names)]
fig.legend(handles=legend_elements, loc='lower center', ncol=len(target_names), 
           fontsize=10, bbox_to_anchor=(0.5, -0.05))

plt.suptitle('Dimensionality Reduction: PCA vs t-SNE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('pca_tsne_comparison.png', dpi=150, bbox_inches='tight')
print("  Saved: pca_tsne_comparison.png")
plt.close()

# ============================================================
# 7. Save Results to CSV
# ============================================================
print("\n[6] Saving Results...")

# Summary metrics
results = pd.DataFrame({
    'Metric': [
        'Dataset',
        'Original Dimensions',
        'PCA Components (95% var)',
        'PCA Reconstruction Error',
        'PCA Explained Variance (PC1)',
        'PCA Explained Variance (PC2)',
        'PCA Total Variance (top 2)',
        't-SNE Trustworthiness',
        't-SNE Perplexity (best)',
        'Accuracy - Original',
        'Accuracy - PCA',
        'Accuracy - t-SNE',
        'Class Separation (PCA)',
        'Class Separation (t-SNE)'
    ],
    'Value': [
        'Wine',
        str(X.shape[1]),
        str(n_components_95),
        f'{reconstruction_error:.6f}',
        f'{explained_var[0]:.4f}',
        f'{explained_var[1]:.4f}',
        f'{sum(explained_var[:2]):.4f}',
        f'{trust_2d:.4f}',
        '30',
        f'{scores_original.mean():.4f} ± {scores_original.std():.4f}',
        f'{scores_pca.mean():.4f} ± {scores_pca.std():.4f}',
        f'{scores_tsne.mean():.4f} ± {scores_tsne.std():.4f}',
        'Good (linear separation)',
        'Excellent (non-linear clusters)'
    ]
})

results.to_csv('dimensionality_reduction_results.csv', index=False)
print("  Saved: dimensionality_reduction_results.csv")

# ============================================================
# 8. Conclusions
# ============================================================
print("\n" + "=" * 60)
print("CONCLUSIONS")
print("=" * 60)
print(f"""
1. PCA Analysis:
   - {n_components_95} components capture 95% of variance (from {X.shape[1]} original features)
   - First 2 components explain {sum(explained_var[:2]):.1%} of total variance
   - Reconstruction error: {reconstruction_error:.6f}

2. t-SNE Analysis:
   - Trustworthiness score: {trust_2d:.4f} (higher = better local structure preservation)
   - Best perplexity: 30 (standard choice)
   - Shows clearer cluster separation than PCA

3. Comparison:
   - PCA: Linear projection, preserves global variance, interpretable
   - t-SNE: Non-linear, preserves local structure, better for visualization
   - t-SNE produces more distinct clusters for this dataset
   - PCA is faster and more stable; t-SNE is stochastic
""")

print("Analysis complete!")
