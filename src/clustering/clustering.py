"""
Performs clustering and value segmentation for baseball batters and pitchers.
- Uses KMeans and PCA for clustering and visualization
- Computes silhouette scores to select optimal cluster count
- Saves cluster results and visualizations for dashboard use
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# Ensure project root is in sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.clean_data import get_batters_df_normalized, get_pitchers_df_normalized, get_batters_df, get_pitchers_df

# ----------------------
# Silhouette Score Helper
# ----------------------
def compute_silhouette_scores(X, min_k=2, max_k=8, random_state=42):
    """
    Computes silhouette scores for KMeans clustering with k in [min_k, max_k].
    Plots silhouette scores and returns best k.
    """
    scores = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
    plt.figure(figsize=(8, 4))
    plt.plot(range(min_k, max_k + 1), scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.tight_layout()
    plt.show()
    best_k = int(np.argmax(scores)) + min_k
    print(f'Best number of clusters by silhouette score: {best_k}')
    return best_k, scores

# ----------------------
# Clustering and Visualization
# ----------------------
def cluster_and_visualize_value_segments(player_type='batters', n_clusters=None, random_state=42):
    """
    Performs KMeans clustering on normalized player data, labels clusters by value, and visualizes with PCA.
    Saves cluster assignments and visualizations to disk.
    """
    if player_type == 'batters':
        df_norm = get_batters_df_normalized()
        df_raw = get_batters_df()
        war_col = 'b_war'
    elif player_type == 'pitchers':
        df_norm = get_pitchers_df_normalized()
        df_raw = get_pitchers_df()
        war_col = 'p_war'
    else:
        raise ValueError("player_type must be 'batters' or 'pitchers'")
    # Use only numeric columns for clustering
    X = df_norm.dropna(axis=1, how='all').dropna()
    # Align with raw for value calculation
    df_raw = df_raw.loc[X.index]
    # If n_clusters not specified, use silhouette score to find best
    if n_clusters is None:
        best_k, _ = compute_silhouette_scores(X.values)
        n_clusters = int(best_k)
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X.values)
    X['cluster'] = clusters
    # Compute value metric for labeling
    if war_col in df_raw.columns and 'salary' in df_raw.columns:
        value = df_raw[war_col] / df_raw['salary']
    else:
        value = pd.Series(np.nan, index=df_raw.index)
    X['value'] = value
    # Label clusters by mean value
    cluster_means = X.groupby('cluster')['value'].mean().sort_values(ascending=True)
    labels = {}
    sorted_clusters = list(cluster_means.index)
    for i, cluster in enumerate(sorted_clusters):
        if i == 0:
            labels[cluster] = "Overvalued"
        elif i == n_clusters - 1:
            labels[cluster] = "Undervalued"
        else:
            labels[cluster] = "Fairly Valued"
    X['value_label'] = X['cluster'].map(labels)  # type: ignore
    # Add fullName for later identification (after clustering)
    if 'fullName' in df_raw.columns:
        X['fullName'] = df_raw['fullName']
    # --- Cluster profiling ---
    print(f"\nCluster Profiling for {player_type.title()} (mean/std/count):")
    # Only use numeric columns for profiling
    profile_cols = [col for col in X.columns.drop(['cluster', 'value_label']) if pd.api.types.is_numeric_dtype(X[col])]
    profile = X.groupby('value_label')[profile_cols].agg(['mean', 'std', 'count'])
    print(profile)
    print("\n--- End of Cluster Profiling ---\n")
    # PCA for visualization
    pca = PCA(n_components=2, random_state=random_state)
    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col]) and col != 'cluster']
    X_pca = pca.fit_transform(X[numeric_cols])
    # Visualization
    plt.figure(figsize=(10, 6))
    colors = {"Overvalued": "red", "Fairly Valued": "gray", "Undervalued": "green"}
    for label, color in colors.items():
        mask = X['value_label'] == label
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=label,
            alpha=0.7,
            c=color
        )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"{player_type.title()} Value Segments (PCA, Normalized, k={n_clusters})")
    plt.legend()
    plt.tight_layout()
    # Save the figure as PNG
    fig_path = f"reports/figures/{player_type}_value_segments.png"
    plt.savefig(fig_path)
    print(f"Saved cluster visualization to {fig_path}")
    plt.show()
    # Save results to CSV for dashboard use
    output_cols = ['fullName', 'value_label', 'value', 'salary']
    output_cols = [col for col in output_cols if col in X.columns]
    output_path = f"data/processed/{player_type}_value_labels.csv"
    X[output_cols].to_csv(output_path, index=False)
    print(f"Saved value labels to {output_path}")
    return X

# ----------------------
# Main Entrypoint
# ----------------------
if __name__ == "__main__":
    print("Batters Clustering:")
    cluster_and_visualize_value_segments('batters')
    print("\nPitchers Clustering:")
    cluster_and_visualize_value_segments('pitchers')
