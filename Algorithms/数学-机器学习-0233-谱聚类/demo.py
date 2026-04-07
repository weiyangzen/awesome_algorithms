"""Minimal runnable MVP for 谱聚类 (Spectral Clustering).

This script is intentionally self-contained:
- Core spectral clustering pipeline is implemented with NumPy.
- If scikit-learn is available, an optional comparison is also printed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import SpectralClustering as SklearnSpectralClustering
except Exception:  # pragma: no cover - optional dependency
    SklearnSpectralClustering = None


@dataclass
class SpectralResult:
    labels: np.ndarray
    eigenvalues: np.ndarray
    affinity: np.ndarray


def make_moons_numpy(n_samples: int, noise: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a two-moons dataset using NumPy only."""
    rng = np.random.default_rng(random_state)

    n_a = n_samples // 2
    n_b = n_samples - n_a

    t_a = rng.uniform(0.0, math.pi, size=n_a)
    t_b = rng.uniform(0.0, math.pi, size=n_b)

    moon_a = np.column_stack([np.cos(t_a), np.sin(t_a)])
    moon_b = np.column_stack([1.0 - np.cos(t_b), -np.sin(t_b) - 0.5])

    X = np.vstack([moon_a, moon_b])
    y = np.concatenate([np.zeros(n_a, dtype=int), np.ones(n_b, dtype=int)])

    X += rng.normal(loc=0.0, scale=noise, size=X.shape)

    order = rng.permutation(n_samples)
    return X[order], y[order]


def pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
    sq_norm = np.sum(X * X, axis=1, keepdims=True)
    dist2 = sq_norm + sq_norm.T - 2.0 * (X @ X.T)
    return np.clip(dist2, 0.0, None)


def build_knn_affinity(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Build symmetric unweighted kNN graph as a dense adjacency matrix."""
    n = X.shape[0]
    dist2 = pairwise_squared_distances(X)
    np.fill_diagonal(dist2, np.inf)

    nn_idx = np.argpartition(dist2, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]

    affinity = np.zeros((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    affinity[rows, nn_idx] = 1.0

    # Symmetrize into an undirected graph.
    affinity = np.maximum(affinity, affinity.T)
    return affinity


def count_connected_components(affinity: np.ndarray) -> int:
    """Count connected components by DFS on the adjacency matrix."""
    n = affinity.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = 0

    for start in range(n):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True

        while stack:
            u = stack.pop()
            neighbors = np.flatnonzero(affinity[u] > 0)
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

    return components


def normalized_laplacian(affinity: np.ndarray) -> np.ndarray:
    degrees = np.sum(affinity, axis=1)
    inv_sqrt = np.zeros_like(degrees)
    nonzero = degrees > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    scaled_affinity = affinity * inv_sqrt[:, None] * inv_sqrt[None, :]
    return np.eye(affinity.shape[0]) - scaled_affinity


def top_k_smallest_eigenvectors(L: np.ndarray, k: int, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    order = np.argsort(eigenvalues)[:k]
    selected_values = eigenvalues[order]
    selected_vectors = eigenvectors[:, order]

    norms = np.linalg.norm(selected_vectors, axis=1, keepdims=True)
    embedding = selected_vectors / np.clip(norms, eps, None)
    return selected_values, embedding


def kmeans_numpy(
    X: np.ndarray,
    n_clusters: int,
    random_state: int,
    max_iter: int = 100,
    n_init: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simple NumPy K-Means implementation."""
    rng = np.random.default_rng(random_state)

    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = np.inf

    n_samples = X.shape[0]

    for _ in range(n_init):
        init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centers = X[init_idx].copy()

        for _ in range(max_iter):
            dist2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dist2, axis=1)

            new_centers = np.empty_like(centers)
            for cid in range(n_clusters):
                mask = labels == cid
                if np.any(mask):
                    new_centers[cid] = X[mask].mean(axis=0)
                else:
                    # Re-seed empty cluster with a random data point.
                    new_centers[cid] = X[rng.integers(0, n_samples)]

            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers

        final_dist2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        final_labels = np.argmin(final_dist2, axis=1)
        inertia = float(np.sum(np.min(final_dist2, axis=1)))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = final_labels
            best_centers = centers.copy()

    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers, best_inertia


def spectral_clustering_mvp(
    X: np.ndarray,
    n_clusters: int,
    n_neighbors: int,
    random_state: int,
) -> SpectralResult:
    affinity = build_knn_affinity(X, n_neighbors=n_neighbors)
    L = normalized_laplacian(affinity)
    eigenvalues, embedding = top_k_smallest_eigenvectors(L, k=n_clusters)
    labels, _, _ = kmeans_numpy(
        embedding,
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=100,
        n_init=10,
    )
    return SpectralResult(labels=labels, eigenvalues=eigenvalues, affinity=affinity)


def comb2(x: np.ndarray | int) -> np.ndarray | float:
    return np.asarray(x) * (np.asarray(x) - 1) / 2.0


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute ARI without sklearn."""
    true_classes, true_inv = np.unique(labels_true, return_inverse=True)
    pred_classes, pred_inv = np.unique(labels_pred, return_inverse=True)

    contingency = np.zeros((true_classes.size, pred_classes.size), dtype=np.int64)
    np.add.at(contingency, (true_inv, pred_inv), 1)

    sum_comb_c = comb2(contingency).sum()
    sum_comb_true = comb2(contingency.sum(axis=1)).sum()
    sum_comb_pred = comb2(contingency.sum(axis=0)).sum()
    n = labels_true.size
    total_comb = comb2(n)

    if total_comb == 0:
        return 1.0

    expected = (sum_comb_true * sum_comb_pred) / total_comb
    max_index = 0.5 * (sum_comb_true + sum_comb_pred)
    denom = max_index - expected

    if denom == 0:
        return 1.0

    return float((sum_comb_c - expected) / denom)


def silhouette_score_numpy(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score in O(n^2) for demo scale."""
    n = X.shape[0]
    dist = np.sqrt(pairwise_squared_distances(X))

    unique_labels = np.unique(labels)
    label_masks = {lb: labels == lb for lb in unique_labels}

    s = np.zeros(n, dtype=float)

    for i in range(n):
        own = labels[i]
        own_mask = label_masks[own]
        own_idx = np.flatnonzero(own_mask)

        if own_idx.size <= 1:
            s[i] = 0.0
            continue

        # Intra-cluster distance a(i)
        a = dist[i, own_idx[own_idx != i]].mean()

        # Nearest-cluster distance b(i)
        b = np.inf
        for lb in unique_labels:
            if lb == own:
                continue
            other_idx = np.flatnonzero(label_masks[lb])
            if other_idx.size == 0:
                continue
            b = min(b, dist[i, other_idx].mean())

        s[i] = (b - a) / max(a, b)

    return float(np.mean(s))


def maybe_run_sklearn_comparison(
    X: np.ndarray,
    n_clusters: int,
    n_neighbors: int,
    random_state: int,
    labels_mvp: np.ndarray,
) -> tuple[bool, dict[str, float], np.ndarray | None]:
    """Run sklearn comparison when dependency is available."""
    if SklearnSpectralClustering is None:
        return False, {}, None

    model = SklearnSpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        random_state=random_state,
    )
    labels_sk = model.fit_predict(X)

    metrics = {
        "ari_mvp_vs_sklearn": adjusted_rand_index(labels_mvp, labels_sk),
        "silhouette_sklearn": silhouette_score_numpy(X, labels_sk),
    }
    return True, metrics, labels_sk


def main() -> None:
    n_samples = 500
    n_clusters = 2
    n_neighbors = 12
    random_state = 42

    X, y_true = make_moons_numpy(n_samples=n_samples, noise=0.06, random_state=0)

    result = spectral_clustering_mvp(
        X=X,
        n_clusters=n_clusters,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    n_components = count_connected_components(result.affinity)

    ari_mvp_truth = adjusted_rand_index(y_true, result.labels)
    silhouette_mvp = silhouette_score_numpy(X, result.labels)

    has_sklearn, sk_metrics, labels_sk = maybe_run_sklearn_comparison(
        X=X,
        n_clusters=n_clusters,
        n_neighbors=n_neighbors,
        random_state=random_state,
        labels_mvp=result.labels,
    )

    df = pd.DataFrame(
        {
            "x1": X[:, 0],
            "x2": X[:, 1],
            "y_true": y_true,
            "label_mvp": result.labels,
        }
    )
    if labels_sk is not None:
        df["label_sklearn"] = labels_sk

    cluster_counts = df["label_mvp"].value_counts().sort_index()
    cluster_centers = df.groupby("label_mvp")[["x1", "x2"]].mean().round(4)

    print("=== Spectral Clustering MVP Demo (NumPy-first) ===")
    print(f"Samples: {n_samples}, clusters: {n_clusters}, n_neighbors: {n_neighbors}")
    print(f"Connected components in affinity graph: {n_components}")
    print(f"Smallest eigenvalues (normalized Laplacian): {np.round(result.eigenvalues, 6)}")
    print()

    print("--- MVP Metrics ---")
    print(f"ARI(MVP vs Truth):      {ari_mvp_truth:.4f}")
    print(f"Silhouette(MVP):        {silhouette_mvp:.4f}")

    if has_sklearn:
        print()
        print("--- sklearn Comparison ---")
        print(f"ARI(MVP vs sklearn):    {sk_metrics['ari_mvp_vs_sklearn']:.4f}")
        print(f"Silhouette(sklearn):    {sk_metrics['silhouette_sklearn']:.4f}")
    else:
        print()
        print("--- sklearn Comparison ---")
        print("Skipped: scikit-learn is not installed in current environment.")

    print()
    print("--- MVP Cluster Stats ---")
    print("Counts by cluster:")
    for cid, count in cluster_counts.items():
        print(f"  cluster {cid}: {count}")
    print("Mean feature location by cluster:")
    print(cluster_centers.to_string())


if __name__ == "__main__":
    main()
