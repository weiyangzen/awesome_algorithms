"""Minimal runnable MVP for k-medoids (PAM-style) clustering.

This script provides:
1) A transparent k-medoids implementation (build + swap style updates).
2) A robustness comparison against k-means on data with injected outliers.
3) Non-interactive, reproducible output with basic quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score


@dataclass
class KMedoidsResult:
    """Container for k-medoids outputs."""

    medoid_indices: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int
    converged: bool


def pairwise_distance_matrix(x: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute full pairwise distance matrix D where D[i, j] = dist(x_i, x_j)."""
    if x.ndim != 2:
        raise ValueError("x must be a 2-D array of shape (n_samples, n_features)")
    d = cdist(x, x, metric=metric)
    if not np.all(np.isfinite(d)):
        raise ValueError("distance matrix contains non-finite values")
    return d


def assign_to_medoids(distance_matrix: np.ndarray, medoid_indices: np.ndarray) -> tuple[np.ndarray, float]:
    """Assign each sample to nearest medoid and return labels and objective value."""
    distances_to_medoids = distance_matrix[:, medoid_indices]
    labels = np.argmin(distances_to_medoids, axis=1)
    min_distances = distances_to_medoids[np.arange(distance_matrix.shape[0]), labels]
    objective = float(np.sum(min_distances))
    return labels, objective


def initialize_medoids_greedy(distance_matrix: np.ndarray, k: int) -> np.ndarray:
    """Greedy medoid initialization (deterministic).

    Step 1: choose global 1-medoid minimizer.
    Step 2+: iteratively add medoid that most reduces objective.
    """
    n = distance_matrix.shape[0]
    if not (1 <= k <= n):
        raise ValueError("k must satisfy 1 <= k <= n_samples")

    first = int(np.argmin(np.sum(distance_matrix, axis=1)))
    medoids: List[int] = [first]

    for _ in range(1, k):
        best_candidate = None
        best_objective = np.inf

        for candidate in range(n):
            if candidate in medoids:
                continue
            trial = np.array(medoids + [candidate], dtype=int)
            _, objective = assign_to_medoids(distance_matrix, trial)
            if objective < best_objective:
                best_objective = objective
                best_candidate = candidate

        if best_candidate is None:
            raise RuntimeError("Failed to initialize medoids")
        medoids.append(best_candidate)

    return np.array(sorted(medoids), dtype=int)


def fit_k_medoids(
    x: np.ndarray,
    k: int,
    metric: str = "euclidean",
    max_iter: int = 100,
) -> KMedoidsResult:
    """Fit k-medoids via a PAM-style local search.

    Update rule:
    - Given current medoids, test every medoid/non-medoid swap.
    - Accept the best improving swap.
    - Stop when no swap reduces objective.
    """
    if x.ndim != 2:
        raise ValueError("x must be 2-D")
    if x.shape[0] < k:
        raise ValueError("n_samples must be >= k")

    distance_matrix = pairwise_distance_matrix(x, metric=metric)
    medoids = initialize_medoids_greedy(distance_matrix, k)
    labels, best_objective = assign_to_medoids(distance_matrix, medoids)

    converged = False
    n = x.shape[0]

    for iteration in range(1, max_iter + 1):
        medoid_set = set(int(m) for m in medoids)
        non_medoids = [idx for idx in range(n) if idx not in medoid_set]

        improved = False
        best_swap_medoids = medoids
        best_swap_labels = labels
        best_swap_objective = best_objective

        for mi, old_medoid in enumerate(medoids):
            for candidate in non_medoids:
                trial = medoids.copy()
                trial[mi] = candidate
                trial = np.array(sorted(trial), dtype=int)

                trial_labels, trial_objective = assign_to_medoids(distance_matrix, trial)
                if trial_objective + 1e-12 < best_swap_objective:
                    improved = True
                    best_swap_objective = trial_objective
                    best_swap_medoids = trial
                    best_swap_labels = trial_labels

        medoids = best_swap_medoids
        labels = best_swap_labels
        best_objective = best_swap_objective

        if not improved:
            converged = True
            return KMedoidsResult(
                medoid_indices=medoids,
                labels=labels,
                inertia=best_objective,
                n_iter=iteration,
                converged=converged,
            )

    return KMedoidsResult(
        medoid_indices=medoids,
        labels=labels,
        inertia=best_objective,
        n_iter=max_iter,
        converged=converged,
    )


def make_dataset(seed: int = 226) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate clustered data and append explicit far outliers."""
    x_core, y_core = make_blobs(
        n_samples=270,
        centers=3,
        cluster_std=[0.75, 0.80, 0.70],
        center_box=(-4.0, 4.0),
        random_state=seed,
    )

    rng = np.random.default_rng(seed + 17)
    outliers = rng.uniform(low=-12.0, high=12.0, size=(30, x_core.shape[1]))

    x = np.vstack([x_core, outliers])
    y = np.concatenate([y_core, np.full(outliers.shape[0], -1, dtype=int)])
    is_outlier = np.concatenate(
        [
            np.zeros(x_core.shape[0], dtype=bool),
            np.ones(outliers.shape[0], dtype=bool),
        ]
    )
    return x.astype(np.float64), y.astype(int), is_outlier


def summarize_cluster_sizes(labels: np.ndarray, prefix: str) -> pd.DataFrame:
    values, counts = np.unique(labels, return_counts=True)
    return pd.DataFrame(
        {
            "cluster": [f"{prefix}{int(v)}" for v in values],
            "size": counts,
        }
    )


def main() -> None:
    x, y, is_outlier = make_dataset(seed=226)
    k = 3

    kmed = fit_k_medoids(x, k=k, metric="euclidean", max_iter=60)

    kmeans = KMeans(n_clusters=k, n_init=20, random_state=226)
    kmeans_labels = kmeans.fit_predict(x)

    sil_kmed = float(silhouette_score(x, kmed.labels, metric="euclidean"))
    sil_kmeans = float(silhouette_score(x, kmeans_labels, metric="euclidean"))

    core_mask = ~is_outlier
    ari_kmed_core = float(adjusted_rand_score(y[core_mask], kmed.labels[core_mask]))
    ari_kmeans_core = float(adjusted_rand_score(y[core_mask], kmeans_labels[core_mask]))

    medoid_points = x[kmed.medoid_indices]
    mean_outlier_to_medoid = float(np.mean(np.min(cdist(x[is_outlier], medoid_points), axis=1)))

    summary = pd.DataFrame(
        {
            "model": ["k-medoids", "k-means"],
            "objective_like": [kmed.inertia, float(kmeans.inertia_)],
            "silhouette": [sil_kmed, sil_kmeans],
            "ARI_on_non_outliers": [ari_kmed_core, ari_kmeans_core],
        }
    )

    size_table = pd.concat(
        [
            summarize_cluster_sizes(kmed.labels, "medoid_"),
            summarize_cluster_sizes(kmeans_labels, "kmeans_"),
        ],
        ignore_index=True,
    )

    assert kmed.medoid_indices.shape == (k,), "medoid count mismatch"
    assert np.unique(kmed.medoid_indices).shape[0] == k, "medoids must be unique"
    assert np.all(kmed.labels >= 0) and np.all(kmed.labels < k), "invalid k-medoids labels"
    assert np.isfinite(kmed.inertia) and kmed.inertia > 0.0, "invalid k-medoids objective"
    assert kmed.converged, "k-medoids did not converge under current setup"

    print("k-medoids (PAM-style) MVP")
    print(f"samples={x.shape[0]}, features={x.shape[1]}, k={k}, outliers={int(np.sum(is_outlier))}")
    print(
        f"kmedoids_iterations={kmed.n_iter}, medoid_indices={kmed.medoid_indices.tolist()}"
    )
    print("medoid_points=")
    print(np.round(medoid_points, 3))
    print()

    print("metric_summary=")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
    print()

    print("cluster_size_summary=")
    print(size_table.to_string(index=False))
    print()

    print(f"mean_outlier_to_nearest_medoid={mean_outlier_to_medoid:.4f}")


if __name__ == "__main__":
    main()
