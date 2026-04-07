"""K-Means clustering MVP (scratch implementation + sklearn baseline).

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


@dataclass
class ClusteringMetrics:
    """Container for clustering quality measurements."""

    inertia: float
    ari: float
    silhouette: float
    n_iter: int


def pairwise_sq_euclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return squared Euclidean distance matrix with shape (n_x, n_y)."""
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    dist_sq = x_norm + y_norm - 2.0 * (x @ y.T)
    return np.maximum(dist_sq, 0.0)


def kmeans_plusplus_init(
    x: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initialize centers with k-means++."""
    n_samples, n_features = x.shape
    if n_clusters > n_samples:
        raise ValueError("n_clusters must be <= number of samples")

    centers = np.empty((n_clusters, n_features), dtype=np.float64)
    first_idx = int(rng.integers(0, n_samples))
    centers[0] = x[first_idx]

    closest_dist_sq = pairwise_sq_euclidean(x, centers[0:1]).reshape(-1)
    for c in range(1, n_clusters):
        total = float(np.sum(closest_dist_sq))
        if total <= 0:
            candidate_idx = int(rng.integers(0, n_samples))
        else:
            probs = closest_dist_sq / total
            candidate_idx = int(rng.choice(n_samples, p=probs))
        centers[c] = x[candidate_idx]
        dist_to_new = pairwise_sq_euclidean(x, centers[c : c + 1]).reshape(-1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_to_new)
    return centers


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-mean, unit-variance standardization."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    x_std = (x - mean) / std
    return x_std, mean, std


def generate_synthetic_blobs(
    n_per_cluster: int = 160,
    seed: int = 225,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a reproducible 2D dataset with three Gaussian clusters."""
    rng = np.random.default_rng(seed)
    true_centers = np.array([[-4.0, -1.2], [0.0, 3.4], [3.8, -0.2]], dtype=np.float64)
    stds = np.array([0.75, 0.9, 0.7], dtype=np.float64)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for label, (center, std) in enumerate(zip(true_centers, stds)):
        pts = rng.normal(loc=center, scale=std, size=(n_per_cluster, 2))
        xs.append(pts)
        ys.append(np.full(n_per_cluster, label, dtype=np.int64))

    x = np.vstack(xs)
    y = np.concatenate(ys)
    order = np.arange(x.shape[0])
    rng.shuffle(order)
    return x[order], y[order], true_centers


class KMeansScratch:
    """A compact Lloyd-style K-Means implementation with multi-start."""

    def __init__(
        self,
        n_clusters: int,
        n_init: int = 12,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 225,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if n_init <= 0:
            raise ValueError("n_init must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tol < 0:
            raise ValueError("tol must be non-negative")

        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)

        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    def _assign(self, x: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dist_sq = pairwise_sq_euclidean(x, centers)
        labels = np.argmin(dist_sq, axis=1)
        min_dist_sq = dist_sq[np.arange(x.shape[0]), labels]
        return labels, min_dist_sq

    def fit(self, x: np.ndarray) -> "KMeansScratch":
        if x.ndim != 2:
            raise ValueError("x must be 2D")
        if x.shape[0] < self.n_clusters:
            raise ValueError("n_clusters cannot exceed sample count")

        x = np.asarray(x, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centers: np.ndarray | None = None
        best_labels: np.ndarray | None = None
        best_n_iter = 0

        for _ in range(self.n_init):
            centers = kmeans_plusplus_init(x, self.n_clusters, rng)
            n_iter_run = 0

            for it in range(1, self.max_iter + 1):
                labels, min_dist_sq = self._assign(x, centers)
                new_centers = np.empty_like(centers)

                for k in range(self.n_clusters):
                    mask = labels == k
                    if np.any(mask):
                        new_centers[k] = x[mask].mean(axis=0)
                    else:
                        # Re-seed empty cluster with current farthest sample.
                        farthest_idx = int(np.argmax(min_dist_sq))
                        new_centers[k] = x[farthest_idx]

                center_shift = float(np.linalg.norm(new_centers - centers))
                centers = new_centers
                n_iter_run = it
                if center_shift <= self.tol:
                    break

            # Final E-step so labels match the final centers.
            labels, min_dist_sq = self._assign(x, centers)
            inertia = float(np.sum(min_dist_sq))

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_n_iter = n_iter_run

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray(x, dtype=np.float64)
        labels, _ = self._assign(x, self.cluster_centers_)
        return labels


def evaluate_clustering(
    x: np.ndarray,
    y_true: np.ndarray,
    labels_pred: np.ndarray,
    inertia: float,
    n_iter: int,
) -> ClusteringMetrics:
    """Compute ARI and silhouette score for diagnostics."""
    ari = float(adjusted_rand_score(y_true, labels_pred))
    sil = float(silhouette_score(x, labels_pred))
    return ClusteringMetrics(inertia=float(inertia), ari=ari, silhouette=sil, n_iter=n_iter)


def center_l2_gap_after_matching(a: np.ndarray, b: np.ndarray) -> float:
    """Compute average center L2 gap after permutation matching."""
    cost = pairwise_sq_euclidean(a, b)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = np.sqrt(cost[row_ind, col_ind])
    return float(np.mean(matched))


def main() -> None:
    print("K-Means clustering MVP (CS-0103)")
    print("=" * 72)

    x_raw, y_true, true_centers = generate_synthetic_blobs(n_per_cluster=160, seed=225)
    x, mean, std = standardize(x_raw)
    n_clusters = 3

    custom = KMeansScratch(
        n_clusters=n_clusters,
        n_init=12,
        max_iter=200,
        tol=1e-4,
        random_state=225,
    ).fit(x)

    sklearn_model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=12,
        max_iter=200,
        tol=1e-4,
        algorithm="lloyd",
        random_state=225,
    ).fit(x)

    metrics_custom = evaluate_clustering(
        x=x,
        y_true=y_true,
        labels_pred=custom.labels_,
        inertia=custom.inertia_,
        n_iter=custom.n_iter_,
    )
    metrics_sklearn = evaluate_clustering(
        x=x,
        y_true=y_true,
        labels_pred=sklearn_model.labels_,
        inertia=sklearn_model.inertia_,
        n_iter=int(sklearn_model.n_iter_),
    )

    summary = pd.DataFrame(
        [
            {
                "model": "scratch-lloyd",
                "inertia": metrics_custom.inertia,
                "ARI_vs_true": metrics_custom.ari,
                "silhouette": metrics_custom.silhouette,
                "n_iter": metrics_custom.n_iter,
            },
            {
                "model": "sklearn-kmeans",
                "inertia": metrics_sklearn.inertia,
                "ARI_vs_true": metrics_sklearn.ari,
                "silhouette": metrics_sklearn.silhouette,
                "n_iter": metrics_sklearn.n_iter,
            },
        ]
    )

    custom_sizes = np.bincount(custom.labels_, minlength=n_clusters)
    sklearn_sizes = np.bincount(sklearn_model.labels_, minlength=n_clusters)
    size_df = pd.DataFrame(
        {
            "cluster_id": np.arange(n_clusters),
            "scratch_size": custom_sizes,
            "sklearn_size": sklearn_sizes,
        }
    )

    center_gap = center_l2_gap_after_matching(custom.cluster_centers_, sklearn_model.cluster_centers_)

    preview = pd.DataFrame(
        {
            "x1_std": x[:8, 0],
            "x2_std": x[:8, 1],
            "true_label": y_true[:8],
            "pred_scratch": custom.labels_[:8],
            "pred_sklearn": sklearn_model.labels_[:8],
        }
    )

    print(f"dataset: n_samples={x.shape[0]}, n_features={x.shape[1]}")
    print(f"raw_center_estimate: {np.round(true_centers, 3).tolist()}")
    print(f"standardize_mean: {np.round(mean, 4).tolist()}")
    print(f"standardize_std: {np.round(std, 4).tolist()}")
    print()
    print("[Model Summary]")
    print(summary.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()
    print("[Cluster Sizes]")
    print(size_df.to_string(index=False))
    print()
    print(f"center_mean_l2_gap(scratch vs sklearn, matched) = {center_gap:.6f}")
    print()
    print("[Prediction Preview]")
    print(preview.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    # Deterministic quality guards for this synthetic dataset.
    if len(np.unique(custom.labels_)) != n_clusters:
        raise RuntimeError("scratch K-Means produced empty/merged clusters in final result")
    if metrics_custom.ari < 0.90:
        raise RuntimeError(f"scratch ARI too low: {metrics_custom.ari:.4f}")
    if metrics_custom.inertia > metrics_sklearn.inertia * 1.10:
        raise RuntimeError("scratch inertia is unexpectedly worse than sklearn baseline")
    if center_gap > 0.30:
        raise RuntimeError(f"center gap too large: {center_gap:.4f}")

    print("=" * 72)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
