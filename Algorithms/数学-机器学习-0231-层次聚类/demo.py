"""Hierarchical clustering (agglomerative) minimal runnable MVP.

The script implements agglomerative clustering from scratch using
single/complete/average linkage, then compares with sklearn/scipy if available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    AgglomerativeClustering = None  # type: ignore[assignment]
    adjusted_rand_score = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False

try:
    from scipy.cluster.hierarchy import fcluster, linkage as scipy_linkage

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    scipy_linkage = None  # type: ignore[assignment]
    fcluster = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False


def adjusted_rand_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Adjusted Rand Index without third-party dependencies."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    n = y_true.size
    if n <= 1:
        return 1.0

    true_labels, true_inverse = np.unique(y_true, return_inverse=True)
    pred_labels, pred_inverse = np.unique(y_pred, return_inverse=True)
    r = true_labels.size
    c = pred_labels.size

    contingency = np.zeros((r, c), dtype=np.int64)
    for i in range(n):
        contingency[true_inverse[i], pred_inverse[i]] += 1

    def comb2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1) // 2

    sum_comb = comb2(contingency).sum(dtype=np.int64)
    row_sum = contingency.sum(axis=1)
    col_sum = contingency.sum(axis=0)
    sum_row = comb2(row_sum).sum(dtype=np.int64)
    sum_col = comb2(col_sum).sum(dtype=np.int64)
    total = n * (n - 1) // 2
    if total == 0:
        return 1.0

    expected = (sum_row * sum_col) / total
    max_index = 0.5 * (sum_row + sum_col)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 0.0
    return float((sum_comb - expected) / denom)


@dataclass
class HierarchicalClusteringMVP:
    linkage: str = "average"

    def __post_init__(self) -> None:
        allowed = {"single", "complete", "average"}
        if self.linkage not in allowed:
            raise ValueError(f"linkage must be one of {allowed}, got {self.linkage}")
        self.linkage_matrix_: np.ndarray | None = None
        self.n_samples_: int = 0
        self.distance_matrix_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "HierarchicalClusteringMVP":
        x = np.asarray(x, dtype=float)
        self._validate_x(x)
        n_samples = x.shape[0]

        dist = self._pairwise_euclidean(x)
        clusters: Dict[int, np.ndarray] = {
            i: np.array([i], dtype=int) for i in range(n_samples)
        }
        active: List[int] = list(range(n_samples))
        next_cluster_id = n_samples
        merges = np.zeros((n_samples - 1, 4), dtype=float)

        for step in range(n_samples - 1):
            best_dist = np.inf
            best_pair: Tuple[int, int] | None = None

            for i_idx in range(len(active) - 1):
                ci = active[i_idx]
                members_i = clusters[ci]
                for j_idx in range(i_idx + 1, len(active)):
                    cj = active[j_idx]
                    members_j = clusters[cj]
                    d = self._cluster_distance(members_i, members_j, dist)

                    if best_pair is None:
                        best_dist = d
                        best_pair = (ci, cj)
                        continue

                    if d < best_dist - 1e-12:
                        best_dist = d
                        best_pair = (ci, cj)
                    elif abs(d - best_dist) <= 1e-12 and (ci, cj) < best_pair:
                        best_pair = (ci, cj)

            assert best_pair is not None
            a, b = best_pair
            merged_members = np.concatenate([clusters[a], clusters[b]])
            merges[step] = [float(a), float(b), float(best_dist), float(merged_members.size)]

            del clusters[a]
            del clusters[b]
            active.remove(a)
            active.remove(b)

            clusters[next_cluster_id] = merged_members
            active.append(next_cluster_id)
            active.sort()
            next_cluster_id += 1

        self.linkage_matrix_ = merges
        self.n_samples_ = n_samples
        self.distance_matrix_ = dist
        return self

    def get_labels(self, n_clusters: int) -> np.ndarray:
        if self.linkage_matrix_ is None:
            raise RuntimeError("Model is not fitted yet.")
        if n_clusters < 2 or n_clusters > self.n_samples_:
            raise ValueError(
                f"n_clusters must be in [2, {self.n_samples_}], got {n_clusters}"
            )

        clusters: Dict[int, List[int]] = {i: [i] for i in range(self.n_samples_)}
        current_count = self.n_samples_

        for step, row in enumerate(self.linkage_matrix_):
            if current_count <= n_clusters:
                break
            left = int(row[0])
            right = int(row[1])
            new_id = self.n_samples_ + step

            merged = clusters.pop(left) + clusters.pop(right)
            clusters[new_id] = merged
            current_count -= 1

        labels = np.empty(self.n_samples_, dtype=int)
        cluster_items = sorted(clusters.values(), key=lambda m: (min(m), len(m)))
        for label, members in enumerate(cluster_items):
            labels[np.array(members, dtype=int)] = label
        return labels

    @staticmethod
    def _validate_x(x: np.ndarray) -> None:
        if x.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={x.shape}")
        if x.shape[0] < 2:
            raise ValueError("Need at least 2 samples for hierarchical clustering")
        if x.shape[1] < 1:
            raise ValueError("Need at least 1 feature")
        if not np.all(np.isfinite(x)):
            raise ValueError("X contains non-finite values")

    @staticmethod
    def _pairwise_euclidean(x: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - x[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        return dist

    def _cluster_distance(
        self, members_a: np.ndarray, members_b: np.ndarray, dist: np.ndarray
    ) -> float:
        block = dist[np.ix_(members_a, members_b)]
        if self.linkage == "single":
            return float(np.min(block))
        if self.linkage == "complete":
            return float(np.max(block))
        return float(np.mean(block))


def make_blobs_like_data(
    seed: int = 2026,
    n_per_cluster: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array([[-3.0, 0.0], [0.5, 3.5], [3.5, -1.5]], dtype=float)

    points = []
    labels = []
    for idx, c in enumerate(centers):
        samples = rng.normal(loc=c, scale=0.6, size=(n_per_cluster, 2))
        points.append(samples)
        labels.extend([idx] * n_per_cluster)
    x = np.vstack(points)
    y = np.array(labels, dtype=int)
    return x, y


def summarize_cluster_sizes(labels: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def run_sklearn_baseline(
    x: np.ndarray, y_true: np.ndarray, n_clusters: int, linkage: str
) -> Tuple[np.ndarray | None, float | None]:
    if not SKLEARN_AVAILABLE:
        return None, None

    try:
        # sklearn>=1.2
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric="euclidean",
        )
    except TypeError:
        # older sklearn fallback
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            affinity="euclidean",
        )

    labels = model.fit_predict(x)
    if adjusted_rand_score is not None:
        ari = float(adjusted_rand_score(y_true, labels))
    else:
        ari = adjusted_rand_index(y_true, labels)
    return labels, ari


def run_scipy_baseline(
    x: np.ndarray, y_true: np.ndarray, n_clusters: int, linkage: str
) -> Tuple[np.ndarray | None, float | None]:
    if not SCIPY_AVAILABLE:
        return None, None
    z = scipy_linkage(x, method=linkage, metric="euclidean")
    labels = fcluster(z, t=n_clusters, criterion="maxclust") - 1
    ari = adjusted_rand_index(y_true, labels)
    return labels.astype(int), ari


def main() -> None:
    linkage = "average"
    n_clusters = 3

    x, y_true = make_blobs_like_data(seed=2026, n_per_cluster=30)
    mvp = HierarchicalClusteringMVP(linkage=linkage).fit(x)
    labels_mvp = mvp.get_labels(n_clusters=n_clusters)

    ari_mvp = adjusted_rand_index(y_true, labels_mvp)

    print("=== Hierarchical Clustering MVP ===")
    print(f"dataset shape: {x.shape}")
    print(f"linkage: {linkage}")
    print(f"n_clusters: {n_clusters}")
    print(f"MVP ARI vs ground truth: {ari_mvp:.6f}")
    print(f"MVP cluster sizes: {summarize_cluster_sizes(labels_mvp)}")

    assert mvp.linkage_matrix_ is not None
    head = mvp.linkage_matrix_[:8]
    print("\nFirst merge rows (left_id, right_id, distance, merged_size):")
    if pd is not None:
        table = pd.DataFrame(
            head, columns=["left_id", "right_id", "distance", "merged_size"]
        )
        with pd.option_context("display.max_rows", 10, "display.width", 120):
            print(table.to_string(index=False))
    else:
        print(np.array2string(head, precision=4, suppress_small=True))

    sk_labels, sk_ari = run_sklearn_baseline(x, y_true, n_clusters, linkage)
    if sk_labels is None:
        print("\nsklearn not available, skip sklearn baseline.")
    else:
        print(f"\nsklearn ARI vs ground truth: {sk_ari:.6f}")
        print(
            "MVP vs sklearn ARI: "
            f"{adjusted_rand_index(labels_mvp, sk_labels):.6f}"
        )

    sp_labels, sp_ari = run_scipy_baseline(x, y_true, n_clusters, linkage)
    if sp_labels is None:
        print("scipy not available, skip scipy baseline.")
    else:
        print(f"scipy ARI vs ground truth: {sp_ari:.6f}")
        print(
            "MVP vs scipy ARI: "
            f"{adjusted_rand_index(labels_mvp, sp_labels):.6f}"
        )


if __name__ == "__main__":
    main()
