"""Educational DBSCAN MVP.

This script implements DBSCAN directly (neighbor query + density expansion)
instead of only calling a library one-liner.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from scipy.spatial.distance import cdist as sp_cdist

    HAS_SCIPY = True
except ModuleNotFoundError:
    sp_cdist = None
    HAS_SCIPY = False

try:
    from sklearn.cluster import DBSCAN as SkDBSCAN
    from sklearn.datasets import make_blobs, make_moons
    from sklearn.metrics import adjusted_rand_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ModuleNotFoundError:
    SkDBSCAN = None
    make_blobs = None
    make_moons = None
    adjusted_rand_score = None
    NearestNeighbors = None
    StandardScaler = None
    HAS_SKLEARN = False


@dataclass
class DBSCANResult:
    labels: np.ndarray
    core_mask: np.ndarray
    neighbor_counts: np.ndarray
    n_clusters: int
    n_noise: int


def standardize(X: np.ndarray) -> np.ndarray:
    """Standardize features to reduce scale sensitivity of eps."""
    if HAS_SKLEARN:
        return StandardScaler().fit_transform(X)
    mu = np.mean(X, axis=0, keepdims=True)
    sigma = np.std(X, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (X - mu) / sigma


def generate_dataset(seed: int = 42) -> np.ndarray:
    """Generate a mixed-density 2D dataset with explicit outliers."""
    rng = np.random.default_rng(seed)

    if HAS_SKLEARN:
        moons_x, _ = make_moons(n_samples=450, noise=0.06, random_state=seed)
        blob_x, _ = make_blobs(
            n_samples=250,
            centers=np.array([[2.5, 2.4], [3.2, 3.1]]),
            cluster_std=[0.20, 0.25],
            random_state=seed,
        )
        noise_x = rng.uniform(low=[-2.0, -1.8], high=[4.8, 4.3], size=(120, 2))
        X = np.vstack([moons_x, blob_x, noise_x]).astype(np.float64)
        return standardize(X)

    # NumPy fallback: two arcs + one Gaussian cluster + uniform noise.
    t = rng.uniform(0.0, np.pi, size=450)
    moon1 = np.column_stack([np.cos(t), np.sin(t)]) + 0.08 * rng.normal(size=(450, 2))
    moon2 = np.column_stack([1.0 - np.cos(t), 0.5 - np.sin(t)]) + 0.08 * rng.normal(size=(450, 2))
    blob = rng.normal(loc=[2.8, 2.7], scale=[0.25, 0.25], size=(220, 2))
    noise = rng.uniform(low=[-2.0, -1.8], high=[4.8, 4.3], size=(120, 2))
    X = np.vstack([moon1, moon2, blob, noise]).astype(np.float64)
    return standardize(X)


def radius_neighbors(X: np.ndarray, eps: float) -> list[np.ndarray]:
    """Return indices within eps-ball for every sample (including itself)."""
    n = X.shape[0]

    if HAS_SKLEARN:
        nn = NearestNeighbors(radius=eps, metric="euclidean")
        nn.fit(X)
        ind = nn.radius_neighbors(X, radius=eps, return_distance=False)
        return [arr.astype(np.int64, copy=False) for arr in ind]

    if HAS_SCIPY:
        D = sp_cdist(X, X, metric="euclidean")
    else:
        diff = X[:, None, :] - X[None, :, :]
        D = np.sqrt(np.sum(diff * diff, axis=2))

    neighbors = [np.flatnonzero(D[i] <= eps).astype(np.int64) for i in range(n)]
    return neighbors


def dbscan_mvp(X: np.ndarray, eps: float, min_samples: int) -> DBSCANResult:
    """Pure algorithmic DBSCAN implementation.

    labels meaning:
    - -99: unassigned (internal)
    - -1 : noise
    - >=0: cluster id
    """
    n = X.shape[0]
    neighbors = radius_neighbors(X, eps=eps)
    neighbor_counts = np.array([arr.size for arr in neighbors], dtype=np.int32)
    core_mask = neighbor_counts >= min_samples

    labels = np.full(n, -99, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        if not core_mask[i]:
            labels[i] = -1
            continue

        # Start a new cluster and expand by density reachability.
        labels[i] = cluster_id
        queue: deque[int] = deque([int(i)])
        in_queue = np.zeros(n, dtype=bool)
        in_queue[i] = True

        while queue:
            j = queue.popleft()

            if not visited[j]:
                visited[j] = True

            if core_mask[j]:
                for nb in neighbors[j]:
                    nb_i = int(nb)
                    if not in_queue[nb_i]:
                        queue.append(nb_i)
                        in_queue[nb_i] = True

            if labels[j] in (-99, -1):
                labels[j] = cluster_id

        cluster_id += 1

    labels[labels == -99] = -1
    n_noise = int(np.sum(labels == -1))
    n_clusters = int(len(set(labels.tolist()) - {-1}))

    return DBSCANResult(
        labels=labels,
        core_mask=core_mask,
        neighbor_counts=neighbor_counts,
        n_clusters=n_clusters,
        n_noise=n_noise,
    )


def evaluate_result(labels: np.ndarray, sk_labels: np.ndarray | None) -> dict[str, float]:
    """Build a compact metrics dictionary."""
    n = labels.shape[0]
    n_clusters = len(set(labels.tolist()) - {-1})
    n_noise = int(np.sum(labels == -1))
    noise_ratio = float(n_noise) / float(max(1, n))

    ari = np.nan
    if sk_labels is not None and HAS_SKLEARN:
        ari = float(adjusted_rand_score(sk_labels, labels))

    return {
        "n_samples": float(n),
        "n_clusters": float(n_clusters),
        "n_noise": float(n_noise),
        "noise_ratio": noise_ratio,
        "ari_vs_sklearn": ari,
    }


def main() -> None:
    seed = 42
    np.random.seed(seed)

    eps = 0.26
    min_samples = 10

    print(f"Backends: sklearn={'yes' if HAS_SKLEARN else 'no'}, scipy={'yes' if HAS_SCIPY else 'no'}")
    print(f"Params: eps={eps}, min_samples={min_samples}")

    X = generate_dataset(seed=seed)

    t0 = time.perf_counter()
    result = dbscan_mvp(X, eps=eps, min_samples=min_samples)
    t1 = time.perf_counter()

    sk_labels = None
    if HAS_SKLEARN:
        sk_labels = SkDBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(X)

    metrics = evaluate_result(result.labels, sk_labels)
    metrics["runtime_sec"] = float(t1 - t0)

    summary = pd.DataFrame(
        [
            {
                "method": "DBSCAN_MVP",
                "n_samples": int(metrics["n_samples"]),
                "n_clusters": int(metrics["n_clusters"]),
                "n_noise": int(metrics["n_noise"]),
                "noise_ratio": round(metrics["noise_ratio"], 4),
                "runtime_sec": round(metrics["runtime_sec"], 4),
                "ari_vs_sklearn": (
                    round(metrics["ari_vs_sklearn"], 4)
                    if not np.isnan(metrics["ari_vs_sklearn"])
                    else np.nan
                ),
            }
        ]
    )

    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    cluster_sizes = (
        pd.Series(result.labels)
        .value_counts()
        .sort_index()
        .rename_axis("label")
        .reset_index(name="count")
    )
    print("\n=== Cluster Size Table (label=-1 is noise) ===")
    print(cluster_sizes.to_string(index=False))

    preview = pd.DataFrame(
        {
            "x0": np.round(X[:, 0], 4),
            "x1": np.round(X[:, 1], 4),
            "label": result.labels.astype(int),
            "is_core": result.core_mask.astype(int),
            "neighbors@eps": result.neighbor_counts.astype(int),
        }
    ).head(12)

    print("\n=== Sample Preview (first 12 rows) ===")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
