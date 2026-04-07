"""Runnable Mean Shift clustering MVP for MATH-0234 (pure NumPy implementation)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_data(random_state: int = 42) -> np.ndarray:
    """Create a reproducible 2D synthetic dataset with uneven cluster spread."""
    rng = np.random.default_rng(random_state)
    centers = np.array([[-5.0, -1.0], [-1.0, 4.0], [2.5, 1.5], [6.0, -2.0]], dtype=float)
    stds = np.array([0.7, 0.9, 0.6, 1.0], dtype=float)
    counts = [120, 120, 120, 120]

    chunks = []
    for center, std, count in zip(centers, stds, counts):
        chunk = rng.normal(loc=center, scale=std, size=(count, 2))
        chunks.append(chunk)
    return np.vstack(chunks)


def pairwise_distances(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Compute Euclidean pairwise distances using broadcasting."""
    if y is None:
        y = x
    diff = x[:, None, :] - y[None, :, :]
    return np.linalg.norm(diff, axis=2)


def estimate_bandwidth(
    x: np.ndarray,
    quantile: float = 0.2,
    sample_size: int = 400,
    random_state: int = 42,
) -> float:
    """Estimate a positive bandwidth from sampled pairwise distances."""
    rng = np.random.default_rng(random_state)
    size = min(sample_size, len(x))
    idx = rng.choice(len(x), size=size, replace=False)
    sampled = x[idx]

    dist = pairwise_distances(sampled)
    upper = dist[np.triu_indices_from(dist, k=1)]
    if upper.size == 0:
        return 1.0

    bw = float(np.quantile(upper, quantile))
    if not np.isfinite(bw) or bw <= 0:
        return 1.0
    return bw


def generate_seeds(x: np.ndarray, bandwidth: float, min_bin_freq: int = 5) -> np.ndarray:
    """Generate seed points from a coarse grid similar to bin seeding."""
    if bandwidth <= 0:
        return x.copy()

    bins = np.floor(x / bandwidth).astype(int)
    unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
    selected = unique_bins[counts >= min_bin_freq]

    if len(selected) == 0:
        return x.copy()

    seeds = (selected.astype(float) + 0.5) * bandwidth
    return seeds


def mean_shift_single_seed(
    x: np.ndarray,
    seed: np.ndarray,
    bandwidth: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, int]:
    """Iteratively move one seed to the local density mode."""
    center = seed.astype(float)
    support = 0

    for _ in range(max_iter):
        dist = np.linalg.norm(x - center, axis=1)
        mask = dist <= bandwidth
        neighbors = x[mask]
        if len(neighbors) == 0:
            break

        support = int(mask.sum())
        new_center = neighbors.mean(axis=0)
        shift = np.linalg.norm(new_center - center)
        center = new_center
        if shift < tol:
            break

    return center, support


def merge_modes(modes: np.ndarray, supports: np.ndarray, bandwidth: float) -> np.ndarray:
    """Greedily merge nearby modes; keep stronger modes first."""
    order = np.argsort(-supports)
    kept: list[np.ndarray] = []
    merge_radius = max(1e-8, 0.5 * bandwidth)

    for idx in order:
        mode = modes[idx]
        if not kept:
            kept.append(mode)
            continue

        dist = np.linalg.norm(np.vstack(kept) - mode, axis=1)
        if np.min(dist) > merge_radius:
            kept.append(mode)

    return np.vstack(kept)


def mean_shift(
    x: np.ndarray,
    bandwidth: float,
    max_iter: int = 300,
    tol: float = 1e-3,
    min_bin_freq: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Mean Shift and return labels with cluster centers."""
    seeds = generate_seeds(x, bandwidth=bandwidth, min_bin_freq=min_bin_freq)

    modes = []
    supports = []
    for seed in seeds:
        mode, support = mean_shift_single_seed(x, seed, bandwidth, max_iter=max_iter, tol=tol)
        modes.append(mode)
        supports.append(support)

    modes_arr = np.vstack(modes)
    supports_arr = np.asarray(supports)
    centers = merge_modes(modes_arr, supports_arr, bandwidth=bandwidth)

    dist_to_centers = pairwise_distances(x, centers)
    labels = np.argmin(dist_to_centers, axis=1)
    return labels, centers


def silhouette_score_numpy(x: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score from scratch for small/medium datasets."""
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")

    dist = pairwise_distances(x)
    scores = np.zeros(len(x), dtype=float)

    for i in range(len(x)):
        same = labels == labels[i]
        same[i] = False

        a = dist[i, same].mean() if np.any(same) else 0.0

        b = np.inf
        for c in unique:
            if c == labels[i]:
                continue
            other = labels == c
            if np.any(other):
                b = min(b, dist[i, other].mean())

        denom = max(a, b)
        scores[i] = 0.0 if denom <= 1e-12 else (b - a) / denom

    return float(scores.mean())


def summarize_clusters(x: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Return per-cluster size and coordinate means."""
    df = pd.DataFrame(x, columns=["x1", "x2"])
    df["cluster"] = labels
    summary = (
        df.groupby("cluster", as_index=False)
        .agg(size=("cluster", "size"), x1_mean=("x1", "mean"), x2_mean=("x2", "mean"))
        .sort_values("cluster")
    )
    return summary


def main() -> None:
    x = generate_data(random_state=42)
    bandwidth = estimate_bandwidth(x, quantile=0.2, sample_size=400, random_state=42)
    labels, centers = mean_shift(x, bandwidth=bandwidth, max_iter=300, tol=1e-3, min_bin_freq=5)

    n_clusters = int(np.unique(labels).size)
    print(f"Estimated bandwidth: {bandwidth:.4f}")
    print(f"Detected clusters: {n_clusters}")
    print(f"Cluster centers shape: {centers.shape}")

    if n_clusters > 1:
        sil = silhouette_score_numpy(x, labels)
        print(f"Silhouette score: {sil:.4f}")
    else:
        print("Silhouette score: N/A (requires at least 2 clusters)")

    summary = summarize_clusters(x, labels)
    print("\nCluster summary:")
    print(summary.to_string(index=False, justify="center", float_format=lambda v: f"{v: .3f}"))

    preview = pd.DataFrame(x[:10], columns=["x1", "x2"])
    preview["label"] = labels[:10]
    print("\nFirst 10 samples with labels:")
    print(preview.to_string(index=False, justify="center", float_format=lambda v: f"{v: .3f}"))


if __name__ == "__main__":
    main()
