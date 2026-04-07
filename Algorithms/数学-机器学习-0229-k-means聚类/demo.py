"""Minimal runnable MVP for k-means clustering (MATH-0229).

This script implements Lloyd's algorithm from scratch with NumPy,
using k-means++ initialization and deterministic synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import List, Tuple

import numpy as np


@dataclass
class KMeansResult:
    """Container for final clustering artifacts."""

    centers: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int
    inertia_history: List[float]


def make_synthetic_blobs(
    n_per_cluster: int = 140,
    seed: int = 229,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a 2D dataset with three well-separated Gaussian clusters."""
    rng = np.random.default_rng(seed)
    true_centers = np.array(
        [
            [-4.0, -2.5],
            [0.5, 4.2],
            [4.6, -1.1],
        ],
        dtype=np.float64,
    )
    stds = np.array([0.60, 0.75, 0.65], dtype=np.float64)

    xs = []
    ys = []
    for cid, (center, std) in enumerate(zip(true_centers, stds)):
        block = rng.normal(loc=center, scale=std, size=(n_per_cluster, 2))
        xs.append(block)
        ys.append(np.full(n_per_cluster, cid, dtype=np.int64))

    x = np.vstack(xs)
    y = np.concatenate(ys)

    # Deterministic shuffle to avoid ordered blocks.
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    return x[idx], y[idx], true_centers


def pairwise_sq_dists(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return squared Euclidean distance matrix with shape (n_samples, k)."""
    diff = x[:, None, :] - centers[None, :, :]
    return np.einsum("nkd,nkd->nk", diff, diff)


def kmeans_plus_plus_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Initialize centers by k-means++."""
    n, d = x.shape
    centers = np.empty((k, d), dtype=np.float64)

    first = int(rng.integers(0, n))
    centers[0] = x[first]

    closest_sq = pairwise_sq_dists(x, centers[:1]).reshape(-1)
    for i in range(1, k):
        total = float(np.sum(closest_sq))
        if total <= 1e-15:
            # Degenerate case: all points coincide with existing centers.
            centers[i] = x[int(rng.integers(0, n))]
            continue
        probs = closest_sq / total
        idx = int(rng.choice(n, p=probs))
        centers[i] = x[idx]
        new_sq = pairwise_sq_dists(x, centers[i : i + 1]).reshape(-1)
        closest_sq = np.minimum(closest_sq, new_sq)

    return centers


def cluster_purity(y_true: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Compute cluster purity in [0, 1]."""
    total = labels.size
    correct = 0
    for cid in range(k):
        mask = labels == cid
        if not np.any(mask):
            continue
        counts = np.bincount(y_true[mask], minlength=k)
        correct += int(np.max(counts))
    return correct / total


def center_rmse_with_best_permutation(
    pred_centers: np.ndarray, true_centers: np.ndarray
) -> float:
    """Match center order by brute force permutation (small k) and return RMSE."""
    k = pred_centers.shape[0]
    best = float("inf")
    for perm in permutations(range(k)):
        mapped = true_centers[np.array(perm)]
        rmse = float(np.sqrt(np.mean((pred_centers - mapped) ** 2)))
        best = min(best, rmse)
    return best


def kmeans_fit(
    x: np.ndarray,
    k: int,
    max_iters: int = 100,
    tol: float = 1e-4,
    seed: int = 229,
) -> KMeansResult:
    """Run Lloyd's algorithm with k-means++ initialization."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if not (1 <= k <= x.shape[0]):
        raise ValueError("k must satisfy 1 <= k <= n_samples")

    rng = np.random.default_rng(seed)
    centers = kmeans_plus_plus_init(x, k=k, rng=rng)
    inertia_history: List[float] = []

    labels = np.zeros(x.shape[0], dtype=np.int64)
    n_iter_done = 0
    for it in range(1, max_iters + 1):
        dist_sq = pairwise_sq_dists(x, centers)
        labels = np.argmin(dist_sq, axis=1)
        min_sq = dist_sq[np.arange(x.shape[0]), labels]
        inertia = float(np.sum(min_sq))
        inertia_history.append(inertia)

        new_centers = np.empty_like(centers)
        for cid in range(k):
            members = x[labels == cid]
            if members.shape[0] == 0:
                # Re-seed empty cluster by currently worst-fitted sample.
                farthest = int(np.argmax(min_sq))
                new_centers[cid] = x[farthest]
            else:
                new_centers[cid] = np.mean(members, axis=0)

        shift = float(np.linalg.norm(new_centers - centers))
        centers = new_centers
        n_iter_done = it
        if shift <= tol:
            break

    final_dist_sq = pairwise_sq_dists(x, centers)
    final_labels = np.argmin(final_dist_sq, axis=1)
    final_min_sq = final_dist_sq[np.arange(x.shape[0]), final_labels]
    final_inertia = float(np.sum(final_min_sq))

    return KMeansResult(
        centers=centers,
        labels=final_labels,
        inertia=final_inertia,
        n_iter=n_iter_done,
        inertia_history=inertia_history,
    )


def main() -> None:
    print("k-means Clustering MVP (MATH-0229)")
    print("=" * 64)

    x, y, true_centers = make_synthetic_blobs(n_per_cluster=140, seed=229)
    result = kmeans_fit(x, k=3, max_iters=80, tol=1e-5, seed=229)

    purity = cluster_purity(y, result.labels, k=3)
    center_rmse = center_rmse_with_best_permutation(result.centers, true_centers)

    print(f"samples: {x.shape[0]}, dim: {x.shape[1]}, k: 3")
    print(f"iterations: {result.n_iter}")
    print(f"initial inertia: {result.inertia_history[0]:.4f}")
    print(f"final inertia: {result.inertia:.4f}")
    print(f"cluster purity: {purity:.4f}")
    print(f"best-permutation center RMSE: {center_rmse:.4f}")
    print("centers:")
    print(np.array2string(result.centers, precision=4))

    # Lloyd iterations should be non-increasing in inertia (up to tiny float noise).
    history = np.array(result.inertia_history, dtype=np.float64)
    if history.size >= 2:
        max_increase = float(np.max(np.diff(history)))
        if max_increase > 1e-8:
            raise RuntimeError(
                f"inertia increased unexpectedly during iterations: {max_increase:.3e}"
            )

    if purity < 0.95:
        raise RuntimeError(f"purity too low: {purity:.4f}")
    if center_rmse > 0.7:
        raise RuntimeError(f"cluster centers are too far from expected: {center_rmse:.4f}")

    print("=" * 64)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
