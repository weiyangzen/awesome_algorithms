"""k-means++ minimal runnable MVP.

This script implements k-means++ initialization and Lloyd iterations from scratch
with NumPy, then validates on fixed synthetic datasets.
"""

from __future__ import annotations

from itertools import permutations
from typing import Dict, List, Sequence, Tuple

import numpy as np

Array = np.ndarray
HistoryItem = Tuple[int, float, float, int]


def check_data_matrix(x: Array) -> None:
    if x.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape={x.shape}.")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("X must be non-empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("X contains non-finite values.")


def squared_distances_to_centers(x: Array, centers: Array) -> Array:
    # Broadcasting pairwise squared Euclidean distances: (n, d) vs (k, d) -> (n, k)
    diff = x[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)


def assign_labels(x: Array, centers: Array) -> Tuple[Array, Array]:
    d2 = squared_distances_to_centers(x, centers)
    labels = np.argmin(d2, axis=1)
    min_d2 = d2[np.arange(x.shape[0]), labels]
    return labels, min_d2


def kmeans_plus_plus_init(x: Array, k: int, rng: np.random.Generator) -> Tuple[Array, Array]:
    n_samples, n_features = x.shape
    if k <= 0 or k > n_samples:
        raise ValueError(f"k must satisfy 1 <= k <= n_samples, got k={k}, n={n_samples}.")

    init_indices = np.empty(k, dtype=int)
    centers = np.empty((k, n_features), dtype=float)

    first_idx = int(rng.integers(0, n_samples))
    init_indices[0] = first_idx
    centers[0] = x[first_idx]

    # Distance to nearest selected center (D(x)^2 in k-means++).
    closest_d2 = np.sum((x - centers[0]) ** 2, axis=1)

    for c in range(1, k):
        total = float(np.sum(closest_d2))

        if not np.isfinite(total):
            raise ValueError("Encountered non-finite probability mass during k-means++ init.")

        if total <= 1e-18:
            # All points are effectively identical w.r.t current centers.
            # Fall back to random pick to keep algorithm running.
            next_idx = int(rng.integers(0, n_samples))
        else:
            probs = closest_d2 / total
            next_idx = int(rng.choice(n_samples, p=probs))

        init_indices[c] = next_idx
        centers[c] = x[next_idx]

        new_d2 = np.sum((x - centers[c]) ** 2, axis=1)
        closest_d2 = np.minimum(closest_d2, new_d2)

    return centers, init_indices


def recompute_centers(
    x: Array,
    labels: Array,
    k: int,
    rng: np.random.Generator,
) -> Tuple[Array, int]:
    n_samples, n_features = x.shape
    centers = np.zeros((k, n_features), dtype=float)
    empty_count = 0

    for j in range(k):
        mask = labels == j
        if np.any(mask):
            centers[j] = np.mean(x[mask], axis=0)
        else:
            empty_count += 1
            # Re-seed empty cluster with a random sample to avoid degenerate NaN center.
            centers[j] = x[int(rng.integers(0, n_samples))]

    return centers, empty_count


def kmeans_fit(
    x: Array,
    k: int,
    seed: int = 0,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> Dict[str, object]:
    check_data_matrix(x)
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol < 0:
        raise ValueError("tol must be non-negative.")

    rng = np.random.default_rng(seed)
    centers, init_indices = kmeans_plus_plus_init(x, k, rng)

    history: List[HistoryItem] = []

    for it in range(1, max_iter + 1):
        labels, min_d2 = assign_labels(x, centers)
        inertia = float(np.sum(min_d2))

        new_centers, empty_count = recompute_centers(x, labels, k, rng)
        shift = float(np.linalg.norm(new_centers - centers))

        history.append((it, inertia, shift, empty_count))
        centers = new_centers

        if shift <= tol:
            break

    final_labels, final_min_d2 = assign_labels(x, centers)
    final_inertia = float(np.sum(final_min_d2))

    return {
        "centers": centers,
        "labels": final_labels,
        "inertia": final_inertia,
        "n_iter": len(history),
        "history": history,
        "init_indices": init_indices,
    }


def make_blob_data(
    centers: Array,
    counts: Sequence[int],
    std: float,
    seed: int,
) -> Tuple[Array, Array]:
    if centers.ndim != 2:
        raise ValueError("centers must be 2D.")
    if len(counts) != centers.shape[0]:
        raise ValueError("counts length must match number of centers.")
    if std <= 0:
        raise ValueError("std must be positive.")

    rng = np.random.default_rng(seed)
    xs: List[Array] = []
    ys: List[Array] = []

    for i, (c, cnt) in enumerate(zip(centers, counts)):
        if cnt <= 0:
            raise ValueError("Each cluster count must be positive.")
        samples = rng.normal(loc=c, scale=std, size=(cnt, centers.shape[1]))
        xs.append(samples)
        ys.append(np.full(cnt, i, dtype=int))

    x = np.vstack(xs)
    y = np.concatenate(ys)
    return x, y


def best_center_rmse(pred_centers: Array, true_centers: Array) -> float:
    if pred_centers.shape != true_centers.shape:
        raise ValueError("pred_centers and true_centers shape mismatch.")

    k = pred_centers.shape[0]
    best = float("inf")
    for perm in permutations(range(k)):
        aligned = pred_centers[list(perm)]
        rmse = float(np.sqrt(np.mean((aligned - true_centers) ** 2)))
        if rmse < best:
            best = rmse
    return best


def best_label_accuracy(pred: Array, true: Array, k: int) -> float:
    if pred.shape != true.shape:
        raise ValueError("pred and true labels shape mismatch.")

    conf = np.zeros((k, k), dtype=int)
    for p, t in zip(pred, true):
        conf[p, t] += 1

    best = 0
    for perm in permutations(range(k)):
        score = sum(conf[p, perm[p]] for p in range(k))
        if score > best:
            best = score
    return float(best / pred.size)


def print_history(history: Sequence[HistoryItem], max_lines: int = 12) -> None:
    print("iter | inertia          | center_shift      | empty_clusters")
    print("-" * 64)
    for it, inertia, shift, empty_count in history[:max_lines]:
        print(f"{it:4d} | {inertia:16.9e} | {shift:16.9e} | {empty_count:14d}")
    if len(history) > max_lines:
        print(f"... ({len(history) - max_lines} more iterations omitted)")


def run_case(
    name: str,
    true_centers: Array,
    counts: Sequence[int],
    std: float,
    data_seed: int,
    fit_seed: int,
    max_iter: int,
    tol: float,
) -> Dict[str, float]:
    print(f"\n=== Case: {name} ===")
    x, y_true = make_blob_data(true_centers, counts, std=std, seed=data_seed)

    res = kmeans_fit(x, k=true_centers.shape[0], seed=fit_seed, max_iter=max_iter, tol=tol)
    pred_centers = np.asarray(res["centers"])
    y_pred = np.asarray(res["labels"])

    center_rmse = best_center_rmse(pred_centers, true_centers)
    label_acc = best_label_accuracy(y_pred, y_true, k=true_centers.shape[0])

    print(f"data shape: {x.shape}")
    print(f"k-means++ init sample indices: {res['init_indices']}")
    print_history(res["history"])
    print(f"final inertia: {res['inertia']:.9e}")
    print(f"iterations used: {res['n_iter']}")
    print(f"center RMSE (best permutation): {center_rmse:.9e}")
    print(f"label accuracy (best permutation): {label_acc:.6f}")

    return {
        "inertia": float(res["inertia"]),
        "n_iter": float(res["n_iter"]),
        "center_rmse": center_rmse,
        "label_acc": label_acc,
    }


def main() -> None:
    cases = [
        {
            "name": "Well-separated 2D blobs (k=3)",
            "true_centers": np.array([[-5.0, -5.0], [0.0, 5.0], [6.0, -1.0]], dtype=float),
            "counts": [150, 140, 160],
            "std": 0.7,
            "data_seed": 2026,
            "fit_seed": 7,
            "max_iter": 100,
            "tol": 1e-4,
        },
        {
            "name": "Moderately-overlapping 2D blobs (k=4)",
            "true_centers": np.array([
                [-3.0, 2.5],
                [0.0, 0.0],
                [3.0, 3.0],
                [4.5, -2.5],
            ], dtype=float),
            "counts": [120, 150, 140, 130],
            "std": 1.0,
            "data_seed": 2027,
            "fit_seed": 11,
            "max_iter": 120,
            "tol": 1e-4,
        },
    ]

    metrics = []
    for cfg in cases:
        m = run_case(
            name=cfg["name"],
            true_centers=cfg["true_centers"],
            counts=cfg["counts"],
            std=cfg["std"],
            data_seed=cfg["data_seed"],
            fit_seed=cfg["fit_seed"],
            max_iter=cfg["max_iter"],
            tol=cfg["tol"],
        )
        metrics.append(m)

    max_center_rmse = max(m["center_rmse"] for m in metrics)
    min_label_acc = min(m["label_acc"] for m in metrics)
    avg_iter = float(np.mean([m["n_iter"] for m in metrics]))

    # Conservative thresholds for stochastic clustering quality.
    pass_flag = bool(max_center_rmse < 1.0 and min_label_acc > 0.70)

    print("\n=== Summary ===")
    print(f"max center RMSE: {max_center_rmse:.9e}")
    print(f"min label accuracy: {min_label_acc:.6f}")
    print(f"average iterations: {avg_iter:.2f}")
    print(f"all cases pass baseline check: {pass_flag}")


if __name__ == "__main__":
    main()
