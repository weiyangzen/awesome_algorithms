"""Random Search: minimal runnable MVP.

This script demonstrates pure random search for box-constrained minimization
with deterministic seeds and batch vectorized objective evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np


Array = np.ndarray
BatchObjective = Callable[[Array], Array]


@dataclass
class SearchResult:
    best_x: Array
    best_value: float
    samples_evaluated: int
    history: List[Tuple[int, float]]


def validate_bounds(bounds: Array) -> Array:
    """Validate bounds and return float ndarray with shape (d, 2)."""
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("bounds must have shape (d, 2)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("bounds must be finite")
    low = arr[:, 0]
    high = arr[:, 1]
    if np.any(low >= high):
        raise ValueError("each bound must satisfy low < high")
    return arr


def make_checkpoints(n_samples: int, log_points: int) -> List[int]:
    """Create sorted unique checkpoints in [1, n_samples]."""
    points = max(1, int(log_points))
    raw = np.linspace(1, n_samples, num=points)
    checkpoints = sorted({int(round(v)) for v in raw})
    checkpoints[-1] = n_samples
    return checkpoints


def random_search(
    objective: BatchObjective,
    bounds: Array,
    n_samples: int,
    seed: int,
    batch_size: int = 1024,
    log_points: int = 10,
) -> SearchResult:
    """Run pure random search on a box domain."""
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    b = validate_bounds(bounds)
    low = b[:, 0]
    high = b[:, 1]
    dim = b.shape[0]

    rng = np.random.default_rng(seed)
    checkpoints = make_checkpoints(n_samples=n_samples, log_points=log_points)
    checkpoint_idx = 0

    best_value = float("inf")
    best_x = np.zeros(dim, dtype=float)
    history: List[Tuple[int, float]] = []

    evaluated = 0
    while evaluated < n_samples:
        k = min(batch_size, n_samples - evaluated)
        unit = rng.random((k, dim), dtype=float)
        samples = low + unit * (high - low)

        values = np.asarray(objective(samples), dtype=float)
        if values.shape != (k,):
            raise RuntimeError(
                f"objective output must have shape ({k},), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise RuntimeError("objective output contains non-finite values")

        batch_best_idx = int(np.argmin(values))
        batch_best_value = float(values[batch_best_idx])
        if batch_best_value < best_value:
            best_value = batch_best_value
            best_x = samples[batch_best_idx].copy()

        evaluated += k
        while checkpoint_idx < len(checkpoints) and evaluated >= checkpoints[checkpoint_idx]:
            history.append((checkpoints[checkpoint_idx], best_value))
            checkpoint_idx += 1

    return SearchResult(
        best_x=best_x,
        best_value=best_value,
        samples_evaluated=evaluated,
        history=history,
    )


def sphere_batch(x: Array) -> Array:
    """Sphere function, global minimum at x=0 with f*=0."""
    return np.sum(x * x, axis=1)


def rastrigin_batch(x: Array) -> Array:
    """Rastrigin function, global minimum at x=0 with f*=0."""
    a = 10.0
    d = x.shape[1]
    return a * d + np.sum(x * x - a * np.cos(2.0 * np.pi * x), axis=1)


def shifted_quadratic_batch(x: Array, center: Array) -> Array:
    """Shifted convex quadratic, global minimum at center with f*=0."""
    diff = x - center.reshape(1, -1)
    return np.sum(diff * diff, axis=1)


def format_vector(v: Array, precision: int = 4) -> str:
    return np.array2string(v, precision=precision, floatmode="fixed")


def run_case(
    name: str,
    objective: BatchObjective,
    bounds: Sequence[Tuple[float, float]],
    n_samples: int,
    seed: int,
    known_optimum: float = 0.0,
    batch_size: int = 2048,
) -> Tuple[str, float]:
    result = random_search(
        objective=objective,
        bounds=np.asarray(bounds, dtype=float),
        n_samples=n_samples,
        seed=seed,
        batch_size=batch_size,
        log_points=8,
    )

    gap = result.best_value - known_optimum

    print(f"\n=== {name} ===")
    print(f"samples_evaluated: {result.samples_evaluated}")
    print(f"best_value: {result.best_value:.8f}")
    print(f"known_optimum: {known_optimum:.8f}")
    print(f"optimality_gap: {gap:.8f}")
    print(f"best_x: {format_vector(result.best_x)}")
    print("history checkpoints (samples, best_value):")
    for samples_seen, best_val in result.history:
        print(f"  - ({samples_seen:6d}, {best_val:.8f})")

    return name, gap


def main() -> None:
    center = np.array([1.5, -2.0, 0.5, 3.0, -1.0], dtype=float)

    cases = [
        {
            "name": "Sphere-2D",
            "objective": sphere_batch,
            "bounds": [(-5.0, 5.0), (-5.0, 5.0)],
            "n_samples": 25_000,
            "seed": 20260361,
            "known_optimum": 0.0,
        },
        {
            "name": "Rastrigin-2D",
            "objective": rastrigin_batch,
            "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
            "n_samples": 40_000,
            "seed": 20260362,
            "known_optimum": 0.0,
        },
        {
            "name": "ShiftedQuadratic-5D",
            "objective": lambda x: shifted_quadratic_batch(x, center=center),
            "bounds": [(-6.0, 6.0)] * 5,
            "n_samples": 60_000,
            "seed": 20260363,
            "known_optimum": 0.0,
        },
    ]

    summary: List[Tuple[str, float]] = []
    for cfg in cases:
        summary.append(
            run_case(
                name=cfg["name"],
                objective=cfg["objective"],
                bounds=cfg["bounds"],
                n_samples=cfg["n_samples"],
                seed=cfg["seed"],
                known_optimum=cfg["known_optimum"],
            )
        )

    print("\n=== Summary ===")
    gaps = np.array([g for _, g in summary], dtype=float)
    for name, gap in summary:
        print(f"{name:24s} gap={gap:.8f}")
    print(f"mean_gap={float(np.mean(gaps)):.8f}")
    print(f"max_gap={float(np.max(gaps)):.8f}")


if __name__ == "__main__":
    main()
