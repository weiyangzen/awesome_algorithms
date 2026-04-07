"""Minimal runnable MVP for numerical differentiation by finite differences."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


def validate_uniform_grid(x: np.ndarray, y: np.ndarray, tol: float = 1e-12) -> float:
    """Validate x/y arrays and return uniform spacing h."""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.size != y.size:
        raise ValueError(f"x and y must have same length, got {x.size} and {y.size}")
    if x.size < 3:
        raise ValueError("at least 3 sample points are required")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite numbers")

    dx = np.diff(x)
    if not np.all(dx > 0.0):
        raise ValueError("x must be strictly increasing")

    h = float(dx[0])
    if h <= 0.0:
        raise ValueError(f"grid spacing must be positive, got {h}")
    if np.max(np.abs(dx - h)) > tol:
        raise ValueError("x must be uniformly spaced for this MVP")
    return h


def forward_difference(y: np.ndarray, h: float) -> np.ndarray:
    """Compute first derivative using forward difference on x[:-1]."""
    return (y[1:] - y[:-1]) / h


def backward_difference(y: np.ndarray, h: float) -> np.ndarray:
    """Compute first derivative using backward difference on x[1:]."""
    return (y[1:] - y[:-1]) / h


def central_difference(y: np.ndarray, h: float) -> np.ndarray:
    """Compute first derivative using central difference on x[1:-1]."""
    return (y[2:] - y[:-2]) / (2.0 * h)


def max_abs_error(approx: np.ndarray, exact: np.ndarray) -> float:
    """Maximum absolute error between two vectors."""
    if approx.shape != exact.shape:
        raise ValueError(f"shape mismatch: {approx.shape} vs {exact.shape}")
    return float(np.max(np.abs(approx - exact)))


def run_single_resolution(
    n_intervals: int,
    x_start: float = 0.0,
    x_end: float = 2.0 * math.pi,
) -> Tuple[float, int, float, float, float]:
    """Run one finite-difference experiment and return errors for three schemes."""
    if n_intervals < 2:
        raise ValueError(f"n_intervals must be >= 2, got {n_intervals}")

    x = np.linspace(x_start, x_end, n_intervals + 1, dtype=float)
    y = np.sin(x)
    exact = np.cos(x)

    h = validate_uniform_grid(x, y)

    df_forward = forward_difference(y, h)
    df_backward = backward_difference(y, h)
    df_central = central_difference(y, h)

    err_forward = max_abs_error(df_forward, exact[:-1])
    err_backward = max_abs_error(df_backward, exact[1:])
    err_central = max_abs_error(df_central, exact[1:-1])

    return h, n_intervals, err_forward, err_backward, err_central


def estimate_orders(
    results: Sequence[Tuple[float, int, float, float, float]],
    method_index: int,
) -> List[Tuple[float, float]]:
    """Estimate empirical order p from consecutive resolutions.

    method_index: 2->forward, 3->backward, 4->central.
    """
    orders: List[Tuple[float, float]] = []
    for i in range(len(results) - 1):
        h1 = results[i][0]
        h2 = results[i + 1][0]
        e1 = results[i][method_index]
        e2 = results[i + 1][method_index]
        if e1 <= 0.0 or e2 <= 0.0:
            continue
        p = math.log(e1 / e2) / math.log(h1 / h2)
        orders.append((h2, p))
    return orders


def print_sample_rows(n_intervals: int = 20, rows: int = 8) -> None:
    """Print a few rows of derivative approximations at one resolution."""
    x = np.linspace(0.0, 2.0 * math.pi, n_intervals + 1, dtype=float)
    y = np.sin(x)
    exact = np.cos(x)
    h = validate_uniform_grid(x, y)

    df_forward = forward_difference(y, h)
    df_backward = backward_difference(y, h)
    df_central = central_difference(y, h)

    print("-" * 96)
    print(f"Sample rows (n_intervals={n_intervals}, h={h:.6f})")
    print(" i      x_i        exact'       forward(i)      backward(i)      central(i)")
    print("-" * 96)

    show = min(rows, n_intervals - 1)
    for i in range(1, show + 1):
        xi = x[i]
        exact_i = exact[i]
        fwd_i = df_forward[i]
        bwd_i = df_backward[i - 1]
        ctr_i = df_central[i - 1]
        print(
            f"{i:2d}  {xi:9.5f}  {exact_i:11.7f}  "
            f"{fwd_i:13.7f}  {bwd_i:13.7f}  {ctr_i:11.7f}"
        )


def main() -> None:
    print("Finite difference demo for f(x)=sin(x), f'(x)=cos(x), x in [0, 2*pi]")

    n_intervals_list = [20, 40, 80, 160]
    results: List[Tuple[float, int, float, float, float]] = []

    print("=" * 96)
    print("Convergence table")
    print(" n_intervals      h            err_forward      err_backward      err_central")
    print("=" * 96)
    for n_intervals in n_intervals_list:
        row = run_single_resolution(n_intervals=n_intervals)
        results.append(row)
        h, n, ef, eb, ec = row
        print(f"{n:8d}      {h:10.6f}    {ef:12.6e}    {eb:12.6e}    {ec:12.6e}")

    print("=" * 96)
    print("Empirical orders p (from consecutive resolutions)")
    for label, idx in (("forward", 2), ("backward", 3), ("central", 4)):
        print(f"{label}:")
        for h, p in estimate_orders(results, method_index=idx):
            print(f"  h={h:10.6f} -> p={p:.4f}")

    print_sample_rows(n_intervals=20, rows=8)


if __name__ == "__main__":
    main()
