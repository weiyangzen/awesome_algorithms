"""Minimal runnable MVP for Aitken's delta-squared acceleration."""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np


def fixed_point_sequence(
    g: Callable[[float], float],
    x0: float,
    steps: int,
) -> np.ndarray:
    """Generate x_0 ... x_steps from x_{k+1} = g(x_k)."""
    if steps < 0:
        raise ValueError("steps must be non-negative")

    seq = np.empty(steps + 1, dtype=float)
    seq[0] = x0
    for k in range(steps):
        seq[k + 1] = g(float(seq[k]))
    return seq


def aitken_delta_squared(
    sequence: np.ndarray,
    eps: float = 1e-14,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Aitken delta-squared to a scalar sequence.

    Returns:
    - accelerated sequence x_hat_k for k = 0..n-3
    - valid mask: False where denominator is too small
    """
    if sequence.ndim != 1:
        raise ValueError("sequence must be 1D")
    if sequence.size < 3:
        return np.empty(0, dtype=float), np.empty(0, dtype=bool)
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    n = sequence.size
    accelerated = np.full(n - 2, np.nan, dtype=float)
    valid = np.zeros(n - 2, dtype=bool)

    for k in range(n - 2):
        xk = float(sequence[k])
        xk1 = float(sequence[k + 1])
        xk2 = float(sequence[k + 2])

        delta = xk1 - xk
        delta2 = xk2 - 2.0 * xk1 + xk

        if abs(delta2) <= eps:
            continue

        accelerated[k] = xk - (delta * delta) / delta2
        valid[k] = True

    return accelerated, valid


def high_precision_fixed_point(
    g: Callable[[float], float],
    x0: float,
    max_iter: int = 10000,
    tol: float = 1e-16,
) -> float:
    """Compute a high-accuracy fixed-point reference by plain iteration."""
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")

    x = float(x0)
    for _ in range(max_iter):
        x_next = float(g(x))
        if abs(x_next - x) <= tol * max(1.0, abs(x_next)):
            return x_next
        x = x_next
    return x


def first_index_below(errors: np.ndarray, tol: float) -> Optional[int]:
    """Return first index i with errors[i] <= tol, else None."""
    for i, value in enumerate(errors):
        if float(value) <= tol:
            return i
    return None


def run_geometric_validation() -> None:
    """Validate Aitken on x_k = x* + c*q^k, where it should be exact in ideal arithmetic."""
    limit = 2.0
    c = -1.5
    q = 0.6
    n = np.arange(0, 10, dtype=float)
    seq = limit + c * (q**n)

    acc, valid = aitken_delta_squared(seq)
    first_valid = int(np.argmax(valid)) if np.any(valid) else None

    print("=" * 88)
    print("Geometric-sequence validation")
    print("=" * 88)
    print(f"true_limit={limit:.16f}, q={q}, c={c}")
    print("first 5 raw terms:", ", ".join(f"{v:.10f}" for v in seq[:5]))

    if first_valid is None:
        print("No valid Aitken point (all denominators too small).")
        return

    estimate = float(acc[first_valid])
    abs_err = abs(estimate - limit)
    print(
        f"first valid accelerated index={first_valid}, "
        f"estimate={estimate:.16f}, abs_error={abs_err:.3e}"
    )


def run_cos_fixed_point_demo() -> None:
    """Compare plain fixed-point sequence and Aitken-accelerated sequence for x = cos(x)."""

    def g(x: float) -> float:
        return math.cos(x)

    steps = 35
    x0 = 1.0
    tol = 1e-5

    raw_seq = fixed_point_sequence(g=g, x0=x0, steps=steps)
    x_star = high_precision_fixed_point(g=g, x0=x0, max_iter=20000, tol=1e-17)
    raw_errors = np.abs(raw_seq - x_star)

    acc_seq, valid = aitken_delta_squared(raw_seq, eps=1e-15)
    acc_errors = np.abs(acc_seq - x_star)

    print("=" * 88)
    print("Fixed-point demo: x_{k+1} = cos(x_k)")
    print("=" * 88)
    print(f"reference fixed point x* = {x_star:.16f}")
    print("k   x_k                 |x_k-x*|         x_hat_k             |x_hat_k-x*|      valid")
    max_rows = min(12, acc_seq.size)
    for k in range(max_rows):
        raw_value = raw_seq[k]
        raw_err = raw_errors[k]
        if valid[k]:
            acc_value = acc_seq[k]
            acc_err = acc_errors[k]
            acc_text = f"{acc_value: .16f}"
            err_text = f"{acc_err: .3e}"
            valid_text = "Y"
        else:
            acc_text = " " * 18 + "nan"
            err_text = " " * 8 + "nan"
            valid_text = "N"
        print(
            f"{k:2d}  {raw_value: .16f}  {raw_err: .3e}  "
            f"{acc_text}  {err_text}     {valid_text}"
        )

    raw_hit = first_index_below(raw_errors, tol=tol)
    valid_acc_errors = acc_errors[valid]
    acc_hit_valid_index = first_index_below(valid_acc_errors, tol=tol)

    if raw_hit is None:
        print(f"plain sequence did not reach tol={tol:.1e} within {steps} steps")
    else:
        print(f"plain sequence first reaches tol={tol:.1e} at raw index k={raw_hit}")

    if acc_hit_valid_index is None:
        print(f"Aitken sequence did not reach tol={tol:.1e} on valid accelerated points")
    else:
        valid_positions = np.flatnonzero(valid)
        original_k = int(valid_positions[acc_hit_valid_index])
        print(
            "Aitken sequence first reaches "
            f"tol={tol:.1e} at accelerated index k={original_k}"
        )

    print(
        f"valid accelerated points: {int(np.sum(valid))}/{valid.size}, "
        f"invalid due to small denominator: {int(np.sum(~valid))}"
    )


def main() -> None:
    run_geometric_validation()
    run_cos_fixed_point_demo()


if __name__ == "__main__":
    main()
