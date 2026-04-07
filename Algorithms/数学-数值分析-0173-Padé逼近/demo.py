"""Padé approximation minimal runnable MVP.

This demo builds a [m/n] Padé approximant from a power-series expansion,
then compares it against an equal-coefficient-budget Taylor polynomial.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional pretty-print dependency
    pd = None


def exp_taylor_coeffs(order: int) -> np.ndarray:
    """Return coefficients c_k of exp(x) = sum c_k x^k up to given order."""
    if order < 0:
        raise ValueError("order must be non-negative")
    return np.array([1.0 / math.factorial(k) for k in range(order + 1)], dtype=float)


def pade_from_series(coeffs: np.ndarray, m: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Construct Padé [m/n] from power-series coefficients around x=0.

    Args:
        coeffs: c_0..c_K where K >= m+n
        m: degree of numerator
        n: degree of denominator

    Returns:
        (p, q) where p are numerator coefficients p_0..p_m (ascending)
        and q are denominator coefficients q_0..q_n (ascending, q_0=1).
    """
    c = np.asarray(coeffs, dtype=float)

    if m < 0 or n < 0:
        raise ValueError("m and n must be non-negative")
    if c.size < (m + n + 1):
        raise ValueError(f"need at least {m+n+1} coefficients, got {c.size}")

    if n == 0:
        q = np.array([1.0], dtype=float)
    else:
        a = np.empty((n, n), dtype=float)
        b = np.empty(n, dtype=float)

        # For k = m+1 .. m+n:
        # c_k + sum_{j=1..n} q_j c_{k-j} = 0
        # => sum_{j=1..n} q_j c_{k-j} = -c_k
        for r in range(1, n + 1):
            k = m + r
            b[r - 1] = -c[k]
            for j in range(1, n + 1):
                a[r - 1, j - 1] = c[k - j]

        try:
            q_tail = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            q_tail = np.linalg.lstsq(a, b, rcond=None)[0]

        q = np.concatenate(([1.0], q_tail))

    p = np.zeros(m + 1, dtype=float)
    for k in range(m + 1):
        upper = min(k, n)
        p[k] = sum(q[j] * c[k - j] for j in range(upper + 1))

    return p, q


def polyval_ascending(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate polynomial with ascending coefficients via Horner."""
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr, dtype=float)
    for ck in reversed(np.asarray(coeffs, dtype=float)):
        y = y * x_arr + ck
    return y


def eval_rational(p: np.ndarray, q: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate rational function P(x)/Q(x), returning value and denominator."""
    num = polyval_ascending(p, x)
    den = polyval_ascending(q, x)
    return num / den, den


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square error."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
    """Maximum absolute error."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def format_table(rows: List[Dict[str, float]]) -> str:
    """Pretty print rows with pandas if available, else plain text."""
    if pd is not None:
        return pd.DataFrame(rows).to_string(index=False)

    if not rows:
        return "(empty)"

    headers = list(rows[0].keys())
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    split_line = "-+-".join("-" * widths[h] for h in headers)
    body_lines = [" | ".join(str(r[h]).ljust(widths[h]) for h in headers) for r in rows]
    return "\n".join([header_line, split_line, *body_lines])


def main() -> None:
    m = 4
    n = 4
    order = m + n

    coeffs = exp_taylor_coeffs(order)
    p, q = pade_from_series(coeffs, m=m, n=n)

    # Verify matching of first (m+n+1) series coefficients in Q*f - P.
    lhs = np.convolve(q, coeffs)[: order + 1]
    p_padded = np.pad(p, (0, n), mode="constant")
    coeff_match_residual = float(np.max(np.abs(lhs - p_padded)))

    x_grid = np.linspace(-4.0, 4.0, 801)
    exact = np.exp(x_grid)

    pade_vals, den_vals = eval_rational(p, q, x_grid)
    taylor_vals = polyval_ascending(coeffs, x_grid)

    metrics = {
        "pade_rmse": rmse(pade_vals, exact),
        "taylor_rmse": rmse(taylor_vals, exact),
        "pade_max_abs_error": max_abs_err(pade_vals, exact),
        "taylor_max_abs_error": max_abs_err(taylor_vals, exact),
        "coeff_match_residual": coeff_match_residual,
        "min_abs_denominator_on_grid": float(np.min(np.abs(den_vals))),
    }
    metrics["rmse_improvement_ratio"] = metrics["taylor_rmse"] / metrics["pade_rmse"]

    sample_points = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], dtype=float)
    sample_exact = np.exp(sample_points)
    sample_pade, sample_den = eval_rational(p, q, sample_points)
    sample_taylor = polyval_ascending(coeffs, sample_points)

    sample_rows: List[Dict[str, float]] = []
    for i, x in enumerate(sample_points):
        sample_rows.append(
            {
                "x": float(x),
                "exp(x)": round(float(sample_exact[i]), 8),
                "pade": round(float(sample_pade[i]), 8),
                "taylor8": round(float(sample_taylor[i]), 8),
                "|Q(x)|": round(float(abs(sample_den[i])), 8),
            }
        )

    print("Padé Approximation MVP ([4/4] for exp(x))")
    print()

    print("Numerator coefficients p (ascending):")
    print(np.array2string(p, precision=10, separator=", "))
    print("Denominator coefficients q (ascending, q0=1):")
    print(np.array2string(q, precision=10, separator=", "))
    print()

    print("Metrics on grid x in [-4, 4]:")
    for key in [
        "pade_rmse",
        "taylor_rmse",
        "rmse_improvement_ratio",
        "pade_max_abs_error",
        "taylor_max_abs_error",
        "coeff_match_residual",
        "min_abs_denominator_on_grid",
    ]:
        print(f"- {key}: {metrics[key]:.10e}")
    print()

    print("Sample-point comparison:")
    print(format_table(sample_rows))


if __name__ == "__main__":
    main()
