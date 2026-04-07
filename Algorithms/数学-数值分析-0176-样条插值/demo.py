"""Minimal runnable MVP for natural cubic spline interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SplineModel:
    """Natural cubic spline model on strictly increasing knots."""

    x: np.ndarray
    y: np.ndarray
    second_derivatives: np.ndarray
    coeffs: np.ndarray  # shape=(n_segments, 4), columns are a,b,c,d


def validate_xy(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize knot arrays."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be 1-D arrays")
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length")
    if x_arr.size < 2:
        raise ValueError("at least two knots are required")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("x and y must contain only finite numbers")

    dx = np.diff(x_arr)
    if not np.all(dx > 0.0):
        raise ValueError("x must be strictly increasing")

    return x_arr, y_arr


def thomas_solve(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal linear system by Thomas algorithm."""
    n = diag.size
    if n == 0:
        return np.array([], dtype=float)
    if lower.size != max(0, n - 1) or upper.size != max(0, n - 1) or rhs.size != n:
        raise ValueError("invalid tridiagonal shapes")

    a = lower.astype(float).copy()
    b = diag.astype(float).copy()
    c = upper.astype(float).copy()
    d = rhs.astype(float).copy()

    for i in range(1, n):
        if abs(b[i - 1]) < 1e-14:
            raise ValueError("tridiagonal system is singular or ill-conditioned")
        w = a[i - 1] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    if abs(b[-1]) < 1e-14:
        raise ValueError("tridiagonal system is singular or ill-conditioned")

    x = np.zeros(n, dtype=float)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def build_natural_cubic_spline(x: Sequence[float], y: Sequence[float]) -> SplineModel:
    """Build a natural cubic spline (M0 = Mn = 0)."""
    x_arr, y_arr = validate_xy(x, y)
    n_segments = x_arr.size - 1
    h = np.diff(x_arr)

    m = np.zeros(x_arr.size, dtype=float)

    # Solve for interior second derivatives M1..M_{n-1}.
    if n_segments >= 2:
        interior_n = n_segments - 1
        diag = 2.0 * (h[:-1] + h[1:])
        rhs = 6.0 * ((y_arr[2:] - y_arr[1:-1]) / h[1:] - (y_arr[1:-1] - y_arr[:-2]) / h[:-1])

        if interior_n == 1:
            interior_m = rhs / diag
        else:
            lower = h[1:-1]
            upper = h[1:-1]
            interior_m = thomas_solve(lower=lower, diag=diag, upper=upper, rhs=rhs)

        m[1:-1] = interior_m

    a = y_arr[:-1]
    b = (y_arr[1:] - y_arr[:-1]) / h - h * (2.0 * m[:-1] + m[1:]) / 6.0
    c = m[:-1] / 2.0
    d = (m[1:] - m[:-1]) / (6.0 * h)
    coeffs = np.column_stack((a, b, c, d))

    return SplineModel(x=x_arr, y=y_arr, second_derivatives=m, coeffs=coeffs)


def evaluate_spline(model: SplineModel, x_query: Sequence[float] | float) -> np.ndarray | float:
    """Evaluate spline values at scalar or vector query points."""
    query_arr = np.asarray(x_query, dtype=float)
    scalar_input = query_arr.ndim == 0
    q = np.atleast_1d(query_arr)

    x_min = model.x[0]
    x_max = model.x[-1]
    if np.any(q < x_min) or np.any(q > x_max):
        raise ValueError(f"query points must be within [{x_min}, {x_max}]")

    idx = np.searchsorted(model.x, q, side="right") - 1
    idx = np.clip(idx, 0, model.coeffs.shape[0] - 1)

    dx = q - model.x[idx]
    seg = model.coeffs[idx]
    values = seg[:, 0] + seg[:, 1] * dx + seg[:, 2] * dx * dx + seg[:, 3] * dx * dx * dx

    if scalar_input:
        return float(values[0])
    return values.reshape(query_arr.shape)


def continuity_report(model: SplineModel) -> tuple[float, float]:
    """Return max jumps of first and second derivatives at interior knots."""
    n_segments = model.coeffs.shape[0]
    if n_segments <= 1:
        return 0.0, 0.0

    h = np.diff(model.x)
    first_jumps = []
    second_jumps = []

    for i in range(n_segments - 1):
        a_l, b_l, c_l, d_l = model.coeffs[i]
        _a_r, b_r, c_r, _d_r = model.coeffs[i + 1]

        dx = h[i]
        d1_left = b_l + 2.0 * c_l * dx + 3.0 * d_l * dx * dx
        d2_left = 2.0 * c_l + 6.0 * d_l * dx

        d1_right = b_r
        d2_right = 2.0 * c_r

        first_jumps.append(abs(d1_left - d1_right))
        second_jumps.append(abs(d2_left - d2_right))

    return float(max(first_jumps)), float(max(second_jumps))


def main() -> None:
    # Sample smooth function with known values.
    x_knots = np.linspace(0.0, 2.0 * np.pi, 9)
    y_knots = np.sin(x_knots)

    model = build_natural_cubic_spline(x_knots, y_knots)

    x_dense = np.linspace(x_knots[0], x_knots[-1], 401)
    y_true = np.sin(x_dense)
    y_spline = evaluate_spline(model, x_dense)
    y_linear = np.interp(x_dense, x_knots, y_knots)

    spline_abs_err = np.abs(y_spline - y_true)
    linear_abs_err = np.abs(y_linear - y_true)

    spline_max = float(np.max(spline_abs_err))
    spline_mae = float(np.mean(spline_abs_err))
    linear_max = float(np.max(linear_abs_err))
    linear_mae = float(np.mean(linear_abs_err))

    d1_jump, d2_jump = continuity_report(model)

    print("Natural cubic spline interpolation demo")
    print("f(x)=sin(x), knots=9 in [0, 2pi], dense eval points=401")
    print("=" * 88)
    print("Error metrics")
    print(f"spline max abs error : {spline_max:.6e}")
    print(f"spline mean abs error: {spline_mae:.6e}")
    print(f"linear max abs error : {linear_max:.6e}")
    print(f"linear mean abs error: {linear_mae:.6e}")
    print(f"MAE improvement (linear/spline): {linear_mae / spline_mae:.2f}x")
    print("=" * 88)
    print("Smoothness and boundary checks")
    print(f"|M0|={abs(model.second_derivatives[0]):.3e}, |Mn|={abs(model.second_derivatives[-1]):.3e}")
    print(f"max first-derivative knot jump : {d1_jump:.3e}")
    print(f"max second-derivative knot jump: {d2_jump:.3e}")

    print("=" * 88)
    print("First 4 segment coefficients: S_i(x)=a+b*dx+c*dx^2+d*dx^3")
    for i in range(min(4, model.coeffs.shape[0])):
        a_i, b_i, c_i, d_i = model.coeffs[i]
        print(
            f"seg {i:02d} [{model.x[i]:.4f}, {model.x[i+1]:.4f}] -> "
            f"a={a_i:+.6f}, b={b_i:+.6f}, c={c_i:+.6f}, d={d_i:+.6f}"
        )

    print("=" * 88)
    sample_points = np.array([0.00, 0.60, 1.10, 2.20, 3.30, 4.40, 5.50, 6.20])
    sample_values = evaluate_spline(model, sample_points)
    print("Sample evaluations")
    print("x        spline(x)      sin(x)         abs_error")
    for xq, ys in zip(sample_points, sample_values):
        yt = np.sin(xq)
        print(f"{xq:5.2f}   {ys:12.8f}   {yt:12.8f}   {abs(ys - yt):10.3e}")


if __name__ == "__main__":
    main()
