"""B-spline curve/surface MVP (non-interactive, runnable with python3)."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

try:
    from scipy.interpolate import BSpline

    SCIPY_AVAILABLE = True
except Exception:
    BSpline = None
    SCIPY_AVAILABLE = False


def make_clamped_uniform_knots(n_ctrl: int, degree: int) -> np.ndarray:
    """Create a clamped uniform knot vector on [0, 1]."""
    if n_ctrl <= degree:
        raise ValueError("Need n_ctrl > degree for a valid B-spline.")

    n_internal = n_ctrl - degree - 1
    if n_internal > 0:
        internal = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
    else:
        internal = np.array([], dtype=float)
    return np.concatenate(
        [np.zeros(degree + 1), internal, np.ones(degree + 1)]
    ).astype(float)


def find_span(t: float, degree: int, knots: np.ndarray, n_ctrl: int) -> int:
    """Find knot span index i such that knots[i] <= t < knots[i+1]."""
    if t >= knots[n_ctrl]:
        return n_ctrl - 1
    if t <= knots[degree]:
        return degree

    low = degree
    high = n_ctrl
    mid = (low + high) // 2
    while t < knots[mid] or t >= knots[mid + 1]:
        if t < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def de_boor_point(
    t: float, degree: int, knots: np.ndarray, control_points: np.ndarray
) -> np.ndarray:
    """Evaluate one B-spline curve point with de Boor algorithm."""
    n_ctrl = control_points.shape[0]
    span = find_span(t, degree, knots, n_ctrl)
    d = [
        control_points[span - degree + j].astype(float).copy()
        for j in range(degree + 1)
    ]

    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            left = knots[span - degree + j]
            right = knots[span + 1 + j - r]
            den = right - left
            alpha = 0.0 if den == 0.0 else (t - left) / den
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[degree]


def evaluate_curve_de_boor(
    ts: np.ndarray, degree: int, knots: np.ndarray, control_points: np.ndarray
) -> np.ndarray:
    """Evaluate a B-spline curve at multiple parameters."""
    return np.vstack([de_boor_point(float(t), degree, knots, control_points) for t in ts])


def evaluate_surface_de_boor(
    us: np.ndarray,
    vs: np.ndarray,
    degree_u: int,
    degree_v: int,
    knots_u: np.ndarray,
    knots_v: np.ndarray,
    ctrl_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate a tensor-product B-spline surface on a parameter grid."""
    n_v = ctrl_grid.shape[1]
    dim = ctrl_grid.shape[2]
    out = np.zeros((len(us), len(vs), dim), dtype=float)

    for iu, u in enumerate(us):
        temp_ctrl = np.vstack(
            [
                de_boor_point(float(u), degree_u, knots_u, ctrl_grid[:, j, :])
                for j in range(n_v)
            ]
        )
        for iv, v in enumerate(vs):
            out[iu, iv, :] = de_boor_point(float(v), degree_v, knots_v, temp_ctrl)
    return out


def basis_funs(span: int, t: float, degree: int, knots: np.ndarray) -> np.ndarray:
    """Compute non-zero basis values N_{span-degree:span, degree}(t)."""
    n = np.zeros(degree + 1, dtype=float)
    n[0] = 1.0
    left = np.zeros(degree + 1, dtype=float)
    right = np.zeros(degree + 1, dtype=float)

    for j in range(1, degree + 1):
        left[j] = t - knots[span + 1 - j]
        right[j] = knots[span + j] - t
        saved = 0.0
        for r in range(j):
            den = right[r + 1] + left[j - r]
            temp = 0.0 if den == 0.0 else n[r] / den
            n[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        n[j] = saved
    return n


def basis_matrix(
    ts: np.ndarray, knots: np.ndarray, degree: int, n_ctrl: int
) -> np.ndarray:
    """Build basis matrix B where B[a, i] = N_{i,p}(t_a)."""
    b = np.zeros((len(ts), n_ctrl), dtype=float)
    if SCIPY_AVAILABLE:
        for i in range(n_ctrl):
            coeff = np.zeros(n_ctrl, dtype=float)
            coeff[i] = 1.0
            basis_i = BSpline(knots, coeff, degree, extrapolate=False)
            b[:, i] = basis_i(ts)
        return b

    for row, t in enumerate(ts):
        span = find_span(float(t), degree, knots, n_ctrl)
        local = basis_funs(span, float(t), degree, knots)
        start = span - degree
        b[row, start : start + degree + 1] = local
    return b


def curve_demo() -> Tuple[float, float, float]:
    """Run curve demo; return key diagnostics."""
    degree = 3
    control = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, -1.0],
            [4.0, 2.5],
            [6.0, 2.0],
            [7.0, -0.5],
            [8.0, 1.5],
            [9.0, 0.0],
        ],
        dtype=float,
    )
    knots = make_clamped_uniform_knots(len(control), degree)
    ts = np.linspace(0.0, 1.0, 201)

    t0 = time.perf_counter()
    curve_custom = evaluate_curve_de_boor(ts, degree, knots, control)
    t1 = time.perf_counter()

    if SCIPY_AVAILABLE:
        spline_x = BSpline(knots, control[:, 0], degree, extrapolate=False)
        spline_y = BSpline(knots, control[:, 1], degree, extrapolate=False)
        curve_scipy = np.column_stack([spline_x(ts), spline_y(ts)])
        max_diff = float(np.max(np.linalg.norm(curve_custom - curve_scipy, axis=1)))
    else:
        max_diff = 0.0
    t2 = time.perf_counter()

    bmat = basis_matrix(ts, knots, degree, len(control))
    partition_error = float(np.max(np.abs(bmat.sum(axis=1) - 1.0)))

    perturbed = control.copy()
    perturbed_idx = 3
    perturbed[perturbed_idx] += np.array([0.7, -0.4])
    curve_perturbed = evaluate_curve_de_boor(ts, degree, knots, perturbed)
    impact = np.linalg.norm(curve_perturbed - curve_custom, axis=1)
    active = impact > 1e-8
    if np.any(active):
        observed_support = (float(ts[active][0]), float(ts[active][-1]))
    else:
        observed_support = (float("nan"), float("nan"))
    theoretical_support = (
        float(knots[perturbed_idx]),
        float(knots[perturbed_idx + degree + 1]),
    )

    print("=== Curve Demo ===")
    print(f"Control points               : {len(control)}")
    print(f"Degree                       : {degree}")
    print(f"Sample count                 : {len(ts)}")
    if SCIPY_AVAILABLE:
        print(f"Max ||custom - scipy||       : {max_diff:.3e}")
    else:
        print("Max ||custom - scipy||       : skipped (scipy not installed)")
    print(f"Partition-of-unity max error : {partition_error:.3e}")
    if SCIPY_AVAILABLE:
        print(
            "Runtime (custom/scipy) [ms]  : "
            f"{(t1 - t0) * 1e3:.2f} / {(t2 - t1) * 1e3:.2f}"
        )
    else:
        print(f"Runtime (custom) [ms]        : {(t1 - t0) * 1e3:.2f}")
    print(
        "Local support [observed]     : "
        f"[{observed_support[0]:.3f}, {observed_support[1]:.3f}]"
    )
    print(
        "Local support [theoretical]  : "
        f"[{theoretical_support[0]:.3f}, {theoretical_support[1]:.3f}]"
    )
    print(f"Endpoint C(0), C(1)          : {curve_custom[0]} , {curve_custom[-1]}")
    return max_diff, partition_error, float(np.max(impact))


def surface_demo() -> Tuple[float, float]:
    """Run tensor-product surface demo; return geometry summary."""
    degree_u, degree_v = 3, 2
    n_u, n_v = 6, 5
    knots_u = make_clamped_uniform_knots(n_u, degree_u)
    knots_v = make_clamped_uniform_knots(n_v, degree_v)

    # Structured 3D control grid with a smooth "hill-like" z profile.
    ctrl_grid = np.zeros((n_u, n_v, 3), dtype=float)
    for i in range(n_u):
        for j in range(n_v):
            x = float(i)
            y = float(j)
            z = (
                1.8 * np.exp(-((i - 2.5) ** 2) / 6.0 - ((j - 2.0) ** 2) / 3.0)
                + 0.15 * i
                - 0.05 * j
            )
            ctrl_grid[i, j] = [x, y, z]

    us = np.linspace(0.0, 1.0, 31)
    vs = np.linspace(0.0, 1.0, 29)
    t0 = time.perf_counter()
    surf = evaluate_surface_de_boor(
        us, vs, degree_u, degree_v, knots_u, knots_v, ctrl_grid
    )
    t1 = time.perf_counter()

    z_min = float(np.min(surf[:, :, 2]))
    z_max = float(np.max(surf[:, :, 2]))
    center = surf[len(us) // 2, len(vs) // 2]

    print("\n=== Surface Demo ===")
    print(f"Control grid                 : {n_u} x {n_v}")
    print(f"Degrees (u, v)               : ({degree_u}, {degree_v})")
    print(f"Sample grid                  : {len(us)} x {len(vs)}")
    print(f"Surface eval runtime [ms]    : {(t1 - t0) * 1e3:.2f}")
    print(f"Z range                      : [{z_min:.4f}, {z_max:.4f}]")
    print(f"Center point S(0.5,0.5)      : {center}")
    return z_min, z_max


def main() -> None:
    max_diff, partition_error, max_impact = curve_demo()
    z_min, z_max = surface_demo()

    scipy_ok = (max_diff < 1e-9) if SCIPY_AVAILABLE else True
    ok = (
        scipy_ok
        and partition_error < 1e-12
        and max_impact > 1e-6
        and z_max > z_min
    )
    print("\nValidation:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
