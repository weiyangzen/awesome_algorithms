"""Bezier curve MVP.

This script demonstrates two equivalent implementations of Bezier evaluation:
1) Bernstein polynomial form
2) de Casteljau recursive interpolation

It also validates subdivision and basic geometric properties.
"""

from __future__ import annotations

from math import comb
from typing import Tuple

import numpy as np


Array = np.ndarray


def _as_control_points(control_points: Array) -> Array:
    cp = np.asarray(control_points, dtype=float)
    if cp.ndim != 2 or cp.shape[0] < 2:
        raise ValueError("control_points must have shape (n+1, dim), n >= 1")
    return cp


def bernstein_basis(n: int, t_values: Array) -> Array:
    t = np.asarray(t_values, dtype=float).reshape(-1)
    omt = 1.0 - t
    basis = np.empty((t.size, n + 1), dtype=float)
    for i in range(n + 1):
        basis[:, i] = comb(n, i) * (t**i) * (omt ** (n - i))
    return basis


def bezier_bernstein(control_points: Array, t_values: Array) -> Array:
    cp = _as_control_points(control_points)
    n = cp.shape[0] - 1
    basis = bernstein_basis(n, t_values)
    return basis @ cp


def de_casteljau_point(control_points: Array, t: float) -> Array:
    cp = _as_control_points(control_points)
    work = cp.copy()
    n = work.shape[0]
    for level in range(1, n):
        work[: n - level] = (1.0 - t) * work[: n - level] + t * work[1 : n - level + 1]
    return work[0]


def bezier_de_casteljau(control_points: Array, t_values: Array) -> Array:
    cp = _as_control_points(control_points)
    t = np.asarray(t_values, dtype=float).reshape(-1)
    out = np.empty((t.size, cp.shape[1]), dtype=float)
    for idx, tt in enumerate(t):
        out[idx] = de_casteljau_point(cp, float(tt))
    return out


def bezier_split(control_points: Array, t: float) -> Tuple[Array, Array]:
    cp = _as_control_points(control_points)
    n = cp.shape[0]

    left = np.empty_like(cp)
    right = np.empty_like(cp)

    work = cp.copy()
    left[0] = work[0]
    right[-1] = work[-1]

    for level in range(1, n):
        work = (1.0 - t) * work[:-1] + t * work[1:]
        left[level] = work[0]
        right[-(level + 1)] = work[-1]

    return left, right


def derivative_control_points(control_points: Array) -> Array:
    cp = _as_control_points(control_points)
    n = cp.shape[0] - 1
    return n * (cp[1:] - cp[:-1])


def polyline_length(points: Array) -> float:
    p = np.asarray(points, dtype=float)
    seg = np.diff(p, axis=0)
    return float(np.linalg.norm(seg, axis=1).sum())


def inside_axis_aligned_hull(points: Array, control_points: Array, tol: float = 1e-12) -> bool:
    p = np.asarray(points, dtype=float)
    cp = _as_control_points(control_points)
    lo = cp.min(axis=0) - tol
    hi = cp.max(axis=0) + tol
    return bool(np.all((p >= lo) & (p <= hi)))


def main() -> None:
    control_points = np.array(
        [
            [0.0, 0.0],
            [1.2, 2.0],
            [2.3, -1.1],
            [4.0, 1.0],
        ],
        dtype=float,
    )

    t_values = np.linspace(0.0, 1.0, 501)

    curve_bernstein = bezier_bernstein(control_points, t_values)
    curve_casteljau = bezier_de_casteljau(control_points, t_values)

    method_diff = float(np.max(np.linalg.norm(curve_bernstein - curve_casteljau, axis=1)))

    endpoint_err = max(
        float(np.linalg.norm(curve_casteljau[0] - control_points[0])),
        float(np.linalg.norm(curve_casteljau[-1] - control_points[-1])),
    )

    deriv_cp = derivative_control_points(control_points)
    tangent_curve_ends = bezier_de_casteljau(deriv_cp, np.array([0.0, 1.0]))
    tangent_formula_ends = np.vstack((
        (control_points.shape[0] - 1) * (control_points[1] - control_points[0]),
        (control_points.shape[0] - 1) * (control_points[-1] - control_points[-2]),
    ))
    tangent_err = float(np.max(np.linalg.norm(tangent_curve_ends - tangent_formula_ends, axis=1)))

    split_t = 0.37
    left_cp, right_cp = bezier_split(control_points, split_t)
    u = np.linspace(0.0, 1.0, 251)

    left_curve = bezier_de_casteljau(left_cp, u)
    right_curve = bezier_de_casteljau(right_cp, u)

    left_ref = bezier_de_casteljau(control_points, split_t * u)
    right_ref = bezier_de_casteljau(control_points, split_t + (1.0 - split_t) * u)

    left_err = float(np.max(np.linalg.norm(left_curve - left_ref, axis=1)))
    right_err = float(np.max(np.linalg.norm(right_curve - right_ref, axis=1)))

    curve_len = polyline_length(curve_casteljau)
    chord_len = float(np.linalg.norm(control_points[-1] - control_points[0]))

    hull_ok = inside_axis_aligned_hull(curve_casteljau, control_points)

    print("Bezier Curve MVP (CS-0276)")
    print(f"Control points: {control_points.shape[0]} (degree {control_points.shape[0] - 1})")
    print(f"Samples: {t_values.size}")
    print(f"Max(Bernstein - de Casteljau): {method_diff:.3e}")
    print(f"Endpoint error: {endpoint_err:.3e}")
    print(f"Endpoint tangent error: {tangent_err:.3e}")
    print(f"Subdivision left error: {left_err:.3e}")
    print(f"Subdivision right error: {right_err:.3e}")
    print(f"Polyline length: {curve_len:.6f}")
    print(f"Chord length: {chord_len:.6f}")
    print(f"Axis-aligned hull check: {hull_ok}")

    assert method_diff < 1e-12, "Bernstein and de Casteljau outputs diverged"
    assert endpoint_err < 1e-12, "Curve endpoint interpolation failed"
    assert tangent_err < 1e-12, "Endpoint tangent formula mismatch"
    assert left_err < 1e-12 and right_err < 1e-12, "Subdivision consistency failed"
    assert curve_len + 1e-12 >= chord_len, "Curve length should be >= chord length"
    assert hull_ok, "Curve violated control-point axis-aligned hull"

    print("All checks passed.")


if __name__ == "__main__":
    main()
