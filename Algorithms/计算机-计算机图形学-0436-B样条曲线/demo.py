"""B-spline curve MVP.

This script implements a clamped open-uniform B-spline curve evaluator using:
1) de Boor algorithm (stable geometric recursion)
2) Cox-de Boor basis recursion (explicit basis definition)

It then validates core properties and optionally compares against SciPy's BSpline.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


Array = np.ndarray


def _as_control_points(control_points: Array) -> Array:
    cp = np.asarray(control_points, dtype=float)
    if cp.ndim != 2 or cp.shape[0] < 2:
        raise ValueError("control_points must have shape (n, dim), n >= 2")
    return cp


def open_uniform_knot_vector(n_ctrl: int, degree: int) -> Array:
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if n_ctrl <= degree:
        raise ValueError("need n_ctrl > degree for a valid B-spline curve")

    interior_count = n_ctrl - degree - 1
    if interior_count > 0:
        interior = np.linspace(0.0, 1.0, interior_count + 2, dtype=float)[1:-1]
    else:
        interior = np.array([], dtype=float)

    knots = np.concatenate(
        (
            np.zeros(degree + 1, dtype=float),
            interior,
            np.ones(degree + 1, dtype=float),
        )
    )
    return knots


def find_knot_span(u: float, degree: int, knots: Array, n_ctrl: int) -> int:
    n = n_ctrl - 1

    if u >= knots[n + 1]:
        return n
    if u <= knots[degree]:
        return degree

    low = degree
    high = n + 1
    mid = (low + high) // 2
    while u < knots[mid] or u >= knots[mid + 1]:
        if u < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def de_boor_point(control_points: Array, degree: int, knots: Array, u: float) -> Array:
    cp = _as_control_points(control_points)
    span = find_knot_span(float(u), degree, knots, cp.shape[0])

    d = np.array([cp[span - degree + j].copy() for j in range(degree + 1)], dtype=float)

    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            i = span - degree + j
            denom = knots[i + degree - r + 1] - knots[i]
            alpha = 0.0 if denom == 0.0 else (u - knots[i]) / denom
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

    return d[degree]


def evaluate_de_boor(control_points: Array, degree: int, knots: Array, u_values: Array) -> Array:
    cp = _as_control_points(control_points)
    u = np.asarray(u_values, dtype=float).reshape(-1)
    out = np.empty((u.size, cp.shape[1]), dtype=float)
    for idx, uu in enumerate(u):
        out[idx] = de_boor_point(cp, degree, knots, float(uu))
    return out


def cox_de_boor_basis_matrix(n_ctrl: int, degree: int, knots: Array, u_values: Array) -> Array:
    u = np.asarray(u_values, dtype=float).reshape(-1)

    @lru_cache(maxsize=None)
    def basis(i: int, p: int, us: float) -> float:
        if p == 0:
            left = knots[i]
            right = knots[i + 1]
            if (left <= us < right) or (us == knots[-1] and right == knots[-1]):
                return 1.0
            return 0.0

        left = 0.0
        left_denom = knots[i + p] - knots[i]
        if left_denom > 0.0:
            left = ((us - knots[i]) / left_denom) * basis(i, p - 1, us)

        right = 0.0
        right_denom = knots[i + p + 1] - knots[i + 1]
        if right_denom > 0.0:
            right = ((knots[i + p + 1] - us) / right_denom) * basis(i + 1, p - 1, us)

        return left + right

    mat = np.empty((u.size, n_ctrl), dtype=float)
    for row, uu in enumerate(u):
        for i in range(n_ctrl):
            mat[row, i] = basis(i, degree, float(uu))
    return mat


def evaluate_by_basis(control_points: Array, degree: int, knots: Array, u_values: Array) -> tuple[Array, Array]:
    cp = _as_control_points(control_points)
    basis = cox_de_boor_basis_matrix(cp.shape[0], degree, knots, u_values)
    return basis @ cp, basis


def polyline_length(points: Array) -> float:
    p = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def main() -> None:
    control_points = np.array(
        [
            [0.0, 0.0],
            [0.8, 1.6],
            [1.8, 2.2],
            [3.0, 0.8],
            [4.2, 2.4],
            [5.6, 1.9],
            [6.4, 0.2],
        ],
        dtype=float,
    )
    degree = 3
    knots = open_uniform_knot_vector(control_points.shape[0], degree)
    u_values = np.linspace(0.0, 1.0, 401)

    curve_de_boor = evaluate_de_boor(control_points, degree, knots, u_values)
    curve_basis, basis_matrix = evaluate_by_basis(control_points, degree, knots, u_values)

    method_diff = float(np.max(np.linalg.norm(curve_de_boor - curve_basis, axis=1)))

    basis_sum = basis_matrix.sum(axis=1)
    partition_err = float(np.max(np.abs(basis_sum - 1.0)))

    active_count = np.sum(basis_matrix > 1e-12, axis=1)
    local_support_max = int(np.max(active_count))

    endpoint_err = max(
        float(np.linalg.norm(curve_de_boor[0] - control_points[0])),
        float(np.linalg.norm(curve_de_boor[-1] - control_points[-1])),
    )

    curve_len = polyline_length(curve_de_boor)
    chord_len = float(np.linalg.norm(control_points[-1] - control_points[0]))

    scipy_diff = None
    try:
        from scipy.interpolate import BSpline

        scipy_curve = BSpline(knots, control_points, degree, extrapolate=False)(u_values)
        scipy_diff = float(np.max(np.linalg.norm(curve_de_boor - scipy_curve, axis=1)))
    except Exception:
        scipy_diff = None

    print("B-spline Curve MVP (CS-0277)")
    print(f"Control points: {control_points.shape[0]}")
    print(f"Degree: {degree}")
    print(f"Knot vector: {knots}")
    print(f"Samples: {u_values.size}")
    print(f"Max(de Boor - Cox basis): {method_diff:.3e}")
    print(f"Partition of unity error: {partition_err:.3e}")
    print(f"Max active basis count: {local_support_max}")
    print(f"Endpoint interpolation error: {endpoint_err:.3e}")
    print(f"Polyline length: {curve_len:.6f}")
    print(f"Chord length: {chord_len:.6f}")
    if scipy_diff is not None:
        print(f"Max(de Boor - SciPy BSpline): {scipy_diff:.3e}")

    assert method_diff < 1e-10, "de Boor and Cox-de Boor basis results diverged"
    assert partition_err < 1e-12, "Basis functions should sum to 1"
    assert local_support_max <= degree + 1, "Local support violated"
    assert endpoint_err < 1e-12, "Clamped B-spline endpoint interpolation failed"
    assert curve_len + 1e-12 >= chord_len, "Polyline length must be >= chord length"
    if scipy_diff is not None:
        assert scipy_diff < 1e-10, "Mismatch with scipy.interpolate.BSpline"

    print("All checks passed.")


if __name__ == "__main__":
    main()
