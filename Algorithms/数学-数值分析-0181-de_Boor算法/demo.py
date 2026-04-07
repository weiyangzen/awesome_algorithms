"""de Boor algorithm MVP.

Non-interactive demo for evaluating a clamped B-spline curve.
The implementation includes:
- de Boor local recursion (main algorithm)
- Cox-de Boor basis-function evaluation (independent cross-check)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SplineReport:
    degree: int
    control_point_count: int
    knot_vector: np.ndarray
    u_values: np.ndarray
    curve_de_boor: np.ndarray
    curve_basis: np.ndarray
    max_diff_deboor_vs_basis: float
    endpoint_error_start: float
    endpoint_error_end: float


def build_open_uniform_knot_vector(n_ctrl: int, degree: int) -> np.ndarray:
    """Build a clamped open-uniform knot vector on [0, 1]."""
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if n_ctrl < degree + 1:
        raise ValueError("Need at least degree + 1 control points.")

    knot_count = n_ctrl + degree + 1
    knots = np.zeros(knot_count, dtype=np.float64)
    knots[-(degree + 1) :] = 1.0

    num_inner = n_ctrl - degree - 1
    if num_inner > 0:
        interior = np.linspace(0.0, 1.0, num_inner + 2, dtype=np.float64)[1:-1]
        knots[degree + 1 : degree + 1 + num_inner] = interior

    return knots


def validate_spline_inputs(control_points: np.ndarray, knots: np.ndarray, degree: int) -> None:
    if control_points.ndim != 2:
        raise ValueError("control_points must have shape (n_ctrl, dim)")
    if degree < 1:
        raise ValueError("degree must be >= 1")

    n_ctrl = control_points.shape[0]
    expected_knots = n_ctrl + degree + 1
    if len(knots) != expected_knots:
        raise ValueError(f"knot vector length must be {expected_knots}, got {len(knots)}")
    if np.any(np.diff(knots) < 0.0):
        raise ValueError("knot vector must be nondecreasing")

    domain_start = knots[degree]
    domain_end = knots[n_ctrl]
    if not domain_end > domain_start:
        raise ValueError("Invalid knot domain: knots[n_ctrl] must be > knots[degree]")


def find_knot_span(u: float, knots: np.ndarray, degree: int, n_ctrl: int) -> int:
    """Find span index k such that U[k] <= u < U[k+1], with right-end handling."""
    n = n_ctrl - 1
    domain_start = knots[degree]
    domain_end = knots[n_ctrl]

    if u <= domain_start:
        return degree
    if u >= domain_end:
        return n

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


def de_boor_point(
    u: float,
    control_points: np.ndarray,
    knots: np.ndarray,
    degree: int,
    eps: float = 1e-14,
) -> np.ndarray:
    """Evaluate one spline point C(u) using de Boor recursion."""
    n_ctrl = control_points.shape[0]
    domain_start = knots[degree]
    domain_end = knots[n_ctrl]

    u_clamped = float(np.clip(u, domain_start, domain_end))
    k = find_knot_span(u_clamped, knots, degree, n_ctrl)

    d = np.array([control_points[k - degree + j] for j in range(degree + 1)], dtype=np.float64)

    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            i = k - degree + j
            left = knots[i]
            right = knots[k + j + 1 - r]
            denom = right - left
            if abs(denom) < eps:
                alpha = 0.0
            else:
                alpha = (u_clamped - left) / denom
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

    return d[degree]


def evaluate_curve_de_boor(
    u_values: Sequence[float],
    control_points: np.ndarray,
    knots: np.ndarray,
    degree: int,
) -> np.ndarray:
    return np.array(
        [de_boor_point(float(u), control_points, knots, degree) for u in u_values],
        dtype=np.float64,
    )


def basis_functions_at(
    u: float,
    knots: np.ndarray,
    degree: int,
    n_ctrl: int,
    eps: float = 1e-14,
) -> np.ndarray:
    """Compute all degree-p basis values N_{i,p}(u) by Cox-de Boor DP."""
    n = n_ctrl - 1
    domain_start = knots[degree]
    domain_end = knots[n_ctrl]
    u_clamped = float(np.clip(u, domain_start, domain_end))

    table = np.zeros((n_ctrl, degree + 1), dtype=np.float64)

    for i in range(n_ctrl):
        left = knots[i]
        right = knots[i + 1]
        is_last = (u_clamped == domain_end) and (i == n)
        if (left <= u_clamped < right) or is_last:
            table[i, 0] = 1.0

    for p in range(1, degree + 1):
        for i in range(n_ctrl):
            first = 0.0
            denom1 = knots[i + p] - knots[i]
            if abs(denom1) > eps:
                first = (u_clamped - knots[i]) / denom1 * table[i, p - 1]

            second = 0.0
            if i + 1 < n_ctrl:
                denom2 = knots[i + p + 1] - knots[i + 1]
                if abs(denom2) > eps:
                    second = (knots[i + p + 1] - u_clamped) / denom2 * table[i + 1, p - 1]

            table[i, p] = first + second

    return table[:, degree]


def evaluate_curve_by_basis(
    u_values: Sequence[float],
    control_points: np.ndarray,
    knots: np.ndarray,
    degree: int,
) -> np.ndarray:
    points = []
    n_ctrl = control_points.shape[0]
    for u in u_values:
        basis = basis_functions_at(float(u), knots, degree, n_ctrl)
        point = basis @ control_points
        points.append(point)
    return np.array(points, dtype=np.float64)


def run_demo() -> SplineReport:
    degree = 3
    control_points = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [4.0, 3.0],
            [5.0, 0.0],
            [6.0, -1.0],
            [8.0, 0.0],
        ],
        dtype=np.float64,
    )

    knots = build_open_uniform_knot_vector(n_ctrl=len(control_points), degree=degree)
    validate_spline_inputs(control_points, knots, degree)

    u_values = np.linspace(0.0, 1.0, 21, dtype=np.float64)

    curve_de_boor = evaluate_curve_de_boor(u_values, control_points, knots, degree)
    curve_basis = evaluate_curve_by_basis(u_values, control_points, knots, degree)

    diffs = np.linalg.norm(curve_de_boor - curve_basis, axis=1)
    max_diff = float(np.max(diffs))

    endpoint_error_start = float(np.linalg.norm(curve_de_boor[0] - control_points[0]))
    endpoint_error_end = float(np.linalg.norm(curve_de_boor[-1] - control_points[-1]))

    return SplineReport(
        degree=degree,
        control_point_count=len(control_points),
        knot_vector=knots,
        u_values=u_values,
        curve_de_boor=curve_de_boor,
        curve_basis=curve_basis,
        max_diff_deboor_vs_basis=max_diff,
        endpoint_error_start=endpoint_error_start,
        endpoint_error_end=endpoint_error_end,
    )


def main() -> None:
    report = run_demo()

    print("de Boor Algorithm Demo")
    print("=" * 80)
    print(f"degree: {report.degree}")
    print(f"control points: {report.control_point_count}")
    print(f"knot vector: {np.array2string(report.knot_vector, precision=4, separator=', ')}")
    print(f"samples: {len(report.u_values)}")
    print(f"max ||deBoor - basis||_2: {report.max_diff_deboor_vs_basis:.3e}")
    print(f"endpoint error (u=0): {report.endpoint_error_start:.3e}")
    print(f"endpoint error (u=1): {report.endpoint_error_end:.3e}")

    print("-" * 80)
    print("Selected sample points (u, x, y):")
    selected_idx = [0, 5, 10, 15, 20]
    for idx in selected_idx:
        u = report.u_values[idx]
        x, y = report.curve_de_boor[idx]
        print(f"u={u:0.2f} -> ({x: .6f}, {y: .6f})")


if __name__ == "__main__":
    main()
