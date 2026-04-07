"""Minimal runnable MVP for NURBS (Non-Uniform Rational B-Splines)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass
class NURBSCurve:
    """Container for one NURBS curve definition."""

    control_points: np.ndarray  # shape: (n+1, dim)
    weights: np.ndarray  # shape: (n+1,)
    degree: int
    knots: np.ndarray  # shape: (n+degree+2,)

    def __post_init__(self) -> None:
        self.control_points = np.asarray(self.control_points, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.knots = np.asarray(self.knots, dtype=float)

        if self.control_points.ndim != 2:
            raise ValueError("control_points must be 2D: (n+1, dim)")
        if self.weights.ndim != 1:
            raise ValueError("weights must be 1D")
        if self.control_points.shape[0] != self.weights.size:
            raise ValueError("weights size must match number of control points")
        if self.degree < 1:
            raise ValueError("degree must be >= 1")

        n = self.num_control_points - 1
        expected_knot_count = n + self.degree + 2
        if self.knots.ndim != 1 or self.knots.size != expected_knot_count:
            raise ValueError(
                f"knots length must be {expected_knot_count}, got {self.knots.size}"
            )
        if not np.all(np.isfinite(self.control_points)):
            raise ValueError("control_points contains non-finite values")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights contains non-finite values")
        if not np.all(np.isfinite(self.knots)):
            raise ValueError("knots contains non-finite values")
        if np.any(self.weights <= 0.0):
            raise ValueError("all weights must be positive")
        if np.any(np.diff(self.knots) < 0.0):
            raise ValueError("knots must be non-decreasing")

        u_min = self.knots[self.degree]
        u_max = self.knots[n + 1]
        if not u_min < u_max:
            raise ValueError("invalid knot domain: knots[p] must be < knots[n+1]")

    @property
    def num_control_points(self) -> int:
        return int(self.control_points.shape[0])

    @property
    def dim(self) -> int:
        return int(self.control_points.shape[1])



def make_open_uniform_knot(n_control_points: int, degree: int) -> np.ndarray:
    """Build clamped open-uniform knot vector on [0, 1]."""
    if n_control_points < degree + 1:
        raise ValueError("n_control_points must be >= degree + 1")
    if degree < 1:
        raise ValueError("degree must be >= 1")

    n = n_control_points - 1
    m = n + degree + 1
    knots = np.zeros(m + 1, dtype=float)

    # Clamped ends.
    knots[: degree + 1] = 0.0
    knots[m - degree :] = 1.0

    # Uniform internal knots.
    num_internal = n - degree
    if num_internal > 0:
        denominator = num_internal + 1
        for j in range(1, num_internal + 1):
            knots[degree + j] = j / denominator

    return knots



def find_span(n: int, degree: int, u: float, knots: np.ndarray) -> int:
    """Find knot span index such that knots[span] <= u < knots[span+1]."""
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



def basis_funs(span: int, u: float, degree: int, knots: np.ndarray) -> np.ndarray:
    """Compute nonzero B-spline basis values N_{span-degree ... span, degree}(u)."""
    n = np.zeros(degree + 1, dtype=float)
    left = np.zeros(degree + 1, dtype=float)
    right = np.zeros(degree + 1, dtype=float)

    n[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u

        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            if denom == 0.0:
                temp = 0.0
            else:
                temp = n[r] / denom
            n[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        n[j] = saved

    return n



def rational_basis_row(curve: NURBSCurve, u: float) -> np.ndarray:
    """Return full rational basis row R_i,p(u), i=0..n."""
    n = curve.num_control_points - 1
    span = find_span(n, curve.degree, u, curve.knots)
    local_n = basis_funs(span, u, curve.degree, curve.knots)

    start = span - curve.degree
    local_weights = curve.weights[start : span + 1]
    weighted = local_n * local_weights
    denom = float(np.sum(weighted))
    if denom <= 0.0:
        raise ZeroDivisionError("rational basis denominator is non-positive")

    local_r = weighted / denom

    row = np.zeros(curve.num_control_points, dtype=float)
    row[start : span + 1] = local_r
    return row



def curve_point(curve: NURBSCurve, u: float) -> np.ndarray:
    """Evaluate one NURBS point C(u)."""
    domain_min = float(curve.knots[curve.degree])
    domain_max = float(curve.knots[curve.num_control_points])
    if u < domain_min or u > domain_max:
        raise ValueError(f"u must be in [{domain_min}, {domain_max}], got {u}")

    r = rational_basis_row(curve, u)
    return r @ curve.control_points



def sample_curve(curve: NURBSCurve, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample NURBS curve at evenly spaced parameters in valid domain."""
    if num_samples < 2:
        raise ValueError("num_samples must be >= 2")

    u_min = float(curve.knots[curve.degree])
    u_max = float(curve.knots[curve.num_control_points])
    params = np.linspace(u_min, u_max, num_samples)
    points = np.vstack([curve_point(curve, float(u)) for u in params])
    return params, points



def build_general_demo_curve() -> NURBSCurve:
    """Create one cubic NURBS curve for generic shape demonstration."""
    control_points = np.array(
        [
            [0.0, 0.0],
            [1.2, 2.2],
            [2.2, -1.0],
            [3.4, 1.5],
            [4.1, 2.6],
            [5.0, 0.2],
        ],
        dtype=float,
    )
    weights = np.array([1.0, 0.8, 1.5, 1.0, 0.9, 1.2], dtype=float)
    degree = 3
    knots = make_open_uniform_knot(n_control_points=control_points.shape[0], degree=degree)
    return NURBSCurve(control_points=control_points, weights=weights, degree=degree, knots=knots)



def build_quarter_circle_curve() -> NURBSCurve:
    """Quadratic NURBS that exactly represents a quarter unit circle."""
    w = 1.0 / math.sqrt(2.0)
    control_points = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    weights = np.array([1.0, w, 1.0], dtype=float)
    degree = 2
    knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    return NURBSCurve(control_points=control_points, weights=weights, degree=degree, knots=knots)



def verify_partition_of_unity(curve: NURBSCurve, params: np.ndarray, tol: float = 1e-10) -> float:
    """Return maximum |sum_i R_i(u) - 1| across sample params."""
    max_err = 0.0
    for u in params:
        r = rational_basis_row(curve, float(u))
        err = abs(float(np.sum(r)) - 1.0)
        max_err = max(max_err, err)
    if max_err > tol:
        raise AssertionError(f"partition of unity failed: max_err={max_err}")
    return max_err



def verify_endpoints(curve: NURBSCurve, tol: float = 1e-10) -> float:
    """Check clamped NURBS endpoint interpolation error."""
    u_min = float(curve.knots[curve.degree])
    u_max = float(curve.knots[curve.num_control_points])

    c0 = curve_point(curve, u_min)
    c1 = curve_point(curve, u_max)

    e0 = float(np.linalg.norm(c0 - curve.control_points[0]))
    e1 = float(np.linalg.norm(c1 - curve.control_points[-1]))
    max_err = max(e0, e1)
    if max_err > tol:
        raise AssertionError(f"endpoint interpolation failed: max_err={max_err}")
    return max_err



def verify_quarter_circle(curve: NURBSCurve, num_samples: int = 121, tol: float = 5e-12) -> float:
    """Check that sampled points satisfy x^2+y^2 ~= 1 for quarter-circle NURBS."""
    _, points = sample_curve(curve, num_samples)
    radius = np.sqrt(np.sum(points**2, axis=1))
    max_err = float(np.max(np.abs(radius - 1.0)))
    if max_err > tol:
        raise AssertionError(f"quarter-circle radius check failed: max_err={max_err}")
    return max_err



def run_demo() -> None:
    """Run deterministic NURBS MVP demo without interactive input."""
    general_curve = build_general_demo_curve()
    circle_curve = build_quarter_circle_curve()

    params_general, points_general = sample_curve(general_curve, num_samples=41)
    _, points_circle = sample_curve(circle_curve, num_samples=41)

    partition_err_general = verify_partition_of_unity(general_curve, params_general)
    endpoint_err_general = verify_endpoints(general_curve)
    circle_radius_err = verify_quarter_circle(circle_curve)

    preview_k = 6
    print("=" * 88)
    print("NURBS MVP (MATH-0178)")
    print("=" * 88)

    print("[General cubic NURBS]")
    print(f"control_points={general_curve.num_control_points}, degree={general_curve.degree}, dim={general_curve.dim}")
    print(f"knot_vector={np.array2string(general_curve.knots, precision=4, separator=', ')}")
    print(f"max_partition_unity_error={partition_err_general:.3e}")
    print(f"max_endpoint_error={endpoint_err_general:.3e}")

    for i in range(preview_k):
        u = params_general[i]
        p = points_general[i]
        print(f"sample_general[{i}] u={u:.4f} -> ({p[0]:+.6f}, {p[1]:+.6f})")

    print("\n[Quarter unit circle NURBS]")
    print(f"control_points={circle_curve.num_control_points}, degree={circle_curve.degree}")
    print(f"weights={np.array2string(circle_curve.weights, precision=6, separator=', ')}")
    print(f"max_radius_error={circle_radius_err:.3e}")

    for i in range(preview_k):
        p = points_circle[i]
        r = math.sqrt(float(p[0] * p[0] + p[1] * p[1]))
        print(f"sample_circle[{i}] -> ({p[0]:+.6f}, {p[1]:+.6f}), radius={r:.12f}")



def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
