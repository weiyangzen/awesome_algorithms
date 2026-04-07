"""NURBS curve MVP.

This script implements a minimal, auditable NURBS curve evaluator:
- Span search on knot vector
- Cox-de Boor B-spline basis recursion
- Rational weighted point evaluation
- Uniform parameter sampling

It runs without interactive input:
    uv run python demo.py
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


class NURBSCurve:
    """A minimal NURBS curve evaluator in Euclidean space R^d."""

    def __init__(
        self,
        control_points: np.ndarray,
        weights: np.ndarray,
        knot_vector: np.ndarray,
        degree: int,
    ) -> None:
        self.control_points = np.asarray(control_points, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.knot_vector = np.asarray(knot_vector, dtype=float)
        self.degree = int(degree)
        self._validate_inputs()
        self.n_ctrl = self.control_points.shape[0]
        self.n = self.n_ctrl - 1
        self.u_min = float(self.knot_vector[self.degree])
        self.u_max = float(self.knot_vector[self.n + 1])

    def _validate_inputs(self) -> None:
        if self.control_points.ndim != 2:
            raise ValueError("control_points must be a 2D array with shape (n, d)")
        if self.weights.ndim != 1:
            raise ValueError("weights must be a 1D array with length n")

        n_ctrl = self.control_points.shape[0]
        if n_ctrl == 0:
            raise ValueError("control_points cannot be empty")
        if self.degree < 1:
            raise ValueError("degree must be >= 1")
        if n_ctrl < self.degree + 1:
            raise ValueError("need at least degree+1 control points")
        if self.weights.shape[0] != n_ctrl:
            raise ValueError("weights length must equal number of control points")

        expected_knot_len = n_ctrl + self.degree + 1
        if self.knot_vector.shape != (expected_knot_len,):
            raise ValueError(
                f"knot_vector length must be {expected_knot_len}, "
                f"got {self.knot_vector.shape[0]}"
            )

        if not np.all(np.isfinite(self.control_points)):
            raise ValueError("control_points contains non-finite values")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights contains non-finite values")
        if not np.all(np.isfinite(self.knot_vector)):
            raise ValueError("knot_vector contains non-finite values")

        if np.any(self.weights <= 0.0):
            raise ValueError("weights must be strictly positive")

        diffs = np.diff(self.knot_vector)
        if np.any(diffs < -EPS):
            raise ValueError("knot_vector must be non-decreasing")

        n = n_ctrl - 1
        u_min = self.knot_vector[self.degree]
        u_max = self.knot_vector[n + 1]
        if u_max - u_min <= EPS:
            raise ValueError("invalid knot domain: [U[p], U[n+1]] has zero length")

    @staticmethod
    def find_span(n: int, degree: int, u: float, knot_vector: np.ndarray) -> int:
        """Find knot span index i such that U[i] <= u < U[i+1], with right-end clamp."""
        if u >= knot_vector[n + 1] - EPS:
            return n

        low = degree
        high = n + 1
        mid = (low + high) // 2

        while u < knot_vector[mid] or u >= knot_vector[mid + 1]:
            if u < knot_vector[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2

        return mid

    @staticmethod
    def basis_functions(span: int, u: float, degree: int, knot_vector: np.ndarray) -> np.ndarray:
        """Compute non-zero B-spline basis values N_{i-degree..i,degree}(u)."""
        basis = np.zeros(degree + 1, dtype=float)
        left = np.zeros(degree + 1, dtype=float)
        right = np.zeros(degree + 1, dtype=float)
        basis[0] = 1.0

        for j in range(1, degree + 1):
            left[j] = u - knot_vector[span + 1 - j]
            right[j] = knot_vector[span + j] - u
            saved = 0.0
            for r in range(j):
                denom = right[r + 1] + left[j - r]
                term = 0.0 if abs(denom) <= EPS else basis[r] / denom
                basis[r] = saved + right[r + 1] * term
                saved = left[j - r] * term
            basis[j] = saved

        return basis

    def evaluate(self, u: float) -> np.ndarray:
        """Evaluate one NURBS curve point C(u)."""
        if u < self.u_min - EPS or u > self.u_max + EPS:
            raise ValueError(f"u={u:.6f} outside domain [{self.u_min:.6f}, {self.u_max:.6f}]")

        uu = float(np.clip(u, self.u_min, self.u_max))
        span = self.find_span(self.n, self.degree, uu, self.knot_vector)
        basis = self.basis_functions(span, uu, self.degree, self.knot_vector)

        start = span - self.degree
        stop = span + 1
        local_points = self.control_points[start:stop]
        local_weights = self.weights[start:stop]

        weighted_basis = basis * local_weights
        denominator = float(np.sum(weighted_basis))
        if abs(denominator) <= EPS:
            raise ZeroDivisionError("NURBS denominator is near zero")

        numerator = np.sum(weighted_basis[:, None] * local_points, axis=0)
        return numerator / denominator

    def sample(self, num_points: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Sample C(u) uniformly in parameter domain."""
        if num_points < 2:
            raise ValueError("num_points must be >= 2")

        params = np.linspace(self.u_min, self.u_max, num_points)
        points = np.vstack([self.evaluate(u) for u in params])
        return params, points


def open_uniform_knot_vector(num_control_points: int, degree: int) -> np.ndarray:
    """Create an open uniform knot vector on [0,1]."""
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if num_control_points < degree + 1:
        raise ValueError("num_control_points must be >= degree+1")

    num_interior = num_control_points - degree - 1
    knots = [0.0] * (degree + 1)

    if num_interior > 0:
        interior = np.linspace(0.0, 1.0, num_interior + 2)[1:-1]
        knots.extend(interior.tolist())

    knots.extend([1.0] * (degree + 1))
    return np.asarray(knots, dtype=float)


def quarter_circle_case() -> tuple[NURBSCurve, np.ndarray, np.ndarray]:
    """Build a quadratic NURBS that exactly represents a quarter circle."""
    control_points = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    weights = np.array([1.0, np.sqrt(2.0) / 2.0, 1.0], dtype=float)
    degree = 2
    knot_vector = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    curve = NURBSCurve(control_points, weights, knot_vector, degree)
    params, points = curve.sample(num_points=21)
    return curve, params, points


def generic_curve_case() -> tuple[NURBSCurve, np.ndarray, np.ndarray]:
    """Build a generic cubic NURBS example with open uniform knots."""
    control_points = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 3.0],
            [4.0, 0.0],
            [5.0, -1.0],
        ],
        dtype=float,
    )
    weights = np.array([1.0, 0.7, 1.4, 0.9, 1.0], dtype=float)
    degree = 3
    knot_vector = open_uniform_knot_vector(num_control_points=5, degree=degree)
    curve = NURBSCurve(control_points, weights, knot_vector, degree)
    params, points = curve.sample(num_points=11)
    return curve, params, points


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    curve_q, params_q, points_q = quarter_circle_case()
    radii = np.linalg.norm(points_q, axis=1)
    max_radius_error = float(np.max(np.abs(radii - 1.0)))
    midpoint = curve_q.evaluate(0.5)

    print("=== NURBS MVP Demo ===")
    print("Case A: quadratic NURBS quarter circle")
    print(f"parameter_domain=[{curve_q.u_min:.3f}, {curve_q.u_max:.3f}]")
    print(f"start_point={points_q[0]}")
    print(f"mid_point(u=0.5)={midpoint}")
    print(f"end_point={points_q[-1]}")
    print(f"max_radius_error={max_radius_error:.12f}")

    print("sample_points_case_A (u, x, y):")
    for idx in [0, 5, 10, 15, 20]:
        u = params_q[idx]
        x, y = points_q[idx]
        print(f"  u={u:.2f}, x={x:.6f}, y={y:.6f}")

    curve_g, params_g, points_g = generic_curve_case()
    print("\nCase B: generic cubic NURBS")
    print(f"parameter_domain=[{curve_g.u_min:.3f}, {curve_g.u_max:.3f}]")
    print(f"knot_vector={curve_g.knot_vector}")
    print(f"start_point={points_g[0]}")
    print(f"end_point={points_g[-1]}")

    finite_ok = bool(np.all(np.isfinite(points_q)) and np.all(np.isfinite(points_g)))
    monotone_u_ok = bool(np.all(np.diff(params_q) >= 0.0) and np.all(np.diff(params_g) >= 0.0))
    print("\nChecks:")
    print(f"  finite_points={finite_ok}")
    print(f"  monotone_parameters={monotone_u_ok}")


if __name__ == "__main__":
    main()
