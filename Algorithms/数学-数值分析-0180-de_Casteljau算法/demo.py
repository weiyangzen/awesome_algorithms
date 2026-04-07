"""Minimal runnable MVP: de Casteljau algorithm for Bezier curves."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DeCasteljauResult:
    """Container for one de Casteljau evaluation at parameter t."""

    t: float
    point: np.ndarray
    levels: list[np.ndarray]


def validate_control_points(control_points: np.ndarray) -> np.ndarray:
    """Validate and normalize control points into float ndarray of shape (n+1, d)."""
    cp = np.asarray(control_points, dtype=float)
    if cp.ndim != 2:
        raise ValueError(f"control_points must be 2D, got shape={cp.shape}")
    if cp.shape[0] < 2:
        raise ValueError("at least 2 control points are required")
    if cp.shape[1] < 1:
        raise ValueError("point dimension must be >= 1")
    if not np.all(np.isfinite(cp)):
        raise ValueError("control_points contain non-finite values")
    return cp


def validate_t(t: float) -> float:
    """Validate Bezier parameter t in [0, 1]."""
    t_val = float(t)
    if not math.isfinite(t_val):
        raise ValueError(f"t must be finite, got {t!r}")
    if not (0.0 <= t_val <= 1.0):
        raise ValueError(f"t must be in [0, 1], got {t_val}")
    return t_val


def de_casteljau_point(control_points: np.ndarray, t: float) -> DeCasteljauResult:
    """Evaluate one Bezier point using the de Casteljau recursion pyramid."""
    cp = validate_control_points(control_points)
    t_val = validate_t(t)

    levels: list[np.ndarray] = [cp.copy()]
    current = cp.copy()

    # Build pyramid levels: (n+1) -> n -> ... -> 1 point.
    while current.shape[0] > 1:
        current = (1.0 - t_val) * current[:-1] + t_val * current[1:]
        levels.append(current.copy())

    return DeCasteljauResult(t=t_val, point=levels[-1][0], levels=levels)


def bernstein_point(control_points: np.ndarray, t: float) -> np.ndarray:
    """Evaluate one Bezier point from Bernstein polynomial definition."""
    cp = validate_control_points(control_points)
    t_val = validate_t(t)

    n = cp.shape[0] - 1
    weights = np.array(
        [math.comb(n, i) * ((1.0 - t_val) ** (n - i)) * (t_val**i) for i in range(n + 1)],
        dtype=float,
    )
    return weights @ cp


def sample_bezier_curve(control_points: np.ndarray, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample Bezier curve on an even grid of t values in [0, 1]."""
    cp = validate_control_points(control_points)
    m = int(num_samples)
    if m < 2:
        raise ValueError(f"num_samples must be >= 2, got {m}")

    t_values = np.linspace(0.0, 1.0, m)
    points = np.array([de_casteljau_point(cp, float(t)).point for t in t_values], dtype=float)
    return t_values, points


def subdivide_control_polygon(control_points: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide one Bezier curve at parameter t into left/right control polygons."""
    result = de_casteljau_point(control_points, t)
    levels = result.levels

    left = np.array([level[0] for level in levels], dtype=float)
    right = np.array([level[-1] for level in reversed(levels)], dtype=float)
    return left, right


def estimate_polyline_length(points: np.ndarray) -> float:
    """Estimate curve length from sampled polyline."""
    if points.shape[0] < 2:
        return 0.0
    segments = points[1:] - points[:-1]
    return float(np.linalg.norm(segments, axis=1).sum())


def format_point(p: np.ndarray, precision: int = 6) -> str:
    """Pretty formatter for vectors."""
    return "[" + ", ".join(f"{x:.{precision}f}" for x in p) + "]"


def run_demo() -> None:
    """Run deterministic de Casteljau MVP examples without interactive input."""
    control_points = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 3.0],
            [4.0, 0.0],
        ],
        dtype=float,
    )

    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    num_samples = 21
    subdivide_t = 0.5

    print("=" * 84)
    print("de Casteljau Algorithm MVP (Bezier curve evaluation + validation)")
    print("=" * 84)
    print(f"degree = {control_points.shape[0] - 1}, dimension = {control_points.shape[1]}")
    print("control_points:")
    for i, p in enumerate(control_points):
        print(f"  P{i} = {format_point(p)}")
    print()

    print("[Single-point evaluation consistency: de Casteljau vs Bernstein]")
    for t in t_values:
        casteljau = de_casteljau_point(control_points, t)
        bernstein = bernstein_point(control_points, t)
        err = float(np.linalg.norm(casteljau.point - bernstein))
        print(f"t={t:>4.2f} | de_casteljau={format_point(casteljau.point)} | "
              f"bernstein={format_point(bernstein)} | l2_error={err:.3e}")

    print()
    t_grid, sampled_points = sample_bezier_curve(control_points, num_samples=num_samples)
    polyline_len = estimate_polyline_length(sampled_points)
    print(f"[Curve sampling] num_samples={num_samples}, approx_polyline_length={polyline_len:.6f}")
    print("sampled_points_head:")
    for i in range(min(5, num_samples)):
        print(f"  t={t_grid[i]:.2f} -> {format_point(sampled_points[i])}")
    print("sampled_points_tail:")
    for i in range(max(0, num_samples - 3), num_samples):
        print(f"  t={t_grid[i]:.2f} -> {format_point(sampled_points[i])}")

    print()
    left_cp, right_cp = subdivide_control_polygon(control_points, t=subdivide_t)
    print(f"[Subdivision at t={subdivide_t:.2f}]")
    print("left_subcurve_control_points:")
    for i, p in enumerate(left_cp):
        print(f"  L{i} = {format_point(p)}")
    print("right_subcurve_control_points:")
    for i, p in enumerate(right_cp):
        print(f"  R{i} = {format_point(p)}")

    # Simple convex-hull proxy check via axis-aligned bounding box.
    lo = control_points.min(axis=0)
    hi = control_points.max(axis=0)
    inside_box = bool(np.all((sampled_points >= lo) & (sampled_points <= hi)))
    print()
    print("[Sanity check]")
    print(f"all sampled points inside control-point bounding box: {inside_box}")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
