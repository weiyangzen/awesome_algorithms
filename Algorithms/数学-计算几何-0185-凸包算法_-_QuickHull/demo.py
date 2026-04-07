"""Minimal runnable MVP for QuickHull convex hull (MATH-0185).

This demo implements QuickHull from scratch with NumPy and validates it
against a monotonic-chain hull implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np


@dataclass
class HullSummary:
    """Convex hull metrics."""

    vertices: np.ndarray
    area: float
    perimeter: float


def unique_points(points: np.ndarray) -> np.ndarray:
    """Return unique 2D points in float64."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    return np.unique(arr, axis=0)


def cross_values(a: np.ndarray, b: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Compute cross((b-a), (p-a)) for each point p in pts."""
    ab = b - a
    ap = pts - a
    return ab[0] * ap[:, 1] - ab[1] * ap[:, 0]


def polygon_signed_area(vertices: np.ndarray) -> float:
    """Signed area; positive when vertices are in CCW order."""
    m = vertices.shape[0]
    if m < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_perimeter(vertices: np.ndarray) -> float:
    """Perimeter of polygon-like hull representation."""
    m = vertices.shape[0]
    if m == 0:
        return 0.0
    if m == 1:
        return 0.0
    if m == 2:
        # Degenerate hull (line segment): count both directions.
        return 2.0 * float(np.linalg.norm(vertices[1] - vertices[0]))
    nxt = np.roll(vertices, -1, axis=0)
    return float(np.sum(np.linalg.norm(nxt - vertices, axis=1)))


def rotate_start_lexicographic(vertices: np.ndarray) -> np.ndarray:
    """Rotate polygon so the lexicographically smallest point is first."""
    if vertices.shape[0] == 0:
        return vertices
    start = int(np.lexsort((vertices[:, 1], vertices[:, 0]))[0])
    return np.roll(vertices, -start, axis=0)


def quickhull(points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute convex hull vertices with QuickHull.

    Returns vertices in counter-clockwise order, without repeating the first point.
    """
    pts = unique_points(points)
    n = pts.shape[0]

    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if n == 1:
        return pts.copy()

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    left_idx = int(order[0])
    right_idx = int(order[-1])

    if left_idx == right_idx:
        return pts[[left_idx]].copy()

    all_idx = np.arange(n, dtype=int)

    top_idx = all_idx[cross_values(pts[left_idx], pts[right_idx], pts) > eps]
    bottom_idx = all_idx[cross_values(pts[right_idx], pts[left_idx], pts) > eps]

    def extend_hull(a_idx: int, b_idx: int, candidates: np.ndarray) -> List[int]:
        if candidates.size == 0:
            return []

        # Point farthest from line AB by signed area magnitude on the chosen side.
        cross_ab = cross_values(pts[a_idx], pts[b_idx], pts[candidates])
        farthest_pos = int(np.argmax(cross_ab))
        p_idx = int(candidates[farthest_pos])

        left_ap = candidates[cross_values(pts[a_idx], pts[p_idx], pts[candidates]) > eps]
        left_pb = candidates[cross_values(pts[p_idx], pts[b_idx], pts[candidates]) > eps]

        return (
            extend_hull(a_idx, p_idx, left_ap)
            + [p_idx]
            + extend_hull(p_idx, b_idx, left_pb)
        )

    hull_idx: List[int] = [left_idx]
    hull_idx.extend(extend_hull(left_idx, right_idx, top_idx))
    hull_idx.append(right_idx)
    hull_idx.extend(extend_hull(right_idx, left_idx, bottom_idx))

    # Remove accidental duplicates while preserving first-seen order.
    ordered_idx: List[int] = []
    seen: Set[int] = set()
    for idx in hull_idx:
        if idx not in seen:
            ordered_idx.append(idx)
            seen.add(idx)

    hull = pts[np.asarray(ordered_idx, dtype=int)]

    if hull.shape[0] >= 3 and polygon_signed_area(hull) < 0.0:
        hull = hull[::-1]
    return rotate_start_lexicographic(hull)


def monotonic_chain_hull(points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Reference hull via monotonic chain (for correctness cross-check)."""
    pts = unique_points(points)
    n = pts.shape[0]

    if n <= 1:
        return pts.copy()

    sorted_pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        oa = a - o
        ob = b - o
        return float(oa[0] * ob[1] - oa[1] * ob[0])

    lower: List[np.ndarray] = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= eps:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in sorted_pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= eps:
            upper.pop()
        upper.append(p)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if hull.shape[0] >= 3 and polygon_signed_area(hull) < 0.0:
        hull = hull[::-1]
    return rotate_start_lexicographic(hull)


def hull_signature(vertices: np.ndarray, decimals: int = 10) -> Set[Tuple[float, float]]:
    """Order-insensitive signature for hull vertex-set comparison."""
    if vertices.size == 0:
        return set()
    rounded = np.round(vertices, decimals=decimals)
    return {tuple(row.tolist()) for row in rounded}


def summarize_hull(vertices: np.ndarray) -> HullSummary:
    """Compute area/perimeter summary."""
    area = abs(polygon_signed_area(vertices))
    perimeter = polygon_perimeter(vertices)
    return HullSummary(vertices=vertices, area=area, perimeter=perimeter)


def build_demo_points(seed: int = 185) -> np.ndarray:
    """Create a deterministic dataset with noise, duplicates, and outliers."""
    rng = np.random.default_rng(seed)

    cloud = rng.normal(loc=[0.0, 0.0], scale=[1.8, 1.1], size=(260, 2))

    extreme = np.array(
        [
            [-7.0, -1.2],
            [-5.8, 3.9],
            [-1.0, 5.7],
            [3.5, 5.1],
            [6.7, 1.6],
            [5.4, -3.8],
            [0.2, -5.5],
            [-4.8, -4.0],
        ],
        dtype=np.float64,
    )

    duplicates = cloud[:8]
    collinear = np.column_stack([np.linspace(-2.0, 2.0, 11), np.zeros(11)])

    return np.vstack([cloud, extreme, duplicates, collinear])


def main() -> None:
    print("QuickHull MVP (MATH-0185)")
    print("=" * 64)

    points = build_demo_points(seed=185)
    unique_count = unique_points(points).shape[0]

    hull_qh = quickhull(points)
    hull_ref = monotonic_chain_hull(points)

    sig_qh = hull_signature(hull_qh)
    sig_ref = hull_signature(hull_ref)
    if sig_qh != sig_ref:
        raise RuntimeError("QuickHull vertex set does not match reference monotonic-chain hull")

    summary = summarize_hull(hull_qh)

    print(f"raw points: {points.shape[0]}")
    print(f"unique points: {unique_count}")
    print(f"hull vertex count: {summary.vertices.shape[0]}")
    print(f"hull area: {summary.area:.6f}")
    print(f"hull perimeter: {summary.perimeter:.6f}")
    print("first hull vertices (x, y):")

    show_n = min(10, summary.vertices.shape[0])
    for i in range(show_n):
        x, y = summary.vertices[i]
        print(f"  {i:02d}: ({x: .6f}, {y: .6f})")

    if summary.vertices.shape[0] < 3:
        raise RuntimeError("Unexpected degenerate hull for the demo dataset")
    if summary.area <= 1e-6:
        raise RuntimeError("Hull area is too small; demo data may be degenerate")

    print("=" * 64)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
