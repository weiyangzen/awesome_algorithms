"""Minimal runnable MVP for Convex Hull (Jarvis March / Gift Wrapping)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class HullResult:
    unique_points: np.ndarray
    hull_indices: List[int]
    hull_points: np.ndarray
    removed_duplicates: int


def check_points(points: np.ndarray) -> np.ndarray:
    """Validate that points is a finite n x 2 numeric array."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if not np.isfinite(arr).all():
        raise ValueError("points must contain only finite values")
    return arr


def deduplicate_points(points: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, int]:
    """Deduplicate points with an epsilon-aware quantization key."""
    scale = max(eps, 1e-15)
    table: dict[tuple[int, int], np.ndarray] = {}
    for p in points:
        key = (int(np.round(p[0] / scale)), int(np.round(p[1] / scale)))
        if key not in table:
            table[key] = p
    unique = np.array(list(table.values()), dtype=float)
    return unique, len(points) - len(unique)


def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """2D cross product: (b-a) x (c-a). Positive means a->b->c is CCW."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def squared_distance(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(d[0] * d[0] + d[1] * d[1])


def are_all_collinear(points: np.ndarray, eps: float = 1e-12) -> bool:
    if len(points) <= 2:
        return True
    a = points[0]
    b = points[1]
    for i in range(2, len(points)):
        if abs(orientation(a, b, points[i])) > eps:
            return False
    return True


def jarvis_march(points: np.ndarray, eps: float = 1e-12) -> HullResult:
    """Compute convex hull by Jarvis march.

    Returns hull vertices in CCW order when hull has at least 3 vertices.
    Degenerate cases:
    - n == 0: empty hull
    - n == 1: single point hull
    - all collinear: two endpoints hull
    """
    arr = check_points(points)
    unique_points, removed = deduplicate_points(arr, eps=eps)
    n = len(unique_points)

    if n == 0:
        return HullResult(unique_points, [], np.empty((0, 2), dtype=float), removed)
    if n == 1:
        return HullResult(unique_points, [0], unique_points[[0]], removed)

    if are_all_collinear(unique_points, eps=eps):
        order = sorted(range(n), key=lambda i: (unique_points[i, 0], unique_points[i, 1]))
        left = order[0]
        right = order[-1]
        if left == right:
            hull_idx = [left]
        else:
            hull_idx = [left, right]
        return HullResult(unique_points, hull_idx, unique_points[hull_idx], removed)

    # Start from leftmost point (tie-breaker: lower y).
    start = min(range(n), key=lambda i: (unique_points[i, 0], unique_points[i, 1]))

    hull_indices: List[int] = []
    current = start

    while True:
        hull_indices.append(current)

        candidate = None
        for i in range(n):
            if i != current:
                candidate = i
                break
        if candidate is None:
            break

        for i in range(n):
            if i == current or i == candidate:
                continue
            turn = orientation(unique_points[current], unique_points[candidate], unique_points[i])
            # Pick the most clockwise candidate from the current point so that
            # traversal from a leftmost start becomes CCW around the hull.
            if turn < -eps:
                candidate = i
            elif abs(turn) <= eps:
                # Collinear: keep the farthest point to preserve extreme vertex.
                if squared_distance(unique_points[current], unique_points[i]) > squared_distance(
                    unique_points[current], unique_points[candidate]
                ):
                    candidate = i

        current = candidate
        if current == start:
            break

    hull = unique_points[hull_indices]
    return HullResult(unique_points, hull_indices, hull, removed)


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> bool:
    if abs(orientation(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def verify_hull(result: HullResult, eps: float = 1e-10) -> dict[str, bool]:
    hull = result.hull_points
    pts = result.unique_points

    if len(hull) <= 1:
        return {"convex": True, "contains_all_points": True, "ccw": True}

    if len(hull) == 2:
        a, b = hull[0], hull[1]
        all_on_segment = all(point_on_segment(p, a, b, eps=eps) for p in pts)
        return {"convex": True, "contains_all_points": all_on_segment, "ccw": True}

    convex_ok = True
    m = len(hull)
    for i in range(m):
        a = hull[i]
        b = hull[(i + 1) % m]
        c = hull[(i + 2) % m]
        if orientation(a, b, c) <= -eps:
            convex_ok = False
            break

    inside_ok = True
    # For CCW convex polygon, each point must be to the left of each edge.
    for p in pts:
        for i in range(m):
            a = hull[i]
            b = hull[(i + 1) % m]
            if orientation(a, b, p) < -eps:
                inside_ok = False
                break
        if not inside_ok:
            break

    area = polygon_area(hull)
    ccw_ok = area > eps
    return {"convex": convex_ok, "contains_all_points": inside_ok, "ccw": ccw_ok}


def summarize_case(name: str, points: np.ndarray) -> None:
    result = jarvis_march(points)
    checks = verify_hull(result)

    print(f"=== {name} ===")
    print(f"input points: {len(points)}")
    print(
        "unique points: "
        f"{len(result.unique_points)} (removed duplicates: {result.removed_duplicates})"
    )
    print(f"hull vertex count: {len(result.hull_indices)}")
    print(f"hull indices: {result.hull_indices}")
    print(f"hull points: {np.array2string(result.hull_points, precision=3)}")
    print(
        "checks: "
        f"convex={checks['convex']}, "
        f"contains_all_points={checks['contains_all_points']}, "
        f"ccw={checks['ccw']}"
    )
    print()


def main() -> None:
    rng = np.random.default_rng(2026)

    case_random = rng.uniform(-2.0, 2.0, size=(20, 2))

    # Mostly rectangular cloud with a few interior points.
    case_rect = np.array(
        [
            [-3.0, -1.0],
            [-2.5, 1.8],
            [0.0, 2.2],
            [2.7, 1.7],
            [3.2, -0.8],
            [1.8, -2.1],
            [-1.4, -2.0],
            [-0.6, 0.3],
            [1.0, 0.0],
            [0.5, -0.5],
            [-1.0, 1.1],
        ],
        dtype=float,
    )

    # Degenerate sample with duplicates and collinear points.
    case_degenerate = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [3.0, 3.0],
            [2.0, 2.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    summarize_case("Uniform random", case_random)
    summarize_case("Skewed rectangle cloud", case_rect)
    summarize_case("Degenerate collinear + duplicates", case_degenerate)


if __name__ == "__main__":
    main()
