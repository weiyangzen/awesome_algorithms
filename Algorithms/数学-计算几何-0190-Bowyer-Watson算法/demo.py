"""Bowyer-Watson incremental Delaunay triangulation MVP.

This script is self-contained and runnable with:
    python3 demo.py

It implements the triangulation logic directly instead of calling a black-box
third-party Delaunay API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Triangle = Tuple[int, int, int]
Edge = Tuple[int, int]


@dataclass
class TriangulationResult:
    unique_points: np.ndarray
    triangles: List[Triangle]
    removed_duplicates: int


def check_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must be an array-like object with shape (n, 2)")
    if not np.isfinite(arr).all():
        raise ValueError("points contain non-finite values")
    return arr


def deduplicate_points(points: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, int]:
    if points.size == 0:
        return points.copy(), 0

    scale = 1.0 / eps
    seen: Dict[Tuple[int, int], int] = {}
    unique: List[np.ndarray] = []

    for p in points:
        key = (int(np.round(p[0] * scale)), int(np.round(p[1] * scale)))
        if key not in seen:
            seen[key] = len(unique)
            unique.append(p)

    unique_arr = np.asarray(unique, dtype=float)
    removed = int(points.shape[0] - unique_arr.shape[0])
    return unique_arr, removed


def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def are_all_collinear(points: np.ndarray, eps: float = 1e-12) -> bool:
    n = points.shape[0]
    if n < 3:
        return True

    a = points[0]
    pivot = None
    for i in range(1, n):
        if np.linalg.norm(points[i] - a) > eps:
            pivot = points[i]
            break
    if pivot is None:
        return True

    for i in range(1, n):
        if abs(orientation(a, pivot, points[i])) > eps:
            return False
    return True


def canonical_edge(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def make_ccw_triangle(i: int, j: int, k: int, pts: np.ndarray, eps: float = 1e-14) -> Triangle:
    a, b, c = pts[i], pts[j], pts[k]
    cross = orientation(a, b, c)
    if abs(cross) <= eps:
        raise ValueError("degenerate triangle with near-zero area")
    if cross > 0.0:
        return (i, j, k)
    return (i, k, j)


def build_super_triangle(points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if span <= 0.0:
        span = 1.0

    # A large triangle that encloses the whole bounding box.
    p1 = np.array([center[0] - 20.0 * span, center[1] - 1.0 * span], dtype=float)
    p2 = np.array([center[0], center[1] + 20.0 * span], dtype=float)
    p3 = np.array([center[0] + 20.0 * span, center[1] - 1.0 * span], dtype=float)
    return np.vstack([p1, p2, p3])


def circumcircle_contains(triangle: Triangle, p_idx: int, pts: np.ndarray, eps: float = 1e-12) -> bool:
    ia, ib, ic = triangle
    a, b, c = pts[ia], pts[ib], pts[ic]
    p = pts[p_idx]

    ax, ay = a[0] - p[0], a[1] - p[1]
    bx, by = b[0] - p[0], b[1] - p[1]
    cx, cy = c[0] - p[0], c[1] - p[1]

    det = (
        (ax * ax + ay * ay) * (bx * cy - by * cx)
        - (bx * bx + by * by) * (ax * cy - ay * cx)
        + (cx * cx + cy * cy) * (ax * by - ay * bx)
    )

    # Triangles are stored CCW, so det > 0 means "inside circumcircle".
    return det > eps


def bowyer_watson(points: Sequence[Sequence[float]], eps: float = 1e-12) -> TriangulationResult:
    pts = check_points(points)
    pts, removed_duplicates = deduplicate_points(pts, eps=eps)

    n = pts.shape[0]
    if n < 3 or are_all_collinear(pts, eps=eps):
        return TriangulationResult(unique_points=pts, triangles=[], removed_duplicates=removed_duplicates)

    super_tri = build_super_triangle(pts)
    aug = np.vstack([pts, super_tri])

    s0, s1, s2 = n, n + 1, n + 2
    triangulation = {make_ccw_triangle(s0, s1, s2, aug)}

    for p_idx in range(n):
        bad_triangles: List[Triangle] = [
            tri for tri in triangulation if circumcircle_contains(tri, p_idx, aug, eps=eps)
        ]

        edge_counter: Dict[Edge, int] = {}
        for tri in bad_triangles:
            i, j, k = tri
            for e in (canonical_edge(i, j), canonical_edge(j, k), canonical_edge(k, i)):
                edge_counter[e] = edge_counter.get(e, 0) + 1

        for tri in bad_triangles:
            triangulation.remove(tri)

        boundary = [e for e, cnt in edge_counter.items() if cnt == 1]

        for u, v in boundary:
            try:
                tri = make_ccw_triangle(u, v, p_idx, aug)
            except ValueError:
                # Happens in degenerate near-collinear local configurations.
                continue
            triangulation.add(tri)

    final_tris = [tri for tri in triangulation if all(v < n for v in tri)]
    final_tris.sort()

    return TriangulationResult(
        unique_points=pts,
        triangles=final_tris,
        removed_duplicates=removed_duplicates,
    )


def triangle_area2(tri: Triangle, pts: np.ndarray) -> float:
    i, j, k = tri
    return orientation(pts[i], pts[j], pts[k])


def verify_delaunay_empty_circumcircle(
    points: np.ndarray, triangles: Iterable[Triangle], eps: float = 1e-10
) -> bool:
    tris = list(triangles)
    for tri in tris:
        tri_set = set(tri)
        for p_idx in range(points.shape[0]):
            if p_idx in tri_set:
                continue
            if circumcircle_contains(tri, p_idx, points, eps=eps):
                return False
    return True


def summarize_case(name: str, points: np.ndarray) -> None:
    result = bowyer_watson(points)
    pts = result.unique_points
    tris = result.triangles

    all_ccw = all(triangle_area2(t, pts) > 0.0 for t in tris)
    delaunay_ok = verify_delaunay_empty_circumcircle(pts, tris)

    print(f"=== {name} ===")
    print(f"input points: {len(points)}")
    print(f"unique points: {len(pts)} (removed duplicates: {result.removed_duplicates})")
    print(f"triangles: {len(tris)}")
    print(f"all triangles CCW: {all_ccw}")
    print(f"empty circumcircle check: {delaunay_ok}")

    preview = tris[: min(8, len(tris))]
    print(f"triangle preview: {preview}")
    print()


def main() -> None:
    rng = np.random.default_rng(190)

    case1 = rng.random((18, 2))

    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 3))
    case2_base = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    noise = rng.normal(loc=0.0, scale=0.015, size=case2_base.shape)
    case2 = case2_base + noise

    case3 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.9],
            [0.3, 0.4],
            [0.7, 0.45],
            [0.2, 0.75],
            [0.8, 0.8],
            [0.3, 0.4],  # duplicate
            [1.0, 0.0],  # duplicate
        ],
        dtype=float,
    )

    summarize_case("Uniform random", case1)
    summarize_case("Noisy grid", case2)
    summarize_case("With duplicates", case3)


if __name__ == "__main__":
    main()
