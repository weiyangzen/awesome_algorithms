"""Ear clipping polygon triangulation MVP.

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

PointArray = np.ndarray
Triangle = Tuple[int, int, int]


@dataclass
class TriangulationResult:
    points: PointArray
    triangles: List[Triangle]


def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Twice signed area of triangle (a, b, c)."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def polygon_signed_area(points: PointArray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return 0.5 * area2


def check_polygon(points: Sequence[Sequence[float]]) -> PointArray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("polygon must have shape (n, 2)")
    if not np.isfinite(arr).all():
        raise ValueError("polygon contains non-finite values")
    return arr


def remove_closing_duplicate(points: PointArray, eps: float) -> PointArray:
    if points.shape[0] >= 2 and np.linalg.norm(points[0] - points[-1]) <= eps:
        return points[:-1].copy()
    return points.copy()


def remove_adjacent_duplicates(points: PointArray, eps: float) -> PointArray:
    if points.shape[0] == 0:
        return points.copy()

    kept = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - kept[-1]) > eps:
            kept.append(p)

    arr = np.asarray(kept, dtype=float)
    if arr.shape[0] >= 2 and np.linalg.norm(arr[0] - arr[-1]) <= eps:
        arr = arr[:-1]
    return arr


def on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float) -> bool:
    if abs(orientation(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray, eps: float) -> bool:
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True

    if abs(o1) <= eps and on_segment(p1, p2, q1, eps):
        return True
    if abs(o2) <= eps and on_segment(p1, p2, q2, eps):
        return True
    if abs(o3) <= eps and on_segment(q1, q2, p1, eps):
        return True
    if abs(o4) <= eps and on_segment(q1, q2, p2, eps):
        return True

    return False


def is_simple_polygon(points: PointArray, eps: float) -> bool:
    n = points.shape[0]
    if n < 3:
        return False

    for i in range(n):
        a1 = points[i]
        a2 = points[(i + 1) % n]
        for j in range(i + 1, n):
            if i == j:
                continue
            if (i + 1) % n == j:
                continue
            if i == (j + 1) % n:
                continue

            b1 = points[j]
            b2 = points[(j + 1) % n]
            if segments_intersect(a1, a2, b1, b2, eps):
                return False

    return True


def remove_collinear_vertices(points: PointArray, eps: float) -> PointArray:
    if points.shape[0] <= 3:
        return points

    changed = True
    arr = points.copy()

    while changed and arr.shape[0] > 3:
        changed = False
        keep_mask = np.ones(arr.shape[0], dtype=bool)
        for i in range(arr.shape[0]):
            prev = arr[(i - 1) % arr.shape[0]]
            curr = arr[i]
            nxt = arr[(i + 1) % arr.shape[0]]
            if abs(orientation(prev, curr, nxt)) <= eps:
                keep_mask[i] = False
                changed = True
        if changed:
            arr = arr[keep_mask]

    return arr


def normalize_polygon(points: Sequence[Sequence[float]], eps: float = 1e-12) -> PointArray:
    arr = check_polygon(points)
    arr = remove_closing_duplicate(arr, eps)
    arr = remove_adjacent_duplicates(arr, eps)
    arr = remove_collinear_vertices(arr, eps)

    if arr.shape[0] < 3:
        raise ValueError("polygon has fewer than 3 effective vertices")

    area = polygon_signed_area(arr)
    if abs(area) <= eps:
        raise ValueError("polygon area is too close to zero")

    if area < 0.0:
        arr = arr[::-1].copy()

    if not is_simple_polygon(arr, eps):
        raise ValueError("polygon is not simple (self-intersection detected)")

    return arr


def point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float) -> bool:
    """Inclusive point-in-triangle for a CCW triangle."""
    o1 = orientation(a, b, p)
    o2 = orientation(b, c, p)
    o3 = orientation(c, a, p)
    return (o1 >= -eps) and (o2 >= -eps) and (o3 >= -eps)


def is_ear(prev_i: int, curr_i: int, next_i: int, active: List[int], points: PointArray, eps: float) -> bool:
    a, b, c = points[prev_i], points[curr_i], points[next_i]
    if orientation(a, b, c) <= eps:
        return False

    for idx in active:
        if idx in (prev_i, curr_i, next_i):
            continue
        if point_in_triangle(points[idx], a, b, c, eps):
            return False

    return True


def ear_clipping_triangulation(points: Sequence[Sequence[float]], eps: float = 1e-12) -> TriangulationResult:
    poly = normalize_polygon(points, eps=eps)
    n = poly.shape[0]

    if n == 3:
        return TriangulationResult(points=poly, triangles=[(0, 1, 2)])

    active: List[int] = list(range(n))
    triangles: List[Triangle] = []

    while len(active) > 3:
        ear_found = False
        m = len(active)

        for i in range(m):
            prev_i = active[(i - 1) % m]
            curr_i = active[i]
            next_i = active[(i + 1) % m]

            if is_ear(prev_i, curr_i, next_i, active, poly, eps):
                triangles.append((prev_i, curr_i, next_i))
                del active[i]
                ear_found = True
                break

        if not ear_found:
            raise ValueError("failed to find an ear; polygon may be invalid or numerically unstable")

    triangles.append((active[0], active[1], active[2]))
    return TriangulationResult(points=poly, triangles=triangles)


def triangle_area_abs(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return abs(orientation(a, b, c)) * 0.5


def triangulation_area(points: PointArray, triangles: Iterable[Triangle]) -> float:
    return float(sum(triangle_area_abs(points[i], points[j], points[k]) for i, j, k in triangles))


def summarize_case(name: str, polygon: Sequence[Sequence[float]]) -> None:
    result = ear_clipping_triangulation(polygon)
    pts = result.points
    tris = result.triangles

    poly_area = abs(polygon_signed_area(pts))
    tri_area = triangulation_area(pts, tris)
    expected_triangle_count = pts.shape[0] - 2
    area_ok = abs(poly_area - tri_area) <= 1e-9
    all_ccw = all(orientation(pts[i], pts[j], pts[k]) > 0 for i, j, k in tris)

    print(f"=== {name} ===")
    print(f"vertices: {pts.shape[0]}")
    print(f"triangles: {len(tris)} (expected: {expected_triangle_count})")
    print(f"all triangles CCW: {all_ccw}")
    print(f"polygon area: {poly_area:.8f}")
    print(f"sum triangle area: {tri_area:.8f}")
    print(f"area consistency: {area_ok}")
    print(f"triangle preview: {tris[: min(8, len(tris))]}")
    print()


def regular_polygon(n: int, radius: float = 1.0, center: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)
    cx, cy = center
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return np.column_stack([x, y])


def main() -> None:
    case1 = regular_polygon(6, radius=1.0)

    case2 = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.8, 1.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [0.8, 1.0],
        ],
        dtype=float,
    )

    case3 = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [1.8, 1.0],
            [1.8, 2.0],
            [3.0, 2.0],
            [3.0, 3.0],
            [0.0, 3.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    summarize_case("Convex hexagon", case1)
    summarize_case("Concave arrow", case2)
    summarize_case("Concave polygon with closing duplicate", case3)


if __name__ == "__main__":
    main()
