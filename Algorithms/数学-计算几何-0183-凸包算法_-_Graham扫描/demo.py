"""Graham scan convex hull MVP (MATH-0183).

Run:
    python3 demo.py

The implementation is from scratch with NumPy and does not call black-box
convex hull APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass
class HullResult:
    unique_points: np.ndarray
    hull: np.ndarray
    removed_duplicates: int


def check_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Validate and convert input to shape (n, 2)."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if not np.isfinite(arr).all():
        raise ValueError("points contain non-finite values")
    return arr


def deduplicate_points(points: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, int]:
    """Remove near-duplicate points by epsilon-quantized hashing."""
    if points.size == 0:
        return points.copy(), 0

    scale = 1.0 / eps
    seen = set()
    unique: List[np.ndarray] = []
    for p in points:
        key = (int(np.round(p[0] * scale)), int(np.round(p[1] * scale)))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    unique_arr = np.asarray(unique, dtype=float)
    removed = int(points.shape[0] - unique_arr.shape[0])
    return unique_arr, removed


def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product (OA x OB)."""
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def is_all_collinear(points: np.ndarray, eps: float = 1e-12) -> bool:
    """Check whether all points are collinear."""
    n = points.shape[0]
    if n <= 2:
        return True

    p0 = points[0]
    k = None
    for i in range(1, n):
        if np.linalg.norm(points[i] - p0) > eps:
            k = i
            break
    if k is None:
        return True

    p1 = points[k]
    for i in range(n):
        if abs(cross(p0, p1, points[i])) > eps:
            return False
    return True


def pivot_index(points: np.ndarray) -> int:
    """Pick pivot with minimum y, tie-broken by x."""
    order = np.lexsort((points[:, 0], points[:, 1]))
    return int(order[0])


def sort_by_polar(points: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Sort points by polar angle around pivot, then by distance to pivot."""
    vec = points - p0
    angles = np.arctan2(vec[:, 1], vec[:, 0])
    d2 = np.einsum("nd,nd->n", vec, vec)
    order = np.lexsort((d2, angles))
    return points[order]


def collinear_hull_endpoints(points: np.ndarray) -> np.ndarray:
    """Return two extreme endpoints for a collinear point set."""
    order = np.lexsort((points[:, 1], points[:, 0]))
    left = points[order[0]]
    right = points[order[-1]]
    if np.allclose(left, right):
        return left.reshape(1, 2)
    return np.vstack([left, right])


def graham_scan(points: Sequence[Sequence[float]], eps: float = 1e-12) -> HullResult:
    """Compute convex hull vertices in CCW order using Graham scan.

    Returns hull without repeating the first point at the end.
    """
    arr = check_points(points)
    pts, removed_duplicates = deduplicate_points(arr, eps=eps)

    n = pts.shape[0]
    if n == 0:
        return HullResult(unique_points=pts, hull=np.empty((0, 2), dtype=float), removed_duplicates=removed_duplicates)
    if n == 1:
        return HullResult(unique_points=pts, hull=pts.copy(), removed_duplicates=removed_duplicates)
    if is_all_collinear(pts, eps=eps):
        return HullResult(
            unique_points=pts,
            hull=collinear_hull_endpoints(pts),
            removed_duplicates=removed_duplicates,
        )

    p0_idx = pivot_index(pts)
    p0 = pts[p0_idx]

    mask = np.ones(n, dtype=bool)
    mask[p0_idx] = False
    others = pts[mask]
    ordered = sort_by_polar(others, p0)

    stack: List[np.ndarray] = [p0, ordered[0]]

    for p in ordered[1:]:
        while len(stack) >= 2 and cross(stack[-2], stack[-1], p) <= eps:
            stack.pop()
        stack.append(p)

    hull = np.vstack(stack)
    return HullResult(unique_points=pts, hull=hull, removed_duplicates=removed_duplicates)


def polygon_area2(poly: np.ndarray) -> float:
    """Twice polygon signed area (positive if CCW)."""
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_strictly_convex_ccw(poly: np.ndarray, eps: float = 1e-12) -> bool:
    """Check strict convexity and CCW for polygon with >= 3 vertices."""
    n = len(poly)
    if n < 3:
        return True
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        c = poly[(i + 2) % n]
        if cross(a, b, c) <= eps:
            return False
    return polygon_area2(poly) > 0.0


def all_points_inside_or_on_hull(points: np.ndarray, hull: np.ndarray, eps: float = 1e-10) -> bool:
    """Check if every point lies inside or on the CCW convex hull."""
    n = len(hull)
    if n == 0:
        return len(points) == 0
    if n == 1:
        return np.all(np.linalg.norm(points - hull[0], axis=1) <= eps)
    if n == 2:
        a, b = hull[0], hull[1]
        ab = b - a
        ab2 = float(np.dot(ab, ab)) + eps
        for p in points:
            ap = p - a
            if abs(ab[0] * ap[1] - ab[1] * ap[0]) > 1e-8:
                return False
            t = float(np.dot(ap, ab) / ab2)
            if t < -1e-8 or t > 1.0 + 1e-8:
                return False
        return True

    for p in points:
        for i in range(n):
            a = hull[i]
            b = hull[(i + 1) % n]
            if cross(a, b, p) < -eps:
                return False
    return True


def summarize_case(name: str, points: np.ndarray) -> None:
    result = graham_scan(points)
    hull = result.hull

    area = 0.5 * abs(polygon_area2(hull))
    convex_ok = is_strictly_convex_ccw(hull)
    cover_ok = all_points_inside_or_on_hull(result.unique_points, hull)

    print(f"=== {name} ===")
    print(f"input points: {len(points)}")
    print(
        "unique points: "
        f"{len(result.unique_points)} (removed duplicates: {result.removed_duplicates})"
    )
    print(f"hull vertices: {len(hull)}")
    print(f"hull area: {area:.6f}")
    print(f"strictly convex + CCW: {convex_ok}")
    print(f"all points covered by hull: {cover_ok}")
    print(f"hull preview:\n{np.array2string(hull, precision=4)}")
    print()

    if len(hull) >= 3:
        if not convex_ok:
            raise RuntimeError("hull is not strictly convex CCW")
        if not cover_ok:
            raise RuntimeError("hull does not cover all input points")


def main() -> None:
    rng = np.random.default_rng(183)

    case1 = rng.normal(loc=(0.0, 0.0), scale=(1.0, 0.7), size=(28, 2))

    case2 = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [1.5, 1.2],
            [0.4, 1.7],
            [0.0, 0.0],
            [2.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    t = np.linspace(-2.0, 2.0, 15)
    case3 = np.column_stack([t, 0.5 * t + 1.0])

    summarize_case("Gaussian cloud", case1)
    summarize_case("Square + interior + duplicates", case2)
    summarize_case("Collinear points", case3)

    print("Run completed successfully.")


if __name__ == "__main__":
    main()
