"""Minimal runnable MVP for Minkowski sum of convex polygons (MATH-0210)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


EPS = 1e-9


def cross_vec(a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product of vectors a x b."""
    return float(a[0] * b[1] - a[1] * b[0])


def polygon_area(poly: np.ndarray) -> float:
    """Signed area; positive means counter-clockwise."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def normalize_polygon(poly: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Remove duplicate points and normalize to CCW orientation."""
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("polygon must be an Nx2 array")

    p = np.asarray(poly, dtype=np.float64)
    if p.shape[0] < 3:
        raise ValueError("polygon must contain at least 3 points")

    # Drop repeated closing vertex if present.
    if np.linalg.norm(p[0] - p[-1]) <= eps:
        p = p[:-1]

    # Remove adjacent duplicates.
    cleaned: List[np.ndarray] = []
    for pt in p:
        if not cleaned or np.linalg.norm(pt - cleaned[-1]) > eps:
            cleaned.append(pt)
    if len(cleaned) >= 2 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= eps:
        cleaned.pop()

    if len(cleaned) < 3:
        raise ValueError("polygon degenerates after removing duplicate points")

    p = np.array(cleaned, dtype=np.float64)
    if polygon_area(p) < 0.0:
        p = p[::-1]

    if abs(polygon_area(p)) <= eps:
        raise ValueError("degenerate polygon with near-zero area")

    return p


def rotate_to_lowest_left(poly: np.ndarray) -> np.ndarray:
    """Rotate vertices so that the lowest-then-leftmost point is first."""
    idx = min(range(len(poly)), key=lambda i: (poly[i, 1], poly[i, 0]))
    return np.roll(poly, -idx, axis=0)


def edge_vectors(poly: np.ndarray) -> np.ndarray:
    """Return edge vectors for a cyclic polygon."""
    return np.roll(poly, -1, axis=0) - poly


def clean_polygon(poly: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Remove duplicated and collinear consecutive points."""
    if len(poly) == 0:
        return poly

    points: List[np.ndarray] = []
    for pt in poly:
        if not points or np.linalg.norm(pt - points[-1]) > eps:
            points.append(pt)

    if len(points) >= 2 and np.linalg.norm(points[0] - points[-1]) <= eps:
        points.pop()

    changed = True
    while changed and len(points) >= 3:
        changed = False
        new_points: List[np.ndarray] = []
        n = len(points)
        for i in range(n):
            a = points[i - 1]
            b = points[i]
            c = points[(i + 1) % n]
            if abs(cross_vec(b - a, c - b)) <= eps:
                changed = True
                continue
            new_points.append(b)
        points = new_points

    if len(points) < 3:
        raise ValueError("result polygon became degenerate during cleanup")

    return np.array(points, dtype=np.float64)


def minkowski_sum_convex(poly_a: np.ndarray, poly_b: np.ndarray) -> np.ndarray:
    """Linear-time Minkowski sum for two convex polygons in 2D.

    Args:
        poly_a, poly_b: convex polygons as Nx2 and Mx2 arrays.

    Returns:
        Vertices of the Minkowski sum polygon in CCW order.
    """
    a = rotate_to_lowest_left(normalize_polygon(poly_a))
    b = rotate_to_lowest_left(normalize_polygon(poly_b))

    ea = edge_vectors(a)
    eb = edge_vectors(b)

    n = len(ea)
    m = len(eb)

    i = 0
    j = 0
    result = [a[0] + b[0]]

    while i < n or j < m:
        if i == n:
            step = eb[j]
            j += 1
        elif j == m:
            step = ea[i]
            i += 1
        else:
            c = cross_vec(ea[i], eb[j])
            if c > EPS:
                step = ea[i]
                i += 1
            elif c < -EPS:
                step = eb[j]
                j += 1
            else:
                step = ea[i] + eb[j]
                i += 1
                j += 1

        result.append(result[-1] + step)

    result_arr = np.array(result, dtype=np.float64)
    return clean_polygon(result_arr)


def convex_hull_monotonic_chain(points: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Convex hull using monotonic chain; output CCW without duplicate end."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        raise ValueError("at least 3 points are required to build a hull")

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    sorted_pts = pts[order]

    unique_pts: List[Tuple[float, float]] = []
    for p in sorted_pts:
        t = (float(p[0]), float(p[1]))
        if not unique_pts or (abs(t[0] - unique_pts[-1][0]) > eps or abs(t[1] - unique_pts[-1][1]) > eps):
            unique_pts.append(t)

    if len(unique_pts) < 3:
        raise ValueError("points are nearly identical; hull is undefined")

    up = [np.array(p, dtype=np.float64) for p in unique_pts]

    lower: List[np.ndarray] = []
    for p in up:
        while len(lower) >= 2 and cross_vec(lower[-1] - lower[-2], p - lower[-1]) <= eps:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in reversed(up):
        while len(upper) >= 2 and cross_vec(upper[-1] - upper[-2], p - upper[-1]) <= eps:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float64)


def bruteforce_minkowski_hull(poly_a: np.ndarray, poly_b: np.ndarray) -> np.ndarray:
    """Reference result via pairwise sums + convex hull."""
    sums = np.array([pa + pb for pa in poly_a for pb in poly_b], dtype=np.float64)
    hull = convex_hull_monotonic_chain(sums)
    hull = normalize_polygon(hull)
    return rotate_to_lowest_left(hull)


def canonical_polygon(poly: np.ndarray, decimals: int = 8) -> np.ndarray:
    """Canonical cyclic representation for robust polygon equality checks."""
    p = normalize_polygon(poly)
    p = rotate_to_lowest_left(p)
    p = np.round(p, decimals=decimals)
    return p


def polygons_equivalent_cycle(poly_a: np.ndarray, poly_b: np.ndarray, tol: float = 1e-7) -> bool:
    """Check equivalence of two polygons up to cyclic rotation."""
    a = canonical_polygon(poly_a)
    b = canonical_polygon(poly_b)
    if len(a) != len(b):
        return False

    n = len(a)
    for shift in range(n):
        if np.allclose(a, np.roll(b, -shift, axis=0), atol=tol, rtol=0.0):
            return True
    return False


def run_case(name: str, poly_a: np.ndarray, poly_b: np.ndarray) -> None:
    """Execute one test case and validate linear algorithm vs brute force."""
    fast = minkowski_sum_convex(poly_a, poly_b)
    ref = bruteforce_minkowski_hull(poly_a, poly_b)

    area_fast = polygon_area(fast)
    area_ref = polygon_area(ref)

    print(f"\n[{name}]")
    print(f"fast vertices: {len(fast)}, ref vertices: {len(ref)}")
    print(f"fast area: {area_fast:.6f}, ref area: {area_ref:.6f}")
    print("fast polygon:")
    print(np.array2string(fast, precision=4, floatmode="fixed"))

    if not polygons_equivalent_cycle(fast, ref):
        raise RuntimeError(f"case {name} failed: fast result != brute-force hull")


def generate_random_convex_polygon(n: int, seed: int) -> np.ndarray:
    """Generate a random convex polygon by sampling angles and radii."""
    if n < 3:
        raise ValueError("n must be >= 3")

    rng = np.random.default_rng(seed)
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n))
    radii = rng.uniform(0.6, 1.6, size=n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    hull = convex_hull_monotonic_chain(pts)
    hull = normalize_polygon(hull)
    return rotate_to_lowest_left(hull)


def main() -> None:
    print("Minkowski Sum MVP (MATH-0210)")
    print("=" * 64)

    square = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    triangle = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    pentagon = np.array(
        [
            [-1.0, 0.0],
            [-0.2, -0.9],
            [1.0, -0.5],
            [1.3, 0.7],
            [0.0, 1.2],
        ],
        dtype=np.float64,
    )
    quad = np.array(
        [
            [-0.7, -0.4],
            [0.8, -0.6],
            [1.1, 0.5],
            [-0.2, 1.0],
        ],
        dtype=np.float64,
    )

    random_a = generate_random_convex_polygon(n=9, seed=210)
    random_b = generate_random_convex_polygon(n=8, seed=102)

    run_case("square + triangle", square, triangle)
    run_case("pentagon + quadrilateral", pentagon, quad)
    run_case("random convex pair", random_a, random_b)

    print("\nAll test cases passed.")
    print("=" * 64)


if __name__ == "__main__":
    main()
