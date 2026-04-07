"""Delaunay triangulation MVP (non-interactive, runnable).

Run:
    python3 demo.py

Implementation strategy:
- Prefer SciPy/Qhull when available.
- Fall back to an in-file Bowyer-Watson implementation when SciPy is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

SCIPY_AVAILABLE = False
try:
    from scipy.spatial import Delaunay, QhullError

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    Delaunay = None

    class QhullError(Exception):
        """Fallback placeholder when SciPy is unavailable."""


Triangle = Tuple[int, int, int]
Edge = Tuple[int, int]


@dataclass
class DelaunayResult:
    unique_points: np.ndarray
    triangles: List[Triangle]
    removed_duplicates: int
    used_qhull: bool


def check_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
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


def make_ccw_triangle(i: int, j: int, k: int, pts: np.ndarray, eps: float = 1e-14) -> Triangle:
    cross = orientation(pts[i], pts[j], pts[k])
    if abs(cross) <= eps:
        raise ValueError("degenerate triangle")
    if cross > 0.0:
        return (i, j, k)
    return (i, k, j)


def canonical_edge(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def circumcircle_contains(tri: Triangle, p_idx: int, pts: np.ndarray, eps: float = 1e-12) -> bool:
    ia, ib, ic = tri
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
    return det > eps


def build_super_triangle(points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if span <= 0.0:
        span = 1.0

    p1 = np.array([center[0] - 20.0 * span, center[1] - 1.0 * span], dtype=float)
    p2 = np.array([center[0], center[1] + 20.0 * span], dtype=float)
    p3 = np.array([center[0] + 20.0 * span, center[1] - 1.0 * span], dtype=float)
    return np.vstack([p1, p2, p3])


def bowyer_watson_triangulation(points: np.ndarray, eps: float = 1e-12) -> List[Triangle]:
    n = points.shape[0]
    super_tri = build_super_triangle(points)
    aug = np.vstack([points, super_tri])

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
                tri = make_ccw_triangle(u, v, p_idx, aug, eps=eps)
            except ValueError:
                continue
            triangulation.add(tri)

    final_tris = [tri for tri in triangulation if all(v < n for v in tri)]
    return sorted(set(final_tris))


def triangulate_with_qhull(points: np.ndarray, eps: float, qhull_options: str) -> List[Triangle]:
    if not SCIPY_AVAILABLE or Delaunay is None:
        raise QhullError("SciPy Delaunay is unavailable")

    tri = Delaunay(points, qhull_options=qhull_options)
    triangles: List[Triangle] = []
    for simplex in tri.simplices:
        i, j, k = int(simplex[0]), int(simplex[1]), int(simplex[2])
        try:
            ccw = make_ccw_triangle(i, j, k, points, eps=eps)
        except ValueError:
            continue
        triangles.append(ccw)
    return sorted(set(triangles))


def build_delaunay(
    points: Sequence[Sequence[float]],
    eps: float = 1e-12,
    qhull_options: str = "Qbb Qc Qz Q12",
) -> DelaunayResult:
    pts = check_points(points)
    pts, removed = deduplicate_points(pts, eps=eps)

    n = pts.shape[0]
    if n < 3 or are_all_collinear(pts, eps=eps):
        return DelaunayResult(
            unique_points=pts,
            triangles=[],
            removed_duplicates=removed,
            used_qhull=False,
        )

    used_qhull = False
    try:
        triangles = triangulate_with_qhull(pts, eps=eps, qhull_options=qhull_options)
        used_qhull = True
    except Exception:
        triangles = bowyer_watson_triangulation(pts, eps=eps)

    return DelaunayResult(
        unique_points=pts,
        triangles=triangles,
        removed_duplicates=removed,
        used_qhull=used_qhull,
    )


def build_edge_set(triangles: Iterable[Triangle]) -> List[Edge]:
    edges: set[Edge] = set()
    for i, j, k in triangles:
        edges.add(canonical_edge(i, j))
        edges.add(canonical_edge(j, k))
        edges.add(canonical_edge(k, i))
    return sorted(edges)


def compute_vertex_degree_stats(num_points: int, edges: Sequence[Edge]) -> Tuple[int, float, int]:
    if num_points == 0:
        return (0, 0.0, 0)

    degree = np.zeros(num_points, dtype=int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    if np.all(degree == 0):
        return (0, 0.0, 0)

    return (int(degree.min()), float(degree.mean()), int(degree.max()))


def max_incircle_determinant(points: np.ndarray, triangles: Iterable[Triangle]) -> float:
    max_det = 0.0
    tris = list(triangles)
    for tri in tris:
        tri_set = set(tri)
        for p_idx in range(points.shape[0]):
            if p_idx in tri_set:
                continue

            ia, ib, ic = tri
            a, b, c = points[ia], points[ib], points[ic]
            p = points[p_idx]
            ax, ay = a[0] - p[0], a[1] - p[1]
            bx, by = b[0] - p[0], b[1] - p[1]
            cx, cy = c[0] - p[0], c[1] - p[1]
            det = (
                (ax * ax + ay * ay) * (bx * cy - by * cx)
                - (bx * bx + by * by) * (ax * cy - ay * cx)
                + (cx * cx + cy * cy) * (ax * by - ay * bx)
            )
            if det > max_det:
                max_det = float(det)
    return max_det


def verify_empty_circumcircle(
    points: np.ndarray,
    triangles: Iterable[Triangle],
    eps: float = 1e-10,
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
    result = build_delaunay(points)
    pts = result.unique_points
    tris = result.triangles

    all_ccw = all(orientation(pts[i], pts[j], pts[k]) > 0.0 for i, j, k in tris)
    edges = build_edge_set(tris)
    deg_min, deg_mean, deg_max = compute_vertex_degree_stats(len(pts), edges)

    delaunay_ok = verify_empty_circumcircle(pts, tris)
    max_det = max_incircle_determinant(pts, tris)

    print(f"=== {name} ===")
    print(f"input points: {len(points)}")
    print(f"unique points: {len(pts)} (removed duplicates: {result.removed_duplicates})")
    print(f"used qhull: {result.used_qhull}")
    print(f"triangles: {len(tris)}")
    print(f"edges: {len(edges)}")
    print(f"vertex degree (min/mean/max): {deg_min}/{deg_mean:.2f}/{deg_max}")
    print(f"all triangles CCW: {all_ccw}")
    print(f"empty circumcircle check: {delaunay_ok}")
    print(f"max incircle determinant (should be near <= 0): {max_det:.3e}")
    print(f"triangle preview: {tris[: min(8, len(tris))]}")
    print()


def main() -> None:
    rng = np.random.default_rng(189)

    case1 = rng.random((24, 2))

    gx, gy = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 4))
    case2_base = np.column_stack([gx.ravel(), gy.ravel()])
    case2 = case2_base + rng.normal(loc=0.0, scale=0.012, size=case2_base.shape)

    case3 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.6, 0.9],
            [0.25, 0.35],
            [0.82, 0.42],
            [0.16, 0.74],
            [0.74, 0.83],
            [0.25, 0.35],
            [1.0, 0.0],
            [0.5000000000001, 0.4999999999999],
        ],
        dtype=float,
    )

    case4 = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.2],
            [0.4, 0.4],
            [0.6, 0.6],
            [0.8, 0.8],
        ],
        dtype=float,
    )

    summarize_case("Uniform random", case1)
    summarize_case("Noisy grid", case2)
    summarize_case("With duplicates", case3)
    summarize_case("Collinear fallback", case4)


if __name__ == "__main__":
    main()
