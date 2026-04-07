"""Constrained Delaunay Triangulation (CDT) minimal runnable MVP.

This demo focuses on a practical 2D polygonal-domain variant:
- Input: one simple polygon (counter-clockwise or clockwise), and
  additional non-crossing constrained segments between existing vertices.
- Output: a triangulation that preserves all constrained edges and is
  locally Delaunay on non-constrained interior edges.

Implementation strategy:
1) Build an initial constrained triangulation by splitting polygon faces
   with constrained diagonals and triangulating each face via ear clipping.
2) Run Lawson edge flips on non-constrained interior edges until no
   illegal edge remains (local constrained Delaunay condition).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


EPS = 1e-10


Point = np.ndarray
Edge = Tuple[int, int]
Triangle = Tuple[int, int, int]


def as_edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def orient(a: Point, b: Point, c: Point) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def polygon_area(points: np.ndarray, polygon: Sequence[int]) -> float:
    area = 0.0
    n = len(polygon)
    for i in range(n):
        p = points[polygon[i]]
        q = points[polygon[(i + 1) % n]]
        area += p[0] * q[1] - p[1] * q[0]
    return 0.5 * area


def point_on_segment(p: Point, a: Point, b: Point, eps: float = EPS) -> bool:
    if abs(orient(a, b, p)) > eps:
        return False
    min_x, max_x = sorted((a[0], b[0]))
    min_y, max_y = sorted((a[1], b[1]))
    return (min_x - eps <= p[0] <= max_x + eps) and (min_y - eps <= p[1] <= max_y + eps)


def point_in_polygon(point: Point, points: np.ndarray, polygon: Sequence[int], eps: float = EPS) -> bool:
    inside = False
    n = len(polygon)
    x, y = point
    for i in range(n):
        a = points[polygon[i]]
        b = points[polygon[(i + 1) % n]]
        if point_on_segment(point, a, b, eps):
            return True
        xi, yi = a
        xj, yj = b
        hit = ((yi > y) != (yj > y))
        if hit:
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x_cross > x:
                inside = not inside
    return inside


def segments_proper_intersect(a: Point, b: Point, c: Point, d: Point, eps: float = EPS) -> bool:
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def segment_intersects_edge_interior(
    points: np.ndarray,
    u: int,
    v: int,
    e0: int,
    e1: int,
    eps: float = EPS,
) -> bool:
    shared = len({u, v}.intersection({e0, e1})) > 0
    if shared:
        return False
    a, b, c, d = points[u], points[v], points[e0], points[e1]
    return segments_proper_intersect(a, b, c, d, eps)


def is_valid_diagonal(points: np.ndarray, polygon: Sequence[int], u: int, v: int, eps: float = EPS) -> bool:
    n = len(polygon)
    if u == v or u not in polygon or v not in polygon:
        return False
    iu, iv = polygon.index(u), polygon.index(v)
    if abs(iu - iv) in (1, n - 1):
        return False

    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if segment_intersects_edge_interior(points, u, v, a, b, eps):
            return False

    midpoint = 0.5 * (points[u] + points[v])
    return point_in_polygon(midpoint, points, polygon, eps)


def split_polygon_with_diagonal(polygon: Sequence[int], u: int, v: int) -> Tuple[List[int], List[int]]:
    iu, iv = polygon.index(u), polygon.index(v)
    if iu > iv:
        iu, iv = iv, iu
        u, v = v, u

    part1 = list(polygon[iu : iv + 1])
    part2 = list(polygon[iv:]) + list(polygon[: iu + 1])
    return part1, part2


def point_in_triangle(p: Point, a: Point, b: Point, c: Point, eps: float = EPS) -> bool:
    o1 = orient(a, b, p)
    o2 = orient(b, c, p)
    o3 = orient(c, a, p)
    has_neg = (o1 < -eps) or (o2 < -eps) or (o3 < -eps)
    has_pos = (o1 > eps) or (o2 > eps) or (o3 > eps)
    return not (has_neg and has_pos)


def is_ear(points: np.ndarray, polygon: Sequence[int], i: int, eps: float = EPS) -> bool:
    n = len(polygon)
    a = polygon[(i - 1) % n]
    b = polygon[i]
    c = polygon[(i + 1) % n]
    pa, pb, pc = points[a], points[b], points[c]

    if orient(pa, pb, pc) <= eps:
        return False

    for j in range(n):
        p = polygon[j]
        if p in (a, b, c):
            continue
        if point_in_triangle(points[p], pa, pb, pc, eps):
            return False

    if n > 3:
        for j in range(n):
            e0 = polygon[j]
            e1 = polygon[(j + 1) % n]
            if len({a, c}.intersection({e0, e1})) > 0:
                continue
            if segments_proper_intersect(pa, pc, points[e0], points[e1], eps):
                return False
    return True


def orient_triangle_ccw(points: np.ndarray, tri: Triangle) -> Triangle:
    a, b, c = tri
    if orient(points[a], points[b], points[c]) >= 0:
        return tri
    return (a, c, b)


def triangulate_polygon_ear_clipping(points: np.ndarray, polygon: Sequence[int], eps: float = EPS) -> List[Triangle]:
    poly = list(polygon)
    if len(poly) < 3:
        raise ValueError("Polygon must have at least 3 vertices.")

    if polygon_area(points, poly) < 0:
        poly.reverse()

    triangles: List[Triangle] = []
    guard = 0
    while len(poly) > 3:
        clipped = False
        n = len(poly)
        for i in range(n):
            if is_ear(points, poly, i, eps):
                a = poly[(i - 1) % n]
                b = poly[i]
                c = poly[(i + 1) % n]
                triangles.append(orient_triangle_ccw(points, (a, b, c)))
                del poly[i]
                clipped = True
                break
        guard += 1
        if not clipped:
            raise RuntimeError("Ear clipping failed: polygon may be invalid or numerically degenerate.")
        if guard > 10000:
            raise RuntimeError("Ear clipping exceeded iteration guard.")

    triangles.append(orient_triangle_ccw(points, (poly[0], poly[1], poly[2])))
    return triangles


def triangle_edges(tri: Triangle) -> List[Edge]:
    a, b, c = tri
    return [as_edge(a, b), as_edge(b, c), as_edge(c, a)]


def build_edge_to_triangles(triangles: Sequence[Triangle]) -> Dict[Edge, List[int]]:
    edge_to_tris: Dict[Edge, List[int]] = {}
    for ti, tri in enumerate(triangles):
        for e in triangle_edges(tri):
            edge_to_tris.setdefault(e, []).append(ti)
    return edge_to_tris


def opposite_vertex(tri: Triangle, u: int, v: int) -> int:
    for x in tri:
        if x != u and x != v:
            return x
    raise ValueError("Degenerate triangle: no opposite vertex.")


def incircle(a: Point, b: Point, c: Point, d: Point) -> float:
    ax, ay = a[0] - d[0], a[1] - d[1]
    bx, by = b[0] - d[0], b[1] - d[1]
    cx, cy = c[0] - d[0], c[1] - d[1]
    det = (
        (ax * ax + ay * ay) * (bx * cy - by * cx)
        - (bx * bx + by * by) * (ax * cy - ay * cx)
        + (cx * cx + cy * cy) * (ax * by - ay * bx)
    )
    return float(det)


def edge_is_illegal(points: np.ndarray, u: int, v: int, w1: int, w2: int, eps: float = EPS) -> bool:
    pu, pv, pw1, pw2 = points[u], points[v], points[w1], points[w2]

    if not segments_proper_intersect(pu, pv, pw1, pw2, eps):
        return False

    a, b, c = pu, pv, pw1
    if orient(a, b, c) < 0:
        b, a = a, b
    return incircle(a, b, c, pw2) > eps


def constrained_delaunay_refine(
    points: np.ndarray,
    triangles: List[Triangle],
    constrained_edges: Set[Edge],
    eps: float = EPS,
    max_flips: int = 20000,
) -> List[Triangle]:
    triangles = [orient_triangle_ccw(points, t) for t in triangles]

    flips = 0
    changed = True
    while changed:
        changed = False
        edge_to_tris = build_edge_to_triangles(triangles)
        for edge, adj in edge_to_tris.items():
            if len(adj) != 2:
                continue
            if edge in constrained_edges:
                continue

            u, v = edge
            t1_idx, t2_idx = adj
            t1 = triangles[t1_idx]
            t2 = triangles[t2_idx]
            w1 = opposite_vertex(t1, u, v)
            w2 = opposite_vertex(t2, u, v)

            if edge_is_illegal(points, u, v, w1, w2, eps):
                new_t1 = orient_triangle_ccw(points, (w1, w2, u))
                new_t2 = orient_triangle_ccw(points, (w2, w1, v))
                triangles[t1_idx] = new_t1
                triangles[t2_idx] = new_t2
                flips += 1
                changed = True
                if flips > max_flips:
                    raise RuntimeError("Exceeded max_flips; possible degeneracy.")
                break
    return triangles


def triangulation_area(points: np.ndarray, triangles: Sequence[Triangle]) -> float:
    total = 0.0
    for a, b, c in triangles:
        total += 0.5 * abs(orient(points[a], points[b], points[c]))
    return total


@dataclass
class CDTResult:
    triangles: List[Triangle]
    constrained_edges: Set[Edge]
    boundary_edges: Set[Edge]


def build_initial_constrained_triangulation(
    points: np.ndarray,
    outer_polygon: Sequence[int],
    constraints: Sequence[Edge],
    eps: float = EPS,
) -> CDTResult:
    poly = list(outer_polygon)
    if polygon_area(points, poly) < 0:
        poly.reverse()

    faces: List[List[int]] = [poly]

    for u, v in constraints:
        inserted = False
        for fi, face in enumerate(faces):
            if u in face and v in face and is_valid_diagonal(points, face, u, v, eps):
                f1, f2 = split_polygon_with_diagonal(face, u, v)
                if len(f1) < 3 or len(f2) < 3:
                    continue
                faces[fi] = f1
                faces.append(f2)
                inserted = True
                break
        if not inserted:
            raise ValueError(f"Constraint ({u}, {v}) cannot be inserted into current faces.")

    triangles: List[Triangle] = []
    for face in faces:
        triangles.extend(triangulate_polygon_ear_clipping(points, face, eps))

    boundary_edges: Set[Edge] = set()
    for i in range(len(poly)):
        boundary_edges.add(as_edge(poly[i], poly[(i + 1) % len(poly)]))

    constrained_edges: Set[Edge] = set(boundary_edges)
    constrained_edges.update(as_edge(u, v) for (u, v) in constraints)

    return CDTResult(
        triangles=triangles,
        constrained_edges=constrained_edges,
        boundary_edges=boundary_edges,
    )


def count_illegal_unconstrained_edges(
    points: np.ndarray,
    triangles: Sequence[Triangle],
    constrained_edges: Set[Edge],
    eps: float = EPS,
) -> int:
    edge_to_tris = build_edge_to_triangles(triangles)
    bad = 0
    for edge, adj in edge_to_tris.items():
        if len(adj) != 2:
            continue
        if edge in constrained_edges:
            continue
        u, v = edge
        t1, t2 = triangles[adj[0]], triangles[adj[1]]
        w1 = opposite_vertex(t1, u, v)
        w2 = opposite_vertex(t2, u, v)
        if edge_is_illegal(points, u, v, w1, w2, eps):
            bad += 1
    return bad


def triangulation_edges(triangles: Sequence[Triangle]) -> Set[Edge]:
    edges: Set[Edge] = set()
    for tri in triangles:
        edges.update(triangle_edges(tri))
    return edges


def run_demo_case() -> None:
    points = np.array(
        [
            [-2.0, 0.0],   # 0
            [-1.0, -1.6],  # 1
            [1.0, -2.0],   # 2
            [3.1, -1.0],   # 3
            [3.6, 1.1],    # 4
            [2.0, 3.0],    # 5
            [0.0, 3.5],    # 6
            [-2.1, 2.0],   # 7
        ],
        dtype=float,
    )

    outer_polygon = [0, 1, 2, 3, 4, 5, 6, 7]
    constraints = [(0, 3), (0, 5)]

    cdt0 = build_initial_constrained_triangulation(points, outer_polygon, constraints)
    illegal_before = count_illegal_unconstrained_edges(
        points, cdt0.triangles, cdt0.constrained_edges
    )

    refined = constrained_delaunay_refine(points, cdt0.triangles, cdt0.constrained_edges)
    illegal_after = count_illegal_unconstrained_edges(
        points, refined, cdt0.constrained_edges
    )

    poly_area = abs(polygon_area(points, outer_polygon))
    tri_area = triangulation_area(points, refined)
    edges = triangulation_edges(refined)

    missing_constraints = [e for e in map(lambda x: as_edge(*x), constraints) if e not in edges]

    print("=== Constrained Delaunay Triangulation MVP ===")
    print(f"Vertices: {len(points)}")
    print(f"Initial triangles: {len(cdt0.triangles)}")
    print(f"Refined triangles: {len(refined)}")
    print(f"Illegal unconstrained edges before flips: {illegal_before}")
    print(f"Illegal unconstrained edges after flips:  {illegal_after}")
    print(f"Polygon area:      {poly_area:.6f}")
    print(f"Triangulation area:{tri_area:.6f}")
    print(f"Missing constraints in final mesh: {missing_constraints}")

    print("\nFinal triangles (vertex indices):")
    for tri in refined:
        print(f"  {tri}")

    assert not missing_constraints, "Constraint recovery failed."
    assert illegal_after == 0, "Not fully locally Delaunay on unconstrained edges."
    assert abs(poly_area - tri_area) < 1e-8, "Area mismatch indicates invalid triangulation."

    print("\nAll checks passed.")


def main() -> None:
    run_demo_case()


if __name__ == "__main__":
    main()
