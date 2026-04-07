"""Weiler-Atherton polygon clipping MVP.

This demo computes intersection polygons between a subject polygon and a clip polygon.
Implementation focus:
- No geometry black-box library.
- Weiler-Atherton style linked-list traversal.
- Handles common non-degenerate simple polygons.

Limitations are documented in README (collinear overlap / tangential corner cases).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-9


@dataclass
class Node:
    point: np.ndarray
    is_intersection: bool = False
    alpha: float = 0.0
    entry: Optional[bool] = None
    neighbor: Optional["Node"] = None
    next: Optional["Node"] = None
    prev: Optional["Node"] = None
    visited: bool = False
    rec_id: int = -1


@dataclass
class IntersectionRecord:
    rec_id: int
    point: np.ndarray
    s_edge: int
    s_alpha: float
    c_edge: int
    c_alpha: float
    s_node: Optional[Node] = None
    c_node: Optional[Node] = None


def as_np_points(poly: Sequence[Tuple[float, float]]) -> List[np.ndarray]:
    return [np.array([float(x), float(y)], dtype=float) for x, y in poly]


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def segment_intersection(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> Optional[Tuple[float, float, np.ndarray]]:
    """Return (t, u, point) if p1->p2 intersects q1->q2.

    For MVP we skip collinear-overlap handling to keep code minimal.
    """
    r = p2 - p1
    s = q2 - q1
    rxs = cross2(r, s)

    if abs(rxs) < EPS:
        return None  # parallel or collinear

    qp = q1 - p1
    t = cross2(qp, s) / rxs
    u = cross2(qp, r) / rxs

    if -EPS <= t <= 1.0 + EPS and -EPS <= u <= 1.0 + EPS:
        t_clamped = min(max(t, 0.0), 1.0)
        u_clamped = min(max(u, 0.0), 1.0)
        point = p1 + t_clamped * r
        return t_clamped, u_clamped, point
    return None


def points_close(a: np.ndarray, b: np.ndarray, eps: float = 1e-7) -> bool:
    return float(np.linalg.norm(a - b)) <= eps


def point_on_segment(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    ap = pt - a
    ab = b - a
    if abs(cross2(ap, ab)) > 1e-8:
        return False
    dotp = float(np.dot(ap, ab))
    if dotp < -1e-8:
        return False
    if dotp - float(np.dot(ab, ab)) > 1e-8:
        return False
    return True


def point_in_polygon(pt: np.ndarray, poly: Sequence[np.ndarray]) -> bool:
    """Ray-casting with boundary-inclusive behavior."""
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if point_on_segment(pt, a, b):
            return True

    inside = False
    x, y = float(pt[0]), float(pt[1])
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        x1, y1 = float(a[0]), float(a[1])
        x2, y2 = float(b[0]), float(b[1])

        if (y1 > y) != (y2 > y):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1
            if x < xinters:
                inside = not inside
    return inside


def cyclic_link(nodes: List[Node]) -> None:
    n = len(nodes)
    for i, node in enumerate(nodes):
        node.next = nodes[(i + 1) % n]
        node.prev = nodes[(i - 1 + n) % n]


def build_polygon_nodes(
    verts: Sequence[np.ndarray],
    edge_hits: Dict[int, List[Tuple[float, int]]],
    records: List[IntersectionRecord],
    is_subject: bool,
) -> List[Node]:
    nodes: List[Node] = []
    n = len(verts)

    for i in range(n):
        nodes.append(Node(point=verts[i].copy(), is_intersection=False))

        hits = sorted(edge_hits.get(i, []), key=lambda x: x[0])
        for alpha, rec_id in hits:
            # Skip edge endpoints in MVP to avoid duplicate corner bookkeeping.
            if alpha <= 1e-8 or alpha >= 1.0 - 1e-8:
                continue

            rec = records[rec_id]
            inode = Node(
                point=rec.point.copy(),
                is_intersection=True,
                alpha=alpha,
                rec_id=rec_id,
            )
            nodes.append(inode)

            if is_subject:
                rec.s_node = inode
            else:
                rec.c_node = inode

    cyclic_link(nodes)
    return nodes


def dedupe_polygon(points: List[np.ndarray]) -> List[np.ndarray]:
    if not points:
        return points

    out: List[np.ndarray] = []
    for p in points:
        if not out or not points_close(out[-1], p):
            out.append(p)

    if len(out) > 1 and points_close(out[0], out[-1]):
        out.pop()
    return out


def classify_entry_exit(subject_nodes: Sequence[Node], clip_verts: Sequence[np.ndarray]) -> None:
    for node in subject_nodes:
        if not node.is_intersection or node.prev is None or node.next is None:
            continue

        before = node.point + 1e-6 * (node.prev.point - node.point)
        after = node.point + 1e-6 * (node.next.point - node.point)
        inside_before = point_in_polygon(before, clip_verts)
        inside_after = point_in_polygon(after, clip_verts)

        # crossing outside->inside => entry
        node.entry = (not inside_before) and inside_after

        if node.neighbor is not None:
            node.neighbor.entry = node.entry


def weiler_atherton_intersection(
    subject: Sequence[Tuple[float, float]],
    clip: Sequence[Tuple[float, float]],
) -> List[List[Tuple[float, float]]]:
    s_verts = as_np_points(subject)
    c_verts = as_np_points(clip)

    s_n = len(s_verts)
    c_n = len(c_verts)

    records: List[IntersectionRecord] = []
    s_hits: Dict[int, List[Tuple[float, int]]] = {}
    c_hits: Dict[int, List[Tuple[float, int]]] = {}

    for i in range(s_n):
        s1 = s_verts[i]
        s2 = s_verts[(i + 1) % s_n]
        for j in range(c_n):
            c1 = c_verts[j]
            c2 = c_verts[(j + 1) % c_n]
            result = segment_intersection(s1, s2, c1, c2)
            if result is None:
                continue

            t, u, point = result
            rec_id = len(records)
            records.append(
                IntersectionRecord(
                    rec_id=rec_id,
                    point=point,
                    s_edge=i,
                    s_alpha=t,
                    c_edge=j,
                    c_alpha=u,
                )
            )
            s_hits.setdefault(i, []).append((t, rec_id))
            c_hits.setdefault(j, []).append((u, rec_id))

    if not records:
        if point_in_polygon(s_verts[0], c_verts):
            return [[(float(p[0]), float(p[1])) for p in s_verts]]
        if point_in_polygon(c_verts[0], s_verts):
            return [[(float(p[0]), float(p[1])) for p in c_verts]]
        return []

    subject_nodes = build_polygon_nodes(s_verts, s_hits, records, is_subject=True)
    _clip_nodes = build_polygon_nodes(c_verts, c_hits, records, is_subject=False)

    # Pair the two copies of each intersection.
    for rec in records:
        if rec.s_node is None or rec.c_node is None:
            continue
        rec.s_node.neighbor = rec.c_node
        rec.c_node.neighbor = rec.s_node

    classify_entry_exit(subject_nodes, c_verts)

    result: List[List[Tuple[float, float]]] = []
    starts = [n for n in subject_nodes if n.is_intersection and n.entry and not n.visited]

    for start in starts:
        if start.visited:
            continue

        loop: List[np.ndarray] = []
        current = start
        start.visited = True
        if start.neighbor is not None:
            start.neighbor.visited = True

        # conservative guard to avoid infinite loops under degeneracy
        guard = 0
        while guard < 20000:
            guard += 1

            # Step A: walk on subject from current intersection to next intersection.
            while guard < 20000:
                guard += 1
                if not loop or not points_close(loop[-1], current.point):
                    loop.append(current.point.copy())

                current = current.next
                if current is None:
                    break
                if current.is_intersection:
                    if not loop or not points_close(loop[-1], current.point):
                        loop.append(current.point.copy())
                    current.visited = True
                    if current.neighbor is not None:
                        current.neighbor.visited = True
                    break

            if current is None or current.neighbor is None:
                break

            # Switch to clip at this intersection.
            current = current.neighbor

            # Step B: walk on clip to next intersection.
            while guard < 20000:
                guard += 1
                if not loop or not points_close(loop[-1], current.point):
                    loop.append(current.point.copy())

                current = current.next
                if current is None:
                    break
                if current.is_intersection:
                    if not loop or not points_close(loop[-1], current.point):
                        loop.append(current.point.copy())
                    current.visited = True
                    if current.neighbor is not None:
                        current.neighbor.visited = True
                    break

            if current is None or current.neighbor is None:
                break

            # If clip-side intersection maps back to start, ring is closed.
            if current.neighbor is start:
                break

            # Switch back to subject and continue tracing.
            current = current.neighbor

        loop = dedupe_polygon(loop)
        if len(loop) >= 3:
            result.append([(float(p[0]), float(p[1])) for p in loop])

    if result:
        return result

    # Fallback for cases where entry classification yielded no explicit start.
    if point_in_polygon(s_verts[0], c_verts):
        return [[(float(p[0]), float(p[1])) for p in s_verts]]
    if point_in_polygon(c_verts[0], s_verts):
        return [[(float(p[0]), float(p[1])) for p in c_verts]]
    return []


def polygon_area(poly: Sequence[Tuple[float, float]]) -> float:
    area2 = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area2 += x1 * y2 - y1 * x2
    return 0.5 * area2


def main() -> None:
    # A non-degenerate example with real edge intersections:
    # subject is concave, clip is a rectangle.
    subject = [
        (0.0, 0.0),
        (6.0, 0.0),
        (6.0, 5.0),
        (3.0, 2.0),
        (0.0, 5.0),
    ]
    clip = [
        (2.0, -1.0),
        (7.0, -1.0),
        (7.0, 4.0),
        (2.0, 4.0),
    ]

    clipped = weiler_atherton_intersection(subject, clip)

    print("Weiler-Atherton clipping demo")
    print(f"subject vertices: {len(subject)}")
    print(f"clip vertices:    {len(clip)}")
    print(f"output polygons:  {len(clipped)}")

    if not clipped:
        print("No intersection.")
        return

    for idx, poly in enumerate(clipped, start=1):
        print(f"\nPolygon #{idx}:")
        for x, y in poly:
            print(f"  ({x:.6f}, {y:.6f})")
        print(f"  signed area: {polygon_area(poly):.6f}")


if __name__ == "__main__":
    main()
