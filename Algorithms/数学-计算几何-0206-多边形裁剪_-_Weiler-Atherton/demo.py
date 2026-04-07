"""Minimal runnable MVP for polygon clipping via Weiler-Atherton (MATH-0206)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


EPS = 1e-9


@dataclass
class Node:
    """Doubly linked list node for polygon boundary traversal."""

    point: np.ndarray
    is_intersection: bool = False
    alpha: float = 0.0
    entry: bool = False
    visited: bool = False
    neighbor: Optional["Node"] = None
    next: Optional["Node"] = None
    prev: Optional["Node"] = None


def cross(a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product a x b."""
    return float(a[0] * b[1] - a[1] * b[0])


def polygon_area(poly: np.ndarray) -> float:
    """Signed polygon area; positive means counter-clockwise."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def normalize_polygon(poly: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Remove repeated points and enforce CCW orientation."""
    p = np.asarray(poly, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("polygon must be an Nx2 array")
    if len(p) < 3:
        raise ValueError("polygon must have at least 3 points")

    if np.linalg.norm(p[0] - p[-1]) <= eps:
        p = p[:-1]

    cleaned: List[np.ndarray] = []
    for pt in p:
        if not cleaned or np.linalg.norm(pt - cleaned[-1]) > eps:
            cleaned.append(pt)
    if len(cleaned) >= 2 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= eps:
        cleaned.pop()

    if len(cleaned) < 3:
        raise ValueError("polygon degenerates after duplicate removal")

    out = np.array(cleaned, dtype=np.float64)
    area = polygon_area(out)
    if abs(area) <= eps:
        raise ValueError("polygon has near-zero area")
    if area < 0.0:
        out = out[::-1]
    return out


def point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = EPS) -> bool:
    """Check whether p lies on segment ab within tolerance."""
    ap = p - a
    ab = b - a
    if abs(cross(ap, ab)) > eps:
        return False
    dot_val = float(np.dot(ap, ab))
    if dot_val < -eps:
        return False
    if dot_val > float(np.dot(ab, ab)) + eps:
        return False
    return True


def point_in_polygon(point: np.ndarray, polygon: np.ndarray, eps: float = EPS) -> bool:
    """Ray casting point-in-polygon; boundary counts as inside."""
    n = len(polygon)
    x, y = float(point[0]), float(point[1])

    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if point_on_segment(point, a, b, eps):
            return True

    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        intersects = (y1 > y) != (y2 > y)
        if intersects:
            x_cross = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
            if x_cross >= x - eps:
                inside = not inside
    return inside


def segment_intersection_strict(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    eps: float = EPS,
) -> Optional[Tuple[np.ndarray, float, float]]:
    """Return strict interior intersection (point, t, u) for p1->p2 and q1->q2.

    This MVP intentionally ignores collinear overlap and endpoint-only touch events.
    """
    r = p2 - p1
    s = q2 - q1
    denom = cross(r, s)
    if abs(denom) <= eps:
        return None

    qp = q1 - p1
    t = cross(qp, s) / denom
    u = cross(qp, r) / denom

    if eps < t < 1.0 - eps and eps < u < 1.0 - eps:
        point = p1 + t * r
        return point, float(t), float(u)
    return None


def build_ring(poly: np.ndarray) -> List[Node]:
    """Build cyclic doubly linked list from polygon vertices."""
    nodes = [Node(point=np.array(pt, dtype=np.float64)) for pt in poly]
    n = len(nodes)
    for i in range(n):
        nodes[i].next = nodes[(i + 1) % n]
        nodes[i].prev = nodes[(i - 1) % n]
    return nodes


def insert_after(node: Node, new_node: Node) -> None:
    """Insert new_node after node in the current ring."""
    nxt = node.next
    if nxt is None:
        raise RuntimeError("broken ring: node.next is None")
    node.next = new_node
    new_node.prev = node
    new_node.next = nxt
    nxt.prev = new_node


def iterate_ring(head: Node) -> Iterator[Node]:
    """Iterate nodes in a cyclic ring exactly once."""
    yield head
    cur = head.next
    while cur is not None and cur is not head:
        yield cur
        cur = cur.next


def collect_intersections_and_insert(
    subj_nodes: List[Node],
    clip_nodes: List[Node],
    subj_poly: np.ndarray,
    clip_poly: np.ndarray,
    eps: float = EPS,
) -> int:
    """Find all strict intersections and insert paired nodes into both rings."""
    subj_hits: Dict[int, List[Node]] = {i: [] for i in range(len(subj_poly))}
    clip_hits: Dict[int, List[Node]] = {i: [] for i in range(len(clip_poly))}
    inter_count = 0

    for i in range(len(subj_poly)):
        p1 = subj_poly[i]
        p2 = subj_poly[(i + 1) % len(subj_poly)]
        for j in range(len(clip_poly)):
            q1 = clip_poly[j]
            q2 = clip_poly[(j + 1) % len(clip_poly)]
            hit = segment_intersection_strict(p1, p2, q1, q2, eps=eps)
            if hit is None:
                continue

            pt, t, u = hit
            subj_inter = Node(point=np.array(pt, dtype=np.float64), is_intersection=True, alpha=t)
            clip_inter = Node(point=np.array(pt, dtype=np.float64), is_intersection=True, alpha=u)
            subj_inter.neighbor = clip_inter
            clip_inter.neighbor = subj_inter

            subj_hits[i].append(subj_inter)
            clip_hits[j].append(clip_inter)
            inter_count += 1

    for i in range(len(subj_poly)):
        if not subj_hits[i]:
            continue
        cur = subj_nodes[i]
        for node in sorted(subj_hits[i], key=lambda x: x.alpha):
            insert_after(cur, node)
            cur = node

    for j in range(len(clip_poly)):
        if not clip_hits[j]:
            continue
        cur = clip_nodes[j]
        for node in sorted(clip_hits[j], key=lambda x: x.alpha):
            insert_after(cur, node)
            cur = node

    return inter_count


def mark_entry_exit(subj_head: Node, clip_poly: np.ndarray, eps: float = EPS) -> None:
    """Mark intersection nodes on subject ring as entry/exit by parity toggling."""
    inside = point_in_polygon(subj_head.point, clip_poly, eps=eps)

    intersections = [node for node in iterate_ring(subj_head) if node.is_intersection]
    for node in intersections:
        node.entry = not inside
        if node.neighbor is not None:
            node.neighbor.entry = inside
        inside = not inside


def clean_output_polygon(poly: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Drop duplicate consecutive points and remove collinear redundancies."""
    if len(poly) == 0:
        return poly

    pts: List[np.ndarray] = []
    for pt in poly:
        if not pts or np.linalg.norm(pt - pts[-1]) > eps:
            pts.append(np.array(pt, dtype=np.float64))

    if len(pts) >= 2 and np.linalg.norm(pts[0] - pts[-1]) <= eps:
        pts.pop()

    changed = True
    while changed and len(pts) >= 3:
        changed = False
        new_pts: List[np.ndarray] = []
        n = len(pts)
        for i in range(n):
            a = pts[i - 1]
            b = pts[i]
            c = pts[(i + 1) % n]
            if abs(cross(b - a, c - b)) <= eps:
                changed = True
                continue
            new_pts.append(b)
        pts = new_pts

    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)
    out = np.array(pts, dtype=np.float64)
    if len(out) >= 3 and polygon_area(out) < 0.0:
        out = out[::-1]
    return out


def traverse_result_polygons(subj_head: Node, eps: float = EPS) -> List[np.ndarray]:
    """Traverse entry intersections and extract clipped polygon loops."""
    results: List[np.ndarray] = []
    ring_len = sum(1 for _ in iterate_ring(subj_head))
    max_steps = max(64, ring_len * 8)

    candidates = [
        node
        for node in iterate_ring(subj_head)
        if node.is_intersection and node.entry and not node.visited
    ]

    for start in candidates:
        if start.visited:
            continue

        loop_points: List[np.ndarray] = []
        current = start
        on_subject = True
        steps = 0

        while True:
            steps += 1
            if steps > max_steps:
                raise RuntimeError("traversal exceeded safety step limit")

            if current is start and current.visited and on_subject:
                break

            if not loop_points or np.linalg.norm(current.point - loop_points[-1]) > eps:
                loop_points.append(np.array(current.point, dtype=np.float64))

            if current.is_intersection:
                current.visited = True
                if current.neighbor is None:
                    raise RuntimeError("intersection node missing neighbor")
                current.neighbor.visited = True
                if not current.entry:
                    current = current.neighbor
                    on_subject = not on_subject
                    if current is start and current.visited and on_subject:
                        break

            step = current.next
            if step is None:
                raise RuntimeError("broken ring: current.next is None")
            current = step

        poly = clean_output_polygon(np.array(loop_points, dtype=np.float64), eps=eps)
        if len(poly) >= 3 and abs(polygon_area(poly)) > eps:
            results.append(poly)

    return results


def weiler_atherton_clip(subject: np.ndarray, clipper: np.ndarray, eps: float = EPS) -> List[np.ndarray]:
    """Compute intersection of two simple polygons via Weiler-Atherton.

    Notes:
    - Supports typical non-degenerate crossings.
    - Endpoint-only touch and collinear-overlap edges are intentionally skipped in this MVP.
    """
    subj_poly = normalize_polygon(subject, eps=eps)
    clip_poly = normalize_polygon(clipper, eps=eps)

    subj_nodes = build_ring(subj_poly)
    clip_nodes = build_ring(clip_poly)

    inter_count = collect_intersections_and_insert(
        subj_nodes=subj_nodes,
        clip_nodes=clip_nodes,
        subj_poly=subj_poly,
        clip_poly=clip_poly,
        eps=eps,
    )

    if inter_count == 0:
        if point_in_polygon(subj_poly[0], clip_poly, eps=eps):
            return [subj_poly]
        if point_in_polygon(clip_poly[0], subj_poly, eps=eps):
            return [clip_poly]
        return []

    mark_entry_exit(subj_nodes[0], clip_poly, eps=eps)
    return traverse_result_polygons(subj_nodes[0], eps=eps)


def format_polygon(poly: np.ndarray) -> str:
    """Readable polygon string."""
    return np.array2string(poly, precision=4, floatmode="fixed")


def run_case(
    name: str,
    subject: np.ndarray,
    clipper: np.ndarray,
    expected_count: int,
    expected_area: Optional[float] = None,
) -> None:
    """Run a single deterministic test case."""
    result = weiler_atherton_clip(subject, clipper, eps=EPS)

    print(f"\n[{name}]")
    print(f"result polygon count: {len(result)}")
    if len(result) != expected_count:
        raise RuntimeError(
            f"case {name} failed: expected {expected_count} polygons, got {len(result)}"
        )

    total_area = 0.0
    for idx, poly in enumerate(result, start=1):
        area = abs(polygon_area(poly))
        total_area += area
        print(f"polygon {idx}: vertices={len(poly)}, area={area:.6f}")
        print(format_polygon(poly))

    if expected_area is not None and abs(total_area - expected_area) > 1e-6:
        raise RuntimeError(
            f"case {name} failed: expected area {expected_area:.6f}, got {total_area:.6f}"
        )


def main() -> None:
    print("Weiler-Atherton Polygon Clipping MVP (MATH-0206)")
    print("=" * 72)

    subject1 = np.array(
        [
            [-1.0, 0.2],
            [1.2, 2.8],
            [3.2, 0.3],
            [2.0, 0.3],
            [2.0, -1.2],
            [0.2, -1.2],
            [0.2, 0.3],
        ],
        dtype=np.float64,
    )
    clip1 = np.array(
        [
            [0.6, -0.6],
            [2.6, -0.6],
            [2.6, 2.2],
            [0.6, 2.2],
        ],
        dtype=np.float64,
    )

    subject2 = np.array(
        [
            [-3.0, -3.0],
            [-2.0, -3.0],
            [-2.0, -2.0],
            [-3.0, -2.0],
        ],
        dtype=np.float64,
    )
    clip2 = np.array(
        [
            [1.0, 1.0],
            [3.0, 1.0],
            [3.0, 3.0],
            [1.0, 3.0],
        ],
        dtype=np.float64,
    )

    subject3 = np.array(
        [
            [-2.0, -2.0],
            [4.0, -2.0],
            [4.0, 4.0],
            [-2.0, 4.0],
        ],
        dtype=np.float64,
    )
    clip3 = np.array(
        [
            [0.0, 0.0],
            [1.8, 0.2],
            [1.6, 1.6],
            [0.8, 2.0],
            [-0.2, 1.2],
        ],
        dtype=np.float64,
    )

    expected_area_case3 = abs(polygon_area(normalize_polygon(clip3)))

    run_case(
        name="concave subject clipped by rectangle",
        subject=subject1,
        clipper=clip1,
        expected_count=1,
    )
    run_case(
        name="disjoint polygons",
        subject=subject2,
        clipper=clip2,
        expected_count=0,
    )
    run_case(
        name="clipper fully inside subject",
        subject=subject3,
        clipper=clip3,
        expected_count=1,
        expected_area=expected_area_case3,
    )

    print("\nAll deterministic test cases passed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
