"""MVP: line segment intersection detection with a sweep-line strategy.

This demo implements a Bentley-Ottmann-style sweep line with:
- endpoint/intersection event queue
- active segment ordering by current sweep x
- neighbor-only intersection checks

To keep the code compact and dependency-light, the active structure is a Python list
(not a balanced BST). Therefore this MVP favors clarity over asymptotic optimality.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Segment:
    sid: int
    p: Point
    q: Point


@dataclass(frozen=True)
class Intersection:
    kind: str  # "point" or "overlap"
    point: Optional[Point]


def normalize_segment(sid: int, a: Tuple[float, float], b: Tuple[float, float]) -> Segment:
    p = Point(float(a[0]), float(a[1]))
    q = Point(float(b[0]), float(b[1]))
    if (p.x, p.y) <= (q.x, q.y):
        return Segment(sid, p, q)
    return Segment(sid, q, p)


def cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def orient(a: Point, b: Point, c: Point) -> float:
    return cross(b.x - a.x, b.y - a.y, c.x - a.x, c.y - a.y)


def almost_equal(a: float, b: float, eps: float = EPS) -> bool:
    return abs(a - b) <= eps


def point_key(p: Point, digits: int = 12) -> Tuple[float, float]:
    return (round(p.x, digits), round(p.y, digits))


def on_segment(a: Point, b: Point, p: Point, eps: float = EPS) -> bool:
    if abs(orient(a, b, p)) > eps:
        return False
    return (
        min(a.x, b.x) - eps <= p.x <= max(a.x, b.x) + eps
        and min(a.y, b.y) - eps <= p.y <= max(a.y, b.y) + eps
    )


def line_intersection_point(s1: Segment, s2: Segment) -> Optional[Point]:
    p = np.array([s1.p.x, s1.p.y], dtype=float)
    r = np.array([s1.q.x - s1.p.x, s1.q.y - s1.p.y], dtype=float)
    q = np.array([s2.p.x, s2.p.y], dtype=float)
    s = np.array([s2.q.x - s2.p.x, s2.q.y - s2.p.y], dtype=float)

    denom = cross(r[0], r[1], s[0], s[1])
    if abs(denom) <= EPS:
        return None
    qp = q - p
    t = cross(qp[0], qp[1], s[0], s[1]) / denom
    inter = p + t * r
    return Point(float(inter[0]), float(inter[1]))


def segment_intersection(s1: Segment, s2: Segment) -> Optional[Intersection]:
    a, b, c, d = s1.p, s1.q, s2.p, s2.q

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    # Proper crossing.
    if o1 * o2 < -EPS and o3 * o4 < -EPS:
        ip = line_intersection_point(s1, s2)
        if ip is None:
            return None
        return Intersection("point", ip)

    # Endpoint touches.
    for p in (c, d):
        if on_segment(a, b, p):
            return Intersection("point", p)
    for p in (a, b):
        if on_segment(c, d, p):
            return Intersection("point", p)

    # Collinear overlap check.
    if abs(o1) <= EPS and abs(o2) <= EPS and abs(o3) <= EPS and abs(o4) <= EPS:
        points = [a, b, c, d]
        overlap_points: Dict[Tuple[float, float], Point] = {}
        for p in points:
            if on_segment(a, b, p) and on_segment(c, d, p):
                overlap_points[point_key(p)] = p
        if not overlap_points:
            return None
        if len(overlap_points) == 1:
            only = next(iter(overlap_points.values()))
            return Intersection("point", only)
        return Intersection("overlap", None)

    return None


def y_at(seg: Segment, x: float) -> float:
    x1, y1 = seg.p.x, seg.p.y
    x2, y2 = seg.q.x, seg.q.y
    if almost_equal(x1, x2):
        return min(y1, y2)
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def status_key(seg: Segment, x: float) -> Tuple[float, int]:
    return (y_at(seg, x), seg.sid)


def sweep_line_intersections(segments: Sequence[Segment]) -> Set[Tuple[int, int]]:
    seg_map: Dict[int, Segment] = {s.sid: s for s in segments}

    LEFT, INTERSECTION, RIGHT = 0, 1, 2
    counter = 0
    events: List[Tuple[float, float, int, int, int, Optional[int]]] = []

    def push_event(
        x: float,
        y: float,
        etype: int,
        a: int,
        b: Optional[int],
    ) -> None:
        nonlocal counter
        counter += 1
        heapq.heappush(events, (x, y, etype, counter, a, b))

    for s in segments:
        push_event(s.p.x, s.p.y, LEFT, s.sid, None)
        push_event(s.q.x, s.q.y, RIGHT, s.sid, None)

    active: List[int] = []
    found_pairs: Set[Tuple[int, int]] = set()
    scheduled_intersections: Set[Tuple[int, int, float, float]] = set()

    current_x = -1e100

    def pair_key(i: int, j: int) -> Tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def index_in_active(sid: int) -> int:
        return active.index(sid)

    def maybe_register_or_schedule(i: int, j: int, source_x: float) -> None:
        if i == j:
            return
        s1, s2 = seg_map[i], seg_map[j]
        inter = segment_intersection(s1, s2)
        if inter is None:
            return

        pkey = pair_key(i, j)
        if inter.kind == "overlap":
            found_pairs.add(pkey)
            return

        assert inter.point is not None
        px, py = inter.point.x, inter.point.y
        if px <= source_x + EPS:
            found_pairs.add(pkey)
            return

        key = (pkey[0], pkey[1], round(px, 12), round(py, 12))
        if key in scheduled_intersections:
            return
        scheduled_intersections.add(key)
        push_event(px, py, INTERSECTION, pkey[0], pkey[1])

    while events:
        x, y, etype, _id, a, b = heapq.heappop(events)
        current_x = x

        if etype == LEFT:
            # Insert according to ordering just to the right of event x.
            x_probe = current_x + EPS
            new_key = status_key(seg_map[a], x_probe)
            idx = 0
            while idx < len(active) and status_key(seg_map[active[idx]], x_probe) < new_key:
                idx += 1
            active.insert(idx, a)

            if idx - 1 >= 0:
                maybe_register_or_schedule(active[idx - 1], active[idx], current_x)
            if idx + 1 < len(active):
                maybe_register_or_schedule(active[idx], active[idx + 1], current_x)

        elif etype == RIGHT:
            if a not in active:
                continue
            idx = index_in_active(a)
            left_neighbor = active[idx - 1] if idx - 1 >= 0 else None
            right_neighbor = active[idx + 1] if idx + 1 < len(active) else None
            active.pop(idx)

            if left_neighbor is not None and right_neighbor is not None:
                maybe_register_or_schedule(left_neighbor, right_neighbor, current_x)

        else:
            assert b is not None
            pkey = pair_key(a, b)
            found_pairs.add(pkey)

            if a not in active or b not in active:
                continue

            i1, i2 = index_in_active(a), index_in_active(b)
            if i1 == i2:
                continue

            if i1 > i2:
                i1, i2 = i2, i1
                a, b = b, a

            # For robustness in degenerate cases, enforce adjacency before swap.
            if i2 - i1 != 1:
                active.sort(key=lambda sid: status_key(seg_map[sid], current_x + EPS))
                if a not in active or b not in active:
                    continue
                i1, i2 = index_in_active(a), index_in_active(b)
                if i1 > i2:
                    i1, i2 = i2, i1
                    a, b = b, a
                if i2 - i1 != 1:
                    continue

            active[i1], active[i2] = active[i2], active[i1]

            # Check new neighboring pairs caused by the swap.
            if i1 - 1 >= 0:
                maybe_register_or_schedule(active[i1 - 1], active[i1], current_x)
            if i2 + 1 < len(active):
                maybe_register_or_schedule(active[i2], active[i2 + 1], current_x)

    return found_pairs


def brute_force_intersections(segments: Sequence[Segment]) -> Set[Tuple[int, int]]:
    result: Set[Tuple[int, int]] = set()
    n = len(segments)
    for i in range(n):
        for j in range(i + 1, n):
            inter = segment_intersection(segments[i], segments[j])
            if inter is not None:
                result.add((segments[i].sid, segments[j].sid))
    return result


def build_segments(raw: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]) -> List[Segment]:
    return [normalize_segment(i, a, b) for i, (a, b) in enumerate(raw)]


def run_case(name: str, raw_segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]) -> None:
    segments = build_segments(raw_segments)
    sweep_ans = sweep_line_intersections(segments)
    brute_ans = brute_force_intersections(segments)

    print(f"\\n=== {name} ===")
    print(f"segment count: {len(segments)}")
    print(f"sweep intersections ({len(sweep_ans)}): {sorted(sweep_ans)}")
    print(f"brute intersections ({len(brute_ans)}): {sorted(brute_ans)}")

    if sweep_ans != brute_ans:
        missing = sorted(brute_ans - sweep_ans)
        extra = sorted(sweep_ans - brute_ans)
        raise AssertionError(
            "Sweep-line result mismatch with brute force. "
            f"Missing={missing}, Extra={extra}"
        )


def random_general_position_case(n_segments: int, seed: int = 203) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    raw: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for _ in range(n_segments):
        while True:
            p = rng.uniform(0.0, 100.0, size=2)
            q = rng.uniform(0.0, 100.0, size=2)
            if np.linalg.norm(q - p) < 1e-3:
                continue
            # Avoid near-vertical segments to reduce tie-heavy degenerate events in MVP.
            if abs(q[0] - p[0]) < 1e-3:
                continue
            raw.append(((float(p[0]), float(p[1])), (float(q[0]), float(q[1]))))
            break
    return raw


def main() -> None:
    # Deterministic handcrafted case.
    case1 = [
        ((0.0, 0.0), (5.0, 5.0)),
        ((0.0, 5.0), (5.0, 0.0)),
        ((0.5, 4.5), (2.0, 4.5)),
        ((6.0, 0.0), (7.0, 1.0)),
        ((2.0, -1.0), (3.0, 6.0)),
    ]

    # Random reproducible case (general position-oriented sampling).
    case2 = random_general_position_case(n_segments=20, seed=203)

    run_case("handcrafted", case1)
    run_case("random-seed-203", case2)

    print("\\nAll checks passed.")


if __name__ == "__main__":
    main()
