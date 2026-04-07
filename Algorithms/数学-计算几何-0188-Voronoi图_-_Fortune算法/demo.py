"""Minimal runnable MVP for Voronoi diagram via Fortune sweep line (MATH-0188).

This implementation focuses on clarity and algorithm traceability:
- site events and circle events (priority queue)
- beachline as a doubly-linked list of arcs
- breakpoint updates and Voronoi edge creation
- finite clipping of infinite rays to a bounding box

It is an educational MVP, not a production-grade robust geometry kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import List, Optional, Tuple

import numpy as np


EPS = 1e-9


@dataclass(frozen=True)
class Site:
    idx: int
    x: float
    y: float


@dataclass
class Edge:
    left_site: int
    right_site: int
    start: Optional[Tuple[float, float]] = None
    end: Optional[Tuple[float, float]] = None


@dataclass
class Event:
    kind: str  # "site" or "circle"
    y: float
    x: float
    site: Optional[Site] = None
    arc: Optional["Arc"] = None
    center: Optional[Tuple[float, float]] = None
    valid: bool = True


@dataclass
class Arc:
    site: Site
    prev: Optional["Arc"] = None
    next: Optional["Arc"] = None
    circle_event: Optional[Event] = None
    edge_left: Optional[Edge] = None
    edge_right: Optional[Edge] = None


class FortuneVoronoi:
    """Fortune sweep-line MVP for 2D Voronoi diagram."""

    def __init__(self, points: np.ndarray):
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be shape (n, 2)")
        if points.shape[0] < 2:
            raise ValueError("need at least 2 points")

        self.sites: List[Site] = [
            Site(idx=i, x=float(points[i, 0]), y=float(points[i, 1]))
            for i in range(points.shape[0])
        ]

        self.event_heap: List[Tuple[float, int, float, int, Event]] = []
        self.counter = 0
        self.head: Optional[Arc] = None

        self.current_y = float("inf")
        self.edges: List[Edge] = []
        self.vertices: List[Tuple[float, float]] = []

        xs = points[:, 0]
        ys = points[:, 1]
        pad = max(float(np.ptp(xs)), float(np.ptp(ys)), 1.0) * 0.2 + 1.0
        self.bbox = (
            float(np.min(xs) - pad),
            float(np.max(xs) + pad),
            float(np.min(ys) - pad),
            float(np.max(ys) + pad),
        )

    def _push_event(self, event: Event) -> None:
        # Higher y first => heap key uses -y. For same y, site events first.
        kind_pri = 0 if event.kind == "site" else 1
        self.counter += 1
        heapq.heappush(
            self.event_heap,
            (-event.y, kind_pri, event.x, self.counter, event),
        )

    def _parabola_y(self, site: Site, x: float, directrix_y: float) -> float:
        denom = 2.0 * (site.y - directrix_y)
        if abs(denom) < EPS:
            return float("inf")
        return ((x - site.x) ** 2 + site.y**2 - directrix_y**2) / denom

    def _breakpoint_x(self, left: Site, right: Site, directrix_y: float) -> float:
        # Handle near-degenerate cases where focus lies on directrix.
        if abs(left.y - directrix_y) < EPS:
            return left.x
        if abs(right.y - directrix_y) < EPS:
            return right.x

        if abs(left.y - right.y) < EPS:
            return 0.5 * (left.x + right.x)

        z0 = 2.0 * (left.y - directrix_y)
        z1 = 2.0 * (right.y - directrix_y)

        a = 1.0 / z0 - 1.0 / z1
        b = -2.0 * (left.x / z0 - right.x / z1)
        c = (
            (left.x * left.x + left.y * left.y - directrix_y * directrix_y) / z0
            - (right.x * right.x + right.y * right.y - directrix_y * directrix_y) / z1
        )

        if abs(a) < EPS:
            if abs(b) < EPS:
                return 0.5 * (left.x + right.x)
            return -c / b

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            disc = 0.0
        sqrt_disc = math.sqrt(disc)
        x1 = (-b - sqrt_disc) / (2.0 * a)
        x2 = (-b + sqrt_disc) / (2.0 * a)

        # Root selection for left-right ordered arcs.
        if left.y < right.y:
            return max(x1, x2)
        return min(x1, x2)

    def _find_arc_above(self, x: float) -> Optional[Arc]:
        if self.head is None:
            return None

        directrix = self.current_y - 1e-10
        arc = self.head
        while arc is not None:
            left_bp = -float("inf")
            right_bp = float("inf")
            if arc.prev is not None:
                left_bp = self._breakpoint_x(arc.prev.site, arc.site, directrix)
            if arc.next is not None:
                right_bp = self._breakpoint_x(arc.site, arc.next.site, directrix)

            if left_bp - EPS <= x <= right_bp + EPS:
                return arc
            arc = arc.next

        # Fallback: return rightmost arc.
        arc = self.head
        while arc is not None and arc.next is not None:
            arc = arc.next
        return arc

    @staticmethod
    def _orientation(a: Site, b: Site, c: Site) -> float:
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    @staticmethod
    def _circumcenter(a: Site, b: Site, c: Site) -> Optional[Tuple[float, float]]:
        d = 2.0 * (
            a.x * (b.y - c.y)
            + b.x * (c.y - a.y)
            + c.x * (a.y - b.y)
        )
        if abs(d) < EPS:
            return None

        a2 = a.x * a.x + a.y * a.y
        b2 = b.x * b.x + b.y * b.y
        c2 = c.x * c.x + c.y * c.y

        ux = (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d
        uy = (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d
        return (ux, uy)

    def _invalidate_circle_event(self, arc: Optional[Arc]) -> None:
        if arc is None:
            return
        if arc.circle_event is not None:
            arc.circle_event.valid = False
            arc.circle_event = None

    def _check_circle_event(self, arc: Optional[Arc]) -> None:
        if arc is None:
            return
        self._invalidate_circle_event(arc)

        if arc.prev is None or arc.next is None:
            return

        a = arc.prev.site
        b = arc.site
        c = arc.next.site

        # Circle event exists only for clockwise turn under this convention.
        if self._orientation(a, b, c) >= -EPS:
            return

        center = self._circumcenter(a, b, c)
        if center is None:
            return

        cx, cy = center
        radius = math.hypot(b.x - cx, b.y - cy)
        y_event = cy - radius

        if y_event >= self.current_y - EPS:
            return

        evt = Event(kind="circle", y=y_event, x=cx, arc=arc, center=center)
        arc.circle_event = evt
        self._push_event(evt)

    def _new_edge(self, left_site: Site, right_site: Site, start: Tuple[float, float]) -> Edge:
        e = Edge(left_site=left_site.idx, right_site=right_site.idx, start=start, end=None)
        self.edges.append(e)
        return e

    def _handle_site_event(self, site: Site) -> None:
        if self.head is None:
            self.head = Arc(site=site)
            return

        arc = self._find_arc_above(site.x)
        if arc is None:
            self.head = Arc(site=site)
            return

        self._invalidate_circle_event(arc)

        directrix = self.current_y - 1e-10
        start_y = self._parabola_y(arc.site, site.x, directrix)
        if not np.isfinite(start_y):
            start_y = site.y
        start = (site.x, start_y)

        left = Arc(site=arc.site)
        center = Arc(site=site)
        right = Arc(site=arc.site)

        # Preserve external neighborhood links.
        left.prev = arc.prev
        if left.prev is not None:
            left.prev.next = left
        left.next = center

        center.prev = left
        center.next = right

        right.prev = center
        right.next = arc.next
        if right.next is not None:
            right.next.prev = right

        # Preserve old boundary breakpoints on far left/right.
        left.edge_left = arc.edge_left
        right.edge_right = arc.edge_right

        # Two newborn breakpoints produce two half-edges from the same start.
        e1 = self._new_edge(left.site, center.site, start)
        e2 = self._new_edge(center.site, right.site, start)

        left.edge_right = e1
        center.edge_left = e1
        center.edge_right = e2
        right.edge_left = e2

        if arc == self.head:
            self.head = left

        # New possible circle events around the changed neighborhood.
        self._check_circle_event(left)
        self._check_circle_event(right)

    def _handle_circle_event(self, event: Event) -> None:
        if not event.valid:
            return
        arc = event.arc
        if arc is None:
            return
        if arc.circle_event is not event:
            return
        if arc.prev is None or arc.next is None:
            return
        if event.center is None:
            return

        vertex = event.center
        self.vertices.append(vertex)

        if arc.edge_left is not None and arc.edge_left.end is None:
            arc.edge_left.end = vertex
        if arc.edge_right is not None and arc.edge_right.end is None:
            arc.edge_right.end = vertex

        left_arc = arc.prev
        right_arc = arc.next

        # Remove disappearing arc from linked list.
        left_arc.next = right_arc
        right_arc.prev = left_arc

        self._invalidate_circle_event(left_arc)
        self._invalidate_circle_event(right_arc)

        # New breakpoint between left_arc and right_arc starts from this vertex.
        e_new = self._new_edge(left_arc.site, right_arc.site, vertex)
        left_arc.edge_right = e_new
        right_arc.edge_left = e_new

        arc.circle_event = None

        self._check_circle_event(left_arc)
        self._check_circle_event(right_arc)

    def _line_box_intersections(
        self,
        p0: Tuple[float, float],
        direction: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        min_x, max_x, min_y, max_y = self.bbox
        px, py = p0
        dx, dy = direction
        pts: List[Tuple[float, float]] = []

        def add_point(xv: float, yv: float) -> None:
            if not (min_x - 1e-8 <= xv <= max_x + 1e-8 and min_y - 1e-8 <= yv <= max_y + 1e-8):
                return
            for qx, qy in pts:
                if abs(qx - xv) < 1e-7 and abs(qy - yv) < 1e-7:
                    return
            pts.append((xv, yv))

        if abs(dx) > EPS:
            t = (min_x - px) / dx
            add_point(min_x, py + t * dy)
            t = (max_x - px) / dx
            add_point(max_x, py + t * dy)
        if abs(dy) > EPS:
            t = (min_y - py) / dy
            add_point(px + t * dx, min_y)
            t = (max_y - py) / dy
            add_point(px + t * dx, max_y)

        return pts

    def _clip_edge(self, edge: Edge) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        s = edge.start
        e = edge.end

        p = self.sites[edge.left_site]
        q = self.sites[edge.right_site]

        mid = ((p.x + q.x) * 0.5, (p.y + q.y) * 0.5)
        # Perpendicular to vector (q - p): direction of Voronoi edge.
        direction = (-(q.y - p.y), q.x - p.x)

        if abs(direction[0]) < EPS and abs(direction[1]) < EPS:
            return None

        if s is None and e is None:
            pts = self._line_box_intersections(mid, direction)
            if len(pts) < 2:
                return None
            # Pick the farthest pair.
            best = (pts[0], pts[1])
            best_d2 = -1.0
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d2 = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
                    if d2 > best_d2:
                        best_d2 = d2
                        best = (pts[i], pts[j])
            return best

        if s is not None and e is None:
            pts = self._line_box_intersections(s, direction)
            if not pts:
                return None
            far = max(pts, key=lambda t: (t[0] - s[0]) ** 2 + (t[1] - s[1]) ** 2)
            return (s, far)

        if s is None and e is not None:
            pts = self._line_box_intersections(e, direction)
            if not pts:
                return None
            far = max(pts, key=lambda t: (t[0] - e[0]) ** 2 + (t[1] - e[1]) ** 2)
            return (far, e)

        assert s is not None and e is not None
        return (s, e)

    def run(self) -> Tuple[List[Tuple[float, float]], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        for site in self.sites:
            evt = Event(kind="site", y=site.y, x=site.x, site=site)
            self._push_event(evt)

        while self.event_heap:
            neg_y, _kind_pri, _x, _id, evt = heapq.heappop(self.event_heap)
            self.current_y = -neg_y

            if evt.kind == "site":
                assert evt.site is not None
                self._handle_site_event(evt.site)
            else:
                self._handle_circle_event(evt)

        # Convert internal half-edges/rays to finite segments via clipping.
        segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for edge in self.edges:
            seg = self._clip_edge(edge)
            if seg is None:
                continue
            (x1, y1), (x2, y2) = seg
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 < 1e-12:
                continue
            segments.append(seg)

        return self.vertices, segments


def make_demo_points(n: int = 12, seed: int = 188) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 100.0, size=(n, 2))

    # Break exact y ties and x ties to reduce degeneracy for this MVP.
    order = np.argsort(pts[:, 1])
    for rank, i in enumerate(order):
        pts[i, 1] += rank * 1e-6
        pts[i, 0] += rank * 1e-6
    return pts


def optional_scipy_check(
    points: np.ndarray,
    our_vertices: List[Tuple[float, float]],
) -> str:
    """Optional consistency check against scipy.spatial.Voronoi.

    Returns a human-readable diagnostic string and never raises.
    """
    try:
        from scipy.spatial import Voronoi  # type: ignore
    except Exception:
        return "SciPy check: skipped (scipy not available)."

    try:
        ref = Voronoi(points)
    except Exception as exc:  # pragma: no cover - defensive path
        return f"SciPy check: failed to build reference Voronoi ({exc})."

    ref_vertices = ref.vertices
    if len(our_vertices) == 0:
        return (
            f"SciPy check: reference has {len(ref_vertices)} vertices, "
            "our sweep produced 0 finite circle vertices."
        )
    if ref_vertices.size == 0:
        return "SciPy check: reference has 0 finite vertices."

    ours = np.array(our_vertices, dtype=np.float64)
    d = np.linalg.norm(ours[:, None, :] - ref_vertices[None, :, :], axis=2)
    nearest = np.min(d, axis=1)

    return (
        "SciPy check: "
        f"ours={len(our_vertices)} vertices, ref={len(ref_vertices)} vertices, "
        f"mean nearest distance={float(np.mean(nearest)):.4f}, "
        f"max nearest distance={float(np.max(nearest)):.4f}."
    )


def summarize_segments(
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> Tuple[float, float, float]:
    lengths = []
    for (x1, y1), (x2, y2) in segments:
        lengths.append(math.hypot(x2 - x1, y2 - y1))
    if not lengths:
        return (0.0, 0.0, 0.0)
    arr = np.array(lengths, dtype=np.float64)
    return (float(np.min(arr)), float(np.mean(arr)), float(np.max(arr)))


def main() -> None:
    print("Voronoi Diagram by Fortune Sweep MVP (MATH-0188)")
    print("=" * 72)

    points = make_demo_points(n=14, seed=188)
    solver = FortuneVoronoi(points)
    vertices, segments = solver.run()

    print(f"input sites: {len(points)}")
    print(f"finite Voronoi vertices (from circle events): {len(vertices)}")
    print(f"clipped Voronoi segments: {len(segments)}")

    mn, avg, mx = summarize_segments(segments)
    print(f"segment length stats: min={mn:.4f}, mean={avg:.4f}, max={mx:.4f}")

    # Show first few artifacts for quick visual sanity in logs.
    preview_n = min(5, len(vertices))
    if preview_n > 0:
        print("sample vertices:")
        for i in range(preview_n):
            vx, vy = vertices[i]
            print(f"  v{i}: ({vx:.4f}, {vy:.4f})")

    preview_m = min(5, len(segments))
    if preview_m > 0:
        print("sample segments:")
        for i in range(preview_m):
            (x1, y1), (x2, y2) = segments[i]
            print(f"  e{i}: ({x1:.4f}, {y1:.4f}) -> ({x2:.4f}, {y2:.4f})")

    print(optional_scipy_check(points, vertices))

    if len(segments) == 0:
        raise RuntimeError("no Voronoi segments generated")

    print("=" * 72)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
