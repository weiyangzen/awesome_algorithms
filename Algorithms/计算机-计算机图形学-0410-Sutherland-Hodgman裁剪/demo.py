"""CS-0253: Sutherland-Hodgman 多边形裁剪最小可运行 MVP。

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

Point = tuple[float, float]
Polygon = list[Point]

EPS = 1e-9


@dataclass(frozen=True)
class ClipCase:
    """Deterministic regression case for polygon clipping."""

    name: str
    subject: Polygon
    clipper: Polygon
    expected: Polygon


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    """2D cross product of vectors a and b."""
    return ax * by - ay * bx


def polygon_signed_area(poly: Sequence[Point]) -> float:
    """Signed area via shoelace formula.

    Positive => counter-clockwise, negative => clockwise.
    """
    if len(poly) < 3:
        return 0.0

    arr = np.asarray(poly, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_convex_polygon(poly: Sequence[Point], eps: float = EPS) -> bool:
    """Check polygon convexity (allows collinear consecutive edges)."""
    n = len(poly)
    if n < 3:
        return False

    prev_sign = 0
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        cx, cy = poly[(i + 2) % n]

        cross_val = _cross(bx - ax, by - ay, cx - bx, cy - by)
        if abs(cross_val) <= eps:
            continue

        sign = 1 if cross_val > 0 else -1
        if prev_sign == 0:
            prev_sign = sign
        elif sign != prev_sign:
            return False

    return prev_sign != 0


def _inside(point: Point, edge_start: Point, edge_end: Point, clip_ccw: bool) -> bool:
    """Half-plane test against one clipping edge."""
    px, py = point
    ax, ay = edge_start
    bx, by = edge_end
    value = _cross(bx - ax, by - ay, px - ax, py - ay)

    if clip_ccw:
        return value >= -EPS
    return value <= EPS


def _line_intersection(segment_start: Point, segment_end: Point, edge_start: Point, edge_end: Point) -> Point:
    """Intersection point between infinite lines (segment_start->segment_end) and (edge_start->edge_end)."""
    sx, sy = segment_start
    ex, ey = segment_end
    ax, ay = edge_start
    bx, by = edge_end

    rx, ry = ex - sx, ey - sy
    tx, ty = bx - ax, by - ay
    qx, qy = ax - sx, ay - sy

    denom = _cross(rx, ry, tx, ty)
    if abs(denom) <= EPS:
        # Nearly parallel. In Sutherland-Hodgman this usually appears near boundary-aligned
        # segments; use segment end as a stable fallback.
        return float(ex), float(ey)

    t = _cross(qx, qy, tx, ty) / denom
    return float(sx + t * rx), float(sy + t * ry)


def _points_close(p: Point, q: Point, tol: float = 1e-7) -> bool:
    return abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol


def _clean_polygon(poly: Sequence[Point]) -> Polygon:
    """Remove near-duplicate consecutive vertices and trailing closure vertex."""
    cleaned: Polygon = []
    for p in poly:
        vertex = (float(p[0]), float(p[1]))
        if not cleaned or not _points_close(vertex, cleaned[-1]):
            cleaned.append(vertex)

    if len(cleaned) > 1 and _points_close(cleaned[0], cleaned[-1]):
        cleaned.pop()

    if len(cleaned) < 3:
        return []
    return cleaned


def sutherland_hodgman(subject_polygon: Sequence[Point], clip_polygon: Sequence[Point]) -> Polygon:
    """Clip `subject_polygon` by convex `clip_polygon` using Sutherland-Hodgman algorithm."""
    if len(subject_polygon) < 3:
        raise ValueError("subject polygon must have at least 3 vertices")
    if len(clip_polygon) < 3:
        raise ValueError("clip polygon must have at least 3 vertices")
    if not is_convex_polygon(clip_polygon):
        raise ValueError("clip polygon must be convex")

    output = _clean_polygon(subject_polygon)
    if not output:
        return []

    clipper = _clean_polygon(clip_polygon)
    if not clipper:
        raise ValueError("clip polygon is degenerate after cleanup")

    clip_ccw = polygon_signed_area(clipper) > 0

    for i in range(len(clipper)):
        edge_start = clipper[i]
        edge_end = clipper[(i + 1) % len(clipper)]

        input_list = output
        output = []
        if not input_list:
            break

        s = input_list[-1]
        s_inside = _inside(s, edge_start, edge_end, clip_ccw)

        for e in input_list:
            e_inside = _inside(e, edge_start, edge_end, clip_ccw)

            if e_inside:
                if not s_inside:
                    output.append(_line_intersection(s, e, edge_start, edge_end))
                output.append(e)
            elif s_inside:
                output.append(_line_intersection(s, e, edge_start, edge_end))

            s = e
            s_inside = e_inside

        output = _clean_polygon(output)

    return output


def polygons_equivalent(poly_a: Sequence[Point], poly_b: Sequence[Point], tol: float = 1e-6) -> bool:
    """Compare polygons up to cyclic shift and orientation reversal."""
    if len(poly_a) != len(poly_b):
        return False
    if not poly_a and not poly_b:
        return True

    a = np.asarray(poly_a, dtype=np.float64)
    b = np.asarray(poly_b, dtype=np.float64)
    n = len(poly_a)

    for reversed_order in (False, True):
        c = b[::-1] if reversed_order else b
        for shift in range(n):
            rolled = np.roll(c, -shift, axis=0)
            if np.allclose(a, rolled, atol=tol, rtol=0.0):
                return True

    return False


def assert_polygon_inside_clip(poly: Sequence[Point], clipper: Sequence[Point]) -> None:
    """Assert every clipped vertex lies inside (or on boundary of) clip polygon."""
    if not poly:
        return

    clip_ccw = polygon_signed_area(clipper) > 0
    for p in poly:
        for i in range(len(clipper)):
            a = clipper[i]
            b = clipper[(i + 1) % len(clipper)]
            if not _inside(p, a, b, clip_ccw):
                raise AssertionError(f"Vertex {p} is outside clipping polygon")


def run_fixed_cases() -> None:
    clipper_ccw: Polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    clipper_cw: Polygon = list(reversed(clipper_ccw))

    cases = [
        ClipCase(
            name="fully inside",
            subject=[(2.0, 2.0), (8.0, 2.0), (5.0, 8.0)],
            clipper=clipper_ccw,
            expected=[(2.0, 2.0), (8.0, 2.0), (5.0, 8.0)],
        ),
        ClipCase(
            name="fully outside",
            subject=[(-5.0, 1.0), (-1.0, 1.0), (-1.0, 4.0), (-5.0, 4.0)],
            clipper=clipper_ccw,
            expected=[],
        ),
        ClipCase(
            name="partially overlap left boundary",
            subject=[(-2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (-2.0, 8.0)],
            clipper=clipper_ccw,
            expected=[(0.0, 2.0), (8.0, 2.0), (8.0, 8.0), (0.0, 8.0)],
        ),
        ClipCase(
            name="subject covers clipper",
            subject=[(-2.0, -2.0), (12.0, -2.0), (12.0, 12.0), (-2.0, 12.0)],
            clipper=clipper_ccw,
            expected=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        ),
        ClipCase(
            name="clockwise clipper orientation",
            subject=[(-2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (-2.0, 8.0)],
            clipper=clipper_cw,
            expected=[(0.0, 2.0), (8.0, 2.0), (8.0, 8.0), (0.0, 8.0)],
        ),
        ClipCase(
            name="concave subject polygon",
            subject=[(-1.0, 1.0), (7.0, 1.0), (7.0, 4.0), (3.0, 4.0), (3.0, 8.0), (-1.0, 8.0)],
            clipper=[(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)],
            expected=[(0.0, 1.0), (6.0, 1.0), (6.0, 4.0), (3.0, 4.0), (3.0, 6.0), (0.0, 6.0)],
        ),
    ]

    print("=== Fixed Cases ===")
    for i, case in enumerate(cases, start=1):
        clipped = sutherland_hodgman(case.subject, case.clipper)
        assert polygons_equivalent(clipped, case.expected), (
            f"Case '{case.name}' mismatch\nexpected={case.expected}\ngot={clipped}"
        )
        assert_polygon_inside_clip(clipped, case.clipper)
        print(f"[{i}] {case.name}:")
        print(f"    subject={case.subject}")
        print(f"    clipped={clipped}")


def run_random_property_checks(seed: int = 253, n_cases: int = 200) -> None:
    """Randomized checks for geometric sanity (not a formal proof)."""
    rng = np.random.default_rng(seed)
    clipper: Polygon = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]

    for _ in range(n_cases):
        n_vertices = int(rng.integers(3, 9))
        center = rng.uniform(low=-2.0, high=14.0, size=2)
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n_vertices))
        radii = rng.uniform(0.5, 6.5, size=n_vertices)

        points = np.column_stack(
            [
                center[0] + radii * np.cos(angles),
                center[1] + radii * np.sin(angles),
            ]
        )
        subject: Polygon = [tuple(map(float, row)) for row in points.tolist()]

        clipped = sutherland_hodgman(subject, clipper)

        if clipped:
            assert_polygon_inside_clip(clipped, clipper)
            if not np.isfinite(np.asarray(clipped, dtype=np.float64)).all():
                raise AssertionError(f"Non-finite coordinates detected: {clipped}")
        clipped_area = abs(polygon_signed_area(clipped))
        clipper_area = abs(polygon_signed_area(clipper))
        if clipped_area > clipper_area + 1e-6:
            raise AssertionError(
                f"Clipped area cannot exceed clipper area: clipped={clipped_area}, clipper={clipper_area}"
            )


def main() -> None:
    run_fixed_cases()
    run_random_property_checks()

    showcase_subject: Polygon = [(-2.0, 2.0), (4.0, 12.0), (12.0, 6.0), (8.0, -1.0)]
    showcase_clipper: Polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)]
    showcase_result = sutherland_hodgman(showcase_subject, showcase_clipper)

    print("\n=== Showcase ===")
    print(f"subject={showcase_subject}")
    print(f"clipper={showcase_clipper}")
    print(f"clipped={showcase_result}")
    print(
        "areas: "
        f"subject={abs(polygon_signed_area(showcase_subject)):.3f}, "
        f"clipped={abs(polygon_signed_area(showcase_result)):.3f}"
    )

    print("\nAll checks passed for CS-0253 (Sutherland-Hodgman裁剪).")


if __name__ == "__main__":
    main()
