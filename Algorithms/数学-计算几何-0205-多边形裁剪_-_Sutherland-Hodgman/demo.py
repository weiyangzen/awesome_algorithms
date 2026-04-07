"""Sutherland-Hodgman polygon clipping MVP (MATH-0205).

Run:
    python3 demo.py

The implementation uses NumPy only and keeps the clipping pipeline explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ClipResult:
    subject: np.ndarray
    clipper: np.ndarray
    clipped: np.ndarray
    subject_area: float
    clipped_area: float
    is_empty: bool


def as_xy_array(points: Sequence[Sequence[float]], name: str) -> np.ndarray:
    """Validate input as a finite (n, 2) float array."""
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def remove_consecutive_duplicates(poly: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Drop consecutive duplicate points and a repeated closing point."""
    if len(poly) == 0:
        return poly.copy()

    out = [poly[0]]
    for p in poly[1:]:
        if float(np.linalg.norm(p - out[-1])) > eps:
            out.append(p)

    out_arr = np.asarray(out, dtype=float)
    if len(out_arr) > 1 and float(np.linalg.norm(out_arr[0] - out_arr[-1])) <= eps:
        out_arr = out_arr[:-1]
    return out_arr


def cross2(u: np.ndarray, v: np.ndarray) -> float:
    """2D cross product of vectors u x v."""
    return float(u[0] * v[1] - u[1] * v[0])


def cross_points(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product (OA x OB)."""
    return cross2(a - o, b - o)


def polygon_area2(poly: np.ndarray) -> float:
    """Twice signed polygon area."""
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def orientation_sign(poly: np.ndarray, eps: float = 1e-12) -> float:
    """Return +1 for CCW, -1 for CW polygon orientation."""
    area2 = polygon_area2(poly)
    if abs(area2) <= eps:
        raise ValueError("clipper polygon is degenerate (area near zero)")
    return 1.0 if area2 > 0.0 else -1.0


def is_convex_polygon(poly: np.ndarray, eps: float = 1e-12) -> bool:
    """Check strict convexity up to numerical tolerance."""
    n = len(poly)
    if n < 3:
        return False

    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        c = poly[(i + 2) % n]
        z = cross_points(a, b, c)
        if abs(z) <= eps:
            continue
        cur = 1 if z > 0.0 else -1
        if sign == 0:
            sign = cur
        elif cur != sign:
            return False
    return sign != 0


def inside_half_plane(
    p: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    clip_orientation: float,
    eps: float,
) -> bool:
    """Check whether point is inside the directed clipping half-plane."""
    z = cross_points(edge_a, edge_b, p)
    return clip_orientation * z >= -eps


def segment_line_intersection(
    s: np.ndarray,
    e: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Intersect segment SE with the infinite line through A->B."""
    r = e - s
    d = b - a
    denom = cross2(r, d)

    if abs(denom) <= eps:
        # Nearly parallel: return midpoint as a stable fallback.
        return 0.5 * (s + e)

    t = cross2(a - s, d) / denom
    return s + t * r


def clip_against_one_edge(
    subject: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    clip_orientation: float,
    eps: float,
) -> np.ndarray:
    """Clip a polygon against one edge of the clipping polygon."""
    if len(subject) == 0:
        return np.empty((0, 2), dtype=float)

    out = []
    prev = subject[-1]
    prev_inside = inside_half_plane(prev, edge_a, edge_b, clip_orientation, eps)

    for curr in subject:
        curr_inside = inside_half_plane(curr, edge_a, edge_b, clip_orientation, eps)

        if prev_inside and curr_inside:
            out.append(curr)
        elif prev_inside and not curr_inside:
            out.append(segment_line_intersection(prev, curr, edge_a, edge_b, eps))
        elif (not prev_inside) and curr_inside:
            out.append(segment_line_intersection(prev, curr, edge_a, edge_b, eps))
            out.append(curr)

        prev = curr
        prev_inside = curr_inside

    if not out:
        return np.empty((0, 2), dtype=float)

    return remove_consecutive_duplicates(np.asarray(out, dtype=float), eps=eps)


def sutherland_hodgman(
    subject_polygon: Sequence[Sequence[float]],
    clip_polygon: Sequence[Sequence[float]],
    eps: float = 1e-9,
) -> ClipResult:
    """Clip subject polygon by a convex clip polygon."""
    subject = remove_consecutive_duplicates(as_xy_array(subject_polygon, "subject_polygon"), eps=eps)
    clipper = remove_consecutive_duplicates(as_xy_array(clip_polygon, "clip_polygon"), eps=eps)

    if len(clipper) < 3:
        raise ValueError("clip_polygon must have at least 3 vertices")
    if not is_convex_polygon(clipper, eps=eps):
        raise ValueError("clip_polygon must be convex for Sutherland-Hodgman")

    clip_orientation = orientation_sign(clipper, eps=eps)

    output = subject.copy()
    for i in range(len(clipper)):
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]
        output = clip_against_one_edge(output, a, b, clip_orientation=clip_orientation, eps=eps)
        if len(output) == 0:
            break

    output = remove_consecutive_duplicates(output, eps=eps)
    if len(output) >= 3 and polygon_area2(output) < 0.0:
        output = np.flipud(output)

    subject_area = 0.5 * abs(polygon_area2(subject))
    clipped_area = 0.5 * abs(polygon_area2(output))

    return ClipResult(
        subject=subject,
        clipper=clipper,
        clipped=output,
        subject_area=subject_area,
        clipped_area=clipped_area,
        is_empty=len(output) == 0,
    )


def all_vertices_inside_clipper(poly: np.ndarray, clipper: np.ndarray, eps: float = 1e-8) -> bool:
    """Check each vertex is inside/on the convex clipper."""
    if len(poly) == 0:
        return True

    orient = orientation_sign(clipper)
    for p in poly:
        for i in range(len(clipper)):
            a = clipper[i]
            b = clipper[(i + 1) % len(clipper)]
            if orient * cross_points(a, b, p) < -eps:
                return False
    return True


def summarize_case(name: str, subject: np.ndarray, clipper: np.ndarray) -> None:
    result = sutherland_hodgman(subject, clipper)

    print(f"=== {name} ===")
    print(f"subject vertices: {len(result.subject)}")
    print(f"clipper vertices: {len(result.clipper)}")
    print(f"output vertices: {len(result.clipped)}")
    print(f"subject area: {result.subject_area:.6f}")
    print(f"clipped area: {result.clipped_area:.6f}")
    print(f"is empty: {result.is_empty}")
    print(f"output preview:\n{np.array2string(result.clipped, precision=4)}")
    print()

    if len(result.clipped) > 0:
        if not all_vertices_inside_clipper(result.clipped, result.clipper):
            raise RuntimeError("clipped polygon has vertices outside clipper")
        if result.clipped_area > result.subject_area + 1e-7:
            raise RuntimeError("clipped area should not exceed subject area")


def main() -> None:
    # Case 1: concave polygon clipped by an axis-aligned rectangle.
    subject1 = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [2.0, 0.5],
            [4.0, 3.0],
            [0.0, 3.0],
        ],
        dtype=float,
    )
    clipper1 = np.array(
        [
            [1.0, -0.2],
            [3.2, -0.2],
            [3.2, 2.6],
            [1.0, 2.6],
        ],
        dtype=float,
    )

    # Case 2: convex polygon clipped by a clockwise triangle (orientation robustness).
    subject2 = np.array(
        [
            [0.2, 0.3],
            [4.5, 0.6],
            [5.0, 2.2],
            [3.1, 4.3],
            [0.7, 3.9],
            [-0.4, 2.1],
        ],
        dtype=float,
    )
    clipper2 = np.array(
        [
            [4.6, 3.0],
            [0.3, 3.5],
            [1.2, 0.1],
        ],
        dtype=float,
    )

    # Case 3: no overlap.
    subject3 = np.array(
        [
            [-3.0, -2.0],
            [-1.5, -2.0],
            [-1.5, -0.8],
            [-3.0, -0.8],
        ],
        dtype=float,
    )
    clipper3 = np.array(
        [
            [0.0, 0.0],
            [1.5, 0.0],
            [1.5, 1.3],
            [0.0, 1.3],
        ],
        dtype=float,
    )

    summarize_case("Concave clipped by rectangle", subject1, clipper1)
    summarize_case("Convex clipped by CW triangle", subject2, clipper2)
    summarize_case("Disjoint polygons", subject3, clipper3)

    print("Run completed successfully.")


if __name__ == "__main__":
    main()
