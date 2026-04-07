"""Minimal runnable MVP for Chan's convex hull algorithm (MATH-0186).

This implementation is intentionally transparent:
- subgroup hulls are built with Andrew monotonic chain;
- the outer wrapping follows Chan's "guess m, then wrap up to m steps" idea;
- per-subhull support point search is linear (not binary tangent),
  which keeps the code short and easy to audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
EPS = 1e-12


@dataclass(frozen=True)
class ChanResult:
    """Convex hull result and lightweight diagnostics."""

    hull: List[Point]
    rounds: int
    last_m: int


def cross(o: Point, a: Point, b: Point) -> float:
    """2D cross product (OA x OB). Positive means left turn."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def dist2(a: Point, b: Point) -> float:
    """Squared Euclidean distance."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def unique_points(points: Iterable[Point]) -> List[Point]:
    """Deduplicate and lexicographically sort points."""
    return sorted(set(points))


def monotonic_chain(points: Sequence[Point]) -> List[Point]:
    """Andrew monotonic chain hull in CCW order, no repeated first point."""
    pts = unique_points(points)
    if len(pts) <= 1:
        return pts

    lower: List[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= EPS:
            lower.pop()
        lower.append(p)

    upper: List[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= EPS:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if not hull:
        return pts[:1]
    return hull


def is_better_candidate(pivot: Point, current: Point, challenger: Point) -> bool:
    """Return True if challenger is a better next hull point than current.

    We maintain CCW hull order, so for edge pivot->next we want all points
    to be on the left side. During scanning, if challenger is to the right
    of pivot->current, it is a better candidate.
    """
    c = cross(pivot, current, challenger)
    if c < -EPS:
        return True
    if abs(c) <= EPS and dist2(pivot, challenger) > dist2(pivot, current) + EPS:
        return True
    return False


def support_point_linear(pivot: Point, hull: Sequence[Point]) -> Optional[Point]:
    """Pick the best wrapping candidate on one sub-hull via linear scan."""
    best: Optional[Point] = None
    for q in hull:
        if q == pivot:
            continue
        if best is None or is_better_candidate(pivot, best, q):
            best = q
    return best


def chan_once(points: Sequence[Point], m: int) -> Optional[List[Point]]:
    """Try one Chan round with cap m. Return hull if closed within <= m steps."""
    n = len(points)
    if n <= 1:
        return list(points)

    groups = [points[i : i + m] for i in range(0, n, m)]
    sub_hulls = [monotonic_chain(g) for g in groups]

    start = min(points)
    hull: List[Point] = [start]
    current = start

    for _ in range(m):
        nxt: Optional[Point] = None
        for h in sub_hulls:
            cand = support_point_linear(current, h)
            if cand is None:
                continue
            if nxt is None or is_better_candidate(current, nxt, cand):
                nxt = cand

        if nxt is None:
            # Degenerate: all points identical.
            return hull
        if nxt == start:
            return hull

        hull.append(nxt)
        current = nxt

    # Not closed within m steps.
    return None


def chan_convex_hull(points: Sequence[Point]) -> ChanResult:
    """Compute convex hull with a Chan-style outer loop.

    Note:
        This educational MVP uses linear support-point search on each sub-hull,
        so it is not the fully optimal O(n log h) implementation.
    """
    pts = unique_points(points)
    n = len(pts)
    if n <= 2:
        return ChanResult(hull=pts, rounds=0, last_m=n)

    # Doubling m is more practical here because support search is linear.
    m = 4
    rounds = 0
    while True:
        rounds += 1
        cap = min(m, n)
        hull = chan_once(pts, cap)
        if hull is not None:
            return ChanResult(hull=hull, rounds=rounds, last_m=cap)

        if cap == n:
            # Fallback should not be needed, but keeps behavior safe.
            return ChanResult(hull=monotonic_chain(pts), rounds=rounds, last_m=cap)

        m *= 2


def normalize_cycle(hull: Sequence[Point]) -> List[Point]:
    """Rotate hull to start from lexicographically smallest point and force CCW."""
    if not hull:
        return []
    if len(hull) <= 2:
        return sorted(hull)

    idx = min(range(len(hull)), key=lambda i: hull[i])
    rot = list(hull[idx:]) + list(hull[:idx])

    if cross(rot[0], rot[1], rot[2]) < 0:
        rot = [rot[0]] + list(reversed(rot[1:]))
    return rot


def points_inside_or_on_hull(points: Sequence[Point], hull: Sequence[Point]) -> bool:
    """Check that every point lies on/inside a CCW convex polygon hull."""
    if len(hull) <= 2:
        if len(hull) == 0:
            return len(points) == 0
        if len(hull) == 1:
            return all(p == hull[0] for p in points)
        a, b = hull
        for p in points:
            if abs(cross(a, b, p)) > 1e-8:
                return False
        return True

    m = len(hull)
    for p in points:
        for i in range(m):
            a = hull[i]
            b = hull[(i + 1) % m]
            if cross(a, b, p) < -1e-8:
                return False
    return True


def make_demo_points(seed: int = 186) -> List[Point]:
    """Generate a deterministic dataset with noise, collinearity, and duplicates."""
    rng = np.random.default_rng(seed)

    cloud = rng.normal(0.0, 1.0, size=(180, 2))
    transform = np.array([[1.9, 0.45], [-0.25, 1.35]])
    cloud = cloud @ transform

    theta = np.linspace(0.0, 2.0 * np.pi, num=24, endpoint=False)
    outer = np.stack(
        [5.4 * np.cos(theta) + 0.35 * np.sin(3 * theta), 4.1 * np.sin(theta)],
        axis=1,
    )

    collinear = np.column_stack([
        np.linspace(-6.0, 6.0, num=17),
        np.linspace(-2.5, 2.5, num=17),
    ])

    duplicates = outer[:6].copy()

    pts = np.vstack([cloud, outer, collinear, duplicates])
    return [tuple(map(float, row)) for row in pts]


def main() -> None:
    print("Chan Convex Hull MVP (MATH-0186)")
    print("=" * 64)

    points = make_demo_points(seed=186)
    uniq = unique_points(points)

    chan_res = chan_convex_hull(uniq)
    chan_hull = normalize_cycle(chan_res.hull)

    # Reference hull for validation.
    ref_hull = normalize_cycle(monotonic_chain(uniq))

    print(f"input points: {len(points)} (unique: {len(uniq)})")
    print(f"Chan rounds: {chan_res.rounds}, last m: {chan_res.last_m}")
    print(f"hull size (Chan): {len(chan_hull)}")
    print(f"hull size (Reference): {len(ref_hull)}")

    print("First 8 hull vertices (x, y):")
    for p in chan_hull[:8]:
        print(f"  ({p[0]:8.4f}, {p[1]:8.4f})")

    same_vertices = set(chan_hull) == set(ref_hull)
    same_size = len(chan_hull) == len(ref_hull)
    contain_ok = points_inside_or_on_hull(uniq, chan_hull)

    if not same_size or not same_vertices:
        raise RuntimeError("Chan hull does not match reference hull vertex set")
    if not contain_ok:
        raise RuntimeError("Computed hull does not contain all input points")

    print("All checks passed.")
    print("=" * 64)


if __name__ == "__main__":
    main()
