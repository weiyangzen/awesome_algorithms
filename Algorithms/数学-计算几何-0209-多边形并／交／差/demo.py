"""Minimal runnable MVP for polygon union/intersection/difference (MATH-0209).

This MVP uses a grid-raster approximation to support general simple polygons
(with possible concavity) without external geometry engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class GridSpec:
    """Regular grid metadata used for raster boolean operations."""

    x0: float
    y0: float
    dx: float
    dy: float
    width: int
    height: int


def polygon_area(poly: np.ndarray) -> float:
    """Signed area of polygon vertices (CCW positive)."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def normalize_polygon(poly: np.ndarray) -> np.ndarray:
    """Normalize polygon shape and orientation (remove duplicated end, force CCW)."""
    p = np.asarray(poly, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 3:
        raise ValueError("polygon must be an Nx2 array with N >= 3")

    if np.linalg.norm(p[0] - p[-1]) <= EPS:
        p = p[:-1]
    if p.shape[0] < 3:
        raise ValueError("polygon degenerates after removing closing vertex")

    # Drop adjacent duplicates.
    cleaned: List[np.ndarray] = []
    for pt in p:
        if not cleaned or np.linalg.norm(pt - cleaned[-1]) > EPS:
            cleaned.append(pt)
    p = np.array(cleaned, dtype=np.float64)
    if p.shape[0] < 3:
        raise ValueError("polygon degenerates after dedup")

    if polygon_area(p) < 0:
        p = p[::-1]

    return p


def make_grid(poly_a: np.ndarray, poly_b: np.ndarray, resolution: int = 320, pad_ratio: float = 0.08) -> GridSpec:
    """Create a regular grid covering both polygons with small padding."""
    all_pts = np.vstack([poly_a, poly_b])
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)

    span = np.maximum(max_xy - min_xy, 1e-6)
    pad = pad_ratio * max(span[0], span[1])
    min_xy = min_xy - pad
    max_xy = max_xy + pad
    span = max_xy - min_xy

    if span[0] >= span[1]:
        width = resolution
        height = max(32, int(np.ceil(resolution * span[1] / span[0])))
    else:
        height = resolution
        width = max(32, int(np.ceil(resolution * span[0] / span[1])))

    dx = float(span[0] / width)
    dy = float(span[1] / height)
    return GridSpec(float(min_xy[0]), float(min_xy[1]), dx, dy, width, height)


def grid_centers(spec: GridSpec) -> np.ndarray:
    """Return all grid-cell centers as an (H*W, 2) array."""
    xs = spec.x0 + (np.arange(spec.width, dtype=np.float64) + 0.5) * spec.dx
    ys = spec.y0 + (np.arange(spec.height, dtype=np.float64) + 0.5) * spec.dy
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def points_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon (boundary-inclusive by tolerance)."""
    x = points[:, 0]
    y = points[:, 1]

    inside = np.zeros(points.shape[0], dtype=bool)

    x0 = poly[:, 0]
    y0 = poly[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)

    for xi, yi, xj, yj in zip(x0, y0, x1, y1):
        # Standard crossing rule.
        cond = (yi > y) != (yj > y)
        x_inter = (xj - xi) * (y - yi) / ((yj - yi) + EPS) + xi
        inside ^= cond & (x <= x_inter)

        # Boundary correction: near an edge => inside.
        vx = xj - xi
        vy = yj - yi
        wx = x - xi
        wy = y - yi
        cross = np.abs(vx * wy - vy * wx)
        dot = wx * (x - xj) + wy * (y - yj)
        on_edge = (cross <= 1e-10) & (dot <= 1e-10)
        inside |= on_edge

    return inside


def rasterize_polygon(poly: np.ndarray, spec: GridSpec) -> np.ndarray:
    """Rasterize polygon into boolean occupancy mask (H, W)."""
    pts = grid_centers(spec)
    inside = points_in_polygon(pts, poly)
    return inside.reshape(spec.height, spec.width)


def mask_to_loops(mask: np.ndarray) -> List[np.ndarray]:
    """Extract polygon loops from occupancy mask via oriented cell-boundary tracing.

    Vertices are returned in grid-index coordinates (not world coordinates yet).
    """
    h, w = mask.shape
    edges: set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

    filled_rc = np.argwhere(mask)
    for r, c in filled_rc:
        down = r - 1 >= 0 and mask[r - 1, c]
        right = c + 1 < w and mask[r, c + 1]
        up = r + 1 < h and mask[r + 1, c]
        left = c - 1 >= 0 and mask[r, c - 1]

        # Cell corners in index lattice:
        # (c, r) --- (c+1, r)
        #   |             |
        # (c, r+1)---(c+1,r+1)
        if not down:
            edges.add(((c, r), (c + 1, r)))
        if not right:
            edges.add(((c + 1, r), (c + 1, r + 1)))
        if not up:
            edges.add(((c + 1, r + 1), (c, r + 1)))
        if not left:
            edges.add(((c, r + 1), (c, r)))

    if not edges:
        return []

    outgoing: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for s, e in edges:
        outgoing.setdefault(s, []).append(e)

    unused = set(edges)
    loops: List[np.ndarray] = []

    while unused:
        start_edge = next(iter(unused))
        start, nxt = start_edge

        loop: List[Tuple[int, int]] = [start]
        unused.remove(start_edge)
        curr = nxt

        safe_guard = 0
        while curr != loop[0]:
            loop.append(curr)
            candidates = outgoing.get(curr, [])

            next_vertex = None
            for cand in candidates:
                if (curr, cand) in unused:
                    next_vertex = cand
                    break

            if next_vertex is None:
                # Open chain due to pathological pixel touching; discard.
                loop = []
                break

            unused.remove((curr, next_vertex))
            curr = next_vertex

            safe_guard += 1
            if safe_guard > (len(edges) + 5):
                loop = []
                break

        if len(loop) >= 3:
            loops.append(np.array(loop, dtype=np.float64))

    return loops


def simplify_loop(loop: np.ndarray) -> np.ndarray:
    """Remove consecutive collinear vertices from a loop."""
    n = loop.shape[0]
    keep: List[np.ndarray] = []
    for i in range(n):
        a = loop[i - 1]
        b = loop[i]
        c = loop[(i + 1) % n]
        ab = b - a
        bc = c - b
        cross = ab[0] * bc[1] - ab[1] * bc[0]
        if abs(cross) > EPS:
            keep.append(b)
    if len(keep) < 3:
        return loop
    return np.array(keep, dtype=np.float64)


def loops_to_world(loops: Sequence[np.ndarray], spec: GridSpec) -> List[np.ndarray]:
    """Map lattice-index loops to world coordinates."""
    out: List[np.ndarray] = []
    for lp in loops:
        world = np.empty_like(lp)
        world[:, 0] = spec.x0 + lp[:, 0] * spec.dx
        world[:, 1] = spec.y0 + lp[:, 1] * spec.dy
        world = simplify_loop(world)
        out.append(world)
    return out


def boolean_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute union/intersection/difference masks."""
    return {
        "union": mask_a | mask_b,
        "intersection": mask_a & mask_b,
        "difference": mask_a & (~mask_b),
    }


def run_demo_case(name: str, poly_a: np.ndarray, poly_b: np.ndarray, resolution: int = 320) -> pd.DataFrame:
    """Run one case and print summary for union/intersection/difference."""
    a = normalize_polygon(poly_a)
    b = normalize_polygon(poly_b)

    spec = make_grid(a, b, resolution=resolution)
    mask_a = rasterize_polygon(a, spec)
    mask_b = rasterize_polygon(b, spec)
    ops = boolean_masks(mask_a, mask_b)

    cell_area = spec.dx * spec.dy
    area_a = float(mask_a.sum() * cell_area)
    area_b = float(mask_b.sum() * cell_area)

    rows = []
    for op_name, mask in ops.items():
        loops = loops_to_world(mask_to_loops(mask), spec)
        loop_areas = [abs(polygon_area(lp)) for lp in loops if lp.shape[0] >= 3]
        approx_area = float(mask.sum() * cell_area)
        rows.append(
            {
                "case": name,
                "op": op_name,
                "components": len(loops),
                "mask_area": approx_area,
                "largest_component_area": float(max(loop_areas) if loop_areas else 0.0),
            }
        )

    df = pd.DataFrame(rows)

    # Consistency checks under the same grid discretization.
    a_and_b = ops["intersection"].sum()
    a_or_b = ops["union"].sum()
    a_minus_b = ops["difference"].sum()

    if a_or_b != (mask_a.sum() + mask_b.sum() - a_and_b):
        raise RuntimeError("union area identity failed in raster domain")
    if mask_a.sum() != (a_minus_b + a_and_b):
        raise RuntimeError("difference/intersection identity failed in raster domain")

    print(f"\n[{name}] grid={spec.width}x{spec.height}, cell_area={cell_area:.6f}")
    print(f"A_area≈{area_a:.6f}, B_area≈{area_b:.6f}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    return df


def main() -> None:
    print("Polygon Boolean Operations MVP (MATH-0209)")
    print("Method: rasterization + boolean mask ops + contour tracing")
    print("=" * 72)

    # Case 1: concave polygon vs rectangle-like polygon.
    poly_a1 = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 3.0],
            [2.6, 3.0],
            [2.6, 1.4],
            [1.4, 1.4],
            [1.4, 3.0],
            [0.0, 3.0],
        ]
    )
    poly_b1 = np.array(
        [
            [1.0, -0.4],
            [5.0, -0.1],
            [5.2, 2.2],
            [1.3, 2.5],
        ]
    )

    # Case 2: star-ish concave polygon vs triangle.
    poly_a2 = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.5],
            [2.0, 0.0],
            [0.3, 1.6],
            [2.3, 1.6],
        ]
    )
    poly_b2 = np.array(
        [
            [0.7, -0.4],
            [2.8, 0.8],
            [1.2, 2.8],
        ]
    )

    df1 = run_demo_case("concave-vs-quad", poly_a1, poly_b1, resolution=360)
    df2 = run_demo_case("star-vs-triangle", poly_a2, poly_b2, resolution=360)

    combined = pd.concat([df1, df2], ignore_index=True)
    print("\nOverall summary")
    print(combined.groupby("op", as_index=False)["mask_area"].mean().to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
