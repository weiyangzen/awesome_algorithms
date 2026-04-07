"""Scanline polygon fill MVP (non-interactive)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


Point = Tuple[float, float]
Pixel = Tuple[int, int]


@dataclass
class EdgeRecord:
    """Active edge item for scanline filling."""

    y_end: int
    x: float
    inv_slope: float


@dataclass(frozen=True)
class FillStats:
    """Summary statistics for one scanline fill run."""

    edges_total: int
    ignored_horizontal_edges: int
    scanlines_processed: int
    filled_segments: int
    filled_pixels: int
    odd_intersection_rows: int


def build_edge_table(
    vertices: Sequence[Point],
    height: int,
) -> Tuple[Dict[int, List[EdgeRecord]], int, int, int, int]:
    """
    Build edge table grouped by scanline start.

    Returns:
        edge_table: {y_start: [EdgeRecord, ...]}
        y_min: first scanline to process (inclusive)
        y_max: last scanline bound (exclusive)
        edges_total: non-horizontal edge count
        ignored_horizontal: skipped horizontal edge count
    """
    if len(vertices) < 3:
        raise ValueError("polygon must contain at least 3 vertices")

    edge_table: Dict[int, List[EdgeRecord]] = {}
    y_min = height
    y_max = 0
    edges_total = 0
    ignored_horizontal = 0

    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        if y1 == y2:
            ignored_horizontal += 1
            continue

        edges_total += 1

        if y1 < y2:
            x_at_ymin = x1
            y_low = y1
            y_high = y2
            inv_slope = (x2 - x1) / (y2 - y1)
        else:
            x_at_ymin = x2
            y_low = y2
            y_high = y1
            inv_slope = (x1 - x2) / (y1 - y2)

        # Pixel-center convention: row y corresponds to center y+0.5.
        y_start = max(0, int(math.ceil(y_low - 0.5)))
        y_end = min(height, int(math.ceil(y_high - 0.5)))
        if y_start >= y_end:
            continue

        first_center_y = y_start + 0.5
        x_start = x_at_ymin + (first_center_y - y_low) * inv_slope

        edge_table.setdefault(y_start, []).append(
            EdgeRecord(y_end=y_end, x=x_start, inv_slope=inv_slope)
        )

        y_min = min(y_min, y_start)
        y_max = max(y_max, y_end)

    if y_min >= y_max:
        y_min = 0
        y_max = 0

    return edge_table, y_min, y_max, edges_total, ignored_horizontal


def scanline_fill_mask(
    vertices: Sequence[Point],
    height: int,
    width: int,
) -> Tuple[np.ndarray, FillStats]:
    """Compute fill mask using scanline ET/AET algorithm."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    (
        edge_table,
        y_min,
        y_max,
        edges_total,
        ignored_horizontal,
    ) = build_edge_table(vertices, height)

    mask = np.zeros((height, width), dtype=bool)
    active_edges: List[EdgeRecord] = []

    scanlines_processed = 0
    filled_segments = 0
    odd_intersection_rows = 0

    for y in range(y_min, y_max):
        active_edges.extend(edge_table.get(y, []))
        active_edges = [edge for edge in active_edges if y < edge.y_end]
        active_edges.sort(key=lambda edge: (edge.x, edge.inv_slope))

        scanlines_processed += 1
        if len(active_edges) % 2 == 1:
            odd_intersection_rows += 1

        for i in range(0, len(active_edges) - 1, 2):
            x_left = active_edges[i].x
            x_right = active_edges[i + 1].x
            if x_right < x_left:
                x_left, x_right = x_right, x_left

            x_start = max(0, int(math.ceil(x_left - 0.5)))
            x_end = min(width, int(math.ceil(x_right - 0.5)))

            if x_start < x_end:
                mask[y, x_start:x_end] = True
                filled_segments += 1

        for edge in active_edges:
            edge.x += edge.inv_slope

    stats = FillStats(
        edges_total=edges_total,
        ignored_horizontal_edges=ignored_horizontal,
        scanlines_processed=scanlines_processed,
        filled_segments=filled_segments,
        filled_pixels=int(np.count_nonzero(mask)),
        odd_intersection_rows=odd_intersection_rows,
    )
    return mask, stats


def fill_polygon(
    canvas: np.ndarray,
    vertices: Sequence[Point],
    fill_value: int,
) -> Tuple[np.ndarray, np.ndarray, FillStats]:
    """Fill polygon on a canvas and return (result, mask, stats)."""
    if not isinstance(canvas, np.ndarray) or canvas.ndim != 2:
        raise ValueError("canvas must be a 2D numpy array")

    height, width = canvas.shape
    mask, stats = scanline_fill_mask(vertices, height=height, width=width)

    result = canvas.copy()
    result[mask] = int(fill_value)
    return result, mask, stats


def point_in_polygon_even_odd(xc: float, yc: float, vertices: Sequence[Point]) -> bool:
    """Return True iff point is inside polygon under even-odd rule."""
    inside = False
    n = len(vertices)

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        if (y1 > yc) == (y2 > yc):
            continue

        x_cross = x1 + (yc - y1) * (x2 - x1) / (y2 - y1)
        if xc < x_cross:
            inside = not inside

    return inside


def reference_mask_even_odd(
    vertices: Sequence[Point],
    height: int,
    width: int,
) -> np.ndarray:
    """Brute-force reference mask on pixel centers (for validation)."""
    ref = np.zeros((height, width), dtype=bool)
    for y in range(height):
        yc = y + 0.5
        for x in range(width):
            xc = x + 0.5
            ref[y, x] = point_in_polygon_even_odd(xc, yc, vertices)
    return ref


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Pixel]:
    """Return integer pixels for a line segment via Bresenham."""
    pixels: List[Pixel] = []

    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    x, y = x0, y0
    while True:
        pixels.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

    return pixels


def draw_polygon_outline(
    canvas: np.ndarray,
    vertices: Sequence[Point],
    outline_value: int = 1,
) -> np.ndarray:
    """Draw polygon edges on canvas for visualization only."""
    out = canvas.copy()
    height, width = out.shape

    n = len(vertices)
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        for px, py in bresenham_line(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))):
            if 0 <= py < height and 0 <= px < width:
                out[py, px] = int(outline_value)

    return out


def render_canvas(canvas: np.ndarray) -> str:
    """Render integer canvas to compact ASCII view."""
    symbols = {
        0: ".",
        1: "o",
        5: "#",
    }
    lines = []
    for row in canvas:
        lines.append("".join(symbols.get(int(v), "?") for v in row))
    return "\n".join(lines)


def run_case(name: str, vertices: Sequence[Point], height: int = 18, width: int = 24) -> None:
    """Run one deterministic fill case and print checks."""
    base = np.zeros((height, width), dtype=np.int32)
    outline = draw_polygon_outline(base, vertices, outline_value=1)

    filled, mask, stats = fill_polygon(base, vertices, fill_value=5)
    reference = reference_mask_even_odd(vertices, height=height, width=width)

    matches_reference = bool(np.array_equal(mask, reference))

    visual = outline.copy()
    visual[mask] = 5

    print(f"=== {name} ===")
    print("Outline:")
    print(render_canvas(outline))
    print("\nFilled:")
    print(render_canvas(visual))
    print()
    print(
        "stats:",
        {
            "edges_total": stats.edges_total,
            "ignored_horizontal_edges": stats.ignored_horizontal_edges,
            "scanlines_processed": stats.scanlines_processed,
            "filled_segments": stats.filled_segments,
            "filled_pixels": stats.filled_pixels,
            "odd_intersection_rows": stats.odd_intersection_rows,
            "matches_reference": matches_reference,
        },
    )
    print()


def main() -> None:
    convex_pentagon: List[Point] = [
        (4, 2),
        (16, 3),
        (20, 10),
        (10, 15),
        (3, 11),
    ]

    concave_polygon: List[Point] = [
        (4, 3),
        (19, 3),
        (19, 7),
        (12, 7),
        (12, 13),
        (8, 13),
        (8, 7),
        (4, 7),
    ]

    run_case("Case 1: Convex Pentagon", convex_pentagon)
    run_case("Case 2: Concave Polygon", concave_polygon)


if __name__ == "__main__":
    main()
