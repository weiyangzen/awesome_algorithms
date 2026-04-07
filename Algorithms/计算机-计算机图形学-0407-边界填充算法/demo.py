"""Boundary fill algorithm MVP (non-interactive)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


Coord = Tuple[int, int]


@dataclass(frozen=True)
class FillStats:
    """Summary metrics for one boundary-fill run."""

    seed: Coord
    connectivity: int
    filled_pixels: int
    visited_steps: int


def neighbor_offsets(connectivity: int) -> Sequence[Coord]:
    """Return neighbor offsets for 4- or 8-connectivity."""
    if connectivity == 4:
        return ((1, 0), (-1, 0), (0, 1), (0, -1))
    if connectivity == 8:
        return (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
    raise ValueError("connectivity must be 4 or 8")


def boundary_fill(
    canvas: np.ndarray,
    seed: Coord,
    boundary_value: int,
    fill_value: int,
    connectivity: int = 4,
) -> Tuple[np.ndarray, FillStats]:
    """
    Perform boundary fill from a seed point on a 2D integer canvas.

    Rules:
    - Stop at boundary_value.
    - Skip already-filled cells (fill_value).
    - Fill all reachable cells via given connectivity.
    """
    if not isinstance(canvas, np.ndarray) or canvas.ndim != 2:
        raise ValueError("canvas must be a 2D numpy array")
    if boundary_value == fill_value:
        raise ValueError("boundary_value and fill_value must be different")

    rows, cols = canvas.shape
    sr, sc = seed
    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError("seed is out of bounds")

    offsets = neighbor_offsets(connectivity)
    result = canvas.copy()

    stack: List[Coord] = [seed]
    filled = 0
    steps = 0

    while stack:
        r, c = stack.pop()
        steps += 1

        if r < 0 or r >= rows or c < 0 or c >= cols:
            continue

        value = int(result[r, c])
        if value == boundary_value or value == fill_value:
            continue

        result[r, c] = fill_value
        filled += 1

        for dr, dc in offsets:
            stack.append((r + dr, c + dc))

    stats = FillStats(
        seed=seed,
        connectivity=connectivity,
        filled_pixels=filled,
        visited_steps=steps,
    )
    return result, stats


def build_demo_canvas(height: int = 16, width: int = 28) -> np.ndarray:
    """Create a deterministic canvas with outer and inner boundaries."""
    canvas = np.zeros((height, width), dtype=np.int32)
    boundary = 1

    # Outer rectangular boundary.
    canvas[0, :] = boundary
    canvas[-1, :] = boundary
    canvas[:, 0] = boundary
    canvas[:, -1] = boundary

    # Inner obstacle boundary (small rectangle).
    canvas[6, 9:19] = boundary
    canvas[10, 9:19] = boundary
    canvas[6:11, 9] = boundary
    canvas[6:11, 18] = boundary

    return canvas


def render_canvas(canvas: np.ndarray) -> str:
    """Render integer canvas to ASCII for terminal output."""
    symbols = {0: ".", 1: "#", 2: "*"}
    lines = []
    for row in canvas:
        lines.append("".join(symbols.get(int(v), "?") for v in row))
    return "\n".join(lines)


def run_case(canvas: np.ndarray, seed: Coord, connectivity: int) -> None:
    """Run one fill case and print diagnostics."""
    boundary_value = 1
    fill_value = 2

    filled_canvas, stats = boundary_fill(
        canvas=canvas,
        seed=seed,
        boundary_value=boundary_value,
        fill_value=fill_value,
        connectivity=connectivity,
    )
    boundary_intact = bool(np.all(filled_canvas[canvas == boundary_value] == boundary_value))

    print(f"\n=== Case connectivity={connectivity} ===")
    print(f"seed={stats.seed}")
    print(f"filled_pixels={stats.filled_pixels}")
    print(f"visited_steps={stats.visited_steps}")
    print(f"boundary_intact={boundary_intact}")
    print(render_canvas(filled_canvas))


def main() -> None:
    canvas = build_demo_canvas()
    seed = (2, 2)

    print("Input canvas:")
    print(render_canvas(canvas))

    run_case(canvas, seed=seed, connectivity=4)
    run_case(canvas, seed=seed, connectivity=8)


if __name__ == "__main__":
    main()
