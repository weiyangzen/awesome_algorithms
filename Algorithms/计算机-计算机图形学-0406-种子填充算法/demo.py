"""Seed fill (flood fill) algorithm MVP (non-interactive)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


Coord = Tuple[int, int]


@dataclass(frozen=True)
class FillStats:
    """Summary metrics for one seed-fill run."""

    seed: Coord
    connectivity: int
    source_value: int
    fill_value: int
    filled_pixels: int
    popped_steps: int


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


def seed_fill(
    canvas: np.ndarray,
    seed: Coord,
    fill_value: int,
    connectivity: int = 4,
) -> Tuple[np.ndarray, FillStats]:
    """
    Fill the connected component that shares seed's original value.

    Expansion rule:
    - Start from seed.
    - Expand to neighbors under given connectivity.
    - A pixel is fillable iff its current value equals source_value.
    """
    if not isinstance(canvas, np.ndarray) or canvas.ndim != 2:
        raise ValueError("canvas must be a 2D numpy array")

    rows, cols = canvas.shape
    sr, sc = seed
    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError("seed is out of bounds")

    offsets = neighbor_offsets(connectivity)
    source_value = int(canvas[sr, sc])

    result = canvas.copy()
    if source_value == int(fill_value):
        stats = FillStats(
            seed=seed,
            connectivity=connectivity,
            source_value=source_value,
            fill_value=int(fill_value),
            filled_pixels=0,
            popped_steps=0,
        )
        return result, stats

    stack: List[Coord] = [seed]
    filled_pixels = 0
    popped_steps = 0

    while stack:
        r, c = stack.pop()
        popped_steps += 1

        if r < 0 or r >= rows or c < 0 or c >= cols:
            continue

        if int(result[r, c]) != source_value:
            continue

        result[r, c] = fill_value
        filled_pixels += 1

        for dr, dc in offsets:
            stack.append((r + dr, c + dc))

    stats = FillStats(
        seed=seed,
        connectivity=connectivity,
        source_value=source_value,
        fill_value=int(fill_value),
        filled_pixels=filled_pixels,
        popped_steps=popped_steps,
    )
    return result, stats


def build_component_canvas(height: int = 12, width: int = 24) -> np.ndarray:
    """Create deterministic canvas with two disconnected same-valued regions."""
    canvas = np.zeros((height, width), dtype=np.int32)

    source = 3
    obstacle = 8

    # Region A (target component).
    canvas[2:6, 2:8] = source
    # Carve a small hole with a different value to prove strict source-value rule.
    canvas[3:5, 4:6] = obstacle

    # Region B (same source value but disconnected from region A).
    canvas[7:10, 15:21] = source

    # Extra unrelated stripe.
    canvas[4:9, 11] = 6

    return canvas


def build_diagonal_canvas(size: int = 7) -> np.ndarray:
    """Create a diagonal chain used to compare 4- and 8-connectivity."""
    canvas = np.zeros((size, size), dtype=np.int32)
    for i in range(1, size - 1):
        canvas[i, i] = 4
    return canvas


def render_canvas(canvas: np.ndarray) -> str:
    """Render integer canvas to compact ASCII symbols."""
    symbols = {
        0: ".",
        2: "*",
        3: "a",
        4: "d",
        6: "x",
        8: "o",
        9: "#",
    }
    lines = []
    for row in canvas:
        lines.append("".join(symbols.get(int(v), "?") for v in row))
    return "\n".join(lines)


def run_component_case() -> None:
    """Run main seed-fill case on disconnected same-valued regions."""
    canvas = build_component_canvas()
    seed = (2, 2)
    fill_value = 2

    filled, stats = seed_fill(canvas, seed=seed, fill_value=fill_value, connectivity=4)

    diff_mask = filled != canvas
    only_source_changed = bool(np.all(canvas[diff_mask] == stats.source_value))
    changed_to_fill_value = bool(np.all(filled[diff_mask] == fill_value))

    disconnected_component_preserved = bool(np.all(filled[7:10, 15:21] == 3))
    expected_pixels = 20

    print("=== Component Case (4-connectivity) ===")
    print("Input:")
    print(render_canvas(canvas))
    print("\nOutput:")
    print(render_canvas(filled))
    print(
        "\n"
        f"source_value={stats.source_value}, fill_value={stats.fill_value}, "
        f"filled_pixels={stats.filled_pixels}, popped_steps={stats.popped_steps}"
    )
    print(f"only_source_changed={only_source_changed}")
    print(f"changed_to_fill_value={changed_to_fill_value}")
    print(f"disconnected_component_preserved={disconnected_component_preserved}")
    print(f"expected_filled_pixels={expected_pixels}")
    print(f"count_check={(stats.filled_pixels == expected_pixels)}")


def run_diagonal_cases() -> None:
    """Run diagonal-only connectivity comparison cases."""
    canvas = build_diagonal_canvas()
    seed = (1, 1)

    fill4, stats4 = seed_fill(canvas, seed=seed, fill_value=9, connectivity=4)
    fill8, stats8 = seed_fill(canvas, seed=seed, fill_value=9, connectivity=8)

    print("\n=== Diagonal Chain Case ===")
    print("Input:")
    print(render_canvas(canvas))

    print("\nOutput (connectivity=4):")
    print(render_canvas(fill4))
    print(f"filled_pixels={stats4.filled_pixels}, expected=1, check={(stats4.filled_pixels == 1)}")

    print("\nOutput (connectivity=8):")
    print(render_canvas(fill8))
    print(f"filled_pixels={stats8.filled_pixels}, expected=5, check={(stats8.filled_pixels == 5)}")


def main() -> None:
    run_component_case()
    run_diagonal_cases()


if __name__ == "__main__":
    main()
