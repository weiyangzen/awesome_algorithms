"""Minimal runnable MVP for A* shortest path on a weighted 2D grid."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

Coord = tuple[int, int]


@dataclass
class AStarResult:
    """Container for A* search diagnostics and solution path."""

    found: bool
    path: list[Coord]
    total_cost: float
    expanded_nodes: int
    pushed_nodes: int
    closed_nodes: int


def parse_ascii_map(lines: list[str]) -> tuple[np.ndarray, np.ndarray, Coord, Coord]:
    """Parse map text into walkable mask, terrain cost matrix, and start/goal."""
    if not lines:
        raise ValueError("Map cannot be empty.")

    width = len(lines[0])
    if width == 0:
        raise ValueError("Map row cannot be empty.")
    if any(len(row) != width for row in lines):
        raise ValueError("Map must be rectangular.")

    n_rows, n_cols = len(lines), width
    walkable = np.zeros((n_rows, n_cols), dtype=bool)
    terrain_cost = np.full((n_rows, n_cols), np.inf, dtype=float)

    start: Coord | None = None
    goal: Coord | None = None

    # Cost model: entering a cell pays that cell's cost.
    cost_table = {
        ".": 1.0,
        "~": 3.0,
        "^": 5.0,
        "S": 1.0,
        "G": 1.0,
    }

    for r, row in enumerate(lines):
        for c, ch in enumerate(row):
            if ch == "#":
                continue
            if ch not in cost_table:
                raise ValueError(f"Unsupported map symbol {ch!r} at ({r}, {c}).")

            walkable[r, c] = True
            terrain_cost[r, c] = cost_table[ch]

            if ch == "S":
                if start is not None:
                    raise ValueError("Map must contain exactly one start 'S'.")
                start = (r, c)
            elif ch == "G":
                if goal is not None:
                    raise ValueError("Map must contain exactly one goal 'G'.")
                goal = (r, c)

    if start is None or goal is None:
        raise ValueError("Map must contain one 'S' and one 'G'.")
    return walkable, terrain_cost, start, goal


def manhattan(a: Coord, b: Coord) -> int:
    """Manhattan distance for 4-neighbor grid moves."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(parent_r: np.ndarray, parent_c: np.ndarray, start: Coord, goal: Coord) -> list[Coord]:
    """Reconstruct path from parent pointers."""
    path: list[Coord] = []
    cur = goal

    while True:
        path.append(cur)
        if cur == start:
            break

        pr = int(parent_r[cur[0], cur[1]])
        pc = int(parent_c[cur[0], cur[1]])
        if pr < 0 or pc < 0:
            raise RuntimeError("Failed to reconstruct path due to broken parent chain.")
        cur = (pr, pc)

    path.reverse()
    return path


def path_cost(path: list[Coord], terrain_cost: np.ndarray) -> float:
    """Compute path total by summing entering-costs except start cell."""
    if len(path) <= 1:
        return 0.0
    return float(sum(float(terrain_cost[r, c]) for r, c in path[1:]))


def astar_search(
    walkable: np.ndarray,
    terrain_cost: np.ndarray,
    start: Coord,
    goal: Coord,
    heuristic_weight: float = 1.0,
) -> AStarResult:
    """Run A* on a weighted 4-neighbor grid."""
    if heuristic_weight < 0.0:
        raise ValueError("heuristic_weight must be non-negative.")
    if walkable.shape != terrain_cost.shape:
        raise ValueError("walkable and terrain_cost must have the same shape.")
    if not walkable[start] or not walkable[goal]:
        raise ValueError("start and goal must be walkable.")
    if np.any((terrain_cost[walkable] <= 0.0) | (~np.isfinite(terrain_cost[walkable]))):
        raise ValueError("All walkable terrain costs must be finite and > 0.")

    n_rows, n_cols = walkable.shape
    min_step_cost = float(np.min(terrain_cost[walkable]))

    def heuristic(node: Coord) -> float:
        return heuristic_weight * min_step_cost * float(manhattan(node, goal))

    g = np.full((n_rows, n_cols), np.inf, dtype=float)
    parent_r = np.full((n_rows, n_cols), -1, dtype=int)
    parent_c = np.full((n_rows, n_cols), -1, dtype=int)
    closed = np.zeros((n_rows, n_cols), dtype=bool)

    g[start] = 0.0
    tie = 0
    heap: list[tuple[float, int, int, int]] = []
    heapq.heappush(heap, (heuristic(start), tie, start[0], start[1]))
    pushed_nodes = 1
    expanded_nodes = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while heap:
        f_cur, _, r, c = heapq.heappop(heap)
        if closed[r, c]:
            continue

        # Skip stale queue entries.
        if f_cur > g[r, c] + heuristic((r, c)) + 1e-12:
            continue

        closed[r, c] = True
        expanded_nodes += 1
        if (r, c) == goal:
            break

        g_rc = g[r, c]
        for dr, dc in directions:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= n_rows or nc < 0 or nc >= n_cols:
                continue
            if not walkable[nr, nc] or closed[nr, nc]:
                continue

            tentative = g_rc + float(terrain_cost[nr, nc])
            if tentative + 1e-12 < g[nr, nc]:
                g[nr, nc] = tentative
                parent_r[nr, nc] = r
                parent_c[nr, nc] = c
                tie += 1
                heapq.heappush(heap, (tentative + heuristic((nr, nc)), tie, nr, nc))
                pushed_nodes += 1

    found = bool(closed[goal])
    if not found:
        return AStarResult(
            found=False,
            path=[],
            total_cost=float(np.inf),
            expanded_nodes=expanded_nodes,
            pushed_nodes=pushed_nodes,
            closed_nodes=int(np.sum(closed)),
        )

    solution_path = reconstruct_path(parent_r=parent_r, parent_c=parent_c, start=start, goal=goal)
    total_cost = float(g[goal])
    return AStarResult(
        found=True,
        path=solution_path,
        total_cost=total_cost,
        expanded_nodes=expanded_nodes,
        pushed_nodes=pushed_nodes,
        closed_nodes=int(np.sum(closed)),
    )


def overlay_path(lines: list[str], path: list[Coord]) -> list[str]:
    """Draw solution path on map with '*' (excluding S/G)."""
    canvas = [list(row) for row in lines]
    for r, c in path:
        if canvas[r][c] in {"S", "G"}:
            continue
        canvas[r][c] = "*"
    return ["".join(row) for row in canvas]


def main() -> None:
    # Deterministic weighted map:
    # '.' cost=1, '~' cost=3, '^' cost=5, '#' blocked, S start, G goal
    ascii_map = [
        "S..#....~...#",
        ".##.#..~~.#.#",
        "...#....~.#..",
        ".~.#.####.#..",
        ".~...#...#..G",
        "....##...#...",
        ".^^....~.....",
    ]

    walkable, terrain_cost, start, goal = parse_ascii_map(ascii_map)

    astar_result = astar_search(
        walkable=walkable,
        terrain_cost=terrain_cost,
        start=start,
        goal=goal,
        heuristic_weight=1.0,
    )
    dijkstra_result = astar_search(
        walkable=walkable,
        terrain_cost=terrain_cost,
        start=start,
        goal=goal,
        heuristic_weight=0.0,
    )

    if not astar_result.found:
        raise AssertionError("A* failed to find a path on a reachable map.")
    if not dijkstra_result.found:
        raise AssertionError("Dijkstra baseline failed to find a path on a reachable map.")

    # Correctness checks: A* with admissible heuristic must match Dijkstra optimum.
    if abs(astar_result.total_cost - dijkstra_result.total_cost) > 1e-9:
        raise AssertionError(
            "A* and Dijkstra costs differ: "
            f"{astar_result.total_cost} vs {dijkstra_result.total_cost}"
        )
    if not astar_result.path or astar_result.path[0] != start or astar_result.path[-1] != goal:
        raise AssertionError("A* path endpoints are invalid.")

    recomputed = path_cost(astar_result.path, terrain_cost)
    if abs(recomputed - astar_result.total_cost) > 1e-9:
        raise AssertionError("Path reconstruction cost mismatch.")

    rendered = overlay_path(ascii_map, astar_result.path)

    print("A* shortest path MVP report")
    print(f"grid_shape                    : {walkable.shape[0]} x {walkable.shape[1]}")
    print(f"start                         : {start}")
    print(f"goal                          : {goal}")
    print(f"astar_found                   : {astar_result.found}")
    print(f"astar_total_cost              : {astar_result.total_cost:.3f}")
    print(f"astar_path_nodes              : {len(astar_result.path)}")
    print(f"astar_expanded_nodes          : {astar_result.expanded_nodes}")
    print(f"astar_pushed_nodes            : {astar_result.pushed_nodes}")
    print(f"dijkstra_total_cost           : {dijkstra_result.total_cost:.3f}")
    print(f"dijkstra_expanded_nodes       : {dijkstra_result.expanded_nodes}")

    print("\nPath coordinates:")
    print(astar_result.path)

    print("\nMap with A* path:")
    for row in rendered:
        print(row)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
