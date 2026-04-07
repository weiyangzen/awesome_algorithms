"""A* pathfinding MVP on 2D grid with deterministic demo cases."""

from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[int, int]


def validate_grid(grid: np.ndarray, start: Point, goal: Point) -> np.ndarray:
    """Validate grid and endpoint settings for A* search."""
    mat = np.asarray(grid, dtype=int)
    if mat.ndim != 2:
        raise ValueError(f"grid must be 2D, got ndim={mat.ndim}")
    if mat.size == 0:
        raise ValueError("grid must be non-empty")
    if not np.all((mat == 0) | (mat == 1)):
        raise ValueError("grid values must be 0 (free) or 1 (blocked)")

    rows, cols = mat.shape
    for name, p in [("start", start), ("goal", goal)]:
        r, c = p
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"{name} {p} is out of bounds for grid shape {mat.shape}")
        if mat[r, c] == 1:
            raise ValueError(f"{name} {p} is blocked")
    return mat


def manhattan(a: Point, b: Point) -> int:
    """Admissible and consistent heuristic for 4-neighbor unit-cost grids."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors4(mat: np.ndarray, p: Point) -> List[Point]:
    """Return walkable 4-neighbors of a grid cell."""
    rows, cols = mat.shape
    r, c = p
    out: List[Point] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and mat[nr, nc] == 0:
            out.append((nr, nc))
    return out


def reconstruct_path(came_from: Dict[Point, Point], end: Point) -> List[Point]:
    """Reconstruct path from parent map."""
    path: List[Point] = [end]
    cur = end
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def astar_search(grid: np.ndarray, start: Point, goal: Point) -> Tuple[Optional[List[Point]], float, int]:
    """
    Run A* and return (path, cost, expanded_nodes).

    - path: list of points from start to goal if found, otherwise None.
    - cost: path length in unit step costs; inf if no path.
    - expanded_nodes: number of nodes popped into closed set.
    """
    mat = validate_grid(grid, start, goal)

    if start == goal:
        return [start], 0.0, 1

    open_heap: List[Tuple[int, int, Point]] = []
    g_score: Dict[Point, int] = {start: 0}
    came_from: Dict[Point, Point] = {}
    closed_set = set()
    expanded_nodes = 0

    heapq.heappush(open_heap, (manhattan(start, goal), 0, start))

    while open_heap:
        f_cur, g_cur, cur = heapq.heappop(open_heap)
        _ = f_cur  # Keep tuple shape explicit and readable.

        if cur in closed_set:
            continue
        closed_set.add(cur)
        expanded_nodes += 1

        if cur == goal:
            path = reconstruct_path(came_from, goal)
            return path, float(g_cur), expanded_nodes

        for nxt in neighbors4(mat, cur):
            if nxt in closed_set:
                continue

            tentative_g = g_cur + 1
            old_g = g_score.get(nxt, 10**12)
            if tentative_g < old_g:
                g_score[nxt] = tentative_g
                came_from[nxt] = cur
                f_nxt = tentative_g + manhattan(nxt, goal)
                heapq.heappush(open_heap, (f_nxt, tentative_g, nxt))

    return None, float("inf"), expanded_nodes


def bfs_shortest_path_length(grid: np.ndarray, start: Point, goal: Point) -> float:
    """Reference shortest path length on unit-cost grid (for consistency check)."""
    mat = validate_grid(grid, start, goal)
    if start == goal:
        return 0.0

    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        cur, dist = queue.popleft()
        for nxt in neighbors4(mat, cur):
            if nxt in visited:
                continue
            if nxt == goal:
                return float(dist + 1)
            visited.add(nxt)
            queue.append((nxt, dist + 1))
    return float("inf")


def is_valid_path(grid: np.ndarray, path: Sequence[Point], start: Point, goal: Point) -> bool:
    """Check path validity on grid."""
    mat = validate_grid(grid, start, goal)
    if len(path) == 0:
        return False
    if tuple(path[0]) != start or tuple(path[-1]) != goal:
        return False

    for p in path:
        r, c = p
        if mat[r, c] == 1:
            return False

    for a, b in zip(path, path[1:]):
        if manhattan(a, b) != 1:
            return False
    return True


def render_grid(grid: np.ndarray, start: Point, goal: Point, path: Optional[Sequence[Point]]) -> str:
    """Create a compact text map for console visualization."""
    mat = np.asarray(grid, dtype=int)
    canvas = np.full(mat.shape, ".", dtype="<U1")
    canvas[mat == 1] = "#"

    if path is not None:
        for r, c in path:
            if (r, c) != start and (r, c) != goal:
                canvas[r, c] = "*"

    canvas[start[0], start[1]] = "S"
    canvas[goal[0], goal[1]] = "G"
    return "\n".join(" ".join(row) for row in canvas)


def run_case(name: str, grid: np.ndarray, start: Point, goal: Point) -> None:
    """Run one deterministic case and print metrics."""
    print(f"\n=== {name} ===")
    print(f"shape={grid.shape}, start={start}, goal={goal}")

    path, cost, expanded = astar_search(grid, start, goal)
    bfs_cost = bfs_shortest_path_length(grid, start, goal)
    found = path is not None
    valid = is_valid_path(grid, path, start, goal) if path is not None else False
    optimal = (cost == bfs_cost) if found else (bfs_cost == float("inf"))

    print(f"path found         : {found}")
    print(f"astar cost         : {cost}")
    print(f"bfs shortest cost  : {bfs_cost}")
    print(f"optimality check   : {optimal}")
    print(f"path validity      : {valid}")
    print(f"expanded nodes     : {expanded}")
    print("grid:")
    print(render_grid(grid, start, goal, path))
    print(f"path sequence      : {path}")


def main() -> None:
    np.set_printoptions(linewidth=120)

    # Case 1: path exists.
    grid_ok = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 0],
        ],
        dtype=int,
    )
    run_case("Case 1 (reachable)", grid_ok, start=(0, 0), goal=(5, 6))

    # Case 2: no path due to blocking wall.
    grid_blocked = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    run_case("Case 2 (unreachable)", grid_blocked, start=(0, 0), goal=(4, 4))

    # Case 3: boundary condition start == goal.
    grid_same = np.zeros((3, 3), dtype=int)
    run_case("Case 3 (start equals goal)", grid_same, start=(1, 1), goal=(1, 1))

    # Optional invalid-input demonstration.
    bad = np.array([[0, 2], [0, 0]], dtype=int)
    try:
        _ = astar_search(bad, (0, 0), (1, 1))
    except ValueError as exc:
        print("\nExpected failure on invalid grid:")
        print(exc)


if __name__ == "__main__":
    main()
