"""A* Search MVP demo.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from itertools import count
from typing import Iterable, Sequence

Coord = tuple[int, int]
Grid = Sequence[Sequence[int]]


@dataclass(frozen=True)
class SearchResult:
    """Result of one A* search run."""

    found: bool
    path: list[Coord]
    cost: int
    expanded_nodes: int


def _validate_grid(grid: Grid, start: Coord, goal: Coord) -> None:
    if not grid or not grid[0]:
        raise ValueError("grid must be a non-empty rectangle")

    row_len = len(grid[0])
    for r, row in enumerate(grid):
        if len(row) != row_len:
            raise ValueError(f"grid must be rectangular: row 0 has len={row_len}, row {r} has len={len(row)}")
        for c, value in enumerate(row):
            if value not in (0, 1):
                raise ValueError(f"grid values must be 0 (free) or 1 (blocked), but grid[{r}][{c}]={value}")

    rows, cols = len(grid), len(grid[0])
    for name, point in (("start", start), ("goal", goal)):
        r, c = point
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"{name}={point} out of bounds for grid size {rows}x{cols}")
        if grid[r][c] == 1:
            raise ValueError(f"{name}={point} lies on a blocked cell")


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors4(grid: Grid, node: Coord) -> Iterable[Coord]:
    rows, cols = len(grid), len(grid[0])
    r, c = node

    # Fixed order keeps the demo deterministic.
    for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            yield (nr, nc)


def _reconstruct_path(came_from: dict[Coord, Coord], goal: Coord) -> list[Coord]:
    path = [goal]
    current = goal
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_search(grid: Grid, start: Coord, goal: Coord) -> SearchResult:
    """Run A* on a 0/1 grid with 4-neighborhood and unit edge cost."""
    _validate_grid(grid, start, goal)

    if start == goal:
        return SearchResult(found=True, path=[start], cost=0, expanded_nodes=0)

    g_score: dict[Coord, int] = {start: 0}
    came_from: dict[Coord, Coord] = {}

    open_heap: list[tuple[int, int, Coord]] = []
    tie_breaker = count()
    heapq.heappush(open_heap, (_manhattan(start, goal), next(tie_breaker), start))

    closed: set[Coord] = set()
    expanded_nodes = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)
        expanded_nodes += 1

        if current == goal:
            path = _reconstruct_path(came_from, goal)
            return SearchResult(found=True, path=path, cost=g_score[goal], expanded_nodes=expanded_nodes)

        current_g = g_score[current]
        for neighbor in _neighbors4(grid, current):
            if neighbor in closed:
                continue

            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, 10**18):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + _manhattan(neighbor, goal)
                heapq.heappush(open_heap, (f_score, next(tie_breaker), neighbor))

    return SearchResult(found=False, path=[], cost=-1, expanded_nodes=expanded_nodes)


def _bfs_shortest_distance(grid: Grid, start: Coord, goal: Coord) -> int | None:
    """Reference shortest-path distance for validation (unit-cost BFS)."""
    if start == goal:
        return 0

    queue: deque[tuple[Coord, int]] = deque([(start, 0)])
    seen: set[Coord] = {start}

    while queue:
        node, dist = queue.popleft()
        for nxt in _neighbors4(grid, node):
            if nxt in seen:
                continue
            if nxt == goal:
                return dist + 1
            seen.add(nxt)
            queue.append((nxt, dist + 1))

    return None


def _is_path_valid(grid: Grid, path: Sequence[Coord], start: Coord, goal: Coord) -> bool:
    if not path:
        return False
    if path[0] != start or path[-1] != goal:
        return False

    rows, cols = len(grid), len(grid[0])
    for i, (r, c) in enumerate(path):
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        if grid[r][c] == 1:
            return False
        if i > 0:
            pr, pc = path[i - 1]
            if abs(pr - r) + abs(pc - c) != 1:
                return False
    return True


def _render_grid_with_path(grid: Grid, start: Coord, goal: Coord, path: Sequence[Coord]) -> str:
    path_set = set(path)
    lines: list[str] = []
    for r, row in enumerate(grid):
        chars: list[str] = []
        for c, cell in enumerate(row):
            point = (r, c)
            if point == start:
                chars.append("S")
            elif point == goal:
                chars.append("G")
            elif cell == 1:
                chars.append("#")
            elif point in path_set:
                chars.append("*")
            else:
                chars.append(".")
        lines.append("".join(chars))
    return "\n".join(lines)


def main() -> None:
    reachable_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]

    blocked_grid = [
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [1, 0, 1, 0],
    ]

    cases = [
        ("reachable", reachable_grid, (0, 0), (4, 5), True),
        ("unreachable", blocked_grid, (0, 0), (3, 3), False),
        ("start_is_goal", reachable_grid, (2, 2), (2, 2), True),
    ]

    print("A* Search MVP demo")
    print("=" * 40)

    for idx, (name, grid, start, goal, should_find) in enumerate(cases, start=1):
        result = astar_search(grid, start, goal)
        bfs_dist = _bfs_shortest_distance(grid, start, goal)

        if result.found != should_find:
            raise AssertionError(
                f"Case {idx} ({name}): expected found={should_find}, got found={result.found}"
            )

        if result.found:
            if not _is_path_valid(grid, result.path, start, goal):
                raise AssertionError(f"Case {idx} ({name}): invalid path returned: {result.path}")

            if bfs_dist is None:
                raise AssertionError(f"Case {idx} ({name}): BFS says unreachable but A* found a path")

            if result.cost != len(result.path) - 1:
                raise AssertionError(
                    f"Case {idx} ({name}): cost={result.cost} but path length implies {len(result.path) - 1}"
                )

            if result.cost != bfs_dist:
                raise AssertionError(
                    f"Case {idx} ({name}): A* cost={result.cost} != BFS shortest={bfs_dist}"
                )

            print(f"Case {idx:02d} [{name}] -> found=True, cost={result.cost}, expanded={result.expanded_nodes}")
            print(_render_grid_with_path(grid, start, goal, result.path))
        else:
            if bfs_dist is not None:
                raise AssertionError(f"Case {idx} ({name}): BFS found {bfs_dist}, but A* reported unreachable")
            if result.path or result.cost != -1:
                raise AssertionError(
                    f"Case {idx} ({name}): expected empty path and cost=-1, got path={result.path}, cost={result.cost}"
                )
            print(f"Case {idx:02d} [{name}] -> found=False, expanded={result.expanded_nodes}")
            print(_render_grid_with_path(grid, start, goal, []))

        print("-" * 40)

    # Explicit input-validation check.
    try:
        astar_search([[0, 1], [1]], (0, 0), (1, 0))
        raise AssertionError("Malformed-grid check failed: ValueError was expected")
    except ValueError:
        print("Malformed-grid check: passed (ValueError raised as expected)")

    print("All checks passed.")


if __name__ == "__main__":
    main()
