"""IDA* (Iterative Deepening A*) minimal runnable MVP.

The demo builds a 2D grid graph with obstacles and finds a shortest path
from start to goal with IDA* using Manhattan distance as an admissible
heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Callable

import numpy as np

Node = tuple[int, int]
Graph = dict[Node, list[tuple[Node, float]]]


@dataclass
class SearchResult:
    found: bool
    path: list[Node]
    cost: float
    expanded_nodes: int
    iterations: int
    bounds: list[float]


def manhattan(a: Node, b: Node) -> float:
    """L1 distance computed with numpy for a compact numeric implementation."""
    return float(np.abs(np.array(a) - np.array(b)).sum())


def build_grid_graph(rows: int, cols: int, blocked: set[Node]) -> Graph:
    """Build a 4-neighbor weighted graph (unit edge cost) from a grid."""
    graph: Graph = {}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(rows):
        for c in range(cols):
            node = (r, c)
            if node in blocked:
                continue
            neighbors: list[tuple[Node, float]] = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if 0 <= nr < rows and 0 <= nc < cols and nxt not in blocked:
                    neighbors.append((nxt, 1.0))
            graph[node] = neighbors
    return graph


def ida_star(
    graph: Graph,
    start: Node,
    goal: Node,
    heuristic: Callable[[Node], float],
) -> SearchResult:
    """Find a shortest path by IDA* under a non-negative edge-weight graph."""
    bound = heuristic(start)
    expanded_nodes = 0
    iterations = 0
    bounds: list[float] = []

    def dfs(
        node: Node,
        g: float,
        current_bound: float,
        path: list[Node],
        in_path: set[Node],
    ) -> tuple[float, list[Node] | None, float | None]:
        nonlocal expanded_nodes

        f = g + heuristic(node)
        if f > current_bound:
            return f, None, None
        if node == goal:
            return g, path.copy(), g

        expanded_nodes += 1
        min_over_bound = inf

        ordered_neighbors = sorted(
            graph.get(node, []),
            key=lambda item: g + item[1] + heuristic(item[0]),
        )

        for nxt, edge_cost in ordered_neighbors:
            if nxt in in_path:
                continue
            path.append(nxt)
            in_path.add(nxt)

            t, found_path, found_cost = dfs(
                node=nxt,
                g=g + edge_cost,
                current_bound=current_bound,
                path=path,
                in_path=in_path,
            )
            if found_path is not None and found_cost is not None:
                return t, found_path, found_cost

            if t < min_over_bound:
                min_over_bound = t

            path.pop()
            in_path.remove(nxt)

        return min_over_bound, None, None

    while True:
        iterations += 1
        bounds.append(bound)
        t, found_path, found_cost = dfs(
            node=start,
            g=0.0,
            current_bound=bound,
            path=[start],
            in_path={start},
        )

        if found_path is not None and found_cost is not None:
            return SearchResult(
                found=True,
                path=found_path,
                cost=found_cost,
                expanded_nodes=expanded_nodes,
                iterations=iterations,
                bounds=bounds,
            )

        if t == inf:
            return SearchResult(
                found=False,
                path=[],
                cost=inf,
                expanded_nodes=expanded_nodes,
                iterations=iterations,
                bounds=bounds,
            )

        bound = t


def render_grid(
    rows: int,
    cols: int,
    blocked: set[Node],
    path: list[Node],
    start: Node,
    goal: Node,
) -> str:
    path_set = set(path)
    lines: list[str] = []
    for r in range(rows):
        chars: list[str] = []
        for c in range(cols):
            node = (r, c)
            if node == start:
                chars.append("S")
            elif node == goal:
                chars.append("G")
            elif node in blocked:
                chars.append("#")
            elif node in path_set:
                chars.append("*")
            else:
                chars.append(".")
        lines.append(" ".join(chars))
    return "\n".join(lines)


def main() -> None:
    rows, cols = 8, 10
    start = (0, 0)
    goal = (7, 9)

    blocked: set[Node] = {
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 7),
        (2, 3),
        (2, 7),
        (3, 3),
        (3, 5),
        (3, 6),
        (3, 7),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 7),
        (5, 7),
        (6, 4),
        (6, 5),
        (6, 7),
    }

    if start in blocked or goal in blocked:
        raise ValueError("start/goal must be in free cells")

    graph = build_grid_graph(rows=rows, cols=cols, blocked=blocked)
    heuristic = lambda node: manhattan(node, goal)
    result = ida_star(graph=graph, start=start, goal=goal, heuristic=heuristic)

    print("=== IDA* Demo (Grid Pathfinding) ===")
    print(f"start={start}, goal={goal}")
    print(f"found={result.found}")
    print(f"iterations={result.iterations}")
    print(f"bounds={result.bounds}")
    print(f"expanded_nodes={result.expanded_nodes}")

    if result.found:
        print(f"path_length={len(result.path)}")
        print(f"path_cost={result.cost}")
        print(f"path={result.path}")
        print("grid:")
        print(render_grid(rows, cols, blocked, result.path, start, goal))
    else:
        print("No path exists under current grid obstacles.")


if __name__ == "__main__":
    main()
