"""Breadth-First Search (BFS) MVP, non-interactive."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class BFSResult:
    order: List[int]
    distance: List[int]
    parent: List[int]


class GraphBFS:
    """Simple adjacency-list graph with BFS shortest-path support (unweighted)."""

    def __init__(self, num_vertices: int, edges: Sequence[Tuple[int, int]], directed: bool = False) -> None:
        if num_vertices <= 0:
            raise ValueError("num_vertices must be positive")

        self.n = num_vertices
        self.directed = directed
        self.adj: List[List[int]] = [[] for _ in range(num_vertices)]

        for u, v in edges:
            if not (0 <= u < num_vertices and 0 <= v < num_vertices):
                raise ValueError(f"edge ({u}, {v}) contains out-of-range vertex")
            self.adj[u].append(v)
            if not directed:
                self.adj[v].append(u)

        # Deterministic traversal order and duplicate-edge cleanup.
        for i in range(num_vertices):
            self.adj[i] = sorted(set(self.adj[i]))

    def bfs(self, start: int) -> BFSResult:
        if not (0 <= start < self.n):
            raise ValueError("start vertex out of range")

        distance = [-1] * self.n
        parent = [-1] * self.n
        order: List[int] = []

        queue: deque[int] = deque([start])
        distance[start] = 0

        while queue:
            u = queue.popleft()
            order.append(u)

            for v in self.adj[u]:
                if distance[v] != -1:
                    continue
                distance[v] = distance[u] + 1
                parent[v] = u
                queue.append(v)

        return BFSResult(order=order, distance=distance, parent=parent)

    def shortest_path(self, start: int, target: int) -> List[int]:
        if not (0 <= target < self.n):
            raise ValueError("target vertex out of range")

        result = self.bfs(start)
        if result.distance[target] == -1:
            return []

        path: List[int] = []
        cur = target
        while cur != -1:
            path.append(cur)
            if cur == start:
                break
            cur = result.parent[cur]

        path.reverse()
        return path


def validate_bfs_result(graph: GraphBFS, start: int, result: BFSResult) -> None:
    """Sanity checks for BFS outputs.

    1) start distance is 0
    2) parent chain increases distance by exactly 1
    3) for undirected graphs, reachable edge endpoints differ by at most 1 in level
    """

    assert result.distance[start] == 0, "start node must have distance 0"

    for v in range(graph.n):
        p = result.parent[v]
        if p != -1:
            assert result.distance[v] == result.distance[p] + 1, "invalid parent-distance relation"

    if not graph.directed:
        for u in range(graph.n):
            for v in graph.adj[u]:
                if result.distance[u] == -1 or result.distance[v] == -1:
                    continue
                assert abs(result.distance[u] - result.distance[v]) <= 1, "invalid BFS layering on edge"


def run_case(
    name: str,
    graph: GraphBFS,
    start: int,
    target: int,
) -> None:
    print(f"=== {name} ===")
    print(f"n={graph.n}, directed={graph.directed}")
    print("adjacency:")
    for i, neighbors in enumerate(graph.adj):
        print(f"  {i}: {neighbors}")

    result = graph.bfs(start)
    validate_bfs_result(graph, start, result)

    print(f"start={start}")
    print(f"bfs_order={result.order}")
    print(f"distance={result.distance}")
    print(f"parent={result.parent}")

    path = graph.shortest_path(start, target)
    if path:
        print(f"shortest_path({start}->{target})={path}, hops={len(path) - 1}")
    else:
        print(f"shortest_path({start}->{target})=UNREACHABLE")

    print("validation=PASS")
    print()


def main() -> None:
    # Case 1: undirected connected graph.
    edges1 = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (3, 4),
        (2, 5),
        (5, 6),
    ]
    g1 = GraphBFS(num_vertices=7, edges=edges1, directed=False)
    run_case("Undirected-Connected", g1, start=0, target=6)

    # Case 2: undirected disconnected graph.
    edges2 = [
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
    ]
    g2 = GraphBFS(num_vertices=6, edges=edges2, directed=False)
    run_case("Undirected-Disconnected", g2, start=0, target=5)


if __name__ == "__main__":
    main()
