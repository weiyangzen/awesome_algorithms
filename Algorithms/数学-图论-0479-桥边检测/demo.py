"""Bridge detection MVP using Tarjan low-link algorithm.

This script is self-contained and runs without interactive input.
It demonstrates:
- O(V+E) bridge detection on undirected graphs
- support for disconnected graphs and parallel edges
- cross-check with a brute-force remove-edge validator
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Graph:
    """Undirected graph with edge-id based adjacency for robust bridge detection."""

    n: int
    edges: List[Tuple[int, int]]
    adjacency: List[List[Tuple[int, int]]]  # adjacency[u] = [(v, edge_id), ...]

    @property
    def m(self) -> int:
        return len(self.edges)


@dataclass
class BridgeResult:
    bridge_edge_ids: List[int]
    tin: np.ndarray
    low: np.ndarray


def build_undirected_graph(num_vertices: int, edges: Iterable[Tuple[int, int]]) -> Graph:
    """Build an undirected graph; each input edge gets a unique edge id."""
    if num_vertices <= 0:
        raise ValueError("num_vertices must be positive")

    edge_list = list(edges)
    adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(num_vertices)]

    for edge_id, (u, v) in enumerate(edge_list):
        if not (0 <= u < num_vertices and 0 <= v < num_vertices):
            raise ValueError(
                f"edge ({u}, {v}) is out of valid range [0, {num_vertices - 1}]"
            )
        if u == v:
            raise ValueError("self-loop is not supported in this MVP")

        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))

    return Graph(n=num_vertices, edges=edge_list, adjacency=adjacency)


def find_bridges_tarjan(graph: Graph) -> BridgeResult:
    """Find all bridges in O(V+E) using DFS timestamps + low-link values."""
    n = graph.n
    visited = np.zeros(n, dtype=bool)
    tin = np.full(n, -1, dtype=int)
    low = np.full(n, -1, dtype=int)

    timer = 0
    bridges: List[int] = []

    def dfs(v: int, parent_edge_id: int) -> None:
        nonlocal timer
        visited[v] = True
        tin[v] = timer
        low[v] = timer
        timer += 1

        for to, edge_id in graph.adjacency[v]:
            if edge_id == parent_edge_id:
                # Skip the exact undirected edge used to enter current node.
                continue

            if visited[to]:
                # Back edge: tighten low-link with ancestor entry time.
                low[v] = min(low[v], int(tin[to]))
                continue

            dfs(to, edge_id)
            low[v] = min(low[v], int(low[to]))

            # No back edge from subtree(to) to v or above => (v,to) is a bridge.
            if low[to] > tin[v]:
                bridges.append(edge_id)

    for start in range(n):
        if not visited[start]:
            dfs(start, parent_edge_id=-1)

    bridges.sort()
    return BridgeResult(bridge_edge_ids=bridges, tin=tin, low=low)


def count_connected_components(graph: Graph, banned_edge_id: int = -1) -> int:
    """Count connected components after optionally removing one edge by id."""
    visited = np.zeros(graph.n, dtype=bool)
    components = 0

    for s in range(graph.n):
        if visited[s]:
            continue

        components += 1
        stack = [s]
        visited[s] = True

        while stack:
            u = stack.pop()
            for v, edge_id in graph.adjacency[u]:
                if edge_id == banned_edge_id:
                    continue
                if visited[v]:
                    continue
                visited[v] = True
                stack.append(v)

    return components


def find_bridges_bruteforce(graph: Graph) -> List[int]:
    """Reference implementation: remove each edge and compare component count."""
    base_components = count_connected_components(graph)
    bridges: List[int] = []

    for edge_id in range(graph.m):
        new_components = count_connected_components(graph, banned_edge_id=edge_id)
        if new_components > base_components:
            bridges.append(edge_id)

    return bridges


def canonical_edge_repr(graph: Graph, edge_id: int) -> Tuple[int, int, int]:
    """Return (min_u, max_v, edge_id) for stable printing/sorting."""
    u, v = graph.edges[edge_id]
    a, b = (u, v) if u <= v else (v, u)
    return (a, b, edge_id)


def format_bridge_set(graph: Graph, bridge_edge_ids: Sequence[int]) -> str:
    if not bridge_edge_ids:
        return "[]"
    parts = []
    for edge_id in sorted(bridge_edge_ids):
        a, b, eid = canonical_edge_repr(graph, edge_id)
        parts.append(f"({a}, {b}, id={eid})")
    return "[" + ", ".join(parts) + "]"


def run_case(name: str, graph: Graph) -> None:
    print(f"\n=== Case: {name} ===")
    print(f"vertices={graph.n}, edges={graph.m}")

    tarjan = find_bridges_tarjan(graph)
    brute = find_bridges_bruteforce(graph)

    print(f"Tarjan bridges   : {format_bridge_set(graph, tarjan.bridge_edge_ids)}")
    print(f"Bruteforce check : {format_bridge_set(graph, brute)}")
    print(f"match            : {tarjan.bridge_edge_ids == brute}")
    print(f"tin              : {tarjan.tin.tolist()}")
    print(f"low              : {tarjan.low.tolist()}")


def main() -> None:
    # Case A: one cycle + one chain + one cycle => two bridges.
    graph_a = build_undirected_graph(
        num_vertices=7,
        edges=[
            (0, 1),
            (1, 2),
            (2, 0),
            (1, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 4),
        ],
    )

    # Case B: parallel edges between 0 and 1 mean those are not bridges.
    graph_b = build_undirected_graph(
        num_vertices=4,
        edges=[
            (0, 1),
            (0, 1),
            (1, 2),
            (2, 3),
        ],
    )

    # Case C: disconnected graph where each component is a cycle => no bridges.
    graph_c = build_undirected_graph(
        num_vertices=6,
        edges=[
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 5),
            (5, 3),
        ],
    )

    run_case("cycle + chain + cycle", graph_a)
    run_case("parallel edges", graph_b)
    run_case("two disconnected cycles", graph_c)


if __name__ == "__main__":
    main()
