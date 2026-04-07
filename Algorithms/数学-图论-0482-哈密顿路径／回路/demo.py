"""Hamiltonian path/cycle MVP using bitmask dynamic programming.

This script is fully self-contained and runs without interactive input.
It demonstrates exact existence + one witness construction for both:
- Hamiltonian Path
- Hamiltonian Cycle
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Sequence, Tuple

import numpy as np

UNREACHABLE = np.int16(-2)
START_SENTINEL = np.int16(-1)
MAX_EXACT_VERTICES = 20


@dataclass(frozen=True)
class Graph:
    """Simple adjacency-matrix graph container."""

    adjacency: np.ndarray
    directed: bool = False

    @property
    def n(self) -> int:
        return int(self.adjacency.shape[0])

    def has_edge(self, u: int, v: int) -> bool:
        return bool(self.adjacency[u, v])


def build_graph_from_edges(
    num_vertices: int,
    edges: Sequence[Tuple[int, int]],
    directed: bool = False,
) -> Graph:
    """Build a graph from edge list.

    Vertices are expected to be indexed from 0 to num_vertices-1.
    """
    if num_vertices <= 0:
        raise ValueError("num_vertices must be positive.")

    adjacency = np.zeros((num_vertices, num_vertices), dtype=np.uint8)

    for u, v in edges:
        if not (0 <= u < num_vertices and 0 <= v < num_vertices):
            raise ValueError(f"Edge ({u}, {v}) uses vertex outside [0, {num_vertices - 1}].")
        if u == v:
            raise ValueError("Self-loops are not supported in this MVP.")
        adjacency[u, v] = 1
        if not directed:
            adjacency[v, u] = 1

    return Graph(adjacency=adjacency, directed=directed)


def _neighbor_bitmasks(graph: Graph) -> List[int]:
    bitmasks: List[int] = []
    for u in range(graph.n):
        bits = 0
        for v in range(graph.n):
            if graph.has_edge(u, v):
                bits |= 1 << v
        bitmasks.append(bits)
    return bitmasks


def _reconstruct_path(parent: np.ndarray, full_mask: int, end_vertex: int) -> List[int]:
    """Reconstruct one path ending at end_vertex using parent table."""
    path: List[int] = []
    mask = full_mask
    current = int(end_vertex)

    while True:
        path.append(current)
        prev = int(parent[mask, current])
        if prev == int(START_SENTINEL):
            break
        mask ^= 1 << current
        current = prev

    path.reverse()
    return path


def find_hamiltonian_path(graph: Graph) -> Optional[List[int]]:
    """Find one Hamiltonian path if it exists, else return None.

    Exact DP complexity is O(n^2 * 2^n), so we guard n <= 20.
    """
    n = graph.n
    if n > MAX_EXACT_VERTICES:
        raise ValueError(
            f"Exact bitmask DP is capped at {MAX_EXACT_VERTICES} vertices, got {n}."
        )

    state_count = 1 << n
    full_mask = state_count - 1
    parent = np.full((state_count, n), fill_value=UNREACHABLE, dtype=np.int16)
    neighbor_bits = _neighbor_bitmasks(graph)

    # Any vertex can be the start of a Hamiltonian path.
    for v in range(n):
        parent[1 << v, v] = START_SENTINEL

    for mask in range(state_count):
        for end in range(n):
            if parent[mask, end] == UNREACHABLE:
                continue

            candidates = neighbor_bits[end] & (~mask) & full_mask
            while candidates:
                lsb = candidates & -candidates
                nxt = lsb.bit_length() - 1
                next_mask = mask | lsb
                if parent[next_mask, nxt] == UNREACHABLE:
                    parent[next_mask, nxt] = np.int16(end)
                candidates ^= lsb

    for end in range(n):
        if parent[full_mask, end] != UNREACHABLE:
            return _reconstruct_path(parent=parent, full_mask=full_mask, end_vertex=end)

    return None


def _find_hamiltonian_cycle_from_start(graph: Graph, start: int) -> Optional[List[int]]:
    n = graph.n
    state_count = 1 << n
    full_mask = state_count - 1

    parent = np.full((state_count, n), fill_value=UNREACHABLE, dtype=np.int16)
    neighbor_bits = _neighbor_bitmasks(graph)

    start_mask = 1 << start
    parent[start_mask, start] = START_SENTINEL

    for mask in range(state_count):
        if (mask & start_mask) == 0:
            continue
        for end in range(n):
            if parent[mask, end] == UNREACHABLE:
                continue

            candidates = neighbor_bits[end] & (~mask) & full_mask
            while candidates:
                lsb = candidates & -candidates
                nxt = lsb.bit_length() - 1
                next_mask = mask | lsb
                if parent[next_mask, nxt] == UNREACHABLE:
                    parent[next_mask, nxt] = np.int16(end)
                candidates ^= lsb

    for end in range(n):
        if end == start:
            continue
        if parent[full_mask, end] == UNREACHABLE:
            continue
        if not graph.has_edge(end, start):
            continue

        path = _reconstruct_path(parent=parent, full_mask=full_mask, end_vertex=end)
        path.append(start)
        return path

    return None


def find_hamiltonian_cycle(graph: Graph) -> Optional[List[int]]:
    """Find one Hamiltonian cycle if it exists, else return None."""
    n = graph.n
    if n > MAX_EXACT_VERTICES:
        raise ValueError(
            f"Exact bitmask DP is capped at {MAX_EXACT_VERTICES} vertices, got {n}."
        )

    # Try each start to avoid missing a cycle due to fixed-root asymmetry.
    for start in range(n):
        cycle = _find_hamiltonian_cycle_from_start(graph, start)
        if cycle is not None:
            return cycle
    return None


def is_valid_hamiltonian_path(graph: Graph, path: Sequence[int]) -> bool:
    n = graph.n
    if len(path) != n:
        return False
    if sorted(path) != list(range(n)):
        return False
    for i in range(n - 1):
        if not graph.has_edge(path[i], path[i + 1]):
            return False
    return True


def is_valid_hamiltonian_cycle(graph: Graph, cycle: Sequence[int]) -> bool:
    n = graph.n
    if len(cycle) != n + 1:
        return False
    if cycle[0] != cycle[-1]:
        return False
    core = cycle[:-1]
    if sorted(core) != list(range(n)):
        return False
    for i in range(n):
        if not graph.has_edge(cycle[i], cycle[i + 1]):
            return False
    return True


def format_walk(nodes: Sequence[int]) -> str:
    return " -> ".join(str(x) for x in nodes)


def run_case(name: str, graph: Graph) -> None:
    print(f"\n=== Case: {name} ===")
    print(f"directed={graph.directed}, vertices={graph.n}")

    t0 = perf_counter()
    path = find_hamiltonian_path(graph)
    t1 = perf_counter()
    cycle = find_hamiltonian_cycle(graph)
    t2 = perf_counter()

    if path is None:
        print("Hamiltonian path: NOT FOUND")
    else:
        print(f"Hamiltonian path: {format_walk(path)}")
        print(f"path valid: {is_valid_hamiltonian_path(graph, path)}")

    if cycle is None:
        print("Hamiltonian cycle: NOT FOUND")
    else:
        print(f"Hamiltonian cycle: {format_walk(cycle)}")
        print(f"cycle valid: {is_valid_hamiltonian_cycle(graph, cycle)}")

    print(f"path search time (ms): {(t1 - t0) * 1000.0:.3f}")
    print(f"cycle search time (ms): {(t2 - t1) * 1000.0:.3f}")


def main() -> None:
    # Case 1: has both Hamiltonian path and cycle.
    graph_cycle = build_graph_from_edges(
        num_vertices=6,
        edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 2),
            (2, 4),
        ],
        directed=False,
    )

    # Case 2: chain graph, has path but no cycle.
    graph_path_only = build_graph_from_edges(
        num_vertices=6,
        edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        ],
        directed=False,
    )

    # Case 3: disconnected graph, neither path nor cycle exists.
    graph_none = build_graph_from_edges(
        num_vertices=6,
        edges=[
            (0, 1),
            (1, 2),
            (3, 4),
            (4, 5),
        ],
        directed=False,
    )

    # Case 4: directed graph with a Hamiltonian cycle.
    directed_cycle = build_graph_from_edges(
        num_vertices=5,
        edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),
            (0, 2),
            (2, 4),
        ],
        directed=True,
    )

    run_case("Undirected: cycle exists", graph_cycle)
    run_case("Undirected: path only", graph_path_only)
    run_case("Undirected: none", graph_none)
    run_case("Directed: cycle exists", directed_cycle)


if __name__ == "__main__":
    main()
