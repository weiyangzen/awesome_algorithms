"""Prim algorithm MVP for minimum spanning tree on undirected weighted graphs.

The script is self-contained and runs without interactive input.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Tuple

Edge = Tuple[int, int, float]


@dataclass
class UnionFind:
    """Disjoint-set union structure used for Kruskal cross-check."""

    parent: List[int]
    rank: List[int]

    @classmethod
    def create(cls, n: int) -> "UnionFind":
        return cls(parent=list(range(n)), rank=[0] * n)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def build_adjacency(num_nodes: int, edges: List[Edge]) -> List[List[Tuple[int, float]]]:
    """Build adjacency list for an undirected weighted graph."""
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for u, v, w in edges:
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"Edge ({u}, {v}, {w}) uses node id out of range [0, {num_nodes}).")
        adjacency[u].append((v, w))
        adjacency[v].append((u, w))
    return adjacency


def prim_mst(num_nodes: int, edges: List[Edge], start: int = 0) -> Tuple[List[Edge], float]:
    """Return MST edges and total weight using Prim's algorithm with a min-heap.

    Raises:
        ValueError: if graph is disconnected or start node is invalid.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if num_nodes == 0:
        return [], 0.0
    if not (0 <= start < num_nodes):
        raise ValueError(f"start node must be in [0, {num_nodes})")

    adjacency = build_adjacency(num_nodes, edges)

    visited = [False] * num_nodes
    visited_count = 0

    mst_edges: List[Edge] = []
    total_weight = 0.0

    # Heap entries are (weight, to_node, from_node).
    heap: List[Tuple[float, int, int]] = [(0.0, start, -1)]

    while heap and visited_count < num_nodes:
        weight, node, parent = heapq.heappop(heap)
        if visited[node]:
            continue

        visited[node] = True
        visited_count += 1

        if parent != -1:
            mst_edges.append((parent, node, weight))
            total_weight += weight

        for nxt, nxt_w in adjacency[node]:
            if not visited[nxt]:
                heapq.heappush(heap, (nxt_w, nxt, node))

    if visited_count != num_nodes:
        raise ValueError("Graph is disconnected; MST does not exist for all nodes.")

    return mst_edges, total_weight


def kruskal_mst(num_nodes: int, edges: List[Edge]) -> Tuple[List[Edge], float]:
    """Reference implementation used to cross-check Prim's result."""
    if num_nodes == 0:
        return [], 0.0

    uf = UnionFind.create(num_nodes)
    mst: List[Edge] = []
    total = 0.0

    for u, v, w in sorted(edges, key=lambda x: x[2]):
        if uf.union(u, v):
            mst.append((u, v, w))
            total += w
            if len(mst) == num_nodes - 1:
                break

    if len(mst) != num_nodes - 1:
        raise ValueError("Graph is disconnected; MST does not exist for all nodes.")

    return mst, total


def normalize_edges(edge_list: List[Edge]) -> List[Edge]:
    """Normalize undirected edge direction for stable display."""
    normalized = []
    for u, v, w in edge_list:
        a, b = (u, v) if u <= v else (v, u)
        normalized.append((a, b, w))
    return sorted(normalized, key=lambda x: (x[2], x[0], x[1]))


def main() -> None:
    # Connected weighted undirected graph with known MST total weight 11.
    num_nodes = 6
    edges: List[Edge] = [
        (0, 1, 4.0),
        (0, 2, 3.0),
        (1, 2, 1.0),
        (1, 3, 2.0),
        (2, 3, 4.0),
        (1, 4, 3.0),
        (3, 4, 2.0),
        (3, 5, 3.0),
        (4, 5, 6.0),
        (2, 5, 8.0),
    ]

    prim_edges, prim_total = prim_mst(num_nodes, edges, start=0)
    kruskal_edges, kruskal_total = kruskal_mst(num_nodes, edges)

    print("Prim MST edges (u, v, w):")
    for item in normalize_edges(prim_edges):
        print(item)
    print(f"Prim total weight: {prim_total:.1f}")

    # Deterministic correctness checks.
    assert len(prim_edges) == num_nodes - 1, "MST must contain n-1 edges."
    assert abs(prim_total - 11.0) < 1e-12, f"Expected total weight 11.0, got {prim_total}"
    assert abs(prim_total - kruskal_total) < 1e-12, (
        f"Prim ({prim_total}) and Kruskal ({kruskal_total}) totals should match"
    )

    print("Kruskal total weight (cross-check):", f"{kruskal_total:.1f}")

    # Disconnected-graph branch check (should raise).
    try:
        prim_mst(4, [(0, 1, 1.0), (2, 3, 1.0)], start=0)
        raise AssertionError("Disconnected graph should raise ValueError")
    except ValueError as exc:
        print("Disconnected graph check:", str(exc))

    # Keep variable used to avoid linter complaints in stricter environments.
    _ = kruskal_edges

    print("All assertions passed.")


if __name__ == "__main__":
    main()
