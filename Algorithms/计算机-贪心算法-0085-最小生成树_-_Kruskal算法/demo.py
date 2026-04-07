"""Kruskal minimum spanning tree (MST) minimal runnable MVP."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Edge:
    """Undirected weighted edge."""

    u: int
    v: int
    weight: float


class UnionFind:
    """Disjoint Set Union with path compression and union by rank."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

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


def kruskal_mst(num_nodes: int, edges: Iterable[Edge]) -> tuple[float, list[Edge], bool]:
    """Return (total_weight, mst_edges, connected).

    If graph is disconnected, returns a minimum spanning forest with connected=False.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if num_nodes <= 1:
        return 0.0, [], True

    clean_edges: list[Edge] = []
    for edge in edges:
        if not (0 <= edge.u < num_nodes and 0 <= edge.v < num_nodes):
            raise ValueError(f"edge endpoint out of range: {edge}")
        if edge.u == edge.v:
            # Self-loop never helps MST.
            continue
        u, v = sorted((edge.u, edge.v))
        clean_edges.append(Edge(u=u, v=v, weight=edge.weight))

    sorted_edges = sorted(clean_edges, key=lambda e: (e.weight, e.u, e.v))
    uf = UnionFind(num_nodes)

    chosen: list[Edge] = []
    total_weight = 0.0

    for edge in sorted_edges:
        if uf.union(edge.u, edge.v):
            chosen.append(edge)
            total_weight += edge.weight
            if len(chosen) == num_nodes - 1:
                break

    connected = len(chosen) == num_nodes - 1
    return total_weight, chosen, connected


def _is_spanning_tree(num_nodes: int, picked: Sequence[Edge]) -> bool:
    if len(picked) != num_nodes - 1:
        return False
    uf = UnionFind(num_nodes)
    for edge in picked:
        if edge.u == edge.v:
            return False
        if not uf.union(edge.u, edge.v):
            return False
    root = uf.find(0)
    return all(uf.find(i) == root for i in range(num_nodes))


def bruteforce_mst_weight(num_nodes: int, edges: Sequence[Edge]) -> float | None:
    """Exact MST weight for small graphs by enumeration.

    Returns None if the graph is disconnected.
    """
    if num_nodes <= 0:
        return 0.0

    best = None
    for comb in combinations(edges, num_nodes - 1):
        if not _is_spanning_tree(num_nodes, comb):
            continue
        weight = sum(edge.weight for edge in comb)
        if best is None or weight < best:
            best = weight
    return best


def make_random_connected_graph(
    num_nodes: int,
    num_edges: int,
    seed: int,
    min_weight: int = 1,
    max_weight: int = 20,
) -> list[Edge]:
    """Create a connected random undirected graph."""
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2")
    max_edges = num_nodes * (num_nodes - 1) // 2
    if not (num_nodes - 1 <= num_edges <= max_edges):
        raise ValueError("num_edges out of valid range")

    rng = random.Random(seed)
    edge_weights: dict[tuple[int, int], int] = {}

    # Step 1: build a random spanning tree to guarantee connectivity.
    for node in range(1, num_nodes):
        parent = rng.randrange(0, node)
        a, b = sorted((node, parent))
        edge_weights[(a, b)] = rng.randint(min_weight, max_weight)

    # Step 2: add extra edges until reaching num_edges.
    while len(edge_weights) < num_edges:
        u = rng.randrange(0, num_nodes)
        v = rng.randrange(0, num_nodes)
        if u == v:
            continue
        a, b = sorted((u, v))
        if (a, b) in edge_weights:
            continue
        edge_weights[(a, b)] = rng.randint(min_weight, max_weight)

    return [Edge(u=a, v=b, weight=w) for (a, b), w in edge_weights.items()]


def format_edges(edges: Sequence[Edge]) -> str:
    return ", ".join(f"{e.u}-{e.v}:{e.weight:g}" for e in edges)


def main() -> None:
    print("=== Kruskal MST MVP Demo ===")

    print("\n[1] Deterministic sample")
    sample_edges = [
        Edge(0, 1, 4),
        Edge(0, 2, 4),
        Edge(1, 2, 2),
        Edge(1, 3, 5),
        Edge(2, 3, 5),
        Edge(2, 4, 11),
        Edge(3, 4, 2),
        Edge(3, 5, 6),
        Edge(4, 5, 1),
    ]
    expected_weight = 14.0
    total_weight, mst_edges, connected = kruskal_mst(6, sample_edges)
    brute_weight = bruteforce_mst_weight(6, sample_edges)

    print(f"input edges : {format_edges(sample_edges)}")
    print(f"mst edges   : {format_edges(mst_edges)}")
    print(f"connected   : {connected}")
    print(f"mst weight  : {total_weight:g}")
    print(f"bruteforce  : {brute_weight}")

    assert connected, "deterministic sample should be connected"
    assert total_weight == expected_weight, "deterministic sample MST weight mismatch"
    assert brute_weight == total_weight, "Kruskal should match brute-force optimum"

    print("\n[2] Random small-graph cross-check")
    for case_id, seed in enumerate((2026, 2027, 2028), start=1):
        edges = make_random_connected_graph(num_nodes=6, num_edges=9, seed=seed)
        k_weight, _, k_connected = kruskal_mst(6, edges)
        b_weight = bruteforce_mst_weight(6, edges)
        ok = k_connected and (k_weight == b_weight)
        print(
            f"case#{case_id}: edges={len(edges)}, "
            f"kruskal={k_weight:g}, brute={b_weight}, pass={ok}"
        )
        assert ok, f"random case#{case_id} failed"

    print("\n[3] Disconnected graph behavior")
    disconnected_edges = [
        Edge(0, 1, 1),
        Edge(1, 2, 3),
        Edge(3, 4, 2),
    ]
    forest_weight, forest_edges, is_connected = kruskal_mst(5, disconnected_edges)
    print(f"input edges : {format_edges(disconnected_edges)}")
    print(f"forest edges: {format_edges(forest_edges)}")
    print(f"connected   : {is_connected}")
    print(f"forest weight: {forest_weight:g}")

    assert not is_connected, "disconnected graph should not return connected=True"
    assert len(forest_edges) == 3, "forest should contain 3 edges for this input"

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
