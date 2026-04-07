"""Boruvka minimum spanning tree (MST) demo.

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

Edge = Tuple[int, int, float]


@dataclass
class BoruvkaResult:
    mst_edges: List[Edge]
    total_weight: float
    rounds: int
    is_spanning_tree: bool
    components: int


class DisjointSetUnion:
    """Union-Find with path compression + union by rank."""

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


def boruvka_mst(num_vertices: int, edges: List[Edge]) -> BoruvkaResult:
    """Compute MST using Boruvka algorithm.

    If the graph is disconnected, this returns a minimum spanning forest and
    marks is_spanning_tree=False.
    """
    if num_vertices < 0:
        raise ValueError("num_vertices must be non-negative")
    if num_vertices == 0:
        return BoruvkaResult([], 0.0, 0, True, 0)

    dsu = DisjointSetUnion(num_vertices)
    components = num_vertices
    mst_edges: List[Edge] = []
    total_weight = 0.0
    rounds = 0

    while components > 1:
        rounds += 1
        cheapest: List[Edge | None] = [None] * num_vertices

        for u, v, w in edges:
            if not (0 <= u < num_vertices and 0 <= v < num_vertices):
                raise ValueError(f"edge ({u}, {v}, {w}) has vertex out of range")

            ru = dsu.find(u)
            rv = dsu.find(v)
            if ru == rv:
                continue

            if cheapest[ru] is None or w < cheapest[ru][2]:
                cheapest[ru] = (u, v, w)
            if cheapest[rv] is None or w < cheapest[rv][2]:
                cheapest[rv] = (u, v, w)

        merged_in_round = 0
        for edge in cheapest:
            if edge is None:
                continue

            u, v, w = edge
            if dsu.union(u, v):
                mst_edges.append(edge)
                total_weight += w
                components -= 1
                merged_in_round += 1

        if merged_in_round == 0:
            break

    return BoruvkaResult(
        mst_edges=mst_edges,
        total_weight=total_weight,
        rounds=rounds,
        is_spanning_tree=(components == 1),
        components=components,
    )


def pretty_print_result(title: str, result: BoruvkaResult) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for idx, (u, v, w) in enumerate(result.mst_edges, start=1):
        print(f"{idx:2d}. ({u}, {v}) weight={w}")
    print(f"total_weight = {result.total_weight}")
    print(f"rounds       = {result.rounds}")
    print(f"components   = {result.components}")
    print(f"spanning     = {result.is_spanning_tree}")


def main() -> None:
    connected_graph_edges: List[Edge] = [
        (0, 1, 7),
        (0, 3, 5),
        (1, 2, 8),
        (1, 3, 9),
        (1, 4, 7),
        (2, 4, 5),
        (3, 4, 15),
        (3, 5, 6),
        (4, 5, 8),
        (4, 6, 9),
        (5, 6, 11),
    ]

    disconnected_graph_edges: List[Edge] = [
        (0, 1, 2),
        (1, 2, 1),
        (3, 4, 3),
    ]

    result_connected = boruvka_mst(num_vertices=7, edges=connected_graph_edges)
    result_disconnected = boruvka_mst(num_vertices=5, edges=disconnected_graph_edges)

    pretty_print_result("Connected graph (MST expected)", result_connected)
    pretty_print_result("Disconnected graph (forest expected)", result_disconnected)


if __name__ == "__main__":
    main()
