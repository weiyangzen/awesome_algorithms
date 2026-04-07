"""Bron-Kerbosch (pivot) MVP for maximum clique.

Run:
    python3 demo.py
"""

from itertools import combinations
from typing import Dict, Iterable, List, Set, Tuple


Graph = Dict[int, Set[int]]


def build_graph(num_vertices: int, edges: Iterable[Tuple[int, int]]) -> Graph:
    """Build an undirected simple graph as adjacency sets."""
    graph: Graph = {v: set() for v in range(num_vertices)}
    for u, v in edges:
        if not (0 <= u < num_vertices and 0 <= v < num_vertices):
            raise ValueError(f"edge ({u}, {v}) contains invalid vertex id")
        if u == v:
            continue
        graph[u].add(v)
        graph[v].add(u)
    return graph


def is_clique(graph: Graph, vertices: Set[int]) -> bool:
    """Check whether every pair of vertices is adjacent."""
    ordered = sorted(vertices)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            u, v = ordered[i], ordered[j]
            if v not in graph[u]:
                return False
    return True


def bron_kerbosch_maximum_clique(graph: Graph) -> Set[int]:
    """Return one maximum clique using Bron-Kerbosch with pivot and pruning."""
    best_clique: Set[int] = set()

    def search(r: Set[int], p: Set[int], x: Set[int]) -> None:
        nonlocal best_clique

        # Safe upper bound: even taking all p cannot beat current best.
        if len(r) + len(p) <= len(best_clique):
            return

        if not p and not x:
            if len(r) > len(best_clique):
                best_clique = set(r)
            return

        pivot_candidates = p | x
        if pivot_candidates:
            pivot = max(pivot_candidates, key=lambda u: len(p & graph[u]))
            candidates = sorted(p - graph[pivot])
        else:
            candidates = sorted(p)

        for v in candidates:
            if v not in p:
                continue
            search(r | {v}, p & graph[v], x & graph[v])
            p.remove(v)
            x.add(v)

    vertices = set(graph.keys())
    search(set(), set(vertices), set())
    return best_clique


def bruteforce_maximum_clique(graph: Graph) -> Set[int]:
    """Reference solver for small graphs, used only for verification."""
    vertices: List[int] = sorted(graph.keys())
    best: Set[int] = set()

    for size in range(1, len(vertices) + 1):
        for comb in combinations(vertices, size):
            cand = set(comb)
            if len(cand) > len(best) and is_clique(graph, cand):
                best = cand
    return best


def main() -> None:
    # A fixed demo graph with maximum clique size 4.
    # Cliques of size 4 include {0,1,2,3} and {2,3,4,5}.
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
        (4, 5),
        (1, 6),
        (6, 7),
    ]

    graph = build_graph(num_vertices=8, edges=edges)

    bk_clique = bron_kerbosch_maximum_clique(graph)
    brute_clique = bruteforce_maximum_clique(graph)

    print("=== Bron-Kerbosch Maximum Clique Demo ===")
    print(f"vertex_count: {len(graph)}")
    print(f"edge_count: {sum(len(nbrs) for nbrs in graph.values()) // 2}")
    print(f"bron_kerbosch_max_clique: {sorted(bk_clique)} (size={len(bk_clique)})")
    print(f"bruteforce_max_clique:   {sorted(brute_clique)} (size={len(brute_clique)})")

    if not is_clique(graph, bk_clique):
        raise RuntimeError("Bron-Kerbosch result is not a clique")

    if len(bk_clique) != len(brute_clique):
        raise RuntimeError("Size mismatch between Bron-Kerbosch and brute-force result")

    print("verification: PASS")


if __name__ == "__main__":
    main()
