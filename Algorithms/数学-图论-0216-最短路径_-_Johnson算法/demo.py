"""Johnson algorithm MVP for all-pairs shortest paths on sparse graphs.

Supports negative edge weights, but rejects negative cycles.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, Hashable, Iterable, List, Tuple

Node = Hashable
Edge = Tuple[Node, Node, float]


def bellman_ford_potential(nodes: List[Node], edges: List[Edge]) -> Dict[Node, float]:
    """Return potential h(v) via Bellman-Ford on graph with a super source.

    Raises:
        ValueError: if a negative cycle exists.
    """
    super_source = "__SUPER_SOURCE__"
    ext_nodes: List[Node] = nodes + [super_source]
    ext_edges: List[Edge] = list(edges) + [(super_source, v, 0.0) for v in nodes]

    dist: Dict[Node, float] = {v: math.inf for v in ext_nodes}
    dist[super_source] = 0.0

    for _ in range(len(ext_nodes) - 1):
        changed = False
        for u, v, w in ext_edges:
            if dist[u] != math.inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break

    for u, v, w in ext_edges:
        if dist[u] != math.inf and dist[u] + w < dist[v]:
            raise ValueError("Graph contains a negative-weight cycle.")

    return {v: dist[v] for v in nodes}


def dijkstra(
    nodes: Iterable[Node],
    adjacency: Dict[Node, List[Tuple[Node, float]]],
    source: Node,
) -> Dict[Node, float]:
    """Single-source shortest path on non-negative weighted graph."""
    dist: Dict[Node, float] = {v: math.inf for v in nodes}
    dist[source] = 0.0

    heap: List[Tuple[float, Node]] = [(0.0, source)]
    while heap:
        cur_dist, u = heapq.heappop(heap)
        if cur_dist > dist[u]:
            continue
        for v, w in adjacency.get(u, []):
            cand = cur_dist + w
            if cand < dist[v]:
                dist[v] = cand
                heapq.heappush(heap, (cand, v))
    return dist


def johnson_all_pairs_shortest_paths(
    nodes: List[Node],
    edges: List[Edge],
) -> Dict[Node, Dict[Node, float]]:
    """Compute all-pairs shortest paths with Johnson algorithm."""
    if not nodes:
        return {}

    h = bellman_ford_potential(nodes, edges)

    reweighted_adj: Dict[Node, List[Tuple[Node, float]]] = {v: [] for v in nodes}
    for u, v, w in edges:
        w_prime = w + h[u] - h[v]
        # Numerical safety for floating-point arithmetic.
        if w_prime < 0 and abs(w_prime) < 1e-12:
            w_prime = 0.0
        reweighted_adj[u].append((v, w_prime))

    all_pairs: Dict[Node, Dict[Node, float]] = {}
    for src in nodes:
        d_prime = dijkstra(nodes, reweighted_adj, src)
        restored: Dict[Node, float] = {}
        for dst in nodes:
            if d_prime[dst] == math.inf:
                restored[dst] = math.inf
            else:
                restored[dst] = d_prime[dst] - h[src] + h[dst]
        all_pairs[src] = restored

    return all_pairs


def pretty_print_matrix(nodes: List[Node], dist: Dict[Node, Dict[Node, float]]) -> None:
    """Print shortest path matrix in table form."""
    header = "src\\dst".ljust(8) + "".join(str(v).rjust(8) for v in nodes)
    print(header)
    for u in nodes:
        row = [str(u).ljust(8)]
        for v in nodes:
            d = dist[u][v]
            row.append(("inf" if d == math.inf else f"{d:.0f}").rjust(8))
        print("".join(row))


def main() -> None:
    # Classic CLRS-like sample graph: negative edges but no negative cycles.
    nodes = [0, 1, 2, 3, 4]
    edges: List[Edge] = [
        (0, 1, 3.0),
        (0, 2, 8.0),
        (0, 4, -4.0),
        (1, 3, 1.0),
        (1, 4, 7.0),
        (2, 1, 4.0),
        (3, 0, 2.0),
        (3, 2, -5.0),
        (4, 3, 6.0),
    ]

    dist = johnson_all_pairs_shortest_paths(nodes, edges)

    print("All-pairs shortest-path distances (Johnson):")
    pretty_print_matrix(nodes, dist)

    # Sanity checks against known shortest-path values for this graph.
    expected_0 = {0: 0.0, 1: 1.0, 2: -3.0, 3: 2.0, 4: -4.0}
    expected_3 = {0: 2.0, 1: -1.0, 2: -5.0, 3: 0.0, 4: -2.0}
    assert dist[0] == expected_0, f"Unexpected row for source 0: {dist[0]}"
    assert dist[3] == expected_3, f"Unexpected row for source 3: {dist[3]}"

    print("\nAssertions passed.")


if __name__ == "__main__":
    main()
