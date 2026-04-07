"""Dijkstra shortest path minimal runnable MVP for CS-0066.

Run:
    uv run python demo.py
"""

from __future__ import annotations

import heapq
import math
import random
from typing import Iterable, List, Sequence, Tuple

Edge = Tuple[int, int, float]


def build_adjacency(num_nodes: int, edges: Iterable[Edge]) -> List[List[Tuple[int, float]]]:
    """Build adjacency list for a directed graph with non-negative edge weights."""
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")

    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for u, v, w in edges:
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"Edge ({u}, {v}, {w}) has endpoint out of range [0, {num_nodes}).")
        if w < 0:
            raise ValueError(f"Dijkstra requires non-negative weights, got edge ({u}, {v}, {w}).")
        adjacency[u].append((v, float(w)))
    return adjacency


def dijkstra_shortest_paths(
    num_nodes: int, edges: Iterable[Edge], source: int = 0
) -> Tuple[List[float], List[int]]:
    """Return shortest distances and predecessor array from source using Dijkstra."""
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if num_nodes == 0:
        return [], []
    if not (0 <= source < num_nodes):
        raise ValueError(f"source must be in [0, {num_nodes}).")

    adjacency = build_adjacency(num_nodes, edges)

    dist = [math.inf] * num_nodes
    prev = [-1] * num_nodes
    settled = [False] * num_nodes

    dist[source] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, source)]

    while heap:
        du, u = heapq.heappop(heap)
        if settled[u]:
            continue
        settled[u] = True

        for v, w in adjacency[u]:
            alt = du + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))

    return dist, prev


def reconstruct_path(prev: Sequence[int], source: int, target: int) -> List[int]:
    """Reconstruct one shortest path from source to target using predecessor links."""
    if not (0 <= source < len(prev) and 0 <= target < len(prev)):
        raise ValueError("source/target out of range for predecessor array")

    if source == target:
        return [source]

    path: List[int] = []
    cur = target
    while cur != -1:
        path.append(cur)
        if cur == source:
            break
        cur = prev[cur]

    if not path or path[-1] != source:
        return []

    path.reverse()
    return path


def bellman_ford_reference(num_nodes: int, edges: Iterable[Edge], source: int) -> List[float]:
    """Reference shortest-path implementation for validation on non-negative graphs."""
    dist = [math.inf] * num_nodes
    dist[source] = 0.0

    edge_list = list(edges)
    for _ in range(num_nodes - 1):
        changed = False
        for u, v, w in edge_list:
            if dist[u] is math.inf:
                continue
            cand = dist[u] + w
            if cand < dist[v]:
                dist[v] = cand
                changed = True
        if not changed:
            break
    return dist


def make_random_graph(num_nodes: int, num_edges: int, seed: int) -> List[Edge]:
    """Create a directed random graph with non-negative integer weights."""
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2")
    if num_edges < 1:
        raise ValueError("num_edges must be >= 1")

    rng = random.Random(seed)
    seen: set[Tuple[int, int]] = set()
    edges: List[Edge] = []

    while len(edges) < num_edges:
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u == v or (u, v) in seen:
            continue
        seen.add((u, v))
        w = float(rng.randint(0, 12))
        edges.append((u, v, w))

    return edges


def undirected_to_directed(edges: Sequence[Edge]) -> List[Edge]:
    """Expand each undirected edge into two directed edges."""
    directed: List[Edge] = []
    for u, v, w in edges:
        directed.append((u, v, float(w)))
        directed.append((v, u, float(w)))
    return directed


def assert_dist_equal(a: Sequence[float], b: Sequence[float], eps: float = 1e-12) -> None:
    """Assert equality for distance arrays with inf-safe comparison."""
    assert len(a) == len(b), "Distance arrays must have same length"
    for i, (x, y) in enumerate(zip(a, b)):
        if math.isinf(x) and math.isinf(y):
            continue
        assert abs(x - y) <= eps, f"Distance mismatch at node {i}: {x} vs {y}"


def main() -> None:
    print("=== Dijkstra Shortest Path MVP ===")

    # Deterministic sample (classic 6-node weighted graph).
    num_nodes = 6
    undirected_edges: List[Edge] = [
        (0, 1, 7.0),
        (0, 2, 9.0),
        (0, 5, 14.0),
        (1, 2, 10.0),
        (1, 3, 15.0),
        (2, 3, 11.0),
        (2, 5, 2.0),
        (3, 4, 6.0),
        (4, 5, 9.0),
    ]
    edges = undirected_to_directed(undirected_edges)

    dist, prev = dijkstra_shortest_paths(num_nodes, edges, source=0)
    expected = [0.0, 7.0, 9.0, 20.0, 20.0, 11.0]
    assert_dist_equal(dist, expected)

    path_to_4 = reconstruct_path(prev, source=0, target=4)
    assert path_to_4 == [0, 2, 5, 4], f"Unexpected path to 4: {path_to_4}"

    print("[deterministic] distances:", dist)
    print("[deterministic] path 0->4:", path_to_4)

    # Random-graph cross-check against Bellman-Ford.
    print("\n[random cross-check] Dijkstra vs Bellman-Ford")
    for case_id, seed in enumerate((2026, 2027, 2028), start=1):
        rg_edges = make_random_graph(num_nodes=7, num_edges=16, seed=seed)
        d1, _ = dijkstra_shortest_paths(7, rg_edges, source=0)
        d2 = bellman_ford_reference(7, rg_edges, source=0)
        assert_dist_equal(d1, d2)
        print(f"case#{case_id}: pass, reachable={sum(0 if math.isinf(x) else 1 for x in d1)}")

    # Unreachable-node branch.
    dist_unreach, prev_unreach = dijkstra_shortest_paths(
        5,
        [(0, 1, 1.0), (1, 2, 2.0)],
        source=0,
    )
    assert math.isinf(dist_unreach[4]), "node 4 should be unreachable"
    assert reconstruct_path(prev_unreach, 0, 4) == [], "unreachable path should be empty"
    print("\n[unreachable] distance to node 4 is inf and path is []")

    # Negative-weight rejection branch.
    try:
        dijkstra_shortest_paths(3, [(0, 1, -1.0)], source=0)
        raise AssertionError("Negative edge should raise ValueError")
    except ValueError as exc:
        print("[negative-edge check]", str(exc))

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
