"""Minimal runnable MVP for Bellman-Ford algorithm."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

INF = float("inf")
NEG_INF = float("-inf")
Edge = Tuple[int, int, float]


def bellman_ford(
    n: int, edges: List[Edge], source: int
) -> Tuple[List[float], List[Optional[int]], bool, List[int]]:
    """Single-source shortest paths with negative-cycle influence marking.

    Args:
        n: Number of vertices indexed in [0, n-1].
        edges: Directed weighted edges (u, v, w).
        source: Source vertex index.

    Returns:
        dist: Distance array (INF for unreachable, NEG_INF for neg-cycle affected).
        parent: Predecessor array for path reconstruction.
        has_negative_cycle: Whether a source-reachable negative cycle exists.
        affected_by_neg_cycle: Vertices whose shortest distance is undefined.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if source < 0 or source >= n:
        raise ValueError("source index out of range")

    adjacency: List[List[int]] = [[] for _ in range(n)]
    for u, v, _ in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("edge endpoint out of range")
        adjacency[u].append(v)

    dist = [INF] * n
    parent: List[Optional[int]] = [None] * n
    dist[source] = 0.0

    # Core relaxation passes.
    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] == INF:
                continue
            candidate = dist[u] + w
            if candidate < dist[v]:
                dist[v] = candidate
                parent[v] = u
                updated = True
        if not updated:
            break

    # Detect vertices that can still be relaxed => reachable from a negative cycle.
    seeds: List[int] = []
    in_seed = [False] * n
    for u, v, w in edges:
        if dist[u] == INF:
            continue
        if dist[u] + w < dist[v] and not in_seed[v]:
            seeds.append(v)
            in_seed[v] = True

    has_negative_cycle = len(seeds) > 0
    if not has_negative_cycle:
        return dist, parent, False, []

    # Propagate negative-cycle influence along outgoing edges.
    affected = [False] * n
    q: Deque[int] = deque(seeds)
    for s in seeds:
        affected[s] = True

    while q:
        x = q.popleft()
        for y in adjacency[x]:
            if not affected[y]:
                affected[y] = True
                q.append(y)

    affected_nodes = [i for i, flag in enumerate(affected) if flag]
    for v in affected_nodes:
        dist[v] = NEG_INF
        parent[v] = None

    return dist, parent, True, affected_nodes


def reconstruct_path(
    parent: List[Optional[int]], dist: List[float], source: int, target: int
) -> List[int]:
    """Reconstruct one shortest path if distance is finite."""
    n = len(parent)
    if not (0 <= source < n and 0 <= target < n):
        raise ValueError("source/target index out of range")
    if dist[target] in (INF, NEG_INF):
        return []

    path: List[int] = []
    cur: Optional[int] = target
    max_steps = n + 1
    for _ in range(max_steps):
        if cur is None:
            return []
        path.append(cur)
        if cur == source:
            path.reverse()
            return path
        cur = parent[cur]

    # Guard against malformed predecessor chains.
    return []


def format_dist(x: float) -> str:
    if x == INF:
        return "inf"
    if x == NEG_INF:
        return "-inf"
    return f"{x:.0f}"


def run_case(
    name: str, n: int, edges: List[Edge], source: int, query_targets: List[int]
) -> None:
    print(f"=== {name} ===")
    dist, parent, has_neg_cycle, affected = bellman_ford(n, edges, source)
    print(f"source: {source}")
    print(f"dist: {[format_dist(x) for x in dist]}")
    print(f"has_negative_cycle: {has_neg_cycle}")
    print(f"affected_by_neg_cycle: {affected}")

    for target in query_targets:
        path = reconstruct_path(parent, dist, source, target)
        if not path:
            status = "unreachable or undefined (-inf)"
            print(f"path {source}->{target}: {status}")
        else:
            print(
                f"path {source}->{target}: {path}, cost={format_dist(dist[target])}"
            )
    print()


def main() -> None:
    # Case A: classic CLRS graph without reachable negative cycle.
    n1 = 5
    edges1: List[Edge] = [
        (0, 1, 6),
        (0, 2, 7),
        (1, 2, 8),
        (1, 3, 5),
        (1, 4, -4),
        (2, 3, -3),
        (2, 4, 9),
        (3, 1, -2),
        (4, 0, 2),
        (4, 3, 7),
    ]
    run_case(
        "Case A: no negative cycle",
        n=n1,
        edges=edges1,
        source=0,
        query_targets=[1, 3, 4],
    )

    # Case B: source-reachable negative cycle 1->2->3->1.
    n2 = 5
    edges2: List[Edge] = [
        (0, 1, 1),
        (1, 2, -1),
        (2, 3, -1),
        (3, 1, -1),
        (0, 4, 5),
        (3, 4, 2),
    ]
    run_case(
        "Case B: reachable negative cycle",
        n=n2,
        edges=edges2,
        source=0,
        query_targets=[1, 3, 4],
    )


if __name__ == "__main__":
    main()
