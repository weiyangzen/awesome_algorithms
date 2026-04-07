"""Minimal runnable MVP for Dijkstra shortest-path algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

Edge = Tuple[int, int, float]


@dataclass
class DijkstraResult:
    source: int
    dist: np.ndarray
    parent: np.ndarray
    push_count: int
    pop_count: int
    relax_count: int


def validate_graph(n: int, edges: Iterable[Edge], source: int) -> List[Edge]:
    """Validate and normalize directed weighted edges for Dijkstra."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= source < n):
        raise ValueError(f"source {source} out of range [0, {n})")

    normalized: List[Edge] = []
    for u, v, w in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}, {w}) has invalid endpoint")
        if not np.isfinite(w):
            raise ValueError(f"edge ({u}, {v}, {w}) has non-finite weight")
        if w < 0:
            raise ValueError(f"edge ({u}, {v}, {w}) has negative weight, not supported")
        normalized.append((int(u), int(v), float(w)))

    return normalized


def build_adjacency(n: int, edges: Sequence[Edge]) -> List[List[Tuple[int, float]]]:
    """Build adjacency list from edge list."""
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        adjacency[u].append((v, w))
    return adjacency


def dijkstra(n: int, edges: Iterable[Edge], source: int) -> DijkstraResult:
    """Compute single-source shortest paths on a non-negative weighted graph."""
    edge_list = validate_graph(n, edges, source)
    adjacency = build_adjacency(n, edge_list)

    dist = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    settled = np.zeros(n, dtype=bool)

    dist[source] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, source)]

    push_count = 1
    pop_count = 0
    relax_count = 0
    eps = 1e-12

    while heap:
        cur_dist, u = heappop(heap)
        pop_count += 1

        # Ignore stale heap items (a shorter distance was already found later).
        if cur_dist > dist[u] + eps:
            continue

        if settled[u]:
            continue
        settled[u] = True

        for v, w in adjacency[u]:
            candidate = cur_dist + w
            if candidate + eps < dist[v]:
                dist[v] = candidate
                parent[v] = u
                relax_count += 1
                heappush(heap, (candidate, v))
                push_count += 1

    return DijkstraResult(
        source=source,
        dist=dist,
        parent=parent,
        push_count=push_count,
        pop_count=pop_count,
        relax_count=relax_count,
    )


def reference_relaxation(n: int, edges: Sequence[Edge], source: int) -> np.ndarray:
    """Reference shortest paths via repeated relaxation (Bellman-Ford style)."""
    dist = np.full(n, np.inf, dtype=float)
    dist[source] = 0.0

    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if np.isfinite(dist[u]) and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break

    return dist


def reconstruct_path(source: int, target: int, parent: np.ndarray) -> Optional[List[int]]:
    """Reconstruct one shortest path from source to target."""
    n = parent.shape[0]
    if not (0 <= source < n and 0 <= target < n):
        raise ValueError("source/target out of range")

    path = [target]
    cur = target
    for _ in range(n + 1):
        if cur == source:
            path.reverse()
            return path
        cur = int(parent[cur])
        if cur == -1:
            return None
        path.append(cur)

    return None


def format_dist(x: float) -> str:
    return f"{x:.1f}" if np.isfinite(x) else "inf"


def run_demo() -> None:
    """Run one fixed non-interactive demo case with assertions."""
    print("=== Dijkstra MVP Demo (directed graph, non-negative weights) ===")

    n = 7
    source = 0
    edges: List[Edge] = [
        (0, 1, 7.0),
        (0, 2, 9.0),
        (0, 5, 14.0),
        (1, 2, 10.0),
        (1, 3, 15.0),
        (2, 3, 11.0),
        (2, 5, 2.0),
        (5, 4, 9.0),
        (3, 4, 6.0),
        # node 6 is unreachable from node 0
    ]

    result = dijkstra(n=n, edges=edges, source=source)
    reference = reference_relaxation(n=n, edges=validate_graph(n, edges, source), source=source)

    if not np.allclose(result.dist, reference, atol=1e-10, equal_nan=False):
        raise AssertionError("dijkstra mismatch against reference relaxation")

    expected = np.array([0.0, 7.0, 9.0, 20.0, 20.0, 11.0, np.inf], dtype=float)
    if not np.allclose(result.dist, expected, atol=1e-10, equal_nan=False):
        raise AssertionError(f"unexpected shortest distances: {result.dist}")

    print(
        "stats:",
        f"push_count={result.push_count},",
        f"pop_count={result.pop_count},",
        f"relax_count={result.relax_count}",
    )
    print("distance summary:")
    for node, d in enumerate(result.dist):
        print(f"  node={node} dist={format_dist(d)} reachable={bool(np.isfinite(d))}")

    targets = [3, 4, 5, 6]
    print("path summary:")
    for t in targets:
        path = reconstruct_path(source, t, result.parent)
        if path is None:
            print(f"  {source}->{t}: unreachable")
        else:
            print(f"  {source}->{t}: {' -> '.join(map(str, path))} (cost={result.dist[t]:.1f})")

    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
