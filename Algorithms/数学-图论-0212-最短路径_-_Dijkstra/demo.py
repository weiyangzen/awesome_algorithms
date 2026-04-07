"""Minimal runnable MVP for Dijkstra single-source shortest paths."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DijkstraResult:
    source: int
    dist: np.ndarray
    pred: np.ndarray
    popped_count: int
    relax_count: int
    push_count: int


def validate_input(n: int, edges: Iterable[Tuple[int, int, float]], source: int) -> List[Tuple[int, int, float]]:
    """Validate graph input and normalize edge records."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= source < n):
        raise ValueError(f"source {source} out of range [0, {n})")

    normalized: List[Tuple[int, int, float]] = []
    for u, v, w in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}, {w}) has invalid node index")
        if not np.isfinite(w):
            raise ValueError(f"edge ({u}, {v}, {w}) has non-finite weight")
        if w < 0:
            raise ValueError(f"edge ({u}, {v}, {w}) has negative weight, not allowed by Dijkstra")
        normalized.append((int(u), int(v), float(w)))

    return normalized


def build_adjacency(n: int, edges: Sequence[Tuple[int, int, float]]) -> List[List[Tuple[int, float]]]:
    """Build adjacency list: adj[u] = [(v, w), ...]."""
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    return adj


def dijkstra(n: int, edges: Iterable[Tuple[int, int, float]], source: int) -> DijkstraResult:
    """Run Dijkstra for a directed graph with non-negative weights."""
    edge_list = validate_input(n, edges, source)
    adj = build_adjacency(n, edge_list)

    dist = np.full(n, np.inf, dtype=float)
    pred = np.full(n, -1, dtype=int)
    settled = np.zeros(n, dtype=bool)

    dist[source] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, source)]
    push_count = 1
    popped_count = 0
    relax_count = 0

    eps = 1e-12
    while heap:
        cur_dist, u = heappop(heap)
        popped_count += 1

        # Skip stale queue entries.
        if cur_dist > dist[u] + eps:
            continue

        if settled[u]:
            continue
        settled[u] = True

        for v, w in adj[u]:
            cand = cur_dist + w
            if cand + eps < dist[v]:
                dist[v] = cand
                pred[v] = u
                relax_count += 1
                heappush(heap, (cand, v))
                push_count += 1

    return DijkstraResult(
        source=source,
        dist=dist,
        pred=pred,
        popped_count=popped_count,
        relax_count=relax_count,
        push_count=push_count,
    )


def relaxation_reference(n: int, edges: Sequence[Tuple[int, int, float]], source: int) -> np.ndarray:
    """Reference shortest paths by repeated edge relaxation (Bellman-Ford style, no negative edges here)."""
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


def reconstruct_path(source: int, target: int, pred: np.ndarray) -> Optional[List[int]]:
    """Reconstruct one shortest path from source to target using predecessor links."""
    n = pred.shape[0]
    if not (0 <= source < n and 0 <= target < n):
        raise ValueError("source/target out of range")

    path = [target]
    cur = target
    for _ in range(n + 1):
        if cur == source:
            path.reverse()
            return path
        cur = int(pred[cur])
        if cur == -1:
            return None
        path.append(cur)

    return None


def format_distance(d: float) -> str:
    """Pretty formatting for distances."""
    if np.isfinite(d):
        return f"{d:.1f}"
    return "inf"


def print_summary(result: DijkstraResult) -> None:
    """Print per-node distance summary and simple search statistics."""
    print(f"source = {result.source}")
    print(
        "stats: "
        f"push_count={result.push_count}, "
        f"popped_count={result.popped_count}, "
        f"relax_count={result.relax_count}"
    )
    print("distance summary from source:")
    for node, d in enumerate(result.dist):
        reachable = bool(np.isfinite(d))
        print(f"  node={node:>2d} | dist={format_distance(d):>5s} | reachable={reachable}")


def run_demo_case() -> None:
    print("=== Dijkstra MVP Demo: directed graph with one unreachable node ===")
    n = 7
    source = 0
    edges = [
        (0, 1, 4.0),
        (0, 2, 1.0),
        (2, 1, 2.0),
        (1, 3, 1.0),
        (2, 3, 5.0),
        (1, 4, 7.0),
        (3, 4, 3.0),
        (4, 5, 1.0),
        (2, 5, 10.0),
        # node 6 remains unreachable from source 0
    ]

    result = dijkstra(n=n, edges=edges, source=source)
    print_summary(result)

    reference = relaxation_reference(n=n, edges=validate_input(n, edges, source), source=source)
    if not np.allclose(result.dist, reference, atol=1e-10, equal_nan=False):
        raise AssertionError("Dijkstra result mismatch against reference relaxation distances")

    targets = [3, 4, 5, 6]
    print("paths from source:")
    for t in targets:
        path = reconstruct_path(source, t, result.pred)
        if path is None:
            print(f"  path {source}->{t}: unreachable")
            continue
        print(f"  path {source}->{t}: {' -> '.join(map(str, path))} (cost={result.dist[t]:.1f})")

    print("All checks passed.")


def main() -> None:
    run_demo_case()


if __name__ == "__main__":
    main()
