"""Minimal runnable MVP for Bellman-Ford single-source shortest paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class BellmanFordResult:
    source: int
    dist: np.ndarray
    pred: np.ndarray
    has_negative_cycle: bool
    negative_cycle_affected: np.ndarray


def validate_input(n: int, edges: Iterable[Tuple[int, int, float]], source: int) -> List[Tuple[int, int, float]]:
    """Validate graph input and return normalized edge list."""
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
        normalized.append((int(u), int(v), float(w)))

    return normalized


def build_adjacency(n: int, edges: Sequence[Tuple[int, int, float]]) -> List[List[int]]:
    """Build adjacency list for reachability propagation."""
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v, _ in edges:
        adj[u].append(v)
    return adj


def bellman_ford(n: int, edges: Iterable[Tuple[int, int, float]], source: int) -> BellmanFordResult:
    """Run Bellman-Ford and detect source-reachable negative cycles."""
    edge_list = validate_input(n, edges, source)

    dist = np.full(n, np.inf, dtype=float)
    pred = np.full(n, -1, dtype=int)
    dist[source] = 0.0

    # Relax all edges up to n-1 rounds.
    for _ in range(n - 1):
        updated = False
        for u, v, w in edge_list:
            if np.isfinite(dist[u]) and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                updated = True
        if not updated:
            break

    # One extra scan: nodes that can still be relaxed are influenced by a reachable negative cycle.
    seeds: List[int] = []
    for u, v, w in edge_list:
        if np.isfinite(dist[u]) and dist[u] + w < dist[v]:
            seeds.append(v)

    negative_cycle_affected = np.zeros(n, dtype=bool)
    if seeds:
        adj = build_adjacency(n, edge_list)
        stack = list(set(seeds))
        while stack:
            node = stack.pop()
            if negative_cycle_affected[node]:
                continue
            negative_cycle_affected[node] = True
            stack.extend(adj[node])

    return BellmanFordResult(
        source=source,
        dist=dist,
        pred=pred,
        has_negative_cycle=bool(np.any(negative_cycle_affected)),
        negative_cycle_affected=negative_cycle_affected,
    )


def reconstruct_path(
    source: int,
    target: int,
    pred: np.ndarray,
    negative_cycle_affected: np.ndarray,
) -> Optional[List[int]]:
    """Reconstruct one source->target path if it is reachable and well-defined."""
    n = pred.shape[0]
    if not (0 <= source < n and 0 <= target < n):
        raise ValueError("source/target out of range")
    if negative_cycle_affected[target]:
        return None

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

    # Defensive fallback for unexpected predecessor loops.
    return None


def format_distance_row(node: int, result: BellmanFordResult) -> str:
    """Format one node distance line for readable output."""
    reachable = bool(np.isfinite(result.dist[node]))
    affected = bool(result.negative_cycle_affected[node])

    if affected:
        dist_text = "-inf(*)"
    elif reachable:
        dist_text = f"{result.dist[node]:.1f}"
    else:
        dist_text = "inf"

    return (
        f"  node={node:>2d} | dist={dist_text:>8s} "
        f"| reachable={str(reachable):<5s} | negative-cycle-affected={affected}"
    )


def print_result_summary(result: BellmanFordResult) -> None:
    print(f"source = {result.source}")
    print(f"has_negative_cycle = {result.has_negative_cycle}")
    print("distance summary from source:")
    for node in range(result.dist.shape[0]):
        print(format_distance_row(node, result))


def show_paths(result: BellmanFordResult, targets: Sequence[int]) -> None:
    source = result.source
    print("paths from source:")
    for t in targets:
        path = reconstruct_path(source, t, result.pred, result.negative_cycle_affected)
        if path is None:
            print(f"  path {source}->{t}: undefined (unreachable or negative-cycle-affected)")
            continue
        cost = result.dist[t]
        print(f"  path {source}->{t}: {' -> '.join(map(str, path))} (cost={cost:.1f})")


def run_case_without_negative_cycle() -> None:
    print("=== Case A: negative edges but no negative cycle ===")
    n = 5
    source = 0
    edges = [
        (0, 1, 6.0),
        (0, 2, 7.0),
        (1, 2, 8.0),
        (1, 3, 5.0),
        (1, 4, -4.0),
        (2, 3, -3.0),
        (2, 4, 9.0),
        (3, 1, -2.0),
        (4, 0, 2.0),
        (4, 3, 7.0),
    ]

    result = bellman_ford(n=n, edges=edges, source=source)
    print_result_summary(result)
    show_paths(result, targets=[1, 2, 3, 4])


def run_case_with_negative_cycle() -> None:
    print("\n=== Case B: source-reachable negative cycle ===")
    n = 6
    source = 0
    edges = [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 1, -4.0),  # cycle 1->2->3->1 has total weight -2
        (2, 4, 2.0),
    ]

    result = bellman_ford(n=n, edges=edges, source=source)
    print_result_summary(result)
    show_paths(result, targets=[1, 2, 3, 4, 5])



def main() -> None:
    run_case_without_negative_cycle()
    run_case_with_negative_cycle()


if __name__ == "__main__":
    main()
