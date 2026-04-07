"""Minimal runnable MVP for bipartite maximum matching via Hopcroft-Karp."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np


@dataclass
class HopcroftKarpResult:
    n_left: int
    n_right: int
    pair_left: np.ndarray
    pair_right: np.ndarray
    matching_size: int
    bfs_rounds: int
    dfs_calls: int
    edge_scans: int
    augmenting_paths: int


def validate_input(
    n_left: int,
    n_right: int,
    edges: Iterable[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Validate bipartite graph input and return a deduplicated edge list."""
    if n_left < 0 or n_right < 0:
        raise ValueError("n_left and n_right must be non-negative")

    normalized: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for u, v in edges:
        if not (0 <= u < n_left):
            raise ValueError(f"left node index out of range: {u}")
        if not (0 <= v < n_right):
            raise ValueError(f"right node index out of range: {v}")
        e = (int(u), int(v))
        if e in seen:
            continue
        seen.add(e)
        normalized.append(e)
    return normalized


def build_adjacency(n_left: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """Build left-to-right adjacency list: adj[u] = [v1, v2, ...]."""
    adj: List[List[int]] = [[] for _ in range(n_left)]
    for u, v in edges:
        adj[u].append(v)
    return adj


def hopcroft_karp(
    n_left: int,
    n_right: int,
    edges: Iterable[Tuple[int, int]],
) -> HopcroftKarpResult:
    """Compute maximum-cardinality matching on a bipartite graph."""
    edge_list = validate_input(n_left, n_right, edges)
    adj = build_adjacency(n_left, edge_list)

    pair_left = np.full(n_left, -1, dtype=int)
    pair_right = np.full(n_right, -1, dtype=int)
    dist = np.full(n_left, -1, dtype=int)

    bfs_rounds = 0
    dfs_calls = 0
    edge_scans = 0
    augmenting_paths = 0

    def bfs() -> bool:
        nonlocal bfs_rounds, edge_scans
        bfs_rounds += 1

        q: deque[int] = deque()
        for u in range(n_left):
            if pair_left[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1

        found_augmenting = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                edge_scans += 1
                u2 = int(pair_right[v])
                if u2 == -1:
                    found_augmenting = True
                elif dist[u2] == -1:
                    dist[u2] = dist[u] + 1
                    q.append(u2)
        return found_augmenting

    def dfs(u: int) -> bool:
        nonlocal dfs_calls, edge_scans
        dfs_calls += 1

        for v in adj[u]:
            edge_scans += 1
            u2 = int(pair_right[v])
            if u2 == -1 or (dist[u2] == dist[u] + 1 and dfs(u2)):
                pair_left[u] = v
                pair_right[v] = u
                return True

        # Mark dead-end in current BFS layer graph to avoid repeated useless DFS.
        dist[u] = -1
        return False

    matching_size = 0
    while bfs():
        for u in range(n_left):
            if pair_left[u] == -1 and dfs(u):
                matching_size += 1
                augmenting_paths += 1

    return HopcroftKarpResult(
        n_left=n_left,
        n_right=n_right,
        pair_left=pair_left,
        pair_right=pair_right,
        matching_size=matching_size,
        bfs_rounds=bfs_rounds,
        dfs_calls=dfs_calls,
        edge_scans=edge_scans,
        augmenting_paths=augmenting_paths,
    )


def matching_pairs(pair_left: np.ndarray) -> List[Tuple[int, int]]:
    """Convert left match array into sorted pair list."""
    pairs: List[Tuple[int, int]] = []
    for u, v in enumerate(pair_left.tolist()):
        if v != -1:
            pairs.append((u, int(v)))
    return pairs


def is_valid_matching(result: HopcroftKarpResult, edges: Sequence[Tuple[int, int]]) -> bool:
    """Validate cardinality and edge legality of the matching result."""
    edge_set = set(edges)

    used_right: Set[int] = set()
    count = 0
    for u, v in matching_pairs(result.pair_left):
        if (u, v) not in edge_set:
            return False
        if v in used_right:
            return False
        used_right.add(v)
        if result.pair_right[v] != u:
            return False
        count += 1

    return count == result.matching_size


def brute_force_maximum_size(n_left: int, adj: Sequence[Sequence[int]], n_right: int) -> int:
    """Reference maximum matching size by backtracking (for small graphs)."""
    used = np.zeros(n_right, dtype=bool)
    best = 0

    def backtrack(u: int, matched: int) -> None:
        nonlocal best
        if u == n_left:
            if matched > best:
                best = matched
            return

        # Branch-and-bound upper bound.
        if matched + (n_left - u) <= best:
            return

        # Option 1: leave this left node unmatched.
        backtrack(u + 1, matched)

        # Option 2: match with any currently free right neighbor.
        for v in adj[u]:
            if used[v]:
                continue
            used[v] = True
            backtrack(u + 1, matched + 1)
            used[v] = False

    backtrack(0, 0)
    return best


def run_demo_case() -> None:
    print("=== Hopcroft-Karp MVP Demo: bipartite maximum matching ===")

    n_left = 7
    n_right = 6
    edges = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
        (4, 5),
        (5, 4),
        (5, 5),
        (0, 1),  # duplicate edge, will be deduplicated by validate_input
        # left node 6 has no edges and stays unmatched
    ]

    normalized_edges = validate_input(n_left, n_right, edges)
    result = hopcroft_karp(n_left, n_right, normalized_edges)

    if not is_valid_matching(result, normalized_edges):
        raise AssertionError("Hopcroft-Karp produced an invalid matching")

    ref_size = brute_force_maximum_size(
        n_left=n_left,
        n_right=n_right,
        adj=build_adjacency(n_left, normalized_edges),
    )
    if result.matching_size != ref_size:
        raise AssertionError(
            f"matching size mismatch: hopcroft_karp={result.matching_size}, reference={ref_size}"
        )

    pairs = matching_pairs(result.pair_left)
    print(f"left size={n_left}, right size={n_right}, edges={len(normalized_edges)}")
    print(f"maximum matching size = {result.matching_size}")
    print(f"matching pairs (left -> right): {pairs}")
    print(
        "stats: "
        f"bfs_rounds={result.bfs_rounds}, "
        f"dfs_calls={result.dfs_calls}, "
        f"edge_scans={result.edge_scans}, "
        f"augmenting_paths={result.augmenting_paths}"
    )
    print("All checks passed.")


def main() -> None:
    run_demo_case()


if __name__ == "__main__":
    main()
