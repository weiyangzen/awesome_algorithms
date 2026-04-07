"""Minimal runnable MVP for bipartite maximum matching via Hungarian/Kuhn DFS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np


@dataclass
class HungarianKuhnResult:
    n_left: int
    n_right: int
    pair_left: np.ndarray
    pair_right: np.ndarray
    matching_size: int
    augment_attempts: int
    augment_successes: int
    dfs_calls: int
    edge_scans: int


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
        edge = (int(u), int(v))
        if edge in seen:
            continue
        seen.add(edge)
        normalized.append(edge)
    return normalized


def build_adjacency(n_left: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """Build left-to-right adjacency list: adj[u] = [v1, v2, ...]."""
    adj: List[List[int]] = [[] for _ in range(n_left)]
    for u, v in edges:
        adj[u].append(v)
    return adj


def hungarian_kuhn(
    n_left: int,
    n_right: int,
    edges: Iterable[Tuple[int, int]],
) -> HungarianKuhnResult:
    """Compute maximum-cardinality matching on bipartite graph by DFS augmenting paths."""
    edge_list = validate_input(n_left, n_right, edges)
    adj = build_adjacency(n_left, edge_list)

    pair_left = np.full(n_left, -1, dtype=int)
    pair_right = np.full(n_right, -1, dtype=int)

    augment_attempts = 0
    augment_successes = 0
    dfs_calls = 0
    edge_scans = 0

    def try_augment(u: int, seen_right: np.ndarray) -> bool:
        nonlocal dfs_calls, edge_scans
        dfs_calls += 1

        for v in adj[u]:
            edge_scans += 1
            if seen_right[v]:
                continue
            seen_right[v] = True

            matched_left = int(pair_right[v])
            if matched_left == -1 or try_augment(matched_left, seen_right):
                pair_right[v] = u
                pair_left[u] = v
                return True

        return False

    matching_size = 0
    for u in range(n_left):
        if pair_left[u] != -1:
            continue
        augment_attempts += 1
        seen_right = np.zeros(n_right, dtype=bool)
        if try_augment(u, seen_right):
            matching_size += 1
            augment_successes += 1

    return HungarianKuhnResult(
        n_left=n_left,
        n_right=n_right,
        pair_left=pair_left,
        pair_right=pair_right,
        matching_size=matching_size,
        augment_attempts=augment_attempts,
        augment_successes=augment_successes,
        dfs_calls=dfs_calls,
        edge_scans=edge_scans,
    )


def matching_pairs(pair_left: np.ndarray) -> List[Tuple[int, int]]:
    """Convert left match array into sorted pair list."""
    pairs: List[Tuple[int, int]] = []
    for u, v in enumerate(pair_left.tolist()):
        if v != -1:
            pairs.append((u, int(v)))
    return pairs


def is_valid_matching(result: HungarianKuhnResult, edges: Sequence[Tuple[int, int]]) -> bool:
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
    used_right = np.zeros(n_right, dtype=bool)
    best = 0

    def backtrack(u: int, matched: int) -> None:
        nonlocal best
        if u == n_left:
            if matched > best:
                best = matched
            return

        if matched + (n_left - u) <= best:
            return

        # Option 1: leave current left node unmatched.
        backtrack(u + 1, matched)

        # Option 2: match with a free right neighbor.
        for v in adj[u]:
            if used_right[v]:
                continue
            used_right[v] = True
            backtrack(u + 1, matched + 1)
            used_right[v] = False

    backtrack(0, 0)
    return best


def run_demo_case() -> None:
    print("=== Hungarian/Kuhn MVP Demo: bipartite maximum matching ===")

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
        (0, 1),  # duplicate edge: validate_input deduplicates it
        # left node 6 has no incident edge, so it remains unmatched
    ]

    normalized_edges = validate_input(n_left, n_right, edges)
    result = hungarian_kuhn(n_left, n_right, normalized_edges)
    if not is_valid_matching(result, normalized_edges):
        raise AssertionError("Hungarian/Kuhn produced an invalid matching")

    ref_size = brute_force_maximum_size(
        n_left=n_left,
        n_right=n_right,
        adj=build_adjacency(n_left, normalized_edges),
    )
    if result.matching_size != ref_size:
        raise AssertionError(
            f"matching size mismatch: hungarian_kuhn={result.matching_size}, reference={ref_size}"
        )

    pairs = matching_pairs(result.pair_left)
    print(f"left size={n_left}, right size={n_right}, edges={len(normalized_edges)}")
    print(f"maximum matching size = {result.matching_size}")
    print(f"matching pairs (left -> right): {pairs}")
    print(
        "stats: "
        f"augment_attempts={result.augment_attempts}, "
        f"augment_successes={result.augment_successes}, "
        f"dfs_calls={result.dfs_calls}, "
        f"edge_scans={result.edge_scans}"
    )
    print("All checks passed.")


def main() -> None:
    run_demo_case()


if __name__ == "__main__":
    main()
