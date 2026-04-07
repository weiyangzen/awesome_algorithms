"""Hamiltonian path MVP using explicit backtracking and bitmask DP check."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def validate_adjacency_matrix(adj: np.ndarray) -> np.ndarray:
    """Validate an undirected simple-graph adjacency matrix."""
    mat = np.asarray(adj, dtype=int)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"adjacency matrix must be square, got shape={mat.shape}")
    if mat.shape[0] == 0:
        raise ValueError("adjacency matrix must be non-empty")
    if not np.all((mat == 0) | (mat == 1)):
        raise ValueError("adjacency matrix must contain only 0/1")
    if not np.array_equal(mat, mat.T):
        raise ValueError("adjacency matrix must be symmetric for undirected graph")
    if not np.all(np.diag(mat) == 0):
        raise ValueError("diagonal entries must be 0 for a simple graph")

    return mat


def build_undirected_adjacency(n: int, edges: Iterable[Tuple[int, int]]) -> np.ndarray:
    """Build a 0/1 undirected adjacency matrix from edge list."""
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    adj = np.zeros((n, n), dtype=int)
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) is out of range for n={n}")
        if u == v:
            raise ValueError("self-loop is not allowed in this MVP")
        adj[u, v] = 1
        adj[v, u] = 1
    return adj


def find_hamiltonian_path_backtracking(adj: np.ndarray) -> Optional[List[int]]:
    """Return one Hamiltonian path using DFS backtracking, or None."""
    mat = validate_adjacency_matrix(adj)
    n = mat.shape[0]

    degrees = np.sum(mat, axis=1)
    used = np.zeros(n, dtype=bool)
    path: List[int] = []

    def dfs(current: int, depth: int) -> bool:
        if depth == n:
            return True

        neighbors = np.where(mat[current] == 1)[0]
        candidates = [int(v) for v in neighbors if not used[v]]
        candidates.sort(key=lambda v: (int(degrees[v]), v))

        for nxt in candidates:
            used[nxt] = True
            path.append(nxt)
            if dfs(nxt, depth + 1):
                return True
            path.pop()
            used[nxt] = False

        return False

    start_order = sorted(range(n), key=lambda v: (int(degrees[v]), v))
    for start in start_order:
        used[:] = False
        path.clear()

        used[start] = True
        path.append(start)

        if dfs(start, 1):
            return path.copy()

    return None


def hamiltonian_path_exists_bitmask_dp(adj: np.ndarray) -> bool:
    """Check existence of Hamiltonian path by bitmask DP."""
    mat = validate_adjacency_matrix(adj)
    n = mat.shape[0]

    if n > 20:
        raise ValueError("bitmask DP is intentionally limited to n <= 20")

    state_count = 1 << n
    dp = np.zeros((state_count, n), dtype=bool)

    for v in range(n):
        dp[1 << v, v] = True

    for mask in range(state_count):
        ends = np.where(dp[mask])[0]
        if ends.size == 0:
            continue
        for end in ends:
            neighbors = np.where(mat[end] == 1)[0]
            for nxt in neighbors:
                bit = 1 << int(nxt)
                if mask & bit:
                    continue
                dp[mask | bit, int(nxt)] = True

    return bool(np.any(dp[state_count - 1]))


def is_valid_hamiltonian_path(adj: np.ndarray, path: Sequence[int]) -> bool:
    """Validate whether `path` is a Hamiltonian path on graph `adj`."""
    mat = validate_adjacency_matrix(adj)
    n = mat.shape[0]

    if len(path) != n:
        return False

    normalized = [int(v) for v in path]
    if sorted(normalized) != list(range(n)):
        return False

    for u, v in zip(normalized, normalized[1:]):
        if mat[u, v] == 0:
            return False

    return True


def run_case(name: str, adj: np.ndarray) -> None:
    """Run one deterministic experiment case and print metrics."""
    print(f"\n=== {name} ===")
    print("adjacency matrix:")
    print(adj)

    path = find_hamiltonian_path_backtracking(adj)
    found_by_backtracking = path is not None
    is_valid = is_valid_hamiltonian_path(adj, path) if path is not None else False
    exists_by_dp = hamiltonian_path_exists_bitmask_dp(adj)

    print(f"backtracking path : {path}")
    print(f"backtracking found: {found_by_backtracking}")
    print(f"path validity     : {is_valid}")
    print(f"bitmask-dp exists : {exists_by_dp}")
    print(f"consistency check : {found_by_backtracking == exists_by_dp}")


def main() -> None:
    np.set_printoptions(linewidth=120)

    # Case 1: a graph with a Hamiltonian path.
    # One valid path is 0-1-2-3-4-5-6-7.
    graph_with_path = build_undirected_adjacency(
        n=8,
        edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (0, 2),
            (1, 4),
            (2, 5),
            (3, 6),
        ],
    )

    # Case 2: star graph K(1,5), which has no Hamiltonian path for n=6.
    graph_without_path = build_undirected_adjacency(
        n=6,
        edges=[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
    )

    run_case("Case 1 (should exist)", graph_with_path)
    run_case("Case 2 (should not exist)", graph_without_path)

    # Optional sanity check for invalid input handling.
    bad = np.array([[0, 1], [0, 0]], dtype=int)
    try:
        _ = find_hamiltonian_path_backtracking(bad)
    except ValueError as exc:
        print("\nExpected failure on invalid adjacency matrix:")
        print(exc)


if __name__ == "__main__":
    main()
