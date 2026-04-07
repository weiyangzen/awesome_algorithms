"""Minimal runnable MVP for maximum independent set by greedy heuristic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

Edge = Tuple[int, int]


try:
    _ = (1).bit_count  # type: ignore[attr-defined]

    def popcount(x: int) -> int:
        return x.bit_count()  # type: ignore[attr-defined]
except AttributeError:

    def popcount(x: int) -> int:
        return bin(x).count("1")


@dataclass
class MISResult:
    """Container for an independent-set solution and run statistics."""

    vertices: List[int]
    iterations: int

    @property
    def size(self) -> int:
        return len(self.vertices)


def build_undirected_adjacency(n: int, edges: Sequence[Edge]) -> np.ndarray:
    """Build adjacency matrix for a simple undirected graph."""
    if n <= 0:
        raise ValueError("n must be positive")

    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) has vertex index out of range")
        if u == v:
            raise ValueError("self-loop is not allowed in this MVP")
        adj[u, v] = True
        adj[v, u] = True

    if not np.array_equal(adj, adj.T):
        raise RuntimeError("internal error: adjacency matrix must be symmetric")
    if np.any(np.diag(adj)):
        raise RuntimeError("internal error: diagonal must be false")
    return adj


def greedy_maximal_independent_set_min_degree(adj: np.ndarray) -> MISResult:
    """
    Greedy heuristic for maximum independent set:
    repeatedly pick one currently active vertex with minimum induced degree.

    This returns a maximal independent set (not guaranteed maximum).
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency matrix must be square")
    if not np.array_equal(adj, adj.T):
        raise ValueError("adjacency matrix must be symmetric (undirected graph)")
    if np.any(np.diag(adj)):
        raise ValueError("adjacency matrix diagonal must be all false")

    n = adj.shape[0]
    active = np.ones(n, dtype=bool)
    chosen: List[int] = []
    rounds = 0

    while np.any(active):
        rounds += 1
        active_idx = np.flatnonzero(active)
        induced = adj[np.ix_(active_idx, active_idx)]
        induced_degree = induced.sum(axis=1)

        min_degree = induced_degree.min()
        candidates = active_idx[induced_degree == min_degree]
        v = int(candidates.min())  # deterministic tie-break

        chosen.append(v)

        # Remove v itself and all its active neighbors.
        active[v] = False
        neighbors = np.flatnonzero(adj[v] & active)
        active[neighbors] = False

    chosen.sort()
    return MISResult(vertices=chosen, iterations=rounds)


def is_independent_set(adj: np.ndarray, vertices: Sequence[int]) -> bool:
    """Check whether a vertex set is independent."""
    if len(vertices) == 0:
        return True

    arr = np.asarray(vertices, dtype=int)
    if len(np.unique(arr)) != len(arr):
        return False

    sub = adj[np.ix_(arr, arr)]
    return not bool(np.any(np.triu(sub, k=1)))


def is_maximal_independent_set(adj: np.ndarray, vertices: Sequence[int]) -> bool:
    """Check whether an independent set is maximal (cannot add any vertex)."""
    if not is_independent_set(adj, vertices):
        return False

    n = adj.shape[0]
    in_set = np.zeros(n, dtype=bool)
    in_set[np.asarray(vertices, dtype=int)] = True

    for v in np.flatnonzero(~in_set):
        # If v has no neighbor in the set, we can add v -> not maximal.
        if not np.any(adj[v] & in_set):
            return False
    return True


def exact_maximum_independent_set_branch_and_bound(
    adj: np.ndarray,
    max_n_for_exact: int = 28,
) -> List[int]:
    """
    Exact MIS via branch-and-bound over bitmasks.

    Intended only as a correctness/quality baseline on small graphs.
    """
    n = adj.shape[0]
    if n > max_n_for_exact:
        raise ValueError(
            f"exact MIS is disabled for n={n}; set max_n_for_exact larger if needed"
        )

    neighbor_masks: List[int] = []
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i, j]:
                mask |= 1 << j
        neighbor_masks.append(mask)

    best_mask = 0

    def dfs(candidate_mask: int, current_mask: int) -> None:
        nonlocal best_mask

        # Upper bound pruning.
        if popcount(current_mask) + popcount(candidate_mask) <= popcount(best_mask):
            return

        if candidate_mask == 0:
            if popcount(current_mask) > popcount(best_mask):
                best_mask = current_mask
            return

        # Branching pivot: vertex with highest degree in current candidate-induced graph.
        temp = candidate_mask
        pivot = -1
        pivot_degree = -1
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = popcount(neighbor_masks[v] & candidate_mask)
            if deg > pivot_degree:
                pivot_degree = deg
                pivot = v
            temp ^= lsb

        v = pivot

        # Branch 1: include v -> remove v and all its neighbors from candidates.
        dfs(candidate_mask & ~(1 << v) & ~neighbor_masks[v], current_mask | (1 << v))
        # Branch 2: exclude v.
        dfs(candidate_mask & ~(1 << v), current_mask)

    dfs((1 << n) - 1, 0)

    exact_vertices = [i for i in range(n) if (best_mask >> i) & 1]
    return exact_vertices


def edge_count(adj: np.ndarray) -> int:
    """Number of undirected edges."""
    return int(np.triu(adj, k=1).sum())


def main() -> None:
    # Deterministic sample where min-degree greedy is valid but not always optimal.
    n = 12
    edges: List[Edge] = [
        (0, 1),
        (0, 2),
        (0, 5),
        (0, 10),
        (1, 7),
        (1, 8),
        (2, 9),
        (2, 10),
        (2, 11),
        (3, 4),
        (3, 6),
        (3, 9),
        (4, 7),
        (4, 9),
        (4, 10),
        (6, 8),
        (8, 9),
        (9, 11),
    ]

    adj = build_undirected_adjacency(n=n, edges=edges)

    greedy = greedy_maximal_independent_set_min_degree(adj)
    exact_vertices = exact_maximum_independent_set_branch_and_bound(adj)

    # Correctness checks.
    if not is_independent_set(adj, greedy.vertices):
        raise AssertionError("greedy result is not an independent set")
    if not is_maximal_independent_set(adj, greedy.vertices):
        raise AssertionError("greedy result should be maximal")

    if not is_independent_set(adj, exact_vertices):
        raise AssertionError("exact baseline returned a non-independent set")
    if len(greedy.vertices) > len(exact_vertices):
        raise AssertionError("greedy set cannot be larger than exact optimum")

    density = (2.0 * edge_count(adj)) / (n * (n - 1))
    approx_ratio = len(greedy.vertices) / len(exact_vertices)

    print("Maximum Independent Set (Greedy MVP)")
    print(f"n_vertices                      : {n}")
    print(f"n_edges                         : {edge_count(adj)}")
    print(f"graph_density                   : {density:.3f}")
    print(f"greedy_vertices                 : {greedy.vertices}")
    print(f"greedy_size                     : {greedy.size}")
    print(f"greedy_iterations               : {greedy.iterations}")
    print(f"exact_vertices                  : {exact_vertices}")
    print(f"exact_size                      : {len(exact_vertices)}")
    print(f"approximation_ratio             : {approx_ratio:.3f}")
    print(
        "quality_note                    : "
        "greedy gives a maximal independent set; it may be smaller than optimum"
    )
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
