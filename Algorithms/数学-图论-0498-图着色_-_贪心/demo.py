"""Greedy graph coloring MVP with an exact small-graph baseline.

This demo focuses on order sensitivity of greedy coloring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class GreedyColoringResult:
    order: List[int]
    colors: List[int]
    num_colors: int


@dataclass(frozen=True)
class ExactColoringResult:
    chromatic_number: int
    colors: List[int]
    search_nodes: int


def build_undirected_adjacency(n: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Build a symmetric adjacency matrix for a simple undirected graph."""
    if n < 0:
        raise ValueError("n must be non-negative")

    adj = np.zeros((n, n), dtype=np.bool_)
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) contains invalid vertex index")
        if u == v:
            raise ValueError(f"self-loop detected at vertex {u}")
        adj[u, v] = True
        adj[v, u] = True
    return adj


def validate_adjacency_matrix(adj: np.ndarray) -> None:
    """Validate that the matrix is square, symmetric, and loop-free."""
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency matrix must be square")
    if np.any(np.diag(adj)):
        raise ValueError("self-loop detected on diagonal")
    if not np.array_equal(adj, adj.T):
        raise ValueError("adjacency matrix must be symmetric for undirected graph")


def is_proper_coloring(adj: np.ndarray, colors: Sequence[int]) -> bool:
    """Check whether a vertex coloring satisfies all edge constraints."""
    validate_adjacency_matrix(adj)
    n = adj.shape[0]
    if len(colors) != n:
        return False
    if any(c < 0 for c in colors):
        return False

    for u in range(n):
        for v in np.flatnonzero(adj[u]):
            vv = int(v)
            if vv > u and colors[u] == colors[vv]:
                return False
    return True


def greedy_coloring(adj: np.ndarray, order: Sequence[int]) -> GreedyColoringResult:
    """Greedy coloring with the 'smallest available color' rule."""
    validate_adjacency_matrix(adj)
    n = adj.shape[0]

    if sorted(order) != list(range(n)):
        raise ValueError("order must be a permutation of 0..n-1")

    colors = [-1] * n
    for v in order:
        neighbors = np.flatnonzero(adj[v])
        used = {colors[int(u)] for u in neighbors if colors[int(u)] != -1}

        color = 0
        while color in used:
            color += 1
        colors[v] = color

    num_colors = 0 if n == 0 else (max(colors) + 1)
    if not is_proper_coloring(adj, colors):
        raise RuntimeError("greedy coloring produced an invalid solution")

    return GreedyColoringResult(order=list(order), colors=colors, num_colors=num_colors)


def largest_degree_first_order(adj: np.ndarray) -> List[int]:
    """Sort vertices by degree descending, tie-break by vertex id ascending."""
    validate_adjacency_matrix(adj)
    n = adj.shape[0]
    degrees = np.sum(adj, axis=1).astype(int)
    return sorted(range(n), key=lambda v: (-degrees[v], v))


def can_color_with_k(
    adj: np.ndarray,
    order: Sequence[int],
    k: int,
) -> Tuple[bool, List[int], int]:
    """Backtracking feasibility check for k-colorability."""
    n = adj.shape[0]
    colors = [-1] * n
    search_nodes = 0

    def dfs(pos: int) -> bool:
        nonlocal search_nodes
        search_nodes += 1

        if pos == n:
            return True

        v = order[pos]
        neighbors = np.flatnonzero(adj[v])
        used = {colors[int(u)] for u in neighbors if colors[int(u)] != -1}

        for c in range(k):
            if c in used:
                continue
            colors[v] = c
            if dfs(pos + 1):
                return True
            colors[v] = -1

        return False

    ok = dfs(0)
    return ok, colors.copy(), search_nodes


def exact_chromatic_number_backtracking(
    adj: np.ndarray,
    max_n_for_exact: int = 18,
) -> ExactColoringResult:
    """Compute exact chromatic number via incremental k-colorability checks."""
    validate_adjacency_matrix(adj)
    n = adj.shape[0]

    if n == 0:
        return ExactColoringResult(chromatic_number=0, colors=[], search_nodes=0)
    if n > max_n_for_exact:
        raise ValueError(
            f"exact solver supports n <= {max_n_for_exact}, got n={n}"
        )

    # Use greedy upper bound so exact search only scans a narrow k-range.
    order = largest_degree_first_order(adj)
    greedy_upper = greedy_coloring(adj, order).num_colors

    total_nodes = 0
    for k in range(1, greedy_upper + 1):
        ok, colors, nodes = can_color_with_k(adj, order, k)
        total_nodes += nodes
        if ok:
            if not is_proper_coloring(adj, colors):
                raise RuntimeError("exact solver returned invalid coloring")
            return ExactColoringResult(
                chromatic_number=k,
                colors=colors,
                search_nodes=total_nodes,
            )

    raise RuntimeError("exact solver failed to find a feasible coloring")


def build_crown_graph(k: int) -> np.ndarray:
    """Build the crown graph on 2k vertices.

    Vertices 0..k-1 are A-part, k..2k-1 are B-part.
    Edge (Ai, Bj) exists iff i != j.
    """
    if k < 2:
        raise ValueError("k must be >= 2 for crown graph")

    n = 2 * k
    edges: List[Tuple[int, int]] = []
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            edges.append((i, k + j))
    return build_undirected_adjacency(n, edges)


def interleaved_crown_order(k: int) -> List[int]:
    """Order A0, B0, A1, B1, ..., A(k-1), B(k-1)."""
    order: List[int] = []
    for i in range(k):
        order.extend([i, k + i])
    return order


def color_classes(colors: Sequence[int]) -> Dict[int, List[int]]:
    """Group vertices by color id for readable reporting."""
    buckets: Dict[int, List[int]] = {}
    for v, c in enumerate(colors):
        buckets.setdefault(c, []).append(v)
    return {c: buckets[c] for c in sorted(buckets)}


def edge_count(adj: np.ndarray) -> int:
    """Count undirected edges from upper triangle."""
    return int(np.count_nonzero(np.triu(adj, k=1)))


def main() -> None:
    k = 5
    adj = build_crown_graph(k)
    n = adj.shape[0]
    m = edge_count(adj)

    orders: Dict[str, List[int]] = {
        "natural": list(range(n)),
        "interleaved": interleaved_crown_order(k),
        "largest_degree_first": largest_degree_first_order(adj),
    }

    greedy_results = {
        name: greedy_coloring(adj, order)
        for name, order in orders.items()
    }

    exact = exact_chromatic_number_backtracking(adj, max_n_for_exact=18)

    # Correctness checks
    for result in greedy_results.values():
        assert is_proper_coloring(adj, result.colors)
    assert is_proper_coloring(adj, exact.colors)

    # Crown graph optimal color count is 2, and interleaved order is intentionally poor.
    assert exact.chromatic_number == 2
    assert greedy_results["natural"].num_colors == exact.chromatic_number
    assert greedy_results["largest_degree_first"].num_colors == exact.chromatic_number
    assert greedy_results["interleaved"].num_colors > exact.chromatic_number

    print("Greedy Graph Coloring Demo (MATH-0498)")
    print(f"Graph: crown graph k={k}, |V|={n}, |E|={m}")
    print()

    for name, result in greedy_results.items():
        print(f"[{name}] colors_used={result.num_colors}")
        print(f"  order={result.order}")
        print(f"  color_classes={color_classes(result.colors)}")

    print()
    print("[exact_backtracking]")
    print(f"  chromatic_number={exact.chromatic_number}")
    print(f"  search_nodes={exact.search_nodes}")
    print(f"  color_classes={color_classes(exact.colors)}")
    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
