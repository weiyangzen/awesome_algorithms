"""Minimal runnable MVP for graph coloring by DSatur heuristic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

Edge = Tuple[int, int]


@dataclass
class ColoringResult:
    """Container for a coloring solution and run statistics."""

    colors: List[int]
    num_colors: int
    order: List[int]
    saturation_updates: int


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


def dsatur_coloring(adj: np.ndarray) -> ColoringResult:
    """
    DSatur heuristic for vertex coloring.

    Selection rule each round:
    1) highest saturation degree;
    2) tie-break by highest static degree;
    3) tie-break by smallest vertex index (deterministic).

    Coloring rule:
    - assign the smallest non-negative color not used by colored neighbors.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency matrix must be square")
    if not np.array_equal(adj, adj.T):
        raise ValueError("adjacency matrix must be symmetric (undirected graph)")
    if np.any(np.diag(adj)):
        raise ValueError("adjacency matrix diagonal must be all false")

    n = adj.shape[0]
    degrees = adj.sum(axis=1).astype(int)

    colors = [-1] * n
    uncolored = np.ones(n, dtype=bool)
    neighbor_used_colors = [set() for _ in range(n)]

    order: List[int] = []
    saturation_updates = 0

    for _ in range(n):
        candidates = np.flatnonzero(uncolored)

        best_v = -1
        best_key = (-1, -1, -10**9)
        for v in candidates:
            sat_v = len(neighbor_used_colors[int(v)])
            key = (sat_v, int(degrees[int(v)]), -int(v))
            if key > best_key:
                best_key = key
                best_v = int(v)

        v = best_v
        forbidden = neighbor_used_colors[v]

        color = 0
        while color in forbidden:
            color += 1

        colors[v] = color
        uncolored[v] = False
        order.append(v)

        for u in np.flatnonzero(adj[v] & uncolored):
            uu = int(u)
            if color not in neighbor_used_colors[uu]:
                neighbor_used_colors[uu].add(color)
                saturation_updates += 1

    num_colors = max(colors) + 1 if colors else 0
    return ColoringResult(
        colors=colors,
        num_colors=num_colors,
        order=order,
        saturation_updates=saturation_updates,
    )


def is_proper_coloring(adj: np.ndarray, colors: Sequence[int]) -> bool:
    """Check whether coloring is valid (adjacent vertices have different colors)."""
    arr = np.asarray(colors, dtype=int)
    if np.any(arr < 0):
        return False

    same_color = arr[:, None] == arr[None, :]
    conflict = np.any(np.triu(adj & same_color, k=1))
    return not bool(conflict)


def edge_count(adj: np.ndarray) -> int:
    """Number of undirected edges."""
    return int(np.triu(adj, k=1).sum())


def color_histogram(colors: Sequence[int]) -> Dict[int, int]:
    """Count how many vertices use each color."""
    hist: Dict[int, int] = {}
    for c in colors:
        hist[c] = hist.get(c, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: kv[0]))


def exact_chromatic_number_backtracking(
    adj: np.ndarray,
    max_n_for_exact: int = 18,
) -> Tuple[int, List[int]]:
    """
    Exact chromatic number by DFS + branch-and-bound.

    This is intentionally small-scale and used only as a quality baseline.
    """
    n = adj.shape[0]
    if n > max_n_for_exact:
        raise ValueError(
            f"exact coloring is disabled for n={n}; set max_n_for_exact larger if needed"
        )

    degrees = adj.sum(axis=1).astype(int)
    neighbors = [np.flatnonzero(adj[v]).astype(int).tolist() for v in range(n)]

    colors = [-1] * n
    best_num = n + 1
    best_coloring: List[int] = []

    def saturation_degree(v: int) -> int:
        used = {colors[u] for u in neighbors[v] if colors[u] >= 0}
        return len(used)

    def choose_vertex() -> int:
        uncolored = [v for v in range(n) if colors[v] < 0]
        return max(uncolored, key=lambda v: (saturation_degree(v), int(degrees[v]), -v))

    def dfs(colored_count: int, used_colors: int) -> None:
        nonlocal best_num, best_coloring

        if used_colors >= best_num:
            return

        if colored_count == n:
            best_num = used_colors
            best_coloring = colors.copy()
            return

        v = choose_vertex()
        forbidden = {colors[u] for u in neighbors[v] if colors[u] >= 0}

        for c in range(used_colors):
            if c in forbidden:
                continue
            colors[v] = c
            dfs(colored_count + 1, used_colors)
            colors[v] = -1

        colors[v] = used_colors
        dfs(colored_count + 1, used_colors + 1)
        colors[v] = -1

    dfs(colored_count=0, used_colors=0)

    if not best_coloring:
        raise RuntimeError("exact solver failed to produce a coloring")
    return best_num, best_coloring


def main() -> None:
    # Fixed undirected graph: an odd wheel plus additional structure.
    # This graph needs 4 colors; DSatur should find an optimal 4-coloring.
    n = 10
    edges: List[Edge] = [
        # Odd wheel W6 equivalent: cycle C5 + hub 5.
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (5, 0),
        (5, 1),
        (5, 2),
        (5, 3),
        (5, 4),
        # Extra triangle and cross edges.
        (6, 7),
        (7, 8),
        (8, 6),
        (6, 5),
        (7, 5),
        (8, 5),
        (6, 1),
        (7, 3),
        (8, 0),
        # One extra vertex connected to mixed colors.
        (9, 2),
        (9, 4),
        (9, 5),
        (9, 7),
    ]

    adj = build_undirected_adjacency(n=n, edges=edges)

    dsatur = dsatur_coloring(adj)
    exact_num, exact_coloring = exact_chromatic_number_backtracking(adj)

    if not is_proper_coloring(adj, dsatur.colors):
        raise AssertionError("DSatur produced an invalid coloring")
    if not is_proper_coloring(adj, exact_coloring):
        raise AssertionError("exact solver produced an invalid coloring")
    if dsatur.num_colors < exact_num:
        raise AssertionError("heuristic cannot use fewer colors than exact optimum")

    density = (2.0 * edge_count(adj)) / (n * (n - 1))
    gap = dsatur.num_colors - exact_num

    print("Graph Coloring - DSatur (MVP)")
    print(f"n_vertices                      : {n}")
    print(f"n_edges                         : {edge_count(adj)}")
    print(f"graph_density                   : {density:.3f}")
    print(f"dsatur_colors                   : {dsatur.colors}")
    print(f"dsatur_num_colors               : {dsatur.num_colors}")
    print(f"dsatur_color_histogram          : {color_histogram(dsatur.colors)}")
    print(f"dsatur_selection_order          : {dsatur.order}")
    print(f"dsatur_saturation_updates       : {dsatur.saturation_updates}")
    print(f"exact_chromatic_number          : {exact_num}")
    print(f"exact_coloring                  : {exact_coloring}")
    print(f"optimality_gap                  : {gap}")
    print(
        "quality_note                    : "
        "DSatur is heuristic; exact baseline is for small-graph verification"
    )
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
