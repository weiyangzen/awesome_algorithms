"""Subgraph Isomorphism MVP (Ullmann-style backtracking).

This script solves non-induced subgraph isomorphism for small undirected graphs.
It is intentionally compact and transparent for algorithm learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class Graph:
    """Simple undirected labeled graph backed by a NumPy adjacency matrix."""

    adj: np.ndarray
    labels: Tuple[str, ...]

    @staticmethod
    def from_edges(
        n: int,
        edges: Iterable[Tuple[int, int]],
        labels: Optional[Sequence[str]] = None,
    ) -> "Graph":
        mat = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"edge ({u}, {v}) out of range for n={n}")
            if u == v:
                raise ValueError("self-loop is not supported in this MVP")
            mat[u, v] = 1
            mat[v, u] = 1

        if labels is None:
            labels = ["*"] * n
        if len(labels) != n:
            raise ValueError("labels length must equal n")

        return Graph(adj=mat, labels=tuple(labels))

    @property
    def n(self) -> int:
        return int(self.adj.shape[0])

    @property
    def degree(self) -> np.ndarray:
        return self.adj.sum(axis=1).astype(np.int32)


@dataclass
class SearchStats:
    nodes_expanded: int = 0
    backtracks: int = 0
    pruned_by_refine: int = 0


def initial_candidate_matrix(pattern: Graph, target: Graph) -> np.ndarray:
    """Build candidate matrix M where M[i, j] means pattern i may map to target j."""
    p_deg = pattern.degree
    t_deg = target.degree

    m = np.zeros((pattern.n, target.n), dtype=bool)
    for i in range(pattern.n):
        for j in range(target.n):
            # Necessary conditions for non-induced subgraph isomorphism.
            if pattern.labels[i] == target.labels[j] and p_deg[i] <= t_deg[j]:
                m[i, j] = True
    return m


def refine_candidates(pattern: Graph, target: Graph, m: np.ndarray) -> bool:
    """Arc-consistency-like pruning used in Ullmann-style algorithms.

    For each candidate pair (i, j), every neighbor of i must have at least one
    candidate among neighbors of j.
    """
    changed = True
    while changed:
        changed = False
        for i in range(pattern.n):
            nbr_i = np.flatnonzero(pattern.adj[i])
            row_candidates = np.flatnonzero(m[i])
            for j in row_candidates:
                nbr_j = np.flatnonzero(target.adj[j])
                feasible = True
                for x in nbr_i:
                    if not np.any(m[x, nbr_j]):
                        feasible = False
                        break
                if not feasible:
                    m[i, j] = False
                    changed = True
                    if not np.any(m[i]):
                        return False
    return True


def choose_next_pattern_vertex(
    m: np.ndarray,
    mapping: Dict[int, int],
    used_target_vertices: Set[int],
) -> Tuple[int, List[int]]:
    """MRV heuristic: choose unmapped pattern vertex with fewest candidates."""
    best_i = -1
    best_candidates: List[int] = []

    for i in range(m.shape[0]):
        if i in mapping:
            continue
        candidates = [j for j in np.flatnonzero(m[i]).tolist() if j not in used_target_vertices]
        if best_i == -1 or len(candidates) < len(best_candidates):
            best_i = i
            best_candidates = candidates

    return best_i, best_candidates


def locally_consistent(
    pattern: Graph,
    target: Graph,
    mapping: Dict[int, int],
    i: int,
    j: int,
) -> bool:
    """Check edge-preservation against already assigned pairs.

    Non-induced variant: if an edge exists in pattern, it must exist in target.
    """
    for pi, tj in mapping.items():
        if pattern.adj[i, pi] == 1 and target.adj[j, tj] != 1:
            return False
    return True


def backtrack_search(
    pattern: Graph,
    target: Graph,
    m: np.ndarray,
    mapping: Dict[int, int],
    used_target_vertices: Set[int],
    stats: SearchStats,
) -> Optional[Dict[int, int]]:
    if len(mapping) == pattern.n:
        return dict(mapping)

    i, candidates = choose_next_pattern_vertex(m, mapping, used_target_vertices)
    if i == -1 or len(candidates) == 0:
        stats.backtracks += 1
        return None

    for j in candidates:
        if not locally_consistent(pattern, target, mapping, i, j):
            continue

        stats.nodes_expanded += 1

        m_next = m.copy()

        # Fix i -> j.
        m_next[i, :] = False
        m_next[i, j] = True

        # Injective mapping: no other pattern vertex can map to j.
        for r in range(pattern.n):
            if r != i:
                m_next[r, j] = False

        # Early row-empty check before deeper refinement.
        if not np.all(np.any(m_next, axis=1)):
            stats.pruned_by_refine += 1
            continue

        if not refine_candidates(pattern, target, m_next):
            stats.pruned_by_refine += 1
            continue

        mapping[i] = j
        used_target_vertices.add(j)

        solved = backtrack_search(pattern, target, m_next, mapping, used_target_vertices, stats)
        if solved is not None:
            return solved

        used_target_vertices.remove(j)
        del mapping[i]

    stats.backtracks += 1
    return None


def solve_subgraph_isomorphism(pattern: Graph, target: Graph) -> Tuple[Optional[Dict[int, int]], SearchStats]:
    """Return one mapping if exists, otherwise None."""
    stats = SearchStats()

    if pattern.n > target.n:
        return None, stats

    m = initial_candidate_matrix(pattern, target)
    if not np.all(np.any(m, axis=1)):
        return None, stats

    if not refine_candidates(pattern, target, m):
        return None, stats

    mapping: Dict[int, int] = {}
    used_target_vertices: Set[int] = set()
    solution = backtrack_search(pattern, target, m, mapping, used_target_vertices, stats)
    return solution, stats


def brute_force_subgraph_isomorphism(pattern: Graph, target: Graph) -> Optional[Dict[int, int]]:
    """Small-graph verifier for demo correctness checks.

    This is exponential and only used on tiny cases to sanity-check the MVP solver.
    """
    if pattern.n > target.n:
        return None

    p_vertices = list(range(pattern.n))
    t_vertices = list(range(target.n))

    for perm in permutations(t_vertices, pattern.n):
        mapping = dict(zip(p_vertices, perm))

        label_ok = all(pattern.labels[i] == target.labels[mapping[i]] for i in p_vertices)
        if not label_ok:
            continue

        edge_ok = True
        for u in p_vertices:
            for v in p_vertices:
                if u < v and pattern.adj[u, v] == 1 and target.adj[mapping[u], mapping[v]] == 0:
                    edge_ok = False
                    break
            if not edge_ok:
                break

        if edge_ok:
            return mapping

    return None


def build_demo_graphs() -> Tuple[Graph, Graph, Graph]:
    """Construct one target graph and two pattern graphs (positive/negative)."""

    target = Graph.from_edges(
        n=9,
        edges=[
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 3),
            (1, 4),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 6),
            (5, 7),
            (6, 7),
            (4, 8),
            (6, 8),
        ],
        labels=["A", "B", "B", "C", "C", "D", "D", "E", "F"],
    )

    pattern_exists = Graph.from_edges(
        n=4,
        # Triangle + tail
        edges=[(0, 1), (1, 2), (0, 2), (2, 3)],
        labels=["A", "B", "C", "D"],
    )

    pattern_not_exists = Graph.from_edges(
        n=4,
        # K4 with the same labels, stricter than the target connectivity.
        edges=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        labels=["A", "B", "C", "D"],
    )

    return target, pattern_exists, pattern_not_exists


def format_mapping(mapping: Optional[Dict[int, int]]) -> str:
    if mapping is None:
        return "None"
    pairs = [f"p{p}->t{t}" for p, t in sorted(mapping.items())]
    return "{" + ", ".join(pairs) + "}"


def run_case(case_name: str, pattern: Graph, target: Graph) -> None:
    mapping, stats = solve_subgraph_isomorphism(pattern, target)

    # Tiny-case sanity check: algorithm result should match brute-force existence.
    brute_mapping = brute_force_subgraph_isomorphism(pattern, target)
    assert (mapping is None) == (brute_mapping is None), (
        f"Mismatch between MVP solver and brute-force checker in {case_name}"
    )

    print(f"\n[{case_name}]")
    print(f"pattern_n={pattern.n}, target_n={target.n}")
    print(f"found={mapping is not None}")
    print(f"mapping={format_mapping(mapping)}")
    print(
        "stats="
        f"nodes_expanded:{stats.nodes_expanded}, "
        f"backtracks:{stats.backtracks}, "
        f"pruned_by_refine:{stats.pruned_by_refine}"
    )


def main() -> None:
    target, pattern_exists, pattern_not_exists = build_demo_graphs()

    print("Subgraph Isomorphism MVP (Ullmann-style)")
    print("- graph type: undirected, labeled")
    print("- matching type: non-induced subgraph isomorphism")

    run_case("Case-1 Exists", pattern_exists, target)
    run_case("Case-2 NotExists", pattern_not_exists, target)


if __name__ == "__main__":
    main()
