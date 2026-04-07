"""Graph Isomorphism MVP.

This script implements an educational, transparent solver for undirected
vertex-labeled graph isomorphism:
1) quick invariant filtering,
2) pairwise 1-WL color refinement,
3) backtracking with adjacency consistency and forward checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
        if n <= 0:
            raise ValueError("n must be positive")

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

    @property
    def edge_count(self) -> int:
        return int(self.adj.sum() // 2)


@dataclass
class SearchStats:
    nodes_expanded: int = 0
    backtracks: int = 0
    pruned_by_forward_check: int = 0


def permute_graph(graph: Graph, perm_old_to_new: Sequence[int]) -> Graph:
    """Return a relabeled copy of graph.

    perm_old_to_new[u] = new index of old vertex u.
    """
    n = graph.n
    if len(perm_old_to_new) != n:
        raise ValueError("permutation length mismatch")
    if sorted(perm_old_to_new) != list(range(n)):
        raise ValueError("invalid permutation")

    inv_new_to_old = np.argsort(np.asarray(perm_old_to_new, dtype=np.int32))
    new_adj = graph.adj[np.ix_(inv_new_to_old, inv_new_to_old)].copy()
    new_labels = tuple(graph.labels[int(old)] for old in inv_new_to_old)
    return Graph(adj=new_adj, labels=new_labels)


def invariant_check(g1: Graph, g2: Graph) -> bool:
    """Fast necessary conditions before refinement/search."""
    if g1.n != g2.n:
        return False
    if g1.edge_count != g2.edge_count:
        return False
    if sorted(g1.degree.tolist()) != sorted(g2.degree.tolist()):
        return False
    if sorted(g1.labels) != sorted(g2.labels):
        return False
    return True


def _compress_pair_tokens(
    tokens1: Sequence[Tuple[object, ...]],
    tokens2: Sequence[Tuple[object, ...]],
) -> Tuple[np.ndarray, np.ndarray]:
    token_to_id: Dict[Tuple[object, ...], int] = {}
    for tok in sorted(set(tokens1) | set(tokens2)):
        token_to_id[tok] = len(token_to_id)

    c1 = np.array([token_to_id[t] for t in tokens1], dtype=np.int32)
    c2 = np.array([token_to_id[t] for t in tokens2], dtype=np.int32)
    return c1, c2


def _neighbor_color_signature(adj_row: np.ndarray, colors: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    nbr_idx = np.flatnonzero(adj_row)
    if nbr_idx.size == 0:
        return ()
    nbr_colors = colors[nbr_idx]
    unique, counts = np.unique(nbr_colors, return_counts=True)
    return tuple((int(c), int(k)) for c, k in zip(unique.tolist(), counts.tolist()))


def pair_color_refinement(g1: Graph, g2: Graph) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Run pairwise 1-WL color refinement and compare color class histograms."""
    init_tokens_1 = [(g1.labels[i], int(g1.degree[i])) for i in range(g1.n)]
    init_tokens_2 = [(g2.labels[i], int(g2.degree[i])) for i in range(g2.n)]
    c1, c2 = _compress_pair_tokens(init_tokens_1, init_tokens_2)

    while True:
        bins1 = np.bincount(c1)
        bins2 = np.bincount(c2)
        if not np.array_equal(bins1, bins2):
            return False, c1, c2

        next_tokens_1 = [
            (int(c1[i]), _neighbor_color_signature(g1.adj[i], c1))
            for i in range(g1.n)
        ]
        next_tokens_2 = [
            (int(c2[i]), _neighbor_color_signature(g2.adj[i], c2))
            for i in range(g2.n)
        ]
        nc1, nc2 = _compress_pair_tokens(next_tokens_1, next_tokens_2)

        if np.array_equal(c1, nc1) and np.array_equal(c2, nc2):
            return True, c1, c2
        c1, c2 = nc1, nc2


def locally_consistent(
    g1: Graph,
    g2: Graph,
    u: int,
    v: int,
    mapping: Dict[int, int],
) -> bool:
    """Check edge/non-edge preservation against already mapped vertices."""
    for mu, mv in mapping.items():
        if g1.adj[u, mu] != g2.adj[v, mv]:
            return False
    return True


def forward_check(
    g1: Graph,
    g2: Graph,
    colors1: np.ndarray,
    class_to_v2: Dict[int, List[int]],
    mapping: Dict[int, int],
    used_v2: set[int],
) -> bool:
    """Ensure every unmapped vertex in g1 still has at least one legal candidate."""
    n = g1.n
    for u in range(n):
        if u in mapping:
            continue
        cls = int(colors1[u])
        feasible = False
        for v in class_to_v2[cls]:
            if v in used_v2:
                continue
            if locally_consistent(g1, g2, u, v, mapping):
                feasible = True
                break
        if not feasible:
            return False
    return True


def choose_next_vertex(
    g1: Graph,
    g2: Graph,
    colors1: np.ndarray,
    class_to_v2: Dict[int, List[int]],
    mapping: Dict[int, int],
    used_v2: set[int],
) -> Tuple[int, List[int]]:
    """MRV-like choice: pick unmapped vertex with the fewest legal candidates."""
    best_u = -1
    best_candidates: List[int] = []

    for u in range(g1.n):
        if u in mapping:
            continue
        cls = int(colors1[u])
        candidates = [
            v
            for v in class_to_v2[cls]
            if v not in used_v2 and locally_consistent(g1, g2, u, v, mapping)
        ]
        if best_u == -1 or len(candidates) < len(best_candidates):
            best_u = u
            best_candidates = candidates
    return best_u, best_candidates


def backtrack_isomorphism(
    g1: Graph,
    g2: Graph,
    colors1: np.ndarray,
    class_to_v2: Dict[int, List[int]],
    mapping: Dict[int, int],
    used_v2: set[int],
    stats: SearchStats,
) -> Optional[Dict[int, int]]:
    if len(mapping) == g1.n:
        return dict(mapping)

    u, candidates = choose_next_vertex(g1, g2, colors1, class_to_v2, mapping, used_v2)
    if u == -1 or not candidates:
        stats.backtracks += 1
        return None

    for v in candidates:
        stats.nodes_expanded += 1
        mapping[u] = v
        used_v2.add(v)

        if forward_check(g1, g2, colors1, class_to_v2, mapping, used_v2):
            solved = backtrack_isomorphism(g1, g2, colors1, class_to_v2, mapping, used_v2, stats)
            if solved is not None:
                return solved
        else:
            stats.pruned_by_forward_check += 1

        used_v2.remove(v)
        del mapping[u]

    stats.backtracks += 1
    return None


def solve_graph_isomorphism(g1: Graph, g2: Graph) -> Tuple[bool, Optional[Dict[int, int]], SearchStats]:
    """Return (isomorphic, mapping, stats)."""
    stats = SearchStats()
    if not invariant_check(g1, g2):
        return False, None, stats

    ok, colors1, colors2 = pair_color_refinement(g1, g2)
    if not ok:
        return False, None, stats

    class_to_v2: Dict[int, List[int]] = {}
    for c in np.unique(colors2):
        class_to_v2[int(c)] = np.flatnonzero(colors2 == c).tolist()

    mapping: Dict[int, int] = {}
    used_v2: set[int] = set()
    sol = backtrack_isomorphism(g1, g2, colors1, class_to_v2, mapping, used_v2, stats)
    return sol is not None, sol, stats


def brute_force_isomorphic(g1: Graph, g2: Graph) -> bool:
    """Exponential verifier for tiny graphs (for cross-checking the MVP)."""
    if g1.n != g2.n:
        return False
    n = g1.n
    if n > 9:
        raise ValueError("brute force verifier is limited to n <= 9")

    for perm in permutations(range(n)):
        label_ok = True
        for u in range(n):
            if g1.labels[u] != g2.labels[perm[u]]:
                label_ok = False
                break
        if not label_ok:
            continue

        p = np.array(perm, dtype=np.int32)
        if np.array_equal(g1.adj, g2.adj[np.ix_(p, p)]):
            return True
    return False


def random_graph(n: int, edge_prob: float, rng: random.Random) -> Graph:
    labels_pool = ("A", "B", "C")
    labels = [labels_pool[rng.randrange(len(labels_pool))] for _ in range(n)]
    edges: List[Tuple[int, int]] = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < edge_prob:
                edges.append((u, v))
    return Graph.from_edges(n=n, edges=edges, labels=labels)


def build_fixed_cases() -> Tuple[Graph, Graph, Graph]:
    base = Graph.from_edges(
        n=7,
        edges=[
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 5),
            (5, 6),
        ],
        labels=["A", "B", "C", "B", "C", "A", "D"],
    )

    # old -> new
    perm = [3, 0, 4, 1, 5, 2, 6]
    iso_graph = permute_graph(base, perm)

    non_iso = Graph.from_edges(
        n=7,
        edges=[
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            # removed (4,5) to break isomorphism while keeping labels same
            (5, 6),
        ],
        labels=["A", "B", "C", "B", "C", "A", "D"],
    )
    return base, iso_graph, non_iso


def run_fixed_demo() -> None:
    g1, g2_iso, g2_non_iso = build_fixed_cases()

    ok1, mapping1, stats1 = solve_graph_isomorphism(g1, g2_iso)
    if not ok1 or mapping1 is None:
        raise AssertionError("fixed positive case failed")

    ok2, mapping2, stats2 = solve_graph_isomorphism(g1, g2_non_iso)
    if ok2 or mapping2 is not None:
        raise AssertionError("fixed negative case failed")

    print("[fixed-1] isomorphic case")
    print(f"  result={ok1}, mapping={mapping1}")
    print(
        "  stats: "
        f"expanded={stats1.nodes_expanded}, backtracks={stats1.backtracks}, "
        f"pruned={stats1.pruned_by_forward_check}"
    )

    print("[fixed-2] non-isomorphic case")
    print(f"  result={ok2}, mapping={mapping2}")
    print(
        "  stats: "
        f"expanded={stats2.nodes_expanded}, backtracks={stats2.backtracks}, "
        f"pruned={stats2.pruned_by_forward_check}"
    )


def run_random_crosscheck(rounds: int = 80, seed: int = 20260407) -> None:
    rng = random.Random(seed)
    for rid in range(1, rounds + 1):
        n = rng.randint(2, 8)
        p = rng.uniform(0.2, 0.7)
        g1 = random_graph(n=n, edge_prob=p, rng=rng)

        perm = list(range(n))
        rng.shuffle(perm)  # old -> new
        g2_pos = permute_graph(g1, perm)

        fast_pos, _, _ = solve_graph_isomorphism(g1, g2_pos)
        brute_pos = brute_force_isomorphic(g1, g2_pos)
        if fast_pos != brute_pos:
            raise AssertionError(
                "positive cross-check mismatch: "
                f"round={rid}, n={n}, fast={fast_pos}, brute={brute_pos}"
            )

        g2_rand = random_graph(n=n, edge_prob=rng.uniform(0.2, 0.7), rng=rng)
        fast_rand, _, _ = solve_graph_isomorphism(g1, g2_rand)
        brute_rand = brute_force_isomorphic(g1, g2_rand)
        if fast_rand != brute_rand:
            raise AssertionError(
                "random-pair cross-check mismatch: "
                f"round={rid}, n={n}, fast={fast_rand}, brute={brute_rand}"
            )

    print(f"[random] passed {rounds} rounds cross-check (seed={seed}).")


def main() -> None:
    run_fixed_demo()
    run_random_crosscheck()
    print("All checks passed.")


if __name__ == "__main__":
    main()
