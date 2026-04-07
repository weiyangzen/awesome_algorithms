"""Minimal runnable MVP for Maximum Independent Set via backtracking."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable, List, Optional, Sequence, Tuple


Edge = Tuple[int, int]


@dataclass
class MISResult:
    size: int
    vertices: List[int]
    nodes_visited: int
    pruned_by_bound: int


def popcount(x: int) -> int:
    """Compatibility popcount for Python versions without int.bit_count()."""
    cnt = 0
    while x:
        x &= x - 1
        cnt += 1
    return cnt


def build_adjacency_masks(n: int, edges: Iterable[Edge]) -> List[int]:
    """Build undirected graph adjacency bitmasks."""
    if n <= 0:
        raise ValueError("n must be positive")

    adj = [0] * n
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) has invalid endpoint")
        if u == v:
            raise ValueError(f"self-loop ({u}, {v}) is not supported for MIS")

        adj[u] |= 1 << v
        adj[v] |= 1 << u

    return adj


def mask_to_vertices(mask: int) -> List[int]:
    """Convert a bitmask to a sorted vertex list."""
    vertices: List[int] = []
    while mask:
        lsb = mask & -mask
        vertices.append(lsb.bit_length() - 1)
        mask ^= lsb
    return vertices


def vertices_to_mask(vertices: Sequence[int]) -> int:
    """Convert vertex list to bitmask."""
    mask = 0
    for v in vertices:
        mask |= 1 << v
    return mask


def is_independent_mask(mask: int, adj: Sequence[int]) -> bool:
    """Check whether the vertex subset represented by mask is independent."""
    remaining = mask
    while remaining:
        lsb = remaining & -remaining
        v = lsb.bit_length() - 1
        if adj[v] & (mask ^ lsb):
            return False
        remaining ^= lsb
    return True


def choose_branch_vertex(candidate_mask: int, adj: Sequence[int]) -> int:
    """Choose a branch vertex with maximum degree inside candidate set."""
    tmp = candidate_mask
    best_v = -1
    best_deg = -1

    while tmp:
        lsb = tmp & -tmp
        v = lsb.bit_length() - 1
        deg_in_candidate = popcount(adj[v] & candidate_mask)
        if deg_in_candidate > best_deg:
            best_deg = deg_in_candidate
            best_v = v
        tmp ^= lsb

    return best_v


def maximum_independent_set_backtracking(n: int, edges: Iterable[Edge]) -> MISResult:
    """Solve MIS with branch-and-bound backtracking."""
    adj = build_adjacency_masks(n, edges)

    best_mask = 0
    best_size = 0
    nodes_visited = 0
    pruned_by_bound = 0

    def dfs(candidate_mask: int, chosen_mask: int, chosen_size: int) -> None:
        nonlocal best_mask, best_size, nodes_visited, pruned_by_bound
        nodes_visited += 1

        # Upper bound: every remaining candidate could be selected in the best case.
        if chosen_size + popcount(candidate_mask) <= best_size:
            pruned_by_bound += 1
            return

        if candidate_mask == 0:
            if chosen_size > best_size:
                best_size = chosen_size
                best_mask = chosen_mask
            return

        v = choose_branch_vertex(candidate_mask, adj)
        v_bit = 1 << v
        remaining_without_v = candidate_mask & ~v_bit

        # Branch 1: include v, so neighbors of v must be removed from candidates.
        dfs(remaining_without_v & ~adj[v], chosen_mask | v_bit, chosen_size + 1)

        # Branch 2: exclude v.
        if chosen_size + popcount(remaining_without_v) <= best_size:
            pruned_by_bound += 1
            return
        dfs(remaining_without_v, chosen_mask, chosen_size)

    all_vertices = (1 << n) - 1
    dfs(all_vertices, 0, 0)

    vertices = mask_to_vertices(best_mask)
    return MISResult(
        size=best_size,
        vertices=vertices,
        nodes_visited=nodes_visited,
        pruned_by_bound=pruned_by_bound,
    )


def brute_force_mis(n: int, edges: Iterable[Edge]) -> MISResult:
    """Exact baseline for small graphs only; used for demo self-check."""
    adj = build_adjacency_masks(n, edges)

    best_mask = 0
    best_size = 0
    total_masks = 1 << n

    for mask in range(total_masks):
        size = popcount(mask)
        if size <= best_size:
            continue
        if is_independent_mask(mask, adj):
            best_mask = mask
            best_size = size

    return MISResult(
        size=best_size,
        vertices=mask_to_vertices(best_mask),
        nodes_visited=total_masks,
        pruned_by_bound=0,
    )


def cycle_graph_edges(n: int) -> List[Edge]:
    if n < 3:
        raise ValueError("cycle graph requires n >= 3")
    return [(i, (i + 1) % n) for i in range(n)]


def complete_bipartite_edges(left_size: int, right_size: int) -> Tuple[int, List[Edge]]:
    n = left_size + right_size
    edges: List[Edge] = []
    for u in range(left_size):
        for v in range(left_size, n):
            edges.append((u, v))
    return n, edges


def random_graph_edges(n: int, edge_probability: float, seed: int) -> List[Edge]:
    if not (0.0 <= edge_probability <= 1.0):
        raise ValueError("edge_probability must be in [0, 1]")

    rng = Random(seed)
    edges: List[Edge] = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < edge_probability:
                edges.append((u, v))
    return edges


def run_case(
    title: str,
    n: int,
    edges: List[Edge],
    expected_size: Optional[int] = None,
    verify_with_bruteforce: bool = False,
) -> None:
    print(f"=== {title} ===")
    print(f"n={n}, |E|={len(edges)}")

    result = maximum_independent_set_backtracking(n, edges)
    print(f"MIS size (backtracking): {result.size}")
    print(f"MIS vertices: {result.vertices}")
    print(f"Search nodes visited: {result.nodes_visited}")
    print(f"Pruned by bound: {result.pruned_by_bound}")

    adj = build_adjacency_masks(n, edges)
    mask = vertices_to_mask(result.vertices)
    assert result.size == len(result.vertices)
    assert is_independent_mask(mask, adj), "Result set must be independent"

    if expected_size is not None:
        assert result.size == expected_size, (
            f"Expected MIS size {expected_size}, got {result.size}"
        )

    if verify_with_bruteforce:
        baseline = brute_force_mis(n, edges)
        print(f"Bruteforce size: {baseline.size}")
        print(f"Bruteforce vertices: {baseline.vertices}")
        assert result.size == baseline.size, "Backtracking result mismatches brute force"

    print()


def main() -> None:
    # Case A: odd cycle C5, alpha(C5)=2.
    n_a = 5
    edges_a = cycle_graph_edges(n_a)
    run_case(
        title="Case A: cycle graph C5",
        n=n_a,
        edges=edges_a,
        expected_size=2,
        verify_with_bruteforce=True,
    )

    # Case B: complete bipartite K3,4, alpha(K3,4)=4.
    n_b, edges_b = complete_bipartite_edges(3, 4)
    run_case(
        title="Case B: complete bipartite graph K3,4",
        n=n_b,
        edges=edges_b,
        expected_size=4,
        verify_with_bruteforce=True,
    )

    # Case C: medium random graph; show branch-and-bound stats.
    n_c = 18
    edges_c = random_graph_edges(n=n_c, edge_probability=0.28, seed=495)
    run_case(
        title="Case C: random graph (n=18, p=0.28, seed=495)",
        n=n_c,
        edges=edges_c,
        expected_size=None,
        verify_with_bruteforce=False,
    )


if __name__ == "__main__":
    main()
