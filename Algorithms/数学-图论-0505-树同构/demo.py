"""Tree isomorphism MVP based on AHU canonical encoding.

The script runs fixed test cases without interactive input.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

Edge = Tuple[int, int]
Adjacency = List[List[int]]


@dataclass(frozen=True)
class Case:
    """Single deterministic test case for tree isomorphism."""

    name: str
    n1: int
    edges1: Sequence[Edge]
    n2: int
    edges2: Sequence[Edge]
    expected: bool


def validate_and_build_tree(n: int, edges: Iterable[Edge]) -> Adjacency:
    """Validate tree constraints and return adjacency list.

    Conditions enforced:
    - n >= 1
    - number of edges == n - 1
    - no self-loop / duplicate edge
    - all vertices in [0, n)
    - graph connected
    """
    if n <= 0:
        raise ValueError(f"n must be >= 1, got {n}")

    edge_list = list(edges)
    if len(edge_list) != n - 1:
        raise ValueError(f"a tree with n={n} must have n-1 edges, got {len(edge_list)}")

    adjacency: Adjacency = [[] for _ in range(n)]
    seen_undirected = set()

    for u, v in edge_list:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) contains invalid vertex id for n={n}")
        if u == v:
            raise ValueError(f"self-loop detected at vertex {u}")

        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen_undirected:
            raise ValueError(f"duplicate edge detected: ({a}, {b})")
        seen_undirected.add((a, b))

        adjacency[u].append(v)
        adjacency[v].append(u)

    visited = [False] * n
    stack = [0]
    visited[0] = True

    while stack:
        u = stack.pop()
        for v in adjacency[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)

    if not all(visited):
        raise ValueError("graph is not connected, therefore not a tree")

    return adjacency


def find_tree_centers(adjacency: Adjacency) -> List[int]:
    """Find one or two centers of an undirected tree by leaf peeling."""
    n = len(adjacency)
    if n == 1:
        return [0]

    degree = [len(nei) for nei in adjacency]
    leaves = deque(i for i, d in enumerate(degree) if d <= 1)
    removed = len(leaves)

    while removed < n:
        layer_size = len(leaves)
        new_leaves: List[int] = []

        for _ in range(layer_size):
            leaf = leaves.popleft()
            for nb in adjacency[leaf]:
                degree[nb] -= 1
                if degree[nb] == 1:
                    new_leaves.append(nb)

        removed += len(new_leaves)
        leaves.extend(new_leaves)

    return sorted(leaves)


def rooted_canonical_code(adjacency: Adjacency, root: int) -> str:
    """Compute AHU canonical code of a rooted tree."""
    n = len(adjacency)
    parent = [-1] * n
    parent[root] = root

    stack = [root]
    order: List[int] = []

    while stack:
        u = stack.pop()
        order.append(u)
        for v in adjacency[u]:
            if parent[v] == -1:
                parent[v] = u
                stack.append(v)

    if len(order) != n:
        raise ValueError("rooted traversal did not cover all vertices")

    code = [""] * n
    for u in reversed(order):
        child_codes = [code[v] for v in adjacency[u] if parent[v] == u]
        child_codes.sort()
        code[u] = "(" + "".join(child_codes) + ")"

    return code[root]


def unrooted_canonical_code(adjacency: Adjacency) -> Tuple[str, List[int], List[str]]:
    """Return canonical code for an unrooted tree and tracing details."""
    centers = find_tree_centers(adjacency)
    candidate_codes = sorted(rooted_canonical_code(adjacency, c) for c in centers)
    return candidate_codes[0], centers, candidate_codes


def are_isomorphic(n1: int, edges1: Sequence[Edge], n2: int, edges2: Sequence[Edge]) -> bool:
    """Main API: decide whether two trees are isomorphic."""
    if n1 != n2:
        return False

    tree1 = validate_and_build_tree(n1, edges1)
    tree2 = validate_and_build_tree(n2, edges2)

    code1, _, _ = unrooted_canonical_code(tree1)
    code2, _, _ = unrooted_canonical_code(tree2)
    return code1 == code2


def relabel_edges(n: int, edges: Sequence[Edge], permutation: Sequence[int]) -> List[Edge]:
    """Relabel tree vertices by an old->new permutation."""
    if len(permutation) != n or sorted(permutation) != list(range(n)):
        raise ValueError("permutation must be a bijection over [0, n)")
    return [(permutation[u], permutation[v]) for u, v in edges]


def run_case(case: Case) -> bool:
    """Run one case, print details, and return the actual result."""
    tree1 = validate_and_build_tree(case.n1, case.edges1)
    tree2 = validate_and_build_tree(case.n2, case.edges2)

    canonical1, centers1, candidates1 = unrooted_canonical_code(tree1)
    canonical2, centers2, candidates2 = unrooted_canonical_code(tree2)
    actual = canonical1 == canonical2 and case.n1 == case.n2

    print(f"[{case.name}]")
    print(f"expected={case.expected}, actual={actual}")
    print(f"tree1_centers={centers1}, tree1_candidate_codes={candidates1}")
    print(f"tree2_centers={centers2}, tree2_candidate_codes={candidates2}")
    print(f"tree1_canonical={canonical1}")
    print(f"tree2_canonical={canonical2}")
    print("-")

    return actual


def build_cases() -> List[Case]:
    """Create deterministic demo cases."""
    balanced_tree = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    balanced_perm = [3, 1, 4, 0, 2, 6, 5]

    unbalanced_tree = [
        (0, 1),
        (1, 2),
        (1, 3),
        (3, 4),
        (3, 5),
        (2, 6),
        (6, 7),
        (6, 8),
        (8, 9),
    ]
    unbalanced_perm = [5, 0, 8, 2, 4, 7, 1, 6, 9, 3]

    path7 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    star7 = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]

    return [
        Case(
            name="isomorphic_balanced_relabel",
            n1=7,
            edges1=balanced_tree,
            n2=7,
            edges2=relabel_edges(7, balanced_tree, balanced_perm),
            expected=True,
        ),
        Case(
            name="non_isomorphic_path_vs_star",
            n1=7,
            edges1=path7,
            n2=7,
            edges2=star7,
            expected=False,
        ),
        Case(
            name="isomorphic_unbalanced_relabel",
            n1=10,
            edges1=unbalanced_tree,
            n2=10,
            edges2=relabel_edges(10, unbalanced_tree, unbalanced_perm),
            expected=True,
        ),
        Case(
            name="single_vertex_tree",
            n1=1,
            edges1=[],
            n2=1,
            edges2=[],
            expected=True,
        ),
    ]


def main() -> None:
    cases = build_cases()
    all_passed = True

    for case in cases:
        actual = run_case(case)
        if actual != case.expected:
            all_passed = False

    # Also expose the API-level function once in output.
    api_check = are_isomorphic(1, [], 1, [])
    print(f"api_smoke_check_are_isomorphic={api_check}")

    if not all_passed:
        raise RuntimeError("At least one test case failed")

    print("ALL_CASES_PASSED=True")


if __name__ == "__main__":
    main()
