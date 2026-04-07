"""Tree DP MVP: Maximum Weight Independent Set on a Tree.

The script builds one fixed tree instance, solves it with tree dynamic
programming in O(n), reconstructs one optimal node set, and validates
against brute-force enumeration for reliability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TreeInstance:
    """Weighted tree instance for MWIS."""

    weights: np.ndarray
    edges: List[Tuple[int, int]]


def validate_tree_instance(instance: TreeInstance) -> List[List[int]]:
    """Validate tree inputs and build adjacency list.

    Requirements:
    - n >= 1
    - weights are 1D and non-negative
    - edge count is n - 1
    - edge endpoints are valid, no self-loop
    - graph is connected
    """

    weights = instance.weights
    edges = instance.edges

    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array")

    n = int(weights.shape[0])
    if n == 0:
        raise ValueError("tree must contain at least one node")
    if np.any(weights < 0):
        raise ValueError("node weights must be non-negative")
    if len(edges) != n - 1:
        raise ValueError("a tree with n nodes must contain exactly n-1 edges")

    adj: List[List[int]] = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge[{idx}] has invalid node index")
        if u == v:
            raise ValueError(f"edge[{idx}] is a self-loop")
        adj[u].append(v)
        adj[v].append(u)

    # Connectivity check: with n-1 edges, connected <=> tree.
    visited = [False] * n
    stack = [0]
    visited[0] = True
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)

    if not all(visited):
        raise ValueError("graph is disconnected, not a valid tree")

    return adj


def build_rooted_parent_and_postorder(
    adj: Sequence[Sequence[int]], root: int
) -> tuple[np.ndarray, List[int]]:
    """Root the tree and return parent array + postorder list."""

    n = len(adj)
    if not (0 <= root < n):
        raise ValueError("root index out of range")

    parent = np.full(n, -1, dtype=np.int64)
    order: List[int] = []

    stack = [root]
    parent[root] = root
    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                stack.append(v)

    if len(order) != n:
        raise ValueError("tree rooting failed due to disconnected graph")

    postorder = list(reversed(order))
    return parent, postorder


def solve_tree_mwis(
    instance: TreeInstance, root: int = 0
) -> tuple[int, List[int], np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
    """Solve MWIS on tree by two-state tree DP and reconstruct one optimum."""

    adj = validate_tree_instance(instance)
    parent, postorder = build_rooted_parent_and_postorder(adj, root)

    n = len(adj)
    take = np.zeros(n, dtype=np.int64)
    skip = np.zeros(n, dtype=np.int64)

    for u in postorder:
        include_u = int(instance.weights[u])
        exclude_u = 0
        for v in adj[u]:
            if v == int(parent[u]):
                continue
            include_u += int(skip[v])
            exclude_u += int(max(take[v], skip[v]))

        take[u] = include_u
        skip[u] = exclude_u

    best_value = int(max(take[root], skip[root]))

    # Reconstruct one optimal solution.
    selected = np.zeros(n, dtype=bool)
    stack: List[Tuple[int, int, bool]] = [(root, int(parent[root]), False)]
    while stack:
        u, p, parent_taken = stack.pop()
        if parent_taken:
            choose_u = False
        else:
            choose_u = int(take[u]) >= int(skip[u])

        selected[u] = choose_u

        for v in adj[u]:
            if v == p:
                continue
            stack.append((v, u, choose_u))

    selected_nodes = [int(i) for i in np.where(selected)[0]]
    selected_value = int(instance.weights[selected].sum())
    if selected_value != best_value:
        raise RuntimeError("reconstructed plan value does not match DP optimum")

    return best_value, selected_nodes, take, skip, parent, adj


def brute_force_mwis(instance: TreeInstance) -> tuple[int, List[int]]:
    """Brute-force optimum for small trees; used only as a correctness oracle."""

    n = int(instance.weights.shape[0])
    if n > 24:
        raise ValueError("brute force is intended only for small n (<= 24)")

    best_value = -1
    best_mask = 0

    for mask in range(1 << n):
        valid = True
        for u, v in instance.edges:
            if ((mask >> u) & 1) and ((mask >> v) & 1):
                valid = False
                break
        if not valid:
            continue

        total = 0
        for i in range(n):
            if (mask >> i) & 1:
                total += int(instance.weights[i])

        if total > best_value:
            best_value = total
            best_mask = mask

    best_nodes = [i for i in range(n) if (best_mask >> i) & 1]
    return int(best_value), best_nodes


def is_independent_set(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> bool:
    """Check whether `nodes` is an independent set under given edges."""

    node_set = set(nodes)
    for u, v in edges:
        if u in node_set and v in node_set:
            return False
    return True


def main() -> None:
    # Deterministic demo instance; no interactive input.
    instance = TreeInstance(
        weights=np.array([6, 4, 7, 3, 8, 5, 2, 9, 4], dtype=np.int64),
        edges=[
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 5),
            (2, 6),
            (5, 7),
            (5, 8),
        ],
    )

    best_value, selected_nodes, take, skip, parent, _adj = solve_tree_mwis(instance, root=0)
    brute_force_value, brute_force_nodes = brute_force_mwis(instance)

    selected_set = set(selected_nodes)
    node_rows = []
    for i in range(int(instance.weights.shape[0])):
        node_rows.append(
            {
                "node": i,
                "weight": int(instance.weights[i]),
                "parent": int(parent[i]) if i != 0 else -1,
                "dp_take": int(take[i]),
                "dp_skip": int(skip[i]),
                "chosen": i in selected_set,
            }
        )

    node_df = pd.DataFrame(node_rows)
    selected_value = int(sum(int(instance.weights[i]) for i in selected_nodes))

    print("=== Tree DP Demo: MWIS on Tree ===")
    print(f"node_count = {instance.weights.shape[0]}")
    print("edges:")
    print(pd.DataFrame(instance.edges, columns=["u", "v"]).to_string(index=False))

    print("\nNode DP table:")
    print(node_df.to_string(index=False))

    print("\nSummary:")
    print(f"best_value_by_dp         = {best_value}")
    print(f"best_value_by_plan       = {selected_value}")
    print(f"best_value_by_bruteforce = {brute_force_value}")
    print(f"selected_nodes_by_dp     = {selected_nodes}")
    print(f"selected_nodes_bruteforce= {brute_force_nodes}")
    print(f"is_independent_set       = {is_independent_set(selected_nodes, instance.edges)}")
    print(f"dp_matches_bruteforce    = {best_value == brute_force_value}")

    if not is_independent_set(selected_nodes, instance.edges):
        raise RuntimeError("reconstructed node set violates independent-set constraint")
    if selected_value != best_value:
        raise RuntimeError("reconstructed node set value does not match DP optimum")
    if best_value != brute_force_value:
        raise RuntimeError("DP optimum does not match brute-force optimum")


if __name__ == "__main__":
    main()
