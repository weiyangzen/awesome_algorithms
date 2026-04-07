"""Tarjan SCC minimal runnable MVP.

Run:
    python3 demo.py
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


def tarjan_scc(n: int, edges: Sequence[Tuple[int, int]]) -> Tuple[List[List[int]], List[int]]:
    """Compute strongly connected components using Tarjan algorithm.

    Args:
        n: Number of nodes labeled from 0 to n-1.
        edges: Directed edges (u, v).

    Returns:
        sccs: SCC list in Tarjan discovery-pop order.
        comp_id: comp_id[u] is the SCC index of node u in `sccs`.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) is out of range for n={n}")
        adj[u].append(v)

    dfn = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: List[int] = []
    sccs: List[List[int]] = []
    comp_id = [-1] * n
    index = 0

    def dfs(u: int) -> None:
        nonlocal index
        dfn[u] = index
        low[u] = index
        index += 1

        stack.append(u)
        on_stack[u] = True

        for v in adj[u]:
            if dfn[v] == -1:
                dfs(v)
                low[u] = min(low[u], low[v])
            elif on_stack[v]:
                low[u] = min(low[u], dfn[v])

        if low[u] == dfn[u]:
            component: List[int] = []
            component_index = len(sccs)
            while True:
                x = stack.pop()
                on_stack[x] = False
                comp_id[x] = component_index
                component.append(x)
                if x == u:
                    break
            sccs.append(component)

    for node in range(n):
        if dfn[node] == -1:
            dfs(node)

    return sccs, comp_id


def canonicalize_sccs(
    sccs: Sequence[Sequence[int]], comp_id: Sequence[int]
) -> Tuple[List[List[int]], List[int]]:
    """Sort nodes inside SCCs and renumber SCC IDs by min node for stable display."""
    if not sccs:
        return [], list(comp_id)

    normalized = [sorted(component) for component in sccs]
    old_order = list(range(len(normalized)))
    old_order.sort(key=lambda i: normalized[i][0])

    old_to_new = {old: new for new, old in enumerate(old_order)}
    canonical_sccs = [normalized[old] for old in old_order]
    canonical_comp_id = [old_to_new[cid] for cid in comp_id]
    return canonical_sccs, canonical_comp_id


def condensation_edges(edges: Sequence[Tuple[int, int]], comp_id: Sequence[int]) -> List[Tuple[int, int]]:
    """Build DAG edges after SCC contraction."""
    dag = set()
    for u, v in edges:
        cu, cv = comp_id[u], comp_id[v]
        if cu != cv:
            dag.add((cu, cv))
    return sorted(dag)


def main() -> None:
    # Example graph:
    # SCC A: 0 -> 1 -> 2 -> 0
    # SCC B: 3 -> 4 -> 5 -> 3
    # SCC C: 6 -> 7 -> 6
    # Cross edges: 2->3, 5->6, 4->6
    n = 8
    edges = [
        (0, 1),
        (1, 2),
        (2, 0),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 3),
        (5, 6),
        (4, 6),
        (6, 7),
        (7, 6),
    ]

    raw_sccs, raw_comp_id = tarjan_scc(n, edges)
    sccs, comp_id = canonicalize_sccs(raw_sccs, raw_comp_id)
    dag = condensation_edges(edges, comp_id)

    print("Tarjan SCC demo")
    print(f"nodes = {n}, edges = {len(edges)}")
    print("SCCs (canonical order):")
    for i, comp in enumerate(sccs):
        print(f"  C{i}: {comp}")

    print("node -> component:")
    for u, cid in enumerate(comp_id):
        print(f"  {u} -> C{cid}")

    print(f"condensation DAG edges: {dag}")


if __name__ == "__main__":
    main()
