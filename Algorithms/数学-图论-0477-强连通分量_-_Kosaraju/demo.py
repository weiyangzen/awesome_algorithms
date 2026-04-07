"""Minimal runnable MVP for strongly connected components via Kosaraju."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class SCCResult:
    components: List[List[int]]
    component_of: List[int]
    condensation_edges: List[Tuple[int, int]]


def build_graph(n: int, edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
    """Build a directed graph as sorted adjacency lists."""
    if n <= 0:
        raise ValueError("n must be positive")

    adjacency_sets = [set() for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) has invalid node index")
        adjacency_sets[u].add(v)

    return [sorted(neighbors) for neighbors in adjacency_sets]


def reverse_graph(graph: Sequence[Sequence[int]]) -> List[List[int]]:
    """Return transpose graph G^T."""
    n = len(graph)
    rev = [[] for _ in range(n)]
    for u, neighbors in enumerate(graph):
        for v in neighbors:
            rev[v].append(u)

    for arr in rev:
        arr.sort()
    return rev


def finish_order(graph: Sequence[Sequence[int]]) -> List[int]:
    """First DFS pass: collect nodes by finishing times (postorder)."""
    n = len(graph)
    visited = [False] * n
    order: List[int] = []

    for start in range(n):
        if visited[start]:
            continue

        # Stack frame: (node, next_neighbor_index).
        stack: List[Tuple[int, int]] = [(start, 0)]
        visited[start] = True

        while stack:
            node, next_idx = stack[-1]
            if next_idx < len(graph[node]):
                nxt = graph[node][next_idx]
                stack[-1] = (node, next_idx + 1)
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append((nxt, 0))
            else:
                order.append(node)
                stack.pop()

    return order


def collect_component(
    graph_t: Sequence[Sequence[int]],
    start: int,
    visited: List[bool],
) -> List[int]:
    """Second DFS pass on G^T: collect one SCC."""
    stack = [start]
    visited[start] = True
    component: List[int] = []

    while stack:
        node = stack.pop()
        component.append(node)
        for nxt in graph_t[node]:
            if not visited[nxt]:
                visited[nxt] = True
                stack.append(nxt)

    component.sort()
    return component


def kosaraju_scc(n: int, edges: Iterable[Tuple[int, int]]) -> SCCResult:
    """Compute SCCs of a directed graph with Kosaraju's algorithm."""
    graph = build_graph(n, edges)
    graph_t = reverse_graph(graph)

    order = finish_order(graph)

    visited = [False] * n
    components: List[List[int]] = []
    component_of = [-1] * n

    for node in reversed(order):
        if visited[node]:
            continue

        comp = collect_component(graph_t, node, visited)
        cid = len(components)
        for x in comp:
            component_of[x] = cid
        components.append(comp)

    cond_set = set()
    for u in range(n):
        cu = component_of[u]
        for v in graph[u]:
            cv = component_of[v]
            if cu != cv:
                cond_set.add((cu, cv))

    condensation_edges = sorted(cond_set)
    return SCCResult(
        components=components,
        component_of=component_of,
        condensation_edges=condensation_edges,
    )


def print_result(title: str, n: int, edges: Sequence[Tuple[int, int]]) -> None:
    print(f"\n=== {title} ===")
    print(f"n = {n}")
    print(f"edges = {list(edges)}")

    result = kosaraju_scc(n, edges)

    print(f"SCC count = {len(result.components)}")
    for cid, comp in enumerate(result.components):
        print(f"  SCC#{cid}: {comp}")

    print(f"component_of(node) = {result.component_of}")
    print(f"condensation DAG edges = {result.condensation_edges}")


def main() -> None:
    # Case A: multiple SCCs.
    n_a = 8
    edges_a = [
        (0, 1),
        (1, 2),
        (2, 0),  # SCC: {0,1,2}
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 3),  # SCC: {3,4,5}
        (5, 6),
        (6, 7),
        (7, 6),  # SCC: {6,7}
    ]
    print_result("Case A: Multiple SCCs", n_a, edges_a)

    # Case B: one giant SCC.
    n_b = 5
    edges_b = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (1, 3),
        (3, 1),
    ]
    print_result("Case B: Single SCC", n_b, edges_b)


if __name__ == "__main__":
    main()
