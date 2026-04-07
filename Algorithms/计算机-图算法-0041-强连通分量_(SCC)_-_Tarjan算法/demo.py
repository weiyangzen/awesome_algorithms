"""Minimal runnable MVP for strongly connected components (Tarjan, directed graph)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

Edge = tuple[int, int]


@dataclass
class TarjanResult:
    sccs: list[list[int]]
    component_id_of_vertex: list[int]
    discovery_time: list[int]
    low_link: list[int]
    dfs_order: list[int]
    condensation_edges: list[Edge]
    condensation_topo_order: list[int]


def build_directed_graph(
    n: int,
    edges: Iterable[Edge],
) -> tuple[list[list[int]], list[Edge]]:
    """Build adjacency list for a directed graph with input validation."""
    if n < 0:
        raise ValueError("n must be non-negative")

    adj: list[list[int]] = [[] for _ in range(n)]
    normalized_edges: list[Edge] = []

    for raw_u, raw_v in edges:
        u = int(raw_u)
        v = int(raw_v)
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) out of range for n={n}")

        adj[u].append(v)
        normalized_edges.append((u, v))

    return adj, normalized_edges


def topological_sort_dag(num_vertices: int, edges: Iterable[Edge]) -> list[int]:
    """Kahn topological sort for SCC condensation DAG."""
    indegree = [0] * num_vertices
    dag_adj: list[list[int]] = [[] for _ in range(num_vertices)]

    for u, v in edges:
        dag_adj[u].append(v)
        indegree[v] += 1

    queue = [v for v in range(num_vertices) if indegree[v] == 0]
    order: list[int] = []
    head = 0

    while head < len(queue):
        u = queue[head]
        head += 1
        order.append(u)

        for v in dag_adj[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    if len(order) != num_vertices:
        raise RuntimeError("condensation graph is expected to be a DAG")

    return order


def tarjan_scc(n: int, edges: Iterable[Edge]) -> TarjanResult:
    """Compute SCCs with Tarjan's algorithm (single DFS pass + low-link)."""
    adj, normalized_edges = build_directed_graph(n, edges)

    discovery_time = [-1] * n
    low_link = [-1] * n
    on_stack = [False] * n
    stack: list[int] = []
    dfs_order: list[int] = []
    raw_components: list[list[int]] = []

    time = 0

    def dfs(u: int) -> None:
        nonlocal time
        discovery_time[u] = time
        low_link[u] = time
        time += 1

        dfs_order.append(u)
        stack.append(u)
        on_stack[u] = True

        for v in adj[u]:
            if discovery_time[v] == -1:
                dfs(v)
                low_link[u] = min(low_link[u], low_link[v])
            elif on_stack[v]:
                low_link[u] = min(low_link[u], discovery_time[v])

        if low_link[u] == discovery_time[u]:
            component: list[int] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == u:
                    break
            raw_components.append(component)

    for start in range(n):
        if discovery_time[start] == -1:
            dfs(start)

    component_id_raw = [-1] * n
    for old_id, component in enumerate(raw_components):
        for vertex in component:
            component_id_raw[vertex] = old_id

    ordered_components_with_old_id = sorted(
        ((sorted(component), old_id) for old_id, component in enumerate(raw_components)),
        key=lambda item: (item[0][0], len(item[0]), item[0]),
    )

    old_to_new = {
        old_id: new_id for new_id, (_component, old_id) in enumerate(ordered_components_with_old_id)
    }

    sccs = [component for component, _old_id in ordered_components_with_old_id]
    component_id_of_vertex = [old_to_new[component_id_raw[v]] for v in range(n)]

    condensation_edge_set: set[Edge] = set()
    for u, v in normalized_edges:
        cu = component_id_of_vertex[u]
        cv = component_id_of_vertex[v]
        if cu != cv:
            condensation_edge_set.add((cu, cv))

    condensation_edges = sorted(condensation_edge_set)
    condensation_topo_order = topological_sort_dag(len(sccs), condensation_edges)

    return TarjanResult(
        sccs=sccs,
        component_id_of_vertex=component_id_of_vertex,
        discovery_time=discovery_time,
        low_link=low_link,
        dfs_order=dfs_order,
        condensation_edges=condensation_edges,
        condensation_topo_order=condensation_topo_order,
    )


def assert_case(
    result: TarjanResult,
    expected_sccs: set[frozenset[int]],
    expected_condensation_edges: set[Edge],
) -> None:
    got_sccs = {frozenset(component) for component in result.sccs}
    if got_sccs != expected_sccs:
        raise AssertionError(
            f"SCC mismatch: got={sorted(map(sorted, got_sccs))}, "
            f"expected={sorted(map(sorted, expected_sccs))}"
        )

    got_edges = set(result.condensation_edges)
    if got_edges != expected_condensation_edges:
        raise AssertionError(
            f"condensation edges mismatch: got={sorted(got_edges)}, "
            f"expected={sorted(expected_condensation_edges)}"
        )

    all_vertices = set(v for comp in result.sccs for v in comp)
    expected_vertices = set(range(len(result.component_id_of_vertex)))
    if all_vertices != expected_vertices:
        raise AssertionError(
            f"vertex coverage mismatch: got={sorted(all_vertices)}, "
            f"expected={sorted(expected_vertices)}"
        )


def run_demo_cases() -> None:
    cases = [
        {
            "name": "Case-1: SCC chain in a directed graph",
            "n": 8,
            "edges": np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 0],
                    [2, 3],
                    [3, 4],
                    [4, 5],
                    [5, 3],
                    [5, 6],
                    [6, 7],
                    [7, 6],
                ],
                dtype=np.int64,
            ),
            "expected_sccs": {
                frozenset({0, 1, 2}),
                frozenset({3, 4, 5}),
                frozenset({6, 7}),
            },
            "expected_condensation_edges": {
                (0, 1),
                (1, 2),
            },
        },
        {
            "name": "Case-2: disconnected graph with isolated vertex",
            "n": 7,
            "edges": np.array(
                [
                    [0, 1],
                    [1, 0],
                    [1, 2],
                    [3, 4],
                    [4, 5],
                    [5, 3],
                ],
                dtype=np.int64,
            ),
            "expected_sccs": {
                frozenset({0, 1}),
                frozenset({2}),
                frozenset({3, 4, 5}),
                frozenset({6}),
            },
            "expected_condensation_edges": {
                (0, 1),
            },
        },
        {
            "name": "Case-3: DAG (every vertex is its own SCC)",
            "n": 5,
            "edges": np.array(
                [
                    [0, 1],
                    [0, 2],
                    [1, 3],
                    [2, 3],
                    [3, 4],
                ],
                dtype=np.int64,
            ),
            "expected_sccs": {
                frozenset({0}),
                frozenset({1}),
                frozenset({2}),
                frozenset({3}),
                frozenset({4}),
            },
            "expected_condensation_edges": {
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 3),
                (3, 4),
            },
        },
    ]

    for idx, case in enumerate(cases, start=1):
        edges_list = [tuple(map(int, edge)) for edge in case["edges"].tolist()]
        result = tarjan_scc(case["n"], edges_list)

        print(f"[{idx}] {case['name']}")
        print(f"  SCCs: {result.sccs}")
        print(f"  component_id_of_vertex: {result.component_id_of_vertex}")
        print(f"  discovery_time: {result.discovery_time}")
        print(f"  low_link: {result.low_link}")
        print(f"  dfs_order: {result.dfs_order}")
        print(f"  condensation_edges: {result.condensation_edges}")
        print(f"  condensation_topo_order: {result.condensation_topo_order}")

        assert_case(
            result=result,
            expected_sccs=case["expected_sccs"],
            expected_condensation_edges=case["expected_condensation_edges"],
        )
        print("  checks: passed\n")


def main() -> None:
    run_demo_cases()
    print("All demo cases passed.")


if __name__ == "__main__":
    main()
