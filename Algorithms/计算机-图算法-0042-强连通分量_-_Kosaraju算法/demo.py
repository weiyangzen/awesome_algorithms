"""Minimal runnable MVP for strongly connected components (Kosaraju, directed graph)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

Edge = tuple[int, int]


@dataclass
class KosarajuResult:
    sccs: list[list[int]]
    component_id_of_vertex: list[int]
    finish_order: list[int]
    second_pass_order: list[int]
    condensation_edges: list[Edge]
    condensation_topo_order: list[int]


def build_directed_graph(
    n: int,
    edges: Iterable[Edge],
) -> tuple[list[list[int]], list[list[int]], list[Edge]]:
    """Build adjacency and reverse-adjacency lists for a directed graph."""
    if n < 0:
        raise ValueError("n must be non-negative")

    adj: list[list[int]] = [[] for _ in range(n)]
    rev_adj: list[list[int]] = [[] for _ in range(n)]
    normalized_edges: list[Edge] = []

    for raw_u, raw_v in edges:
        u = int(raw_u)
        v = int(raw_v)
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) out of range for n={n}")

        adj[u].append(v)
        rev_adj[v].append(u)
        normalized_edges.append((u, v))

    return adj, rev_adj, normalized_edges


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


def kosaraju_scc(n: int, edges: Iterable[Edge]) -> KosarajuResult:
    """Compute SCCs with Kosaraju's algorithm (two DFS passes)."""
    adj, rev_adj, normalized_edges = build_directed_graph(n, edges)

    seen = [False] * n
    finish_order: list[int] = []

    def dfs1(u: int) -> None:
        seen[u] = True
        for v in adj[u]:
            if not seen[v]:
                dfs1(v)
        finish_order.append(u)

    for start in range(n):
        if not seen[start]:
            dfs1(start)

    component_id_raw = [-1] * n
    raw_components: list[list[int]] = []

    def dfs2(u: int, cid: int) -> None:
        component_id_raw[u] = cid
        raw_components[cid].append(u)
        for v in rev_adj[u]:
            if component_id_raw[v] == -1:
                dfs2(v, cid)

    second_pass_order = list(reversed(finish_order))
    for u in second_pass_order:
        if component_id_raw[u] == -1:
            cid = len(raw_components)
            raw_components.append([])
            dfs2(u, cid)

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

    return KosarajuResult(
        sccs=sccs,
        component_id_of_vertex=component_id_of_vertex,
        finish_order=finish_order,
        second_pass_order=second_pass_order,
        condensation_edges=condensation_edges,
        condensation_topo_order=condensation_topo_order,
    )


def assert_case(
    result: KosarajuResult,
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
        result = kosaraju_scc(case["n"], edges_list)

        print(f"[{idx}] {case['name']}")
        print(f"  SCCs: {result.sccs}")
        print(f"  component_id_of_vertex: {result.component_id_of_vertex}")
        print(f"  finish_order: {result.finish_order}")
        print(f"  second_pass_order: {result.second_pass_order}")
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
