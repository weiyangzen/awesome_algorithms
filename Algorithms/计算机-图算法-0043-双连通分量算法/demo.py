"""Minimal runnable MVP for biconnected components (Tarjan, undirected graph)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

Edge = tuple[int, int]


@dataclass
class BiconnectedResult:
    biconnected_components: list[list[int]]
    biconnected_component_edges: list[list[Edge]]
    articulation_points: list[int]
    bridges: list[Edge]
    tin: list[int]
    low: list[int]


def canonical_edge(u: int, v: int) -> Edge:
    return (u, v) if u <= v else (v, u)


def build_undirected_graph(
    n: int,
    edges: Iterable[Edge],
) -> tuple[list[list[tuple[int, int]]], list[Edge]]:
    if n < 0:
        raise ValueError("n must be non-negative")

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    normalized_edges: list[Edge] = []

    for edge_id, (raw_u, raw_v) in enumerate(edges):
        u = int(raw_u)
        v = int(raw_v)
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) out of range for n={n}")
        if u == v:
            raise ValueError("self-loop is not supported in this MVP")

        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))
        normalized_edges.append((u, v))

    return adjacency, normalized_edges


def analyze_biconnected_components(n: int, edges: Iterable[Edge]) -> BiconnectedResult:
    adjacency, _ = build_undirected_graph(n, edges)

    tin = [-1] * n
    low = [-1] * n
    timer = 0

    edge_stack: list[tuple[int, int, int]] = []
    articulation_points: set[int] = set()
    bridges: list[Edge] = []
    components_vertices: list[set[int]] = []
    components_edges: list[list[Edge]] = []

    def pop_component(stop_edge_id: int) -> None:
        comp_vertices: set[int] = set()
        comp_edges: list[Edge] = []

        while edge_stack:
            a, b, popped_id = edge_stack.pop()
            comp_vertices.add(a)
            comp_vertices.add(b)
            comp_edges.append(canonical_edge(a, b))
            if popped_id == stop_edge_id:
                break

        if comp_vertices:
            components_vertices.append(comp_vertices)
            components_edges.append(sorted(comp_edges))

    def dfs(u: int, parent_edge_id: int) -> None:
        nonlocal timer

        tin[u] = timer
        low[u] = timer
        timer += 1

        child_count = 0

        for v, edge_id in adjacency[u]:
            if edge_id == parent_edge_id:
                continue

            if tin[v] == -1:
                edge_stack.append((u, v, edge_id))
                child_count += 1
                dfs(v, edge_id)
                low[u] = min(low[u], low[v])

                if low[v] >= tin[u]:
                    if parent_edge_id != -1 or child_count > 1:
                        articulation_points.add(u)
                    pop_component(edge_id)

                if low[v] > tin[u]:
                    bridges.append(canonical_edge(u, v))

            elif tin[v] < tin[u]:
                edge_stack.append((u, v, edge_id))
                low[u] = min(low[u], tin[v])

    for start in range(n):
        if tin[start] != -1:
            continue

        if not adjacency[start]:
            tin[start] = timer
            low[start] = timer
            timer += 1
            components_vertices.append({start})
            components_edges.append([])
            continue

        dfs(start, -1)

        if edge_stack:
            # Safety guard: flush any remaining edges from the current DFS tree.
            pop_component(edge_stack[-1][2])

    biconnected_components = sorted(
        (sorted(component) for component in components_vertices),
        key=lambda comp: (len(comp), comp),
    )

    biconnected_component_edges = sorted(
        (sorted(component_edges) for component_edges in components_edges),
        key=lambda comp_edges: (len(comp_edges), comp_edges),
    )

    return BiconnectedResult(
        biconnected_components=biconnected_components,
        biconnected_component_edges=biconnected_component_edges,
        articulation_points=sorted(articulation_points),
        bridges=sorted(bridges),
        tin=tin,
        low=low,
    )


def assert_case(
    result: BiconnectedResult,
    expected_articulation_points: set[int],
    expected_bridges: set[Edge],
    expected_components: set[frozenset[int]],
) -> None:
    got_art = set(result.articulation_points)
    if got_art != expected_articulation_points:
        raise AssertionError(
            f"articulation points mismatch: got={sorted(got_art)}, "
            f"expected={sorted(expected_articulation_points)}"
        )

    got_bridges = {canonical_edge(u, v) for (u, v) in result.bridges}
    expected_bridge_norm = {canonical_edge(u, v) for (u, v) in expected_bridges}
    if got_bridges != expected_bridge_norm:
        raise AssertionError(
            f"bridges mismatch: got={sorted(got_bridges)}, "
            f"expected={sorted(expected_bridge_norm)}"
        )

    got_components = {frozenset(component) for component in result.biconnected_components}
    if got_components != expected_components:
        raise AssertionError(
            f"biconnected components mismatch: got={sorted(map(sorted, got_components))}, "
            f"expected={sorted(map(sorted, expected_components))}"
        )


def run_demo_cases() -> None:
    cases = [
        {
            "name": "Case-1: mixed graph with articulation points and bridges",
            "n": 8,
            "edges": np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 0],
                    [1, 3],
                    [3, 4],
                    [4, 5],
                    [5, 3],
                    [3, 6],
                    [6, 7],
                ],
                dtype=np.int64,
            ),
            "expected_articulation_points": {1, 3, 6},
            "expected_bridges": {(1, 3), (3, 6), (6, 7)},
            "expected_components": {
                frozenset({0, 1, 2}),
                frozenset({1, 3}),
                frozenset({3, 4, 5}),
                frozenset({3, 6}),
                frozenset({6, 7}),
            },
        },
        {
            "name": "Case-2: disconnected graph with one isolated vertex",
            "n": 6,
            "edges": np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 0],
                    [3, 4],
                ],
                dtype=np.int64,
            ),
            "expected_articulation_points": set(),
            "expected_bridges": {(3, 4)},
            "expected_components": {
                frozenset({0, 1, 2}),
                frozenset({3, 4}),
                frozenset({5}),
            },
        },
    ]

    for idx, case in enumerate(cases, start=1):
        edges_list = [tuple(map(int, edge)) for edge in case["edges"].tolist()]
        result = analyze_biconnected_components(case["n"], edges_list)

        print(f"[{idx}] {case['name']}")
        print(f"  articulation points: {result.articulation_points}")
        print(f"  bridges: {result.bridges}")
        print(f"  biconnected components (vertices): {result.biconnected_components}")
        print(f"  biconnected components (edges): {result.biconnected_component_edges}")
        print(f"  tin: {result.tin}")
        print(f"  low: {result.low}")

        assert_case(
            result,
            expected_articulation_points=case["expected_articulation_points"],
            expected_bridges=case["expected_bridges"],
            expected_components=case["expected_components"],
        )
        print("  checks: passed\n")



def main() -> None:
    run_demo_cases()
    print("All demo cases passed.")


if __name__ == "__main__":
    main()
