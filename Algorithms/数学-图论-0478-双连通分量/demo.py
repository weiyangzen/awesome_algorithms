"""双连通分量（点双，Biconnected Components）最小可运行 MVP.

实现范围：
1) 无向图 Tarjan 点双连通分量分解（边栈法）
2) 同步输出割点（Articulation Points）
3) 通过暴力删点法对拍割点，增强正确性可验证性

运行：
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np

Edge = Tuple[int, int]


@dataclass(frozen=True)
class Graph:
    """带 edge_id 的无向图表示，支持重边。"""

    n: int
    edges: List[Edge]
    adjacency: List[List[Tuple[int, int]]]  # adjacency[u] = [(v, edge_id), ...]

    @property
    def m(self) -> int:
        return len(self.edges)


@dataclass
class BiconnectedResult:
    """Tarjan 点双结果。"""

    bcc_vertices: List[List[int]]
    bcc_edges: List[List[int]]
    articulation_points: List[int]
    disc: np.ndarray
    low: np.ndarray


@dataclass(frozen=True)
class GraphCase:
    """测试图用例。"""

    name: str
    n: int
    edges: Sequence[Edge]
    expected_bcc_vertices: Sequence[Sequence[int]]
    expected_articulation_points: Sequence[int]


def build_undirected_graph(num_vertices: int, edges: Iterable[Edge]) -> Graph:
    """构建无向图并做基础校验。"""
    if not isinstance(num_vertices, int) or num_vertices < 0:
        raise ValueError(f"`num_vertices` must be a non-negative int, got {num_vertices!r}")

    edge_list = list(edges)
    if num_vertices == 0 and edge_list:
        raise ValueError("num_vertices=0 时不允许包含边")

    adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(num_vertices)]
    for edge_id, (u, v) in enumerate(edge_list):
        if not isinstance(u, int) or not isinstance(v, int):
            raise ValueError(f"edge #{edge_id} has non-int endpoint: {(u, v)!r}")
        if not (0 <= u < num_vertices and 0 <= v < num_vertices):
            raise ValueError(
                f"edge #{edge_id} endpoint out of range [0, {num_vertices - 1}]: {(u, v)!r}"
            )
        if u == v:
            raise ValueError("self-loop is not supported in this MVP")
        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))

    return Graph(n=num_vertices, edges=edge_list, adjacency=adjacency)


def _sort_component_order(
    bcc_vertices: List[List[int]], bcc_edges: List[List[int]]
) -> Tuple[List[List[int]], List[List[int]]]:
    order = sorted(
        range(len(bcc_vertices)),
        key=lambda i: (
            bcc_vertices[i][0] if bcc_vertices[i] else -1,
            len(bcc_vertices[i]),
            bcc_vertices[i],
            bcc_edges[i],
        ),
    )
    return [bcc_vertices[i] for i in order], [bcc_edges[i] for i in order]


def biconnected_components_tarjan(graph: Graph) -> BiconnectedResult:
    """Tarjan 点双连通分量分解（边栈法），时间复杂度 O(V+E)。"""
    n = graph.n
    disc = np.full(n, -1, dtype=int)
    low = np.full(n, -1, dtype=int)
    is_articulation = np.zeros(n, dtype=bool)

    timer = 0
    edge_stack: List[int] = []
    bcc_vertices: List[List[int]] = []
    bcc_edges: List[List[int]] = []

    def pop_component_until(stop_edge_id: int) -> None:
        comp_edges: List[int] = []
        comp_vertices: Set[int] = set()
        while edge_stack:
            eid = edge_stack.pop()
            comp_edges.append(eid)
            u, v = graph.edges[eid]
            comp_vertices.add(u)
            comp_vertices.add(v)
            if eid == stop_edge_id:
                break
        if comp_vertices:
            bcc_edges.append(sorted(comp_edges))
            bcc_vertices.append(sorted(comp_vertices))

    def dfs(u: int, parent_edge_id: int, parent_vertex: int) -> None:
        nonlocal timer
        disc[u] = timer
        low[u] = timer
        timer += 1

        child_count = 0
        for v, edge_id in graph.adjacency[u]:
            if edge_id == parent_edge_id:
                continue

            if disc[v] == -1:
                child_count += 1
                edge_stack.append(edge_id)
                dfs(v, edge_id, u)
                low[u] = min(low[u], int(low[v]))

                if low[v] >= disc[u]:
                    if parent_vertex != -1 or child_count > 1:
                        is_articulation[u] = True
                    pop_component_until(edge_id)
            elif disc[v] < disc[u]:
                # 只把指向祖先的返祖边入栈，避免无向边重复入栈。
                edge_stack.append(edge_id)
                low[u] = min(low[u], int(disc[v]))

    for root in range(n):
        if disc[root] != -1:
            continue
        dfs(root, parent_edge_id=-1, parent_vertex=-1)

        # 防御性收尾：理论上正常流程不会残留边栈。
        if edge_stack:
            residual_edges = sorted(edge_stack)
            residual_vertices: Set[int] = set()
            for eid in residual_edges:
                a, b = graph.edges[eid]
                residual_vertices.add(a)
                residual_vertices.add(b)
            bcc_edges.append(residual_edges)
            bcc_vertices.append(sorted(residual_vertices))
            edge_stack.clear()

        # 孤立点是一个平凡 block。
        if not graph.adjacency[root]:
            bcc_edges.append([])
            bcc_vertices.append([root])

    articulation_points = [i for i, flag in enumerate(is_articulation.tolist()) if flag]
    bcc_vertices, bcc_edges = _sort_component_order(bcc_vertices, bcc_edges)
    return BiconnectedResult(
        bcc_vertices=bcc_vertices,
        bcc_edges=bcc_edges,
        articulation_points=articulation_points,
        disc=disc,
        low=low,
    )


def count_connected_components(graph: Graph, removed_vertex: int | None = None) -> int:
    """统计删除某点后的连通分量数量（无向图）。"""
    if removed_vertex is not None and not (0 <= removed_vertex < graph.n):
        raise ValueError(f"`removed_vertex` out of range: {removed_vertex}")

    visited = np.zeros(graph.n, dtype=bool)
    components = 0

    for start in range(graph.n):
        if start == removed_vertex or visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True

        while stack:
            u = stack.pop()
            for v, _ in graph.adjacency[u]:
                if v == removed_vertex or visited[v]:
                    continue
                visited[v] = True
                stack.append(v)
    return components


def articulation_points_bruteforce(graph: Graph) -> List[int]:
    """暴力删点法求割点，作为 Tarjan 结果对拍基线。"""
    base_components = count_connected_components(graph)
    points: List[int] = []
    for x in range(graph.n):
        after = count_connected_components(graph, removed_vertex=x)
        if after > base_components:
            points.append(x)
    return points


def _canonicalize_vertex_components(components: Sequence[Sequence[int]]) -> List[List[int]]:
    canonical = [sorted(set(comp)) for comp in components]
    canonical.sort(key=lambda comp: (comp[0] if comp else -1, len(comp), comp))
    return canonical


def _format_vertex_components(components: Sequence[Sequence[int]]) -> str:
    if not components:
        return "[]"
    parts = ["{" + ",".join(str(x) for x in comp) + "}" for comp in components]
    return "[" + ", ".join(parts) + "]"


def _validate_edge_partition(graph: Graph, bcc_edges: Sequence[Sequence[int]]) -> None:
    all_ids: List[int] = []
    for comp in bcc_edges:
        all_ids.extend(comp)
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError(f"edge ids are duplicated among components: {all_ids}")
    if sorted(all_ids) != list(range(graph.m)):
        raise AssertionError(
            f"edge ids are not fully covered by BCC decomposition: got {sorted(all_ids)}"
        )


def run_case(case: GraphCase) -> None:
    graph = build_undirected_graph(case.n, case.edges)
    result = biconnected_components_tarjan(graph)
    brute_ap = articulation_points_bruteforce(graph)

    expected_bcc = _canonicalize_vertex_components(case.expected_bcc_vertices)
    actual_bcc = _canonicalize_vertex_components(result.bcc_vertices)
    expected_ap = sorted(case.expected_articulation_points)
    actual_ap = sorted(result.articulation_points)

    _validate_edge_partition(graph, result.bcc_edges)

    print(f"\n=== {case.name} ===")
    print(f"n={graph.n}, m={graph.m}")
    print(f"BCC vertices (Tarjan): {_format_vertex_components(actual_bcc)}")
    print(f"BCC edge ids         : {result.bcc_edges}")
    print(f"APs (Tarjan)         : {actual_ap}")
    print(f"APs (Bruteforce)     : {brute_ap}")
    print(f"APs (Expected)       : {expected_ap}")
    print(f"disc                 : {result.disc.tolist()}")
    print(f"low                  : {result.low.tolist()}")

    if actual_bcc != expected_bcc:
        raise AssertionError(
            f"[{case.name}] BCC mismatch: Tarjan={actual_bcc}, Expected={expected_bcc}"
        )
    if actual_ap != brute_ap:
        raise AssertionError(f"[{case.name}] AP mismatch: Tarjan={actual_ap}, Bruteforce={brute_ap}")
    if actual_ap != expected_ap:
        raise AssertionError(f"[{case.name}] AP mismatch: Tarjan={actual_ap}, Expected={expected_ap}")


def main() -> None:
    cases = [
        GraphCase(
            name="two triangles sharing one articulation",
            n=5,
            edges=[(0, 1), (1, 2), (2, 0), (1, 3), (3, 4), (4, 1)],
            expected_bcc_vertices=[[0, 1, 2], [1, 3, 4]],
            expected_articulation_points=[1],
        ),
        GraphCase(
            name="chain graph",
            n=4,
            edges=[(0, 1), (1, 2), (2, 3)],
            expected_bcc_vertices=[[0, 1], [1, 2], [2, 3]],
            expected_articulation_points=[1, 2],
        ),
        GraphCase(
            name="disconnected with isolated vertex",
            n=7,
            edges=[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5)],
            expected_bcc_vertices=[[0, 1, 2], [3, 4], [4, 5], [6]],
            expected_articulation_points=[4],
        ),
        GraphCase(
            name="parallel-edge multigraph",
            n=2,
            edges=[(0, 1), (0, 1)],
            expected_bcc_vertices=[[0, 1]],
            expected_articulation_points=[],
        ),
        GraphCase(
            name="single isolated vertex",
            n=1,
            edges=[],
            expected_bcc_vertices=[[0]],
            expected_articulation_points=[],
        ),
    ]

    for case in cases:
        run_case(case)

    print("\nAll biconnected-component checks passed.")


if __name__ == "__main__":
    main()
