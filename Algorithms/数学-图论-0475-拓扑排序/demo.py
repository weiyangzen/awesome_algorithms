"""Minimal runnable MVP for Topological Sorting (MATH-0475).

This demo implements Kahn's algorithm from scratch for DAG topological
sorting and cycle detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, Iterable, List, Sequence, Tuple


Node = str
Edge = Tuple[Node, Node]


@dataclass
class TopoResult:
    """Result of a topological sort attempt."""

    order: List[Node]
    has_cycle: bool
    processed_count: int
    total_nodes: int


def build_graph(nodes: Sequence[Node], edges: Iterable[Edge]) -> Tuple[Dict[Node, List[Node]], Dict[Node, int]]:
    """Build adjacency list and indegree table.

    Args:
        nodes: known node list.
        edges: directed edges (u, v), meaning u must come before v.

    Returns:
        adjacency: node -> outgoing neighbors.
        indegree: node -> incoming edge count.
    """
    adjacency: Dict[Node, List[Node]] = {n: [] for n in nodes}
    indegree: Dict[Node, int] = {n: 0 for n in nodes}

    for u, v in edges:
        if u not in adjacency:
            adjacency[u] = []
            indegree[u] = 0
        if v not in adjacency:
            adjacency[v] = []
            indegree[v] = 0
        adjacency[u].append(v)
        indegree[v] += 1

    return adjacency, indegree


def kahn_topological_sort(nodes: Sequence[Node], edges: Iterable[Edge]) -> TopoResult:
    """Topological sort with Kahn's indegree algorithm.

    A min-heap is used for deterministic tie-breaking.
    """
    adjacency, indegree = build_graph(nodes, edges)

    heap: List[Node] = []
    for n in adjacency:
        if indegree[n] == 0:
            heappush(heap, n)

    order: List[Node] = []
    processed = 0

    while heap:
        u = heappop(heap)
        order.append(u)
        processed += 1

        for v in adjacency[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                heappush(heap, v)

    total = len(adjacency)
    has_cycle = processed != total
    return TopoResult(order=order, has_cycle=has_cycle, processed_count=processed, total_nodes=total)


def validate_topological_order(order: Sequence[Node], nodes: Sequence[Node], edges: Iterable[Edge]) -> bool:
    """Check whether `order` is a valid topological order for given graph."""
    if len(order) != len(set(nodes)):
        return False

    position: Dict[Node, int] = {node: idx for idx, node in enumerate(order)}
    if len(position) != len(order):
        return False

    for u, v in edges:
        if u not in position or v not in position:
            return False
        if position[u] >= position[v]:
            return False
    return True


def format_order(order: Sequence[Node]) -> str:
    """Pretty formatter for printing an order."""
    return " -> ".join(order)


def main() -> None:
    print("Topological Sorting MVP (MATH-0475)")
    print("=" * 68)

    dag_nodes = [
        "需求分析",
        "数据库建模",
        "接口设计",
        "后端实现",
        "前端实现",
        "联调测试",
        "上线发布",
    ]
    dag_edges: List[Edge] = [
        ("需求分析", "数据库建模"),
        ("需求分析", "接口设计"),
        ("数据库建模", "后端实现"),
        ("接口设计", "后端实现"),
        ("接口设计", "前端实现"),
        ("后端实现", "联调测试"),
        ("前端实现", "联调测试"),
        ("联调测试", "上线发布"),
    ]

    dag_result = kahn_topological_sort(dag_nodes, dag_edges)
    print("DAG 排序结果:")
    print(f"  节点数: {dag_result.total_nodes}, 已处理: {dag_result.processed_count}")
    print(f"  是否有环: {dag_result.has_cycle}")
    print(f"  顺序: {format_order(dag_result.order)}")

    dag_valid = validate_topological_order(dag_result.order, dag_nodes, dag_edges)
    print(f"  顺序合法性校验: {dag_valid}")

    if dag_result.has_cycle:
        raise RuntimeError("DAG 示例不应检测到环")
    if not dag_valid:
        raise RuntimeError("DAG 示例拓扑序校验失败")

    print("-" * 68)

    cyclic_nodes = ["A", "B", "C", "D"]
    cyclic_edges: List[Edge] = [
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
        ("D", "A"),
    ]

    cyclic_result = kahn_topological_sort(cyclic_nodes, cyclic_edges)
    print("含环图检测结果:")
    print(f"  节点数: {cyclic_result.total_nodes}, 已处理: {cyclic_result.processed_count}")
    print(f"  是否有环: {cyclic_result.has_cycle}")
    print(f"  可输出的前缀顺序: {format_order(cyclic_result.order) if cyclic_result.order else '(空)'}")

    if not cyclic_result.has_cycle:
        raise RuntimeError("含环图应被检测为有环")

    print("=" * 68)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
