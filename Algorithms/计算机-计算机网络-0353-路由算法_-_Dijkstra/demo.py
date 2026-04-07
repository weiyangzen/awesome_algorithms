"""Dijkstra 路由算法最小可运行示例（MVP）。

目标：
- 在一个小型链路状态拓扑上计算从源路由器到所有节点的最短路径。
- 输出距离表与下一跳路由表，展示路由算法落地结果。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import inf
from typing import Dict, List, Optional, Tuple

Graph = Dict[str, List[Tuple[str, float]]]


@dataclass
class DijkstraResult:
    source: str
    distances: Dict[str, float]
    predecessors: Dict[str, Optional[str]]
    heap_pop_count: int
    relax_count: int


@dataclass
class RouteEntry:
    target: str
    next_hop: str
    total_cost: float
    path: List[str]


def build_sample_topology() -> Graph:
    """构造一个无向、非负权重的示例网络拓扑。"""
    return {
        "R1": [("R2", 2), ("R3", 5)],
        "R2": [("R1", 2), ("R3", 1), ("R4", 2)],
        "R3": [("R1", 5), ("R2", 1), ("R4", 3), ("R5", 1)],
        "R4": [("R2", 2), ("R3", 3), ("R6", 4)],
        "R5": [("R3", 1), ("R6", 2), ("R7", 5)],
        "R6": [("R4", 4), ("R5", 2), ("R7", 1)],
        "R7": [("R5", 5), ("R6", 1), ("R8", 2)],
        "R8": [("R7", 2)],
    }


def validate_graph(graph: Graph) -> None:
    if not graph:
        raise ValueError("graph must not be empty")

    for node, neighbors in graph.items():
        if not isinstance(node, str) or not node:
            raise ValueError("node id must be non-empty str")
        for neighbor, weight in neighbors:
            if weight < 0:
                raise ValueError("Dijkstra requires non-negative edge weights")
            if not isinstance(neighbor, str) or not neighbor:
                raise ValueError("neighbor id must be non-empty str")


def dijkstra(graph: Graph, source: str) -> DijkstraResult:
    """使用最小堆实现 Dijkstra 单源最短路径。"""
    validate_graph(graph)
    if source not in graph:
        raise ValueError(f"source {source!r} is not present in graph")

    distances: Dict[str, float] = {node: inf for node in graph}
    predecessors: Dict[str, Optional[str]] = {node: None for node in graph}
    distances[source] = 0.0

    priority_queue: List[Tuple[float, str]] = [(0.0, source)]
    visited: set[str] = set()

    heap_pop_count = 0
    relax_count = 0

    while priority_queue:
        current_distance, node = heapq.heappop(priority_queue)
        heap_pop_count += 1

        if node in visited:
            continue
        if current_distance > distances[node]:
            continue

        visited.add(node)

        for neighbor, weight in graph[node]:
            candidate = current_distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                predecessors[neighbor] = node
                heapq.heappush(priority_queue, (candidate, neighbor))
                relax_count += 1

    return DijkstraResult(
        source=source,
        distances=distances,
        predecessors=predecessors,
        heap_pop_count=heap_pop_count,
        relax_count=relax_count,
    )


def reconstruct_path(
    predecessors: Dict[str, Optional[str]],
    source: str,
    target: str,
) -> List[str]:
    """从前驱映射回溯 source -> target 路径。不可达时返回空列表。"""
    if source == target:
        return [source]

    path: List[str] = []
    cursor: Optional[str] = target

    while cursor is not None:
        path.append(cursor)
        if cursor == source:
            path.reverse()
            return path
        cursor = predecessors[cursor]

    return []


def build_routing_table(result: DijkstraResult) -> List[RouteEntry]:
    entries: List[RouteEntry] = []

    for target in sorted(result.distances):
        if target == result.source:
            continue

        distance = result.distances[target]
        if distance == inf:
            continue

        path = reconstruct_path(result.predecessors, result.source, target)
        if len(path) < 2:
            continue

        entries.append(
            RouteEntry(
                target=target,
                next_hop=path[1],
                total_cost=distance,
                path=path,
            )
        )

    return entries


def print_distance_table(result: DijkstraResult) -> None:
    print("=== Distance Table ===")
    print(f"source: {result.source}")
    print("node  distance")
    for node in sorted(result.distances):
        d = result.distances[node]
        text = "INF" if d == inf else f"{d:.1f}"
        print(f"{node:>4}  {text:>8}")


def print_routing_table(entries: List[RouteEntry]) -> None:
    print("\n=== Routing Table (Shortest-Path Based) ===")
    print("target  next_hop  cost  path")
    for entry in entries:
        path_str = " -> ".join(entry.path)
        print(
            f"{entry.target:>6}  {entry.next_hop:>8}  "
            f"{entry.total_cost:>4.1f}  {path_str}"
        )


def main() -> None:
    graph = build_sample_topology()
    source = "R1"

    result = dijkstra(graph, source)
    routes = build_routing_table(result)

    print_distance_table(result)
    print_routing_table(routes)

    print("\n=== Debug Stats ===")
    print(f"heap_pop_count: {result.heap_pop_count}")
    print(f"relax_count: {result.relax_count}")
    print(f"route_count: {len(routes)}")


if __name__ == "__main__":
    main()
