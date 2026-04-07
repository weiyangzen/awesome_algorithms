"""可见性图算法（Visibility Graph）最小可运行示例。

运行方式：
    uv run python Algorithms/数学-计算几何-0211-可见性图算法/demo.py
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class Node:
    """可见性图中的节点。"""

    idx: int
    point: tuple[float, float]
    kind: str  # start / goal / vertex
    obstacle_id: int | None = None
    vertex_id: int | None = None


def as_point(p: Iterable[float]) -> np.ndarray:
    arr = np.asarray(tuple(p), dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"Point must be 2D, got shape={arr.shape}")
    return arr


def same_point(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> bool:
    return float(np.linalg.norm(a - b)) <= eps


def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """二维有向面积（叉积 z 分量）。"""

    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def on_segment(a: np.ndarray, p: np.ndarray, b: np.ndarray, eps: float = EPS) -> bool:
    """判断点 p 是否在线段 ab 上（含端点）。"""

    if abs(orient(a, p, b)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    """判断两线段是否相交（含端点接触与共线重叠）。"""

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    if (o1 > EPS and o2 < -EPS or o1 < -EPS and o2 > EPS) and (
        o3 > EPS and o4 < -EPS or o3 < -EPS and o4 > EPS
    ):
        return True

    if abs(o1) <= EPS and on_segment(p1, q1, p2):
        return True
    if abs(o2) <= EPS and on_segment(p1, q2, p2):
        return True
    if abs(o3) <= EPS and on_segment(q1, p1, q2):
        return True
    if abs(o4) <= EPS and on_segment(q1, p2, q2):
        return True

    return False


def point_in_polygon_strict(point: np.ndarray, polygon: np.ndarray) -> bool:
    """射线法判断点是否在多边形严格内部。边界点返回 False。"""

    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if on_segment(a, point, b):
            return False

    x, y = float(point[0]), float(point[1])
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 0.0) + x1
            if x < xinters:
                inside = not inside
    return inside


def are_visible(p: np.ndarray, q: np.ndarray, obstacles: list[np.ndarray]) -> bool:
    """判断线段 pq 是否可见（不穿越任意障碍物内部）。"""

    if same_point(p, q):
        return False

    # 1) 与障碍物边相交检查。
    for polygon in obstacles:
        n = len(polygon)
        for i in range(n):
            a = polygon[i]
            b = polygon[(i + 1) % n]

            # 完全同一条边（允许，可沿障碍边界移动）。
            same_edge = (same_point(p, a) and same_point(q, b)) or (
                same_point(p, b) and same_point(q, a)
            )
            if same_edge:
                continue

            if not segments_intersect(p, q, a, b):
                continue

            # 仅在共享端点处接触可放行。
            shared_endpoint = (
                same_point(p, a)
                or same_point(p, b)
                or same_point(q, a)
                or same_point(q, b)
            )
            if not shared_endpoint:
                return False

            # 共线重叠（不仅仅是单点接触）判为不可见，避免贴边穿过。
            if abs(orient(p, q, a)) <= EPS and abs(orient(p, q, b)) <= EPS:
                axis = 0 if abs(p[0] - q[0]) >= abs(p[1] - q[1]) else 1
                p_min, p_max = sorted([float(p[axis]), float(q[axis])])
                e_min, e_max = sorted([float(a[axis]), float(b[axis])])
                overlap = min(p_max, e_max) - max(p_min, e_min)
                if overlap > EPS:
                    return False

    # 2) 中点落在障碍物严格内部则不可见。
    mid = (p + q) * 0.5
    for polygon in obstacles:
        if point_in_polygon_strict(mid, polygon):
            return False

    return True


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def build_visibility_graph(
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: list[np.ndarray],
) -> tuple[list[Node], dict[int, list[tuple[int, float]]]]:
    """构建可见性图（无向带权图）。"""

    s = as_point(start)
    t = as_point(goal)

    for oid, polygon in enumerate(obstacles):
        if point_in_polygon_strict(s, polygon):
            raise ValueError(f"Start is inside obstacle {oid}")
        if point_in_polygon_strict(t, polygon):
            raise ValueError(f"Goal is inside obstacle {oid}")

    nodes: list[Node] = [
        Node(0, (float(s[0]), float(s[1])), "start"),
        Node(1, (float(t[0]), float(t[1])), "goal"),
    ]

    for oid, polygon in enumerate(obstacles):
        for vid, v in enumerate(polygon):
            nodes.append(
                Node(
                    idx=len(nodes),
                    point=(float(v[0]), float(v[1])),
                    kind="vertex",
                    obstacle_id=oid,
                    vertex_id=vid,
                )
            )

    adjacency: dict[int, list[tuple[int, float]]] = {node.idx: [] for node in nodes}
    points = [as_point(node.point) for node in nodes]

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if are_visible(points[i], points[j], obstacles):
                w = euclidean(points[i], points[j])
                adjacency[i].append((j, w))
                adjacency[j].append((i, w))

    return nodes, adjacency


def dijkstra(
    adjacency: dict[int, list[tuple[int, float]]], start_idx: int, goal_idx: int
) -> tuple[float, list[int]]:
    """Dijkstra 最短路。"""

    dist = {node: math.inf for node in adjacency}
    prev: dict[int, int | None] = {node: None for node in adjacency}
    dist[start_idx] = 0.0

    heap: list[tuple[float, int]] = [(0.0, start_idx)]
    visited: set[int] = set()

    while heap:
        cur_dist, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        if u == goal_idx:
            break

        for v, w in adjacency[u]:
            cand = cur_dist + w
            if cand < dist[v]:
                dist[v] = cand
                prev[v] = u
                heapq.heappush(heap, (cand, v))

    if math.isinf(dist[goal_idx]):
        return math.inf, []

    path = []
    cur: int | None = goal_idx
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return dist[goal_idx], path


def demo_scene() -> tuple[tuple[float, float], tuple[float, float], list[np.ndarray]]:
    """给出固定示例场景（无交互）。"""

    start = (0.0, 0.0)
    goal = (9.0, 6.0)

    # 障碍物按逆时针给出。
    obstacles = [
        np.array([(2.0, 1.0), (4.2, 1.0), (4.2, 3.2), (2.0, 3.2)], dtype=float),
        np.array([(5.0, 3.8), (6.8, 3.8), (6.0, 5.6)], dtype=float),
        np.array([(6.8, 1.0), (8.4, 1.2), (8.2, 2.8), (6.5, 2.6)], dtype=float),
    ]
    return start, goal, obstacles


def main() -> None:
    start, goal, obstacles = demo_scene()
    nodes, adjacency = build_visibility_graph(start, goal, obstacles)
    distance, path = dijkstra(adjacency, start_idx=0, goal_idx=1)

    edge_count = sum(len(v) for v in adjacency.values()) // 2
    print("=== Visibility Graph MVP ===")
    print(f"start={start}, goal={goal}")
    print(f"obstacles={len(obstacles)}, nodes={len(nodes)}, edges={edge_count}")

    if not path:
        print("No path found.")
        return

    print(f"shortest_distance={distance:.6f}")
    print("path_node_indices=", path)
    print("path_points=")
    for idx in path:
        print(f"  {idx}: {nodes[idx].point}")


if __name__ == "__main__":
    main()
