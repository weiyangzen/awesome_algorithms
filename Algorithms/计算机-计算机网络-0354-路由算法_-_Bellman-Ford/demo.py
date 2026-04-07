"""Bellman-Ford MVP for routing-style shortest path demonstration."""

from __future__ import annotations

from collections import deque
from math import inf
from typing import List, Optional, Sequence, Set, Tuple

Edge = Tuple[int, int, float]


def bellman_ford(
    n: int,
    edges: Sequence[Edge],
    source: int,
) -> Tuple[List[float], List[Optional[int]], Set[int]]:
    """Return (dist, predecessor, affected_by_negative_cycle)."""
    dist: List[float] = [inf] * n
    pred: List[Optional[int]] = [None] * n
    dist[source] = 0.0

    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] == inf:
                continue
            candidate = dist[u] + w
            if candidate < dist[v]:
                dist[v] = candidate
                pred[v] = u
                updated = True
        if not updated:
            break

    adjacency: List[List[int]] = [[] for _ in range(n)]
    for u, v, _ in edges:
        adjacency[u].append(v)

    affected: Set[int] = set()
    queue: deque[int] = deque()

    for u, v, w in edges:
        if dist[u] == inf:
            continue
        if dist[u] + w < dist[v]:
            if v not in affected:
                affected.add(v)
                queue.append(v)

    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if nxt not in affected:
                affected.add(nxt)
                queue.append(nxt)

    return dist, pred, affected


def reconstruct_path(
    pred: Sequence[Optional[int]],
    source: int,
    target: int,
) -> Optional[List[int]]:
    if source == target:
        return [source]

    path: List[int] = []
    seen: Set[int] = set()
    cur: Optional[int] = target
    while cur is not None and cur not in seen:
        seen.add(cur)
        path.append(cur)
        if cur == source:
            path.reverse()
            return path
        cur = pred[cur]
    return None


def path_to_string(path: Optional[List[int]], names: Sequence[str]) -> str:
    if path is None:
        return "UNREACHABLE"
    return " -> ".join(names[idx] for idx in path)


def print_result_table(
    names: Sequence[str],
    source: int,
    dist: Sequence[float],
    pred: Sequence[Optional[int]],
    affected: Set[int],
) -> None:
    print(f"源点: {names[source]}")
    print(f"{'节点':<8}{'距离':<14}{'路径'}")
    print("-" * 48)
    for i, name in enumerate(names):
        if i in affected:
            print(f"{name:<8}{'UNDEFINED':<14}{'UNDEFINED (negative cycle)'}")
            continue
        if dist[i] == inf:
            print(f"{name:<8}{'INF':<14}{'UNREACHABLE'}")
            continue
        path = reconstruct_path(pred, source, i)
        print(f"{name:<8}{dist[i]:<14.2f}{path_to_string(path, names)}")


def run_demo_without_negative_cycle() -> None:
    print("=" * 72)
    print("Demo A: 图中有负权边，但不存在负权环")
    print("=" * 72)

    names = ["A", "B", "C", "D", "E", "F"]
    edges: List[Edge] = [
        (0, 1, 4),
        (0, 2, 2),
        (1, 2, -1),
        (1, 3, 2),
        (2, 3, 3),
        (2, 4, 2),
        (4, 3, -2),
        (3, 5, 2),
        (4, 5, 4),
    ]
    source = 0

    dist, pred, affected = bellman_ford(len(names), edges, source)
    if affected:
        print("存在可达负权环，受影响节点:", ", ".join(names[i] for i in sorted(affected)))
    else:
        print("不存在可达负权环。")
    print_result_table(names, source, dist, pred, affected)
    print()


def run_demo_with_negative_cycle() -> None:
    print("=" * 72)
    print("Demo B: 图中存在可达负权环")
    print("=" * 72)

    names = ["S", "T", "X", "Y"]
    edges: List[Edge] = [
        (0, 1, 1),
        (1, 2, 1),
        (2, 1, -3),
        (2, 3, 2),
    ]
    source = 0

    dist, pred, affected = bellman_ford(len(names), edges, source)
    if affected:
        print("存在可达负权环，受影响节点:", ", ".join(names[i] for i in sorted(affected)))
    else:
        print("不存在可达负权环。")
    print_result_table(names, source, dist, pred, affected)
    print()


def main() -> None:
    run_demo_without_negative_cycle()
    run_demo_with_negative_cycle()


if __name__ == "__main__":
    main()
