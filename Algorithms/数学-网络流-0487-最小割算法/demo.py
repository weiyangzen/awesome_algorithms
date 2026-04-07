"""最小割算法（基于 Edmonds-Karp 最大流）的最小可运行示例。

运行方式：
    uv run python demo.py

脚本内置多个小图样例，自动验证：
1) max_flow == residual_min_cut
2) max_flow == brute_force_min_cut
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


Edge = Tuple[int, int, int]


@dataclass
class MinCutResult:
    """单个测试样例的结果。"""

    max_flow: int
    min_cut_capacity: int
    s_side: List[int]
    t_side: List[int]
    cut_edges: List[Edge]


def build_capacity_matrix(n: int, edges: Sequence[Edge]) -> List[List[int]]:
    """把边列表转成 n x n 容量矩阵，支持重边容量累加。"""
    if n <= 1:
        raise ValueError("n must be >= 2")

    capacity = [[0] * n for _ in range(n)]
    for u, v, c in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"invalid edge ({u}, {v}, {c})")
        if c < 0:
            raise ValueError("capacity must be non-negative")
        if u == v:
            continue
        capacity[u][v] += c
    return capacity


def bfs_augmenting_path(
    residual: Sequence[Sequence[int]], source: int, sink: int
) -> Tuple[int, List[int]]:
    """在残量网络中用 BFS 查找一条增广路。

    返回：
    - bottleneck: 路径瓶颈容量；若不可达则为 0
    - parent: 前驱数组
    """
    n = len(residual)
    parent = [-1] * n
    parent[source] = source

    q: deque[int] = deque([source])

    while q:
        u = q.popleft()
        for v in range(n):
            if parent[v] != -1:
                continue
            if residual[u][v] <= 0:
                continue
            parent[v] = u
            if v == sink:
                q.clear()
                break
            q.append(v)

    if parent[sink] == -1:
        return 0, parent

    bottleneck = 10**18
    cur = sink
    while cur != source:
        prev = parent[cur]
        bottleneck = min(bottleneck, residual[prev][cur])
        cur = prev
    return int(bottleneck), parent


def edmonds_karp_max_flow(
    capacity: Sequence[Sequence[int]], source: int, sink: int
) -> Tuple[int, List[List[int]]]:
    """返回最大流值和最终残量网络。"""
    if source == sink:
        raise ValueError("source and sink must be different")

    n = len(capacity)
    residual = [list(row) for row in capacity]
    max_flow = 0

    while True:
        bottleneck, parent = bfs_augmenting_path(residual, source, sink)
        if bottleneck == 0:
            break

        max_flow += bottleneck
        cur = sink
        while cur != source:
            prev = parent[cur]
            residual[prev][cur] -= bottleneck
            residual[cur][prev] += bottleneck
            cur = prev

    return max_flow, residual


def extract_min_cut_from_residual(
    capacity: Sequence[Sequence[int]], residual: Sequence[Sequence[int]], source: int
) -> Tuple[List[int], List[int], List[Edge], int]:
    """根据最终残量网络提取最小割。"""
    n = len(capacity)
    visited = [False] * n
    q: deque[int] = deque([source])
    visited[source] = True

    while q:
        u = q.popleft()
        for v in range(n):
            if visited[v]:
                continue
            if residual[u][v] <= 0:
                continue
            visited[v] = True
            q.append(v)

    s_side = [i for i, flag in enumerate(visited) if flag]
    t_side = [i for i, flag in enumerate(visited) if not flag]

    cut_edges: List[Edge] = []
    cut_capacity = 0
    for u in s_side:
        for v in t_side:
            cap_uv = capacity[u][v]
            if cap_uv > 0:
                cut_edges.append((u, v, cap_uv))
                cut_capacity += cap_uv

    return s_side, t_side, cut_edges, cut_capacity


def brute_force_min_cut(
    capacity: Sequence[Sequence[int]], source: int, sink: int
) -> Tuple[int, Tuple[List[int], List[int]]]:
    """穷举所有 s-t 划分，返回最小割容量（仅用于小图校验）。"""
    n = len(capacity)
    others = [v for v in range(n) if v not in (source, sink)]

    best = 10**18
    best_partition: Tuple[List[int], List[int]] | None = None

    # 掩码表示 others 中哪些点放入 S。
    for mask in range(1 << len(others)):
        in_s = [False] * n
        in_s[source] = True
        in_s[sink] = False

        for bit, node in enumerate(others):
            if (mask >> bit) & 1:
                in_s[node] = True

        s_side = [i for i in range(n) if in_s[i]]
        t_side = [i for i in range(n) if not in_s[i]]

        cut_cap = 0
        for u in s_side:
            row = capacity[u]
            for v in t_side:
                cut_cap += row[v]

        if cut_cap < best:
            best = cut_cap
            best_partition = (s_side, t_side)

    if best_partition is None:
        raise RuntimeError("failed to enumerate cuts")
    return int(best), best_partition


def solve_min_cut(n: int, edges: Sequence[Edge], source: int, sink: int) -> MinCutResult:
    """对单个图实例求最小割，并给出结构化结果。"""
    capacity = build_capacity_matrix(n, edges)
    max_flow, residual = edmonds_karp_max_flow(capacity, source, sink)
    s_side, t_side, cut_edges, cut_capacity = extract_min_cut_from_residual(
        capacity, residual, source
    )

    return MinCutResult(
        max_flow=max_flow,
        min_cut_capacity=cut_capacity,
        s_side=s_side,
        t_side=t_side,
        cut_edges=cut_edges,
    )


def run_case(name: str, n: int, edges: Sequence[Edge], source: int, sink: int) -> None:
    """运行单个样例并做一致性断言。"""
    print(f"\n=== Case: {name} ===")
    result = solve_min_cut(n=n, edges=edges, source=source, sink=sink)

    capacity = build_capacity_matrix(n, edges)
    brute_value, brute_partition = brute_force_min_cut(capacity, source, sink)

    print(f"source={source}, sink={sink}")
    print(f"max_flow={result.max_flow}")
    print(f"min_cut_by_residual={result.min_cut_capacity}")
    print(f"min_cut_by_bruteforce={brute_value}")
    print(f"S={result.s_side}, T={result.t_side}")
    print(f"cut_edges={result.cut_edges}")
    print(f"bruteforce_partition={brute_partition}")

    assert result.max_flow == result.min_cut_capacity, (
        f"max_flow({result.max_flow}) != residual_cut({result.min_cut_capacity})"
    )
    assert result.max_flow == brute_value, f"max_flow({result.max_flow}) != brute_cut({brute_value})"


def case_definitions() -> Iterable[Tuple[str, int, List[Edge], int, int]]:
    """固定样例集合。"""
    yield (
        "classic_clrs",
        6,
        [
            (0, 1, 16),
            (0, 2, 13),
            (1, 2, 10),
            (2, 1, 4),
            (1, 3, 12),
            (2, 4, 14),
            (3, 2, 9),
            (4, 3, 7),
            (3, 5, 20),
            (4, 5, 4),
        ],
        0,
        5,
    )

    yield (
        "parallel_edges",
        5,
        [
            (0, 1, 3),
            (0, 1, 2),
            (0, 2, 4),
            (1, 2, 1),
            (1, 3, 3),
            (2, 3, 2),
            (2, 4, 3),
            (3, 4, 4),
        ],
        0,
        4,
    )

    yield (
        "sparse_chain_like",
        7,
        [
            (0, 1, 5),
            (0, 2, 3),
            (1, 3, 4),
            (2, 3, 2),
            (2, 4, 2),
            (3, 5, 6),
            (4, 5, 1),
            (5, 6, 5),
        ],
        0,
        6,
    )


def main() -> None:
    for name, n, edges, source, sink in case_definitions():
        run_case(name=name, n=n, edges=edges, source=source, sink=sink)

    print("\nAll min-cut checks passed.")


if __name__ == "__main__":
    main()
