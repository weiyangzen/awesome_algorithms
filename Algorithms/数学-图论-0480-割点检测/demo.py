"""割点检测（Articulation Points）最小可运行 MVP.

实现内容：
1) Tarjan DFS 时间戳法（手写，不依赖图算法黑盒库）
2) 暴力校验器（删点后重算连通分量）用于正确性对拍
3) 多组固定测试用例，脚本可直接运行
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

Edge = Tuple[int, int]


@dataclass(frozen=True)
class GraphCase:
    """单个测试图用例。"""

    name: str
    n: int
    edges: Sequence[Edge]
    expected: Sequence[int]


def _validate_graph(n: int, edges: Iterable[Edge]) -> None:
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"`n` must be a non-negative int, got {n!r}")

    for i, (u, v) in enumerate(edges):
        if not isinstance(u, int) or not isinstance(v, int):
            raise ValueError(f"edge #{i} has non-int endpoint: {(u, v)!r}")
        if n == 0:
            raise ValueError("n=0 时不应包含边")
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge #{i} endpoint out of range [0, {n - 1}]: {(u, v)!r}")


def build_adj_list(n: int, edges: Iterable[Edge]) -> List[List[int]]:
    """构建无向图邻接表，自动去重与过滤自环。"""
    edges_list = list(edges)
    _validate_graph(n, edges_list)
    adj_sets: List[Set[int]] = [set() for _ in range(n)]

    for u, v in edges_list:
        if u == v:
            # 自环不影响割点性质，这里忽略
            continue
        adj_sets[u].add(v)
        adj_sets[v].add(u)

    return [sorted(nei) for nei in adj_sets]


def articulation_points_tarjan(n: int, edges: Iterable[Edge]) -> List[int]:
    """Tarjan 时间戳法求无向图所有割点（升序返回）。"""
    if n == 0:
        return []

    adj = build_adj_list(n, edges)
    disc = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    is_ap = [False] * n
    time = 0

    def dfs(u: int) -> None:
        nonlocal time
        disc[u] = time
        low[u] = time
        time += 1

        children = 0
        for v in adj[u]:
            if disc[v] == -1:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])

                # 情况 1: u 是 DFS 树根，且至少有两个子树
                if parent[u] == -1 and children > 1:
                    is_ap[u] = True
                # 情况 2: u 不是根，存在子节点 v 使 low[v] >= disc[u]
                if parent[u] != -1 and low[v] >= disc[u]:
                    is_ap[u] = True
            elif v != parent[u]:
                # 返祖边更新 low[u]
                low[u] = min(low[u], disc[v])

    for start in range(n):
        if disc[start] == -1:
            dfs(start)

    return [i for i, flag in enumerate(is_ap) if flag]


def _components_count_with_removed(adj: Sequence[Sequence[int]], removed: int | None) -> int:
    n = len(adj)
    seen = [False] * n

    if removed is not None and (removed < 0 or removed >= n):
        raise ValueError(f"removed out of range: {removed}")

    comp = 0
    for s in range(n):
        if s == removed or seen[s]:
            continue
        comp += 1
        q: deque[int] = deque([s])
        seen[s] = True
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == removed or seen[v]:
                    continue
                seen[v] = True
                q.append(v)
    return comp


def articulation_points_bruteforce(n: int, edges: Iterable[Edge]) -> List[int]:
    """暴力法：删除每个点后判断连通分量是否增加。"""
    if n == 0:
        return []
    adj = build_adj_list(n, edges)
    base_components = _components_count_with_removed(adj, removed=None)

    ans = []
    for x in range(n):
        after = _components_count_with_removed(adj, removed=x)
        if after > base_components:
            ans.append(x)
    return ans


def run_case(case: GraphCase) -> None:
    tarjan = articulation_points_tarjan(case.n, case.edges)
    brute = articulation_points_bruteforce(case.n, case.edges)
    expected = sorted(case.expected)

    print(f"\n=== {case.name} ===")
    print(f"n={case.n}, m={len(case.edges)}")
    print(f"Tarjan APs    : {tarjan}")
    print(f"Bruteforce APs: {brute}")
    print(f"Expected APs  : {expected}")

    if tarjan != brute:
        raise AssertionError(f"[{case.name}] Tarjan != Bruteforce: {tarjan} vs {brute}")
    if tarjan != expected:
        raise AssertionError(f"[{case.name}] Tarjan != Expected: {tarjan} vs {expected}")


def main() -> None:
    cases = [
        GraphCase(
            name="Tree-like graph",
            n=7,
            edges=[(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (5, 6)],
            expected=[1, 3, 5],
        ),
        GraphCase(
            name="Cycle graph",
            n=6,
            edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
            expected=[],
        ),
        GraphCase(
            name="Disconnected graph",
            n=8,
            edges=[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6), (6, 4)],
            expected=[4],
        ),
        GraphCase(
            name="Single vertex",
            n=1,
            edges=[],
            expected=[],
        ),
    ]

    for case in cases:
        run_case(case)

    print("\nAll articulation-point checks passed.")


if __name__ == "__main__":
    main()
