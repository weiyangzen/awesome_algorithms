"""深度优先搜索 (DFS) 的最小可运行 MVP.

功能:
- 建图（无向/有向）
- 递归 DFS 与迭代 DFS
- 路径可达性判定
- 无向图连通分量分解
- 无向图环检测
- 固定样例与随机对拍（BFS + 并查集校验）
"""

from __future__ import annotations

from collections import deque
import random
from typing import Iterable, List, Sequence, Set, Tuple

Edge = Tuple[int, int]
Graph = List[List[int]]


def build_graph(n: int, edges: Sequence[Edge], directed: bool = False) -> Graph:
    """从边集构造邻接表，并做基础合法性校验。"""
    if n <= 0:
        raise ValueError("n must be positive")

    adj_sets: List[Set[int]] = [set() for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge index {idx} has vertex out of range: {(u, v)}")
        adj_sets[u].add(v)
        if not directed:
            adj_sets[v].add(u)

    # 固定邻居顺序以保证遍历结果可复现。
    return [sorted(neis) for neis in adj_sets]


def dfs_recursive(graph: Graph, start: int) -> List[int]:
    """递归版 DFS，返回首次访问顺序（前序）。"""
    n = len(graph)
    if not (0 <= start < n):
        raise ValueError(f"start {start} out of range [0, {n - 1}]")

    visited = [False] * n
    order: List[int] = []

    def visit(u: int) -> None:
        visited[u] = True
        order.append(u)
        for v in graph[u]:
            if not visited[v]:
                visit(v)

    visit(start)
    return order


def dfs_iterative(graph: Graph, start: int) -> List[int]:
    """迭代版 DFS，返回首次访问顺序（前序）。"""
    n = len(graph)
    if not (0 <= start < n):
        raise ValueError(f"start {start} out of range [0, {n - 1}]")

    visited = [False] * n
    order: List[int] = []
    stack: List[int] = [start]

    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)

        # 逆序压栈，保证出栈时仍按升序邻居访问，贴近递归前序。
        for v in reversed(graph[u]):
            if not visited[v]:
                stack.append(v)

    return order


def reachable_dfs(graph: Graph, src: int, dst: int) -> bool:
    """使用 DFS 判断 src 是否可达 dst。"""
    n = len(graph)
    if not (0 <= src < n and 0 <= dst < n):
        raise ValueError("src/dst out of range")
    if src == dst:
        return True

    visited = [False] * n
    stack = [src]
    visited[src] = True

    while stack:
        u = stack.pop()
        for v in graph[u]:
            if not visited[v]:
                if v == dst:
                    return True
                visited[v] = True
                stack.append(v)
    return False


def connected_components(graph: Graph) -> List[List[int]]:
    """无向图连通分量（基于 DFS）。"""
    n = len(graph)
    visited = [False] * n
    components: List[List[int]] = []

    for s in range(n):
        if visited[s]:
            continue
        comp: List[int] = []
        stack = [s]
        visited[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in reversed(graph[u]):
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        components.append(comp)

    return components


def has_cycle_undirected(graph: Graph) -> bool:
    """无向图环检测（迭代 DFS + 父节点过滤）。"""
    n = len(graph)
    visited = [False] * n

    for s in range(n):
        if visited[s]:
            continue

        visited[s] = True
        # 栈元素: (当前点, 父节点, 当前扫描到的邻居下标)
        stack: List[Tuple[int, int, int]] = [(s, -1, 0)]

        while stack:
            u, parent, idx = stack[-1]
            if idx >= len(graph[u]):
                stack.pop()
                continue

            v = graph[u][idx]
            stack[-1] = (u, parent, idx + 1)
            if not visited[v]:
                visited[v] = True
                stack.append((v, u, 0))
            elif v != parent:
                return True

    return False


def bfs_reachable(graph: Graph, src: int, dst: int) -> bool:
    """用于对拍的 BFS 可达性判定。"""
    if src == dst:
        return True
    n = len(graph)
    vis = [False] * n
    dq = deque([src])
    vis[src] = True

    while dq:
        u = dq.popleft()
        for v in graph[u]:
            if not vis[v]:
                if v == dst:
                    return True
                vis[v] = True
                dq.append(v)
    return False


def has_cycle_dsu_reference(n: int, edges: Sequence[Edge]) -> bool:
    """并查集版无向图环检测，仅用于随机对拍参考。"""
    parent = list(range(n))
    size = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        return True

    seen: Set[Edge] = set()
    for u, v in edges:
        e = (u, v) if u <= v else (v, u)
        if e in seen:
            continue
        seen.add(e)
        if u == v:
            return True
        if not union(u, v):
            return True
    return False


def format_graph(graph: Graph) -> str:
    lines = []
    for u, neis in enumerate(graph):
        lines.append(f"{u}: {neis}")
    return "\n".join(lines)


def random_edges(n: int, p: float, rng: random.Random) -> List[Edge]:
    edges: List[Edge] = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def run_fixed_demo() -> None:
    print("=== Fixed Demo ===")
    n = 8
    edges: List[Edge] = [
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 4),
        (2, 5),
        (5, 6),
        (4, 6),  # 形成一个环: 1-4-6-5-2-0-1 的子环结构
    ]

    graph = build_graph(n, edges, directed=False)
    print("Graph adjacency list:")
    print(format_graph(graph))

    start = 0
    rec_order = dfs_recursive(graph, start)
    it_order = dfs_iterative(graph, start)
    print(f"\nDFS recursive order from {start}: {rec_order}")
    print(f"DFS iterative order from {start}: {it_order}")
    if rec_order != it_order:
        raise AssertionError("recursive and iterative DFS order mismatch")

    comps = connected_components(graph)
    print(f"\nConnected components: {comps}")

    cycle_flag = has_cycle_undirected(graph)
    print(f"Has cycle (undirected): {cycle_flag}")
    if not cycle_flag:
        raise AssertionError("fixed demo should contain a cycle")

    queries = [(0, 6), (0, 7), (3, 5), (6, 7)]
    print("\nReachability queries:")
    for src, dst in queries:
        ans = reachable_dfs(graph, src, dst)
        print(f"reachable_dfs({src}, {dst}) = {ans}")

    print()


def run_random_crosscheck() -> None:
    print("=== Random Cross Check ===")
    rng = random.Random(20260407)
    trials = 30

    for tid in range(1, trials + 1):
        n = rng.randint(6, 14)
        p = rng.uniform(0.12, 0.42)
        edges = random_edges(n, p, rng)
        graph = build_graph(n, edges, directed=False)

        # 1) 可达性对拍: DFS vs BFS
        for _ in range(20):
            src = rng.randrange(n)
            dst = rng.randrange(n)
            a = reachable_dfs(graph, src, dst)
            b = bfs_reachable(graph, src, dst)
            if a != b:
                raise AssertionError(
                    f"reachability mismatch at trial {tid}: src={src}, dst={dst}, dfs={a}, bfs={b}"
                )

        # 2) 访问集合一致性: 递归 DFS 与迭代 DFS
        start = rng.randrange(n)
        rec_order = dfs_recursive(graph, start)
        it_order = dfs_iterative(graph, start)
        if set(rec_order) != set(it_order):
            raise AssertionError(f"visited-set mismatch at trial {tid}, start={start}")

        # 3) 环检测对拍: DFS vs 并查集参考判定
        cycle_dfs = has_cycle_undirected(graph)
        cycle_ref = has_cycle_dsu_reference(n, edges)
        if cycle_dfs != cycle_ref:
            raise AssertionError(
                f"cycle mismatch at trial {tid}: dfs={cycle_dfs}, ref={cycle_ref}"
            )

    print(f"All {trials} random trials passed.\n")


def main() -> None:
    run_fixed_demo()
    run_random_crosscheck()
    print("DFS MVP finished successfully.")


if __name__ == "__main__":
    main()
