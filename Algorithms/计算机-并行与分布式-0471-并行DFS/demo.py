"""并行 DFS 最小可运行 MVP.

运行:
    uv run python demo.py
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np

Graph = list[list[int]]
Edge = tuple[int, int]


def build_undirected_graph(n: int, edges: Iterable[Edge]) -> Graph:
    """由边集构造无向图邻接表，并固定邻居顺序。"""
    if n <= 0:
        raise ValueError("n must be positive")

    adj = [set() for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge[{idx}] out of range: {(u, v)}")
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return [sorted(nei) for nei in adj]


def generate_connected_sparse_graph(n: int, m: int, seed: int) -> Graph:
    """生成一个连通稀疏无向图。

    策略:
    - 先添加链边保证连通;
    - 再随机补边到总边数 m。
    """
    if n <= 0:
        raise ValueError("n must be positive")
    min_edges = n - 1
    max_edges = n * (n - 1) // 2
    if m < min_edges or m > max_edges:
        raise ValueError(f"m must be in [{min_edges}, {max_edges}]")

    rng = np.random.default_rng(seed)
    edges: set[Edge] = {(i, i + 1) for i in range(n - 1)}

    while len(edges) < m:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n - 1))
        if v >= u:
            v += 1
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))

    return build_undirected_graph(n, edges)


def sequential_dfs(graph: Graph, start: int = 0) -> list[int]:
    """迭代版顺序 DFS，返回访问序列。"""
    n = len(graph)
    if n == 0:
        return []
    if not (0 <= start < n):
        raise ValueError(f"start {start} out of range")

    visited = [False] * n
    order: list[int] = []
    stack = [start]
    visited[start] = True

    while stack:
        u = stack.pop()
        order.append(u)
        # 逆序压栈，保持较稳定的前序访问语义
        for v in reversed(graph[u]):
            if not visited[v]:
                visited[v] = True
                stack.append(v)

    return order


def parallel_dfs(graph: Graph, start: int = 0, workers: int | None = None) -> list[int]:
    """线程并行 DFS（共享 visited + 锁）。

    说明:
    - 起点由主线程访问;
    - 起点邻居作为并行种子;
    - 各线程在本地栈上深搜，争抢全局 visited 访问权。
    """
    n = len(graph)
    if n == 0:
        return []
    if not (0 <= start < n):
        raise ValueError(f"start {start} out of range")

    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    workers = max(1, workers)
    if workers == 1:
        return sequential_dfs(graph, start)

    visited = [False] * n
    visited_lock = threading.Lock()

    visited[start] = True
    order: list[int] = [start]

    seeds: list[int] = []
    for nei in graph[start]:
        if not visited[nei]:
            visited[nei] = True
            seeds.append(nei)

    if not seeds:
        return order

    def _worker(seed: int) -> list[int]:
        local_order: list[int] = []
        stack = [seed]

        while stack:
            u = stack.pop()
            local_order.append(u)

            for v in reversed(graph[u]):
                should_push = False
                with visited_lock:
                    if not visited[v]:
                        visited[v] = True
                        should_push = True
                if should_push:
                    stack.append(v)

        return local_order

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_worker, seed) for seed in seeds]
        for fut in futures:
            order.extend(fut.result())

    return order


def _validate_orders(seq_order: list[int], par_order: list[int], n: int) -> None:
    """检查并行结果与顺序结果在可达集合和唯一性上的一致性。"""
    seq_set = set(seq_order)
    par_set = set(par_order)

    if len(seq_order) != len(seq_set):
        raise AssertionError("sequential_dfs has duplicate visits")
    if len(par_order) != len(par_set):
        raise AssertionError("parallel_dfs has duplicate visits")
    if seq_set != par_set:
        raise AssertionError("reachable vertex set mismatch between sequential and parallel")
    if len(seq_set) != n:
        raise AssertionError("generated graph is expected to be connected but traversal did not cover all")


@dataclass
class BenchResult:
    name: str
    seconds: float


def _benchmark(name: str, fn) -> tuple[list[int], BenchResult]:
    start = perf_counter()
    result = fn()
    elapsed = perf_counter() - start
    return result, BenchResult(name=name, seconds=elapsed)


def run_self_test() -> None:
    """小规模正确性自检。"""
    graph = build_undirected_graph(
        8,
        [
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 5),
            (5, 6),
            (4, 6),
            (6, 7),
        ],
    )
    seq = sequential_dfs(graph, start=0)
    par = parallel_dfs(graph, start=0, workers=4)
    _validate_orders(seq, par, n=8)
    print("[self-test] passed")


def main() -> None:
    run_self_test()

    n = 8_000
    m = 32_000
    seed = 20260407
    workers = min(8, os.cpu_count() or 1)

    graph = generate_connected_sparse_graph(n=n, m=m, seed=seed)
    print(f"[config] n={n}, m={m}, workers={workers}, seed={seed}")

    seq_order, seq_metric = _benchmark("sequential_dfs", lambda: sequential_dfs(graph, start=0))
    par_order, par_metric = _benchmark(
        "parallel_dfs(threaded)",
        lambda: parallel_dfs(graph, start=0, workers=workers),
    )

    _validate_orders(seq_order, par_order, n=n)

    speedup = seq_metric.seconds / par_metric.seconds if par_metric.seconds > 0 else float("inf")
    print(f"[time] {seq_metric.name}: {seq_metric.seconds:.4f}s")
    print(f"[time] {par_metric.name}: {par_metric.seconds:.4f}s")
    print(f"[time] speedup(sequential/parallel): {speedup:.3f}x")
    print("[check] reachable set and uniqueness validation passed")


if __name__ == "__main__":
    main()
