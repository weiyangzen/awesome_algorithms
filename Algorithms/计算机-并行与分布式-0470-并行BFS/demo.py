"""并行 BFS 最小可运行 MVP.

运行:
    uv run python demo.py
"""

from __future__ import annotations

import os
import threading
from collections import deque
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
    """生成可复现的连通稀疏无向图。

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


def sequential_bfs(graph: Graph, start: int = 0) -> tuple[list[int], list[int]]:
    """顺序 BFS，返回访问顺序与最短层数距离。"""
    n = len(graph)
    if n == 0:
        return [], []
    if not (0 <= start < n):
        raise ValueError(f"start {start} out of range")

    visited = [False] * n
    dist = [-1] * n
    order: list[int] = []

    queue: deque[int] = deque([start])
    visited[start] = True
    dist[start] = 0

    while queue:
        u = queue.popleft()
        order.append(u)

        for v in graph[u]:
            if not visited[v]:
                visited[v] = True
                dist[v] = dist[u] + 1
                queue.append(v)

    return order, dist


def _chunk_frontier(frontier: list[int], parts: int) -> list[list[int]]:
    """把当前 frontier 切分为若干块，供并行 worker 处理。"""
    if not frontier:
        return []
    parts = max(1, min(parts, len(frontier)))
    chunk_size = (len(frontier) + parts - 1) // parts
    return [frontier[i : i + chunk_size] for i in range(0, len(frontier), chunk_size)]


def parallel_bfs(graph: Graph, start: int = 0, workers: int | None = None) -> tuple[list[int], list[int]]:
    """层同步线程并行 BFS.

    关键点:
    - 每一轮只扩展同一层 frontier;
    - 并行处理 frontier 分块;
    - 用锁保护 visited 的检查+标记，避免重复入队。
    """
    n = len(graph)
    if n == 0:
        return [], []
    if not (0 <= start < n):
        raise ValueError(f"start {start} out of range")

    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    workers = max(1, workers)
    if workers == 1:
        return sequential_bfs(graph, start)

    visited = [False] * n
    dist = [-1] * n
    visit_lock = threading.Lock()

    visited[start] = True
    dist[start] = 0
    order: list[int] = [start]

    frontier: list[int] = [start]
    depth = 0

    while frontier:
        chunks = _chunk_frontier(frontier, workers)

        def _worker(chunk: list[int]) -> list[int]:
            local_next: list[int] = []
            for u in chunk:
                for v in graph[u]:
                    claimed = False
                    with visit_lock:
                        if not visited[v]:
                            visited[v] = True
                            dist[v] = depth + 1
                            claimed = True
                    if claimed:
                        local_next.append(v)
            return local_next

        next_frontier: list[int] = []
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [executor.submit(_worker, chunk) for chunk in chunks]
            for fut in futures:
                next_frontier.extend(fut.result())

        order.extend(next_frontier)
        frontier = next_frontier
        depth += 1

    return order, dist


def _validate_results(
    seq_order: list[int],
    seq_dist: list[int],
    par_order: list[int],
    par_dist: list[int],
    n: int,
) -> None:
    """校验可达集合、唯一性与距离一致性。"""
    seq_set = set(seq_order)
    par_set = set(par_order)

    if len(seq_order) != len(seq_set):
        raise AssertionError("sequential_bfs produced duplicated visits")
    if len(par_order) != len(par_set):
        raise AssertionError("parallel_bfs produced duplicated visits")
    if seq_set != par_set:
        raise AssertionError("reachable vertex set mismatch between sequential and parallel")
    if seq_dist != par_dist:
        raise AssertionError("distance array mismatch between sequential and parallel")
    if len(seq_set) != n:
        raise AssertionError("graph is expected to be connected but traversal did not cover all")


@dataclass
class BenchResult:
    name: str
    seconds: float


def _benchmark(name: str, fn) -> tuple[tuple[list[int], list[int]], BenchResult]:
    t0 = perf_counter()
    result = fn()
    elapsed = perf_counter() - t0
    return result, BenchResult(name=name, seconds=elapsed)


def run_self_test() -> None:
    """小图正确性自检。"""
    graph = build_undirected_graph(
        9,
        [
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 5),
            (4, 6),
            (5, 7),
            (6, 8),
            (7, 8),
        ],
    )
    seq_order, seq_dist = sequential_bfs(graph, start=0)
    par_order, par_dist = parallel_bfs(graph, start=0, workers=4)

    expected_dist = [0, 1, 1, 2, 2, 2, 3, 3, 4]
    if seq_dist != expected_dist:
        raise AssertionError(f"unexpected sequential distances: {seq_dist}")

    _validate_results(seq_order, seq_dist, par_order, par_dist, n=9)
    print("[self-test] passed")


def main() -> None:
    run_self_test()

    n = 10_000
    m = 50_000
    seed = 20260407
    workers = min(8, os.cpu_count() or 1)

    graph = generate_connected_sparse_graph(n=n, m=m, seed=seed)
    print(f"[config] n={n}, m={m}, workers={workers}, seed={seed}")

    (seq_order, seq_dist), seq_metric = _benchmark(
        "sequential_bfs",
        lambda: sequential_bfs(graph, start=0),
    )
    (par_order, par_dist), par_metric = _benchmark(
        "parallel_bfs(level-synchronous, threaded)",
        lambda: parallel_bfs(graph, start=0, workers=workers),
    )

    _validate_results(seq_order, seq_dist, par_order, par_dist, n=n)

    speedup = seq_metric.seconds / par_metric.seconds if par_metric.seconds > 0 else float("inf")
    max_depth = max(seq_dist)

    print(f"[time] {seq_metric.name}: {seq_metric.seconds:.4f}s")
    print(f"[time] {par_metric.name}: {par_metric.seconds:.4f}s")
    print(f"[time] speedup(sequential/parallel): {speedup:.3f}x")
    print(f"[check] distance consistency passed, max_depth={max_depth}")


if __name__ == "__main__":
    main()
