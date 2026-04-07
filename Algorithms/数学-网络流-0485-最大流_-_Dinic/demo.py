"""Minimal runnable MVP for Dinic maximum flow."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Edge:
    to: int
    rev: int
    cap: int


class Dinic:
    """Dinic maximum flow on directed graph with integral capacities."""

    def __init__(self, n: int) -> None:
        if n <= 1:
            raise ValueError("n must be >= 2")
        self.n = n
        self.graph: List[List[Edge]] = [[] for _ in range(n)]
        self.level: List[int] = [-1] * n
        self.it: List[int] = [0] * n

        # Small instrumentation for demo output.
        self.phase_count = 0
        self.dfs_calls = 0
        self.augmentations = 0

    def add_edge(self, u: int, v: int, cap: int) -> None:
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise ValueError("edge endpoint out of range")
        if cap < 0:
            raise ValueError("capacity must be non-negative")

        fwd = Edge(to=v, rev=len(self.graph[v]), cap=cap)
        rev = Edge(to=u, rev=len(self.graph[u]), cap=0)
        self.graph[u].append(fwd)
        self.graph[v].append(rev)

    def _bfs_level_graph(self, s: int, t: int) -> bool:
        self.level = [-1] * self.n
        q: deque[int] = deque([s])
        self.level[s] = 0

        while q:
            u = q.popleft()
            for e in self.graph[u]:
                if e.cap > 0 and self.level[e.to] < 0:
                    self.level[e.to] = self.level[u] + 1
                    q.append(e.to)

        return self.level[t] >= 0

    def _dfs_blocking_flow(self, u: int, t: int, f: int) -> int:
        self.dfs_calls += 1
        if u == t:
            return f

        while self.it[u] < len(self.graph[u]):
            i = self.it[u]
            e = self.graph[u][i]
            if e.cap > 0 and self.level[e.to] == self.level[u] + 1:
                pushed = self._dfs_blocking_flow(e.to, t, min(f, e.cap))
                if pushed > 0:
                    e.cap -= pushed
                    rev = self.graph[e.to][e.rev]
                    rev.cap += pushed
                    return pushed
            self.it[u] += 1

        return 0

    def max_flow(self, s: int, t: int) -> int:
        if not (0 <= s < self.n and 0 <= t < self.n):
            raise ValueError("source/sink out of range")
        if s == t:
            return 0

        flow = 0
        inf = 10**18
        while self._bfs_level_graph(s, t):
            self.phase_count += 1
            self.it = [0] * self.n
            while True:
                pushed = self._dfs_blocking_flow(s, t, inf)
                if pushed == 0:
                    break
                self.augmentations += 1
                flow += pushed
        return flow

    def min_cut_reachable(self, s: int) -> List[bool]:
        """Nodes reachable from s in final residual graph."""
        vis = [False] * self.n
        q: deque[int] = deque([s])
        vis[s] = True

        while q:
            u = q.popleft()
            for e in self.graph[u]:
                if e.cap > 0 and not vis[e.to]:
                    vis[e.to] = True
                    q.append(e.to)

        return vis


def edmonds_karp_max_flow(n: int, edges: Sequence[Tuple[int, int, int]], s: int, t: int) -> int:
    """Reference implementation for cross-checking Dinic."""
    cap = np.zeros((n, n), dtype=np.int64)
    for u, v, c in edges:
        cap[u, v] += np.int64(c)

    flow = np.int64(0)
    while True:
        parent = [-1] * n
        parent[s] = s
        q: deque[int] = deque([s])

        while q and parent[t] == -1:
            u = q.popleft()
            row = cap[u]
            for v in range(n):
                if parent[v] == -1 and row[v] > 0:
                    parent[v] = u
                    q.append(v)
                    if v == t:
                        break

        if parent[t] == -1:
            break

        aug = np.int64(1 << 60)
        v = t
        while v != s:
            u = parent[v]
            aug = min(aug, cap[u, v])
            v = u

        v = t
        while v != s:
            u = parent[v]
            cap[u, v] -= aug
            cap[v, u] += aug
            v = u

        flow += aug

    return int(flow)


def build_dinic(n: int, edges: Sequence[Tuple[int, int, int]]) -> Dinic:
    d = Dinic(n)
    for u, v, c in edges:
        d.add_edge(u, v, c)
    return d


def cut_capacity(edges: Sequence[Tuple[int, int, int]], reachable: Sequence[bool]) -> int:
    total = 0
    for u, v, c in edges:
        if reachable[u] and not reachable[v]:
            total += c
    return total


def run_classic_case() -> None:
    # Classic textbook network (CLRS style), expected max flow = 23.
    n = 6
    s, t = 0, 5
    edges = [
        (0, 1, 16),
        (0, 2, 13),
        (1, 2, 10),
        (2, 1, 4),
        (1, 3, 12),
        (3, 2, 9),
        (2, 4, 14),
        (4, 3, 7),
        (3, 5, 20),
        (4, 5, 4),
    ]

    dinic = build_dinic(n, edges)
    flow = dinic.max_flow(s, t)
    reachable = dinic.min_cut_reachable(s)
    cut = cut_capacity(edges, reachable)

    print("[Classic] max flow:", flow)
    print("[Classic] min-cut capacity:", cut)
    print(
        "[Classic] instrumentation:",
        f"phases={dinic.phase_count}, augmentations={dinic.augmentations}, dfs_calls={dinic.dfs_calls}",
    )

    assert flow == 23, f"expected 23, got {flow}"
    assert cut == flow, "max-flow min-cut theorem check failed"


def random_graph_case(rng: np.random.Generator, case_id: int) -> None:
    n = int(rng.integers(6, 10))
    s, t = 0, n - 1

    # Ensure s->t is reachable by a base chain.
    edges: List[Tuple[int, int, int]] = []
    for u in range(n - 1):
        edges.append((u, u + 1, int(rng.integers(1, 12))))

    # Add random directed edges (allowing anti-parallel edges).
    p = 0.30
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if rng.random() < p:
                edges.append((u, v, int(rng.integers(1, 15))))

    dinic = build_dinic(n, edges)
    flow_dinic = dinic.max_flow(s, t)
    flow_ek = edmonds_karp_max_flow(n, edges, s, t)

    print(
        f"[Random {case_id}] n={n}, m={len(edges)}, flow_dinic={flow_dinic}, flow_edmonds_karp={flow_ek}"
    )

    assert flow_dinic == flow_ek, "Dinic != Edmonds-Karp on random case"


def main() -> None:
    print("Dinic maximum flow MVP demo")
    print("=" * 36)

    run_classic_case()

    rng = np.random.default_rng(20260407)
    for case_id in range(1, 7):
        random_graph_case(rng, case_id)

    print("=" * 36)
    print("All checks passed.")


if __name__ == "__main__":
    main()
