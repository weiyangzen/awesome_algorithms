"""Minimal runnable MVP for Edmonds-Karp maximum flow."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class EKResult:
    max_flow: int
    residual: np.ndarray
    bfs_runs: int
    augmentations: int


def _validate_input(n: int, edges: Sequence[Tuple[int, int, int]], source: int, sink: int) -> None:
    if n <= 1:
        raise ValueError("n must be >= 2")
    if not (0 <= source < n and 0 <= sink < n):
        raise ValueError("source/sink out of range")
    if source == sink:
        raise ValueError("source and sink must be different")

    for u, v, c in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge endpoint out of range: {(u, v, c)}")
        if c < 0:
            raise ValueError(f"capacity must be non-negative: {(u, v, c)}")


def build_capacity_matrix(n: int, edges: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    """Build capacity matrix, merging parallel edges by summation."""
    cap = np.zeros((n, n), dtype=np.int64)
    for u, v, c in edges:
        cap[u, v] += np.int64(c)
    return cap


def edmonds_karp_max_flow(
    n: int,
    edges: Sequence[Tuple[int, int, int]],
    source: int,
    sink: int,
) -> EKResult:
    """Edmonds-Karp using BFS shortest augmenting paths on residual graph."""
    _validate_input(n, edges, source, sink)

    capacity = build_capacity_matrix(n, edges)
    residual = capacity.copy()
    parent = np.full(n, -1, dtype=np.int64)

    max_flow = np.int64(0)
    bfs_runs = 0
    augmentations = 0

    while True:
        bfs_runs += 1
        parent.fill(-1)
        parent[source] = source
        q: deque[int] = deque([source])

        while q and parent[sink] == -1:
            u = q.popleft()
            row = residual[u]
            for v in np.flatnonzero(row > 0):
                if parent[v] != -1:
                    continue
                parent[v] = u
                q.append(int(v))
                if v == sink:
                    break

        if parent[sink] == -1:
            break

        aug = np.int64(1 << 60)
        v = sink
        while v != source:
            u = int(parent[v])
            aug = min(aug, residual[u, v])
            v = u

        v = sink
        while v != source:
            u = int(parent[v])
            residual[u, v] -= aug
            residual[v, u] += aug
            v = u

        max_flow += aug
        augmentations += 1

    return EKResult(
        max_flow=int(max_flow),
        residual=residual,
        bfs_runs=bfs_runs,
        augmentations=augmentations,
    )


def min_cut_reachable(residual: np.ndarray, source: int) -> List[bool]:
    """Reachability from source in final residual graph."""
    n = residual.shape[0]
    vis = [False] * n
    vis[source] = True
    q: deque[int] = deque([source])

    while q:
        u = q.popleft()
        for v in np.flatnonzero(residual[u] > 0):
            iv = int(v)
            if not vis[iv]:
                vis[iv] = True
                q.append(iv)

    return vis


def cut_capacity(edges: Sequence[Tuple[int, int, int]], reachable: Sequence[bool]) -> int:
    total = 0
    for u, v, c in edges:
        if reachable[u] and not reachable[v]:
            total += c
    return total


@dataclass
class Edge:
    to: int
    rev: int
    cap: int


class Dinic:
    """Reference implementation for random-case cross-checking."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.graph: List[List[Edge]] = [[] for _ in range(n)]
        self.level: List[int] = [-1] * n
        self.it: List[int] = [0] * n

    def add_edge(self, u: int, v: int, cap: int) -> None:
        fwd = Edge(to=v, rev=len(self.graph[v]), cap=cap)
        rev = Edge(to=u, rev=len(self.graph[u]), cap=0)
        self.graph[u].append(fwd)
        self.graph[v].append(rev)

    def _bfs(self, source: int, sink: int) -> bool:
        self.level = [-1] * self.n
        self.level[source] = 0
        q: deque[int] = deque([source])

        while q:
            u = q.popleft()
            for e in self.graph[u]:
                if e.cap > 0 and self.level[e.to] == -1:
                    self.level[e.to] = self.level[u] + 1
                    q.append(e.to)

        return self.level[sink] != -1

    def _dfs(self, u: int, sink: int, f: int) -> int:
        if u == sink:
            return f

        while self.it[u] < len(self.graph[u]):
            i = self.it[u]
            e = self.graph[u][i]
            if e.cap > 0 and self.level[e.to] == self.level[u] + 1:
                pushed = self._dfs(e.to, sink, min(f, e.cap))
                if pushed > 0:
                    e.cap -= pushed
                    self.graph[e.to][e.rev].cap += pushed
                    return pushed
            self.it[u] += 1

        return 0

    def max_flow(self, source: int, sink: int) -> int:
        flow = 0
        inf = 10**18
        while self._bfs(source, sink):
            self.it = [0] * self.n
            while True:
                pushed = self._dfs(source, sink, inf)
                if pushed == 0:
                    break
                flow += pushed
        return flow


def dinic_reference_max_flow(
    n: int,
    edges: Sequence[Tuple[int, int, int]],
    source: int,
    sink: int,
) -> int:
    d = Dinic(n)
    for u, v, c in edges:
        d.add_edge(u, v, c)
    return d.max_flow(source, sink)


def run_classic_case() -> None:
    # Classic CLRS case, expected max flow = 23.
    n = 6
    source, sink = 0, 5
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

    result = edmonds_karp_max_flow(n, edges, source, sink)
    reachable = min_cut_reachable(result.residual, source)
    cut = cut_capacity(edges, reachable)

    print("[Classic] max flow:", result.max_flow)
    print("[Classic] min-cut capacity:", cut)
    print(
        "[Classic] instrumentation:",
        f"bfs_runs={result.bfs_runs}, augmentations={result.augmentations}",
    )

    assert result.max_flow == 23, f"expected 23, got {result.max_flow}"
    assert cut == result.max_flow, "max-flow min-cut theorem check failed"


def random_case(rng: np.random.Generator, case_id: int) -> None:
    n = int(rng.integers(6, 10))
    source, sink = 0, n - 1

    edges: List[Tuple[int, int, int]] = []

    # Make sure source reaches sink via a base chain.
    for u in range(n - 1):
        edges.append((u, u + 1, int(rng.integers(1, 10))))

    # Add random directed edges (allow anti-parallel edges).
    p = 0.28
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if rng.random() < p:
                edges.append((u, v, int(rng.integers(1, 15))))

    flow_ek = edmonds_karp_max_flow(n, edges, source, sink).max_flow
    flow_dinic = dinic_reference_max_flow(n, edges, source, sink)

    print(
        f"[Random {case_id}] n={n}, m={len(edges)}, flow_ek={flow_ek}, flow_dinic={flow_dinic}"
    )

    assert flow_ek == flow_dinic, "Edmonds-Karp != Dinic on random case"


def main() -> None:
    print("Edmonds-Karp maximum flow MVP demo")
    print("=" * 42)

    run_classic_case()

    rng = np.random.default_rng(20260407)
    for case_id in range(1, 7):
        random_case(rng, case_id)

    print("=" * 42)
    print("All checks passed.")


if __name__ == "__main__":
    main()
