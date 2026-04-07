"""Tarjan offline LCA minimal runnable MVP.

Run:
    python3 demo.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class DSU:
    """Disjoint Set Union with path compression + union by rank."""

    parent: List[int]
    rank: List[int]

    @classmethod
    def make(cls, n: int) -> "DSU":
        return cls(parent=list(range(n)), rank=[0] * n)

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


def tarjan_offline_lca(
    n: int,
    edges: Sequence[Tuple[int, int]],
    queries: Sequence[Tuple[int, int]],
) -> List[Optional[int]]:
    """Answer LCA queries in a tree/forest using Tarjan offline algorithm.

    Returns:
        answers[i] is the LCA for queries[i], or None if endpoints are disconnected.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []

    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) out of range for n={n}")
        adj[u].append(v)
        adj[v].append(u)

    query_bucket: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for qi, (u, v) in enumerate(queries):
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"query ({u}, {v}) out of range for n={n}")
        query_bucket[u].append((v, qi))
        query_bucket[v].append((u, qi))

    dsu = DSU.make(n)
    ancestor = list(range(n))
    visited = [False] * n
    seen = [False] * n
    component = [-1] * n
    answers: List[Optional[int]] = [None] * len(queries)

    def dfs(u: int, parent: int, comp_id: int) -> None:
        seen[u] = True
        component[u] = comp_id
        ancestor[dsu.find(u)] = u

        for v in adj[u]:
            if v == parent or seen[v]:
                continue
            dfs(v, u, comp_id)
            dsu.union(u, v)
            ancestor[dsu.find(u)] = u

        visited[u] = True

        for other, qi in query_bucket[u]:
            if not visited[other]:
                continue
            if component[other] != component[u]:
                answers[qi] = None
                continue
            answers[qi] = ancestor[dsu.find(other)]

    comp_id = 0
    for node in range(n):
        if seen[node]:
            continue
        dfs(node, -1, comp_id)
        comp_id += 1

    return answers


def naive_lca_forest(
    n: int,
    edges: Sequence[Tuple[int, int]],
    queries: Sequence[Tuple[int, int]],
) -> List[Optional[int]]:
    """Reference implementation for verification (O(height) per query)."""
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    parent = [-1] * n
    depth = [0] * n
    comp = [-1] * n

    cid = 0
    for start in range(n):
        if comp[start] != -1:
            continue
        comp[start] = cid
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if comp[v] != -1:
                    continue
                comp[v] = cid
                parent[v] = u
                depth[v] = depth[u] + 1
                queue.append(v)
        cid += 1

    result: List[Optional[int]] = []
    for u, v in queries:
        if comp[u] != comp[v]:
            result.append(None)
            continue

        a, b = u, v
        while depth[a] > depth[b]:
            a = parent[a]
        while depth[b] > depth[a]:
            b = parent[b]
        while a != b:
            a = parent[a]
            b = parent[b]
        result.append(a)

    return result


def main() -> None:
    # Forest example:
    # component 0: 0..8
    # component 1: 9--10
    n = 11
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 4),
        (2, 5),
        (2, 6),
        (5, 7),
        (5, 8),
        (9, 10),
    ]

    queries = [
        (3, 4),   # 1
        (3, 6),   # 0
        (7, 8),   # 5
        (7, 6),   # 2
        (0, 8),   # 0
        (9, 10),  # 9
        (3, 10),  # None (different components)
        (2, 2),   # 2
    ]

    expected = [1, 0, 5, 2, 0, 9, None, 2]

    tarjan_answers = tarjan_offline_lca(n, edges, queries)
    naive_answers = naive_lca_forest(n, edges, queries)

    print("Tarjan Offline LCA demo")
    print(f"nodes={n}, edges={len(edges)}, queries={len(queries)}")
    print("query results:")
    for i, ((u, v), ans, exp) in enumerate(zip(queries, tarjan_answers, expected)):
        print(f"  q{i}: LCA({u}, {v}) = {ans}  (expected={exp})")

    print("consistency checks:")
    print(f"  tarjan == expected: {tarjan_answers == expected}")
    print(f"  tarjan == naive   : {tarjan_answers == naive_answers}")

    if tarjan_answers != expected or tarjan_answers != naive_answers:
        raise RuntimeError("LCA answers mismatch; implementation should be checked.")


if __name__ == "__main__":
    main()
