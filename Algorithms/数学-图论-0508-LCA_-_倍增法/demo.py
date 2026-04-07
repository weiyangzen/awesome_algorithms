"""Minimal runnable MVP for LCA (Lowest Common Ancestor) with binary lifting."""

from __future__ import annotations

from collections import deque
import random
from typing import List, Sequence, Tuple


Edge = Tuple[int, int]


class BinaryLiftingLCA:
    """LCA solver on a rooted tree using binary lifting.

    Nodes are indexed as integers in [0, n-1].
    """

    def __init__(self, n: int, edges: Sequence[Edge], root: int = 0) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        if not (0 <= root < n):
            raise ValueError("root out of range")
        if len(edges) != n - 1:
            raise ValueError("a tree with n nodes must have exactly n-1 edges")

        self.n = n
        self.root = root
        self.log = max(1, n.bit_length())
        self.adj: List[List[int]] = [[] for _ in range(n)]

        for idx, (u, v) in enumerate(edges):
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"edge #{idx} has node out of range: {(u, v)}")
            if u == v:
                raise ValueError(f"self-loop is not allowed in a tree: {(u, v)}")
            self.adj[u].append(v)
            self.adj[v].append(u)

        self.depth = [0] * n
        parent0 = [-1] * n
        visited = [False] * n

        queue: deque[int] = deque([root])
        visited[root] = True
        parent0[root] = root

        while queue:
            node = queue.popleft()
            for nxt in self.adj[node]:
                if visited[nxt]:
                    continue
                visited[nxt] = True
                parent0[nxt] = node
                self.depth[nxt] = self.depth[node] + 1
                queue.append(nxt)

        if not all(visited):
            raise ValueError("graph is disconnected; expected one connected tree")

        self.up: List[List[int]] = [[0] * n for _ in range(self.log)]
        self.up[0] = parent0

        for k in range(1, self.log):
            prev = self.up[k - 1]
            cur = self.up[k]
            for v in range(n):
                cur[v] = prev[prev[v]]

    def _check_node(self, node: int) -> None:
        if not (0 <= node < self.n):
            raise ValueError(f"node out of range: {node}")

    def kth_ancestor(self, node: int, k: int) -> int:
        """Return the k-th ancestor of node (0-th ancestor is itself)."""
        self._check_node(node)
        if k < 0:
            raise ValueError("k must be non-negative")

        cur = node
        bit = 0
        steps = k
        while steps:
            if steps & 1:
                cur = self.up[bit][cur]
            steps >>= 1
            bit += 1
            if bit >= self.log and steps:
                # For a valid tree root points to itself, so climbing further is stable.
                break
        return cur

    def lca(self, u: int, v: int) -> int:
        """Return LCA(u, v) in O(log n)."""
        self._check_node(u)
        self._check_node(v)

        if self.depth[u] < self.depth[v]:
            u, v = v, u

        u = self.kth_ancestor(u, self.depth[u] - self.depth[v])
        if u == v:
            return u

        for k in range(self.log - 1, -1, -1):
            pu = self.up[k][u]
            pv = self.up[k][v]
            if pu != pv:
                u, v = pu, pv

        return self.up[0][u]

    def distance(self, u: int, v: int) -> int:
        """Number of edges on the simple path from u to v."""
        w = self.lca(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[w]


def naive_lca(parent0: Sequence[int], depth: Sequence[int], u: int, v: int) -> int:
    """Reference implementation in O(height) for correctness checks."""
    uu, vv = u, v
    while depth[uu] > depth[vv]:
        uu = parent0[uu]
    while depth[vv] > depth[uu]:
        vv = parent0[vv]
    while uu != vv:
        uu = parent0[uu]
        vv = parent0[vv]
    return uu


def run_fixed_case() -> None:
    print("=== Fixed Case: hand-crafted tree ===")

    n = 9
    edges: List[Edge] = [
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 4),
        (2, 5),
        (2, 6),
        (5, 7),
        (5, 8),
    ]
    solver = BinaryLiftingLCA(n=n, edges=edges, root=0)

    queries = [
        (3, 4, 1),
        (3, 6, 0),
        (7, 8, 5),
        (7, 6, 2),
        (4, 8, 0),
        (2, 8, 2),
        (0, 8, 0),
    ]

    print("u  v  lca  dist")
    print("----------------")
    for u, v, expected in queries:
        got = solver.lca(u, v)
        if got != expected:
            raise AssertionError(f"fixed case mismatch for ({u}, {v}): {got} != {expected}")
        dist = solver.distance(u, v)
        print(f"{u:>1}  {v:>1}  {got:>3}  {dist:>4}")

    print("Fixed case passed.\n")


def generate_random_tree(n: int, rng: random.Random) -> List[Edge]:
    if n == 1:
        return []
    edges: List[Edge] = []
    for node in range(1, n):
        parent = rng.randrange(0, node)
        edges.append((node, parent))
    return edges


def run_random_regression(seed: int = 2026) -> None:
    print("=== Random Regression: compare with naive LCA ===")

    rng = random.Random(seed)
    sizes = [1, 2, 7, 31, 128]
    total_queries = 0

    for n in sizes:
        edges = generate_random_tree(n, rng)
        solver = BinaryLiftingLCA(n=n, edges=edges, root=0)
        parent0 = solver.up[0]

        q = 200
        for _ in range(q):
            u = rng.randrange(n)
            v = rng.randrange(n)
            ans_fast = solver.lca(u, v)
            ans_ref = naive_lca(parent0, solver.depth, u, v)
            if ans_fast != ans_ref:
                raise AssertionError(
                    f"random mismatch (n={n}, u={u}, v={v}): {ans_fast} != {ans_ref}"
                )
            total_queries += 1

        print(f"n={n:>3}, edges={len(edges):>3}, checked_queries={q}")

    print(f"Random regression passed, total checked queries = {total_queries}.\n")


def run_small_api_showcase() -> None:
    print("=== API Showcase: kth ancestor ===")

    n = 6
    edges: List[Edge] = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)]
    solver = BinaryLiftingLCA(n=n, edges=edges, root=0)

    examples = [(4, 0), (4, 1), (4, 2), (4, 3), (5, 2)]
    for node, k in examples:
        anc = solver.kth_ancestor(node, k)
        print(f"kth_ancestor(node={node}, k={k}) = {anc}")

    print()


def main() -> None:
    run_fixed_case()
    run_random_regression(seed=2026)
    run_small_api_showcase()
    print("All checks completed successfully.")


if __name__ == "__main__":
    main()
