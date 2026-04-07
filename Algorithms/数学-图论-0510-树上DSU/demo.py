"""Tree DSU (DSU on Tree) MVP.

Task implemented: for each node u, compute the number of distinct colors
in subtree(u), rooted at node 0.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from typing import DefaultDict, List, Tuple


class TreeDSUDistinctColors:
    """Small-to-large (Sack) implementation for subtree distinct color count."""

    def __init__(
        self,
        n: int,
        edges: List[Tuple[int, int]],
        colors: List[int],
        root: int = 0,
    ) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        if len(colors) != n:
            raise ValueError("len(colors) must equal n")
        if len(edges) != n - 1:
            raise ValueError("tree must have exactly n-1 edges")
        if not (0 <= root < n):
            raise ValueError("root out of range")

        self.n = n
        self.root = root
        self.colors = colors
        self.graph: List[List[int]] = [[] for _ in range(n)]

        for u, v in edges:
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"edge ({u}, {v}) has endpoint out of range")
            if u == v:
                raise ValueError("self-loop is not allowed in a tree")
            self.graph[u].append(v)
            self.graph[v].append(u)

        self.parent = [-1] * n
        self.size = [0] * n
        self.heavy = [-1] * n
        self.tin = [0] * n
        self.tout = [0] * n
        self.order = [0] * n
        self.timer = 0

        self.freq: DefaultDict[int, int] = defaultdict(int)
        self.distinct_colors = 0
        self.answer = [0] * n

    def solve(self) -> List[int]:
        sys.setrecursionlimit(max(10**6, 2 * self.n + 10))
        self._dfs_size(self.root, -1)
        if self.timer != self.n:
            raise ValueError("input graph is not connected; expected a tree")
        self._dfs_sack(self.root, -1, True)
        return self.answer[:]

    def _dfs_size(self, u: int, p: int) -> None:
        self.parent[u] = p
        self.tin[u] = self.timer
        self.order[self.timer] = u
        self.timer += 1

        self.size[u] = 1
        best_child = -1
        best_size = 0

        for v in self.graph[u]:
            if v == p:
                continue
            self._dfs_size(v, u)
            self.size[u] += self.size[v]
            if self.size[v] > best_size:
                best_size = self.size[v]
                best_child = v

        self.heavy[u] = best_child
        self.tout[u] = self.timer

    def _apply_color(self, color: int, delta: int) -> None:
        prev = self.freq[color]
        cur = prev + delta
        if cur < 0:
            raise RuntimeError("color frequency became negative")

        if prev == 0 and cur > 0:
            self.distinct_colors += 1
        elif prev > 0 and cur == 0:
            self.distinct_colors -= 1

        if cur == 0:
            del self.freq[color]
        else:
            self.freq[color] = cur

    def _apply_subtree_range(self, u: int, delta: int) -> None:
        for pos in range(self.tin[u], self.tout[u]):
            node = self.order[pos]
            self._apply_color(self.colors[node], delta)

    def _dfs_sack(self, u: int, p: int, keep: bool) -> None:
        heavy_child = self.heavy[u]

        for v in self.graph[u]:
            if v == p or v == heavy_child:
                continue
            self._dfs_sack(v, u, False)

        if heavy_child != -1:
            self._dfs_sack(heavy_child, u, True)

        for v in self.graph[u]:
            if v == p or v == heavy_child:
                continue
            self._apply_subtree_range(v, +1)

        self._apply_color(self.colors[u], +1)
        self.answer[u] = self.distinct_colors

        if not keep:
            self._apply_subtree_range(u, -1)


def naive_subtree_distinct_counts(
    n: int,
    edges: List[Tuple[int, int]],
    colors: List[int],
    root: int = 0,
) -> List[int]:
    """O(n^2) baseline for cross-checking."""
    if n <= 0:
        raise ValueError("n must be positive")
    if len(colors) != n:
        raise ValueError("len(colors) must equal n")

    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    parent = [-1] * n
    children = [[] for _ in range(n)]
    stack = [root]
    order = []

    while stack:
        u = stack.pop()
        order.append(u)
        for v in graph[u]:
            if v == parent[u]:
                continue
            if parent[v] != -1 or v == root:
                continue
            parent[v] = u
            children[u].append(v)
            stack.append(v)

    if len(order) != n:
        raise ValueError("input graph is not connected; expected a tree")

    ans = [0] * n
    for u in range(n):
        seen = set()
        st = [u]
        while st:
            x = st.pop()
            seen.add(colors[x])
            st.extend(children[x])
        ans[u] = len(seen)

    return ans


def build_fixed_case() -> Tuple[int, List[Tuple[int, int]], List[int], List[int]]:
    n = 7
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    colors = [1, 2, 1, 3, 2, 1, 4]
    expected = [4, 2, 2, 1, 1, 1, 1]
    return n, edges, colors, expected


def run_fixed_demo() -> None:
    n, edges, colors, expected = build_fixed_case()
    solver = TreeDSUDistinctColors(n, edges, colors, root=0)
    got = solver.solve()

    if got != expected:
        raise AssertionError(f"fixed demo failed: got={got}, expected={expected}")

    print("[fixed] subtree distinct colors:")
    for u, value in enumerate(got):
        print(f"  node={u:2d}, answer={value}")


def generate_random_tree(n: int, rng: random.Random) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for v in range(1, n):
        p = rng.randrange(v)
        edges.append((p, v))
    return edges


def run_random_crosscheck(rounds: int = 60, seed: int = 20260407) -> None:
    rng = random.Random(seed)
    for rid in range(1, rounds + 1):
        n = rng.randint(2, 120)
        edges = generate_random_tree(n, rng)
        max_color = max(2, n // 3)
        colors = [rng.randint(0, max_color) for _ in range(n)]

        fast = TreeDSUDistinctColors(n, edges, colors, root=0).solve()
        slow = naive_subtree_distinct_counts(n, edges, colors, root=0)

        if fast != slow:
            raise AssertionError(
                "random cross-check failed on round "
                f"{rid}: n={n}, fast={fast}, slow={slow}, edges={edges}, colors={colors}"
            )

    print(f"[random] passed {rounds} rounds of cross-check (seed={seed}).")


def main() -> None:
    run_fixed_demo()
    run_random_crosscheck()
    print("All checks passed.")


if __name__ == "__main__":
    main()
