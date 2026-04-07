"""Minimal runnable Link-Cut Tree MVP.

This script implements a source-level Link-Cut Tree (LCT) using splay trees,
then validates it against a brute-force forest model.
"""

from __future__ import annotations

from collections import deque
from typing import List, Set, Tuple

import numpy as np


class LinkCutTree:
    """Link-Cut Tree for dynamic forests with node-value path-sum queries."""

    def __init__(self, values_1based: List[int]) -> None:
        if len(values_1based) < 2:
            raise ValueError("values_1based must be 1-based and contain at least one node")

        self.n = len(values_1based) - 1
        self.ch = [[0, 0] for _ in range(self.n + 1)]
        self.fa = [0] * (self.n + 1)
        self.rev = [False] * (self.n + 1)

        self.val = [0] * (self.n + 1)
        self.sum = [0] * (self.n + 1)
        for i in range(1, self.n + 1):
            self.val[i] = int(values_1based[i])
            self.sum[i] = int(values_1based[i])

    def _check_node(self, x: int) -> None:
        if not (1 <= x <= self.n):
            raise IndexError(f"node id out of range: {x}")

    def _is_root(self, x: int) -> bool:
        p = self.fa[x]
        return p == 0 or (self.ch[p][0] != x and self.ch[p][1] != x)

    def _pull(self, x: int) -> None:
        left, right = self.ch[x]
        self.sum[x] = self.sum[left] + self.val[x] + self.sum[right]

    def _push(self, x: int) -> None:
        if not self.rev[x]:
            return
        left, right = self.ch[x]
        self.ch[x][0], self.ch[x][1] = right, left
        if left:
            self.rev[left] = not self.rev[left]
        if right:
            self.rev[right] = not self.rev[right]
        self.rev[x] = False

    def _rotate(self, x: int) -> None:
        p = self.fa[x]
        g = self.fa[p]

        if self.ch[p][0] == x:
            b = self.ch[x][1]
            self.ch[x][1] = p
            self.ch[p][0] = b
            if b:
                self.fa[b] = p
        else:
            b = self.ch[x][0]
            self.ch[x][0] = p
            self.ch[p][1] = b
            if b:
                self.fa[b] = p

        self.fa[p] = x
        self.fa[x] = g

        if g:
            if self.ch[g][0] == p:
                self.ch[g][0] = x
            elif self.ch[g][1] == p:
                self.ch[g][1] = x

        self._pull(p)
        self._pull(x)

    def _splay(self, x: int) -> None:
        stack = [x]
        y = x
        while not self._is_root(y):
            y = self.fa[y]
            stack.append(y)
        while stack:
            self._push(stack.pop())

        while not self._is_root(x):
            p = self.fa[x]
            g = self.fa[p]
            if not self._is_root(p):
                zigzig = (self.ch[p][0] == x) == (self.ch[g][0] == p)
                if zigzig:
                    self._rotate(p)
                else:
                    self._rotate(x)
            self._rotate(x)

    def _access(self, x: int) -> int:
        last = 0
        y = x
        while y:
            self._splay(y)
            self.ch[y][1] = last
            if last:
                self.fa[last] = y
            self._pull(y)
            last = y
            y = self.fa[y]
        self._splay(x)
        return last

    def _make_root(self, x: int) -> None:
        self._access(x)
        self.rev[x] = not self.rev[x]

    def _find_root(self, x: int) -> int:
        self._access(x)
        while True:
            self._push(x)
            left = self.ch[x][0]
            if left == 0:
                break
            x = left
        self._splay(x)
        return x

    def connected(self, u: int, v: int) -> bool:
        self._check_node(u)
        self._check_node(v)
        if u == v:
            return True
        return self._find_root(u) == self._find_root(v)

    def link(self, u: int, v: int) -> bool:
        self._check_node(u)
        self._check_node(v)
        if u == v:
            return False

        self._make_root(u)
        if self._find_root(v) == u:
            return False

        self.fa[u] = v
        return True

    def cut(self, u: int, v: int) -> bool:
        self._check_node(u)
        self._check_node(v)
        if u == v:
            return False

        self._make_root(u)
        self._access(v)

        if self.ch[v][0] != u or self.fa[u] != v or self.ch[u][1] != 0:
            return False

        self.ch[v][0] = 0
        self.fa[u] = 0
        self._pull(v)
        return True

    def set_value(self, x: int, new_value: int) -> None:
        self._check_node(x)
        self._access(x)
        self.val[x] = int(new_value)
        self._pull(x)

    def path_sum(self, u: int, v: int) -> int:
        self._check_node(u)
        self._check_node(v)
        if not self.connected(u, v):
            raise ValueError(f"path_sum requires connected nodes, got ({u}, {v})")
        self._make_root(u)
        self._access(v)
        return int(self.sum[v])


def norm_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def brute_connected(adj: List[Set[int]], u: int, v: int) -> bool:
    if u == v:
        return True
    q = deque([u])
    seen = {u}
    while q:
        x = q.popleft()
        for y in adj[x]:
            if y in seen:
                continue
            if y == v:
                return True
            seen.add(y)
            q.append(y)
    return False


def brute_path_sum(adj: List[Set[int]], values: List[int], u: int, v: int) -> int:
    parent = {u: 0}
    q = deque([u])
    while q and v not in parent:
        x = q.popleft()
        for y in adj[x]:
            if y in parent:
                continue
            parent[y] = x
            q.append(y)

    if v not in parent:
        raise ValueError(f"nodes are disconnected in brute model: ({u}, {v})")

    total = 0
    cur = v
    while cur != 0:
        total += values[cur]
        cur = parent[cur]
    return total


def run_deterministic_case() -> None:
    print("[Case] deterministic")

    values = [0, 5, 3, 7, 2, 6, 4]  # nodes 1..6
    lct = LinkCutTree(values)

    assert lct.link(1, 2)
    assert lct.link(2, 3)
    assert lct.link(2, 4)
    assert lct.link(4, 5)

    s1 = lct.path_sum(3, 5)
    print(f"path_sum(3,5) = {s1}")
    assert s1 == 18  # 7 + 3 + 2 + 6

    lct.set_value(4, 10)
    s2 = lct.path_sum(3, 5)
    print(f"after set_value(4,10), path_sum(3,5) = {s2}")
    assert s2 == 26  # 7 + 3 + 10 + 6

    assert lct.link(5, 6)
    s3 = lct.path_sum(3, 6)
    print(f"after link(5,6), path_sum(3,6) = {s3}")
    assert s3 == 30  # 7 + 3 + 10 + 6 + 4

    assert lct.cut(2, 4)
    c = lct.connected(3, 5)
    print(f"after cut(2,4), connected(3,5) = {c}")
    assert not c

    s4 = lct.path_sum(5, 6)
    print(f"path_sum(5,6) = {s4}")
    assert s4 == 10  # 6 + 4

    print("deterministic case passed")


def run_randomized_regression(seed: int = 20260407, rounds: int = 400) -> None:
    print("[Case] randomized regression")
    rng = np.random.default_rng(seed)

    n = 16
    values = [0] + rng.integers(-8, 9, size=n).astype(int).tolist()
    lct = LinkCutTree(values)

    adj: List[Set[int]] = [set() for _ in range(n + 1)]
    edges: Set[Tuple[int, int]] = set()

    for _ in range(rounds):
        op = int(rng.integers(0, 4))

        if op == 0:
            linked = False
            for _attempt in range(24):
                u = int(rng.integers(1, n + 1))
                v = int(rng.integers(1, n + 1))
                if u == v:
                    continue
                e = norm_edge(u, v)
                if e in edges:
                    continue
                if brute_connected(adj, u, v):
                    continue

                ok = lct.link(u, v)
                assert ok
                adj[u].add(v)
                adj[v].add(u)
                edges.add(e)
                linked = True
                break
            if not linked:
                # Current forest may already be dense under acyclic constraints.
                pass

        elif op == 1:
            if edges:
                edge_list = list(edges)
                idx = int(rng.integers(0, len(edge_list)))
                u, v = edge_list[idx]
                ok = lct.cut(u, v)
                assert ok
                adj[u].remove(v)
                adj[v].remove(u)
                edges.remove((u, v))

        elif op == 2:
            u = int(rng.integers(1, n + 1))
            v = int(rng.integers(1, n + 1))
            c1 = lct.connected(u, v)
            c2 = brute_connected(adj, u, v)
            assert c1 == c2
            if c1:
                s1 = lct.path_sum(u, v)
                s2 = brute_path_sum(adj, values, u, v)
                assert s1 == s2

        else:
            u = int(rng.integers(1, n + 1))
            new_val = int(rng.integers(-12, 13))
            lct.set_value(u, new_val)
            values[u] = new_val

            # Immediate local sanity check after update.
            x = int(rng.integers(1, n + 1))
            y = int(rng.integers(1, n + 1))
            if brute_connected(adj, x, y):
                s1 = lct.path_sum(x, y)
                s2 = brute_path_sum(adj, values, x, y)
                assert s1 == s2

    # Final exhaustive consistency check on connectivity and path sums.
    for u in range(1, n + 1):
        for v in range(1, n + 1):
            c1 = lct.connected(u, v)
            c2 = brute_connected(adj, u, v)
            assert c1 == c2
            if c1:
                s1 = lct.path_sum(u, v)
                s2 = brute_path_sum(adj, values, u, v)
                assert s1 == s2

    print(f"randomized regression passed (seed={seed}, rounds={rounds}, n={n})")


def main() -> None:
    run_deterministic_case()
    run_randomized_regression()
    print("Link-Cut Tree MVP finished successfully.")


if __name__ == "__main__":
    main()
