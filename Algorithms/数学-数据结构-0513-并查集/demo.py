"""Union-Find (Disjoint Set Union) minimal runnable MVP.

Run:
    uv run python demo.py
"""

from __future__ import annotations

import random
from typing import List


class UnionFind:
    """Disjoint Set Union with path compression and union by size."""

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self.n = n
        self.parent: List[int] = list(range(n))
        self.size_arr: List[int] = [1] * n
        self.components = n

    def _check_index(self, x: int) -> None:
        if not 0 <= x < self.n:
            raise IndexError(f"index out of range: {x}")

    def find(self, x: int) -> int:
        """Find root of x with path halving compression."""
        self._check_index(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Merge sets containing a and b; return whether merge happened."""
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False

        # Union by size: attach smaller tree under larger root.
        if self.size_arr[ra] < self.size_arr[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size_arr[ra] += self.size_arr[rb]
        self.components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def component_size(self, x: int) -> int:
        return self.size_arr[self.find(x)]

    def groups(self) -> List[List[int]]:
        buckets: dict[int, List[int]] = {}
        for i in range(self.n):
            root = self.find(i)
            buckets.setdefault(root, []).append(i)
        return sorted((sorted(g) for g in buckets.values()), key=lambda g: g[0])


class NaiveDisjointSet:
    """Slow baseline implementation for randomized cross-check."""

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self.n = n
        self.label: List[int] = list(range(n))
        self.components = n

    def _check_index(self, x: int) -> None:
        if not 0 <= x < self.n:
            raise IndexError(f"index out of range: {x}")

    def find(self, x: int) -> int:
        self._check_index(x)
        return self.label[x]

    def union(self, a: int, b: int) -> bool:
        la = self.find(a)
        lb = self.find(b)
        if la == lb:
            return False
        for i in range(self.n):
            if self.label[i] == lb:
                self.label[i] = la
        self.components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def groups(self) -> List[List[int]]:
        buckets: dict[int, List[int]] = {}
        for i, lab in enumerate(self.label):
            buckets.setdefault(lab, []).append(i)
        return sorted((sorted(g) for g in buckets.values()), key=lambda g: g[0])


def deterministic_demo() -> None:
    print("=== Deterministic DSU demo ===")
    uf = UnionFind(10)
    operations = [(0, 1), (1, 2), (3, 4), (5, 6), (2, 6), (7, 8), (8, 9), (4, 5)]

    for step, (a, b) in enumerate(operations, start=1):
        merged = uf.union(a, b)
        print(
            f"step={step:02d} union({a}, {b}) -> merged={merged:<5} "
            f"components={uf.components}"
        )

    queries = [(0, 6), (3, 9), (7, 9), (0, 9)]
    for a, b in queries:
        print(f"connected({a}, {b}) = {uf.connected(a, b)}")

    print(f"component_size(0) = {uf.component_size(0)}")
    print(f"groups = {uf.groups()}")


def randomized_cross_check(seed: int = 513, n: int = 30, rounds: int = 500) -> None:
    print("\n=== Randomized cross-check vs naive baseline ===")
    rng = random.Random(seed)
    uf = UnionFind(n)
    base = NaiveDisjointSet(n)

    for _ in range(rounds):
        a = rng.randrange(n)
        b = rng.randrange(n)
        op_choice = rng.random()

        if op_choice < 0.6:
            merged_uf = uf.union(a, b)
            merged_base = base.union(a, b)
            assert merged_uf == merged_base
            assert uf.components == base.components
        elif op_choice < 0.9:
            assert uf.connected(a, b) == base.connected(a, b)
        else:
            assert uf.component_size(a) == len(
                next(group for group in base.groups() if a in group)
            )

    assert uf.groups() == base.groups()
    print(
        "randomized check passed: "
        f"seed={seed}, n={n}, rounds={rounds}, components={uf.components}"
    )


def main() -> None:
    deterministic_demo()
    randomized_cross_check()


if __name__ == "__main__":
    main()
