"""Union-Find (Disjoint Set Union) demo with union by rank.

Run:
    uv run python demo.py
"""

from __future__ import annotations

import random
from typing import List


class UnionFindRank:
    """Disjoint Set Union with path compression and union by rank."""

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self.n = n
        self.parent: List[int] = list(range(n))
        self.rank: List[int] = [0] * n
        self.comp_size: List[int] = [1] * n
        self.components = n

    def _check_index(self, x: int) -> None:
        if not 0 <= x < self.n:
            raise IndexError(f"index out of range: {x}")

    def find(self, x: int) -> int:
        self._check_index(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        # Path compression: flatten x -> root chain.
        while self.parent[x] != x:
            parent = self.parent[x]
            self.parent[x] = root
            x = parent
        return root

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.comp_size[ra] += self.comp_size[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def size(self, x: int) -> int:
        return self.comp_size[self.find(x)]

    def groups(self) -> List[List[int]]:
        buckets: dict[int, List[int]] = {}
        for i in range(self.n):
            root = self.find(i)
            buckets.setdefault(root, []).append(i)
        return sorted((sorted(group) for group in buckets.values()), key=lambda g: g[0])


class NaiveDisjointSet:
    """Slow baseline for randomized correctness checks."""

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
        return sorted((sorted(group) for group in buckets.values()), key=lambda g: g[0])


def deterministic_demo() -> None:
    print("=== Deterministic demo: union by rank ===")
    uf = UnionFindRank(8)
    operations = [(0, 1), (1, 2), (3, 4), (2, 4), (5, 6), (6, 7), (4, 7), (0, 7)]

    for step, (a, b) in enumerate(operations, start=1):
        merged = uf.union(a, b)
        print(
            f"step={step:02d} union({a}, {b}) -> merged={merged:<5} "
            f"components={uf.components}"
        )

    queries = [(0, 7), (1, 5), (3, 6)]
    for a, b in queries:
        print(f"connected({a}, {b}) = {uf.connected(a, b)}")

    print(f"size(0) = {uf.size(0)}")
    print(f"groups = {uf.groups()}")


def randomized_cross_check(seed: int = 2026, n: int = 25, rounds: int = 400) -> None:
    print("\n=== Randomized cross-check against naive baseline ===")
    rng = random.Random(seed)
    uf = UnionFindRank(n)
    baseline = NaiveDisjointSet(n)

    for _ in range(rounds):
        a = rng.randrange(n)
        b = rng.randrange(n)

        if rng.random() < 0.65:
            merged_uf = uf.union(a, b)
            merged_base = baseline.union(a, b)
            assert merged_uf == merged_base
            assert uf.components == baseline.components
        else:
            assert uf.connected(a, b) == baseline.connected(a, b)

    assert uf.groups() == baseline.groups()
    print(
        "randomized check passed: "
        f"seed={seed}, n={n}, rounds={rounds}, components={uf.components}"
    )


def main() -> None:
    deterministic_demo()
    randomized_cross_check()


if __name__ == "__main__":
    main()
