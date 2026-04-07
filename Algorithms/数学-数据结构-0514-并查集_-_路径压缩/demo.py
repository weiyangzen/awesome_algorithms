"""Union-Find (Disjoint Set Union) demo focused on path compression.

Run:
    uv run python demo.py
"""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np


class UnionFindPathCompression:
    """Disjoint Set Union using path compression in find()."""

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self.n = n
        self.parent: np.ndarray = np.arange(n, dtype=np.int64)
        self.comp_size: np.ndarray = np.ones(n, dtype=np.int64)
        self.components = n

    def _check_index(self, x: int) -> None:
        if not 0 <= x < self.n:
            raise IndexError(f"index out of range: {x}")

    def _root_no_compress(self, x: int) -> int:
        self._check_index(x)
        while int(self.parent[x]) != x:
            x = int(self.parent[x])
        return x

    def depth_no_compress(self, x: int) -> int:
        """Return path length from x to root without mutating parent."""
        self._check_index(x)
        depth = 0
        while int(self.parent[x]) != x:
            x = int(self.parent[x])
            depth += 1
        return depth

    def max_depth_no_compress(self) -> int:
        if self.n == 0:
            return 0
        return max(self.depth_no_compress(i) for i in range(self.n))

    def find(self, x: int) -> int:
        self._check_index(x)
        path: List[int] = []
        curr = x
        while int(self.parent[curr]) != curr:
            path.append(curr)
            curr = int(self.parent[curr])
        root = curr
        for node in path:
            self.parent[node] = root
        return root

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False

        # Deliberately simple linking to keep the demo centered on path compression.
        self.parent[ra] = rb
        self.comp_size[rb] += self.comp_size[ra]
        self.components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def size(self, x: int) -> int:
        root = self.find(x)
        return int(self.comp_size[root])

    def groups(self) -> List[List[int]]:
        buckets: Dict[int, List[int]] = {}
        for i in range(self.n):
            root = self.find(i)
            buckets.setdefault(root, []).append(i)
        return sorted((sorted(group) for group in buckets.values()), key=lambda g: g[0])

    def parent_snapshot(self) -> List[int]:
        return [int(v) for v in self.parent.tolist()]


class NaiveDisjointSet:
    """Slow baseline for correctness cross-check."""

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
            if self.label[i] == la:
                self.label[i] = lb
        self.components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def groups(self) -> List[List[int]]:
        buckets: Dict[int, List[int]] = {}
        for i, lab in enumerate(self.label):
            buckets.setdefault(lab, []).append(i)
        return sorted((sorted(group) for group in buckets.values()), key=lambda g: g[0])


def deterministic_demo() -> None:
    print("=== Deterministic demo: path compression ===")
    uf = UnionFindPathCompression(10)
    operations = [(i, i + 1) for i in range(9)]

    for step, (a, b) in enumerate(operations, start=1):
        merged = uf.union(a, b)
        print(
            f"step={step:02d} union({a}, {b}) -> merged={merged:<5} "
            f"components={uf.components}"
        )

    print(f"parent before compression = {uf.parent_snapshot()}")
    depth_before = uf.max_depth_no_compress()
    root0 = uf.find(0)
    depth_after = uf.max_depth_no_compress()
    print(f"find(0) -> root={root0}")
    print(f"max depth before={depth_before}, after one find(0)={depth_after}")
    print(f"parent after compression  = {uf.parent_snapshot()}")
    print(f"connected(0, 9) = {uf.connected(0, 9)}")
    print(f"size(3) = {uf.size(3)}")
    print(f"groups = {uf.groups()}")


def randomized_cross_check(seed: int = 2026, n: int = 30, rounds: int = 500) -> None:
    print("\n=== Randomized cross-check against naive baseline ===")
    rng = random.Random(seed)
    uf = UnionFindPathCompression(n)
    baseline = NaiveDisjointSet(n)

    for _ in range(rounds):
        a = rng.randrange(n)
        b = rng.randrange(n)
        if rng.random() < 0.65:
            merged_uf = uf.union(a, b)
            merged_baseline = baseline.union(a, b)
            assert merged_uf == merged_baseline
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
