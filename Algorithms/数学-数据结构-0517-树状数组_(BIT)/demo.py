"""Fenwick Tree (Binary Indexed Tree) minimal runnable MVP.

Features:
1) O(n) linear build
2) O(log n) point update
3) O(log n) prefix/range sum query
4) O(log n) index lookup by prefix sum (order statistics)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FenwickTree:
    """1-based Fenwick Tree for additive aggregation on int64 values."""

    n: int

    def __init__(self, values_1based: np.ndarray) -> None:
        vals = np.asarray(values_1based, dtype=np.int64)
        if vals.ndim != 1:
            raise ValueError("values_1based must be a 1-D array")

        self.n = int(vals.shape[0])
        self.tree = np.zeros(self.n + 1, dtype=np.int64)
        self._build_linear(vals)

    @staticmethod
    def _lowbit(x: int) -> int:
        return x & -x

    def _build_linear(self, vals_1based: np.ndarray) -> None:
        """O(n) build from a 1-based logical array passed as 0-based ndarray."""
        self.tree[1:] = vals_1based
        for i in range(1, self.n + 1):
            parent = i + self._lowbit(i)
            if parent <= self.n:
                self.tree[parent] += self.tree[i]

    def add(self, index_1based: int, delta: int) -> None:
        if index_1based < 1 or index_1based > self.n:
            raise IndexError(f"index_1based out of range: {index_1based}")

        i = index_1based
        d = np.int64(delta)
        while i <= self.n:
            self.tree[i] += d
            i += self._lowbit(i)

    def prefix_sum(self, index_1based: int) -> int:
        if index_1based <= 0:
            return 0
        if index_1based > self.n:
            index_1based = self.n

        s = np.int64(0)
        i = index_1based
        while i > 0:
            s += self.tree[i]
            i -= self._lowbit(i)
        return int(s)

    def range_sum(self, left_1based: int, right_1based: int) -> int:
        if left_1based > right_1based:
            return 0
        if right_1based < 1 or left_1based > self.n:
            return 0

        l = max(1, left_1based)
        r = min(self.n, right_1based)
        return self.prefix_sum(r) - self.prefix_sum(l - 1)

    def find_by_prefix(self, target: int) -> int:
        """Return smallest idx such that prefix_sum(idx) >= target, else n+1.

        Requires non-negative values for meaningful monotonic prefix sums.
        """
        if target <= 0:
            return 1

        total = self.prefix_sum(self.n)
        if target > total:
            return self.n + 1

        idx = 0
        acc = np.int64(0)

        bit = 1
        while (bit << 1) <= self.n:
            bit <<= 1

        while bit > 0:
            nxt = idx + bit
            if nxt <= self.n and acc + self.tree[nxt] < target:
                acc += self.tree[nxt]
                idx = nxt
            bit >>= 1

        return idx + 1


def naive_range_sum(arr_1based: np.ndarray, left_1based: int, right_1based: int) -> int:
    if left_1based > right_1based:
        return 0
    l = max(1, left_1based)
    r = min(arr_1based.shape[0], right_1based)
    if l > r:
        return 0
    return int(arr_1based[l - 1 : r].sum())


def main() -> None:
    rng = np.random.default_rng(2026)

    n = 20
    base = rng.integers(0, 16, size=n, dtype=np.int64)
    bit = FenwickTree(base)

    print("Fenwick Tree (BIT) MVP")
    print(f"n={n}")
    print("initial array:", base.tolist())

    # 1) Validate prefix sums against NumPy baseline.
    prefix_np = np.cumsum(base)
    for i in range(1, n + 1):
        got = bit.prefix_sum(i)
        exp = int(prefix_np[i - 1])
        assert got == exp, f"prefix mismatch at i={i}: got {got}, expected {exp}"

    # 2) Random mixed updates + range queries.
    for step in range(1, 121):
        if rng.random() < 0.45:
            idx = int(rng.integers(1, n + 1))
            delta = int(rng.integers(-5, 8))
            bit.add(idx, delta)
            base[idx - 1] += delta
        else:
            l = int(rng.integers(1, n + 1))
            r = int(rng.integers(1, n + 1))
            if l > r:
                l, r = r, l
            got = bit.range_sum(l, r)
            exp = naive_range_sum(base, l, r)
            assert got == exp, f"range mismatch at step={step}, [{l},{r}] got={got}, expected={exp}"

    # 3) Validate order-statistics lookup on non-negative frequencies.
    freq = rng.integers(0, 8, size=15, dtype=np.int64)
    bit_freq = FenwickTree(freq)
    total = int(freq.sum())
    csum = np.cumsum(freq)

    for t in [1, max(1, total // 3), max(1, total // 2), max(1, total - 1), total]:
        idx = bit_freq.find_by_prefix(t)
        exp = int(np.searchsorted(csum, t, side="left") + 1)
        assert idx == exp, f"find_by_prefix mismatch: t={t}, got={idx}, expected={exp}"

    miss_idx = bit_freq.find_by_prefix(total + 1)
    assert miss_idx == bit_freq.n + 1, "target > total should return n+1"

    print("all checks passed")
    print("final array:", base.tolist())
    print("sample queries:")
    for l, r in [(1, 5), (4, 10), (8, n), (1, n)]:
        print(f"  range_sum({l},{r}) = {bit.range_sum(l, r)}")


if __name__ == "__main__":
    main()
