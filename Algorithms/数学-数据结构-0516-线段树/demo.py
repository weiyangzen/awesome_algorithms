"""Minimal runnable MVP for segment tree (range sum + range add + point set)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SegmentTree:
    """Segment tree supporting:
    - range sum query on closed interval [l, r]
    - range add update on closed interval [l, r]
    - point set update
    """

    data: List[int]

    def __post_init__(self) -> None:
        if len(self.data) == 0:
            raise ValueError("data must be non-empty")
        self.n = len(self.data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(1, 0, self.n - 1)

    def _build(self, node: int, left: int, right: int) -> None:
        if left == right:
            self.tree[node] = int(self.data[left])
            return
        mid = (left + right) // 2
        self._build(node * 2, left, mid)
        self._build(node * 2 + 1, mid + 1, right)
        self._pull(node)

    def _pull(self, node: int) -> None:
        self.tree[node] = self.tree[node * 2] + self.tree[node * 2 + 1]

    def _apply(self, node: int, left: int, right: int, delta: int) -> None:
        self.tree[node] += delta * (right - left + 1)
        self.lazy[node] += delta

    def _push(self, node: int, left: int, right: int) -> None:
        if self.lazy[node] == 0 or left == right:
            return
        mid = (left + right) // 2
        delta = self.lazy[node]
        self._apply(node * 2, left, mid, delta)
        self._apply(node * 2 + 1, mid + 1, right, delta)
        self.lazy[node] = 0

    def _check_range(self, l: int, r: int) -> None:
        if not (0 <= l <= r < self.n):
            raise IndexError(f"invalid range [{l}, {r}] for n={self.n}")

    def _range_add(
        self,
        node: int,
        left: int,
        right: int,
        ql: int,
        qr: int,
        delta: int,
    ) -> None:
        if ql <= left and right <= qr:
            self._apply(node, left, right, delta)
            return

        self._push(node, left, right)
        mid = (left + right) // 2
        if ql <= mid:
            self._range_add(node * 2, left, mid, ql, qr, delta)
        if qr > mid:
            self._range_add(node * 2 + 1, mid + 1, right, ql, qr, delta)
        self._pull(node)

    def range_add(self, l: int, r: int, delta: int) -> None:
        self._check_range(l, r)
        self._range_add(1, 0, self.n - 1, l, r, int(delta))

    def _range_sum(
        self,
        node: int,
        left: int,
        right: int,
        ql: int,
        qr: int,
    ) -> int:
        if ql <= left and right <= qr:
            return self.tree[node]

        self._push(node, left, right)
        mid = (left + right) // 2
        total = 0
        if ql <= mid:
            total += self._range_sum(node * 2, left, mid, ql, qr)
        if qr > mid:
            total += self._range_sum(node * 2 + 1, mid + 1, right, ql, qr)
        return total

    def range_sum(self, l: int, r: int) -> int:
        self._check_range(l, r)
        return self._range_sum(1, 0, self.n - 1, l, r)

    def point_set(self, idx: int, value: int) -> None:
        if not (0 <= idx < self.n):
            raise IndexError(f"invalid idx={idx} for n={self.n}")
        current = self.range_sum(idx, idx)
        delta = int(value) - current
        self.range_add(idx, idx, delta)


def run_deterministic_case() -> None:
    rng = np.random.default_rng(2026)
    arr = rng.integers(low=-5, high=10, size=12).tolist()
    seg = SegmentTree(arr.copy())
    brute = arr.copy()

    print("Initial array:", arr)

    ops = [
        ("sum", 0, 5),
        ("add", 2, 7, 3),
        ("sum", 3, 8),
        ("set", 4, -11),
        ("sum", 0, 11),
        ("add", 0, 11, -2),
        ("sum", 4, 4),
        ("sum", 1, 10),
    ]

    for op in ops:
        kind = op[0]
        if kind == "sum":
            l, r = op[1], op[2]
            ans_tree = seg.range_sum(l, r)
            ans_brute = sum(brute[l : r + 1])
            print(f"sum[{l},{r}] -> tree={ans_tree}, brute={ans_brute}")
            assert ans_tree == ans_brute
        elif kind == "add":
            l, r, delta = op[1], op[2], op[3]
            seg.range_add(l, r, delta)
            for i in range(l, r + 1):
                brute[i] += delta
            print(f"add[{l},{r}] += {delta}")
        elif kind == "set":
            idx, value = op[1], op[2]
            seg.point_set(idx, value)
            brute[idx] = value
            print(f"set[{idx}] = {value}")
        else:
            raise RuntimeError("unknown operation")


def run_randomized_regression() -> None:
    rng = np.random.default_rng(99)
    n = 20
    arr = rng.integers(-20, 21, size=n).tolist()
    seg = SegmentTree(arr.copy())
    brute = arr.copy()

    for _ in range(200):
        mode = int(rng.integers(0, 3))
        l = int(rng.integers(0, n))
        r = int(rng.integers(l, n))

        if mode == 0:
            got = seg.range_sum(l, r)
            expect = sum(brute[l : r + 1])
            assert got == expect
        elif mode == 1:
            delta = int(rng.integers(-8, 9))
            seg.range_add(l, r, delta)
            for i in range(l, r + 1):
                brute[i] += delta
        else:
            idx = int(rng.integers(0, n))
            val = int(rng.integers(-30, 31))
            seg.point_set(idx, val)
            brute[idx] = val

    total_tree = seg.range_sum(0, n - 1)
    total_brute = sum(brute)
    assert total_tree == total_brute
    print("Randomized regression passed: all checks matched brute-force results.")


def main() -> None:
    run_deterministic_case()
    run_randomized_regression()
    print("SegmentTree MVP finished successfully.")


if __name__ == "__main__":
    main()
