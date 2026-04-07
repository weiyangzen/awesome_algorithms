"""Minimal runnable MVP for square-root decomposition (分块/平方分解).

Supported operations:
- range_add(l, r, delta): add delta to all elements in [l, r]
- range_sum(l, r): sum of elements in [l, r]
- point_set(i, value): set a[i] = value

Run:
    uv run python demo.py
"""

from __future__ import annotations

import math
import random


class SqrtDecomposition:
    """Range add / range sum structure based on sqrt decomposition."""

    def __init__(self, arr: list[int]) -> None:
        if not arr:
            raise ValueError("input array must be non-empty")

        self.n = len(arr)
        self.block_size = max(1, int(math.sqrt(self.n)))
        self.num_blocks = (self.n + self.block_size - 1) // self.block_size

        self.data = list(arr)
        self.block_sum = [0] * self.num_blocks
        self.lazy = [0] * self.num_blocks

        for i, v in enumerate(self.data):
            self.block_sum[self._block_id(i)] += v

    def _block_id(self, idx: int) -> int:
        return idx // self.block_size

    def _block_left(self, bid: int) -> int:
        return bid * self.block_size

    def _block_right(self, bid: int) -> int:
        return min(self.n - 1, (bid + 1) * self.block_size - 1)

    def _block_len(self, bid: int) -> int:
        return self._block_right(bid) - self._block_left(bid) + 1

    def _rebuild_block(self, bid: int) -> None:
        left = self._block_left(bid)
        right = self._block_right(bid)
        self.block_sum[bid] = sum(self.data[left : right + 1])

    def _push(self, bid: int) -> None:
        tag = self.lazy[bid]
        if tag == 0:
            return
        left = self._block_left(bid)
        right = self._block_right(bid)
        for i in range(left, right + 1):
            self.data[i] += tag
        self.lazy[bid] = 0

    @staticmethod
    def _check_int(x: int, name: str) -> None:
        if not isinstance(x, int):
            raise TypeError(f"{name} must be int, got {type(x).__name__}")

    def _check_index(self, idx: int) -> None:
        self._check_int(idx, "idx")
        if idx < 0 or idx >= self.n:
            raise IndexError(f"index out of range: {idx}")

    def _check_range(self, left: int, right: int) -> None:
        self._check_index(left)
        self._check_index(right)
        if left > right:
            raise IndexError(f"invalid range: left={left}, right={right}")

    def range_add(self, left: int, right: int, delta: int) -> None:
        self._check_range(left, right)
        self._check_int(delta, "delta")

        bl = self._block_id(left)
        br = self._block_id(right)

        if bl == br:
            self._push(bl)
            for i in range(left, right + 1):
                self.data[i] += delta
            self._rebuild_block(bl)
            return

        self._push(bl)
        left_end = self._block_right(bl)
        for i in range(left, left_end + 1):
            self.data[i] += delta
        self._rebuild_block(bl)

        self._push(br)
        right_start = self._block_left(br)
        for i in range(right_start, right + 1):
            self.data[i] += delta
        self._rebuild_block(br)

        for bid in range(bl + 1, br):
            self.lazy[bid] += delta
            self.block_sum[bid] += delta * self._block_len(bid)

    def range_sum(self, left: int, right: int) -> int:
        self._check_range(left, right)

        bl = self._block_id(left)
        br = self._block_id(right)

        if bl == br:
            self._push(bl)
            return sum(self.data[left : right + 1])

        total = 0

        self._push(bl)
        left_end = self._block_right(bl)
        total += sum(self.data[left : left_end + 1])

        self._push(br)
        right_start = self._block_left(br)
        total += sum(self.data[right_start : right + 1])

        for bid in range(bl + 1, br):
            total += self.block_sum[bid]

        return total

    def point_set(self, idx: int, value: int) -> None:
        self._check_index(idx)
        self._check_int(value, "value")

        bid = self._block_id(idx)
        self._push(bid)
        delta = value - self.data[idx]
        self.data[idx] = value
        self.block_sum[bid] += delta

    def materialize(self) -> list[int]:
        for bid in range(self.num_blocks):
            self._push(bid)
        return list(self.data)


def run_demo() -> None:
    arr = [2, -1, 3, 5, 0, 4, -2, 7, 1, 6, -3, 8]
    sd = SqrtDecomposition(arr)

    print("=== Sqrt Decomposition Demo ===")
    print("Initial array:", arr)

    got0 = sd.range_sum(0, len(arr) - 1)
    exp0 = sum(arr)
    print(f"Initial range_sum(0, {len(arr)-1}) = {got0}, expected = {exp0}")
    assert got0 == exp0

    sd.range_add(2, 9, 3)
    for i in range(2, 10):
        arr[i] += 3
    got1 = sd.range_sum(3, 8)
    exp1 = sum(arr[3:9])
    print(f"After range_add(2, 9, 3), range_sum(3, 8) = {got1}, expected = {exp1}")
    assert got1 == exp1

    sd.point_set(5, 100)
    arr[5] = 100
    got2 = sd.range_sum(0, 11)
    exp2 = sum(arr)
    print(f"After point_set(5, 100), range_sum(0, 11) = {got2}, expected = {exp2}")
    assert got2 == exp2

    materialized = sd.materialize()
    print("Materialized array:", materialized)
    assert materialized == arr

    rng = random.Random(20260407)
    n_random = 64
    brute = [rng.randint(-50, 50) for _ in range(n_random)]
    sd2 = SqrtDecomposition(brute)

    ops = 200
    for _ in range(ops):
        op = rng.choice(["add", "sum", "set"])
        l = rng.randrange(n_random)
        r = rng.randrange(n_random)
        if l > r:
            l, r = r, l

        if op == "add":
            delta = rng.randint(-20, 20)
            sd2.range_add(l, r, delta)
            for i in range(l, r + 1):
                brute[i] += delta
        elif op == "sum":
            got = sd2.range_sum(l, r)
            exp = sum(brute[l : r + 1])
            assert got == exp, (
                f"Mismatch on sum({l},{r}): got={got}, exp={exp}"
            )
        else:
            idx = rng.randrange(n_random)
            value = rng.randint(-100, 100)
            sd2.point_set(idx, value)
            brute[idx] = value

    assert sd2.materialize() == brute
    print(f"Random cross-check passed ({ops} operations).")
    print("All assertions passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
