"""Minimal runnable MVP for Sparse Table (static RMQ with O(1) query)."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import List


@dataclass
class SparseTableMin:
    """Sparse table for static range minimum query on closed interval [l, r]."""

    data: List[int]

    def __post_init__(self) -> None:
        if len(self.data) == 0:
            raise ValueError("data must be non-empty")
        self.n = len(self.data)
        self.log2 = self._build_log_table(self.n)
        self.st = self._build_sparse_table(self.data)

    @staticmethod
    def _build_log_table(n: int) -> List[int]:
        log2 = [0] * (n + 1)
        for x in range(2, n + 1):
            log2[x] = log2[x // 2] + 1
        return log2

    def _build_sparse_table(self, arr: List[int]) -> List[List[int]]:
        st: List[List[int]] = [arr.copy()]
        max_k = self.log2[self.n]

        for k in range(1, max_k + 1):
            span = 1 << k
            half = span >> 1
            width = self.n - span + 1
            prev = st[k - 1]
            row = [0] * width
            for i in range(width):
                left = prev[i]
                right = prev[i + half]
                row[i] = left if left <= right else right
            st.append(row)
        return st

    def _check_query(self, l: int, r: int) -> None:
        if not (0 <= l <= r < self.n):
            raise IndexError(f"invalid query [{l}, {r}] for n={self.n}")

    def range_min(self, l: int, r: int) -> int:
        self._check_query(l, r)
        length = r - l + 1
        k = self.log2[length]
        span = 1 << k
        left = self.st[k][l]
        right = self.st[k][r - span + 1]
        return left if left <= right else right


def run_deterministic_case() -> None:
    data = [7, 2, 3, 0, 5, 10, 3, 12, 18, -1, 4, 4]
    queries = [
        (0, 0),
        (0, 3),
        (2, 7),
        (3, 3),
        (4, 11),
        (0, 11),
        (8, 10),
    ]

    st = SparseTableMin(data)
    print("Deterministic case:")
    print("data =", data)
    for l, r in queries:
        got = st.range_min(l, r)
        expect = min(data[l : r + 1])
        print(f"min[{l},{r}] -> st={got}, brute={expect}")
        assert got == expect


def run_randomized_regression() -> None:
    rng = Random(20260407)
    num_cases = 80
    queries_per_case = 150

    for _ in range(num_cases):
        n = rng.randint(1, 120)
        data = [rng.randint(-1000, 1000) for _ in range(n)]
        st = SparseTableMin(data)

        for _ in range(queries_per_case):
            l = rng.randint(0, n - 1)
            r = rng.randint(l, n - 1)
            got = st.range_min(l, r)
            expect = min(data[l : r + 1])
            assert got == expect

    print(
        "Randomized regression passed:",
        f"{num_cases} arrays x {queries_per_case} queries each.",
    )


def main() -> None:
    run_deterministic_case()
    run_randomized_regression()
    print("Sparse table MVP finished successfully.")


if __name__ == "__main__":
    main()
