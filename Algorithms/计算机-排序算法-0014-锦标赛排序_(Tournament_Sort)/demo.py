"""Tournament Sort MVP.

This script implements tournament sort explicitly (winner tree), not by calling
third-party sorting routines as the algorithm itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


Key = Tuple[float, int]


@dataclass
class TournamentSortResult:
    sorted_values: List[float]
    comparisons: int
    path_updates: int


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class TournamentTreeSorter:
    """Winner-tree based tournament sorter (stable by original index)."""

    def __init__(self, values: Sequence[float]) -> None:
        self.values = [float(v) for v in values]
        self.n = len(self.values)
        self.m = next_power_of_two(self.n)
        self.base = self.m
        self.tree: List[Key] = [(float("inf"), 10**18)] * (2 * self.m)
        self.sentinel: Key = (float("inf"), 10**18)
        self.index_to_leaf: Dict[int, int] = {}

        self.comparisons = 0
        self.path_updates = 0

        self._initialize_leaves()
        self._build_tree()

    def _better(self, a: Key, b: Key) -> Key:
        self.comparisons += 1
        if a[0] < b[0]:
            return a
        if a[0] > b[0]:
            return b
        return a if a[1] <= b[1] else b

    def _initialize_leaves(self) -> None:
        for i in range(self.m):
            pos = self.base + i
            if i < self.n:
                key = (self.values[i], i)
                self.tree[pos] = key
                self.index_to_leaf[i] = pos
            else:
                self.tree[pos] = self.sentinel

    def _build_tree(self) -> None:
        for node in range(self.base - 1, 0, -1):
            left = self.tree[2 * node]
            right = self.tree[2 * node + 1]
            self.tree[node] = self._better(left, right)

    def _recompute_path(self, leaf_pos: int) -> None:
        node = leaf_pos // 2
        while node >= 1:
            left = self.tree[2 * node]
            right = self.tree[2 * node + 1]
            self.tree[node] = self._better(left, right)
            self.path_updates += 1
            node //= 2

    def sort(self) -> TournamentSortResult:
        if self.n == 0:
            return TournamentSortResult(sorted_values=[], comparisons=0, path_updates=0)

        output: List[float] = []
        for _ in range(self.n):
            winner = self.tree[1]
            winner_value, winner_idx = winner

            output.append(winner_value)

            leaf_pos = self.index_to_leaf[winner_idx]
            self.tree[leaf_pos] = self.sentinel
            self._recompute_path(leaf_pos)

        return TournamentSortResult(
            sorted_values=output,
            comparisons=self.comparisons,
            path_updates=self.path_updates,
        )


def run_case(case_name: str, data: Sequence[float]) -> bool:
    sorter = TournamentTreeSorter(data)
    result = sorter.sort()
    expected = sorted(float(x) for x in data)
    ok = result.sorted_values == expected

    print(f"\n[{case_name}]")
    print(f"input           : {list(float(x) for x in data)}")
    print(f"tournament_sort : {result.sorted_values}")
    print(f"python_sorted   : {expected}")
    print(f"match           : {ok}")
    print(f"comparisons     : {result.comparisons}")
    print(f"path_updates    : {result.path_updates}")
    return ok


def main() -> None:
    rng = np.random.default_rng(2026)

    fixed_case = [5, 3, 7, 3, -1, 4, 0, -1, 9]
    random_case = rng.integers(low=-20, high=21, size=15).tolist()
    empty_case: List[float] = []
    single_case = [42.0]

    checks = [
        run_case("fixed_with_duplicates", fixed_case),
        run_case("random_integers", random_case),
        run_case("empty", empty_case),
        run_case("single", single_case),
    ]

    print("\n=== Summary ===")
    print(f"all_cases_passed={all(checks)}")
    print(f"num_cases={len(checks)}")


if __name__ == "__main__":
    main()
