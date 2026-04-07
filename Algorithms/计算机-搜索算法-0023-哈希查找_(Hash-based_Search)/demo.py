"""Hash-based Search MVP demo.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Hashable, Sequence


@dataclass
class Entry:
    """One chain-node entry in a bucket."""

    key: Hashable
    indices: list[int]


class HashTableIndex:
    """A minimal chained hash index: key -> list of positions in the original array."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.buckets: list[list[Entry]] = [[] for _ in range(capacity)]
        self.size = 0

    def _slot(self, key: Hashable) -> int:
        try:
            return hash(key) % self.capacity
        except TypeError as exc:
            raise TypeError(f"Key must be hashable, got {key!r}") from exc

    def insert(self, key: Hashable, index: int) -> None:
        """Insert one (key, index). If key exists, append index to its posting list."""
        bucket = self.buckets[self._slot(key)]
        for entry in bucket:
            if entry.key == key:
                entry.indices.append(index)
                return
        bucket.append(Entry(key=key, indices=[index]))
        self.size += 1

    def find_first(self, key: Hashable) -> int:
        """Return first position of key, or -1 if key is absent."""
        bucket = self.buckets[self._slot(key)]
        for entry in bucket:
            if entry.key == key:
                return entry.indices[0]
        return -1

    def find_all(self, key: Hashable) -> list[int]:
        """Return all positions of key, or [] if absent."""
        bucket = self.buckets[self._slot(key)]
        for entry in bucket:
            if entry.key == key:
                return list(entry.indices)
        return []

    def bucket_stats(self) -> tuple[int, int, int]:
        """Return (non_empty_bucket_count, max_bucket_chain_length, key_count)."""
        non_empty = 0
        max_chain = 0
        for bucket in self.buckets:
            if bucket:
                non_empty += 1
                max_chain = max(max_chain, len(bucket))
        return non_empty, max_chain, self.size


def _next_power_of_two(x: int) -> int:
    value = 1
    while value < x:
        value <<= 1
    return value


def build_hash_index(arr: Sequence[Hashable]) -> HashTableIndex:
    """Build a hash index from arr."""
    capacity = _next_power_of_two(max(8, 2 * len(arr)))
    index = HashTableIndex(capacity=capacity)
    for i, value in enumerate(arr):
        index.insert(value, i)
    return index


def hash_search_first(arr: Sequence[Hashable], target: Hashable) -> int:
    """One-shot query helper: build index then query first index."""
    return build_hash_index(arr).find_first(target)


def _linear_first(arr: Sequence[Hashable], target: Hashable) -> int:
    for i, value in enumerate(arr):
        if value == target:
            return i
    return -1


def _linear_all(arr: Sequence[Hashable], target: Hashable) -> list[int]:
    return [i for i, value in enumerate(arr) if value == target]


def main() -> None:
    cases = [
        ([], 3),
        ([5], 5),
        ([5], 7),
        ([42, 7, 13, 7, 99], 7),
        ([42, 7, 13, 7, 99], 100),
        ([0, -1, -1, 3, 8, 8, 8], -1),
        ([0, -1, -1, 3, 8, 8, 8], 8),
        ([10, 20, 30, 40, 50], 30),
        ([10, 20, 30, 40, 50], -10),
        (["aa", "bb", "aa", "cc"], "aa"),
        (["aa", "bb", "aa", "cc"], "dd"),
    ]

    print("Hash-based Search MVP demo")
    print("-" * 40)

    for idx, (arr, target) in enumerate(cases, start=1):
        hash_index = build_hash_index(arr)
        first = hash_index.find_first(target)
        all_pos = hash_index.find_all(target)

        expected_first = _linear_first(arr, target)
        expected_all = _linear_all(arr, target)

        if first != expected_first:
            raise AssertionError(
                f"Case {idx}: first={first}, expected_first={expected_first}, arr={arr}, target={target}"
            )
        if all_pos != expected_all:
            raise AssertionError(
                f"Case {idx}: all_pos={all_pos}, expected_all={expected_all}, arr={arr}, target={target}"
            )

        non_empty, max_chain, key_count = hash_index.bucket_stats()
        print(
            f"Case {idx:02d}: target={target!r}, first={first}, all={all_pos}, "
            f"keys={key_count}, non_empty_buckets={non_empty}, max_chain={max_chain}"
        )

    # Repeated-query scenario: build once, query many times.
    rng = Random(2026)
    arr = [rng.randint(0, 50) for _ in range(200)]
    queries = [rng.randint(0, 55) for _ in range(120)]

    shared_index = build_hash_index(arr)
    for q in queries:
        got = shared_index.find_first(q)
        expect = _linear_first(arr, q)
        if got != expect:
            raise AssertionError(f"Batch query mismatch: q={q}, got={got}, expect={expect}")

    # Explicitly verify TypeError on unhashable key.
    try:
        hash_search_first([[1], [2]], [1])
        raise AssertionError("Unhashable-key check failed: TypeError was expected")
    except TypeError:
        print("Unhashable key check: passed (TypeError raised as expected)")

    print("Batch query check: passed")
    print("All checks passed.")


if __name__ == "__main__":
    main()
