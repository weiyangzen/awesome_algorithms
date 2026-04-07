"""Minimal runnable MVP for Mo's algorithm (MATH-0519).

This demo solves offline range-distinct queries on a static array.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class Query:
    """Closed interval query [l, r], 0-based."""

    l: int
    r: int


@dataclass(frozen=True)
class _OrderedQuery:
    """Internal query record with index and block metadata."""

    l: int
    r: int
    idx: int
    block: int


def _compress_values(arr: Sequence[int]) -> List[int]:
    """Coordinate-compress values in arr to [0, unique_count)."""
    uniq = sorted(set(arr))
    rank: Dict[int, int] = {v: i for i, v in enumerate(uniq)}
    return [rank[v] for v in arr]


def mo_distinct_count(arr: Sequence[int], queries: Sequence[Query]) -> List[int]:
    """Answer offline distinct-count queries using Mo's algorithm.

    Args:
        arr: Static integer array.
        queries: Query list, each as closed interval [l, r].

    Returns:
        Distinct count for each query in original query order.
    """
    n = len(arr)
    q = len(queries)
    if q == 0:
        return []
    if n == 0:
        raise ValueError("arr cannot be empty when queries are non-empty")

    for i, qu in enumerate(queries):
        if qu.l < 0 or qu.r < 0 or qu.l > qu.r or qu.r >= n:
            raise ValueError(f"invalid query at index {i}: [{qu.l}, {qu.r}]")

    block_size = max(1, int(math.sqrt(n)))
    ordered: List[_OrderedQuery] = [
        _OrderedQuery(l=qu.l, r=qu.r, idx=i, block=qu.l // block_size)
        for i, qu in enumerate(queries)
    ]

    # Odd-even ordering reduces pointer oscillation on R.
    ordered.sort(key=lambda x: (x.block, x.r if (x.block % 2 == 0) else -x.r))

    compressed = _compress_values(arr)
    freq = [0] * len(set(arr))
    ans = [0] * q

    cur_l, cur_r = 0, -1
    distinct = 0

    def add(pos: int) -> int:
        nonlocal distinct
        v = compressed[pos]
        if freq[v] == 0:
            distinct += 1
        freq[v] += 1
        return distinct

    def remove(pos: int) -> int:
        nonlocal distinct
        v = compressed[pos]
        freq[v] -= 1
        if freq[v] == 0:
            distinct -= 1
        return distinct

    for qu in ordered:
        while cur_l > qu.l:
            cur_l -= 1
            add(cur_l)
        while cur_r < qu.r:
            cur_r += 1
            add(cur_r)
        while cur_l < qu.l:
            remove(cur_l)
            cur_l += 1
        while cur_r > qu.r:
            remove(cur_r)
            cur_r -= 1
        ans[qu.idx] = distinct

    return ans


def naive_distinct_count(arr: Sequence[int], queries: Sequence[Query]) -> List[int]:
    """Baseline O(Q*N) solver for verification."""
    out: List[int] = []
    for qu in queries:
        out.append(len(set(arr[qu.l : qu.r + 1])))
    return out


def _fixed_demo() -> None:
    arr = [1, 1, 2, 1, 3, 4, 5, 2, 8]
    queries = [
        Query(0, 4),
        Query(1, 3),
        Query(2, 4),
        Query(0, 8),
        Query(3, 7),
    ]

    mo_ans = mo_distinct_count(arr, queries)
    naive_ans = naive_distinct_count(arr, queries)

    print("Fixed example")
    print(f"arr = {arr}")
    for i, q in enumerate(queries):
        print(
            f"  q{i}: [{q.l}, {q.r}] -> mo={mo_ans[i]}, naive={naive_ans[i]}"
        )
    assert mo_ans == naive_ans


def _random_regression(seed: int = 519) -> None:
    rng = random.Random(seed)
    n = 120
    q = 220
    arr = [rng.randint(0, 40) for _ in range(n)]

    queries: List[Query] = []
    for _ in range(q):
        l = rng.randint(0, n - 1)
        r = rng.randint(l, n - 1)
        queries.append(Query(l, r))

    mo_ans = mo_distinct_count(arr, queries)
    naive_ans = naive_distinct_count(arr, queries)
    assert mo_ans == naive_ans, "Mo result mismatch against naive baseline"

    print("Random regression")
    print(f"  n={n}, q={q}, seed={seed}")
    print("  check: mo_distinct_count == naive_distinct_count")


def main() -> None:
    print("Mo's Algorithm MVP (MATH-0519)")
    print("=" * 64)
    _fixed_demo()
    print("=" * 64)
    _random_regression()
    print("=" * 64)
    print("All checks passed.")


if __name__ == "__main__":
    main()
