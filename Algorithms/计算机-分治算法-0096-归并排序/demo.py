"""Minimal runnable MVP for merge sort (divide and conquer)."""

from __future__ import annotations

from random import Random
from time import perf_counter
from typing import Callable, Iterable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")
K = TypeVar("K")


def merge(left: Sequence[T], right: Sequence[T], key: Callable[[T], K]) -> List[T]:
    """Merge two sorted sequences into one sorted list.

    Stability is guaranteed by preferring the left element when keys are equal.
    """
    i = 0
    j = 0
    merged: List[T] = []

    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    if i < len(left):
        merged.extend(left[i:])
    if j < len(right):
        merged.extend(right[j:])

    return merged


def merge_sort(seq: Iterable[T], key: Callable[[T], K] | None = None) -> List[T]:
    """Return a new list containing all items from *seq* in sorted order."""
    items = list(seq)
    key_fn: Callable[[T], K]

    if key is None:
        key_fn = lambda x: x  # type: ignore[assignment]
    else:
        key_fn = key

    if len(items) <= 1:
        return items

    mid = len(items) // 2
    left_sorted = merge_sort(items[:mid], key=key_fn)
    right_sorted = merge_sort(items[mid:], key=key_fn)
    return merge(left_sorted, right_sorted, key_fn)


def run_basic_case() -> None:
    data = [38, 27, 43, 3, 9, 82, 10, 27, -5]
    result = merge_sort(data)

    print("[Case 1] 基础整数排序")
    print("原始:", data)
    print("结果:", result)

    expected = sorted(data)
    assert result == expected, "Basic case failed"


def run_stability_case() -> None:
    records: List[Tuple[str, int]] = [
        ("alice", 90),
        ("bob", 75),
        ("carol", 90),
        ("david", 75),
        ("eve", 90),
    ]

    by_score = merge_sort(records, key=lambda x: x[1])

    print("\n[Case 2] 稳定性验证（按分数升序）")
    print("原始:", records)
    print("结果:", by_score)

    # For equal score=75, bob should remain before david.
    # For equal score=90, alice should remain before carol before eve.
    assert [name for name, score in by_score if score == 75] == ["bob", "david"]
    assert [name for name, score in by_score if score == 90] == ["alice", "carol", "eve"]


def run_random_regression(seed: int = 7, rounds: int = 20) -> None:
    rng = Random(seed)

    for _ in range(rounds):
        n = rng.randint(0, 200)
        data = [rng.randint(-1000, 1000) for _ in range(n)]
        got = merge_sort(data)
        expected = sorted(data)
        assert got == expected, "Random regression failed"

    print(f"\n[Case 3] 随机回归测试通过: rounds={rounds}, seed={seed}")


def benchmark(size: int = 20_000, seed: int = 42) -> None:
    rng = Random(seed)
    data = [rng.randint(-1_000_000, 1_000_000) for _ in range(size)]

    t0 = perf_counter()
    out_merge = merge_sort(data)
    t1 = perf_counter()

    t2 = perf_counter()
    out_builtin = sorted(data)
    t3 = perf_counter()

    assert out_merge == out_builtin, "Benchmark correctness check failed"

    print("\n[Case 4] 小规模性能对照")
    print(f"n={size}")
    print(f"merge_sort: {t1 - t0:.6f}s")
    print(f"built-in sorted: {t3 - t2:.6f}s")


def main() -> None:
    run_basic_case()
    run_stability_case()
    run_random_regression()
    benchmark()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
