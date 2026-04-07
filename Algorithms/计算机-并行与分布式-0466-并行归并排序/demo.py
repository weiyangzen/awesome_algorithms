"""并行归并排序最小可运行 MVP.

运行:
    uv run python demo.py
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import numpy as np


def merge_two_sorted_lists(left: list[int], right: list[int]) -> list[int]:
    """将两个有序数组线性归并为一个有序数组."""
    merged: list[int] = []
    i = 0
    j = 0
    left_len = len(left)
    right_len = len(right)

    while i < left_len and j < right_len:
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    if i < left_len:
        merged.extend(left[i:])
    if j < right_len:
        merged.extend(right[j:])
    return merged


def sequential_merge_sort(values: list[int]) -> list[int]:
    """标准递归归并排序，用作基线和并行子任务."""
    n = len(values)
    if n <= 1:
        return values[:]

    mid = n // 2
    left_sorted = sequential_merge_sort(values[:mid])
    right_sorted = sequential_merge_sort(values[mid:])
    return merge_two_sorted_lists(left_sorted, right_sorted)


def _merge_pair(pair: tuple[list[int], list[int]]) -> list[int]:
    """给进程池使用的二路归并包装函数."""
    return merge_two_sorted_lists(pair[0], pair[1])


def _split_into_chunks(values: list[int], num_chunks: int) -> list[list[int]]:
    n = len(values)
    if num_chunks <= 1 or n == 0:
        return [values[:]]

    chunk_size = math.ceil(n / num_chunks)
    chunks: list[list[int]] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append(values[start:end])
    return chunks


def parallel_merge_sort(
    values: list[int],
    max_workers: int | None = None,
    min_chunk_size: int = 25_000,
) -> list[int]:
    """并行归并排序.

    策略:
    1) 先把输入拆成多个块并行做顺序归并排序;
    2) 再按层做并行二路归并，直到只剩一个块。
    """
    n = len(values)
    if n <= 1:
        return values[:]

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) - 1)
    max_workers = max(1, max_workers)

    if max_workers == 1 or n < 2 * min_chunk_size:
        return sequential_merge_sort(values)

    num_chunks = min(max_workers, math.ceil(n / min_chunk_size))
    num_chunks = max(2, num_chunks)
    chunks = _split_into_chunks(values, num_chunks)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        sorted_chunks = list(pool.map(sequential_merge_sort, chunks))

        while len(sorted_chunks) > 1:
            pair_tasks: list[tuple[list[int], list[int]]] = []
            carry: list[int] | None = None

            idx = 0
            total = len(sorted_chunks)
            while idx < total:
                if idx + 1 < total:
                    pair_tasks.append((sorted_chunks[idx], sorted_chunks[idx + 1]))
                    idx += 2
                else:
                    carry = sorted_chunks[idx]
                    idx += 1

            merged_chunks = list(pool.map(_merge_pair, pair_tasks))
            if carry is not None:
                merged_chunks.append(carry)
            sorted_chunks = merged_chunks

    return sorted_chunks[0]


def _assert_sorted(values: list[int]) -> None:
    for i in range(1, len(values)):
        if values[i - 1] > values[i]:
            raise AssertionError(f"数组未排序: index={i - 1} > {i}")


def _self_test() -> None:
    cases = [
        [],
        [7],
        [3, 1, 2],
        [5, -2, 5, 0, -2, 9, 9, 1],
        list(range(30, -1, -1)),
    ]
    for case in cases:
        seq = sequential_merge_sort(case)
        par = parallel_merge_sort(case, max_workers=2, min_chunk_size=4)
        expected = sorted(case)
        if seq != expected or par != expected:
            raise AssertionError("自测失败: 结果与 sorted(case) 不一致")
    print("[self-test] passed")


@dataclass
class BenchmarkResult:
    name: str
    seconds: float


def _benchmark(name: str, fn, data: list[int]) -> tuple[list[int], BenchmarkResult]:
    start = perf_counter()
    result = fn(data)
    elapsed = perf_counter() - start
    return result, BenchmarkResult(name=name, seconds=elapsed)


def main() -> None:
    _self_test()

    rng = np.random.default_rng(seed=20260407)
    n = 150_000
    data = rng.integers(0, 2_000_000, size=n, dtype=np.int64).tolist()

    workers = min(4, os.cpu_count() or 1)
    min_chunk_size = 20_000
    print(
        f"[config] n={n}, workers={workers}, min_chunk_size={min_chunk_size}, "
        f"python={os.sys.version.split()[0]}"
    )

    seq_sorted, seq_metric = _benchmark("sequential_merge_sort", sequential_merge_sort, data)
    par_sorted, par_metric = _benchmark(
        "parallel_merge_sort",
        lambda arr: parallel_merge_sort(arr, max_workers=workers, min_chunk_size=min_chunk_size),
        data,
    )

    _assert_sorted(seq_sorted)
    _assert_sorted(par_sorted)
    if seq_sorted != par_sorted:
        raise AssertionError("并行与顺序结果不一致")

    speedup = seq_metric.seconds / par_metric.seconds if par_metric.seconds > 0 else float("inf")
    print(f"[time] {seq_metric.name}: {seq_metric.seconds:.4f}s")
    print(f"[time] {par_metric.name}: {par_metric.seconds:.4f}s")
    print(f"[time] speedup(sequential/parallel): {speedup:.3f}x")
    print("[check] sorted outputs are identical")


if __name__ == "__main__":
    main()
