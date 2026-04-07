"""并行快速排序最小可运行 MVP.

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


def _choose_pivot(values: list[int]) -> int:
    """中位三数法选主元，降低极端输入下的不平衡概率."""
    first = values[0]
    middle = values[len(values) // 2]
    last = values[-1]
    trio = [first, middle, last]
    trio.sort()
    return trio[1]


def _partition_three_way(values: list[int], pivot: int) -> tuple[list[int], list[int], list[int]]:
    """按 pivot 进行三路划分."""
    less: list[int] = []
    equal: list[int] = []
    greater: list[int] = []
    for v in values:
        if v < pivot:
            less.append(v)
        elif v > pivot:
            greater.append(v)
        else:
            equal.append(v)
    return less, equal, greater


def sequential_quick_sort(values: list[int]) -> list[int]:
    """顺序快速排序，用作基线和并行子任务."""
    n = len(values)
    if n <= 1:
        return values[:]

    pivot = _choose_pivot(values)
    less, equal, greater = _partition_three_way(values, pivot)
    if len(equal) == n:
        return equal
    return sequential_quick_sort(less) + equal + sequential_quick_sort(greater)


def _expand_partitions(values: list[int], depth: int, min_partition_size: int) -> list[list[int]]:
    """在主进程内按有限深度展开快排分区树，得到有序拼接顺序的子问题列表."""
    n = len(values)
    if n <= 1:
        return [values[:]]
    if depth <= 0 or n < min_partition_size:
        return [values[:]]

    pivot = _choose_pivot(values)
    less, equal, greater = _partition_three_way(values, pivot)
    if len(equal) == n:
        return [equal]

    partitions: list[list[int]] = []
    if less:
        partitions.extend(_expand_partitions(less, depth - 1, min_partition_size))
    if equal:
        partitions.append(equal)
    if greater:
        partitions.extend(_expand_partitions(greater, depth - 1, min_partition_size))
    return partitions


def parallel_quick_sort(
    values: list[int],
    max_workers: int | None = None,
    min_partition_size: int = 25_000,
    partition_depth: int | None = None,
) -> list[int]:
    """并行快速排序.

    策略:
    1) 主进程按有限深度展开快排分区树，获得按大小关系有序的子分区列表;
    2) 进程池并行地对每个子分区执行顺序快排;
    3) 按分区顺序拼接结果得到全局有序数组。
    """
    n = len(values)
    if n <= 1:
        return values[:]

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) - 1)
    max_workers = max(1, max_workers)

    if max_workers == 1 or n < 2 * min_partition_size:
        return sequential_quick_sort(values)

    if partition_depth is None:
        partition_depth = max(1, math.ceil(math.log2(max_workers)))

    partitions = _expand_partitions(values, partition_depth, min_partition_size)
    if len(partitions) == 1:
        return sequential_quick_sort(values)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        sorted_parts = list(pool.map(sequential_quick_sort, partitions))

    merged: list[int] = []
    for part in sorted_parts:
        merged.extend(part)
    return merged


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
        [4, 4, 4, 4, 4, 4],
    ]
    for case in cases:
        seq = sequential_quick_sort(case)
        par = parallel_quick_sort(case, max_workers=2, min_partition_size=4, partition_depth=2)
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
    n = 140_000
    data = rng.integers(0, 2_000_000, size=n, dtype=np.int64).tolist()

    workers = min(4, os.cpu_count() or 1)
    min_partition_size = 20_000
    partition_depth = max(1, math.ceil(math.log2(workers)))
    print(
        f"[config] n={n}, workers={workers}, min_partition_size={min_partition_size}, "
        f"partition_depth={partition_depth}, python={os.sys.version.split()[0]}"
    )

    seq_sorted, seq_metric = _benchmark("sequential_quick_sort", sequential_quick_sort, data)
    par_sorted, par_metric = _benchmark(
        "parallel_quick_sort",
        lambda arr: parallel_quick_sort(
            arr,
            max_workers=workers,
            min_partition_size=min_partition_size,
            partition_depth=partition_depth,
        ),
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
