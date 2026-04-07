"""并行前缀和最小可运行 MVP.

运行:
    uv run python demo.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np


ArrayLike = np.ndarray


def sequential_prefix_sum(values: ArrayLike) -> np.ndarray:
    """顺序前缀和（inclusive scan），作为正确性与性能基线."""
    arr = np.asarray(values, dtype=np.int64)
    n = arr.size
    if n == 0:
        return arr.copy()

    out = np.empty_like(arr)
    running = 0
    for i in range(n):
        running += int(arr[i])
        out[i] = running
    return out


def hillis_steele_prefix_sum(values: ArrayLike) -> np.ndarray:
    """Hillis-Steele 并行扫描（inclusive scan）。

    每轮步长 offset 翻倍：1,2,4,...
    对所有 i>=offset 并行执行 out[i] = prev[i] + prev[i-offset]。
    """
    out = np.asarray(values, dtype=np.int64).copy()
    n = out.size
    if n == 0:
        return out

    offset = 1
    while offset < n:
        prev = out.copy()
        out[offset:] = prev[offset:] + prev[:-offset]
        offset <<= 1
    return out


def _assert_prefix_sum(original: np.ndarray, prefix: np.ndarray) -> None:
    """检查 prefix 是否是 original 的包含型前缀和."""
    if original.shape != prefix.shape:
        raise AssertionError("形状不一致")

    n = original.size
    if n == 0:
        return

    if prefix[0] != original[0]:
        raise AssertionError("前缀和首元素错误")

    reconstructed = prefix[1:] - prefix[:-1]
    if not np.array_equal(reconstructed, original[1:]):
        raise AssertionError("前缀和差分还原失败")


@dataclass
class BenchmarkResult:
    name: str
    seconds: float


def _benchmark(name: str, fn, data: np.ndarray) -> tuple[np.ndarray, BenchmarkResult]:
    start = perf_counter()
    out = fn(data)
    elapsed = perf_counter() - start
    return out, BenchmarkResult(name=name, seconds=elapsed)


def _self_test() -> None:
    cases = [
        np.array([], dtype=np.int64),
        np.array([7], dtype=np.int64),
        np.array([1, 2, 3, 4], dtype=np.int64),
        np.array([5, -2, 9, 0, -1, 3], dtype=np.int64),
        np.array([4, 4, 4, 4, 4], dtype=np.int64),
        np.array([-3, -1, -7, 2, 9], dtype=np.int64),
    ]

    for case in cases:
        seq = sequential_prefix_sum(case)
        hs = hillis_steele_prefix_sum(case)
        ref = np.cumsum(case, dtype=np.int64)
        _assert_prefix_sum(case, seq)
        _assert_prefix_sum(case, hs)

        if not np.array_equal(seq, ref):
            raise AssertionError("sequential_prefix_sum 与 numpy.cumsum 不一致")
        if not np.array_equal(hs, ref):
            raise AssertionError("hillis_steele_prefix_sum 与 numpy.cumsum 不一致")

    print("[self-test] passed")


def main() -> None:
    _self_test()

    rng = np.random.default_rng(seed=20260407)
    n = 800_000
    data = rng.integers(-8, 9, size=n, dtype=np.int64)
    rounds = math.ceil(math.log2(n)) if n > 1 else 0

    print(f"[config] n={n}, rounds={rounds}, dtype={data.dtype}, seed=20260407")

    seq, seq_metric = _benchmark("sequential_prefix_sum", sequential_prefix_sum, data)
    hs, hs_metric = _benchmark("hillis_steele_prefix_sum", hillis_steele_prefix_sum, data)
    ref, ref_metric = _benchmark(
        "numpy_cumsum_reference",
        lambda arr: np.cumsum(arr, dtype=np.int64),
        data,
    )

    _assert_prefix_sum(data, seq)
    _assert_prefix_sum(data, hs)

    if not np.array_equal(seq, hs):
        raise AssertionError("并行扫描结果与顺序基线不一致")
    if not np.array_equal(seq, ref):
        raise AssertionError("并行扫描结果与 numpy.cumsum 参考不一致")

    print(f"[time] {seq_metric.name}: {seq_metric.seconds:.4f}s")
    print(f"[time] {hs_metric.name}: {hs_metric.seconds:.4f}s")
    print(f"[time] {ref_metric.name}: {ref_metric.seconds:.4f}s")
    rel = hs_metric.seconds / seq_metric.seconds if seq_metric.seconds > 0 else float('inf')
    print(f"[time] relative(hillis_steele/sequential): {rel:.3f}x")
    print("[check] sequential == hillis_steele == numpy_cumsum")
    print(f"[sample] first10_data={data[:10].tolist()}")
    print(f"[sample] first10_prefix={hs[:10].tolist()}")


if __name__ == "__main__":
    main()
