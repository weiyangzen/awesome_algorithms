"""并行矩阵乘法最小可运行 MVP.

运行:
    uv run python demo.py
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

# 为避免与 NumPy/BLAS 自带多线程叠加，默认把底层 BLAS 线程数限制为 1。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

Matrix = np.ndarray


def generate_matrix(rows: int, cols: int, seed: int) -> Matrix:
    """生成可复现的随机矩阵（float64）。"""
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols)).astype(np.float64, copy=False)


def _validate_shapes(a: Matrix, b: Matrix) -> None:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must both be 2-D matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes: {a.shape} x {b.shape}")


def _iter_row_ranges(total_rows: int, parts: int) -> list[tuple[int, int]]:
    if total_rows <= 0:
        return []
    parts = max(1, min(parts, total_rows))
    chunk_size = (total_rows + parts - 1) // parts
    return [(start, min(start + chunk_size, total_rows)) for start in range(0, total_rows, chunk_size)]


def sequential_blocked_matmul(a: Matrix, b: Matrix, block_size: int = 128) -> Matrix:
    """顺序分块矩阵乘法 C = A x B。"""
    _validate_shapes(a, b)
    block_size = max(1, block_size)
    m, k = a.shape
    _, n = b.shape
    c = np.zeros((m, n), dtype=np.float64)

    for i0 in range(0, m, block_size):
        i1 = min(i0 + block_size, m)
        c_rows = c[i0:i1, :]
        a_rows = a[i0:i1, :]
        for k0 in range(0, k, block_size):
            k1 = min(k0 + block_size, k)
            c_rows += a_rows[:, k0:k1] @ b[k0:k1, :]

    return c


def _worker_row_block(a_rows: Matrix, b: Matrix, block_size: int) -> Matrix:
    """计算 C 的一个行块。"""
    rows, k = a_rows.shape
    n = b.shape[1]
    c_rows = np.zeros((rows, n), dtype=np.float64)

    for k0 in range(0, k, block_size):
        k1 = min(k0 + block_size, k)
        c_rows += a_rows[:, k0:k1] @ b[k0:k1, :]

    return c_rows


def parallel_blocked_matmul(
    a: Matrix,
    b: Matrix,
    workers: int | None = None,
    block_size: int = 128,
) -> Matrix:
    """按行切分 + 线程池并行的分块矩阵乘法。"""
    _validate_shapes(a, b)
    block_size = max(1, block_size)
    m = a.shape[0]
    n = b.shape[1]

    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    workers = max(1, workers)

    if workers == 1 or m == 1:
        return sequential_blocked_matmul(a, b, block_size=block_size)

    ranges = _iter_row_ranges(m, workers)
    c = np.zeros((m, n), dtype=np.float64)

    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = [executor.submit(_worker_row_block, a[start:end, :], b, block_size) for start, end in ranges]
        for (start, end), future in zip(ranges, futures):
            c[start:end, :] = future.result()

    return c


def max_abs_error(x: Matrix, y: Matrix) -> float:
    return float(np.max(np.abs(x - y)))


def relative_fro_error(x: Matrix, y: Matrix) -> float:
    numerator = float(np.linalg.norm(x - y, ord="fro"))
    denominator = float(np.linalg.norm(y, ord="fro"))
    if denominator == 0.0:
        return numerator
    return numerator / denominator


@dataclass
class BenchResult:
    name: str
    seconds: float


def _benchmark(name: str, fn) -> tuple[Matrix, BenchResult]:
    t0 = perf_counter()
    result = fn()
    elapsed = perf_counter() - t0
    return result, BenchResult(name=name, seconds=elapsed)


def run_self_test() -> None:
    """小规模精确自检。"""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float64)
    expected = np.array([[58.0, 64.0], [139.0, 154.0]], dtype=np.float64)

    seq = sequential_blocked_matmul(a, b, block_size=2)
    par = parallel_blocked_matmul(a, b, workers=2, block_size=2)
    ref = a @ b

    if not np.array_equal(seq, expected):
        raise AssertionError(f"sequential result mismatch: {seq}")
    if not np.array_equal(par, expected):
        raise AssertionError(f"parallel result mismatch: {par}")
    if not np.array_equal(ref, expected):
        raise AssertionError(f"numpy result mismatch: {ref}")

    print("[self-test] passed")


def main() -> None:
    run_self_test()

    m = 768
    k = 768
    n = 768
    block_size = 128
    workers = min(8, os.cpu_count() or 1)
    seed = 20260407

    a = generate_matrix(m, k, seed=seed)
    b = generate_matrix(k, n, seed=seed + 1)

    print(f"[config] A=({m},{k}), B=({k},{n}), block_size={block_size}, workers={workers}, seed={seed}")

    seq_c, seq_metric = _benchmark(
        "sequential_blocked_matmul",
        lambda: sequential_blocked_matmul(a, b, block_size=block_size),
    )
    par_c, par_metric = _benchmark(
        "parallel_blocked_matmul(row-partition + threaded)",
        lambda: parallel_blocked_matmul(a, b, workers=workers, block_size=block_size),
    )
    ref_c, ref_metric = _benchmark(
        "numpy_matmul_reference",
        lambda: a @ b,
    )

    if not np.allclose(seq_c, ref_c, rtol=1e-10, atol=1e-10):
        raise AssertionError("sequential_blocked_matmul is not consistent with numpy reference")
    if not np.allclose(par_c, ref_c, rtol=1e-10, atol=1e-10):
        raise AssertionError("parallel_blocked_matmul is not consistent with numpy reference")

    abs_err = max_abs_error(par_c, ref_c)
    rel_err = relative_fro_error(par_c, ref_c)

    speedup_vs_seq = seq_metric.seconds / par_metric.seconds if par_metric.seconds > 0 else float("inf")
    speedup_vs_np = ref_metric.seconds / par_metric.seconds if par_metric.seconds > 0 else float("inf")

    print(f"[time] {seq_metric.name}: {seq_metric.seconds:.4f}s")
    print(f"[time] {par_metric.name}: {par_metric.seconds:.4f}s")
    print(f"[time] {ref_metric.name}: {ref_metric.seconds:.4f}s")
    print(f"[time] speedup(seq/parallel): {speedup_vs_seq:.3f}x")
    print(f"[time] speedup(numpy_reference/parallel): {speedup_vs_np:.3f}x")
    print(f"[check] max_abs_error={abs_err:.3e}, rel_fro_error={rel_err:.3e}")


if __name__ == "__main__":
    main()
