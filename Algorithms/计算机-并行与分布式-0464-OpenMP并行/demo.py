"""Minimal runnable MVP for OpenMP-style parallel for + reduction (CS-0303).

Run:
    uv run python demo.py
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass
class RunStat:
    """One benchmark record."""

    mode: str
    total: float
    elapsed_sec: float
    rel_err_vs_expected: float


def kernel(value: float) -> float:
    """CPU-bound scalar kernel used for demonstrating reduction."""
    return math.sin(value) * math.sin(value) + math.cos(value) * math.cos(value) + math.sqrt(value)


def reduce_chunk(chunk: np.ndarray) -> float:
    """Worker-side reduction over one chunk."""
    acc = 0.0
    for value in chunk:
        acc += kernel(float(value))
    return acc


def serial_reduction(data: np.ndarray) -> float:
    """Serial baseline: one loop on one process."""
    acc = 0.0
    for value in data:
        acc += kernel(float(value))
    return acc


def make_static_chunks(data: np.ndarray, workers: int) -> list[np.ndarray]:
    """OpenMP schedule(static): split contiguous ranges once at launch."""
    return [chunk for chunk in np.array_split(data, workers) if chunk.size > 0]


def make_dynamic_chunks(data: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    """OpenMP schedule(dynamic): many small tasks fetched at runtime."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [data[i : i + chunk_size] for i in range(0, data.size, chunk_size)]


def parallel_reduction(chunks: list[np.ndarray], workers: int) -> float:
    """Parallel reduction by process pool + partial-sum merge."""
    with ProcessPoolExecutor(max_workers=workers) as executor:
        partials = list(executor.map(reduce_chunk, chunks))
    return float(sum(partials))


def timed_call(fn, *args, **kwargs) -> tuple[float, float]:
    """Return (result, elapsed_seconds)."""
    start = perf_counter()
    result = float(fn(*args, **kwargs))
    elapsed = perf_counter() - start
    return result, elapsed


def relative_error(value: float, target: float) -> float:
    denom = max(1.0, abs(target))
    return abs(value - target) / denom


def print_result_table(stats: list[RunStat], serial_time: float) -> None:
    header = f"{'mode':<20} {'sum':>18} {'seconds':>10} {'speedup':>10} {'rel_err':>12}"
    print(header)
    print("-" * len(header))
    for item in stats:
        speedup = serial_time / item.elapsed_sec if item.elapsed_sec > 0 else float("inf")
        print(
            f"{item.mode:<20} {item.total:>18.6f} {item.elapsed_sec:>10.4f} "
            f"{speedup:>10.2f} {item.rel_err_vs_expected:>12.3e}"
        )


def main() -> None:
    print("=== OpenMP Parallel MVP (Python Simulation) ===")

    cpu_count = os.cpu_count() or 1
    workers = min(8, max(1, cpu_count))

    n = 1_200_000
    dynamic_chunk_size = 40_000
    data = np.linspace(1.0, 400.0, num=n, dtype=np.float64)

    # Expected value uses identity: sin^2(x) + cos^2(x) = 1.
    expected = float(np.sum(1.0 + np.sqrt(data)))

    serial_sum, serial_time = timed_call(serial_reduction, data)
    static_chunks = make_static_chunks(data, workers)
    dynamic_chunks = make_dynamic_chunks(data, dynamic_chunk_size)
    static_sum, static_time = timed_call(parallel_reduction, static_chunks, workers)
    dynamic_sum, dynamic_time = timed_call(parallel_reduction, dynamic_chunks, workers)

    stats = [
        RunStat("serial", serial_sum, serial_time, relative_error(serial_sum, expected)),
        RunStat("parallel-static", static_sum, static_time, relative_error(static_sum, expected)),
        RunStat("parallel-dynamic", dynamic_sum, dynamic_time, relative_error(dynamic_sum, expected)),
    ]

    tolerance = 2e-12
    for item in stats:
        if item.rel_err_vs_expected > tolerance:
            raise AssertionError(
                f"{item.mode} result mismatch: rel_err={item.rel_err_vs_expected:.3e} > {tolerance:.1e}"
            )

    print(f"Input size N={n}, workers={workers}, dynamic_chunk_size={dynamic_chunk_size}")
    print(
        "OpenMP mapping: #pragma omp parallel for reduction(+:sum) "
        "schedule(static|dynamic)"
    )
    print_result_table(stats, serial_time=serial_time)
    print("\nAll checks passed for CS-0303 (OpenMP并行).")


if __name__ == "__main__":
    main()
