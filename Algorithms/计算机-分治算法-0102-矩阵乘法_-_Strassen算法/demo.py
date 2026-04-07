"""Matrix multiplication MVP using Strassen's divide-and-conquer algorithm."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StrassenConfig:
    """Configuration for Strassen recursion."""

    leaf_size: int = 64


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pad_operands(a: NDArray[np.floating], b: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating], int, int]:
    """Pad two matrices to a common square whose side length is a power of two."""
    m, k_a = a.shape
    k_b, n = b.shape
    if k_a != k_b:
        raise ValueError(f"Shape mismatch: {a.shape} cannot multiply {b.shape}")

    target = _next_power_of_two(max(m, k_a, n))
    out_dtype = np.result_type(a.dtype, b.dtype)

    a_pad = np.zeros((target, target), dtype=out_dtype)
    b_pad = np.zeros((target, target), dtype=out_dtype)
    a_pad[:m, :k_a] = a
    b_pad[:k_b, :n] = b
    return a_pad, b_pad, m, n


def _strassen_recursive(a: NDArray[np.floating], b: NDArray[np.floating], leaf_size: int) -> NDArray[np.floating]:
    n = a.shape[0]
    if n <= leaf_size:
        return a @ b

    half = n // 2

    a11 = a[:half, :half]
    a12 = a[:half, half:]
    a21 = a[half:, :half]
    a22 = a[half:, half:]

    b11 = b[:half, :half]
    b12 = b[:half, half:]
    b21 = b[half:, :half]
    b22 = b[half:, half:]

    # Seven recursive multiplications in Strassen's formulation.
    m1 = _strassen_recursive(a11 + a22, b11 + b22, leaf_size)
    m2 = _strassen_recursive(a21 + a22, b11, leaf_size)
    m3 = _strassen_recursive(a11, b12 - b22, leaf_size)
    m4 = _strassen_recursive(a22, b21 - b11, leaf_size)
    m5 = _strassen_recursive(a11 + a12, b22, leaf_size)
    m6 = _strassen_recursive(a21 - a11, b11 + b12, leaf_size)
    m7 = _strassen_recursive(a12 - a22, b21 + b22, leaf_size)

    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6

    c = np.empty((n, n), dtype=np.result_type(c11.dtype, c22.dtype))
    c[:half, :half] = c11
    c[:half, half:] = c12
    c[half:, :half] = c21
    c[half:, half:] = c22
    return c


def strassen_multiply(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    config: StrassenConfig | None = None,
) -> NDArray[np.floating]:
    """Multiply two matrices with Strassen + padding for non-power-of-two sizes."""
    cfg = config or StrassenConfig()
    if cfg.leaf_size <= 0:
        raise ValueError("leaf_size must be a positive integer")

    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.ndim != 2 or b_arr.ndim != 2:
        raise ValueError("Both operands must be 2D matrices")

    a_pad, b_pad, out_rows, out_cols = _pad_operands(a_arr, b_arr)
    c_pad = _strassen_recursive(a_pad, b_pad, cfg.leaf_size)
    return c_pad[:out_rows, :out_cols]


def _benchmark_once(a: NDArray[np.floating], b: NDArray[np.floating], leaf_size: int) -> tuple[float, float, float]:
    t0 = time.perf_counter()
    c_strassen = strassen_multiply(a, b, StrassenConfig(leaf_size=leaf_size))
    t1 = time.perf_counter()

    c_numpy = a @ b
    t2 = time.perf_counter()

    max_abs_err = float(np.max(np.abs(c_strassen - c_numpy)))
    return (t1 - t0), (t2 - t1), max_abs_err


def main() -> None:
    rng = np.random.default_rng(42)
    cases = [
        (3, 5, 4),
        (8, 8, 8),
        (31, 31, 31),
        (64, 64, 64),
    ]

    leaf_size = 32
    print("Strassen matrix multiplication MVP")
    print(f"leaf_size={leaf_size}")
    print("shape(A) x shape(B) | strassen(s) | numpy(s) | max_abs_err")

    for m, k, n in cases:
        a = rng.normal(size=(m, k))
        b = rng.normal(size=(k, n))
        t_strassen, t_numpy, err = _benchmark_once(a, b, leaf_size)
        print(f"({m},{k}) x ({k},{n}) | {t_strassen:10.6f} | {t_numpy:8.6f} | {err:.3e}")
        if not np.allclose(strassen_multiply(a, b, StrassenConfig(leaf_size=leaf_size)), a @ b, atol=1e-9):
            raise RuntimeError("Validation failed: Strassen result does not match NumPy matmul")

    print("All validation checks passed.")


if __name__ == "__main__":
    main()
