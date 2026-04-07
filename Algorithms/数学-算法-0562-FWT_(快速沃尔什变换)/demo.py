"""MATH-0562: FWT (Fast Walsh-Hadamard Transform) MVP.

This demo implements XOR convolution using FWHT:
1) Transform two vectors with FWHT.
2) Multiply point-wise in transform domain.
3) Inverse FWHT to recover XOR convolution.
"""

from __future__ import annotations

import numpy as np


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def fwht_inplace(a: np.ndarray, inverse: bool = False) -> None:
    """In-place Fast Walsh-Hadamard Transform.

    The transform matrix contains only +1/-1, so each stage is built by
    butterfly updates:
        (u, v) -> (u + v, u - v)
    For inverse transform under this normalization, divide by n once at end.
    """
    n = a.shape[0]
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("FWHT length must be a non-zero power of two.")

    step = 1
    while step < n:
        jump = step * 2
        for i in range(0, n, jump):
            left = a[i : i + step].copy()
            right = a[i + step : i + jump].copy()
            a[i : i + step] = left + right
            a[i + step : i + jump] = left - right
        step = jump

    if inverse:
        a /= n


def xor_convolution_fwt(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute XOR convolution by FWHT in O(n log n)."""
    n = next_power_of_two(max(len(x), len(y)))
    fx = np.zeros(n, dtype=np.float64)
    fy = np.zeros(n, dtype=np.float64)
    fx[: len(x)] = x
    fy[: len(y)] = y

    fwht_inplace(fx, inverse=False)
    fwht_inplace(fy, inverse=False)
    fz = fx * fy
    fwht_inplace(fz, inverse=True)
    return fz


def xor_convolution_naive(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Reference O(n^2) XOR convolution for validation."""
    n = next_power_of_two(max(len(x), len(y)))
    z = np.zeros(n, dtype=np.float64)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            z[i ^ j] += xi * yj
    return z


def main() -> None:
    # Deterministic example (non-power-of-two inputs are padded internally).
    x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y = np.array([2, -1, 0, 3], dtype=np.float64)

    fwt_res = xor_convolution_fwt(x, y)
    naive_res = xor_convolution_naive(x, y)

    print("Input x:", x.tolist())
    print("Input y:", y.tolist())
    print("FWT XOR convolution:", np.round(fwt_res, 6).tolist())
    print("Naive XOR convolution:", np.round(naive_res, 6).tolist())

    if not np.allclose(fwt_res, naive_res, atol=1e-9):
        raise AssertionError("FWT result does not match naive result.")

    # Quick randomized checks on power-of-two lengths.
    rng = np.random.default_rng(42)
    for bits in range(1, 7):
        n = 1 << bits
        xr = rng.integers(-3, 4, size=n).astype(np.float64)
        yr = rng.integers(-3, 4, size=n).astype(np.float64)
        lhs = xor_convolution_fwt(xr, yr)
        rhs = xor_convolution_naive(xr, yr)
        if not np.allclose(lhs, rhs, atol=1e-9):
            raise AssertionError(f"Random test failed for n={n}")

    print("All checks passed.")


if __name__ == "__main__":
    main()
