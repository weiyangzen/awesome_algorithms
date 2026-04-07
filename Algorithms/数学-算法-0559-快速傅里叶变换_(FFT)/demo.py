"""MATH-0559: Fast Fourier Transform (FFT) MVP.

This script implements radix-2 iterative Cooley-Tukey FFT from scratch
(with numpy only for arrays/math helpers), then uses it to compute linear
convolution and validate against naive references.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def bit_reverse_permute(a: np.ndarray) -> None:
    """Reorder array in-place by bit-reversed index."""
    n = a.shape[0]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]


def fft_iterative(x: Sequence[complex], inverse: bool = False) -> np.ndarray:
    """Radix-2 iterative FFT/IFFT.

    Args:
        x: Input sequence, length must be power of two.
        inverse: If True, compute inverse transform.

    Returns:
        Complex ndarray of transformed values.
    """
    a = np.asarray(x, dtype=np.complex128).copy()
    n = a.shape[0]
    if not is_power_of_two(n):
        raise ValueError("FFT input length must be a power of two.")

    bit_reverse_permute(a)

    length = 2
    while length <= n:
        half = length // 2
        angle = (2.0 * math.pi / length) * (1 if inverse else -1)
        wlen = complex(math.cos(angle), math.sin(angle))

        for start in range(0, n, length):
            w = 1.0 + 0.0j
            for j in range(half):
                u = a[start + j]
                v = a[start + j + half] * w
                a[start + j] = u + v
                a[start + j + half] = u - v
                w *= wlen

        length <<= 1

    if inverse:
        a /= n

    return a


def linear_convolution_fft(x: Sequence[float], y: Sequence[float]) -> np.ndarray:
    """Compute linear convolution via FFT."""
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Input sequences for convolution must be non-empty.")

    need = len(x) + len(y) - 1
    n = next_power_of_two(need)

    a = np.zeros(n, dtype=np.complex128)
    b = np.zeros(n, dtype=np.complex128)
    a[: len(x)] = np.asarray(x, dtype=np.float64)
    b[: len(y)] = np.asarray(y, dtype=np.float64)

    fa = fft_iterative(a, inverse=False)
    fb = fft_iterative(b, inverse=False)
    fc = fa * fb
    c = fft_iterative(fc, inverse=True)

    return c.real[:need]


def linear_convolution_naive(x: Sequence[float], y: Sequence[float]) -> np.ndarray:
    """Reference O(n^2) linear convolution."""
    out = np.zeros(len(x) + len(y) - 1, dtype=np.float64)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            out[i + j] += xi * yj
    return out


def dft_naive(x: Sequence[complex], inverse: bool = False) -> np.ndarray:
    """Reference O(n^2) DFT/IFFT for small-size validation."""
    a = np.asarray(x, dtype=np.complex128)
    n = a.shape[0]
    out = np.zeros(n, dtype=np.complex128)
    sign = 1 if inverse else -1

    for k in range(n):
        s = 0.0 + 0.0j
        for t in range(n):
            theta = sign * 2.0 * math.pi * k * t / n
            s += a[t] * complex(math.cos(theta), math.sin(theta))
        out[k] = s / n if inverse else s

    return out


def main() -> None:
    print("FFT MVP (MATH-0559)")
    print("=" * 72)

    # Fixed example for convolution.
    x = np.array([1.0, 2.0, -1.0, 3.0, 0.5], dtype=np.float64)
    y = np.array([2.0, -2.0, 4.0, 1.5], dtype=np.float64)

    fft_conv = linear_convolution_fft(x, y)
    naive_conv = linear_convolution_naive(x, y)
    np_conv = np.convolve(x, y)

    print("x:", x.tolist())
    print("y:", y.tolist())
    print("FFT convolution:", np.round(fft_conv, 9).tolist())
    print("Naive convolution:", np.round(naive_conv, 9).tolist())

    if not np.allclose(fft_conv, naive_conv, atol=1e-9):
        raise AssertionError("FFT convolution does not match naive convolution.")
    if not np.allclose(fft_conv, np_conv, atol=1e-9):
        raise AssertionError("FFT convolution does not match numpy.convolve.")

    # Validate FFT core (not only convolution) against naive DFT and numpy FFT.
    test_signal = np.array([1.0, -0.5, 2.0, 0.0, -1.0, 3.0, 0.5, -2.0], dtype=np.float64)
    fft_res = fft_iterative(test_signal, inverse=False)
    dft_res = dft_naive(test_signal, inverse=False)
    np_fft_res = np.fft.fft(test_signal)

    if not np.allclose(fft_res, dft_res, atol=1e-9):
        raise AssertionError("FFT does not match naive DFT on fixed signal.")
    if not np.allclose(fft_res, np_fft_res, atol=1e-9):
        raise AssertionError("FFT does not match numpy.fft.fft on fixed signal.")

    recovered = fft_iterative(fft_res, inverse=True)
    if not np.allclose(recovered.real, test_signal, atol=1e-9):
        raise AssertionError("IFFT failed to reconstruct original signal.")

    # Randomized checks for convolution correctness.
    rng = np.random.default_rng(559)
    for n1 in range(1, 10):
        for n2 in range(1, 10):
            a = rng.integers(-5, 6, size=n1).astype(np.float64)
            b = rng.integers(-5, 6, size=n2).astype(np.float64)
            lhs = linear_convolution_fft(a, b)
            rhs = linear_convolution_naive(a, b)
            if not np.allclose(lhs, rhs, atol=1e-9):
                raise AssertionError(f"Random convolution mismatch for n1={n1}, n2={n2}")

    print("DFT cross-check: PASS")
    print("Convolution random checks: PASS")
    print("All FFT checks passed.")


if __name__ == "__main__":
    main()
