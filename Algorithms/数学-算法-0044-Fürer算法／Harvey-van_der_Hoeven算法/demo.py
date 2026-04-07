"""MVP: FFT-style big integer multiplication inspired by FurER/HvdH pipeline.

This demo is intentionally small and runnable:
- maps integers to base-B coefficient vectors
- uses numpy FFT for polynomial convolution
- performs exact carry normalization back to integer

Run:
    python3 demo.py
"""

from __future__ import annotations

import random
import time
from typing import List, Sequence

import numpy as np


BASE = 10_000


def int_to_chunks(n: int, base: int = BASE) -> List[int]:
    """Encode a nonnegative integer as little-endian base-`base` chunks."""
    if n < 0:
        raise ValueError("int_to_chunks expects n >= 0")
    if n == 0:
        return [0]

    chunks: List[int] = []
    while n:
        n, r = divmod(n, base)
        chunks.append(r)
    return chunks


def chunks_to_int(chunks: Sequence[int], base: int = BASE) -> int:
    """Decode little-endian base chunks back to an integer."""
    value = 0
    for d in reversed(chunks):
        value = value * base + int(d)
    return value


def fft_convolution(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """Convolve two nonnegative integer sequences via real FFT."""
    if not a or not b:
        return []

    out_len = len(a) + len(b) - 1
    fft_len = 1 << (out_len - 1).bit_length()

    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    spec_a = np.fft.rfft(arr_a, fft_len)
    spec_b = np.fft.rfft(arr_b, fft_len)
    spec_c = spec_a * spec_b

    conv = np.fft.irfft(spec_c, fft_len)[:out_len]
    return np.rint(conv).astype(np.int64).tolist()


def normalize_carries(coeffs: Sequence[int], base: int = BASE) -> List[int]:
    """Normalize coefficient list into legal base digits."""
    digits: List[int] = []
    carry = 0

    for coeff in coeffs:
        total = int(coeff) + carry
        digit = total % base
        carry = total // base
        digits.append(digit)

    while carry > 0:
        carry, digit = divmod(carry, base)
        digits.append(digit)

    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()

    return digits


def fft_bigint_multiply(x: int, y: int, base: int = BASE) -> int:
    """Multiply two Python integers using chunking + FFT convolution."""
    if x == 0 or y == 0:
        return 0

    sign = -1 if (x < 0) ^ (y < 0) else 1
    ax, ay = abs(x), abs(y)

    a = int_to_chunks(ax, base)
    b = int_to_chunks(ay, base)

    coeffs = fft_convolution(a, b)
    digits = normalize_carries(coeffs, base)
    product = chunks_to_int(digits, base)

    return sign * product


def random_int(bits: int, rng: random.Random) -> int:
    """Generate a positive integer with exactly `bits` bits."""
    if bits < 1:
        raise ValueError("bits must be >= 1")
    n = rng.getrandbits(bits)
    n |= 1 << (bits - 1)
    return n


def benchmark(bit_sizes: Sequence[int], trials: int = 3, seed: int = 2026) -> None:
    """Compare FFT MVP with Python built-in multiplication."""
    rng = random.Random(seed)

    print("Benchmark (milliseconds, lower is better)")
    print("bits | fft_mvp_ms | python_builtin_ms | verified")
    print("-----+------------+-------------------+---------")

    for bits in bit_sizes:
        fft_total_ms = 0.0
        py_total_ms = 0.0
        ok = True

        for _ in range(trials):
            x = random_int(bits, rng)
            y = random_int(bits, rng)

            t0 = time.perf_counter()
            p_fft = fft_bigint_multiply(x, y)
            t1 = time.perf_counter()

            p_py = x * y
            t2 = time.perf_counter()

            fft_total_ms += (t1 - t0) * 1000.0
            py_total_ms += (t2 - t1) * 1000.0
            ok = ok and (p_fft == p_py)

        fft_avg = fft_total_ms / trials
        py_avg = py_total_ms / trials
        print(f"{bits:4d} | {fft_avg:10.3f} | {py_avg:17.3f} | {ok}")


def main() -> None:
    print("Fürer/HvdH-inspired integer multiplication MVP")
    print("Kernel: integer -> polynomial -> FFT convolution -> carry normalization")
    print("=" * 74)

    examples = [
        (1234, 5678),
        (12345678, 87654321),
        (-3141592653589, 2718281828459),
        (0, 987654321),
    ]

    for x, y in examples:
        p_fft = fft_bigint_multiply(x, y)
        p_py = x * y
        print(f"x={x}, y={y}")
        print(f"fft_product={p_fft}")
        print(f"py_product ={p_py}")
        print(f"match={p_fft == p_py}")
        print("-" * 74)

    benchmark(bit_sizes=[128, 512, 1024, 2048, 4096], trials=3, seed=2026)


if __name__ == "__main__":
    main()
