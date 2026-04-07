"""Minimal runnable MVP for FFT-based integer multiplication."""

from __future__ import annotations

import random
import time
from typing import List, Sequence

import numpy as np

BASE = 10_000  # 10^4 keeps coefficients moderate for float FFT rounding.


def int_to_digits(value: int, base: int = BASE) -> List[int]:
    """Convert non-negative integer to little-endian digit list in given base."""
    if value < 0:
        raise ValueError("int_to_digits expects non-negative value")
    if value == 0:
        return [0]

    digits: List[int] = []
    while value:
        value, rem = divmod(value, base)
        digits.append(rem)
    return digits


def digits_to_int(digits: Sequence[int], base: int = BASE) -> int:
    """Convert little-endian base digits back to integer."""
    value = 0
    for d in reversed(digits):
        value = value * base + int(d)
    return value


def trim_leading_zeros(digits: List[int]) -> List[int]:
    """Remove high-end zeros while preserving zero representation."""
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
    return digits


def next_power_of_two(n: int) -> int:
    """Small helper for FFT length selection."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def fft_convolution(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """Linear convolution via complex FFT with rounding to nearest integers."""
    out_len = len(a) + len(b) - 1
    n_fft = next_power_of_two(out_len)

    fa = np.fft.fft(np.array(a, dtype=np.float64), n=n_fft)
    fb = np.fft.fft(np.array(b, dtype=np.float64), n=n_fft)
    c = np.fft.ifft(fa * fb)

    # Imaginary leakage is rounding noise; real part should be near integers.
    coeffs = np.rint(c.real[:out_len]).astype(np.int64)
    return coeffs.tolist()


def carry_normalize(coeffs: Sequence[int], base: int = BASE) -> List[int]:
    """Normalize possibly large convolution coefficients into base digits."""
    digits: List[int] = []
    carry = 0

    for coeff in coeffs:
        total = int(coeff) + carry
        carry, digit = divmod(total, base)
        digits.append(digit)

    while carry:
        carry, digit = divmod(carry, base)
        digits.append(digit)

    return trim_leading_zeros(digits)


def multiply_integers_fft(a: int, b: int, base: int = BASE) -> int:
    """Multiply two integers using FFT-based convolution of base digits."""
    if a == 0 or b == 0:
        return 0

    sign = -1 if (a < 0) ^ (b < 0) else 1
    da = int_to_digits(abs(a), base)
    db = int_to_digits(abs(b), base)

    coeffs = fft_convolution(da, db)
    prod_digits = carry_normalize(coeffs, base)
    result = digits_to_int(prod_digits, base)
    return sign * result


def schoolbook_multiply_digits(a: Sequence[int], b: Sequence[int], base: int = BASE) -> List[int]:
    """O(n^2) baseline multiplication on digit arrays."""
    out = [0] * (len(a) + len(b))
    for i, ai in enumerate(a):
        carry = 0
        for j, bj in enumerate(b):
            idx = i + j
            total = out[idx] + ai * bj + carry
            carry, out[idx] = divmod(total, base)
        k = i + len(b)
        while carry:
            total = out[k] + carry
            carry, out[k] = divmod(total, base)
            k += 1
    return trim_leading_zeros(out)


def random_bigint(rng: random.Random, digits10: int) -> int:
    """Generate a random signed integer with approximately digits10 decimal digits."""
    if digits10 <= 1:
        return rng.randint(-9, 9)

    lo = 10 ** (digits10 - 1)
    hi = 10**digits10 - 1
    val = rng.randint(lo, hi)
    if rng.random() < 0.5:
        val = -val
    return val


def run_self_checks() -> None:
    """Deterministic validation against Python's exact integer multiplication."""
    fixed_cases = [
        (0, 0),
        (0, 123456),
        (-999, 0),
        (12345, 6789),
        (-12345, 6789),
        (-12345, -6789),
        (10**60 + 12345, 10**55 + 98765),
    ]

    for a, b in fixed_cases:
        got = multiply_integers_fft(a, b)
        expect = a * b
        assert got == expect, f"fixed case failed: {a} * {b}"

    rng = random.Random(20260407)
    for digits in [1, 2, 5, 20, 80, 200]:
        for _ in range(25):
            a = random_bigint(rng, digits)
            b = random_bigint(rng, digits)
            got = multiply_integers_fft(a, b)
            expect = a * b
            assert got == expect, f"random case failed: digits={digits}, a={a}, b={b}"


def benchmark() -> None:
    """Quick timing comparison: FFT pipeline vs schoolbook digit multiplication."""
    rng = random.Random(7)
    sizes = [80, 200, 400]

    print("\n=== Benchmark (single sample per size) ===")
    for digits10 in sizes:
        a = abs(random_bigint(rng, digits10))
        b = abs(random_bigint(rng, digits10))

        t0 = time.perf_counter()
        fft_val = multiply_integers_fft(a, b)
        t1 = time.perf_counter()

        da = int_to_digits(a)
        db = int_to_digits(b)
        t2 = time.perf_counter()
        sb_digits = schoolbook_multiply_digits(da, db)
        sb_val = digits_to_int(sb_digits)
        t3 = time.perf_counter()

        assert fft_val == sb_val == a * b
        print(
            f"digits~{digits10:>4} | "
            f"fft={((t1 - t0) * 1e3):>8.3f} ms | "
            f"schoolbook={((t3 - t2) * 1e3):>8.3f} ms"
        )


def main() -> None:
    run_self_checks()

    print("=== FFT Multiplication Demo ===")
    samples = [
        (12345, 6789),
        (-3141592653589793, 2718281828459045),
        (10**80 + 123456789, 10**75 + 987654321),
    ]

    for a, b in samples:
        prod = multiply_integers_fft(a, b)
        print(f"a={a}")
        print(f"b={b}")
        print(f"a*b={prod}")
        print(f"verified={prod == a * b}")
        print("-")

    benchmark()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
