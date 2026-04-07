"""Karatsuba multiplication MVP.

Run:
    python3 demo.py
"""

from __future__ import annotations

import random
import time
from typing import List, Tuple


def _karatsuba_nonneg(x: int, y: int, cutoff_bits: int) -> int:
    """Multiply two non-negative integers using Karatsuba recursion."""
    if x == 0 or y == 0:
        return 0

    n = max(x.bit_length(), y.bit_length())
    if n <= cutoff_bits:
        return x * y

    m = n // 2
    x_high = x >> m
    x_low = x - (x_high << m)
    y_high = y >> m
    y_low = y - (y_high << m)

    z2 = _karatsuba_nonneg(x_high, y_high, cutoff_bits)
    z0 = _karatsuba_nonneg(x_low, y_low, cutoff_bits)
    z1 = _karatsuba_nonneg(x_high + x_low, y_high + y_low, cutoff_bits) - z2 - z0

    return (z2 << (2 * m)) + (z1 << m) + z0


def karatsuba_mul(x: int, y: int, cutoff_bits: int = 128) -> int:
    """Multiply signed integers via Karatsuba, with cutoff to native multiplication."""
    if x == 0 or y == 0:
        return 0

    sign = -1 if (x < 0) ^ (y < 0) else 1
    result = _karatsuba_nonneg(abs(x), abs(y), cutoff_bits)
    return sign * result


def schoolbook_mul_decimal(x: int, y: int) -> int:
    """Reference O(n^2) multiplication using decimal digit arrays."""
    if x == 0 or y == 0:
        return 0

    sign = -1 if (x < 0) ^ (y < 0) else 1
    a = str(abs(x))
    b = str(abs(y))
    n = len(a)
    m = len(b)
    out = [0] * (n + m)

    for i in range(n - 1, -1, -1):
        di = ord(a[i]) - ord("0")
        carry = 0
        for j in range(m - 1, -1, -1):
            dj = ord(b[j]) - ord("0")
            pos = i + j + 1
            total = out[pos] + di * dj + carry
            out[pos] = total % 10
            carry = total // 10
        out[i] += carry

    # Normalize possible carry chains in the leading part.
    for k in range(n + m - 1, 0, -1):
        if out[k] >= 10:
            out[k - 1] += out[k] // 10
            out[k] %= 10

    first_non_zero = 0
    while first_non_zero < len(out) - 1 and out[first_non_zero] == 0:
        first_non_zero += 1

    product = int("".join(str(d) for d in out[first_non_zero:]))
    return sign * product


def random_int_with_bits(bits: int, rng: random.Random) -> int:
    """Return a random non-negative integer with exact bit width."""
    if bits <= 0:
        return 0
    return (1 << (bits - 1)) | rng.getrandbits(bits - 1)


def run_correctness_checks() -> None:
    print("[1] Correctness checks")

    fixed_cases: List[Tuple[int, int]] = [
        (0, 123456789),
        (1, 987654321),
        (1234, 5678),
        (-123456789, 987654321),
        (10**50 + 12345, 10**45 + 67890),
        (
            int("314159265358979323846264338327950288419716939937510"),
            int("271828182845904523536028747135266249775724709369995"),
        ),
    ]

    for idx, (x, y) in enumerate(fixed_cases, start=1):
        karatsuba_value = karatsuba_mul(x, y)
        builtin_value = x * y
        schoolbook_value = schoolbook_mul_decimal(x, y)
        ok = karatsuba_value == builtin_value == schoolbook_value
        print(f"  fixed-{idx}: ok={ok}")
        if not ok:
            raise AssertionError("Mismatch on fixed case")

    rng = random.Random(20260407)
    for bits in (8, 16, 32, 64, 128, 256, 512, 1024):
        for _ in range(20):
            x = random_int_with_bits(bits, rng)
            y = random_int_with_bits(bits, rng)
            if rng.random() < 0.3:
                x = -x
            if rng.random() < 0.3:
                y = -y

            karatsuba_value = karatsuba_mul(x, y)
            builtin_value = x * y
            if karatsuba_value != builtin_value:
                raise AssertionError(f"Mismatch at bits={bits}")

    print("  random tests: all passed")


def benchmark_once(fn, x: int, y: int, repeats: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(x, y)
    elapsed = time.perf_counter() - t0
    return elapsed * 1000.0 / repeats


def run_benchmarks() -> None:
    print("\n[2] Performance snapshot (ms per call)")

    rng = random.Random(7)
    rows: List[Tuple[int, float, float, float]] = []

    # Keep sizes moderate so the demo runs quickly.
    for bits in (64, 128, 256, 512, 1024, 2048):
        x = random_int_with_bits(bits, rng)
        y = random_int_with_bits(bits, rng)

        repeats = 5 if bits <= 512 else 3

        t_schoolbook = benchmark_once(schoolbook_mul_decimal, x, y, repeats)
        t_karatsuba = benchmark_once(karatsuba_mul, x, y, repeats)
        t_builtin = benchmark_once(lambda a, b: a * b, x, y, repeats)
        rows.append((bits, t_schoolbook, t_karatsuba, t_builtin))

    header = f"{'bits':>6} | {'schoolbook':>12} | {'karatsuba':>12} | {'builtin':>10}"
    print(header)
    print("-" * len(header))
    for bits, t_schoolbook, t_karatsuba, t_builtin in rows:
        print(
            f"{bits:6d} | {t_schoolbook:12.4f} | {t_karatsuba:12.4f} | {t_builtin:10.4f}"
        )


def main() -> None:
    run_correctness_checks()
    run_benchmarks()


if __name__ == "__main__":
    main()
