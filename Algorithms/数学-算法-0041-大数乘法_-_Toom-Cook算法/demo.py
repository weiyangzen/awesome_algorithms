"""Minimal runnable MVP for big integer multiplication via Toom-Cook (Toom-3)."""

from __future__ import annotations

import random
import time
from typing import Iterable, List, Sequence, Tuple

BASE = 10_000
TOOM3_CUTOFF_LIMBS = 18


def _exact_div(value: int, divisor: int) -> int:
    """Return value // divisor and assert exact divisibility."""
    q, r = divmod(value, divisor)
    if r != 0:
        raise ValueError(f"interpolation exact division failed: {value} / {divisor}")
    return q


def _limb_count(n: int, base: int = BASE) -> int:
    """Count base-`base` limbs for non-negative integer n."""
    if n < 0:
        raise ValueError("_limb_count expects non-negative input")
    if n == 0:
        return 1

    count = 0
    while n:
        n //= base
        count += 1
    return count


def _split_three(n: int, m: int, base: int = BASE) -> Tuple[int, int, int, int]:
    """Split n = a0 + a1*z + a2*z^2 where z = base^m."""
    if n < 0:
        raise ValueError("_split_three expects non-negative input")
    z = base**m
    a0 = n % z
    n //= z
    a1 = n % z
    a2 = n // z
    return a0, a1, a2, z


def _to_limbs(n: int, base: int = BASE) -> List[int]:
    """Convert non-negative integer to little-endian limb list."""
    if n < 0:
        raise ValueError("_to_limbs expects non-negative input")
    if n == 0:
        return [0]

    limbs: List[int] = []
    while n:
        n, rem = divmod(n, base)
        limbs.append(rem)
    return limbs


def _from_limbs(limbs: Sequence[int], base: int = BASE) -> int:
    """Convert little-endian limb list back to integer."""
    value = 0
    for limb in reversed(limbs):
        value = value * base + limb
    return value


def schoolbook_multiply(a: int, b: int, base: int = BASE) -> int:
    """Grade-school multiplication for signed integers over base-`base` limbs."""
    if a == 0 or b == 0:
        return 0

    sign = -1 if (a < 0) ^ (b < 0) else 1
    x, y = abs(a), abs(b)

    ax = _to_limbs(x, base)
    by = _to_limbs(y, base)

    out = [0] * (len(ax) + len(by) + 1)

    for i, ai in enumerate(ax):
        carry = 0
        for j, bj in enumerate(by):
            total = out[i + j] + ai * bj + carry
            carry, out[i + j] = divmod(total, base)

        k = i + len(by)
        while carry:
            total = out[k] + carry
            carry, out[k] = divmod(total, base)
            k += 1

    while len(out) > 1 and out[-1] == 0:
        out.pop()

    return sign * _from_limbs(out, base)


def _interpolate_toom3(v0: int, v1: int, vm1: int, v2: int, vinf: int) -> Tuple[int, int, int, int, int]:
    """Recover coefficients r0..r4 from Toom-3 evaluation values."""
    r0 = v0
    r4 = vinf

    s1 = _exact_div(v1 + vm1, 2)  # r0 + r2 + r4
    s2 = _exact_div(v1 - vm1, 2)  # r1 + r3

    r2 = s1 - r0 - r4

    t = v2 - r0 - 16 * r4
    t = _exact_div(t, 2)  # r1 + 2*r2 + 4*r3

    r3 = _exact_div(t - s2 - 2 * r2, 3)
    r1 = s2 - r3

    return r0, r1, r2, r3, r4


def _recompose(coeffs: Sequence[int], z: int) -> int:
    """Evaluate polynomial with coeffs [r0, r1, ...] at point z via Horner."""
    acc = 0
    for c in reversed(coeffs):
        acc = acc * z + c
    return acc


def _toom3_signed(a: int, b: int, base: int, cutoff_limbs: int) -> int:
    """Signed wrapper for recursive Toom-3 multiplication."""
    if a == 0 or b == 0:
        return 0

    sign = -1 if (a < 0) ^ (b < 0) else 1
    out = _toom3_nonnegative(abs(a), abs(b), base, cutoff_limbs)
    return sign * out


def _toom3_nonnegative(a: int, b: int, base: int, cutoff_limbs: int) -> int:
    """Toom-3 core for non-negative integers."""
    if a == 0 or b == 0:
        return 0

    la = _limb_count(a, base)
    lb = _limb_count(b, base)

    # Small sizes and very unbalanced inputs usually favor schoolbook in pure Python.
    if max(la, lb) <= cutoff_limbs or min(la, lb) <= cutoff_limbs // 2:
        return schoolbook_multiply(a, b, base)

    n = max(la, lb)
    m = (n + 2) // 3

    a0, a1, a2, z = _split_three(a, m, base)
    b0, b1, b2, _ = _split_three(b, m, base)

    p0 = a0
    p1 = a0 + a1 + a2
    pm1 = a0 - a1 + a2
    p2 = a0 + 2 * a1 + 4 * a2
    pinf = a2

    q0 = b0
    q1 = b0 + b1 + b2
    qm1 = b0 - b1 + b2
    q2 = b0 + 2 * b1 + 4 * b2
    qinf = b2

    v0 = _toom3_signed(p0, q0, base, cutoff_limbs)
    v1 = _toom3_signed(p1, q1, base, cutoff_limbs)
    vm1 = _toom3_signed(pm1, qm1, base, cutoff_limbs)
    v2 = _toom3_signed(p2, q2, base, cutoff_limbs)
    vinf = _toom3_signed(pinf, qinf, base, cutoff_limbs)

    coeffs = _interpolate_toom3(v0, v1, vm1, v2, vinf)
    return _recompose(coeffs, z)


def toom3_multiply(a: int, b: int, base: int = BASE, cutoff_limbs: int = TOOM3_CUTOFF_LIMBS) -> int:
    """Public Toom-3 multiplication API."""
    if base <= 1:
        raise ValueError("base must be > 1")
    if cutoff_limbs < 2:
        raise ValueError("cutoff_limbs must be >= 2")
    return _toom3_signed(a, b, base, cutoff_limbs)


def _random_bigint(decimal_digits: int, rng: random.Random) -> int:
    """Generate a signed random integer with exact decimal digit length."""
    if decimal_digits <= 0:
        raise ValueError("decimal_digits must be positive")

    if decimal_digits == 1:
        value = rng.randrange(10)
    else:
        first = str(rng.randrange(1, 10))
        rest = "".join(str(rng.randrange(10)) for _ in range(decimal_digits - 1))
        value = int(first + rest)

    if rng.random() < 0.5:
        value = -value
    return value


def _one_level_trace(a: int, b: int, base: int = BASE) -> List[str]:
    """Create a human-readable single-level Toom-3 trace."""
    ax, bx = abs(a), abs(b)
    n = max(_limb_count(ax, base), _limb_count(bx, base))
    m = (n + 2) // 3

    a0, a1, a2, z = _split_three(ax, m, base)
    b0, b1, b2, _ = _split_three(bx, m, base)

    p0 = a0
    p1 = a0 + a1 + a2
    pm1 = a0 - a1 + a2
    p2 = a0 + 2 * a1 + 4 * a2
    pinf = a2

    q0 = b0
    q1 = b0 + b1 + b2
    qm1 = b0 - b1 + b2
    q2 = b0 + 2 * b1 + 4 * b2
    qinf = b2

    v0 = p0 * q0
    v1 = p1 * q1
    vm1 = pm1 * qm1
    v2 = p2 * q2
    vinf = pinf * qinf

    coeffs = _interpolate_toom3(v0, v1, vm1, v2, vinf)
    recovered = _recompose(coeffs, z)

    lines = [
        f"split base B={base}, m={m}, z=B^m={z}",
        f"a blocks: a0={a0}, a1={a1}, a2={a2}",
        f"b blocks: b0={b0}, b1={b1}, b2={b2}",
        f"point products: v0={v0}, v1={v1}, vm1={vm1}, v2={v2}, vinf={vinf}",
        (
            "interpolated coeffs: "
            f"r0={coeffs[0]}, r1={coeffs[1]}, r2={coeffs[2]}, r3={coeffs[3]}, r4={coeffs[4]}"
        ),
        f"recompose R(z)={recovered}",
        f"builtin abs(a*b)={ax * bx}",
        f"trace check pass={recovered == ax * bx}",
    ]
    return lines


def _time_call(fn, *args, repeat: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    end = time.perf_counter()
    return (end - start) / repeat


def main() -> None:
    print("=== Toom-Cook (Toom-3) Big Integer Multiplication MVP ===")
    print(f"BASE={BASE}, TOOM3_CUTOFF_LIMBS={TOOM3_CUTOFF_LIMBS}")

    fixed_cases = [
        (0, 123456789),
        (123456789, 987654321),
        (-987654321987654321, 123456789123456789),
        (10**60 + 13579, 10**55 + 24680),
        (-10**95 + 7, -(10**92 + 11)),
    ]

    print("\n[1] Fixed-case correctness checks")
    for idx, (a, b) in enumerate(fixed_cases, start=1):
        got = toom3_multiply(a, b)
        expected = a * b
        assert got == expected, f"fixed case failed: a={a}, b={b}"
        print(f"case#{idx}: digits(a)={len(str(abs(a)))}, digits(b)={len(str(abs(b)))}, pass=True")

    print("\n[2] Randomized correctness checks")
    rng = random.Random(20260407)
    plans: Iterable[Tuple[int, int]] = [
        (8, 12),
        (32, 8),
        (96, 5),
        (180, 3),
    ]

    total = 0
    for digits, samples in plans:
        for _ in range(samples):
            a = _random_bigint(digits, rng)
            b = _random_bigint(digits, rng)
            got = toom3_multiply(a, b)
            expected = a * b
            assert got == expected, f"random case failed at digits={digits}"
            total += 1
        print(f"digits={digits:>3}: samples={samples:>2}, pass=True")

    print(f"random total checks passed: {total}")

    print("\n[3] One-level Toom-3 trace")
    trace_a = int("123456789012345678901234567890")
    trace_b = int("998877665544332211009988776655")
    for line in _one_level_trace(trace_a, trace_b):
        print(line)

    print("\n[4] Tiny timing snapshot (lower is better)")
    for digits in (40, 120):
        a = _random_bigint(digits, rng)
        b = _random_bigint(digits, rng)
        t_toom = _time_call(toom3_multiply, a, b, repeat=2)
        t_builtin = _time_call(lambda x, y: x * y, a, b, repeat=2)
        ratio = t_toom / t_builtin if t_builtin > 0 else float("inf")
        print(
            f"digits={digits:>3}: toom3={t_toom:.6f}s, builtin={t_builtin:.6f}s, ratio={ratio:.2f}x"
        )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
