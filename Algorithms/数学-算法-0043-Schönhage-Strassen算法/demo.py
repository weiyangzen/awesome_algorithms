"""MVP for MATH-0043: Schönhage-Strassen-style large integer multiplication.

This implementation is intentionally educational:
- It uses an FFT-like transform in the ring Z/(2^K + 1)Z.
- Roots of unity are powers of 2, matching the key Schönhage-Strassen idea.
- It does not implement every optimization from production libraries (e.g., GMP).

Run:
    python3 demo.py
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Sequence, Tuple


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _to_limbs(value: int, limb_bits: int) -> List[int]:
    """Convert non-negative int to little-endian limbs in base 2^limb_bits."""
    if value < 0:
        raise ValueError("_to_limbs expects non-negative input")
    if value == 0:
        return [0]

    mask = (1 << limb_bits) - 1
    out: List[int] = []
    while value:
        out.append(value & mask)
        value >>= limb_bits
    return out


def _from_limbs(limbs: Sequence[int], limb_bits: int) -> int:
    """Convert little-endian limbs in base 2^limb_bits back to int."""
    out = 0
    for d in reversed(limbs):
        out = (out << limb_bits) + d
    return out


def _order_of_two_mod_fermat_like(k: int, modulus: int) -> int:
    """Return multiplicative order of 2 modulo (2^k + 1), power-of-two order."""
    order = 2 * k
    while order % 2 == 0 and pow(2, order // 2, modulus) == 1:
        order //= 2
    return order


def _choose_ring_params(conv_len: int, limb_bits: int) -> Tuple[int, int, int]:
    """Choose (modulus, k, order_of_two) for an NTT of length conv_len.

    Conditions:
    1) modulus = 2^k + 1 is large enough to avoid coefficient wrap-around.
    2) order_of_two is divisible by conv_len so we can build a conv_len-th root.
    """
    max_coeff = conv_len * ((1 << limb_bits) - 1) ** 2

    k = 1
    while (1 << k) <= max_coeff or 2 * k < conv_len:
        k <<= 1

    while True:
        modulus = (1 << k) + 1
        order = _order_of_two_mod_fermat_like(k, modulus)
        if order >= conv_len and order % conv_len == 0:
            return modulus, k, order
        k <<= 1


def _bit_reverse_permute(a: List[int]) -> None:
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]


def _ntt(a: List[int], modulus: int, omega: int) -> None:
    """In-place iterative NTT over a ring modulo modulus."""
    n = len(a)
    _bit_reverse_permute(a)

    length = 2
    while length <= n:
        wlen = pow(omega, n // length, modulus)
        half = length // 2
        for i in range(0, n, length):
            w = 1
            for j in range(i, i + half):
                u = a[j]
                v = (a[j + half] * w) % modulus
                a[j] = (u + v) % modulus
                a[j + half] = (u - v) % modulus
                w = (w * wlen) % modulus
        length <<= 1


def _intt(a: List[int], modulus: int, omega: int) -> None:
    n = len(a)
    omega_inv = pow(omega, -1, modulus)
    _ntt(a, modulus, omega_inv)
    n_inv = pow(n, -1, modulus)
    for i in range(n):
        a[i] = (a[i] * n_inv) % modulus


def _cyclic_convolution_ring(a: Sequence[int], b: Sequence[int], limb_bits: int) -> Tuple[List[int], Dict[str, int]]:
    """Convolution using ring NTT; returns exact coefficients (no modular wrap)."""
    needed = len(a) + len(b) - 1
    n = _next_power_of_two(needed)

    modulus, k, order = _choose_ring_params(n, limb_bits)
    omega = pow(2, order // n, modulus)

    if pow(omega, n, modulus) != 1 or (n > 1 and pow(omega, n // 2, modulus) == 1):
        raise RuntimeError("Failed to build primitive n-th root of unity")

    fa = list(a) + [0] * (n - len(a))
    fb = list(b) + [0] * (n - len(b))

    _ntt(fa, modulus, omega)
    _ntt(fb, modulus, omega)

    for i in range(n):
        fa[i] = (fa[i] * fb[i]) % modulus

    _intt(fa, modulus, omega)

    coeffs = fa[:needed]
    # Map balanced residues to non-negative exact coefficients.
    # Under our chosen bound, true coefficients are in [0, max_coeff] and < modulus.
    coeffs = [c if c >= 0 else c + modulus for c in coeffs]

    stats = {
        "n": n,
        "k": k,
        "modulus_bits": modulus.bit_length(),
        "order_of_two": order,
    }
    return coeffs, stats


def _carry_propagate(coeffs: Sequence[int], limb_bits: int) -> List[int]:
    base = 1 << limb_bits
    mask = base - 1

    out: List[int] = []
    carry = 0
    for c in coeffs:
        v = c + carry
        out.append(v & mask)
        carry = v >> limb_bits

    while carry:
        out.append(carry & mask)
        carry >>= limb_bits

    while len(out) > 1 and out[-1] == 0:
        out.pop()

    return out


def schonhage_strassen_multiply(x: int, y: int, limb_bits: int = 15) -> Tuple[int, Dict[str, int]]:
    """Schönhage-Strassen-style multiply using ring-NTT + carry reconstruction."""
    if limb_bits <= 0:
        raise ValueError("limb_bits must be positive")

    if x == 0 or y == 0:
        return 0, {"n": 1, "k": 1, "modulus_bits": 2, "order_of_two": 2}

    sign = -1 if (x < 0) ^ (y < 0) else 1
    ax = abs(x)
    ay = abs(y)

    a = _to_limbs(ax, limb_bits)
    b = _to_limbs(ay, limb_bits)

    coeffs, stats = _cyclic_convolution_ring(a, b, limb_bits)
    prod_limbs = _carry_propagate(coeffs, limb_bits)
    prod = _from_limbs(prod_limbs, limb_bits)

    return sign * prod, stats


def _schoolbook_multiply(x: int, y: int, limb_bits: int = 15) -> int:
    """Reference O(n^2) limb multiplication for timing comparison."""
    if x == 0 or y == 0:
        return 0

    sign = -1 if (x < 0) ^ (y < 0) else 1
    a = _to_limbs(abs(x), limb_bits)
    b = _to_limbs(abs(y), limb_bits)

    tmp = [0] * (len(a) + len(b))
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            tmp[i + j] += ai * bj

    out_limbs = _carry_propagate(tmp, limb_bits)
    return sign * _from_limbs(out_limbs, limb_bits)


def _fixed_demo_cases() -> List[Tuple[int, int]]:
    random.seed(43)
    cases: List[Tuple[int, int]] = [
        (0, 123456),
        (1, 1),
        (123456789, 987654321),
        (-(2**120 + 7), 2**130 + 9),
    ]

    for bits in (256, 512, 1024):
        a = random.getrandbits(bits)
        b = random.getrandbits(bits)
        cases.append((a, b))

    return cases


def _run_correctness_suite() -> None:
    print("Correctness checks")
    print("-" * 72)

    for idx, (a, b) in enumerate(_fixed_demo_cases(), start=1):
        ours, stats = schonhage_strassen_multiply(a, b, limb_bits=15)
        truth = a * b
        assert ours == truth, f"Mismatch on case {idx}"
        print(
            f"case={idx:02d} | bits=({a.bit_length():4d},{b.bit_length():4d}) "
            f"| n={stats['n']:4d} | k={stats['k']:4d} | mod_bits={stats['modulus_bits']:4d}"
        )

    print("All deterministic checks passed.")


def _benchmark() -> None:
    print("\nSimple timing (single run each)")
    print("-" * 72)
    random.seed(4300)

    for bits in (256, 512, 1024, 1536):
        a = random.getrandbits(bits)
        b = random.getrandbits(bits)

        t0 = time.perf_counter()
        r_school = _schoolbook_multiply(a, b, limb_bits=15)
        t1 = time.perf_counter()

        r_ss, stats = schonhage_strassen_multiply(a, b, limb_bits=15)
        t2 = time.perf_counter()

        assert r_school == r_ss == a * b

        school_ms = (t1 - t0) * 1000.0
        ss_ms = (t2 - t1) * 1000.0
        print(
            f"bits={bits:4d} | schoolbook={school_ms:8.3f} ms | "
            f"ss_style={ss_ms:8.3f} ms | n={stats['n']:4d}, k={stats['k']:4d}"
        )


def main() -> None:
    print("Schönhage-Strassen-style MVP (MATH-0043)")
    print("=" * 72)
    _run_correctness_suite()
    _benchmark()
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
