"""Number Theoretic Transform (NTT) MVP.

This script demonstrates:
1) iterative in-place NTT / inverse NTT over a finite field,
2) polynomial convolution using NTT,
3) correctness checks against naive convolution,
4) lightweight scaling metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List, Sequence


MOD = 998244353
PRIMITIVE_ROOT = 3


@dataclass
class NTTStats:
    """Runtime counters for visibility into arithmetic work."""

    butterfly_ops: int = 0
    stages: int = 0


def _bit_reverse_permute(a: List[int]) -> None:
    """Reorder in-place by bit-reversed indices."""
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


def ntt(a: List[int], invert: bool, stats: NTTStats | None = None) -> None:
    """In-place iterative Cooley-Tukey NTT."""
    n = len(a)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("NTT length must be a positive power of two.")
    if (MOD - 1) % n != 0:
        raise ValueError("NTT length must divide MOD - 1.")

    _bit_reverse_permute(a)

    length = 2
    while length <= n:
        w_len = pow(PRIMITIVE_ROOT, (MOD - 1) // length, MOD)
        if invert:
            w_len = pow(w_len, MOD - 2, MOD)

        for start in range(0, n, length):
            w = 1
            half = length // 2
            for offset in range(half):
                u = a[start + offset]
                v = (a[start + offset + half] * w) % MOD
                a[start + offset] = (u + v) % MOD
                a[start + offset + half] = (u - v) % MOD
                w = (w * w_len) % MOD
                if stats is not None:
                    stats.butterfly_ops += 1
            if stats is not None:
                stats.stages += 1
        length <<= 1

    if invert:
        n_inv = pow(n, MOD - 2, MOD)
        for i in range(n):
            a[i] = (a[i] * n_inv) % MOD


def convolution_ntt(a: Sequence[int], b: Sequence[int], stats: NTTStats | None = None) -> List[int]:
    """Compute polynomial convolution under MOD via NTT."""
    if not a or not b:
        return []

    out_size = len(a) + len(b) - 1
    n = 1
    while n < out_size:
        n <<= 1

    fa = [x % MOD for x in a] + [0] * (n - len(a))
    fb = [x % MOD for x in b] + [0] * (n - len(b))

    ntt(fa, invert=False, stats=stats)
    ntt(fb, invert=False, stats=stats)

    for i in range(n):
        fa[i] = (fa[i] * fb[i]) % MOD

    ntt(fa, invert=True, stats=stats)
    return fa[:out_size]


def convolution_naive(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """Reference O(n*m) convolution under MOD."""
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] = (out[i + j] + ai * bj) % MOD
    return out


def correctness_demo() -> None:
    """Verify NTT results against naive convolution on fixed cases."""
    print("=== Correctness Demo (NTT vs Naive) ===")
    cases = [
        ([1, 2, 3], [4, 5]),
        ([2, 0, 1, 7], [3, 9, 2]),
        ([5, 1, 4, 2, 8], [6, 0, 3, 7]),
        ([123, 456, 789, 1011], [13, 17, 19, 23, 29]),
    ]
    for idx, (a, b) in enumerate(cases, start=1):
        ntt_res = convolution_ntt(a, b)
        naive_res = convolution_naive(a, b)
        ok = ntt_res == naive_res
        print(f"case {idx}: len(a)={len(a)}, len(b)={len(b)}, equal={ok}")
        print(f"  result={ntt_res}")
    print()


def scaling_demo() -> None:
    """Show runtime trend as polynomial length grows."""
    print("=== Scaling Snapshot ===")
    sizes = [64, 256, 1024, 4096]
    print(f"{'n':>8} | {'time_ms':>10} | {'butterflies':>12} | {'stages_acc':>10}")
    print("-" * 52)

    for n in sizes:
        a = [(i * 17 + 3) % MOD for i in range(n)]
        b = [(i * 19 + 5) % MOD for i in range(n)]
        stats = NTTStats()
        t0 = perf_counter()
        _ = convolution_ntt(a, b, stats=stats)
        elapsed_ms = (perf_counter() - t0) * 1_000
        print(
            f"{n:>8} | {elapsed_ms:>10.3f} | {stats.butterfly_ops:>12} | {stats.stages:>10}"
        )
    print()


def main() -> None:
    print(f"MOD={MOD}, primitive_root={PRIMITIVE_ROOT}")
    correctness_demo()
    scaling_demo()


if __name__ == "__main__":
    main()
