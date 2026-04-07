"""Matrix-fast-power optimization for linear DP (Tribonacci example).

This script provides:
1. Baseline O(n) dynamic programming.
2. Optimized O(log n) matrix fast exponentiation.
3. Deterministic verification and timing comparison.
"""

from __future__ import annotations

import time

import numpy as np

MOD_DEFAULT = 1_000_000_007


def validate_inputs(n: int, mod: int) -> None:
    """Validate scalar inputs for recurrence computation."""
    if not isinstance(n, int):
        raise TypeError(f"n must be int, got {type(n).__name__}")
    if not isinstance(mod, int):
        raise TypeError(f"mod must be int, got {type(mod).__name__}")
    if n < 0:
        raise ValueError("n must be non-negative")
    if mod <= 0:
        raise ValueError("mod must be positive")


def dp_tribonacci_mod(n: int, mod: int = MOD_DEFAULT) -> int:
    """O(n) baseline DP for f(n)=f(n-1)+f(n-2)+f(n-3)."""
    validate_inputs(n, mod)

    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1

    f0, f1, f2 = 0, 1, 1
    for _ in range(3, n + 1):
        f0, f1, f2 = f1, f2, (f0 + f1 + f2) % mod
    return f2


def matmul_mod(a: np.ndarray, b: np.ndarray, mod: int) -> np.ndarray:
    """Matrix multiplication under modulo."""
    return (a @ b) % mod


def mat_pow_mod(base: np.ndarray, exp: int, mod: int) -> np.ndarray:
    """Binary exponentiation for square matrix under modulo."""
    if exp < 0:
        raise ValueError("exp must be non-negative")

    size = base.shape[0]
    result = np.eye(size, dtype=np.int64)
    power = base.astype(np.int64, copy=True) % mod
    e = exp

    while e > 0:
        if e & 1:
            result = matmul_mod(result, power, mod)
        power = matmul_mod(power, power, mod)
        e >>= 1

    return result


def tribonacci_matrix_mod(n: int, mod: int = MOD_DEFAULT) -> int:
    """O(log n) solution via transition-matrix fast power."""
    validate_inputs(n, mod)

    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1

    transition = np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.int64,
    )
    state_n2 = np.array([[1], [1], [0]], dtype=np.int64)

    power = mat_pow_mod(transition, n - 2, mod)
    state_n = matmul_mod(power, state_n2, mod)
    return int(state_n[0, 0])


def verify_consistency(max_n: int = 200, mod: int = MOD_DEFAULT) -> None:
    """Cross-check matrix method against baseline DP for n in [0, max_n]."""
    for n in range(max_n + 1):
        dp_val = dp_tribonacci_mod(n, mod)
        mat_val = tribonacci_matrix_mod(n, mod)
        if dp_val != mat_val:
            raise AssertionError(f"Mismatch at n={n}: dp={dp_val}, matrix={mat_val}")


def benchmark(n: int, mod: int = MOD_DEFAULT) -> None:
    """Print timing and value comparison on a medium-sized n."""
    print(f"\n[Benchmark] n={n}, mod={mod}")

    t0 = time.perf_counter()
    dp_val = dp_tribonacci_mod(n, mod)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    mat_val = tribonacci_matrix_mod(n, mod)
    t3 = time.perf_counter()

    if dp_val != mat_val:
        raise AssertionError("Benchmark mismatch: DP and matrix answers are different")

    dp_ms = (t1 - t0) * 1000.0
    mat_ms = (t3 - t2) * 1000.0
    speedup = dp_ms / mat_ms if mat_ms > 0 else float("inf")

    print(f"value={dp_val}")
    print(f"dp_time_ms={dp_ms:.3f}")
    print(f"matrix_time_ms={mat_ms:.3f}")
    print(f"speedup(dp/matrix)={speedup:.2f}x")


def huge_n_demo(n: int, mod: int = MOD_DEFAULT) -> None:
    """Show practicality of O(log n) on huge index."""
    t0 = time.perf_counter()
    val = tribonacci_matrix_mod(n, mod)
    t1 = time.perf_counter()

    print(f"\n[Huge n] n={n}, mod={mod}")
    print(f"value={val}")
    print(f"matrix_time_ms={(t1 - t0) * 1000.0:.3f}")


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    print("=== Matrix Fast Power Optimized DP: Tribonacci ===")
    print(f"mod={MOD_DEFAULT}")

    verify_consistency(max_n=300, mod=MOD_DEFAULT)
    print("consistency_check=PASS (n in [0, 300])")

    benchmark(n=500_000, mod=MOD_DEFAULT)
    huge_n_demo(n=10**18, mod=MOD_DEFAULT)


if __name__ == "__main__":
    main()
