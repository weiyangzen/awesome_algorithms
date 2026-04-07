"""不同路径问题 MVP：二维 DP + 一维 DP + 组合数公式交叉校验。"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from numbers import Integral

import numpy as np


@dataclass
class UniquePathsResult:
    m: int
    n: int
    paths: int


def validate_grid_shape(m: int, n: int) -> tuple[int, int]:
    """Validate m, n as positive integers."""
    if not isinstance(m, Integral) or not isinstance(n, Integral):
        raise ValueError(f"m and n must be integers, got m={m!r}, n={n!r}")
    m_int = int(m)
    n_int = int(n)
    if m_int <= 0 or n_int <= 0:
        raise ValueError(f"m and n must be positive, got m={m_int}, n={n_int}")
    return m_int, n_int


def unique_paths_dp_2d(m: int, n: int) -> int:
    """Count paths with a full 2D DP table."""
    m, n = validate_grid_shape(m, n)
    dp = np.ones((m, n), dtype=object)

    for i in range(1, m):
        for j in range(1, n):
            dp[i, j] = int(dp[i - 1, j]) + int(dp[i, j - 1])

    return int(dp[m - 1, n - 1])


def unique_paths_dp_1d(m: int, n: int) -> int:
    """Count paths with O(n) rolling DP array."""
    m, n = validate_grid_shape(m, n)
    row = np.ones(n, dtype=object)

    for _ in range(1, m):
        for j in range(1, n):
            row[j] = int(row[j]) + int(row[j - 1])

    return int(row[n - 1])


def unique_paths_combinatorial(m: int, n: int) -> int:
    """Count paths by combinatorics: C(m+n-2, m-1)."""
    m, n = validate_grid_shape(m, n)
    total_steps = m + n - 2
    down_steps = m - 1
    return int(comb(total_steps, down_steps))


def solve_unique_paths(m: int, n: int) -> UniquePathsResult:
    """Primary API used by this MVP (1D DP)."""
    paths = unique_paths_dp_1d(m, n)
    return UniquePathsResult(m=int(m), n=int(n), paths=paths)


def run_case(name: str, m: int, n: int, expected: int | None = None) -> None:
    dp_2d = unique_paths_dp_2d(m, n)
    dp_1d = unique_paths_dp_1d(m, n)
    by_comb = unique_paths_combinatorial(m, n)
    result = solve_unique_paths(m, n)

    print(f"=== {name} ===")
    print(f"grid: m={m}, n={n}")
    print(f"dp_2d={dp_2d}, dp_1d={dp_1d}, comb={by_comb}, api={result.paths}")
    if expected is not None:
        print(f"expected={expected}")
    print()

    if not (dp_2d == dp_1d == by_comb == result.paths):
        raise AssertionError("Inconsistent results among implementations")
    if expected is not None and dp_1d != expected:
        raise AssertionError(f"Unexpected answer: got {dp_1d}, expected {expected}")


def randomized_cross_check(trials: int = 250, max_dim: int = 20, seed: int = 2026) -> None:
    """Randomized regression among 2D-DP, 1D-DP and combinatorics."""
    rng = np.random.default_rng(seed)

    for _ in range(trials):
        m = int(rng.integers(1, max_dim + 1))
        n = int(rng.integers(1, max_dim + 1))

        a = unique_paths_dp_2d(m, n)
        b = unique_paths_dp_1d(m, n)
        c = unique_paths_combinatorial(m, n)

        if not (a == b == c):
            raise AssertionError(
                f"Random cross-check failed for (m={m}, n={n}): {a}, {b}, {c}"
            )

    print(
        f"Randomized cross-check passed: trials={trials}, max_dim={max_dim}, seed={seed}."
    )


def main() -> None:
    run_case(name="Case 1: single cell", m=1, n=1, expected=1)
    run_case(name="Case 2: single row", m=1, n=7, expected=1)
    run_case(name="Case 3: small rectangle", m=3, n=2, expected=3)
    run_case(name="Case 4: classic", m=3, n=7, expected=28)
    run_case(name="Case 5: square 10x10", m=10, n=10, expected=48620)
    run_case(name="Case 6: medium", m=15, n=12)

    randomized_cross_check(trials=250, max_dim=20, seed=2026)


if __name__ == "__main__":
    main()
