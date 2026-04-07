"""Interval DP MVP: Matrix Chain Multiplication.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np


def validate_dims(dims: Sequence[int]) -> np.ndarray:
    """Validate and normalize matrix-chain dimensions."""
    arr = np.asarray(dims, dtype=np.int64)

    if arr.ndim != 1:
        raise ValueError("dims must be a 1D sequence")
    if arr.size < 2:
        raise ValueError("dims length must be at least 2")
    if not np.all(np.isfinite(arr)):
        raise ValueError("dims must contain only finite values")
    if np.any(arr <= 0):
        raise ValueError("all dimensions must be positive")

    return arr


def matrix_chain_interval_dp(dims: Sequence[int]) -> tuple[int, np.ndarray, np.ndarray]:
    """Compute minimum multiplication cost via interval DP.

    Returns:
        min_cost, dp_table, split_table
    """
    p = validate_dims(dims)
    n = p.size - 1

    if n == 1:
        dp = np.zeros((1, 1), dtype=np.int64)
        split = np.zeros((1, 1), dtype=np.int64)
        return 0, dp, split

    inf = np.iinfo(np.int64).max // 4
    dp = np.full((n, n), inf, dtype=np.int64)
    split = np.full((n, n), -1, dtype=np.int64)

    for i in range(n):
        dp[i, i] = 0
        split[i, i] = i

    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            best_cost = inf
            best_k = -1

            for k in range(i, j):
                # cost = left + right + multiply-two-result-matrices
                cand = int(dp[i, k]) + int(dp[k + 1, j]) + int(p[i] * p[k + 1] * p[j + 1])
                if cand < best_cost:
                    best_cost = cand
                    best_k = k

            dp[i, j] = best_cost
            split[i, j] = best_k

    return int(dp[0, n - 1]), dp, split


def reconstruct_parenthesization(split: np.ndarray, i: int, j: int) -> str:
    """Recover one optimal parenthesization from split table."""
    if i == j:
        return f"A{i + 1}"

    k = int(split[i, j])
    if k < i or k >= j:
        raise RuntimeError(f"invalid split at ({i}, {j}): {k}")

    left = reconstruct_parenthesization(split, i, k)
    right = reconstruct_parenthesization(split, k + 1, j)
    return f"({left} x {right})"


def brute_force_min_cost(dims: Sequence[int]) -> int:
    """Reference solution for small n (exponential states with caching)."""
    p = tuple(int(x) for x in validate_dims(dims))
    n = len(p) - 1

    if n == 1:
        return 0

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> int:
        if i == j:
            return 0
        return min(
            solve(i, k) + solve(k + 1, j) + p[i] * p[k + 1] * p[j + 1]
            for k in range(i, j)
        )

    return solve(0, n - 1)


def format_upper_triangle(dp: np.ndarray, width: int = 10) -> str:
    """Pretty-print only upper-triangle entries of DP table."""
    n = dp.shape[0]
    lines: list[str] = []
    for i in range(n):
        row: list[str] = []
        for j in range(n):
            if j < i:
                cell = "-"
            else:
                cell = str(int(dp[i, j]))
            row.append(cell.rjust(width))
        lines.append(" ".join(row))
    return "\n".join(lines)


def run_case(
    name: str,
    dims: Sequence[int],
    expected_cost: int | None,
    brute_force_limit: int = 8,
) -> bool:
    """Execute one deterministic case and validate outputs."""
    min_cost, dp, split = matrix_chain_interval_dp(dims)
    matrix_count = len(dims) - 1

    if matrix_count == 1:
        parenthesization = "A1"
    else:
        parenthesization = reconstruct_parenthesization(split, 0, matrix_count - 1)

    brute_cost: int | None = None
    if matrix_count <= brute_force_limit:
        brute_cost = brute_force_min_cost(dims)

    ok = True
    if expected_cost is not None and min_cost != expected_cost:
        ok = False
    if brute_cost is not None and min_cost != brute_cost:
        ok = False

    print(f"=== {name} ===")
    print(f"dims: {list(dims)}")
    print(f"matrix_count: {matrix_count}")
    print(f"min_cost: {min_cost}")
    print(f"optimal_parenthesization: {parenthesization}")
    if expected_cost is not None:
        print(f"expected_cost: {expected_cost}")
    if brute_cost is not None:
        print(f"brute_force_cost: {brute_cost}")
    print("dp_table_upper_triangle:")
    print(format_upper_triangle(dp))
    print(f"case_passed: {ok}")
    print()

    return ok


def main() -> None:
    cases = [
        {
            "name": "CLRS classic",
            "dims": [30, 35, 15, 5, 10, 20, 25],
            "expected_cost": 15125,
        },
        {
            "name": "Two-way split",
            "dims": [10, 20, 30],
            "expected_cost": 6000,
        },
        {
            "name": "GeeksforGeeks classic",
            "dims": [5, 10, 3, 12, 5, 50, 6],
            "expected_cost": 2010,
        },
        {
            "name": "Single matrix",
            "dims": [8, 13],
            "expected_cost": 0,
        },
    ]

    all_ok = True
    for case in cases:
        ok = run_case(
            name=case["name"],
            dims=case["dims"],
            expected_cost=case["expected_cost"],
        )
        all_ok = all_ok and ok

    print("=== Summary ===")
    print(f"all_cases_passed: {all_ok}")

    if not all_ok:
        raise RuntimeError("At least one case failed validation")


if __name__ == "__main__":
    main()
