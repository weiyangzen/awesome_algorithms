"""矩阵链乘法 MVP：自底向上动态规划 + 记忆化基线 + 小规模暴力校验。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np


@dataclass
class MultiplyStep:
    left: tuple[int, int]
    right: tuple[int, int]
    merged: tuple[int, int]
    scalar_cost: int


@dataclass
class MatrixChainResult:
    n_matrices: int
    dims: list[int]
    min_scalar_multiplications: int
    parenthesization: str
    split_table: np.ndarray
    steps: list[MultiplyStep]


def to_dims_array(dims: Sequence[int] | np.ndarray) -> np.ndarray:
    raw = np.asarray(dims, dtype=float)
    if raw.ndim != 1:
        raise ValueError(f"dims must be a 1D sequence, got shape={raw.shape}")
    if raw.size < 2:
        raise ValueError("dims must contain at least two numbers (for at least one matrix)")
    if not np.all(np.isfinite(raw)):
        raise ValueError("dims contains non-finite values")
    if np.any(raw <= 0):
        raise ValueError("all dimensions must be positive")
    rounded = np.round(raw)
    if not np.allclose(raw, rounded):
        raise ValueError("all dimensions must be integers")

    arr = rounded.astype(np.int64)
    return arr


def build_parenthesization(split: np.ndarray, i: int, j: int) -> str:
    if i == j:
        return f"A{i + 1}"
    k = int(split[i, j])
    if k < i or k >= j:
        raise RuntimeError(f"invalid split for parenthesization: i={i}, j={j}, k={k}")
    left = build_parenthesization(split, i, k)
    right = build_parenthesization(split, k + 1, j)
    return f"({left} x {right})"


def rebuild_steps(split: np.ndarray, dims: np.ndarray) -> list[MultiplyStep]:
    n = int(split.shape[0])
    steps: list[MultiplyStep] = []

    def dfs(i: int, j: int) -> tuple[int, int]:
        if i == j:
            return (i, j)
        k = int(split[i, j])
        if k < i or k >= j:
            raise RuntimeError(f"invalid split for step rebuild: i={i}, j={j}, k={k}")
        left = dfs(i, k)
        right = dfs(k + 1, j)
        steps.append(
            MultiplyStep(
                left=(left[0] + 1, left[1] + 1),
                right=(right[0] + 1, right[1] + 1),
                merged=(i + 1, j + 1),
                scalar_cost=int(dims[i] * dims[k + 1] * dims[j + 1]),
            )
        )
        return (i, j)

    if n > 0:
        dfs(0, n - 1)
    return steps


def matrix_chain_order(dims: Sequence[int] | np.ndarray) -> MatrixChainResult:
    """自底向上 DP：返回最小乘法次数、一种最优括号化以及分割表。"""
    p = to_dims_array(dims)
    n = int(p.size - 1)

    inf = np.iinfo(np.int64).max
    cost = np.full((n, n), inf, dtype=np.int64)
    split = np.full((n, n), -1, dtype=int)

    for i in range(n):
        cost[i, i] = 0

    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            best = inf
            best_k = -1
            for k in range(i, j):
                candidate = (
                    int(cost[i, k])
                    + int(cost[k + 1, j])
                    + int(p[i] * p[k + 1] * p[j + 1])
                )
                if candidate < best:
                    best = candidate
                    best_k = k
            cost[i, j] = best
            split[i, j] = best_k

    parenthesization = build_parenthesization(split, 0, n - 1)
    steps = rebuild_steps(split, p)

    return MatrixChainResult(
        n_matrices=n,
        dims=[int(x) for x in p.tolist()],
        min_scalar_multiplications=int(cost[0, n - 1]),
        parenthesization=parenthesization,
        split_table=split,
        steps=steps,
    )


def matrix_chain_top_down_cost(dims: Sequence[int] | np.ndarray) -> int:
    """记忆化递归基线：只返回最小乘法次数。"""
    p = to_dims_array(dims)
    n = int(p.size - 1)

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> int:
        if i == j:
            return 0
        best = np.iinfo(np.int64).max
        for k in range(i, j):
            candidate = solve(i, k) + solve(k + 1, j) + int(p[i] * p[k + 1] * p[j + 1])
            if candidate < best:
                best = candidate
        return int(best)

    return int(solve(0, n - 1))


def matrix_chain_bruteforce_cost(dims: Sequence[int] | np.ndarray) -> int:
    """无缓存暴力枚举所有括号化，仅用于极小规模校验。"""
    p = to_dims_array(dims)
    n = int(p.size - 1)

    def solve(i: int, j: int) -> int:
        if i == j:
            return 0
        best = np.iinfo(np.int64).max
        for k in range(i, j):
            candidate = solve(i, k) + solve(k + 1, j) + int(p[i] * p[k + 1] * p[j + 1])
            if candidate < best:
                best = candidate
        return int(best)

    return int(solve(0, n - 1))


def left_to_right_cost(dims: Sequence[int] | np.ndarray) -> int:
    p = to_dims_array(dims)
    n = int(p.size - 1)
    if n <= 1:
        return 0

    total = 0
    rows = int(p[0])
    cols = int(p[1])
    for t in range(2, p.size):
        nxt_cols = int(p[t])
        total += rows * cols * nxt_cols
        cols = nxt_cols
    return int(total)


def generate_random_chain_matrices(
    dims: Sequence[int] | np.ndarray,
    seed: int,
) -> list[np.ndarray]:
    p = to_dims_array(dims)
    rng = np.random.default_rng(seed)
    mats: list[np.ndarray] = []
    for i in range(p.size - 1):
        mats.append(rng.normal(loc=0.0, scale=1.0, size=(int(p[i]), int(p[i + 1]))))
    return mats


def multiply_by_split(mats: Sequence[np.ndarray], split: np.ndarray, i: int, j: int) -> np.ndarray:
    if i == j:
        return mats[i]
    k = int(split[i, j])
    left = multiply_by_split(mats, split, i, k)
    right = multiply_by_split(mats, split, k + 1, j)
    return left @ right


def multiply_left_to_right(mats: Sequence[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for mat in mats[1:]:
        out = out @ mat
    return out


def format_steps(steps: Sequence[MultiplyStep]) -> str:
    if not steps:
        return "[]"
    parts = []
    for step in steps:
        parts.append(
            f"{step.left}+{step.right}->{step.merged}(cost={step.scalar_cost})"
        )
    return "[" + ", ".join(parts) + "]"


def run_case(
    name: str,
    dims: Sequence[int],
    expected_cost: int | None = None,
    bruteforce_limit: int = 6,
    verify_numeric: bool = True,
) -> None:
    result = matrix_chain_order(dims)
    baseline = matrix_chain_top_down_cost(dims)

    brute_force = None
    if result.n_matrices <= bruteforce_limit:
        brute_force = matrix_chain_bruteforce_cost(dims)

    sum_step_cost = int(sum(step.scalar_cost for step in result.steps))
    ltr_cost = left_to_right_cost(dims)

    print(f"=== {name} ===")
    print(f"dims={list(dims)} (n={result.n_matrices})")
    print(
        "optimal => "
        f"cost={result.min_scalar_multiplications}, "
        f"parenthesization={result.parenthesization}"
    )
    print(f"steps={format_steps(result.steps)}")
    print(
        "cross-check => "
        f"top_down={baseline}, "
        f"brute_force={('N/A' if brute_force is None else str(brute_force))}, "
        f"sum_step_cost={sum_step_cost}, "
        f"left_to_right_cost={ltr_cost}\n"
    )

    if result.min_scalar_multiplications != baseline:
        raise AssertionError("bottom-up DP and top-down baseline mismatch")
    if brute_force is not None and result.min_scalar_multiplications != brute_force:
        raise AssertionError("bottom-up DP and brute-force baseline mismatch")
    if result.min_scalar_multiplications != sum_step_cost:
        raise AssertionError("step costs do not sum to the optimal DP cost")
    if expected_cost is not None and result.min_scalar_multiplications != expected_cost:
        raise AssertionError(
            f"unexpected optimal cost: got {result.min_scalar_multiplications}, expected {expected_cost}"
        )

    if verify_numeric and max(dims) <= 40:
        mats = generate_random_chain_matrices(dims, seed=2026 + len(dims))
        optimal_product = multiply_by_split(mats, result.split_table, 0, result.n_matrices - 1)
        left_product = multiply_left_to_right(mats)
        if optimal_product.shape != left_product.shape:
            raise AssertionError("matrix product shape mismatch")
        if not np.allclose(optimal_product, left_product, atol=1e-10, rtol=1e-10):
            raise AssertionError("matrix product value mismatch")


def randomized_cross_check(
    trials: int = 200,
    max_matrices: int = 7,
    max_dim: int = 25,
    bruteforce_limit: int = 5,
    seed: int = 2026,
) -> None:
    rng = np.random.default_rng(seed)

    for _ in range(trials):
        n = int(rng.integers(1, max_matrices + 1))
        dims = rng.integers(2, max_dim + 1, size=n + 1).astype(int)

        result = matrix_chain_order(dims)
        baseline = matrix_chain_top_down_cost(dims)
        if result.min_scalar_multiplications != baseline:
            raise AssertionError("random check failed: DP vs top-down mismatch")

        if n <= bruteforce_limit:
            brute_force = matrix_chain_bruteforce_cost(dims)
            if result.min_scalar_multiplications != brute_force:
                raise AssertionError("random check failed: DP vs brute-force mismatch")

        sum_step_cost = int(sum(step.scalar_cost for step in result.steps))
        if result.min_scalar_multiplications != sum_step_cost:
            raise AssertionError("random check failed: step sum mismatch")

    print(
        f"Randomized cross-check passed: {trials} trials "
        f"(max_matrices={max_matrices}, max_dim={max_dim}, seed={seed})."
    )


def main() -> None:
    run_case(
        name="Case 1: CLRS classic",
        dims=[30, 35, 15, 5, 10, 20, 25],
        expected_cost=15125,
        bruteforce_limit=0,
        verify_numeric=True,
    )

    run_case(
        name="Case 2: textbook variant",
        dims=[10, 20, 30, 40, 30],
        expected_cost=30000,
        bruteforce_limit=6,
        verify_numeric=True,
    )

    run_case(
        name="Case 3: two matrices",
        dims=[5, 10, 3],
        expected_cost=150,
        bruteforce_limit=6,
        verify_numeric=True,
    )

    run_case(
        name="Case 4: one matrix",
        dims=[12, 8],
        expected_cost=0,
        bruteforce_limit=6,
        verify_numeric=False,
    )

    randomized_cross_check(
        trials=200,
        max_matrices=7,
        max_dim=25,
        bruteforce_limit=5,
        seed=2026,
    )


if __name__ == "__main__":
    main()
