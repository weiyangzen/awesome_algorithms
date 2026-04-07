"""最小路径和 MVP：二维 DP 主解 + 两种基线交叉验证。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

Coord = tuple[int, int]


@dataclass
class GridMinPathResult:
    min_sum: float
    path_indices: list[Coord]
    path_values: list[float]


def to_matrix(grid: Sequence[Sequence[float]]) -> np.ndarray:
    """校验并标准化网格输入。"""
    if len(grid) == 0:
        raise ValueError("grid must not be empty")

    row_lengths = [len(row) for row in grid]
    if any(length == 0 for length in row_lengths):
        raise ValueError("grid rows must not be empty")
    if len(set(row_lengths)) != 1:
        raise ValueError(f"grid must be rectangular, got row lengths={row_lengths}")

    mat = np.asarray(grid, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape={mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ValueError("grid contains non-finite values")

    return mat


def min_path_sum_dp(grid: Sequence[Sequence[float]]) -> GridMinPathResult:
    """自底向上表格 DP：返回最小路径和及一条对应路径。"""
    mat = to_matrix(grid)
    m, n = mat.shape

    dp = np.empty((m, n), dtype=float)
    parent = np.full((m, n, 2), -1, dtype=int)

    dp[0, 0] = float(mat[0, 0])

    for j in range(1, n):
        dp[0, j] = float(dp[0, j - 1] + mat[0, j])
        parent[0, j] = (0, j - 1)

    for i in range(1, m):
        dp[i, 0] = float(dp[i - 1, 0] + mat[i, 0])
        parent[i, 0] = (i - 1, 0)

    for i in range(1, m):
        for j in range(1, n):
            up = float(dp[i - 1, j])
            left = float(dp[i, j - 1])
            if up <= left:
                best_prev = up
                parent[i, j] = (i - 1, j)
            else:
                best_prev = left
                parent[i, j] = (i, j - 1)
            dp[i, j] = float(mat[i, j] + best_prev)

    rev_path: list[Coord] = []
    i, j = m - 1, n - 1
    while True:
        rev_path.append((i, j))
        if i == 0 and j == 0:
            break
        pi, pj = parent[i, j]
        i, j = int(pi), int(pj)

    path_indices = list(reversed(rev_path))
    path_values = [float(mat[r, c]) for r, c in path_indices]

    return GridMinPathResult(
        min_sum=float(dp[m - 1, n - 1]),
        path_indices=path_indices,
        path_values=path_values,
    )


def min_path_sum_top_down(grid: Sequence[Sequence[float]]) -> float:
    """记忆化递归基线，只返回最小路径和。"""
    mat = to_matrix(grid)
    m, n = mat.shape

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> float:
        if i == 0 and j == 0:
            return float(mat[0, 0])

        best = float("inf")
        if i > 0:
            best = min(best, solve(i - 1, j))
        if j > 0:
            best = min(best, solve(i, j - 1))
        return float(mat[i, j] + best)

    return solve(m - 1, n - 1)


def min_path_sum_bruteforce(grid: Sequence[Sequence[float]]) -> float:
    """暴力 DFS（无记忆化）基线，仅用于小规模验证。"""
    mat = to_matrix(grid)
    m, n = mat.shape

    def dfs(i: int, j: int) -> float:
        current = float(mat[i, j])
        if i == m - 1 and j == n - 1:
            return current

        best = float("inf")
        if i + 1 < m:
            best = min(best, dfs(i + 1, j))
        if j + 1 < n:
            best = min(best, dfs(i, j + 1))
        return current + best

    return dfs(0, 0)


def is_valid_grid_path(
    grid: Sequence[Sequence[float]], path_indices: Sequence[Coord]
) -> bool:
    mat = to_matrix(grid)
    m, n = mat.shape

    if len(path_indices) != (m + n - 1):
        return False
    if path_indices[0] != (0, 0):
        return False
    if path_indices[-1] != (m - 1, n - 1):
        return False

    for k in range(1, len(path_indices)):
        prev_i, prev_j = path_indices[k - 1]
        i, j = path_indices[k]
        di, dj = i - prev_i, j - prev_j
        if (di, dj) not in ((1, 0), (0, 1)):
            return False
        if not (0 <= i < m and 0 <= j < n):
            return False

    return True


def run_case(
    name: str,
    grid: Sequence[Sequence[float]],
    expected_min_sum: float | None = None,
    use_bruteforce: bool = True,
) -> None:
    result = min_path_sum_dp(grid)
    top_down = min_path_sum_top_down(grid)
    brute = min_path_sum_bruteforce(grid) if use_bruteforce else None

    print(f"=== {name} ===")
    print("grid:")
    for row in grid:
        print(f"  {list(row)}")
    print(
        "bottom-up => "
        f"min_sum={result.min_sum:.2f}, "
        f"path_indices={result.path_indices}, "
        f"path_values={result.path_values}"
    )
    print(f"top-down  => min_sum={top_down:.2f}")
    if brute is not None:
        print(f"bruteforce => min_sum={brute:.2f}")
    print()

    if not is_valid_grid_path(grid, result.path_indices):
        raise AssertionError("invalid path indices reconstructed")

    path_sum = float(sum(result.path_values))
    if abs(path_sum - result.min_sum) > 1e-9:
        raise AssertionError("path values sum does not match reported min_sum")

    if abs(result.min_sum - top_down) > 1e-9:
        raise AssertionError("bottom-up and top-down mismatch")

    if brute is not None and abs(result.min_sum - brute) > 1e-9:
        raise AssertionError("bottom-up and brute-force mismatch")

    if expected_min_sum is not None and abs(result.min_sum - expected_min_sum) > 1e-9:
        raise AssertionError(
            f"unexpected min_sum: got {result.min_sum}, expected {expected_min_sum}"
        )


def random_grid(
    rng: np.random.Generator, rows: int, cols: int, low: int = -8, high: int = 16
) -> list[list[int]]:
    return rng.integers(low, high, size=(rows, cols)).tolist()


def randomized_cross_check(
    trials: int = 300,
    max_rows: int = 6,
    max_cols: int = 6,
    seed: int = 2026,
) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        rows = int(rng.integers(1, max_rows + 1))
        cols = int(rng.integers(1, max_cols + 1))
        grid = random_grid(rng, rows=rows, cols=cols)

        main_res = min_path_sum_dp(grid)
        td = min_path_sum_top_down(grid)
        bf = min_path_sum_bruteforce(grid)

        if abs(main_res.min_sum - td) > 1e-9:
            raise AssertionError("random check failed: bottom-up vs top-down mismatch")
        if abs(main_res.min_sum - bf) > 1e-9:
            raise AssertionError("random check failed: bottom-up vs brute-force mismatch")
        if not is_valid_grid_path(grid, main_res.path_indices):
            raise AssertionError("random check failed: invalid reconstructed path")

    print(
        f"Randomized cross-check passed: {trials} trials "
        f"(max_rows={max_rows}, max_cols={max_cols}, seed={seed})."
    )


def main() -> None:
    run_case(
        name="Case 1: canonical example",
        grid=[
            [1, 3, 1],
            [1, 5, 1],
            [4, 2, 1],
        ],
        expected_min_sum=7.0,
    )

    run_case(
        name="Case 2: includes negative values",
        grid=[
            [1, -2, 4],
            [3, -5, 2],
            [6, 1, -1],
        ],
        expected_min_sum=-6.0,
    )

    run_case(
        name="Case 3: single cell",
        grid=[[42]],
        expected_min_sum=42.0,
    )

    randomized_cross_check(trials=300, max_rows=6, max_cols=6, seed=2026)


if __name__ == "__main__":
    main()
