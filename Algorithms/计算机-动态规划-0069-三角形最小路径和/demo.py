"""三角形最小路径和 MVP：动态规划主解 + 两种基线交叉验证。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np


@dataclass
class TriangleMinPathResult:
    min_sum: float
    path_indices: list[tuple[int, int]]
    path_values: list[float]


def to_triangle(triangle: Sequence[Sequence[float]]) -> list[np.ndarray]:
    """校验并规范化三角形输入。"""
    if len(triangle) == 0:
        raise ValueError("triangle must not be empty")

    rows: list[np.ndarray] = []
    for i, row in enumerate(triangle):
        arr = np.asarray(row, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"row {i} must be 1D, got shape={arr.shape}")
        expected_len = i + 1
        if int(arr.size) != expected_len:
            raise ValueError(
                f"row {i} length must be {expected_len}, got {arr.size}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"row {i} contains non-finite values")
        rows.append(arr)

    return rows


def triangle_min_path_bottom_up(
    triangle: Sequence[Sequence[float]],
) -> TriangleMinPathResult:
    """自底向上 DP：返回最小路径和及一条对应路径。"""
    rows = to_triangle(triangle)
    n = len(rows)

    if n == 1:
        value = float(rows[0][0])
        return TriangleMinPathResult(
            min_sum=value,
            path_indices=[(0, 0)],
            path_values=[value],
        )

    # dp[j] 表示“当前处理层下一行从位置 j 出发到底部的最小和”。
    dp = rows[-1].copy()

    # choices[r][c] 记录在 (r, c) 位置下一步应该走到第 r+1 行的哪一列。
    choices = [np.zeros(i + 1, dtype=int) for i in range(n - 1)]

    for r in range(n - 2, -1, -1):
        next_dp = np.empty(r + 1, dtype=float)
        for c in range(r + 1):
            left_child = float(dp[c])
            right_child = float(dp[c + 1])
            if left_child <= right_child:
                child_col = c
                child_best = left_child
            else:
                child_col = c + 1
                child_best = right_child

            choices[r][c] = child_col
            next_dp[c] = float(rows[r][c]) + child_best
        dp = next_dp

    path_indices: list[tuple[int, int]] = [(0, 0)]
    path_values: list[float] = [float(rows[0][0])]
    col = 0
    for r in range(0, n - 1):
        col = int(choices[r][col])
        path_indices.append((r + 1, col))
        path_values.append(float(rows[r + 1][col]))

    return TriangleMinPathResult(
        min_sum=float(dp[0]),
        path_indices=path_indices,
        path_values=path_values,
    )


def triangle_min_path_top_down(triangle: Sequence[Sequence[float]]) -> float:
    """记忆化递归基线，只返回最小路径和。"""
    rows = to_triangle(triangle)
    n = len(rows)

    @lru_cache(maxsize=None)
    def solve(r: int, c: int) -> float:
        if r == n - 1:
            return float(rows[r][c])

        down = solve(r + 1, c)
        down_right = solve(r + 1, c + 1)
        return float(rows[r][c]) + min(down, down_right)

    return solve(0, 0)


def triangle_min_path_bruteforce(triangle: Sequence[Sequence[float]]) -> float:
    """暴力 DFS（无记忆化）基线，仅用于小规模验证。"""
    rows = to_triangle(triangle)
    n = len(rows)

    def dfs(r: int, c: int) -> float:
        if r == n - 1:
            return float(rows[r][c])
        return float(rows[r][c]) + min(dfs(r + 1, c), dfs(r + 1, c + 1))

    return dfs(0, 0)


def is_valid_triangle_path(
    triangle: Sequence[Sequence[float]],
    path_indices: Sequence[tuple[int, int]],
) -> bool:
    rows = to_triangle(triangle)
    n = len(rows)

    if len(path_indices) != n:
        return False
    if path_indices[0] != (0, 0):
        return False

    for i in range(n):
        r, c = path_indices[i]
        if r != i:
            return False
        if c < 0 or c > r:
            return False
        if i > 0:
            _, prev_c = path_indices[i - 1]
            if c not in (prev_c, prev_c + 1):
                return False

    return True


def run_case(
    name: str,
    triangle: Sequence[Sequence[float]],
    expected_min_sum: float | None = None,
    use_bruteforce: bool = True,
) -> None:
    result = triangle_min_path_bottom_up(triangle)
    top_down = triangle_min_path_top_down(triangle)
    brute = triangle_min_path_bruteforce(triangle) if use_bruteforce else None

    print(f"=== {name} ===")
    print("triangle:")
    for row in triangle:
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

    if not is_valid_triangle_path(triangle, result.path_indices):
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


def random_triangle(rows: int, rng: np.random.Generator) -> list[list[int]]:
    return [rng.integers(-10, 21, size=r + 1).tolist() for r in range(rows)]


def randomized_cross_check(
    trials: int = 300,
    max_rows: int = 9,
    seed: int = 2026,
) -> None:
    rng = np.random.default_rng(seed)

    for _ in range(trials):
        rows = int(rng.integers(1, max_rows + 1))
        tri = random_triangle(rows, rng)

        main_res = triangle_min_path_bottom_up(tri)
        td = triangle_min_path_top_down(tri)
        bf = triangle_min_path_bruteforce(tri)

        if abs(main_res.min_sum - td) > 1e-9:
            raise AssertionError("random check failed: bottom-up vs top-down mismatch")
        if abs(main_res.min_sum - bf) > 1e-9:
            raise AssertionError("random check failed: bottom-up vs brute-force mismatch")
        if not is_valid_triangle_path(tri, main_res.path_indices):
            raise AssertionError("random check failed: invalid reconstructed path")

    print(
        f"Randomized cross-check passed: {trials} trials "
        f"(max_rows={max_rows}, seed={seed})."
    )


def main() -> None:
    run_case(
        name="Case 1: canonical positive triangle",
        triangle=[
            [2],
            [3, 4],
            [6, 5, 7],
            [4, 1, 8, 3],
        ],
        expected_min_sum=11.0,
    )

    run_case(
        name="Case 2: includes negative values",
        triangle=[
            [-1],
            [2, 3],
            [1, -1, -3],
        ],
        expected_min_sum=-1.0,
    )

    run_case(
        name="Case 3: single row",
        triangle=[[42]],
        expected_min_sum=42.0,
    )

    randomized_cross_check(trials=300, max_rows=9, seed=2026)


if __name__ == "__main__":
    main()
