"""最小路径和 MVP：动态规划主解 + 记忆化/暴力基线交叉验证。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np


@dataclass
class GridMinPathResult:
    min_sum: float
    path_cells: list[tuple[int, int]]
    path_values: list[float]


def to_grid(grid: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """校验并标准化网格输入。"""
    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"grid must be a 2D numeric matrix, got shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("grid must not be empty")
    rows, cols = arr.shape
    if rows <= 0 or cols <= 0:
        raise ValueError(f"grid dimensions must be positive, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("grid contains non-finite values")
    return arr


def min_path_sum_bottom_up(
    grid: Sequence[Sequence[float]] | np.ndarray,
) -> GridMinPathResult:
    """自底向上 DP：返回最小路径和及一条最优路径。"""
    arr = to_grid(grid)
    rows, cols = arr.shape

    dp = np.empty((rows, cols), dtype=float)
    parent_r = np.full((rows, cols), -1, dtype=int)
    parent_c = np.full((rows, cols), -1, dtype=int)

    dp[0, 0] = float(arr[0, 0])

    for c in range(1, cols):
        dp[0, c] = float(arr[0, c]) + float(dp[0, c - 1])
        parent_r[0, c] = 0
        parent_c[0, c] = c - 1

    for r in range(1, rows):
        dp[r, 0] = float(arr[r, 0]) + float(dp[r - 1, 0])
        parent_r[r, 0] = r - 1
        parent_c[r, 0] = 0

    for r in range(1, rows):
        for c in range(1, cols):
            up = float(dp[r - 1, c])
            left = float(dp[r, c - 1])
            if up <= left:
                best_prev = up
                pr, pc = r - 1, c
            else:
                best_prev = left
                pr, pc = r, c - 1

            dp[r, c] = float(arr[r, c]) + best_prev
            parent_r[r, c] = pr
            parent_c[r, c] = pc

    path_cells_rev: list[tuple[int, int]] = []
    path_values_rev: list[float] = []
    r, c = rows - 1, cols - 1
    while True:
        path_cells_rev.append((r, c))
        path_values_rev.append(float(arr[r, c]))
        pr = int(parent_r[r, c])
        pc = int(parent_c[r, c])
        if pr == -1 and pc == -1:
            break
        if pr < 0 or pc < 0:
            raise RuntimeError("invalid parent pointer during reconstruction")
        r, c = pr, pc

    path_cells = list(reversed(path_cells_rev))
    path_values = list(reversed(path_values_rev))
    return GridMinPathResult(
        min_sum=float(dp[rows - 1, cols - 1]),
        path_cells=path_cells,
        path_values=path_values,
    )


def min_path_sum_top_down(grid: Sequence[Sequence[float]] | np.ndarray) -> float:
    """记忆化递归基线：只返回最小路径和。"""
    arr = to_grid(grid)
    rows, cols = arr.shape

    @lru_cache(maxsize=None)
    def solve(r: int, c: int) -> float:
        if r == 0 and c == 0:
            return float(arr[0, 0])

        best = np.inf
        if r > 0:
            best = min(best, solve(r - 1, c))
        if c > 0:
            best = min(best, solve(r, c - 1))
        return float(arr[r, c]) + float(best)

    return solve(rows - 1, cols - 1)


def min_path_sum_bruteforce(grid: Sequence[Sequence[float]] | np.ndarray) -> float:
    """暴力 DFS（无记忆化）基线，仅用于小规模验证。"""
    arr = to_grid(grid)
    rows, cols = arr.shape

    def dfs(r: int, c: int) -> float:
        if r == rows - 1 and c == cols - 1:
            return float(arr[r, c])

        ans = np.inf
        if r + 1 < rows:
            ans = min(ans, float(arr[r, c]) + dfs(r + 1, c))
        if c + 1 < cols:
            ans = min(ans, float(arr[r, c]) + dfs(r, c + 1))
        return float(ans)

    return dfs(0, 0)


def is_valid_grid_path(
    shape: tuple[int, int],
    path_cells: Sequence[tuple[int, int]],
) -> bool:
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        return False
    if len(path_cells) != rows + cols - 1:
        return False
    if path_cells[0] != (0, 0):
        return False
    if path_cells[-1] != (rows - 1, cols - 1):
        return False

    for i in range(1, len(path_cells)):
        pr, pc = path_cells[i - 1]
        cr, cc = path_cells[i]
        if not (0 <= cr < rows and 0 <= cc < cols):
            return False
        dr, dc = cr - pr, cc - pc
        if (dr, dc) not in ((1, 0), (0, 1)):
            return False
    return True


def run_case(
    name: str,
    grid: Sequence[Sequence[float]],
    expected_min_sum: float | None = None,
    use_bruteforce: bool = True,
) -> None:
    arr = to_grid(grid)
    result = min_path_sum_bottom_up(arr)
    top_down = min_path_sum_top_down(arr)
    brute = min_path_sum_bruteforce(arr) if use_bruteforce else None

    print(f"=== {name} ===")
    print("grid:")
    for row in arr.tolist():
        print(f"  {row}")
    print(
        "bottom-up => "
        f"min_sum={result.min_sum:.2f}, "
        f"path_cells={result.path_cells}, "
        f"path_values={result.path_values}"
    )
    print(f"top-down  => min_sum={top_down:.2f}")
    if brute is not None:
        print(f"bruteforce => min_sum={brute:.2f}")
    print()

    if not is_valid_grid_path(arr.shape, result.path_cells):
        raise AssertionError("invalid path reconstructed")

    if abs(float(sum(result.path_values)) - result.min_sum) > 1e-9:
        raise AssertionError("path values sum does not match min_sum")
    if abs(result.min_sum - top_down) > 1e-9:
        raise AssertionError("bottom-up and top-down mismatch")
    if brute is not None and abs(result.min_sum - brute) > 1e-9:
        raise AssertionError("bottom-up and brute-force mismatch")
    if expected_min_sum is not None and abs(result.min_sum - expected_min_sum) > 1e-9:
        raise AssertionError(
            f"unexpected min_sum: got {result.min_sum}, expected {expected_min_sum}"
        )


def random_grid(rows: int, cols: int, rng: np.random.Generator) -> list[list[int]]:
    return rng.integers(-8, 16, size=(rows, cols)).tolist()


def randomized_cross_check(
    trials: int = 250,
    max_rows: int = 6,
    max_cols: int = 6,
    seed: int = 2026,
) -> None:
    rng = np.random.default_rng(seed)

    for _ in range(trials):
        rows = int(rng.integers(1, max_rows + 1))
        cols = int(rng.integers(1, max_cols + 1))
        grid = random_grid(rows, cols, rng)

        main_res = min_path_sum_bottom_up(grid)
        td = min_path_sum_top_down(grid)
        bf = min_path_sum_bruteforce(grid)

        if abs(main_res.min_sum - td) > 1e-9:
            raise AssertionError("random check failed: bottom-up vs top-down mismatch")
        if abs(main_res.min_sum - bf) > 1e-9:
            raise AssertionError("random check failed: bottom-up vs brute-force mismatch")
        if not is_valid_grid_path((rows, cols), main_res.path_cells):
            raise AssertionError("random check failed: invalid reconstructed path")
        if abs(float(sum(main_res.path_values)) - main_res.min_sum) > 1e-9:
            raise AssertionError("random check failed: path sum mismatch")

    print(
        f"Randomized cross-check passed: {trials} trials "
        f"(max_rows={max_rows}, max_cols={max_cols}, seed={seed})."
    )


def main() -> None:
    run_case(
        name="Case 1: canonical sample",
        grid=[
            [1, 3, 1],
            [1, 5, 1],
            [4, 2, 1],
        ],
        expected_min_sum=7.0,
    )

    run_case(
        name="Case 2: rectangular grid",
        grid=[
            [1, 2, 3],
            [4, 5, 6],
        ],
        expected_min_sum=12.0,
    )

    run_case(
        name="Case 3: includes negative values",
        grid=[
            [1, -3, 2],
            [2, 5, -10],
            [4, 2, 1],
        ],
        expected_min_sum=-9.0,
    )

    run_case(
        name="Case 4: single cell",
        grid=[[42]],
        expected_min_sum=42.0,
    )

    randomized_cross_check(trials=250, max_rows=6, max_cols=6, seed=2026)


if __name__ == "__main__":
    main()
