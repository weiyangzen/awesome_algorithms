"""CS-0087: 搜索二维矩阵II（分治）最小可运行 MVP。

实现目标：
1) 实现子矩阵裁剪型分治搜索；
2) 提供 O(m+n) 楼梯搜索作为基线；
3) 运行脚本时自动执行固定样例、随机回归和小基准（无交互输入）。
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class SearchStats:
    recursive_calls: int = 0
    comparisons: int = 0
    range_prunes: int = 0
    max_depth: int = 0


@dataclass
class SearchResult:
    found: bool
    row: int
    col: int
    stats: SearchStats


def _to_2d_int_array(matrix: Iterable[Iterable[int]]) -> np.ndarray:
    rows = [list(row) for row in matrix]

    if not rows:
        return np.empty((0, 0), dtype=np.int64)

    n_cols = len(rows[0])
    for row in rows:
        if len(row) != n_cols:
            raise ValueError("Input matrix must be rectangular (all rows same length).")

    if n_cols == 0:
        return np.empty((len(rows), 0), dtype=np.int64)

    return np.asarray(rows, dtype=np.int64)


def search_matrix_divide_conquer(matrix: Iterable[Iterable[int]], target: int) -> SearchResult:
    """分治查找：按中列扫描 + 两个候选子矩阵递归。"""
    arr = _to_2d_int_array(matrix)
    stats = SearchStats()

    if arr.size == 0:
        return SearchResult(found=False, row=-1, col=-1, stats=stats)

    n_rows, n_cols = arr.shape

    def solve(top: int, bottom: int, left: int, right: int, depth: int) -> tuple[int, int]:
        stats.recursive_calls += 1
        stats.max_depth = max(stats.max_depth, depth)

        if top > bottom or left > right:
            return -1, -1

        stats.comparisons += 2
        if target < int(arr[top, left]) or target > int(arr[bottom, right]):
            stats.range_prunes += 1
            return -1, -1

        mid_col = (left + right) // 2
        scan_row = top

        while scan_row <= bottom:
            stats.comparisons += 1
            value = int(arr[scan_row, mid_col])
            if value == target:
                return scan_row, mid_col
            if value > target:
                break
            scan_row += 1

        # 候选空间被切成右上与左下两个互补子矩阵。
        r, c = solve(top, scan_row - 1, mid_col + 1, right, depth + 1)
        if r != -1:
            return r, c

        return solve(scan_row, bottom, left, mid_col - 1, depth + 1)

    row, col = solve(0, n_rows - 1, 0, n_cols - 1, 1)
    return SearchResult(found=(row != -1), row=int(row), col=int(col), stats=stats)


def search_matrix_staircase(matrix: Iterable[Iterable[int]], target: int) -> tuple[bool, int, int]:
    """楼梯搜索基线：从右上角出发，O(m+n)。"""
    arr = _to_2d_int_array(matrix)
    if arr.size == 0:
        return False, -1, -1

    n_rows, n_cols = arr.shape
    r, c = 0, n_cols - 1

    while r < n_rows and c >= 0:
        value = int(arr[r, c])
        if value == target:
            return True, r, c
        if value > target:
            c -= 1
        else:
            r += 1

    return False, -1, -1


def search_matrix_bruteforce(matrix: Iterable[Iterable[int]], target: int) -> tuple[bool, int, int]:
    """暴力真值器：用于回归测试对拍。"""
    arr = _to_2d_int_array(matrix)
    if arr.size == 0:
        return False, -1, -1

    positions = np.argwhere(arr == target)
    if positions.size == 0:
        return False, -1, -1

    r, c = positions[0]
    return True, int(r), int(c)


def _make_sorted_matrix(rng: np.random.Generator, n_rows: int, n_cols: int) -> np.ndarray:
    """构造行列都严格递增的矩阵。"""
    if n_rows <= 0 or n_cols <= 0:
        return np.empty((max(n_rows, 0), max(n_cols, 0)), dtype=np.int64)

    base = rng.integers(1, 5, size=(n_rows, n_cols), dtype=np.int64)
    # 双重前缀和保证向右、向下都严格增长。
    return np.cumsum(np.cumsum(base, axis=0), axis=1)


def _check_hit(arr: np.ndarray, found: bool, row: int, col: int, target: int) -> None:
    if not found:
        return
    if not (0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]):
        raise AssertionError(f"Hit index out of bounds: ({row}, {col})")
    if int(arr[row, col]) != target:
        raise AssertionError(
            f"Hit mismatch at ({row}, {col}): got {arr[row, col]}, target {target}"
        )


def run_fixed_cases() -> None:
    matrix = np.array(
        [
            [1, 4, 7, 11, 15],
            [2, 5, 8, 12, 19],
            [3, 6, 9, 16, 22],
            [10, 13, 14, 17, 24],
            [18, 21, 23, 26, 30],
        ],
        dtype=np.int64,
    )

    cases = [
        (matrix, 5, True),
        (matrix, 20, False),
        (np.array([[7]], dtype=np.int64), 7, True),
        (np.array([[7]], dtype=np.int64), 6, False),
        (np.array([[1, 3, 5, 9]], dtype=np.int64), 5, True),
        (np.array([[1], [2], [4], [8]], dtype=np.int64), 3, False),
        (np.empty((0, 0), dtype=np.int64), 1, False),
    ]

    print("[Fixed cases]")
    for arr, target, expected in cases:
        fast = search_matrix_divide_conquer(arr, target)
        slow_found, slow_r, slow_c = search_matrix_staircase(arr, target)
        brute_found, brute_r, brute_c = search_matrix_bruteforce(arr, target)

        if fast.found != expected:
            raise AssertionError(
                f"Expected={expected}, got={fast.found}, target={target}, matrix={arr.tolist()}"
            )

        if fast.found != slow_found or slow_found != brute_found:
            raise AssertionError(
                "Mismatch among methods: "
                f"fast={fast.found}, staircase={slow_found}, brute={brute_found}, target={target}"
            )

        _check_hit(arr, fast.found, fast.row, fast.col, target)
        _check_hit(arr, slow_found, slow_r, slow_c, target)
        _check_hit(arr, brute_found, brute_r, brute_c, target)

        print(
            f"target={target:>3}, found={fast.found}, "
            f"fast_pos=({fast.row},{fast.col}), calls={fast.stats.recursive_calls}, "
            f"prunes={fast.stats.range_prunes}, comps={fast.stats.comparisons}"
        )


def run_random_regression() -> None:
    rng = np.random.default_rng(87)
    shapes = [(1, 1), (1, 6), (6, 1), (3, 4), (4, 7), (8, 8), (12, 16)]
    total = 0

    for n_rows, n_cols in shapes:
        for _ in range(40):
            arr = _make_sorted_matrix(rng, n_rows, n_cols)

            # 约 60% 采样命中目标，40% 采样不命中目标。
            if rng.random() < 0.6:
                r = int(rng.integers(0, n_rows))
                c = int(rng.integers(0, n_cols))
                target = int(arr[r, c])
            else:
                if rng.random() < 0.5:
                    target = int(arr[0, 0] - int(rng.integers(1, 6)))
                else:
                    target = int(arr[-1, -1] + int(rng.integers(1, 6)))

            fast = search_matrix_divide_conquer(arr, target)
            slow_found, slow_r, slow_c = search_matrix_staircase(arr, target)
            brute_found, brute_r, brute_c = search_matrix_bruteforce(arr, target)

            if fast.found != slow_found or slow_found != brute_found:
                raise AssertionError(
                    f"Random mismatch: shape=({n_rows},{n_cols}), target={target}, "
                    f"fast={fast.found}, staircase={slow_found}, brute={brute_found}"
                )

            _check_hit(arr, fast.found, fast.row, fast.col, target)
            _check_hit(arr, slow_found, slow_r, slow_c, target)
            _check_hit(arr, brute_found, brute_r, brute_c, target)
            total += 1

    print(f"Random cross-check cases: {total}")


def run_benchmark() -> None:
    rng = np.random.default_rng(2026)
    sizes = [8, 16, 24, 32, 48, 64]
    repeats = 50
    rows: list[dict[str, float]] = []

    for n in sizes:
        matrices = [_make_sorted_matrix(rng, n, n) for _ in range(repeats)]
        targets = [
            int(mat[int(rng.integers(0, n)), int(rng.integers(0, n))]) if rng.random() < 0.5 else int(mat[-1, -1] + 3)
            for mat in matrices
        ]

        t0 = perf_counter()
        for mat, target in zip(matrices, targets):
            _ = search_matrix_divide_conquer(mat, target)
        t1 = perf_counter()

        for mat, target in zip(matrices, targets):
            _ = search_matrix_staircase(mat, target)
        t2 = perf_counter()

        dc_ms = (t1 - t0) * 1000.0 / repeats
        stair_ms = (t2 - t1) * 1000.0 / repeats

        rows.append(
            {
                "n": float(n),
                "divide_conquer_ms": dc_ms,
                "staircase_ms": stair_ms,
                "stair_over_dc": (stair_ms / dc_ms) if dc_ms > 0 else float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n[Benchmark] average time per case (ms)")
    print(df.to_string(index=False))


def main() -> None:
    run_fixed_cases()
    run_random_regression()
    run_benchmark()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
