"""CS-0080 最近点对问题：分治法最小可运行 MVP。

实现内容:
1) closest_pair_divide_conquer: O(n log n) 分治算法。
2) closest_pair_naive: O(n^2) 朴素算法（用于对拍验证）。
3) main: 固定样例 + 随机回归 + 小规模性能对比。

运行方式:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Point:
    """内部点结构，使用 idx 保证同坐标点可区分。"""

    idx: int
    x: float
    y: float


def _squared_distance(a: Point, b: Point) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy


def _brute_force_points(points: Sequence[Point]) -> tuple[float, tuple[Point, Point] | None]:
    n = len(points)
    if n < 2:
        return float("inf"), None

    best_sq = float("inf")
    best_pair: tuple[Point, Point] | None = None

    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = _squared_distance(points[i], points[j])
            if dist_sq < best_sq:
                best_sq = dist_sq
                best_pair = (points[i], points[j])

    return best_sq, best_pair


def _closest_pair_rec(
    px: Sequence[Point],
    py: Sequence[Point],
) -> tuple[float, tuple[Point, Point] | None]:
    n = len(px)
    if n <= 3:
        return _brute_force_points(px)

    mid = n // 2
    mid_x = px[mid].x

    left_x = px[:mid]
    right_x = px[mid:]
    left_ids = {p.idx for p in left_x}

    left_y: list[Point] = []
    right_y: list[Point] = []
    for p in py:
        if p.idx in left_ids:
            left_y.append(p)
        else:
            right_y.append(p)

    left_sq, left_pair = _closest_pair_rec(left_x, left_y)
    right_sq, right_pair = _closest_pair_rec(right_x, right_y)

    if left_sq <= right_sq:
        delta_sq = left_sq
        best_pair = left_pair
    else:
        delta_sq = right_sq
        best_pair = right_pair

    strip = [p for p in py if (p.x - mid_x) * (p.x - mid_x) < delta_sq]

    # 经典性质: strip 按 y 排序后，每个点只需检查后续常数个点（常用上界 7）。
    for i in range(len(strip)):
        upper = min(i + 8, len(strip))
        for j in range(i + 1, upper):
            dy = strip[j].y - strip[i].y
            if dy * dy >= delta_sq:
                break
            dist_sq = _squared_distance(strip[i], strip[j])
            if dist_sq < delta_sq:
                delta_sq = dist_sq
                best_pair = (strip[i], strip[j])

    return delta_sq, best_pair


def closest_pair_divide_conquer(
    points_xy: Iterable[tuple[float, float]],
) -> tuple[float, tuple[tuple[float, float], tuple[float, float]]]:
    """返回最近点对距离与对应坐标点对（分治 O(n log n)）。"""
    points_list = [(float(x), float(y)) for x, y in points_xy]
    if len(points_list) < 2:
        raise ValueError("closest_pair_divide_conquer requires at least 2 points")

    points = [Point(idx=i, x=x, y=y) for i, (x, y) in enumerate(points_list)]
    px = sorted(points, key=lambda p: (p.x, p.y, p.idx))
    py = sorted(points, key=lambda p: (p.y, p.x, p.idx))

    best_sq, best_pair = _closest_pair_rec(px, py)
    if best_pair is None:
        raise RuntimeError("Unexpected empty result from closest-pair recursion")

    p1, p2 = best_pair
    return sqrt(best_sq), ((p1.x, p1.y), (p2.x, p2.y))


def closest_pair_naive(
    points_xy: Iterable[tuple[float, float]],
) -> tuple[float, tuple[tuple[float, float], tuple[float, float]]]:
    """朴素 O(n^2) 最近点对，用于对拍。"""
    points = [(float(x), float(y)) for x, y in points_xy]
    n = len(points)
    if n < 2:
        raise ValueError("closest_pair_naive requires at least 2 points")

    best_sq = float("inf")
    best_pair: tuple[tuple[float, float], tuple[float, float]] | None = None

    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_sq:
                best_sq = dist_sq
                best_pair = ((x1, y1), (x2, y2))

    if best_pair is None:
        raise RuntimeError("Unexpected empty result from naive closest-pair")

    return sqrt(best_sq), best_pair


def _run_fixed_cases() -> None:
    print("[Case 1] 固定样例")
    example = np.array(
        [
            [2.0, 3.0],
            [12.0, 30.0],
            [40.0, 50.0],
            [5.0, 1.0],
            [12.0, 10.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    fast_dist, fast_pair = closest_pair_divide_conquer(map(tuple, example.tolist()))
    slow_dist, _ = closest_pair_naive(map(tuple, example.tolist()))

    expected = sqrt(2.0)
    print(f"fast distance: {fast_dist:.12f}, pair: {fast_pair}")
    print(f"slow distance: {slow_dist:.12f}")

    if not np.isclose(fast_dist, expected, atol=1e-12):
        raise AssertionError(f"Expected {expected}, got {fast_dist}")
    if not np.isclose(fast_dist, slow_dist, atol=1e-12):
        raise AssertionError("Fixed-case mismatch between fast and naive.")

    duplicate_case = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]], dtype=np.float64)
    dup_dist, _ = closest_pair_divide_conquer(map(tuple, duplicate_case.tolist()))
    if dup_dist != 0.0:
        raise AssertionError(f"Expected duplicate-point distance 0.0, got {dup_dist}")


def _run_random_regression(seed: int = 80, rounds_per_size: int = 25) -> None:
    print("\n[Case 2] 随机对拍")
    rng = np.random.default_rng(seed)

    total = 0
    for n in [2, 3, 5, 8, 16, 32, 64, 128]:
        for _ in range(rounds_per_size):
            points = rng.uniform(-100.0, 100.0, size=(n, 2))
            fast_dist, _ = closest_pair_divide_conquer(map(tuple, points.tolist()))
            slow_dist, _ = closest_pair_naive(map(tuple, points.tolist()))

            if not np.isclose(fast_dist, slow_dist, atol=1e-10):
                raise AssertionError(
                    f"Mismatch for n={n}: fast={fast_dist}, slow={slow_dist}, points={points.tolist()}"
                )
            total += 1

    print(f"random cases={total}, seed={seed}: passed")


def _run_perf_snapshot(seed: int = 2026) -> None:
    print("\n[Case 3] 小规模性能快照")
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, 1.0, size=(800, 2))
    points_list = points.tolist()

    t0 = perf_counter()
    fast_dist, _ = closest_pair_divide_conquer(map(tuple, points_list))
    t1 = perf_counter()

    t2 = perf_counter()
    slow_dist, _ = closest_pair_naive(map(tuple, points_list))
    t3 = perf_counter()

    if not np.isclose(fast_dist, slow_dist, atol=1e-10):
        raise AssertionError("Performance snapshot mismatch between fast and naive.")

    print(f"n=800, fast={t1 - t0:.6f}s, naive={t3 - t2:.6f}s")


def main() -> None:
    _run_fixed_cases()
    _run_random_regression()
    _run_perf_snapshot()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
