"""CS-0245: DDA 直线算法最小可运行 MVP。"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Iterable

import numpy as np

Point = tuple[int, int]


def _round_half_away_from_zero(v: float) -> int:
    """实现 0.5 远离 0 的舍入规则。"""
    if v >= 0:
        return int(math.floor(v + 0.5))
    return int(math.ceil(v - 0.5))


def dda_line(x0: int, y0: int, x1: int, y1: int) -> list[Point]:
    """生成从 (x0, y0) 到 (x1, y1) 的 DDA 离散像素路径。"""
    values = (x0, y0, x1, y1)
    if not all(isinstance(v, Integral) for v in values):
        raise TypeError("x0, y0, x1, y1 must be integers")

    x0, y0, x1, y1 = (int(v) for v in values)
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))

    if steps == 0:
        return [(x0, y0)]

    x_inc = dx / steps
    y_inc = dy / steps

    x = float(x0)
    y = float(y0)
    points: list[Point] = []

    for _ in range(steps + 1):
        points.append((_round_half_away_from_zero(x), _round_half_away_from_zero(y)))
        x += x_inc
        y += y_inc

    return points


def render_ascii(points: Iterable[Point]) -> str:
    """把点序列渲染为 ASCII 网格。"""
    pts = list(points)
    if not pts:
        raise ValueError("points must be non-empty")

    xs = np.array([p[0] for p in pts], dtype=np.int64)
    ys = np.array([p[1] for p in pts], dtype=np.int64)
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    canvas = np.full((height, width), ".", dtype="<U1")

    for i, (x, y) in enumerate(pts):
        row = max_y - y
        col = x - min_x
        mark = "#"
        if i == 0:
            mark = "S"
        if i == len(pts) - 1:
            mark = "E" if i != 0 else "S"
        canvas[row, col] = mark

    return "\n".join("".join(row.tolist()) for row in canvas)


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


def validate_path_properties(points: list[Point], start: Point, end: Point) -> None:
    """验证 DDA 输出点序列的关键性质。"""
    if not points:
        raise AssertionError("Path is empty.")
    if points[0] != start:
        raise AssertionError(f"Start mismatch: expected {start}, got {points[0]}")
    if points[-1] != end:
        raise AssertionError(f"End mismatch: expected {end}, got {points[-1]}")

    dx = abs(end[0] - start[0])
    dy = abs(end[1] - start[1])
    expected_len = max(dx, dy) + 1
    if len(points) != expected_len:
        raise AssertionError(
            f"Length mismatch: expected {expected_len}, got {len(points)}"
        )

    sx = _sign(end[0] - start[0])
    sy = _sign(end[1] - start[1])
    allowed_dx = {0} if sx == 0 else {0, sx}
    allowed_dy = {0} if sy == 0 else {0, sy}

    for (x_prev, y_prev), (x_cur, y_cur) in zip(points, points[1:]):
        step_x = x_cur - x_prev
        step_y = y_cur - y_prev

        if step_x == 0 and step_y == 0:
            raise AssertionError("Zero step is not allowed.")
        if step_x not in allowed_dx:
            raise AssertionError(f"Invalid x step: {step_x}, allowed={allowed_dx}")
        if step_y not in allowed_dy:
            raise AssertionError(f"Invalid y step: {step_y}, allowed={allowed_dy}")

        if abs(step_x) > 1 or abs(step_y) > 1:
            raise AssertionError(f"Step too large: ({step_x}, {step_y})")


def run_fixed_regression_cases() -> None:
    """固定样例回归。"""
    cases: list[tuple[Point, Point, list[Point]]] = [
        ((0, 0), (5, 3), [(0, 0), (1, 1), (2, 1), (3, 2), (4, 2), (5, 3)]),
        ((0, 0), (3, 5), [(0, 0), (1, 1), (1, 2), (2, 3), (2, 4), (3, 5)]),
        ((0, 0), (5, -3), [(0, 0), (1, -1), (2, -1), (3, -2), (4, -2), (5, -3)]),
        ((2, 2), (2, 7), [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)]),
        ((1, 1), (6, 1), [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]),
        ((6, 4), (1, 1), [(6, 4), (5, 3), (4, 3), (3, 2), (2, 2), (1, 1)]),
        ((-2, 3), (1, -3), [(-2, 3), (-2, 2), (-1, 1), (-1, 0), (0, -1), (1, -2), (1, -3)]),
        ((4, 4), (4, 4), [(4, 4)]),
    ]

    for start, end, expected in cases:
        got = dda_line(start[0], start[1], end[0], end[1])
        if got != expected:
            raise AssertionError(
                f"Fixed case mismatch: {start}->{end}\nexpected={expected}\ngot={got}"
            )
        validate_path_properties(got, start, end)


def run_random_property_tests(seed: int = 245, n_cases: int = 300) -> None:
    """随机属性测试。"""
    rng = np.random.default_rng(seed)
    for _ in range(n_cases):
        x0, y0, x1, y1 = rng.integers(-20, 21, size=4, dtype=np.int64).tolist()
        start, end = (int(x0), int(y0)), (int(x1), int(y1))
        points = dda_line(*start, *end)
        validate_path_properties(points, start, end)


def main() -> None:
    run_fixed_regression_cases()
    run_random_property_tests()

    showcase_start = (-5, -2)
    showcase_end = (7, 4)
    showcase_points = dda_line(*showcase_start, *showcase_end)

    print(f"Showcase line: {showcase_start} -> {showcase_end}")
    print("Raster points:", showcase_points)
    print("ASCII preview:")
    print(render_ascii(showcase_points))
    print("All checks passed.")


if __name__ == "__main__":
    main()
