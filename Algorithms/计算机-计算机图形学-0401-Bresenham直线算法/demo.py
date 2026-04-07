"""CS-0244: Bresenham 直线算法最小可运行 MVP。"""

from __future__ import annotations

from typing import Iterable

import numpy as np

Point = tuple[int, int]


def _sign(v: int) -> int:
    """返回 -1/0/1。"""
    return (v > 0) - (v < 0)


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[Point]:
    """生成从 (x0, y0) 到 (x1, y1) 的离散像素路径。

    采用全象限整数写法：仅用加减与比较，不用浮点运算。
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = _sign(x1 - x0)
    sy = _sign(y1 - y0)

    err = dx - dy
    x, y = x0, y0
    points: list[Point] = []

    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return points


def render_ascii(points: Iterable[Point]) -> str:
    """把点序列渲染为 ASCII 网格。

    约定：
    - S: 起点
    - E: 终点
    - #: 中间点
    - .: 空白
    """
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


def validate_path_properties(points: list[Point], start: Point, end: Point) -> None:
    """对输出路径做结构性断言。"""
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

    seen = set()
    min_x, max_x = sorted((start[0], end[0]))
    min_y, max_y = sorted((start[1], end[1]))

    for i, (x, y) in enumerate(points):
        if (x, y) in seen:
            raise AssertionError(f"Repeated point detected at index {i}: {(x, y)}")
        seen.add((x, y))

        if x < min_x or x > max_x or y < min_y or y > max_y:
            raise AssertionError(f"Point out of bounding box at index {i}: {(x, y)}")

    for (x_prev, y_prev), (x_cur, y_cur) in zip(points, points[1:]):
        step_x = x_cur - x_prev
        step_y = y_cur - y_prev

        if step_x == 0 and step_y == 0:
            raise AssertionError("Zero step is not allowed.")
        if step_x not in allowed_dx:
            raise AssertionError(f"Invalid x step: {step_x}, allowed={allowed_dx}")
        if step_y not in allowed_dy:
            raise AssertionError(f"Invalid y step: {step_y}, allowed={allowed_dy}")


def run_fixed_regression_cases() -> None:
    """固定回归样例：覆盖常见斜率与边界情形。"""
    cases: list[tuple[Point, Point, list[Point]]] = [
        ((0, 0), (5, 3), [(0, 0), (1, 1), (2, 1), (3, 2), (4, 2), (5, 3)]),
        ((0, 0), (3, 5), [(0, 0), (1, 1), (1, 2), (2, 3), (2, 4), (3, 5)]),
        ((0, 0), (5, -3), [(0, 0), (1, -1), (2, -1), (3, -2), (4, -2), (5, -3)]),
        ((2, 2), (2, 7), [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)]),
        ((1, 1), (6, 1), [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]),
        ((4, 4), (4, 4), [(4, 4)]),
    ]

    for start, end, expected in cases:
        got = bresenham_line(start[0], start[1], end[0], end[1])
        if got != expected:
            raise AssertionError(
                f"Fixed case mismatch: {start}->{end}\nexpected={expected}\ngot={got}"
            )
        validate_path_properties(got, start, end)


def run_random_property_tests(seed: int = 244, n_cases: int = 300) -> None:
    """随机属性测试：验证全象限与不同方向。"""
    rng = np.random.default_rng(seed)

    for _ in range(n_cases):
        x0, y0, x1, y1 = rng.integers(-20, 21, size=4, dtype=np.int64).tolist()
        start, end = (int(x0), int(y0)), (int(x1), int(y1))

        pts = bresenham_line(start[0], start[1], end[0], end[1])
        validate_path_properties(pts, start, end)


def main() -> None:
    run_fixed_regression_cases()
    run_random_property_tests()

    showcase_start = (-4, -1)
    showcase_end = (7, 5)
    showcase_points = bresenham_line(*showcase_start, *showcase_end)

    print(f"Showcase line: {showcase_start} -> {showcase_end}")
    print("Raster points:", showcase_points)
    print("ASCII preview:")
    print(render_ascii(showcase_points))

    print("All checks passed.")


if __name__ == "__main__":
    main()
