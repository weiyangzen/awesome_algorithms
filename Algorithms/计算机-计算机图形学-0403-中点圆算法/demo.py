"""CS-0246: 中点圆算法最小可运行 MVP。"""

from __future__ import annotations

from numbers import Integral
from typing import Iterable

import numpy as np

Point = tuple[int, int]


def _eight_symmetric_points(cx: int, cy: int, x: int, y: int) -> tuple[Point, ...]:
    """返回相对圆心 (cx, cy) 的 8 对称像素。"""
    return (
        (cx + x, cy + y),
        (cx + y, cy + x),
        (cx + y, cy - x),
        (cx + x, cy - y),
        (cx - x, cy - y),
        (cx - y, cy - x),
        (cx - y, cy + x),
        (cx - x, cy + y),
    )


def midpoint_circle(cx: int, cy: int, radius: int) -> list[Point]:
    """生成整数圆周像素点（中点圆算法）。"""
    values = (cx, cy, radius)
    if not all(isinstance(v, Integral) for v in values):
        raise TypeError("cx, cy, radius must be integers")

    cx, cy, radius = (int(v) for v in values)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return [(cx, cy)]

    x = 0
    y = radius
    decision = 1 - radius

    points: list[Point] = []
    seen: set[Point] = set()

    while x <= y:
        for p in _eight_symmetric_points(cx, cy, x, y):
            if p not in seen:
                seen.add(p)
                points.append(p)

        x += 1
        if decision < 0:
            decision += 2 * x + 1
        else:
            y -= 1
            decision += 2 * (x - y) + 1

    return points


def midpoint_circle_octant_trace(radius: int) -> list[tuple[int, int, int]]:
    """返回第一八分圆迭代轨迹: (x, y, decision_before_update)。"""
    if not isinstance(radius, Integral):
        raise TypeError("radius must be an integer")
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")

    x = 0
    y = radius
    decision = 1 - radius
    trace: list[tuple[int, int, int]] = []

    while x <= y:
        trace.append((x, y, decision))
        x += 1
        if decision < 0:
            decision += 2 * x + 1
        else:
            y -= 1
            decision += 2 * (x - y) + 1

    return trace


def render_ascii(points: Iterable[Point], center: Point | None = None) -> str:
    """将圆周点渲染为 ASCII 字符网格。"""
    pts = list(points)
    if not pts:
        raise ValueError("points must be non-empty")

    all_pts = list(pts)
    if center is not None:
        all_pts.append(center)

    xs = np.array([p[0] for p in all_pts], dtype=np.int64)
    ys = np.array([p[1] for p in all_pts], dtype=np.int64)

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    canvas = np.full((height, width), ".", dtype="<U1")

    for x, y in pts:
        row = max_y - y
        col = x - min_x
        canvas[row, col] = "#"

    if center is not None:
        row = max_y - center[1]
        col = center[0] - min_x
        canvas[row, col] = "C"

    return "\n".join("".join(row.tolist()) for row in canvas)


def _expected_from_trace(center: Point, trace: list[tuple[int, int, int]]) -> set[Point]:
    cx, cy = center
    expected: set[Point] = set()
    for x, y, _ in trace:
        expected.update(_eight_symmetric_points(cx, cy, x, y))
    return expected


def validate_circle_properties(points: list[Point], center: Point, radius: int) -> None:
    """验证中点圆输出点集的关键性质。"""
    cx, cy = center

    if radius == 0:
        if points != [center]:
            raise AssertionError(f"radius=0 must return [center], got {points}")
        return

    if not points:
        raise AssertionError("Point list is empty.")

    point_set = set(points)
    if len(point_set) != len(points):
        raise AssertionError("Duplicate points detected.")

    # 边界盒与圆方程误差检查。
    max_d2_err = max(2 * radius, 1)
    for x, y in points:
        dx = x - cx
        dy = y - cy

        if abs(dx) > radius or abs(dy) > radius:
            raise AssertionError(f"Point out of radius bbox: {(x, y)}")

        d2 = dx * dx + dy * dy
        if abs(d2 - radius * radius) > max_d2_err:
            raise AssertionError(
                f"Point too far from circle: {(x, y)}, d2={d2}, r2={radius * radius}"
            )

    # 对称性 + 八分圆轨迹一致性。
    trace = midpoint_circle_octant_trace(radius)
    expected = _expected_from_trace(center, trace)
    if point_set != expected:
        missing = sorted(expected - point_set)
        extras = sorted(point_set - expected)
        raise AssertionError(
            "Point-set mismatch with octant symmetry reconstruction:\n"
            f"missing={missing}\nextras={extras}"
        )


def _shift_points(base: set[Point], dx: int, dy: int) -> set[Point]:
    return {(x + dx, y + dy) for x, y in base}


def run_fixed_regression_cases() -> None:
    """固定回归：半径 0/1/2/3 的可枚举结果。"""
    base_r0 = {(0, 0)}
    base_r1 = {(0, 1), (1, 0), (0, -1), (-1, 0)}
    base_r2 = {
        (0, 2),
        (1, 2),
        (2, 1),
        (2, 0),
        (2, -1),
        (1, -2),
        (0, -2),
        (-1, -2),
        (-2, -1),
        (-2, 0),
        (-2, 1),
        (-1, 2),
    }
    base_r3 = {
        (0, 3),
        (1, 3),
        (2, 2),
        (3, 1),
        (3, 0),
        (3, -1),
        (2, -2),
        (1, -3),
        (0, -3),
        (-1, -3),
        (-2, -2),
        (-3, -1),
        (-3, 0),
        (-3, 1),
        (-2, 2),
        (-1, 3),
    }

    cases: list[tuple[Point, int, set[Point]]] = [
        ((0, 0), 0, base_r0),
        ((0, 0), 1, base_r1),
        ((0, 0), 2, base_r2),
        ((0, 0), 3, base_r3),
        ((3, -2), 3, _shift_points(base_r3, 3, -2)),
    ]

    for center, radius, expected in cases:
        got = midpoint_circle(center[0], center[1], radius)
        got_set = set(got)
        if got_set != expected:
            raise AssertionError(
                f"Fixed case mismatch: center={center}, radius={radius}\n"
                f"expected={sorted(expected)}\n"
                f"got={sorted(got_set)}"
            )
        validate_circle_properties(got, center, radius)


def run_random_property_tests(seed: int = 246, n_cases: int = 300) -> None:
    """随机性质测试：覆盖随机圆心和半径。"""
    rng = np.random.default_rng(seed)

    for _ in range(n_cases):
        cx, cy = rng.integers(-25, 26, size=2, dtype=np.int64).tolist()
        radius = int(rng.integers(0, 26, dtype=np.int64))

        center = (int(cx), int(cy))
        pts = midpoint_circle(center[0], center[1], radius)
        validate_circle_properties(pts, center, radius)


def main() -> None:
    run_fixed_regression_cases()
    run_random_property_tests()

    showcase_center = (2, -1)
    showcase_radius = 8
    showcase_points = midpoint_circle(showcase_center[0], showcase_center[1], showcase_radius)
    trace = midpoint_circle_octant_trace(showcase_radius)

    print(f"Showcase circle: center={showcase_center}, radius={showcase_radius}")
    print(f"Pixel count: {len(showcase_points)}")
    print("First-octant trace (x, y, decision):", trace)
    print("ASCII preview (#=circle, C=center):")
    print(render_ascii(showcase_points, center=showcase_center))
    print("All checks passed.")


if __name__ == "__main__":
    main()
