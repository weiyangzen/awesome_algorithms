"""CS-0081 凸包算法（分治法 / QuickHull）最小可运行 MVP。

实现内容:
1) convex_hull_quickhull: 分治版 QuickHull，返回逆时针凸包顶点序列。
2) convex_hull_monotonic_chain: O(n log n) 基线算法（Andrew），用于对拍。
3) main: 固定样例 + 随机回归 + 小规模性能快照。

运行方式:
    uv run python demo.py
"""

from __future__ import annotations

from time import perf_counter
from typing import Iterable

import numpy as np

Point = tuple[float, float]


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _signed_distance_to_line(a: Point, b: Point, p: Point) -> float:
    """等价于 2x 三角形有向面积（未归一化），足够用于比较远近和左右侧。"""
    return _cross(a, b, p)


def _deduplicate_points(points: Iterable[Point]) -> list[Point]:
    # 先归一为 float，再按集合去重，最后排序，确保结果稳定可复现。
    uniq = {(float(x), float(y)) for x, y in points}
    return sorted(uniq, key=lambda t: (t[0], t[1]))


def _find_hull(points: list[Point], a: Point, b: Point) -> list[Point]:
    """递归求由有向边 a->b 与点集 points 形成的外侧凸包链。"""
    if not points:
        return []

    farthest = max(points, key=lambda p: abs(_signed_distance_to_line(a, b, p)))

    left_of_ap = [p for p in points if _signed_distance_to_line(a, farthest, p) > 0]
    left_of_pb = [p for p in points if _signed_distance_to_line(farthest, b, p) > 0]

    # a -> farthest -> b 顺序拼接，保持链路拓扑顺序。
    return _find_hull(left_of_ap, a, farthest) + [farthest] + _find_hull(left_of_pb, farthest, b)


def convex_hull_quickhull(points: Iterable[Point]) -> list[Point]:
    """QuickHull（分治）求二维点集凸包，返回逆时针顶点，不重复首尾点。"""
    pts = _deduplicate_points(points)
    n = len(pts)
    if n <= 1:
        return pts
    if n == 2:
        return pts

    leftmost = min(pts, key=lambda p: (p[0], p[1]))
    rightmost = max(pts, key=lambda p: (p[0], p[1]))

    if leftmost == rightmost:
        return [leftmost]

    upper = [p for p in pts if _signed_distance_to_line(leftmost, rightmost, p) > 0]
    lower = [p for p in pts if _signed_distance_to_line(leftmost, rightmost, p) < 0]

    # 逆时针顺序: leftmost -> upper chain -> rightmost -> lower chain(反向边方向计算)。
    hull = (
        [leftmost]
        + _find_hull(upper, leftmost, rightmost)
        + [rightmost]
        + _find_hull(lower, rightmost, leftmost)
    )

    # 再做一次去重，防止极端退化输入下重复点进入链路。
    seen: set[Point] = set()
    deduped: list[Point] = []
    for p in hull:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    if len(deduped) >= 3 and _polygon_area2(deduped) < 0:
        deduped.reverse()
    return deduped


def convex_hull_monotonic_chain(points: Iterable[Point]) -> list[Point]:
    """Andrew 单调链基线算法，返回逆时针凸包顶点，不重复首尾点。"""
    pts = _deduplicate_points(points)
    if len(pts) <= 1:
        return pts

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # 去掉重复端点。
    return lower[:-1] + upper[:-1]


def _polygon_area2(poly: list[Point]) -> float:
    """返回 2x 多边形有向面积（凸包通常为正）。"""
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - y1 * x2
    return s


def _normalize_hull_as_set(hull: list[Point]) -> set[Point]:
    # 统一小数位，避免极少量浮点噪声影响集合比较。
    return {(round(x, 12), round(y, 12)) for x, y in hull}


def _run_fixed_case() -> None:
    print("[Case 1] 固定样例")
    points = [
        (0.0, 0.0),
        (2.0, 0.0),
        (3.0, 1.0),
        (2.0, 3.0),
        (0.0, 2.0),
        (1.0, 1.0),  # 内点
        (1.5, 1.5),  # 内点
    ]
    expected = {(0.0, 0.0), (2.0, 0.0), (3.0, 1.0), (2.0, 3.0), (0.0, 2.0)}

    hull = convex_hull_quickhull(points)
    hull_set = _normalize_hull_as_set(hull)
    print("quickhull hull:", hull)

    if hull_set != expected:
        raise AssertionError(f"Fixed case mismatch: got={hull_set}, expected={expected}")

    if _polygon_area2(hull) <= 0:
        raise AssertionError("Hull should be in counter-clockwise order with positive area.")


def _run_random_regression(seed: int = 81, rounds_per_size: int = 20) -> None:
    print("\n[Case 2] 随机对拍 (QuickHull vs Monotonic Chain)")
    rng = np.random.default_rng(seed)
    total = 0

    for n in [5, 8, 16, 32, 64, 128, 256]:
        for _ in range(rounds_per_size):
            points = rng.uniform(-100.0, 100.0, size=(n, 2))

            # 注入少量重复点，验证健壮性。
            if n >= 16:
                points[:3] = points[5]

            pts = [tuple(map(float, p)) for p in points.tolist()]
            hull_fast = convex_hull_quickhull(pts)
            hull_base = convex_hull_monotonic_chain(pts)

            set_fast = _normalize_hull_as_set(hull_fast)
            set_base = _normalize_hull_as_set(hull_base)
            if set_fast != set_base:
                raise AssertionError(
                    f"Hull mismatch for n={n}: quickhull={set_fast}, baseline={set_base}"
                )
            total += 1

    print(f"random cases={total}, seed={seed}: passed")


def _run_perf_snapshot(seed: int = 2026) -> None:
    print("\n[Case 3] 小规模性能快照")
    rng = np.random.default_rng(seed)
    points = rng.uniform(-10_000.0, 10_000.0, size=(2_000, 2))
    pts = [tuple(map(float, p)) for p in points.tolist()]

    t0 = perf_counter()
    hull_fast = convex_hull_quickhull(pts)
    t1 = perf_counter()

    t2 = perf_counter()
    hull_base = convex_hull_monotonic_chain(pts)
    t3 = perf_counter()

    if _normalize_hull_as_set(hull_fast) != _normalize_hull_as_set(hull_base):
        raise AssertionError("Performance snapshot mismatch between quickhull and baseline.")

    print(
        f"n=2000, quickhull={t1 - t0:.6f}s, baseline(monotonic_chain)={t3 - t2:.6f}s, "
        f"hull_vertices={len(hull_fast)}"
    )


def main() -> None:
    _run_fixed_case()
    _run_random_regression()
    _run_perf_snapshot()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
