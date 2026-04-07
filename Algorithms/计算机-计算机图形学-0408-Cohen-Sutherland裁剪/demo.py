"""CS-0251 Cohen-Sutherland 线段裁剪最小可运行 MVP。

运行方式:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import numpy as np

LEFT = 1
RIGHT = 2
BOTTOM = 4
TOP = 8


@dataclass(frozen=True)
class Rect:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def validate(self) -> None:
        if not (self.xmin <= self.xmax and self.ymin <= self.ymax):
            raise ValueError(f"Invalid rect: {self}")


Point = tuple[float, float]
Segment = tuple[Point, Point]


def compute_outcode(x: float, y: float, rect: Rect) -> int:
    code = 0
    if x < rect.xmin:
        code |= LEFT
    elif x > rect.xmax:
        code |= RIGHT
    if y < rect.ymin:
        code |= BOTTOM
    elif y > rect.ymax:
        code |= TOP
    return code


def cohen_sutherland_clip(
    p1: Point,
    p2: Point,
    rect: Rect,
    eps: float = 1e-12,
) -> tuple[bool, Optional[Segment], int]:
    """返回 (是否可见, 裁剪后线段, 迭代次数)。"""
    rect.validate()

    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    code1 = compute_outcode(x1, y1, rect)
    code2 = compute_outcode(x2, y2, rect)

    iterations = 0
    while True:
        iterations += 1
        if (code1 | code2) == 0:
            return True, ((x1, y1), (x2, y2)), iterations
        if (code1 & code2) != 0:
            return False, None, iterations

        code_out = code1 if code1 != 0 else code2

        if code_out & TOP:
            dy = y2 - y1
            if abs(dy) < eps:
                return False, None, iterations
            x = x1 + (x2 - x1) * (rect.ymax - y1) / dy
            y = rect.ymax
        elif code_out & BOTTOM:
            dy = y2 - y1
            if abs(dy) < eps:
                return False, None, iterations
            x = x1 + (x2 - x1) * (rect.ymin - y1) / dy
            y = rect.ymin
        elif code_out & RIGHT:
            dx = x2 - x1
            if abs(dx) < eps:
                return False, None, iterations
            y = y1 + (y2 - y1) * (rect.xmax - x1) / dx
            x = rect.xmax
        else:  # code_out & LEFT
            dx = x2 - x1
            if abs(dx) < eps:
                return False, None, iterations
            y = y1 + (y2 - y1) * (rect.xmin - x1) / dx
            x = rect.xmin

        if code_out == code1:
            x1, y1 = x, y
            code1 = compute_outcode(x1, y1, rect)
        else:
            x2, y2 = x, y
            code2 = compute_outcode(x2, y2, rect)


def liang_barsky_clip(
    p1: Point,
    p2: Point,
    rect: Rect,
    eps: float = 1e-12,
) -> tuple[bool, Optional[Segment]]:
    """参数方程裁剪，作为对照实现。"""
    rect.validate()
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    dy = y2 - y1

    p = np.array([-dx, dx, -dy, dy], dtype=np.float64)
    q = np.array(
        [x1 - rect.xmin, rect.xmax - x1, y1 - rect.ymin, rect.ymax - y1],
        dtype=np.float64,
    )

    u1 = 0.0
    u2 = 1.0

    for pi, qi in zip(p, q):
        if abs(float(pi)) < eps:
            if qi < 0.0:
                return False, None
            continue

        t = float(qi / pi)
        if pi < 0:
            u1 = max(u1, t)
        else:
            u2 = min(u2, t)

        if u1 - u2 > eps:
            return False, None

    cx1 = x1 + u1 * dx
    cy1 = y1 + u1 * dy
    cx2 = x1 + u2 * dx
    cy2 = y1 + u2 * dy
    return True, ((cx1, cy1), (cx2, cy2))


def _segment_to_array(seg: Segment) -> np.ndarray:
    return np.array([[seg[0][0], seg[0][1]], [seg[1][0], seg[1][1]]], dtype=np.float64)


def _segments_equivalent(seg_a: Segment, seg_b: Segment, atol: float = 1e-9) -> bool:
    a = _segment_to_array(seg_a)
    b = _segment_to_array(seg_b)
    return bool(
        np.allclose(a, b, atol=atol, rtol=0.0)
        or np.allclose(a, b[::-1], atol=atol, rtol=0.0)
    )


def _run_fixed_cases(rect: Rect) -> None:
    print("[Case 1] 固定样例")
    cases = [
        ("inside", (1.0, 1.0), (9.0, 7.0), True, ((1.0, 1.0), (9.0, 7.0))),
        ("outside_same_side", (-5.0, 9.0), (-1.0, 10.0), False, None),
        ("cross_lr", (-4.0, 2.0), (12.0, 6.0), True, ((0.0, 3.0), (10.0, 5.5))),
        ("cross_tb", (3.0, -4.0), (7.0, 10.0), True, ((4.142857142857143, 0.0), (6.428571428571429, 8.0))),
        ("touch_corner", (-2.0, -2.0), (0.0, 0.0), True, ((0.0, 0.0), (0.0, 0.0))),
    ]

    for name, p1, p2, expected_ok, expected_seg in cases:
        ok, seg, loops = cohen_sutherland_clip(p1, p2, rect)
        if ok != expected_ok:
            raise AssertionError(f"{name}: expected ok={expected_ok}, got {ok}")
        if expected_seg is None:
            if seg is not None:
                raise AssertionError(f"{name}: expected None, got {seg}")
        else:
            if seg is None or not _segments_equivalent(seg, expected_seg):
                raise AssertionError(f"{name}: unexpected clipped segment: {seg}, expected {expected_seg}")
        print(f"  - {name:16s} ok={ok}, loops={loops}, seg={seg}")


def _run_random_regression(rect: Rect, seed: int = 251, n_cases: int = 2000) -> None:
    print("\n[Case 2] 随机回归对拍 (Cohen-Sutherland vs Liang-Barsky)")
    rng = np.random.default_rng(seed)
    accepted = 0
    max_loops = 0

    for _ in range(n_cases):
        p1 = tuple(rng.uniform(-15.0, 15.0, size=2).tolist())
        p2 = tuple(rng.uniform(-15.0, 15.0, size=2).tolist())

        ok_cs, seg_cs, loops = cohen_sutherland_clip(p1, p2, rect)
        ok_lb, seg_lb = liang_barsky_clip(p1, p2, rect)
        max_loops = max(max_loops, loops)

        if ok_cs != ok_lb:
            raise AssertionError(
                f"accept mismatch: cs={ok_cs}, lb={ok_lb}, p1={p1}, p2={p2}, cs_seg={seg_cs}, lb_seg={seg_lb}"
            )
        if ok_cs:
            accepted += 1
            if seg_cs is None or seg_lb is None:
                raise AssertionError(f"unexpected None segment for accepted case: p1={p1}, p2={p2}")
            if not _segments_equivalent(seg_cs, seg_lb, atol=1e-8):
                raise AssertionError(
                    f"segment mismatch: p1={p1}, p2={p2}, cs={seg_cs}, lb={seg_lb}"
                )

    print(f"random cases={n_cases}, accepted={accepted}, max_loops={max_loops}, seed={seed}: passed")


def _run_perf_snapshot(rect: Rect, seed: int = 2026, n_cases: int = 50_000) -> None:
    print("\n[Case 3] 性能快照")
    rng = np.random.default_rng(seed)
    p1s = rng.uniform(-20.0, 20.0, size=(n_cases, 2))
    p2s = rng.uniform(-20.0, 20.0, size=(n_cases, 2))

    t0 = perf_counter()
    cnt_cs = 0
    loop_sum = 0
    for i in range(n_cases):
        ok, _, loops = cohen_sutherland_clip(tuple(p1s[i]), tuple(p2s[i]), rect)
        cnt_cs += int(ok)
        loop_sum += loops
    t1 = perf_counter()

    t2 = perf_counter()
    cnt_lb = 0
    for i in range(n_cases):
        ok, _ = liang_barsky_clip(tuple(p1s[i]), tuple(p2s[i]), rect)
        cnt_lb += int(ok)
    t3 = perf_counter()

    if cnt_cs != cnt_lb:
        raise AssertionError(f"performance snapshot accept mismatch: cs={cnt_cs}, lb={cnt_lb}")

    print(
        f"n={n_cases}, cs_time={t1 - t0:.4f}s, lb_time={t3 - t2:.4f}s, "
        f"accept={cnt_cs}, avg_cs_loops={loop_sum / n_cases:.3f}"
    )


def main() -> None:
    rect = Rect(xmin=0.0, xmax=10.0, ymin=0.0, ymax=8.0)
    _run_fixed_cases(rect)
    _run_random_regression(rect)
    _run_perf_snapshot(rect)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
