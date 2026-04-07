"""Minimal runnable MVP for Cohen-Sutherland line clipping.

This script:
1. Implements Cohen-Sutherland clipping against an axis-aligned rectangle;
2. Builds deterministic edge-case + random test segments;
3. Cross-checks results with a hand-written Liang-Barsky reference implementation;
4. Prints a compact detail table and summary table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12

LEFT = 1
RIGHT = 2
BOTTOM = 4
TOP = 8


@dataclass(frozen=True)
class ClipWindow:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains(self, point: np.ndarray, tol: float = 1e-9) -> bool:
        x, y = float(point[0]), float(point[1])
        return (
            self.xmin - tol <= x <= self.xmax + tol
            and self.ymin - tol <= y <= self.ymax + tol
        )


@dataclass(frozen=True)
class ClipResult:
    accepted: bool
    p0: np.ndarray | None
    p1: np.ndarray | None
    reason: str
    iterations: int


@dataclass(frozen=True)
class Segment:
    idx: int
    p0: np.ndarray
    p1: np.ndarray
    tag: str


def region_code(window: ClipWindow, point: np.ndarray) -> int:
    x, y = float(point[0]), float(point[1])
    code = 0
    if x < window.xmin:
        code |= LEFT
    elif x > window.xmax:
        code |= RIGHT

    if y < window.ymin:
        code |= BOTTOM
    elif y > window.ymax:
        code |= TOP
    return code


def cohen_sutherland_clip(
    window: ClipWindow,
    p0: np.ndarray,
    p1: np.ndarray,
    max_iters: int = 24,
) -> ClipResult:
    """Clip one segment with Cohen-Sutherland iterative outcode method."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])

    for it in range(1, max_iters + 1):
        code0 = region_code(window, np.array([x0, y0], dtype=np.float64))
        code1 = region_code(window, np.array([x1, y1], dtype=np.float64))

        if (code0 | code1) == 0:
            return ClipResult(
                True,
                np.array([x0, y0], dtype=np.float64),
                np.array([x1, y1], dtype=np.float64),
                "accepted",
                it,
            )

        if code0 & code1:
            return ClipResult(False, None, None, "trivial_reject", it)

        out_code = code0 if code0 != 0 else code1
        dx = x1 - x0
        dy = y1 - y0

        if out_code & TOP:
            if abs(dy) <= EPS:
                return ClipResult(False, None, None, "numeric_parallel_top", it)
            x = x0 + dx * (window.ymax - y0) / dy
            y = window.ymax
        elif out_code & BOTTOM:
            if abs(dy) <= EPS:
                return ClipResult(False, None, None, "numeric_parallel_bottom", it)
            x = x0 + dx * (window.ymin - y0) / dy
            y = window.ymin
        elif out_code & RIGHT:
            if abs(dx) <= EPS:
                return ClipResult(False, None, None, "numeric_parallel_right", it)
            y = y0 + dy * (window.xmax - x0) / dx
            x = window.xmax
        else:
            if abs(dx) <= EPS:
                return ClipResult(False, None, None, "numeric_parallel_left", it)
            y = y0 + dy * (window.xmin - x0) / dx
            x = window.xmin

        if out_code == code0:
            x0, y0 = x, y
        else:
            x1, y1 = x, y

    return ClipResult(False, None, None, "max_iter_guard", max_iters)


def liang_barsky_reference(window: ClipWindow, p0: np.ndarray, p1: np.ndarray) -> ClipResult:
    """Reference clipper for correctness cross-check."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx = x1 - x0
    dy = y1 - y0

    p = np.array([-dx, dx, -dy, dy], dtype=np.float64)
    q = np.array(
        [x0 - window.xmin, window.xmax - x0, y0 - window.ymin, window.ymax - y0],
        dtype=np.float64,
    )

    u_enter = 0.0
    u_leave = 1.0
    for pi, qi in zip(p, q):
        if abs(pi) <= EPS:
            if qi < 0.0:
                return ClipResult(False, None, None, "parallel_outside", 1)
            continue

        r = qi / pi
        if pi < 0.0:
            u_enter = max(u_enter, r)
        else:
            u_leave = min(u_leave, r)

        if u_enter - u_leave > EPS:
            return ClipResult(False, None, None, "empty_interval", 1)

    c0 = p0 + u_enter * (p1 - p0)
    c1 = p0 + u_leave * (p1 - p0)
    return ClipResult(True, c0, c1, "accepted", 1)


def segment_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a))


def build_segments(seed: int = 2026, random_count: int = 44) -> list[Segment]:
    manual = [
        Segment(0, np.array([1.0, 1.0]), np.array([9.0, 7.0]), "inside"),
        Segment(1, np.array([-3.0, 4.0]), np.array([12.0, 4.0]), "horizontal_cross"),
        Segment(2, np.array([5.0, -2.0]), np.array([5.0, 11.0]), "vertical_cross"),
        Segment(3, np.array([-3.0, -2.0]), np.array([-1.0, -1.0]), "outside_disjoint"),
        Segment(4, np.array([0.0, 0.0]), np.array([10.0, 8.0]), "corner_to_corner"),
        Segment(5, np.array([-2.0, 8.0]), np.array([12.0, 8.0]), "on_top_edge"),
        Segment(6, np.array([10.0, 3.0]), np.array([10.0, 3.0]), "point_on_boundary"),
        Segment(7, np.array([3.0, 3.0]), np.array([3.0, 3.0]), "point_inside"),
        Segment(8, np.array([12.0, 9.0]), np.array([12.0, 9.0]), "point_outside"),
        Segment(9, np.array([-2.0, 4.0]), np.array([0.0, 4.0]), "touch_left_edge"),
    ]

    rng = np.random.default_rng(seed)
    random_segments: list[Segment] = []
    for i in range(random_count):
        a = rng.uniform(low=[-8.0, -6.0], high=[18.0, 14.0], size=2)
        b = rng.uniform(low=[-8.0, -6.0], high=[18.0, 14.0], size=2)
        random_segments.append(Segment(len(manual) + i, a, b, "random"))

    return manual + random_segments


def run_experiment(window: ClipWindow, segments: list[Segment]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    max_endpoint_delta = 0.0
    total_iterations = 0

    for seg in segments:
        cs = cohen_sutherland_clip(window, seg.p0, seg.p1)
        lb = liang_barsky_reference(window, seg.p0, seg.p1)
        total_iterations += cs.iterations

        if cs.accepted != lb.accepted:
            raise AssertionError(f"accept mismatch on segment {seg.idx}")

        clipped_len = 0.0
        endpoint_delta = 0.0
        if cs.accepted:
            assert cs.p0 is not None and cs.p1 is not None
            assert lb.p0 is not None and lb.p1 is not None
            assert window.contains(cs.p0) and window.contains(cs.p1)

            clipped_len = segment_length(cs.p0, cs.p1)
            endpoint_delta = max(
                float(np.linalg.norm(cs.p0 - lb.p0)),
                float(np.linalg.norm(cs.p1 - lb.p1)),
            )
            max_endpoint_delta = max(max_endpoint_delta, endpoint_delta)

        c0 = region_code(window, seg.p0)
        c1 = region_code(window, seg.p1)
        records.append(
            {
                "idx": seg.idx,
                "tag": seg.tag,
                "code0": c0,
                "code1": c1,
                "accepted": cs.accepted,
                "reason": cs.reason,
                "iterations": cs.iterations,
                "x0": float(seg.p0[0]),
                "y0": float(seg.p0[1]),
                "x1": float(seg.p1[0]),
                "y1": float(seg.p1[1]),
                "input_len": segment_length(seg.p0, seg.p1),
                "clipped_len": clipped_len,
                "endpoint_delta_vs_lb": endpoint_delta,
            }
        )

    detail_df = pd.DataFrame(records).sort_values("idx").reset_index(drop=True)
    accepted_df = detail_df[detail_df["accepted"]]
    rejected_df = detail_df[~detail_df["accepted"]]

    summary_df = pd.DataFrame(
        [
            {
                "total_segments": int(len(detail_df)),
                "accepted_segments": int(accepted_df.shape[0]),
                "rejected_segments": int(rejected_df.shape[0]),
                "accept_rate": float(detail_df["accepted"].mean()),
                "avg_iterations": float(total_iterations / len(detail_df)),
                "avg_input_len": float(detail_df["input_len"].mean()),
                "avg_clipped_len_accept_only": (
                    float(accepted_df["clipped_len"].mean())
                    if not accepted_df.empty
                    else 0.0
                ),
                "max_endpoint_delta_vs_lb": max_endpoint_delta,
                "trivial_reject_count": int((detail_df["reason"] == "trivial_reject").sum()),
            }
        ]
    )

    if max_endpoint_delta > 1e-8:
        raise AssertionError(f"endpoint delta too large vs Liang-Barsky: {max_endpoint_delta}")

    if int(summary_df.loc[0, "accepted_segments"]) == 0:
        raise AssertionError("No accepted segments; dataset is not informative.")

    if int(summary_df.loc[0, "rejected_segments"]) == 0:
        raise AssertionError("No rejected segments; dataset is not informative.")

    bad_reasons = {"max_iter_guard", "numeric_parallel_top", "numeric_parallel_bottom"}
    bad_reasons |= {"numeric_parallel_left", "numeric_parallel_right"}
    if any(reason in bad_reasons for reason in detail_df["reason"].unique()):
        raise AssertionError("Unexpected numeric guard rejection found in this dataset.")

    return detail_df, summary_df


def main() -> None:
    window = ClipWindow(xmin=0.0, xmax=10.0, ymin=0.0, ymax=8.0)
    segments = build_segments(seed=2026, random_count=44)
    detail_df, summary_df = run_experiment(window, segments)

    preview_cols = [
        "idx",
        "tag",
        "code0",
        "code1",
        "accepted",
        "reason",
        "iterations",
        "input_len",
        "clipped_len",
    ]

    print("Window:")
    print(
        f"xmin={window.xmin}, xmax={window.xmax}, "
        f"ymin={window.ymin}, ymax={window.ymax}"
    )
    print("\nPreview (first 12 segments):")
    print(detail_df.loc[:11, preview_cols].to_string(index=False, justify="center"))

    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
