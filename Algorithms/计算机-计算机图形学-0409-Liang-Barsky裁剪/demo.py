"""Minimal runnable MVP for Liang-Barsky line clipping.

This script implements Liang-Barsky clipping against an axis-aligned rectangle,
prints a compact report, and performs deterministic sanity checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12


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
    u_enter: float | None
    u_leave: float | None
    reason: str


@dataclass(frozen=True)
class Segment:
    idx: int
    p0: np.ndarray
    p1: np.ndarray
    tag: str


def liang_barsky_clip(window: ClipWindow, p0: np.ndarray, p1: np.ndarray) -> ClipResult:
    """Clip one segment against `window` using Liang-Barsky parameter inequalities."""
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
                return ClipResult(False, None, None, None, None, "parallel_outside")
            continue

        r = qi / pi
        if pi < 0.0:
            u_enter = max(u_enter, r)
        else:
            u_leave = min(u_leave, r)

        if u_enter - u_leave > EPS:
            return ClipResult(False, None, None, None, None, "empty_interval")

    clipped_start = p0 + u_enter * (p1 - p0)
    clipped_end = p0 + u_leave * (p1 - p0)
    return ClipResult(True, clipped_start, clipped_end, u_enter, u_leave, "accepted")


def _region_code(window: ClipWindow, point: np.ndarray) -> int:
    x, y = float(point[0]), float(point[1])
    code = 0
    if x < window.xmin:
        code |= 1
    elif x > window.xmax:
        code |= 2
    if y < window.ymin:
        code |= 4
    elif y > window.ymax:
        code |= 8
    return code


def cohen_sutherland_clip(
    window: ClipWindow,
    p0: np.ndarray,
    p1: np.ndarray,
    max_iters: int = 16,
) -> ClipResult:
    """Reference clipper used only for correctness cross-checks."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])

    for _ in range(max_iters):
        code0 = _region_code(window, np.array([x0, y0], dtype=np.float64))
        code1 = _region_code(window, np.array([x1, y1], dtype=np.float64))

        if code0 == 0 and code1 == 0:
            return ClipResult(
                True,
                np.array([x0, y0], dtype=np.float64),
                np.array([x1, y1], dtype=np.float64),
                None,
                None,
                "accepted",
            )
        if code0 & code1:
            return ClipResult(False, None, None, None, None, "trivial_reject")

        out_code = code0 if code0 != 0 else code1
        dx = x1 - x0
        dy = y1 - y0

        if out_code & 8:
            x = x0 + dx * (window.ymax - y0) / dy
            y = window.ymax
        elif out_code & 4:
            x = x0 + dx * (window.ymin - y0) / dy
            y = window.ymin
        elif out_code & 2:
            y = y0 + dy * (window.xmax - x0) / dx
            x = window.xmax
        else:
            y = y0 + dy * (window.xmin - x0) / dx
            x = window.xmin

        if out_code == code0:
            x0, y0 = x, y
        else:
            x1, y1 = x, y

    return ClipResult(False, None, None, None, None, "max_iter_guard")


def _segment_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a))


def build_segments(seed: int = 2026, random_count: int = 40) -> list[Segment]:
    """Create deterministic mixed test segments: manual edge-cases + random cases."""
    manual = [
        Segment(0, np.array([1.0, 1.0]), np.array([9.0, 7.0]), "inside"),
        Segment(1, np.array([-3.0, 3.0]), np.array([12.0, 3.0]), "horizontal_cross"),
        Segment(2, np.array([5.0, -2.0]), np.array([5.0, 10.0]), "vertical_cross"),
        Segment(3, np.array([-2.0, -2.0]), np.array([-1.0, -1.0]), "outside_disjoint"),
        Segment(4, np.array([2.0, 8.0]), np.array([2.0, 10.0]), "parallel_outside_top"),
        Segment(5, np.array([0.0, 0.0]), np.array([10.0, 8.0]), "corner_to_corner"),
        Segment(6, np.array([10.0, 4.0]), np.array([10.0, 4.0]), "point_on_boundary"),
        Segment(7, np.array([-2.0, 4.0]), np.array([0.0, 4.0]), "touch_left_edge"),
    ]

    rng = np.random.default_rng(seed)
    random_segments = []
    for i in range(random_count):
        a = rng.uniform(low=[-6.0, -5.0], high=[16.0, 13.0], size=2)
        b = rng.uniform(low=[-6.0, -5.0], high=[16.0, 13.0], size=2)
        random_segments.append(Segment(len(manual) + i, a, b, "random"))

    return manual + random_segments


def run_experiment(window: ClipWindow, segments: list[Segment]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    max_endpoint_delta = 0.0

    for seg in segments:
        lb = liang_barsky_clip(window, seg.p0, seg.p1)
        cs = cohen_sutherland_clip(window, seg.p0, seg.p1)

        same_accept = lb.accepted == cs.accepted
        if not same_accept:
            raise AssertionError(f"accept mismatch at segment {seg.idx}")

        clipped_len = 0.0
        endpoint_delta = 0.0
        if lb.accepted:
            assert lb.p0 is not None and lb.p1 is not None
            assert window.contains(lb.p0) and window.contains(lb.p1)
            clipped_len = _segment_length(lb.p0, lb.p1)

            assert cs.p0 is not None and cs.p1 is not None
            endpoint_delta = max(
                float(np.linalg.norm(lb.p0 - cs.p0)),
                float(np.linalg.norm(lb.p1 - cs.p1)),
            )
            max_endpoint_delta = max(max_endpoint_delta, endpoint_delta)

        records.append(
            {
                "idx": seg.idx,
                "tag": seg.tag,
                "x0": float(seg.p0[0]),
                "y0": float(seg.p0[1]),
                "x1": float(seg.p1[0]),
                "y1": float(seg.p1[1]),
                "accepted": lb.accepted,
                "reason": lb.reason,
                "u_enter": np.nan if lb.u_enter is None else lb.u_enter,
                "u_leave": np.nan if lb.u_leave is None else lb.u_leave,
                "input_len": _segment_length(seg.p0, seg.p1),
                "clipped_len": clipped_len,
                "endpoint_delta_vs_cs": endpoint_delta,
            }
        )

    detail_df = pd.DataFrame(records).sort_values("idx").reset_index(drop=True)

    accepted_df = detail_df[detail_df["accepted"]]
    summary = pd.DataFrame(
        [
            {
                "total_segments": int(len(detail_df)),
                "accepted_segments": int(detail_df["accepted"].sum()),
                "rejected_segments": int((~detail_df["accepted"]).sum()),
                "accept_rate": float(detail_df["accepted"].mean()),
                "avg_input_len": float(detail_df["input_len"].mean()),
                "avg_clipped_len_accept_only": (
                    float(accepted_df["clipped_len"].mean())
                    if not accepted_df.empty
                    else 0.0
                ),
                "max_endpoint_delta_vs_cs": max_endpoint_delta,
            }
        ]
    )

    if max_endpoint_delta > 1e-8:
        raise AssertionError(
            f"Liang-Barsky and Cohen-Sutherland mismatch too large: {max_endpoint_delta}"
        )

    return detail_df, summary


def main() -> None:
    window = ClipWindow(xmin=0.0, xmax=10.0, ymin=0.0, ymax=8.0)
    segments = build_segments(seed=2026, random_count=40)

    detail_df, summary_df = run_experiment(window, segments)

    preview_cols = [
        "idx",
        "tag",
        "accepted",
        "reason",
        "u_enter",
        "u_leave",
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

    if int(summary_df.loc[0, "accepted_segments"]) == 0:
        raise AssertionError("No accepted segments, dataset is not informative.")

    if int(summary_df.loc[0, "rejected_segments"]) == 0:
        raise AssertionError("No rejected segments, dataset is not informative.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
