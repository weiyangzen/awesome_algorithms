"""双线性插值: minimal runnable MVP.

实现目标:
1. 手写双线性插值（向量化 + 朴素循环）；
2. 提供边界策略（raise / clip）；
3. 在解析线性场上做正确性验证；
4. 输出可读的样本表与性能指标。
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

Array = np.ndarray


def _validate_grid(grid: Array) -> Array:
    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"grid must be 2D with shape (H, W), got {arr.shape}")
    return arr


def _validate_points(points: Array) -> Array:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2) in (x, y) order, got {pts.shape}")
    return pts


def _handle_bounds(points: Array, shape_hw: tuple[int, int], mode: str) -> Array:
    """Process out-of-bound points.

    Parameters
    ----------
    points:
        (N, 2) array in (x, y) order.
    shape_hw:
        Grid shape (H, W), indexing as G[y, x].
    mode:
        - "raise": throw error if any point is outside valid range.
        - "clip": clip each coordinate into valid range.
    """
    h, w = shape_hw
    lower = np.zeros(2, dtype=float)
    upper = np.array([w - 1, h - 1], dtype=float)

    if mode == "clip":
        return np.clip(points, lower, upper)

    if mode == "raise":
        oob_mask = np.any((points < lower) | (points > upper), axis=1)
        if np.any(oob_mask):
            bad_idx = np.where(oob_mask)[0][:8]
            raise ValueError(
                "query points out of bounds for mode='raise'; "
                f"first bad indices: {bad_idx.tolist()}"
            )
        return points

    raise ValueError(f"unsupported mode: {mode!r}, expected 'raise' or 'clip'")


def bilinear_interpolate(grid: Array, points: Array, mode: str = "raise") -> Array:
    """Vectorized bilinear interpolation.

    Notes
    -----
    - grid axis order is (H, W), indexing as G[y, x]
    - points are in (x, y) order
    """
    g = _validate_grid(grid)
    pts = _validate_points(points)
    pts = _handle_bounds(pts, g.shape, mode=mode)

    h, w = g.shape
    x = pts[:, 0]
    y = pts[:, 1]

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)

    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    xd = x - x0
    yd = y - y0

    f00 = g[y0, x0]
    f10 = g[y0, x1]
    f01 = g[y1, x0]
    f11 = g[y1, x1]

    fx0 = f00 * (1.0 - xd) + f10 * xd
    fx1 = f01 * (1.0 - xd) + f11 * xd

    return fx0 * (1.0 - yd) + fx1 * yd


def bilinear_interpolate_naive(grid: Array, points: Array, mode: str = "raise") -> Array:
    """Naive Python-loop implementation for correctness/perf comparison."""
    g = _validate_grid(grid)
    pts = _validate_points(points)
    pts = _handle_bounds(pts, g.shape, mode=mode)

    h, w = g.shape
    out = np.empty(pts.shape[0], dtype=float)

    for i in range(pts.shape[0]):
        x, y = float(pts[i, 0]), float(pts[i, 1])

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)

        xd = x - x0
        yd = y - y0

        f00 = g[y0, x0]
        f10 = g[y0, x1]
        f01 = g[y1, x0]
        f11 = g[y1, x1]

        fx0 = f00 * (1.0 - xd) + f10 * xd
        fx1 = f01 * (1.0 - xd) + f11 * xd
        out[i] = fx0 * (1.0 - yd) + fx1 * yd

    return out


def build_linear_grid(
    shape_hw: tuple[int, int],
    coeff_xy: tuple[float, float],
    bias: float,
) -> Array:
    """Build G[y, x] = ax*x + ay*y + bias."""
    h, w = shape_hw
    y, x = np.indices((h, w), dtype=float)
    ax, ay = coeff_xy
    return ax * x + ay * y + bias


def analytic_linear_field(
    points_xy: Array,
    coeff_xy: tuple[float, float],
    bias: float,
) -> Array:
    pts = _validate_points(points_xy)
    ax, ay = coeff_xy
    return ax * pts[:, 0] + ay * pts[:, 1] + bias


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    shape_hw = (480, 640)
    coeff_xy = (1.75, -0.48)
    bias = 2.30
    grid = build_linear_grid(shape_hw, coeff_xy, bias)

    rng = np.random.default_rng(2026)
    n_points = 80_000
    queries = np.column_stack(
        [
            rng.uniform(0.0, shape_hw[1] - 1.0, size=n_points),  # x in [0, W-1]
            rng.uniform(0.0, shape_hw[0] - 1.0, size=n_points),  # y in [0, H-1]
        ]
    )

    t0 = time.perf_counter()
    pred_vec = bilinear_interpolate(grid, queries, mode="raise")
    vec_ms = (time.perf_counter() - t0) * 1000.0

    truth = analytic_linear_field(queries, coeff_xy, bias)
    abs_err = np.abs(pred_vec - truth)
    rmse = float(np.sqrt(np.mean((pred_vec - truth) ** 2)))
    max_abs_err = float(abs_err.max())

    # On a linear field, bilinear interpolation should be exact up to fp noise.
    if max_abs_err > 1e-10:
        raise AssertionError(f"max abs error too large on linear field: {max_abs_err}")

    n_subset = 8_000
    subset = queries[:n_subset]
    t1 = time.perf_counter()
    pred_naive = bilinear_interpolate_naive(grid, subset, mode="raise")
    naive_ms = (time.perf_counter() - t1) * 1000.0
    pred_subset_vec = pred_vec[:n_subset]
    if not np.allclose(pred_naive, pred_subset_vec, atol=1e-12):
        raise AssertionError("naive and vectorized implementations mismatch")

    oob_points = np.array(
        [
            [-1.3, 10.0],
            [22.5, -5.2],
            [800.0, 200.0],
            [120.0, 999.0],
        ],
        dtype=float,
    )

    clipped_queries = _handle_bounds(oob_points, shape_hw, mode="clip")
    clipped_values = bilinear_interpolate(grid, oob_points, mode="clip")

    raise_msg = ""
    try:
        _ = bilinear_interpolate(grid, oob_points, mode="raise")
    except ValueError as exc:
        raise_msg = str(exc)
    else:
        raise AssertionError("mode='raise' should fail for out-of-bound points")

    sample_n = 8
    sample_table = pd.DataFrame(
        {
            "x": queries[:sample_n, 0],
            "y": queries[:sample_n, 1],
            "pred": pred_vec[:sample_n],
            "truth": truth[:sample_n],
            "abs_err": abs_err[:sample_n],
        }
    )

    clip_table = pd.DataFrame(
        {
            "x_raw": oob_points[:, 0],
            "y_raw": oob_points[:, 1],
            "x_clip": clipped_queries[:, 0],
            "y_clip": clipped_queries[:, 1],
            "value_after_clip": clipped_values,
        }
    )

    metric_table = pd.DataFrame(
        {
            "metric": [
                "grid_H",
                "grid_W",
                "num_queries",
                "vectorized_ms",
                "naive_ms_subset",
                "naive_over_vectorized_speedup",
                "max_abs_err_linear_field",
                "rmse_linear_field",
                "out_of_bound_points",
            ],
            "value": [
                float(shape_hw[0]),
                float(shape_hw[1]),
                float(n_points),
                vec_ms,
                naive_ms,
                (naive_ms / vec_ms) if vec_ms > 0 else np.inf,
                max_abs_err,
                rmse,
                float(oob_points.shape[0]),
            ],
        }
    )

    print("=== Sample Predictions (Linear Field; should be nearly exact) ===")
    print(sample_table.to_string(index=False))
    print()

    print("=== Boundary Handling (mode='clip') ===")
    print(clip_table.to_string(index=False))
    print()

    print("=== Metrics ===")
    print(metric_table.to_string(index=False))
    print()

    print("=== mode='raise' message ===")
    print(raise_msg)


if __name__ == "__main__":
    main()
