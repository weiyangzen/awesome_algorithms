"""三线性插值: minimal runnable MVP.

实现目标:
1. 手写三线性插值（向量化 + 朴素循环）；
2. 提供边界策略（raise / clip）；
3. 在解析线性场上做正确性验证；
4. 输出可读的样本表与性能指标。
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


Array = np.ndarray


def _validate_volume(volume: Array) -> Array:
    vol = np.asarray(volume, dtype=float)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3D with shape (D, H, W), got {vol.shape}")
    return vol


def _validate_points(points: Array) -> Array:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3) in (x, y, z) order, got {pts.shape}")
    return pts


def _handle_bounds(points: Array, shape_dhw: tuple[int, int, int], mode: str) -> Array:
    """Process out-of-bound points.

    Parameters
    ----------
    points:
        (N, 3) array in (x, y, z) order.
    shape_dhw:
        Volume shape (D, H, W), indexing as V[z, y, x].
    mode:
        - "raise": throw error if any point is outside valid range.
        - "clip": clip each coordinate into valid range.
    """
    d, h, w = shape_dhw
    lower = np.zeros(3, dtype=float)
    upper = np.array([w - 1, h - 1, d - 1], dtype=float)

    if mode == "clip":
        return np.clip(points, lower, upper)

    if mode == "raise":
        oob_mask = np.any((points < lower) | (points > upper), axis=1)
        if np.any(oob_mask):
            bad_idx = np.where(oob_mask)[0][:8]
            raise ValueError(
                f"query points out of bounds for mode='raise'; first bad indices: {bad_idx.tolist()}"
            )
        return points

    raise ValueError(f"unsupported mode: {mode!r}, expected 'raise' or 'clip'")


def trilinear_interpolate(volume: Array, points: Array, mode: str = "raise") -> Array:
    """Vectorized trilinear interpolation.

    Notes
    -----
    - volume axis order is (D, H, W) i.e. V[z, y, x]
    - points are in (x, y, z) order
    """
    vol = _validate_volume(volume)
    pts = _validate_points(points)
    pts = _handle_bounds(pts, vol.shape, mode=mode)

    d, h, w = vol.shape
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)

    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    z1 = np.minimum(z0 + 1, d - 1)

    xd = x - x0
    yd = y - y0
    zd = z - z0

    c000 = vol[z0, y0, x0]
    c100 = vol[z0, y0, x1]
    c010 = vol[z0, y1, x0]
    c110 = vol[z0, y1, x1]
    c001 = vol[z1, y0, x0]
    c101 = vol[z1, y0, x1]
    c011 = vol[z1, y1, x0]
    c111 = vol[z1, y1, x1]

    c00 = c000 * (1.0 - xd) + c100 * xd
    c10 = c010 * (1.0 - xd) + c110 * xd
    c01 = c001 * (1.0 - xd) + c101 * xd
    c11 = c011 * (1.0 - xd) + c111 * xd

    c0 = c00 * (1.0 - yd) + c10 * yd
    c1 = c01 * (1.0 - yd) + c11 * yd

    return c0 * (1.0 - zd) + c1 * zd


def trilinear_interpolate_naive(volume: Array, points: Array, mode: str = "raise") -> Array:
    """Naive Python-loop implementation for correctness/perf comparison."""
    vol = _validate_volume(volume)
    pts = _validate_points(points)
    pts = _handle_bounds(pts, vol.shape, mode=mode)

    d, h, w = vol.shape
    out = np.empty(pts.shape[0], dtype=float)

    for i in range(pts.shape[0]):
        x, y, z = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        z0 = int(np.floor(z))

        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        z1 = min(z0 + 1, d - 1)

        xd = x - x0
        yd = y - y0
        zd = z - z0

        c000 = vol[z0, y0, x0]
        c100 = vol[z0, y0, x1]
        c010 = vol[z0, y1, x0]
        c110 = vol[z0, y1, x1]
        c001 = vol[z1, y0, x0]
        c101 = vol[z1, y0, x1]
        c011 = vol[z1, y1, x0]
        c111 = vol[z1, y1, x1]

        c00 = c000 * (1.0 - xd) + c100 * xd
        c10 = c010 * (1.0 - xd) + c110 * xd
        c01 = c001 * (1.0 - xd) + c101 * xd
        c11 = c011 * (1.0 - xd) + c111 * xd

        c0 = c00 * (1.0 - yd) + c10 * yd
        c1 = c01 * (1.0 - yd) + c11 * yd

        out[i] = c0 * (1.0 - zd) + c1 * zd

    return out


def build_linear_volume(
    shape_dhw: tuple[int, int, int],
    coeff_xyz: tuple[float, float, float],
    bias: float,
) -> Array:
    """Build V[z, y, x] = ax*x + ay*y + az*z + bias."""
    d, h, w = shape_dhw
    z, y, x = np.indices((d, h, w), dtype=float)
    ax, ay, az = coeff_xyz
    return ax * x + ay * y + az * z + bias


def analytic_linear_field(
    points_xyz: Array,
    coeff_xyz: tuple[float, float, float],
    bias: float,
) -> Array:
    pts = _validate_points(points_xyz)
    ax, ay, az = coeff_xyz
    return ax * pts[:, 0] + ay * pts[:, 1] + az * pts[:, 2] + bias


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    shape_dhw = (20, 24, 28)
    coeff_xyz = (1.35, -0.72, 2.10)
    bias = 3.40
    volume = build_linear_volume(shape_dhw, coeff_xyz, bias)

    rng = np.random.default_rng(2026)
    n_points = 50_000
    queries = np.column_stack(
        [
            rng.uniform(0.0, shape_dhw[2] - 1.0, size=n_points),  # x in [0, W-1]
            rng.uniform(0.0, shape_dhw[1] - 1.0, size=n_points),  # y in [0, H-1]
            rng.uniform(0.0, shape_dhw[0] - 1.0, size=n_points),  # z in [0, D-1]
        ]
    )

    t0 = time.perf_counter()
    pred_vec = trilinear_interpolate(volume, queries, mode="raise")
    vec_ms = (time.perf_counter() - t0) * 1000.0

    truth = analytic_linear_field(queries, coeff_xyz, bias)
    abs_err = np.abs(pred_vec - truth)
    rmse = float(np.sqrt(np.mean((pred_vec - truth) ** 2)))
    max_abs_err = float(abs_err.max())

    # On a linear field, trilinear interpolation should be exact up to fp noise.
    if max_abs_err > 1e-10:
        raise AssertionError(f"max abs error too large on linear field: {max_abs_err}")

    n_subset = 5_000
    subset = queries[:n_subset]
    t1 = time.perf_counter()
    pred_naive = trilinear_interpolate_naive(volume, subset, mode="raise")
    naive_ms = (time.perf_counter() - t1) * 1000.0
    pred_subset_vec = pred_vec[:n_subset]
    if not np.allclose(pred_naive, pred_subset_vec, atol=1e-12):
        raise AssertionError("naive and vectorized implementations mismatch")

    oob_points = np.array(
        [
            [-1.2, 5.0, 3.0],
            [10.0, -2.4, 7.5],
            [30.0, 22.0, 19.9],
            [11.5, 40.0, -1.0],
        ],
        dtype=float,
    )

    clipped_queries = _handle_bounds(oob_points, shape_dhw, mode="clip")
    clipped_values = trilinear_interpolate(volume, oob_points, mode="clip")

    raise_msg = ""
    try:
        _ = trilinear_interpolate(volume, oob_points, mode="raise")
    except ValueError as exc:
        raise_msg = str(exc)
    else:
        raise AssertionError("mode='raise' should fail for out-of-bound points")

    sample_n = 8
    sample_table = pd.DataFrame(
        {
            "x": queries[:sample_n, 0],
            "y": queries[:sample_n, 1],
            "z": queries[:sample_n, 2],
            "pred": pred_vec[:sample_n],
            "truth": truth[:sample_n],
            "abs_err": abs_err[:sample_n],
        }
    )

    clip_table = pd.DataFrame(
        {
            "x_raw": oob_points[:, 0],
            "y_raw": oob_points[:, 1],
            "z_raw": oob_points[:, 2],
            "x_clip": clipped_queries[:, 0],
            "y_clip": clipped_queries[:, 1],
            "z_clip": clipped_queries[:, 2],
            "value_after_clip": clipped_values,
        }
    )

    metric_table = pd.DataFrame(
        {
            "metric": [
                "grid_D",
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
                float(shape_dhw[0]),
                float(shape_dhw[1]),
                float(shape_dhw[2]),
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
