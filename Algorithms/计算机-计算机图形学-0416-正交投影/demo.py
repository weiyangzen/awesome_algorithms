"""CS-0258 正交投影：最小可运行 MVP。

实现内容：
1) 手写 4x4 正交投影矩阵并做齐次坐标批量投影。
2) 用直接线性归一化公式做数值对照，验证矩阵实现正确。
3) 验证正交投影的两个核心性质：保持平行、无透视缩短。

运行方式：
    uv run python demo.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


Array = np.ndarray


@dataclass(frozen=True)
class OrthoBox:
    """Axis-aligned orthographic view volume in camera space."""

    left: float
    right: float
    bottom: float
    top: float
    near: float
    far: float


def _validate_box(box: OrthoBox) -> None:
    if not (box.left < box.right):
        raise ValueError("left must be smaller than right")
    if not (box.bottom < box.top):
        raise ValueError("bottom must be smaller than top")
    if not (box.near < box.far):
        raise ValueError("near must be smaller than far")


def _as_points(points: Array) -> Array:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points contain non-finite values")
    return pts


def orthographic_matrix(box: OrthoBox) -> Array:
    """Build a 4x4 orthographic projection matrix mapping box -> NDC [-1,1]^3."""
    _validate_box(box)

    sx = 2.0 / (box.right - box.left)
    sy = 2.0 / (box.top - box.bottom)
    sz = 2.0 / (box.far - box.near)

    tx = -(box.right + box.left) / (box.right - box.left)
    ty = -(box.top + box.bottom) / (box.top - box.bottom)
    tz = -(box.far + box.near) / (box.far - box.near)

    return np.array(
        [
            [sx, 0.0, 0.0, tx],
            [0.0, sy, 0.0, ty],
            [0.0, 0.0, sz, tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def to_homogeneous(points: Array) -> Array:
    pts = _as_points(points)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    return np.hstack((pts, ones))


def project_points_matrix(points: Array, proj_matrix: Array) -> tuple[Array, Array]:
    """Project points via homogeneous matrix multiplication.

    Returns:
      ndc: shape (N, 3)
      clip: shape (N, 4)
    """
    pts = _as_points(points)
    mat = np.asarray(proj_matrix, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"projection matrix must be 4x4, got {mat.shape}")

    ph = to_homogeneous(pts)
    clip = (mat @ ph.T).T

    w = clip[:, 3:4]
    if np.any(np.isclose(w, 0.0)):
        raise ValueError("clip w contains zeros; cannot divide")

    ndc = clip[:, :3] / w
    return ndc, clip


def project_points_direct(points: Array, box: OrthoBox) -> Array:
    """Project points via explicit axis-wise normalization formulas."""
    _validate_box(box)
    pts = _as_points(points)

    x_ndc = (2.0 * pts[:, 0] - (box.right + box.left)) / (box.right - box.left)
    y_ndc = (2.0 * pts[:, 1] - (box.top + box.bottom)) / (box.top - box.bottom)
    z_ndc = (2.0 * pts[:, 2] - (box.far + box.near)) / (box.far - box.near)
    return np.column_stack((x_ndc, y_ndc, z_ndc))


def ndc_to_screen(ndc_points: Array, width: int, height: int) -> Array:
    """Map NDC x/y in [-1,1] to pixel coordinates with top-left origin."""
    ndc = np.asarray(ndc_points, dtype=float)
    if ndc.ndim != 2 or ndc.shape[1] < 2:
        raise ValueError(f"ndc_points must be (N, >=2), got {ndc.shape}")
    if width <= 1 or height <= 1:
        raise ValueError("width and height must be larger than 1")

    x = (ndc[:, 0] * 0.5 + 0.5) * (width - 1)
    y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (height - 1)
    return np.column_stack((x, y))


def inside_ndc_cube(ndc_points: Array, eps: float = 1e-12) -> Array:
    ndc = np.asarray(ndc_points, dtype=float)
    if ndc.ndim != 2 or ndc.shape[1] != 3:
        raise ValueError(f"ndc_points must be (N, 3), got {ndc.shape}")
    return np.all(np.abs(ndc) <= (1.0 + eps), axis=1)


def _slope_2d(a: Array, b: Array) -> float:
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    if abs(dx) < 1e-15:
        return np.inf
    return dy / dx


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    box = OrthoBox(left=-4.0, right=4.0, bottom=-3.0, top=3.0, near=1.0, far=11.0)
    proj = orthographic_matrix(box)

    # Structured sample points: some inside volume, some outside.
    sample_points = np.array(
        [
            [-4.0, -3.0, 1.0],
            [4.0, 3.0, 11.0],
            [0.0, 0.0, 6.0],
            [2.0, -1.5, 8.0],
            [5.0, 0.0, 6.0],
            [0.0, -3.5, 6.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 12.0],
        ],
        dtype=float,
    )

    ndc_matrix, clip = project_points_matrix(sample_points, proj)
    ndc_direct = project_points_direct(sample_points, box)

    max_diff = float(np.max(np.abs(ndc_matrix - ndc_direct)))
    if not np.allclose(ndc_matrix, ndc_direct, atol=1e-12):
        raise AssertionError("matrix projection and direct projection mismatch")

    w_dev = float(np.max(np.abs(clip[:, 3] - 1.0)))
    if w_dev > 1e-12:
        raise AssertionError("orthographic projection should keep w=1")

    inside_mask = inside_ndc_cube(ndc_matrix)
    screen_xy = ndc_to_screen(ndc_matrix, width=960, height=540)

    # Verify "parallel stays parallel" on two equal-direction segments at different z.
    seg1_world = np.array([[-3.5, -1.0, 2.0], [2.5, 2.0, 2.0]], dtype=float)
    seg2_world = np.array([[-3.5, 0.0, 9.0], [2.5, 3.0, 9.0]], dtype=float)

    seg1_ndc = project_points_matrix(seg1_world, proj)[0][:, :2]
    seg2_ndc = project_points_matrix(seg2_world, proj)[0][:, :2]

    d1 = seg1_ndc[1] - seg1_ndc[0]
    d2 = seg2_ndc[1] - seg2_ndc[0]
    parallel_cross = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(parallel_cross) > 1e-12:
        raise AssertionError("parallelism should be preserved in orthographic projection")

    # Verify "no perspective foreshortening": same-length segments remain same length.
    seg_near = np.array([[-2.0, 0.5, 2.0], [2.0, 0.5, 2.0]], dtype=float)
    seg_far = np.array([[-2.0, 0.5, 10.0], [2.0, 0.5, 10.0]], dtype=float)
    near_xy = project_points_matrix(seg_near, proj)[0][:, :2]
    far_xy = project_points_matrix(seg_far, proj)[0][:, :2]
    near_len = float(np.linalg.norm(near_xy[1] - near_xy[0]))
    far_len = float(np.linalg.norm(far_xy[1] - far_xy[0]))
    length_ratio_near_far = near_len / far_len if far_len > 0.0 else np.inf
    if not np.isclose(length_ratio_near_far, 1.0, atol=1e-12):
        raise AssertionError("orthographic projection should not introduce depth foreshortening")

    slope1 = _slope_2d(seg1_ndc[0], seg1_ndc[1])
    slope2 = _slope_2d(seg2_ndc[0], seg2_ndc[1])

    # Batch comparison for performance and numerical consistency.
    rng = np.random.default_rng(2026)
    cloud = np.column_stack(
        (
            rng.uniform(-6.0, 6.0, size=250_000),
            rng.uniform(-5.0, 5.0, size=250_000),
            rng.uniform(-1.0, 13.0, size=250_000),
        )
    )

    t0 = time.perf_counter()
    cloud_ndc_mat = project_points_matrix(cloud, proj)[0]
    matrix_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    cloud_ndc_direct = project_points_direct(cloud, box)
    direct_ms = (time.perf_counter() - t1) * 1000.0

    cloud_max_diff = float(np.max(np.abs(cloud_ndc_mat - cloud_ndc_direct)))
    if cloud_max_diff > 1e-12:
        raise AssertionError(f"cloud projection mismatch too large: {cloud_max_diff:.3e}")

    cloud_inside_ratio = float(np.mean(inside_ndc_cube(cloud_ndc_mat)))

    point_table = pd.DataFrame(
        {
            "x": sample_points[:, 0],
            "y": sample_points[:, 1],
            "z": sample_points[:, 2],
            "x_ndc": ndc_matrix[:, 0],
            "y_ndc": ndc_matrix[:, 1],
            "z_ndc": ndc_matrix[:, 2],
            "x_px": screen_xy[:, 0],
            "y_px": screen_xy[:, 1],
            "inside_ndc": inside_mask,
        }
    )

    metric_table = pd.DataFrame(
        {
            "metric": [
                "sample_count",
                "sample_inside_count",
                "max_diff_matrix_vs_direct_sample",
                "max_w_deviation",
                "parallel_cross",
                "slope_line1",
                "slope_line2",
                "near_far_length_ratio",
                "cloud_count",
                "cloud_inside_ratio",
                "cloud_max_diff_matrix_vs_direct",
                "matrix_projection_ms",
                "direct_projection_ms",
                "matrix_over_direct_time_ratio",
            ],
            "value": [
                float(sample_points.shape[0]),
                float(np.count_nonzero(inside_mask)),
                max_diff,
                w_dev,
                parallel_cross,
                slope1,
                slope2,
                length_ratio_near_far,
                float(cloud.shape[0]),
                cloud_inside_ratio,
                cloud_max_diff,
                matrix_ms,
                direct_ms,
                (matrix_ms / direct_ms) if direct_ms > 0 else np.inf,
            ],
        }
    )

    print("=== Orthographic Projection Matrix (4x4) ===")
    print(proj)
    print()

    print("=== Sample Point Projection Table ===")
    print(point_table.to_string(index=False))
    print()

    print("=== Metrics ===")
    print(metric_table.to_string(index=False))


if __name__ == "__main__":
    main()
