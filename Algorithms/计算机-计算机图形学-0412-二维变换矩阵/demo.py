"""二维变换矩阵: minimal runnable MVP.

本脚本演示二维齐次坐标下的仿射变换链：
- 平移
- 旋转
- 缩放
- 错切
并验证“逐步变换 == 组合矩阵一次变换”。
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd


Array = np.ndarray


def _as_points(points: Array) -> Array:
    """Validate and normalize point array to shape (N, 2)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {pts.shape}")
    return pts


def to_homogeneous(points: Array) -> Array:
    """Convert (N, 2) Euclidean points to (N, 3) homogeneous coordinates."""
    pts = _as_points(points)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    return np.hstack((pts, ones))


def from_homogeneous(points_h: Array) -> Array:
    """Convert (N, 3) homogeneous points back to Euclidean (N, 2)."""
    ph = np.asarray(points_h, dtype=float)
    if ph.ndim != 2 or ph.shape[1] != 3:
        raise ValueError(f"points_h must have shape (N, 3), got {ph.shape}")

    w = ph[:, 2:3]
    if np.any(np.isclose(w, 0.0)):
        raise ValueError("homogeneous coordinate w contains zeros; cannot project")
    return ph[:, :2] / w


def translation_matrix(tx: float, ty: float) -> Array:
    return np.array(
        [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def rotation_matrix(theta_deg: float) -> Array:
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def scaling_matrix(sx: float, sy: float) -> Array:
    return np.array(
        [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def shear_matrix(shx: float = 0.0, shy: float = 0.0) -> Array:
    return np.array(
        [[1.0, shx, 0.0], [shy, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def transform_about_pivot(base: Array, pivot_xy: tuple[float, float]) -> Array:
    """Wrap a transform matrix so that it acts around a pivot instead of origin."""
    px, py = pivot_xy
    to_origin = translation_matrix(-px, -py)
    back = translation_matrix(px, py)
    return back @ base @ to_origin


def apply_transform(points: Array, matrix: Array) -> Array:
    """Apply one 3x3 transform matrix to a batch of (N, 2) points."""
    pts = _as_points(points)
    mat = np.asarray(matrix, dtype=float)
    if mat.shape != (3, 3):
        raise ValueError(f"matrix must be 3x3, got {mat.shape}")

    ph = to_homogeneous(pts)
    transformed_h = (mat @ ph.T).T
    return from_homogeneous(transformed_h)


def compose_transforms(matrices: Iterable[Array]) -> Array:
    """Compose transforms in application order.

    If matrices are [M1, M2, M3] and points are column vectors,
    the result is M = M3 @ M2 @ M1.
    """
    composite = np.eye(3, dtype=float)
    for m in matrices:
        mat = np.asarray(m, dtype=float)
        if mat.shape != (3, 3):
            raise ValueError(f"all matrices must be 3x3, got {mat.shape}")
        composite = mat @ composite
    return composite


def polygon_area(points: Array) -> float:
    """Signed area via shoelace formula; absolute value means geometric area."""
    pts = _as_points(points)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    # A simple rectangle as the demo shape.
    shape = np.array(
        [[0.0, 0.0], [3.0, 0.0], [3.0, 1.5], [0.0, 1.5]],
        dtype=float,
    )
    centroid = tuple(shape.mean(axis=0))

    # Build transform chain.
    t_scale = transform_about_pivot(scaling_matrix(1.4, 0.7), centroid)
    t_rotate = transform_about_pivot(rotation_matrix(30.0), centroid)
    t_shear = shear_matrix(shx=0.25, shy=-0.10)
    t_translate = translation_matrix(2.5, -1.2)
    chain = [t_scale, t_rotate, t_shear, t_translate]

    # 1) Sequential application.
    sequential = shape.copy()
    for mat in chain:
        sequential = apply_transform(sequential, mat)

    # 2) One-shot composite matrix application.
    composite = compose_transforms(chain)
    one_shot = apply_transform(shape, composite)

    # They should be numerically identical up to floating-point tolerance.
    if not np.allclose(sequential, one_shot, atol=1e-10):
        raise AssertionError("sequential result and composite result mismatch")

    # Inverse transform should map transformed points back to original points.
    inv_composite = np.linalg.inv(composite)
    restored = apply_transform(one_shot, inv_composite)
    restore_err = np.linalg.norm(restored - shape, axis=1)
    if float(restore_err.max()) > 1e-9:
        raise AssertionError("inverse transform restoration error is too large")

    # Performance demo on a larger random point cloud.
    rng = np.random.default_rng(2026)
    cloud = rng.normal(loc=0.0, scale=3.0, size=(200_000, 2))

    t0 = time.perf_counter()
    cloud_seq = cloud.copy()
    for mat in chain:
        cloud_seq = apply_transform(cloud_seq, mat)
    seq_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    cloud_comp = apply_transform(cloud, composite)
    comp_ms = (time.perf_counter() - t1) * 1000.0

    if not np.allclose(cloud_seq, cloud_comp, atol=1e-10):
        raise AssertionError("cloud mismatch between sequential and composite")

    # Area scaling check: |det(A)| controls area change for affine transform.
    linear_part = composite[:2, :2]
    det_linear = float(np.linalg.det(linear_part))
    area_before = abs(polygon_area(shape))
    area_after = abs(polygon_area(one_shot))
    area_ratio = area_after / area_before

    point_table = pd.DataFrame(
        {
            "x0": shape[:, 0],
            "y0": shape[:, 1],
            "x1": one_shot[:, 0],
            "y1": one_shot[:, 1],
            "x_restore": restored[:, 0],
            "y_restore": restored[:, 1],
            "restore_err": restore_err,
        }
    )

    metric_table = pd.DataFrame(
        {
            "metric": [
                "num_points_cloud",
                "sequential_ms",
                "composite_ms",
                "speedup_seq_over_comp",
                "area_before",
                "area_after",
                "area_ratio",
                "abs_det_linear",
                "max_restore_err",
            ],
            "value": [
                float(cloud.shape[0]),
                seq_ms,
                comp_ms,
                (seq_ms / comp_ms) if comp_ms > 0 else np.inf,
                area_before,
                area_after,
                area_ratio,
                abs(det_linear),
                float(restore_err.max()),
            ],
        }
    )

    print("=== Composite Matrix (3x3) ===")
    print(composite)
    print()

    print("=== Point Mapping Table (shape vertices) ===")
    print(point_table.to_string(index=False))
    print()

    print("=== Metrics ===")
    print(metric_table.to_string(index=False))


if __name__ == "__main__":
    main()
