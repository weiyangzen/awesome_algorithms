"""CS-0256 三维变换矩阵：最小可运行 MVP。

实现内容：
1) 构建 4x4 齐次坐标下的缩放、旋转、平移矩阵。
2) 按指定顺序组合总变换矩阵，并批量作用到三维点云。
3) 做数值自检：分步变换 == 组合变换，逆变换可恢复原始点。

运行方式：
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class TransformComponents:
    scale: np.ndarray
    rot_x: np.ndarray
    rot_y: np.ndarray
    rot_z: np.ndarray
    translation: np.ndarray
    composed: np.ndarray


def _validate_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("points must contain at least one point")
    if not np.all(np.isfinite(arr)):
        raise ValueError("points contains non-finite values")
    return arr


def _validate_finite_tuple(values: Tuple[float, ...], name: str, expected_len: int) -> tuple[float, ...]:
    if len(values) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(values)}")
    casted = tuple(float(v) for v in values)
    if not np.all(np.isfinite(casted)):
        raise ValueError(f"{name} contains non-finite values: {casted}")
    return casted


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    """(N,3) -> (N,4)，最后一列补 1。"""
    pts = _validate_points(points)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return np.hstack([pts, ones])


def from_homogeneous(points_h: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """(N,4) -> (N,3)，支持一般齐次坐标（会除以 w）。"""
    arr = np.asarray(points_h, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"homogeneous points must have shape (N, 4), got {arr.shape}")

    w = arr[:, 3]
    if np.any(np.abs(w) <= tol):
        raise ValueError("homogeneous coordinate w is too close to zero")
    xyz = arr[:, :3] / w[:, None]
    return xyz


def scale_matrix_3d(sx: float, sy: float, sz: float) -> np.ndarray:
    sx, sy, sz = _validate_finite_tuple((sx, sy, sz), "scale", 3)
    return np.array(
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def translation_matrix_3d(tx: float, ty: float, tz: float) -> np.ndarray:
    tx, ty, tz = _validate_finite_tuple((tx, ty, tz), "translation", 3)
    return np.array(
        [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_x_matrix(theta_rad: float) -> np.ndarray:
    (theta_rad,) = _validate_finite_tuple((theta_rad,), "theta_rad", 1)
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_y_matrix(theta_rad: float) -> np.ndarray:
    (theta_rad,) = _validate_finite_tuple((theta_rad,), "theta_rad", 1)
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_z_matrix(theta_rad: float) -> np.ndarray:
    (theta_rad,) = _validate_finite_tuple((theta_rad,), "theta_rad", 1)
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def compose_transform(
    scale_xyz: tuple[float, float, float],
    rotation_deg_xyz: tuple[float, float, float],
    translation_xyz: tuple[float, float, float],
) -> TransformComponents:
    sx, sy, sz = _validate_finite_tuple(scale_xyz, "scale_xyz", 3)
    rx_deg, ry_deg, rz_deg = _validate_finite_tuple(rotation_deg_xyz, "rotation_deg_xyz", 3)
    tx, ty, tz = _validate_finite_tuple(translation_xyz, "translation_xyz", 3)

    s = scale_matrix_3d(sx, sy, sz)
    rx = rotation_x_matrix(np.deg2rad(rx_deg))
    ry = rotation_y_matrix(np.deg2rad(ry_deg))
    rz = rotation_z_matrix(np.deg2rad(rz_deg))
    t = translation_matrix_3d(tx, ty, tz)

    # 列向量约定: p' = T * Rz * Ry * Rx * S * p
    composed = t @ rz @ ry @ rx @ s

    return TransformComponents(scale=s, rot_x=rx, rot_y=ry, rot_z=rz, translation=t, composed=composed)


def apply_transform(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    pts = _validate_points(points)
    mat = np.asarray(transform_4x4, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"transform matrix must have shape (4, 4), got {mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ValueError("transform matrix contains non-finite values")

    pts_h = to_homogeneous(pts)
    # 批量点采用行向量存储，因此右乘 M^T，等价于列向量左乘 M。
    transformed_h = pts_h @ mat.T
    return from_homogeneous(transformed_h)


def inverse_transform(transform_4x4: np.ndarray) -> np.ndarray:
    mat = np.asarray(transform_4x4, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"transform matrix must have shape (4, 4), got {mat.shape}")
    det = np.linalg.det(mat)
    if np.isclose(det, 0.0, atol=1e-12):
        raise ValueError("transform matrix is singular and cannot be inverted")
    return np.linalg.inv(mat)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    # 8 个立方体顶点
    cube = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    comps = compose_transform(
        scale_xyz=(1.2, 0.8, 1.5),
        rotation_deg_xyz=(30.0, -20.0, 40.0),
        translation_xyz=(2.0, -1.0, 3.5),
    )

    transformed = apply_transform(cube, comps.composed)

    # 校验 1: 分步应用和组合矩阵应用应一致
    stepwise = apply_transform(cube, comps.scale)
    stepwise = apply_transform(stepwise, comps.rot_x)
    stepwise = apply_transform(stepwise, comps.rot_y)
    stepwise = apply_transform(stepwise, comps.rot_z)
    stepwise = apply_transform(stepwise, comps.translation)

    composition_error = float(np.linalg.norm(transformed - stepwise, ord="fro"))
    if composition_error > 1e-10:
        raise AssertionError(f"composition mismatch too large: {composition_error}")

    # 校验 2: 逆变换应恢复到原始点
    inv_m = inverse_transform(comps.composed)
    recovered = apply_transform(transformed, inv_m)
    recovery_error = float(np.linalg.norm(recovered - cube, ord="fro"))
    if recovery_error > 1e-10:
        raise AssertionError(f"inverse recovery mismatch too large: {recovery_error}")

    # 校验 3: 变换顺序不可交换（给出一个可复现的数值差异）
    reordered = comps.translation @ comps.scale @ comps.rot_z @ comps.rot_y @ comps.rot_x
    non_comm_error = float(np.linalg.norm(comps.composed - reordered, ord="fro"))
    if non_comm_error <= 1e-8:
        raise AssertionError("expected non-commutativity, but difference is too small")

    print("=== 3D Transform Matrix MVP ===")
    print("Input points shape:", cube.shape)
    print("Composed matrix M (T*Rz*Ry*Rx*S):")
    print(comps.composed)
    print("\nFirst 3 transformed points:")
    print(transformed[:3])
    print(f"\ncomposition_error (Frobenius): {composition_error:.3e}")
    print(f"inverse_recovery_error (Frobenius): {recovery_error:.3e}")
    print(f"non_commutativity_gap (Frobenius): {non_comm_error:.3e}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
