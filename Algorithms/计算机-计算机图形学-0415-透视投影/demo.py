"""透视投影（Perspective Projection）最小可运行示例。

运行方式：
    uv run python demo.py
"""

from __future__ import annotations

import numpy as np


np.set_printoptions(precision=4, suppress=True)


def perspective_matrix(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """构造 OpenGL 风格右手坐标系透视投影矩阵。

    假设相机朝向 -Z，near/far 为正数，且 near < far。
    输出 4x4 矩阵，将相机空间点映射到裁剪空间。
    """
    if not (0.0 < near < far):
        raise ValueError("near/far must satisfy 0 < near < far")
    if not (0.0 < fov_y_deg < 180.0):
        raise ValueError("fov_y_deg must be in (0, 180)")

    f = 1.0 / np.tan(np.deg2rad(fov_y_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def to_homogeneous(points_xyz: np.ndarray) -> np.ndarray:
    """将 Nx3 点转换为 Nx4 齐次坐标。"""
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    return np.concatenate([points_xyz, ones], axis=1)


def perspective_project(points_camera_xyz: np.ndarray, proj_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """执行透视投影，返回 clip、ndc、w。"""
    pts_h = to_homogeneous(points_camera_xyz)  # (N,4)
    clip = (proj_mat @ pts_h.T).T  # (N,4)

    w = clip[:, 3:4]
    eps = 1e-12
    if np.any(np.abs(w) < eps):
        raise ZeroDivisionError("w is too close to zero during perspective divide")

    ndc = clip[:, :3] / w
    return clip, ndc, w[:, 0]


def ndc_to_screen(ndc_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """将 NDC(-1..1) 映射到屏幕像素坐标。"""
    x = (ndc_xy[:, 0] + 1.0) * 0.5 * width
    y = (1.0 - (ndc_xy[:, 1] + 1.0) * 0.5) * height
    return np.stack([x, y], axis=1)


def main() -> None:
    width, height = 800, 600
    aspect = width / height
    fov_y = 60.0
    near, far = 0.1, 100.0

    proj = perspective_matrix(fov_y_deg=fov_y, aspect=aspect, near=near, far=far)

    # 一个立方体的 8 个顶点，已经在相机空间中（z < 0 表示在相机前方）
    cube_points = np.array(
        [
            [-1.0, -1.0, -3.0],
            [1.0, -1.0, -3.0],
            [1.0, 1.0, -3.0],
            [-1.0, 1.0, -3.0],
            [-1.0, -1.0, -5.0],
            [1.0, -1.0, -5.0],
            [1.0, 1.0, -5.0],
            [-1.0, 1.0, -5.0],
        ],
        dtype=np.float64,
    )

    clip, ndc, w = perspective_project(cube_points, proj)
    screen = ndc_to_screen(ndc[:, :2], width=width, height=height)

    print("=== Perspective Projection MVP ===")
    print(f"viewport: {width}x{height}, fov_y={fov_y}, near={near}, far={far}")
    print("projection matrix:\n", proj)
    print()

    header = (
        "idx |"
        " camera(x,y,z)            |"
        " clip(x,y,z,w)                    |"
        " ndc(x,y,z)          |"
        " screen(x,y)"
    )
    print(header)
    print("-" * len(header))
    for i, (cam, c, n, s) in enumerate(zip(cube_points, clip, ndc, screen)):
        print(
            f"{i:>3} |"
            f" [{cam[0]:>6.2f},{cam[1]:>6.2f},{cam[2]:>6.2f}] |"
            f" [{c[0]:>7.3f},{c[1]:>7.3f},{c[2]:>7.3f},{c[3]:>7.3f}] |"
            f" [{n[0]:>6.3f},{n[1]:>6.3f},{n[2]:>6.3f}] |"
            f" [{s[0]:>7.2f},{s[1]:>7.2f}]"
        )

    # 深度导致“近大远小”的数值验证
    test_near = np.array([[1.0, 0.0, -2.0]], dtype=np.float64)
    test_far = np.array([[1.0, 0.0, -6.0]], dtype=np.float64)
    _, ndc_near, _ = perspective_project(test_near, proj)
    _, ndc_far, _ = perspective_project(test_far, proj)

    print("\n=== Sanity Check: near objects appear larger ===")
    print(f"same x=1.0, z=-2 => ndc_x = {ndc_near[0, 0]:.6f}")
    print(f"same x=1.0, z=-6 => ndc_x = {ndc_far[0, 0]:.6f}")
    print("expect |ndc_x(z=-2)| > |ndc_x(z=-6)|")

    if abs(ndc_near[0, 0]) <= abs(ndc_far[0, 0]):
        raise AssertionError("Perspective effect check failed")

    # 裁剪体合法性检查（立方体顶点都应在 NDC 立方体范围内）
    in_frustum = np.all(np.abs(ndc) <= 1.0 + 1e-9, axis=1)
    print(f"all cube vertices inside NDC cube: {bool(np.all(in_frustum))}")


if __name__ == "__main__":
    main()
