"""纹理映射: minimal runnable MVP.

实现内容:
1. 程序化生成纹理（棋盘格）；
2. 三角形栅格化（重心坐标）；
3. UV 仿射插值与透视校正插值；
4. 双线性纹理采样；
5. 输出误差指标与样本像素，形成可验证闭环。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Array = np.ndarray


def generate_checkerboard(size: int = 256, checks: int = 16) -> Array:
    """Generate RGB checkerboard texture in [0, 1]."""
    if size <= 1 or checks <= 0:
        raise ValueError("size must be > 1 and checks must be positive")

    y = np.arange(size, dtype=np.float64)[:, None]
    x = np.arange(size, dtype=np.float64)[None, :]
    cell = size / checks
    pattern = ((np.floor(x / cell) + np.floor(y / cell)) % 2.0).astype(np.float64)

    red = pattern
    green = 1.0 - pattern
    blue = 0.25 + 0.75 * (1.0 - pattern)
    return np.stack([red, green, blue], axis=-1)


def bilinear_sample_repeat(texture: Array, u: Array, v: Array) -> Array:
    """Bilinear sampling with repeat-wrap for vectorized u,v arrays."""
    if texture.ndim != 3 or texture.shape[2] != 3:
        raise ValueError("texture must have shape (H, W, 3)")

    h, w, _ = texture.shape
    u = np.asarray(u, dtype=np.float64) % 1.0
    v = np.asarray(v, dtype=np.float64) % 1.0

    x = u * (w - 1)
    y = v * (h - 1)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h

    tx = x - x0
    ty = y - y0

    c00 = texture[y0, x0]
    c10 = texture[y0, x1]
    c01 = texture[y1, x0]
    c11 = texture[y1, x1]

    c0 = c00 * (1.0 - tx)[:, None] + c10 * tx[:, None]
    c1 = c01 * (1.0 - tx)[:, None] + c11 * tx[:, None]
    return c0 * (1.0 - ty)[:, None] + c1 * ty[:, None]


def barycentric_for_points(points_xy: Array, tri_xy: Array) -> Array:
    """Compute barycentric weights for 2D points against one triangle."""
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    if tri_xy.shape != (3, 2):
        raise ValueError("tri_xy must have shape (3, 2)")

    mat = np.array(
        [
            [tri_xy[0, 0], tri_xy[1, 0], tri_xy[2, 0]],
            [tri_xy[0, 1], tri_xy[1, 1], tri_xy[2, 1]],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    det = np.linalg.det(mat)
    if abs(det) < 1e-12:
        raise ValueError("degenerate triangle: area too small")

    inv_mat = np.linalg.inv(mat)
    points_h = np.column_stack([points_xy, np.ones(points_xy.shape[0], dtype=np.float64)])
    # Row-wise form: w^T = p^T * inv(M)^T
    return points_h @ inv_mat.T


def triangle_bbox(tri_xy: Array, width: int, height: int) -> tuple[int, int, int, int]:
    """Compute integer pixel bbox [x_min, x_max], [y_min, y_max] clamped to image."""
    x_min = max(int(np.floor(np.min(tri_xy[:, 0]))), 0)
    x_max = min(int(np.ceil(np.max(tri_xy[:, 0]))), width - 1)
    y_min = max(int(np.floor(np.min(tri_xy[:, 1]))), 0)
    y_max = min(int(np.ceil(np.max(tri_xy[:, 1]))), height - 1)
    if x_min > x_max or y_min > y_max:
        raise ValueError("triangle bbox out of image")
    return x_min, x_max, y_min, y_max


def rasterize_triangle(
    width: int,
    height: int,
    tri_xy: Array,
    tri_uv: Array,
    tri_z: Array,
    texture: Array,
) -> dict[str, Array | float | int]:
    """Rasterize one textured triangle with affine and perspective-correct UV mapping."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if tri_xy.shape != (3, 2) or tri_uv.shape != (3, 2) or tri_z.shape != (3,):
        raise ValueError("tri_xy, tri_uv, tri_z shapes must be (3,2), (3,2), (3,)")
    if np.any(tri_z <= 0.0):
        raise ValueError("tri_z must be strictly positive for perspective correction")

    x_min, x_max, y_min, y_max = triangle_bbox(tri_xy, width, height)

    yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    points = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])

    bary = barycentric_for_points(points, tri_xy)
    inside = np.all(bary >= -1e-9, axis=1)

    inside_points = points[inside]
    inside_bary = bary[inside]
    if inside_points.shape[0] == 0:
        raise ValueError("triangle does not cover any pixel center")

    uv_affine = inside_bary @ tri_uv

    inv_z = 1.0 / tri_z
    denom = inside_bary @ inv_z
    uv_over_z = tri_uv * inv_z[:, None]
    uv_perspective = (inside_bary @ uv_over_z) / denom[:, None]

    col_affine = bilinear_sample_repeat(texture, uv_affine[:, 0], uv_affine[:, 1])
    col_perspective = bilinear_sample_repeat(texture, uv_perspective[:, 0], uv_perspective[:, 1])

    background = np.array([0.08, 0.08, 0.08], dtype=np.float64)
    img_affine = np.broadcast_to(background, (height, width, 3)).copy()
    img_perspective = np.broadcast_to(background, (height, width, 3)).copy()

    px = inside_points[:, 0].astype(np.int64)
    py = inside_points[:, 1].astype(np.int64)
    img_affine[py, px] = col_affine
    img_perspective[py, px] = col_perspective

    uv_abs_diff = np.abs(uv_affine - uv_perspective)
    color_abs_diff = np.abs(col_affine - col_perspective)

    mse = float(np.mean((col_affine - col_perspective) ** 2))
    mae = float(np.mean(color_abs_diff))

    return {
        "img_affine": img_affine,
        "img_perspective": img_perspective,
        "inside_points": inside_points,
        "inside_bary": inside_bary,
        "uv_affine": uv_affine,
        "uv_perspective": uv_perspective,
        "color_affine": col_affine,
        "color_perspective": col_perspective,
        "uv_l1_mean": float(np.mean(uv_abs_diff)),
        "uv_linf_max": float(np.max(uv_abs_diff)),
        "mse": mse,
        "mae": mae,
        "coverage": int(inside_points.shape[0]),
        "bbox_w": int(x_max - x_min + 1),
        "bbox_h": int(y_max - y_min + 1),
    }


def make_sample_table(
    inside_points: Array,
    uv_affine: Array,
    uv_perspective: Array,
    color_affine: Array,
    color_perspective: Array,
    sample_count: int = 8,
) -> pd.DataFrame:
    n = inside_points.shape[0]
    idx = np.linspace(0, n - 1, num=min(sample_count, n), dtype=np.int64)
    return pd.DataFrame(
        {
            "x": inside_points[idx, 0],
            "y": inside_points[idx, 1],
            "u_aff": uv_affine[idx, 0],
            "v_aff": uv_affine[idx, 1],
            "u_persp": uv_perspective[idx, 0],
            "v_persp": uv_perspective[idx, 1],
            "dR": color_perspective[idx, 0] - color_affine[idx, 0],
            "dG": color_perspective[idx, 1] - color_affine[idx, 1],
            "dB": color_perspective[idx, 2] - color_affine[idx, 2],
        }
    )


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    width, height = 320, 240
    texture = generate_checkerboard(size=256, checks=20)

    tri_xy = np.array(
        [
            [36.0, 24.0],
            [292.0, 66.0],
            [154.0, 218.0],
        ],
        dtype=np.float64,
    )
    tri_uv = np.array(
        [
            [0.0, 0.0],
            [3.4, 0.2],
            [0.6, 3.0],
        ],
        dtype=np.float64,
    )
    tri_z = np.array([0.55, 1.7, 3.8], dtype=np.float64)

    result = rasterize_triangle(
        width=width,
        height=height,
        tri_xy=tri_xy,
        tri_uv=tri_uv,
        tri_z=tri_z,
        texture=texture,
    )

    inside_bary = result["inside_bary"]
    bary_sum_err = float(np.max(np.abs(np.sum(inside_bary, axis=1) - 1.0)))

    sample_table = make_sample_table(
        inside_points=result["inside_points"],
        uv_affine=result["uv_affine"],
        uv_perspective=result["uv_perspective"],
        color_affine=result["color_affine"],
        color_perspective=result["color_perspective"],
        sample_count=8,
    )

    metric_table = pd.DataFrame(
        {
            "metric": [
                "image_width",
                "image_height",
                "texture_size",
                "covered_pixels",
                "bbox_width",
                "bbox_height",
                "barycentric_sum_max_error",
                "uv_l1_mean",
                "uv_linf_max",
                "color_mse_affine_vs_perspective",
                "color_mae_affine_vs_perspective",
            ],
            "value": [
                float(width),
                float(height),
                float(texture.shape[0]),
                float(result["coverage"]),
                float(result["bbox_w"]),
                float(result["bbox_h"]),
                bary_sum_err,
                float(result["uv_l1_mean"]),
                float(result["uv_linf_max"]),
                float(result["mse"]),
                float(result["mae"]),
            ],
        }
    )

    print("=== Texture Mapping MVP ===")
    print("Triangle vertices (screen xy):")
    print(tri_xy)
    print("Triangle UV:")
    print(tri_uv)
    print(f"Triangle depth z: {tri_z}")
    print()
    print("-- Metrics --")
    print(metric_table.to_string(index=False))
    print()
    print("-- Sample Pixels (Affine vs Perspective-Correct UV) --")
    print(sample_table.to_string(index=False))

    if bary_sum_err > 1e-10:
        raise AssertionError(f"barycentric sum error too large: {bary_sum_err}")
    if float(result["coverage"]) <= 0:
        raise AssertionError("no covered pixels")
    if float(result["mse"]) <= 1e-7:
        raise AssertionError(
            "expected visible difference between affine and perspective-correct mapping"
        )


if __name__ == "__main__":
    main()
