"""CS-0259 Z-Buffer算法：最小可运行 MVP。

实现内容：
1) 手写三角形光栅化与重心坐标深度插值。
2) 使用 Z-Buffer 进行逐像素隐藏面消除。
3) 验证 Z-Buffer 对绘制顺序不敏感，并对比无深度测试的顺序敏感性。

运行方式：
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


Array = np.ndarray


@dataclass(frozen=True)
class Triangle:
    """A colored triangle in NDC coordinates.

    vertices_ndc: shape (3, 3), each vertex is (x, y, z)
      - x, y in [-1, 1]
      - z in [0, 1], where smaller means closer to camera.
    """

    name: str
    vertices_ndc: Array
    color: tuple[int, int, int]


@dataclass
class RenderResult:
    color_buffer: Array
    depth_buffer: Array
    owner_buffer: Array
    tri_stats: pd.DataFrame
    global_metrics: pd.DataFrame


def _validate_triangle(tri: Triangle) -> Triangle:
    verts = np.asarray(tri.vertices_ndc, dtype=float)
    if verts.shape != (3, 3):
        raise ValueError(f"triangle {tri.name} vertices must be (3,3), got {verts.shape}")
    if not np.all(np.isfinite(verts)):
        raise ValueError(f"triangle {tri.name} vertices contain non-finite values")

    if np.any(verts[:, :2] < -1.0) or np.any(verts[:, :2] > 1.0):
        raise ValueError(f"triangle {tri.name} xy must be inside [-1,1]")
    if np.any(verts[:, 2] < 0.0) or np.any(verts[:, 2] > 1.0):
        raise ValueError(f"triangle {tri.name} z must be inside [0,1]")

    r, g, b = tri.color
    if any((c < 0 or c > 255) for c in (r, g, b)):
        raise ValueError(f"triangle {tri.name} color must be 0..255")
    return tri


def ndc_to_screen(vertices_ndc: Array, width: int, height: int) -> Array:
    """Map NDC vertices to screen coordinates.

    x_ndc in [-1,1] -> x_screen in [0,width-1]
    y_ndc in [-1,1] -> y_screen in [0,height-1] with top-left origin
    z is kept as-is for depth test.
    """
    v = np.asarray(vertices_ndc, dtype=float)
    x = (v[:, 0] * 0.5 + 0.5) * (width - 1)
    y = (1.0 - (v[:, 1] * 0.5 + 0.5)) * (height - 1)
    z = v[:, 2]
    return np.column_stack((x, y, z))


def edge_function(a_xy: Array, b_xy: Array, p_xy: Array) -> float:
    """Signed doubled area of triangle (a, b, p) in 2D."""
    return float((p_xy[0] - a_xy[0]) * (b_xy[1] - a_xy[1]) - (p_xy[1] - a_xy[1]) * (b_xy[0] - a_xy[0]))


def rasterize_triangle(
    tri_screen: Array,
    tri_color: tuple[int, int, int],
    tri_id: int,
    depth_buffer: Array,
    color_buffer: Array,
    owner_buffer: Array,
    use_zbuffer: bool,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Rasterize one triangle and optionally apply depth test."""
    h, w = depth_buffer.shape

    a, b, c = tri_screen
    area = edge_function(a[:2], b[:2], c[:2])
    if abs(area) < eps:
        return {
            "tri_id": float(tri_id),
            "candidate_pixels": 0.0,
            "inside_fragments": 0.0,
            "depth_pass": 0.0,
            "depth_fail": 0.0,
            "overwritten": 0.0,
        }

    min_x = max(int(np.floor(np.min(tri_screen[:, 0]))), 0)
    max_x = min(int(np.ceil(np.max(tri_screen[:, 0]))), w - 1)
    min_y = max(int(np.floor(np.min(tri_screen[:, 1]))), 0)
    max_y = min(int(np.ceil(np.max(tri_screen[:, 1]))), h - 1)

    candidate_pixels = 0
    inside_fragments = 0
    depth_pass = 0
    depth_fail = 0
    overwritten = 0

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            candidate_pixels += 1
            p = np.array([x + 0.5, y + 0.5], dtype=float)

            w0 = edge_function(b[:2], c[:2], p)
            w1 = edge_function(c[:2], a[:2], p)
            w2 = edge_function(a[:2], b[:2], p)

            if area > 0:
                inside = (w0 >= -eps) and (w1 >= -eps) and (w2 >= -eps)
            else:
                inside = (w0 <= eps) and (w1 <= eps) and (w2 <= eps)

            if not inside:
                continue

            inside_fragments += 1

            alpha = w0 / area
            beta = w1 / area
            gamma = w2 / area
            z = alpha * a[2] + beta * b[2] + gamma * c[2]

            if use_zbuffer:
                if z < depth_buffer[y, x]:
                    if owner_buffer[y, x] >= 0:
                        overwritten += 1
                    depth_buffer[y, x] = z
                    color_buffer[y, x] = tri_color
                    owner_buffer[y, x] = tri_id
                    depth_pass += 1
                else:
                    depth_fail += 1
            else:
                if owner_buffer[y, x] >= 0:
                    overwritten += 1
                depth_buffer[y, x] = z
                color_buffer[y, x] = tri_color
                owner_buffer[y, x] = tri_id
                depth_pass += 1

    return {
        "tri_id": float(tri_id),
        "candidate_pixels": float(candidate_pixels),
        "inside_fragments": float(inside_fragments),
        "depth_pass": float(depth_pass),
        "depth_fail": float(depth_fail),
        "overwritten": float(overwritten),
    }


def render_scene(
    triangles: list[Triangle],
    draw_order: Iterable[int],
    width: int,
    height: int,
    use_zbuffer: bool,
) -> RenderResult:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    for tri in triangles:
        _validate_triangle(tri)

    depth_buffer = np.full((height, width), np.inf, dtype=float)
    color_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    owner_buffer = np.full((height, width), -1, dtype=np.int32)

    stats_rows: list[dict[str, float]] = []
    for tri_idx in draw_order:
        tri = triangles[int(tri_idx)]
        tri_screen = ndc_to_screen(tri.vertices_ndc, width=width, height=height)
        one_stats = rasterize_triangle(
            tri_screen=tri_screen,
            tri_color=tri.color,
            tri_id=int(tri_idx),
            depth_buffer=depth_buffer,
            color_buffer=color_buffer,
            owner_buffer=owner_buffer,
            use_zbuffer=use_zbuffer,
        )
        stats_rows.append(one_stats)

    tri_stats = pd.DataFrame(stats_rows)
    tri_stats.insert(1, "tri_name", [triangles[int(i)].name for i in tri_stats["tri_id"].astype(int)])

    visible_mask = owner_buffer >= 0
    visible_pixels = int(np.count_nonzero(visible_mask))

    finite_depth = depth_buffer[np.isfinite(depth_buffer)]
    if finite_depth.size == 0:
        min_depth = np.nan
        max_depth = np.nan
    else:
        min_depth = float(np.min(finite_depth))
        max_depth = float(np.max(finite_depth))

    global_metrics = pd.DataFrame(
        {
            "metric": [
                "width",
                "height",
                "total_pixels",
                "visible_pixels",
                "coverage_ratio",
                "depth_min_visible",
                "depth_max_visible",
                "sum_inside_fragments",
                "sum_depth_pass",
                "sum_depth_fail",
                "sum_overwritten",
            ],
            "value": [
                float(width),
                float(height),
                float(width * height),
                float(visible_pixels),
                float(visible_pixels / (width * height)),
                min_depth,
                max_depth,
                float(tri_stats["inside_fragments"].sum()),
                float(tri_stats["depth_pass"].sum()),
                float(tri_stats["depth_fail"].sum()),
                float(tri_stats["overwritten"].sum()),
            ],
        }
    )

    return RenderResult(
        color_buffer=color_buffer,
        depth_buffer=depth_buffer,
        owner_buffer=owner_buffer,
        tri_stats=tri_stats,
        global_metrics=global_metrics,
    )


def summarize_owner_counts(owner_buffer: Array, triangles: list[Triangle]) -> pd.DataFrame:
    ids, counts = np.unique(owner_buffer, return_counts=True)
    rows = []
    for idx, c in zip(ids.tolist(), counts.tolist()):
        if idx < 0:
            name = "background"
        else:
            name = triangles[idx].name
        rows.append({"id": int(idx), "name": name, "pixels": int(c)})
    return pd.DataFrame(rows)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    width = 96
    height = 72

    # Three overlapping triangles.
    # z smaller => closer to camera.
    triangles = [
        Triangle(
            name="far_green",
            vertices_ndc=np.array(
                [[-0.85, -0.55, 0.75], [0.80, -0.75, 0.75], [0.05, 0.88, 0.75]],
                dtype=float,
            ),
            color=(20, 180, 60),
        ),
        Triangle(
            name="near_red",
            vertices_ndc=np.array(
                [[-0.68, -0.10, 0.20], [0.78, -0.18, 0.20], [0.12, 0.90, 0.20]],
                dtype=float,
            ),
            color=(220, 40, 40),
        ),
        Triangle(
            name="slanted_blue",
            vertices_ndc=np.array(
                [[-0.92, 0.62, 0.55], [0.92, 0.52, 0.40], [0.00, -0.92, 0.65]],
                dtype=float,
            ),
            color=(40, 90, 230),
        ),
    ]

    # Different draw orders.
    order_a = [0, 1, 2]
    order_b = [2, 0, 1]

    # Z-Buffer rendering: should be order-invariant.
    z_a = render_scene(triangles, order_a, width, height, use_zbuffer=True)
    z_b = render_scene(triangles, order_b, width, height, use_zbuffer=True)

    if not np.array_equal(z_a.owner_buffer, z_b.owner_buffer):
        raise AssertionError("Z-buffer owner buffer should be order-invariant")
    if not np.allclose(z_a.depth_buffer, z_b.depth_buffer, equal_nan=False):
        raise AssertionError("Z-buffer depth buffer should be order-invariant")

    # Painter-style rendering without depth test: usually order-dependent.
    p_a = render_scene(triangles, order_a, width, height, use_zbuffer=False)
    p_b = render_scene(triangles, order_b, width, height, use_zbuffer=False)

    painter_diff_pixels = int(np.count_nonzero(p_a.owner_buffer != p_b.owner_buffer))
    if painter_diff_pixels <= 0:
        raise AssertionError("Painter rendering should differ under different orders for this scene")

    # Z-buffer should differ from at least one painter result in this occluded scene.
    z_vs_painter_diff = int(np.count_nonzero(z_a.owner_buffer != p_a.owner_buffer))
    if z_vs_painter_diff <= 0:
        raise AssertionError("Z-buffer result should differ from painter result in an occluded scene")

    owner_counts = summarize_owner_counts(z_a.owner_buffer, triangles)

    headline_metrics = pd.DataFrame(
        {
            "metric": [
                "zbuffer_order_diff_pixels",
                "painter_order_diff_pixels",
                "zbuffer_vs_painter_diff_pixels",
            ],
            "value": [
                float(np.count_nonzero(z_a.owner_buffer != z_b.owner_buffer)),
                float(painter_diff_pixels),
                float(z_vs_painter_diff),
            ],
        }
    )

    print("=== Z-Buffer MVP (CS-0259) ===")
    print(f"resolution: {width} x {height}")
    print(f"draw_order_a: {order_a}")
    print(f"draw_order_b: {order_b}")
    print()

    print("=== Z-Buffer Triangle Stats (order_a) ===")
    print(z_a.tri_stats.to_string(index=False))
    print()

    print("=== Z-Buffer Global Metrics (order_a) ===")
    print(z_a.global_metrics.to_string(index=False))
    print()

    print("=== Visible Pixel Ownership (Z-Buffer order_a) ===")
    print(owner_counts.to_string(index=False))
    print()

    print("=== Order-Sensitivity Comparison ===")
    print(headline_metrics.to_string(index=False))
    print("All checks passed.")


if __name__ == "__main__":
    main()
