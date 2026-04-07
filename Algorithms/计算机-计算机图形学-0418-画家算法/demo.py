"""CS-0260 画家算法：最小可运行 MVP。

实现内容：
1) 手写三角形光栅化（边函数 + 重心深度插值）。
2) 实现画家算法（按平均深度远到近绘制）。
3) 以 Z-Buffer 作为参考，展示画家算法在不同场景下的正确性与局限。

运行方式：
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


Array = np.ndarray


@dataclass(frozen=True)
class Triangle:
    """Triangle in NDC coordinates.

    vertices_ndc has shape (3,3), each row is (x, y, z):
    - x, y in [-1, 1]
    - z in [0, 1], where smaller means closer to camera
    """

    name: str
    vertices_ndc: Array
    color: tuple[int, int, int]


@dataclass
class RenderResult:
    color_buffer: Array
    owner_buffer: Array
    depth_buffer: Array
    tri_stats: pd.DataFrame
    global_metrics: pd.DataFrame
    draw_order: list[int]


def _validate_triangle(tri: Triangle) -> None:
    verts = np.asarray(tri.vertices_ndc, dtype=float)
    if verts.shape != (3, 3):
        raise ValueError(f"{tri.name}: vertices shape must be (3,3), got {verts.shape}")
    if not np.all(np.isfinite(verts)):
        raise ValueError(f"{tri.name}: vertices contain non-finite values")

    if np.any(verts[:, :2] < -1.0) or np.any(verts[:, :2] > 1.0):
        raise ValueError(f"{tri.name}: x/y must be inside [-1,1]")
    if np.any(verts[:, 2] < 0.0) or np.any(verts[:, 2] > 1.0):
        raise ValueError(f"{tri.name}: z must be inside [0,1]")

    if any((c < 0 or c > 255) for c in tri.color):
        raise ValueError(f"{tri.name}: color channels must be in [0,255]")


def ndc_to_screen(vertices_ndc: Array, width: int, height: int) -> Array:
    """Map NDC to screen coordinates (top-left origin)."""
    v = np.asarray(vertices_ndc, dtype=float)
    x = (v[:, 0] * 0.5 + 0.5) * (width - 1)
    y = (1.0 - (v[:, 1] * 0.5 + 0.5)) * (height - 1)
    z = v[:, 2]
    return np.column_stack((x, y, z))


def edge_function(a_xy: Array, b_xy: Array, p_xy: Array) -> float:
    """Signed doubled area for triangle (a, b, p) in 2D."""
    return float((p_xy[0] - a_xy[0]) * (b_xy[1] - a_xy[1]) - (p_xy[1] - a_xy[1]) * (b_xy[0] - a_xy[0]))


def rasterize_triangle(tri_screen: Array, width: int, height: int, eps: float = 1e-12) -> dict[str, object]:
    """Rasterize one triangle and return pixel fragments.

    Returns:
      - candidate_pixels: int
      - inside_fragments: int
      - fragments: list[(x, y, z)]
    """
    a, b, c = tri_screen
    area = edge_function(a[:2], b[:2], c[:2])
    if abs(area) < eps:
        return {"candidate_pixels": 0, "inside_fragments": 0, "fragments": []}

    min_x = max(int(np.floor(np.min(tri_screen[:, 0]))), 0)
    max_x = min(int(np.ceil(np.max(tri_screen[:, 0]))), width - 1)
    min_y = max(int(np.floor(np.min(tri_screen[:, 1]))), 0)
    max_y = min(int(np.ceil(np.max(tri_screen[:, 1]))), height - 1)

    candidate_pixels = 0
    fragments: list[tuple[int, int, float]] = []

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

            alpha = w0 / area
            beta = w1 / area
            gamma = w2 / area
            z = float(alpha * a[2] + beta * b[2] + gamma * c[2])
            fragments.append((x, y, z))

    return {
        "candidate_pixels": candidate_pixels,
        "inside_fragments": len(fragments),
        "fragments": fragments,
    }


def _build_global_metrics(
    width: int,
    height: int,
    owner_buffer: Array,
    depth_buffer: Array,
    tri_stats: pd.DataFrame,
) -> pd.DataFrame:
    visible_pixels = int(np.count_nonzero(owner_buffer >= 0))
    finite_depth = depth_buffer[np.isfinite(depth_buffer)]
    if finite_depth.size == 0:
        depth_min = np.nan
        depth_max = np.nan
    else:
        depth_min = float(np.min(finite_depth))
        depth_max = float(np.max(finite_depth))

    return pd.DataFrame(
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
                depth_min,
                depth_max,
                float(tri_stats["inside_fragments"].sum()),
                float(tri_stats["depth_pass"].sum()),
                float(tri_stats["depth_fail"].sum()),
                float(tri_stats["overwritten"].sum()),
            ],
        }
    )


def render_painter(triangles: list[Triangle], width: int, height: int) -> RenderResult:
    """Painter's algorithm: sort by average depth (far to near) and overdraw."""
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    for tri in triangles:
        _validate_triangle(tri)

    depth_keys = np.array([float(np.mean(np.asarray(t.vertices_ndc)[:, 2])) for t in triangles], dtype=float)
    draw_order = sorted(range(len(triangles)), key=lambda i: (-depth_keys[i], i))

    color_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    owner_buffer = np.full((height, width), -1, dtype=np.int32)
    depth_buffer = np.full((height, width), np.inf, dtype=float)

    rows: list[dict[str, float | str]] = []
    for tri_id in draw_order:
        tri = triangles[tri_id]
        tri_screen = ndc_to_screen(tri.vertices_ndc, width=width, height=height)
        ras = rasterize_triangle(tri_screen, width=width, height=height)

        depth_pass = 0
        overwritten = 0
        for x, y, z in ras["fragments"]:
            if owner_buffer[y, x] >= 0:
                overwritten += 1
            color_buffer[y, x] = tri.color
            owner_buffer[y, x] = tri_id
            depth_buffer[y, x] = z
            depth_pass += 1

        rows.append(
            {
                "tri_id": float(tri_id),
                "tri_name": tri.name,
                "depth_key_mean_z": float(depth_keys[tri_id]),
                "candidate_pixels": float(ras["candidate_pixels"]),
                "inside_fragments": float(ras["inside_fragments"]),
                "depth_pass": float(depth_pass),
                "depth_fail": 0.0,
                "overwritten": float(overwritten),
            }
        )

    tri_stats = pd.DataFrame(rows)
    global_metrics = _build_global_metrics(width, height, owner_buffer, depth_buffer, tri_stats)
    return RenderResult(
        color_buffer=color_buffer,
        owner_buffer=owner_buffer,
        depth_buffer=depth_buffer,
        tri_stats=tri_stats,
        global_metrics=global_metrics,
        draw_order=draw_order,
    )


def render_zbuffer(triangles: list[Triangle], width: int, height: int, draw_order: list[int]) -> RenderResult:
    """Reference renderer with per-pixel Z test."""
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    for tri in triangles:
        _validate_triangle(tri)

    color_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    owner_buffer = np.full((height, width), -1, dtype=np.int32)
    depth_buffer = np.full((height, width), np.inf, dtype=float)

    rows: list[dict[str, float | str]] = []
    for tri_id in draw_order:
        tri = triangles[tri_id]
        tri_screen = ndc_to_screen(tri.vertices_ndc, width=width, height=height)
        ras = rasterize_triangle(tri_screen, width=width, height=height)

        depth_pass = 0
        depth_fail = 0
        overwritten = 0
        for x, y, z in ras["fragments"]:
            if z < depth_buffer[y, x]:
                if owner_buffer[y, x] >= 0:
                    overwritten += 1
                color_buffer[y, x] = tri.color
                owner_buffer[y, x] = tri_id
                depth_buffer[y, x] = z
                depth_pass += 1
            else:
                depth_fail += 1

        rows.append(
            {
                "tri_id": float(tri_id),
                "tri_name": tri.name,
                "depth_key_mean_z": float(np.mean(np.asarray(tri.vertices_ndc)[:, 2])),
                "candidate_pixels": float(ras["candidate_pixels"]),
                "inside_fragments": float(ras["inside_fragments"]),
                "depth_pass": float(depth_pass),
                "depth_fail": float(depth_fail),
                "overwritten": float(overwritten),
            }
        )

    tri_stats = pd.DataFrame(rows)
    global_metrics = _build_global_metrics(width, height, owner_buffer, depth_buffer, tri_stats)
    return RenderResult(
        color_buffer=color_buffer,
        owner_buffer=owner_buffer,
        depth_buffer=depth_buffer,
        tri_stats=tri_stats,
        global_metrics=global_metrics,
        draw_order=list(draw_order),
    )


def compare_buffers(owner_a: Array, owner_b: Array) -> pd.DataFrame:
    mismatches = int(np.count_nonzero(owner_a != owner_b))
    total = int(owner_a.size)
    return pd.DataFrame(
        {
            "metric": ["owner_mismatch_pixels", "owner_mismatch_ratio"],
            "value": [float(mismatches), float(mismatches / total)],
        }
    )


def build_scene_monotonic() -> list[Triangle]:
    """Scene where painter ordering is sufficient."""
    return [
        Triangle(
            name="far_blue",
            vertices_ndc=np.array([[-0.92, -0.72, 0.82], [0.92, -0.72, 0.82], [0.00, 0.90, 0.82]], dtype=float),
            color=(40, 100, 220),
        ),
        Triangle(
            name="mid_green",
            vertices_ndc=np.array([[-0.78, -0.25, 0.55], [0.84, -0.35, 0.55], [0.10, 0.92, 0.55]], dtype=float),
            color=(40, 190, 90),
        ),
        Triangle(
            name="near_red",
            vertices_ndc=np.array([[-0.70, -0.08, 0.24], [0.70, -0.10, 0.24], [0.00, 0.86, 0.24]], dtype=float),
            color=(220, 50, 50),
        ),
    ]


def build_scene_interpenetrating() -> list[Triangle]:
    """Scene where average-depth painter sorting fails."""
    return [
        Triangle(
            name="left_near_right_far",
            vertices_ndc=np.array([[-0.90, -0.82, 0.12], [0.90, -0.82, 0.88], [0.00, 0.88, 0.88]], dtype=float),
            color=(235, 80, 80),
        ),
        Triangle(
            name="left_far_right_near",
            vertices_ndc=np.array([[-0.90, -0.82, 0.88], [0.90, -0.82, 0.12], [0.00, 0.88, 0.12]], dtype=float),
            color=(60, 135, 245),
        ),
    ]


def run_scene(scene_name: str, triangles: list[Triangle], width: int, height: int) -> tuple[RenderResult, RenderResult, pd.DataFrame]:
    painter = render_painter(triangles, width=width, height=height)
    zref = render_zbuffer(triangles, width=width, height=height, draw_order=list(range(len(triangles))))
    cmp_df = compare_buffers(painter.owner_buffer, zref.owner_buffer)

    print(f"=== Scene: {scene_name} ===")
    print(f"resolution: {width} x {height}")
    print(f"painter_draw_order (far->near): {painter.draw_order}")
    print()

    print("[Painter] triangle stats")
    print(painter.tri_stats.to_string(index=False))
    print()
    print("[Painter] global metrics")
    print(painter.global_metrics.to_string(index=False))
    print()

    print("[Z-Buffer Reference] triangle stats")
    print(zref.tri_stats.to_string(index=False))
    print()
    print("[Z-Buffer Reference] global metrics")
    print(zref.global_metrics.to_string(index=False))
    print()

    print("[Painter vs Z-Buffer]")
    print(cmp_df.to_string(index=False))
    print()
    return painter, zref, cmp_df


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    width = 120
    height = 90

    scene_a = build_scene_monotonic()
    _, _, cmp_a = run_scene("monotonic_depth", scene_a, width, height)
    mismatch_a = int(cmp_a.loc[cmp_a["metric"] == "owner_mismatch_pixels", "value"].iloc[0])
    if mismatch_a != 0:
        raise AssertionError("Scene A should have zero mismatch between painter and Z-buffer.")

    scene_b = build_scene_interpenetrating()
    _, _, cmp_b = run_scene("interpenetrating_depth", scene_b, width, height)
    mismatch_b = int(cmp_b.loc[cmp_b["metric"] == "owner_mismatch_pixels", "value"].iloc[0])
    if mismatch_b <= 0:
        raise AssertionError("Scene B should expose painter failure (mismatch must be > 0).")

    print("=== Summary Checks ===")
    print(f"scene_a_mismatch_pixels = {mismatch_a}")
    print(f"scene_b_mismatch_pixels = {mismatch_b}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
