"""Flat shading MVP: minimal triangle rasterizer with z-buffer and Lambert face lighting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

EPS = 1e-12


@dataclass
class RenderStats:
    triangles_total: int
    triangles_rasterized: int
    pixels_shaded: int
    mean_intensity: float
    min_depth: float
    max_depth: float


def normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if (not np.isfinite(norm)) or norm < EPS:
        raise ValueError(f"Cannot normalize vector with norm={norm}.")
    return v / norm


def make_rotation_x(theta_rad: float) -> np.ndarray:
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def make_rotation_y(theta_rad: float) -> np.ndarray:
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def make_look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up))
    up_true = np.cross(right, forward)

    view = np.eye(4, dtype=float)
    view[0, :3] = right
    view[1, :3] = up_true
    view[2, :3] = -forward

    view[0, 3] = -float(np.dot(right, eye))
    view[1, 3] = -float(np.dot(up_true, eye))
    view[2, 3] = float(np.dot(forward, eye))
    return view


def make_perspective(fov_y_deg: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
    if fov_y_deg <= 0.0 or fov_y_deg >= 180.0:
        raise ValueError("fov_y_deg must be in (0, 180).")
    if aspect <= 0.0:
        raise ValueError("aspect must be > 0.")
    if z_near <= 0.0 or z_far <= z_near:
        raise ValueError("Require 0 < z_near < z_far.")

    f = 1.0 / np.tan(np.deg2rad(fov_y_deg) * 0.5)
    p = np.zeros((4, 4), dtype=float)
    p[0, 0] = f / aspect
    p[1, 1] = f
    p[2, 2] = (z_far + z_near) / (z_near - z_far)
    p[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    p[3, 2] = -1.0
    return p


def generate_cube_mesh() -> Tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    triangles = np.array(
        [
            [4, 5, 6],
            [4, 6, 7],
            [1, 0, 3],
            [1, 3, 2],
            [0, 4, 7],
            [0, 7, 3],
            [5, 1, 2],
            [5, 2, 6],
            [3, 7, 6],
            [3, 6, 2],
            [0, 1, 5],
            [0, 5, 4],
        ],
        dtype=np.int64,
    )

    return vertices, triangles


def apply_transform(points_xyz: np.ndarray, matrix_4x4: np.ndarray) -> np.ndarray:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N, 3), got {points_xyz.shape}.")

    ones = np.ones((points_xyz.shape[0], 1), dtype=float)
    points_h = np.concatenate([points_xyz, ones], axis=1)
    transformed_h = (matrix_4x4 @ points_h.T).T
    w = transformed_h[:, 3:4]

    if np.any(np.abs(w) < EPS):
        raise RuntimeError("Encountered homogeneous w close to zero during transform.")

    return transformed_h[:, :3] / w


def project_to_ndc(points_xyz: np.ndarray, mvp: np.ndarray) -> np.ndarray:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N, 3), got {points_xyz.shape}.")

    ones = np.ones((points_xyz.shape[0], 1), dtype=float)
    points_h = np.concatenate([points_xyz, ones], axis=1)
    clip = (mvp @ points_h.T).T
    w = clip[:, 3:4]

    if np.any(np.abs(w) < EPS):
        raise RuntimeError("Encountered clip-space w close to zero.")

    ndc = clip[:, :3] / w
    if not np.all(np.isfinite(ndc)):
        raise RuntimeError("NDC coordinates contain non-finite values.")
    return ndc


def ndc_to_screen(ndc: np.ndarray, width: int, height: int) -> np.ndarray:
    x = (ndc[:, 0] * 0.5 + 0.5) * float(width - 1)
    y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * float(height - 1)
    z = ndc[:, 2] * 0.5 + 0.5
    return np.stack([x, y, z], axis=1)


def face_normal(points_xyz: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a, b, c = points_xyz[tri[0]], points_xyz[tri[1]], points_xyz[tri[2]]
    return normalize(np.cross(b - a, c - a))


def shade_face(
    normal_view: np.ndarray,
    normal_world: np.ndarray,
    light_dir_view: np.ndarray,
    ambient: float = 0.18,
    diffuse: float = 0.82,
) -> Tuple[np.ndarray, float]:
    lambert = max(0.0, float(np.dot(normal_view, light_dir_view)))
    intensity = float(np.clip(ambient + diffuse * lambert, 0.0, 1.0))

    # Use world-space normal magnitude to assign a stable per-face base color.
    base_rgb = np.clip(0.25 + 0.75 * np.abs(normal_world), 0.0, 1.0)
    rgb = np.clip(np.round(255.0 * base_rgb * intensity), 0.0, 255.0).astype(np.uint8)
    return rgb, intensity


def edge_value(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax)


def rasterize_triangle(
    image: np.ndarray,
    depth_buffer: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    color: np.ndarray,
) -> int:
    area = edge_value(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1])
    if abs(area) < 1e-8:
        return 0

    h, w, _ = image.shape
    min_x = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
    max_x = min(w - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
    min_y = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
    max_y = min(h - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))

    if min_x > max_x or min_y > max_y:
        return 0

    shaded_pixels = 0

    for py in range(min_y, max_y + 1):
        y = float(py) + 0.5
        for px in range(min_x, max_x + 1):
            x = float(px) + 0.5

            w0 = edge_value(p1[0], p1[1], p2[0], p2[1], x, y) / area
            w1 = edge_value(p2[0], p2[1], p0[0], p0[1], x, y) / area
            w2 = edge_value(p0[0], p0[1], p1[0], p1[1], x, y) / area

            inside = (w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0) or (w0 <= 0.0 and w1 <= 0.0 and w2 <= 0.0)
            if not inside:
                continue

            depth = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
            if depth < 0.0 or depth > 1.0:
                continue

            if depth < depth_buffer[py, px]:
                depth_buffer[py, px] = depth
                image[py, px] = color
                shaded_pixels += 1

    return shaded_pixels


def save_ppm(path: Path, image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image with shape (H, W, 3), got {image.shape}.")

    h, w, _ = image.shape
    if image.dtype != np.uint8:
        image = np.clip(np.round(image), 0.0, 255.0).astype(np.uint8)

    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(image.tobytes())


def render_flat_shaded(width: int = 480, height: int = 360) -> Tuple[np.ndarray, RenderStats]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    vertices, triangles = generate_cube_mesh()

    model = make_rotation_y(np.deg2rad(32.0)) @ make_rotation_x(np.deg2rad(-24.0))
    eye = np.array([2.6, 2.4, 1.8], dtype=float)
    target = np.array([0.0, 0.0, 0.0], dtype=float)
    up = np.array([0.0, 1.0, 0.0], dtype=float)

    view = make_look_at(eye, target, up)
    projection = make_perspective(fov_y_deg=60.0, aspect=float(width) / float(height), z_near=0.1, z_far=10.0)

    model_view = view @ model
    mvp = projection @ model_view

    vertices_world = apply_transform(vertices, model)
    vertices_view = apply_transform(vertices, model_view)
    vertices_ndc = project_to_ndc(vertices, mvp)
    vertices_screen = ndc_to_screen(vertices_ndc, width=width, height=height)

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = np.array([18, 18, 24], dtype=np.uint8)
    depth_buffer = np.full((height, width), np.inf, dtype=float)

    light_dir_view = normalize(np.array([0.4, 0.65, 1.0], dtype=float))

    triangles_rasterized = 0
    pixels_shaded = 0
    intensities: List[float] = []

    for tri in triangles:
        n_view = face_normal(vertices_view, tri)
        n_world = face_normal(vertices_world, tri)
        face_color, face_intensity = shade_face(
            normal_view=n_view,
            normal_world=n_world,
            light_dir_view=light_dir_view,
        )

        p0 = vertices_screen[tri[0]]
        p1 = vertices_screen[tri[1]]
        p2 = vertices_screen[tri[2]]

        shaded = rasterize_triangle(
            image=image,
            depth_buffer=depth_buffer,
            p0=p0,
            p1=p1,
            p2=p2,
            color=face_color,
        )

        if shaded > 0:
            triangles_rasterized += 1
            pixels_shaded += shaded
            intensities.append(face_intensity)

    if pixels_shaded == 0:
        raise RuntimeError("No pixels were shaded; check camera/projection settings.")

    finite_depth = depth_buffer[np.isfinite(depth_buffer)]
    stats = RenderStats(
        triangles_total=int(triangles.shape[0]),
        triangles_rasterized=triangles_rasterized,
        pixels_shaded=pixels_shaded,
        mean_intensity=float(np.mean(intensities)) if intensities else 0.0,
        min_depth=float(np.min(finite_depth)),
        max_depth=float(np.max(finite_depth)),
    )

    return image, stats


def main() -> None:
    width = 480
    height = 360

    image, stats = render_flat_shaded(width=width, height=height)

    out_path = Path(__file__).with_name("flat_shading_output.ppm")
    save_ppm(out_path, image)

    print("Flat shading MVP finished.")
    print(f"output_image: {out_path}")
    print(f"resolution: {width}x{height}")
    print(f"triangles_total: {stats.triangles_total}")
    print(f"triangles_rasterized: {stats.triangles_rasterized}")
    print(f"pixels_shaded: {stats.pixels_shaded}")
    print(f"mean_face_intensity: {stats.mean_intensity:.6f}")
    print(f"depth_range: [{stats.min_depth:.6f}, {stats.max_depth:.6f}]")


if __name__ == "__main__":
    main()
