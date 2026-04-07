"""Phong illumination MVP with analytic ray-sphere intersection.

This script renders one sphere under point lights using ambient + Lambert diffuse
+ Phong specular (r·v)^n, writes a PPM image, and prints deterministic diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 1e-6


@dataclass(frozen=True)
class Material:
    albedo: np.ndarray
    ambient: float
    diffuse: float
    specular: float
    shininess: float


@dataclass(frozen=True)
class PointLight:
    position: np.ndarray
    color: np.ndarray
    intensity: float


@dataclass(frozen=True)
class Camera:
    origin: np.ndarray
    fov_deg: float


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < EPSILON:
        return v
    return v / n


def reflect(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - 2.0 * float(np.dot(v, n)) * n


def ray_sphere_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
) -> float | None:
    oc = ray_origin - sphere_center
    a = float(np.dot(ray_direction, ray_direction))
    b = 2.0 * float(np.dot(oc, ray_direction))
    c = float(np.dot(oc, oc) - sphere_radius * sphere_radius)

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None

    sqrt_disc = float(np.sqrt(disc))
    t0 = (-b - sqrt_disc) / (2.0 * a)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    if t0 > EPSILON:
        return t0
    if t1 > EPSILON:
        return t1
    return None


def background_color(direction: np.ndarray) -> np.ndarray:
    t = 0.5 * (direction[1] + 1.0)
    top = np.array([0.14, 0.24, 0.42], dtype=np.float64)
    bottom = np.array([0.90, 0.95, 1.00], dtype=np.float64)
    return (1.0 - t) * bottom + t * top


def phong_shade(
    point: np.ndarray,
    normal: np.ndarray,
    view_dir: np.ndarray,
    material: Material,
    lights: list[PointLight],
    ambient_light: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    color = material.ambient * material.albedo * ambient_light

    ndotl_sum = 0.0
    rdotv_sum = 0.0
    phong_spec_sum = 0.0
    lit_samples = 0.0

    for light in lights:
        light_vec = light.position - point
        light_dist = float(np.linalg.norm(light_vec))
        if light_dist < EPSILON:
            continue

        light_dir = light_vec / light_dist
        ndotl = max(float(np.dot(normal, light_dir)), 0.0)
        if ndotl <= 0.0:
            continue

        diffuse = (
            material.diffuse
            * ndotl
            * material.albedo
            * light.color
            * light.intensity
        )

        reflected = normalize(reflect(-light_dir, normal))
        rdotv = max(float(np.dot(reflected, view_dir)), 0.0)
        phong_strength = rdotv**material.shininess
        specular = material.specular * phong_strength * light.color * light.intensity

        color += diffuse + specular

        ndotl_sum += ndotl
        rdotv_sum += rdotv
        phong_spec_sum += phong_strength
        lit_samples += 1.0

    diagnostics = {
        "ndotl_sum": ndotl_sum,
        "rdotv_sum": rdotv_sum,
        "phong_spec_sum": phong_spec_sum,
        "lit_samples": lit_samples,
    }
    return np.clip(color, 0.0, 1.0), diagnostics


def render_phong_sphere(
    width: int,
    height: int,
    camera: Camera,
    sphere_center: np.ndarray,
    sphere_radius: float,
    material: Material,
    lights: list[PointLight],
    ambient_light: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    image = np.zeros((height, width, 3), dtype=np.float64)
    aspect = width / float(height)
    scale = np.tan(np.deg2rad(0.5 * camera.fov_deg))

    hit_pixels = 0
    background_pixels = 0
    ndotl_sum = 0.0
    rdotv_sum = 0.0
    phong_spec_sum = 0.0
    lit_samples = 0.0

    for y in range(height):
        py = (1.0 - 2.0 * ((y + 0.5) / height)) * scale
        for x in range(width):
            px = (2.0 * ((x + 0.5) / width) - 1.0) * aspect * scale
            ray_dir = normalize(np.array([px, py, 1.0], dtype=np.float64))

            t = ray_sphere_intersection(
                ray_origin=camera.origin,
                ray_direction=ray_dir,
                sphere_center=sphere_center,
                sphere_radius=sphere_radius,
            )
            if t is None:
                image[y, x] = background_color(ray_dir)
                background_pixels += 1
                continue

            hit_point = camera.origin + t * ray_dir
            normal = normalize(hit_point - sphere_center)
            view_dir = normalize(camera.origin - hit_point)

            shaded, diag = phong_shade(
                point=hit_point,
                normal=normal,
                view_dir=view_dir,
                material=material,
                lights=lights,
                ambient_light=ambient_light,
            )
            image[y, x] = shaded
            hit_pixels += 1

            ndotl_sum += diag["ndotl_sum"]
            rdotv_sum += diag["rdotv_sum"]
            phong_spec_sum += diag["phong_spec_sum"]
            lit_samples += diag["lit_samples"]

    stats = {
        "width": float(width),
        "height": float(height),
        "pixels_total": float(width * height),
        "hit_pixels": float(hit_pixels),
        "background_pixels": float(background_pixels),
        "hit_ratio": float(hit_pixels) / float(width * height),
        "mean_luminance": float(np.mean(image)),
        "mean_ndotl": ndotl_sum / max(lit_samples, 1.0),
        "mean_rdotv": rdotv_sum / max(lit_samples, 1.0),
        "mean_phong_spec_strength": phong_spec_sum / max(lit_samples, 1.0),
        "lit_samples": lit_samples,
    }
    return image, stats


def save_ppm(path: Path, image: np.ndarray) -> None:
    rgb8 = np.clip(image * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    h, w, _ = rgb8.shape
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb8.tobytes())


def build_probe_table(image: np.ndarray) -> pd.DataFrame:
    h, w, _ = image.shape
    probes = {
        "top_left": image[0, 0],
        "center": image[h // 2, w // 2],
        "sphere_top": image[int(0.30 * h), w // 2],
        "sphere_left": image[h // 2, int(0.32 * w)],
        "bottom_right": image[h - 1, w - 1],
    }

    rows: list[dict[str, float | str]] = []
    for name, rgb in probes.items():
        rows.append(
            {
                "probe": name,
                "r": float(rgb[0]),
                "g": float(rgb[1]),
                "b": float(rgb[2]),
                "luminance": float(np.mean(rgb)),
            }
        )
    return pd.DataFrame(rows)


def run_checks(image: np.ndarray, stats: dict[str, float]) -> None:
    assert image.ndim == 3 and image.shape[2] == 3
    assert float(np.min(image)) >= 0.0
    assert float(np.max(image)) <= 1.0

    assert stats["hit_pixels"] > 0.0
    assert stats["background_pixels"] > 0.0
    assert 0.10 <= stats["hit_ratio"] <= 0.80
    assert 0.10 <= stats["mean_luminance"] <= 0.95
    assert stats["lit_samples"] > 0.0
    assert stats["mean_ndotl"] > 0.0
    assert stats["mean_phong_spec_strength"] >= 0.0


def main() -> None:
    width, height = 320, 220
    camera = Camera(
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        fov_deg=54.0,
    )

    sphere_center = np.array([0.0, -0.03, 3.75], dtype=np.float64)
    sphere_radius = 1.12

    material = Material(
        albedo=np.array([0.56, 0.62, 0.93], dtype=np.float64),
        ambient=0.18,
        diffuse=0.74,
        specular=0.56,
        shininess=42.0,
    )

    lights = [
        PointLight(
            position=np.array([2.2, 2.4, 0.5], dtype=np.float64),
            color=np.array([1.0, 0.98, 0.93], dtype=np.float64),
            intensity=1.08,
        ),
        PointLight(
            position=np.array([-3.1, 1.0, 1.8], dtype=np.float64),
            color=np.array([0.74, 0.84, 1.0], dtype=np.float64),
            intensity=0.52,
        ),
    ]

    ambient_light = np.array([0.20, 0.21, 0.25], dtype=np.float64)

    image, stats = render_phong_sphere(
        width=width,
        height=height,
        camera=camera,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        material=material,
        lights=lights,
        ambient_light=ambient_light,
    )

    output_path = Path(__file__).resolve().parent / "phong_sphere.ppm"
    save_ppm(output_path, image)
    run_checks(image, stats)
    probe_df = build_probe_table(image)

    print("Phong model MVP completed.")
    print(f"Image file: {output_path}")
    print(f"Resolution: {width}x{height}, FOV: {camera.fov_deg:.1f} deg")
    print("Render statistics:")
    for key, value in stats.items():
        print(f"  - {key}: {value:.6f}")

    print("\nProbe colors:")
    print(probe_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
