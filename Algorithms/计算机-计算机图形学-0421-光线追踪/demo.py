"""Minimal runnable MVP for basic ray tracing.

This script renders a tiny scene with:
- ray-sphere / ray-plane intersection
- ambient + Lambert diffuse + Phong specular shading
- hard shadows through shadow rays
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 1e-4


@dataclass(frozen=True)
class Material:
    color: np.ndarray
    ambient: float
    diffuse: float
    specular: float
    shininess: float


@dataclass(frozen=True)
class HitRecord:
    distance: float
    point: np.ndarray
    normal: np.ndarray
    material: Material
    object_name: str


@dataclass(frozen=True)
class Sphere:
    center: np.ndarray
    radius: float
    material: Material
    name: str

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> HitRecord | None:
        oc = origin - self.center
        a = float(np.dot(direction, direction))
        b = 2.0 * float(np.dot(oc, direction))
        c = float(np.dot(oc, oc) - self.radius * self.radius)

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None

        sqrt_disc = float(np.sqrt(disc))
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        t = None
        if t1 > EPSILON:
            t = t1
        elif t2 > EPSILON:
            t = t2

        if t is None:
            return None

        point = origin + t * direction
        normal = normalize(point - self.center)
        return HitRecord(t, point, normal, self.material, self.name)


@dataclass(frozen=True)
class Plane:
    point: np.ndarray
    normal: np.ndarray
    material: Material
    name: str

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> HitRecord | None:
        denom = float(np.dot(direction, self.normal))
        if abs(denom) < 1e-8:
            return None

        t = float(np.dot(self.point - origin, self.normal) / denom)
        if t <= EPSILON:
            return None

        hit_point = origin + t * direction
        corrected_normal = self.normal if denom < 0.0 else -self.normal
        return HitRecord(t, hit_point, corrected_normal, self.material, self.name)


@dataclass(frozen=True)
class PointLight:
    position: np.ndarray
    color: np.ndarray
    intensity: float


def normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length < 1e-12:
        return v
    return v / length


def reflect(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - 2.0 * float(np.dot(v, n)) * n


class RayTracer:
    def __init__(
        self,
        objects: list[Sphere | Plane],
        lights: list[PointLight],
        ambient_light: np.ndarray | None = None,
    ) -> None:
        self.objects = objects
        self.lights = lights
        self.ambient_light = (
            np.array([0.14, 0.14, 0.14], dtype=np.float64)
            if ambient_light is None
            else ambient_light.astype(np.float64)
        )
        self.stats = {
            "rays_cast": 0,
            "shadow_rays": 0,
            "intersection_tests": 0,
            "surface_hits": 0,
            "shadowed_light_samples": 0,
        }

    def background(self, direction: np.ndarray) -> np.ndarray:
        t = 0.5 * (direction[1] + 1.0)
        horizon = np.array([0.78, 0.86, 0.98], dtype=np.float64)
        zenith = np.array([0.22, 0.36, 0.62], dtype=np.float64)
        return (1.0 - t) * horizon + t * zenith

    def find_nearest_hit(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float = np.inf,
    ) -> HitRecord | None:
        nearest: HitRecord | None = None
        nearest_t = max_distance

        for obj in self.objects:
            self.stats["intersection_tests"] += 1
            hit = obj.intersect(origin, direction)
            if hit is not None and hit.distance < nearest_t:
                nearest_t = hit.distance
                nearest = hit

        if nearest is not None:
            self.stats["surface_hits"] += 1
        return nearest

    def is_shadowed(
        self,
        origin: np.ndarray,
        direction_to_light: np.ndarray,
        light_distance: float,
    ) -> bool:
        self.stats["shadow_rays"] += 1
        hit = self.find_nearest_hit(
            origin=origin,
            direction=direction_to_light,
            max_distance=light_distance - EPSILON,
        )
        return hit is not None

    def shade(self, hit: HitRecord, ray_direction: np.ndarray) -> np.ndarray:
        mat = hit.material
        view_dir = normalize(-ray_direction)

        color = mat.ambient * mat.color * self.ambient_light

        for light in self.lights:
            light_vec = light.position - hit.point
            light_distance = float(np.linalg.norm(light_vec))
            if light_distance <= EPSILON:
                continue

            light_dir = light_vec / light_distance
            shadow_origin = hit.point + hit.normal * (4.0 * EPSILON)
            if self.is_shadowed(shadow_origin, light_dir, light_distance):
                self.stats["shadowed_light_samples"] += 1
                continue

            ndotl = max(float(np.dot(hit.normal, light_dir)), 0.0)
            diffuse = mat.diffuse * ndotl * mat.color * light.color * light.intensity

            reflected = normalize(reflect(-light_dir, hit.normal))
            spec_angle = max(float(np.dot(view_dir, reflected)), 0.0)
            specular = (
                mat.specular
                * (spec_angle**mat.shininess)
                * light.color
                * light.intensity
            )

            color += diffuse + specular

        return np.clip(color, 0.0, 1.0)

    def trace_ray(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        self.stats["rays_cast"] += 1
        hit = self.find_nearest_hit(origin, direction)
        if hit is None:
            return self.background(direction)
        return self.shade(hit, direction)

    def render(
        self,
        width: int,
        height: int,
        fov_deg: float,
        camera_origin: np.ndarray,
    ) -> np.ndarray:
        image = np.zeros((height, width, 3), dtype=np.float64)
        aspect = width / float(height)
        scale = np.tan(np.deg2rad(0.5 * fov_deg))

        for y in range(height):
            py = (1.0 - 2.0 * ((y + 0.5) / height)) * scale
            for x in range(width):
                px = (2.0 * ((x + 0.5) / width) - 1.0) * aspect * scale
                direction = normalize(np.array([px, py, 1.0], dtype=np.float64))
                image[y, x] = self.trace_ray(camera_origin, direction)

        return image


def save_ppm(path: Path, image: np.ndarray) -> None:
    rgb8 = np.clip(image * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    height, width, _ = rgb8.shape
    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(rgb8.tobytes())


def build_default_scene() -> tuple[list[Sphere | Plane], list[PointLight]]:
    clay = Material(
        color=np.array([0.88, 0.42, 0.22], dtype=np.float64),
        ambient=0.20,
        diffuse=0.70,
        specular=0.20,
        shininess=24.0,
    )
    cobalt = Material(
        color=np.array([0.25, 0.46, 0.90], dtype=np.float64),
        ambient=0.16,
        diffuse=0.66,
        specular=0.48,
        shininess=56.0,
    )
    floor = Material(
        color=np.array([0.80, 0.80, 0.78], dtype=np.float64),
        ambient=0.22,
        diffuse=0.62,
        specular=0.08,
        shininess=10.0,
    )

    objects: list[Sphere | Plane] = [
        Sphere(
            center=np.array([-0.92, -0.02, 4.3], dtype=np.float64),
            radius=0.95,
            material=clay,
            name="clay_sphere",
        ),
        Sphere(
            center=np.array([1.08, -0.28, 5.5], dtype=np.float64),
            radius=0.75,
            material=cobalt,
            name="cobalt_sphere",
        ),
        Plane(
            point=np.array([0.0, -1.0, 0.0], dtype=np.float64),
            normal=normalize(np.array([0.0, 1.0, 0.0], dtype=np.float64)),
            material=floor,
            name="ground_plane",
        ),
    ]

    lights = [
        PointLight(
            position=np.array([2.6, 4.2, 0.2], dtype=np.float64),
            color=np.array([1.0, 0.98, 0.95], dtype=np.float64),
            intensity=1.00,
        ),
        PointLight(
            position=np.array([-3.1, 2.4, 1.8], dtype=np.float64),
            color=np.array([0.72, 0.82, 1.0], dtype=np.float64),
            intensity=0.62,
        ),
    ]
    return objects, lights


def build_image_summary(image: np.ndarray) -> pd.DataFrame:
    h, w, _ = image.shape
    probes = {
        "top_left": image[0, 0],
        "center": image[h // 2, w // 2],
        "bottom_right": image[h - 1, w - 1],
        "global_mean": np.mean(image, axis=(0, 1)),
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


def run_sanity_checks(image: np.ndarray, stats: dict[str, int]) -> None:
    assert image.ndim == 3 and image.shape[2] == 3
    assert float(np.min(image)) >= 0.0
    assert float(np.max(image)) <= 1.0

    mean_luminance = float(np.mean(image))
    assert 0.10 <= mean_luminance <= 0.90

    assert stats["rays_cast"] > 0
    assert stats["surface_hits"] > 0
    assert stats["shadow_rays"] > 0
    assert stats["intersection_tests"] > 0
    assert stats["shadowed_light_samples"] > 0


def main() -> None:
    width, height = 240, 160
    fov_deg = 58.0
    camera_origin = np.array([0.0, 0.1, 0.0], dtype=np.float64)

    objects, lights = build_default_scene()
    tracer = RayTracer(objects=objects, lights=lights)
    image = tracer.render(
        width=width,
        height=height,
        fov_deg=fov_deg,
        camera_origin=camera_origin,
    )

    output_path = Path(__file__).resolve().parent / "render_ray_tracing.ppm"
    save_ppm(output_path, image)

    run_sanity_checks(image, tracer.stats)
    summary = build_image_summary(image)

    print("Basic ray tracing MVP completed.")
    print(f"Image file: {output_path}")
    print(f"Resolution: {width}x{height}, FOV(deg): {fov_deg}")
    print("Ray statistics:")
    for key, value in tracer.stats.items():
        print(f"  - {key}: {value}")

    print("\nImage summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
