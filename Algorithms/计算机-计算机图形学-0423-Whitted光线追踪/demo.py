"""Minimal runnable MVP for Whitted-style ray tracing.

This script renders a tiny scene with spheres and a plane, featuring:
- hard shadows (shadow rays)
- local Phong lighting
- recursive mirror reflection
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
    reflectivity: float


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
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None

        sqrt_disc = float(np.sqrt(discriminant))
        t_small = (-b - sqrt_disc) / (2.0 * a)
        t_large = (-b + sqrt_disc) / (2.0 * a)

        t = None
        if t_small > EPSILON:
            t = t_small
        elif t_large > EPSILON:
            t = t_large

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


class WhittedTracer:
    def __init__(
        self,
        objects: list[Sphere | Plane],
        lights: list[PointLight],
        max_depth: int = 3,
        ambient_light: np.ndarray | None = None,
    ) -> None:
        self.objects = objects
        self.lights = lights
        self.max_depth = max_depth
        self.ambient_light = (
            np.array([0.12, 0.12, 0.12], dtype=np.float64)
            if ambient_light is None
            else ambient_light.astype(np.float64)
        )
        self.stats = {
            "rays_cast": 0,
            "reflection_rays": 0,
            "shadow_rays": 0,
            "intersection_tests": 0,
            "surface_hits": 0,
        }

    def background(self, direction: np.ndarray) -> np.ndarray:
        t = 0.5 * (direction[1] + 1.0)
        bottom = np.array([0.72, 0.84, 1.00], dtype=np.float64)
        top = np.array([0.13, 0.25, 0.50], dtype=np.float64)
        return (1.0 - t) * bottom + t * top

    def find_nearest_hit(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float = np.inf,
    ) -> HitRecord | None:
        nearest: HitRecord | None = None
        min_distance = max_distance

        for obj in self.objects:
            self.stats["intersection_tests"] += 1
            hit = obj.intersect(origin, direction)
            if hit is not None and hit.distance < min_distance:
                min_distance = hit.distance
                nearest = hit

        if nearest is not None:
            self.stats["surface_hits"] += 1
        return nearest

    def occluded(
        self,
        origin: np.ndarray,
        direction_to_light: np.ndarray,
        light_distance: float,
    ) -> bool:
        self.stats["shadow_rays"] += 1
        for obj in self.objects:
            self.stats["intersection_tests"] += 1
            hit = obj.intersect(origin, direction_to_light)
            if hit is not None and hit.distance < (light_distance - EPSILON):
                return True
        return False

    def shade(self, hit: HitRecord, ray_direction: np.ndarray, depth: int) -> np.ndarray:
        mat = hit.material
        view_dir = normalize(-ray_direction)

        local_color = mat.ambient * mat.color * self.ambient_light

        for light in self.lights:
            to_light = light.position - hit.point
            light_distance = float(np.linalg.norm(to_light))
            if light_distance <= EPSILON:
                continue

            light_dir = to_light / light_distance
            shadow_origin = hit.point + hit.normal * (4.0 * EPSILON)
            if self.occluded(shadow_origin, light_dir, light_distance):
                continue

            ndotl = max(float(np.dot(hit.normal, light_dir)), 0.0)
            diffuse = mat.diffuse * ndotl * mat.color * light.color * light.intensity

            reflected_light = normalize(reflect(-light_dir, hit.normal))
            spec_angle = max(float(np.dot(view_dir, reflected_light)), 0.0)
            specular = (
                mat.specular
                * (spec_angle**mat.shininess)
                * light.color
                * light.intensity
            )
            local_color += diffuse + specular

        if mat.reflectivity > 0.0 and depth < self.max_depth:
            self.stats["reflection_rays"] += 1
            reflected_dir = normalize(reflect(ray_direction, hit.normal))
            reflected_origin = hit.point + hit.normal * (4.0 * EPSILON)
            reflected_color = self.trace_ray(reflected_origin, reflected_dir, depth + 1)
            local_color = (
                (1.0 - mat.reflectivity) * local_color
                + mat.reflectivity * reflected_color
            )

        return np.clip(local_color, 0.0, 1.0)

    def trace_ray(self, origin: np.ndarray, direction: np.ndarray, depth: int) -> np.ndarray:
        self.stats["rays_cast"] += 1
        if depth > self.max_depth:
            return self.background(direction)

        hit = self.find_nearest_hit(origin, direction)
        if hit is None:
            return self.background(direction)

        return self.shade(hit, direction, depth)

    def render(
        self,
        width: int,
        height: int,
        fov_deg: float,
        camera_origin: np.ndarray,
    ) -> np.ndarray:
        image = np.zeros((height, width, 3), dtype=np.float64)
        aspect_ratio = width / float(height)
        scale = np.tan(np.deg2rad(0.5 * fov_deg))

        for y in range(height):
            py = (1.0 - 2.0 * ((y + 0.5) / height)) * scale
            for x in range(width):
                px = (2.0 * ((x + 0.5) / width) - 1.0) * aspect_ratio * scale
                direction = normalize(np.array([px, py, 1.0], dtype=np.float64))
                image[y, x] = self.trace_ray(camera_origin, direction, depth=0)

        return image


def save_ppm(path: Path, image: np.ndarray) -> None:
    clipped = np.clip(image * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    height, width, _ = clipped.shape
    with path.open("wb") as f:
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        f.write(header)
        f.write(clipped.tobytes())


def build_default_scene() -> tuple[list[Sphere | Plane], list[PointLight]]:
    ruby = Material(
        color=np.array([0.90, 0.20, 0.25], dtype=np.float64),
        ambient=0.16,
        diffuse=0.72,
        specular=0.36,
        shininess=48.0,
        reflectivity=0.18,
    )
    glass = Material(
        color=np.array([0.18, 0.45, 0.85], dtype=np.float64),
        ambient=0.10,
        diffuse=0.58,
        specular=0.88,
        shininess=96.0,
        reflectivity=0.65,
    )
    floor = Material(
        color=np.array([0.78, 0.80, 0.78], dtype=np.float64),
        ambient=0.18,
        diffuse=0.66,
        specular=0.14,
        shininess=20.0,
        reflectivity=0.07,
    )

    objects: list[Sphere | Plane] = [
        Sphere(
            center=np.array([-0.95, 0.0, 4.6], dtype=np.float64),
            radius=1.0,
            material=ruby,
            name="ruby_sphere",
        ),
        Sphere(
            center=np.array([1.05, -0.18, 5.8], dtype=np.float64),
            radius=1.0,
            material=glass,
            name="glass_sphere",
        ),
        Plane(
            point=np.array([0.0, -1.05, 0.0], dtype=np.float64),
            normal=normalize(np.array([0.0, 1.0, 0.0], dtype=np.float64)),
            material=floor,
            name="ground_plane",
        ),
    ]

    lights = [
        PointLight(
            position=np.array([2.8, 4.4, 0.6], dtype=np.float64),
            color=np.array([1.0, 0.98, 0.96], dtype=np.float64),
            intensity=1.0,
        ),
        PointLight(
            position=np.array([-3.5, 2.6, 2.2], dtype=np.float64),
            color=np.array([0.75, 0.84, 1.0], dtype=np.float64),
            intensity=0.65,
        ),
    ]

    return objects, lights


def build_image_summary(image: np.ndarray) -> pd.DataFrame:
    h, w, _ = image.shape
    probes = {
        "top_left": image[0, 0],
        "center": image[h // 2, w // 2],
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

    rows.append(
        {
            "probe": "global_mean",
            "r": float(np.mean(image[:, :, 0])),
            "g": float(np.mean(image[:, :, 1])),
            "b": float(np.mean(image[:, :, 2])),
            "luminance": float(np.mean(image)),
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
    assert stats["reflection_rays"] > 0


def main() -> None:
    width, height = 240, 160
    fov_deg = 58.0
    camera_origin = np.array([0.0, 0.15, 0.0], dtype=np.float64)

    objects, lights = build_default_scene()
    tracer = WhittedTracer(objects=objects, lights=lights, max_depth=3)
    image = tracer.render(width=width, height=height, fov_deg=fov_deg, camera_origin=camera_origin)

    output_path = Path(__file__).resolve().parent / "render_whitted.ppm"
    save_ppm(output_path, image)

    run_sanity_checks(image, tracer.stats)
    summary = build_image_summary(image)

    print("Whitted ray tracing MVP completed.")
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
