"""Minimal runnable MVP for Monte Carlo path tracing.

This script renders a tiny diffuse scene using:
- ray-sphere / ray-plane intersection
- cosine-weighted hemisphere sampling
- multi-bounce global illumination estimation
- Russian roulette termination
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 1e-4


@dataclass(frozen=True)
class Material:
    albedo: np.ndarray
    emission: np.ndarray


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


def normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length < 1e-12:
        return v
    return v / length


def sample_cosine_hemisphere(normal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u1, u2 = rng.random(2)
    r = float(np.sqrt(u1))
    theta = float(2.0 * np.pi * u2)

    local = np.array(
        [r * np.cos(theta), r * np.sin(theta), np.sqrt(max(0.0, 1.0 - u1))],
        dtype=np.float64,
    )

    if abs(normal[2]) < 0.999:
        tangent = normalize(np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), normal))
    else:
        tangent = normalize(np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), normal))
    bitangent = np.cross(normal, tangent)

    world = local[0] * tangent + local[1] * bitangent + local[2] * normal
    return normalize(world)


class PathTracer:
    def __init__(
        self,
        objects: list[Sphere | Plane],
        max_depth: int = 6,
        rr_start_depth: int = 3,
    ) -> None:
        self.objects = objects
        self.max_depth = max_depth
        self.rr_start_depth = rr_start_depth
        self.stats = {
            "camera_rays": 0,
            "path_segments": 0,
            "intersection_tests": 0,
            "surface_hits": 0,
            "escaped_rays": 0,
            "russian_roulette_terminated": 0,
        }

    def background(self, direction: np.ndarray) -> np.ndarray:
        t = 0.5 * (direction[1] + 1.0)
        horizon = np.array([0.88, 0.93, 1.00], dtype=np.float64)
        zenith = np.array([0.32, 0.46, 0.72], dtype=np.float64)
        return (1.0 - t) * horizon + t * zenith

    def find_nearest_hit(self, origin: np.ndarray, direction: np.ndarray) -> HitRecord | None:
        nearest: HitRecord | None = None
        nearest_t = np.inf

        for obj in self.objects:
            self.stats["intersection_tests"] += 1
            hit = obj.intersect(origin, direction)
            if hit is not None and hit.distance < nearest_t:
                nearest = hit
                nearest_t = hit.distance

        if nearest is not None:
            self.stats["surface_hits"] += 1
        return nearest

    def trace_path(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        radiance = np.zeros(3, dtype=np.float64)
        throughput = np.ones(3, dtype=np.float64)

        ray_origin = origin
        ray_dir = direction

        for depth in range(self.max_depth):
            self.stats["path_segments"] += 1
            hit = self.find_nearest_hit(ray_origin, ray_dir)

            if hit is None:
                radiance += throughput * self.background(ray_dir)
                self.stats["escaped_rays"] += 1
                break

            material = hit.material
            radiance += throughput * material.emission

            new_dir = sample_cosine_hemisphere(hit.normal, rng)
            throughput *= material.albedo

            if depth >= self.rr_start_depth:
                survive_prob = min(0.95, float(np.max(throughput)))
                if survive_prob <= 1e-8 or float(rng.random()) > survive_prob:
                    self.stats["russian_roulette_terminated"] += 1
                    break
                throughput /= survive_prob

            ray_origin = hit.point + hit.normal * (4.0 * EPSILON)
            ray_dir = new_dir

        return radiance

    def render(
        self,
        width: int,
        height: int,
        fov_deg: float,
        camera_origin: np.ndarray,
        samples_per_pixel: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        image = np.zeros((height, width, 3), dtype=np.float64)
        aspect = width / float(height)
        scale = np.tan(np.deg2rad(0.5 * fov_deg))

        for y in range(height):
            for x in range(width):
                pixel_radiance = np.zeros(3, dtype=np.float64)
                for _ in range(samples_per_pixel):
                    jx, jy = rng.random(2)
                    px = (2.0 * ((x + jx) / width) - 1.0) * aspect * scale
                    py = (1.0 - 2.0 * ((y + jy) / height)) * scale
                    direction = normalize(np.array([px, py, 1.0], dtype=np.float64))

                    self.stats["camera_rays"] += 1
                    pixel_radiance += self.trace_path(camera_origin, direction, rng)

                image[y, x] = pixel_radiance / float(samples_per_pixel)

        return image


def tonemap_and_gamma(image_linear: np.ndarray) -> np.ndarray:
    mapped = image_linear / (1.0 + image_linear)
    gamma = np.power(np.clip(mapped, 0.0, 1.0), 1.0 / 2.2)
    return np.clip(gamma, 0.0, 1.0)


def save_ppm(path: Path, image: np.ndarray) -> None:
    rgb8 = np.clip(image * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    height, width, _ = rgb8.shape
    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(rgb8.tobytes())


def build_default_scene() -> list[Sphere | Plane]:
    matte_red = Material(
        albedo=np.array([0.80, 0.28, 0.24], dtype=np.float64),
        emission=np.zeros(3, dtype=np.float64),
    )
    matte_blue = Material(
        albedo=np.array([0.24, 0.45, 0.84], dtype=np.float64),
        emission=np.zeros(3, dtype=np.float64),
    )
    matte_floor = Material(
        albedo=np.array([0.78, 0.78, 0.74], dtype=np.float64),
        emission=np.zeros(3, dtype=np.float64),
    )
    emissive = Material(
        albedo=np.zeros(3, dtype=np.float64),
        emission=np.array([8.0, 7.6, 6.8], dtype=np.float64),
    )

    objects: list[Sphere | Plane] = [
        Sphere(
            center=np.array([-0.85, -0.08, 4.4], dtype=np.float64),
            radius=0.92,
            material=matte_red,
            name="red_sphere",
        ),
        Sphere(
            center=np.array([1.10, -0.22, 5.6], dtype=np.float64),
            radius=0.78,
            material=matte_blue,
            name="blue_sphere",
        ),
        Sphere(
            center=np.array([0.25, 3.5, 4.6], dtype=np.float64),
            radius=0.62,
            material=emissive,
            name="light_sphere",
        ),
        Plane(
            point=np.array([0.0, -1.0, 0.0], dtype=np.float64),
            normal=normalize(np.array([0.0, 1.0, 0.0], dtype=np.float64)),
            material=matte_floor,
            name="ground_plane",
        ),
    ]
    return objects


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


def run_sanity_checks(image_linear: np.ndarray, image_display: np.ndarray, stats: dict[str, int]) -> None:
    assert image_linear.ndim == 3 and image_linear.shape[2] == 3
    assert image_display.ndim == 3 and image_display.shape[2] == 3

    assert float(np.min(image_display)) >= 0.0
    assert float(np.max(image_display)) <= 1.0

    mean_linear = float(np.mean(image_linear))
    mean_display = float(np.mean(image_display))
    assert 0.02 <= mean_linear <= 8.0
    assert 0.08 <= mean_display <= 0.92

    assert stats["camera_rays"] > 0
    assert stats["path_segments"] >= stats["camera_rays"]
    assert stats["intersection_tests"] > 0
    assert stats["surface_hits"] > 0
    assert stats["escaped_rays"] > 0


def main() -> None:
    width, height = 200, 132
    fov_deg = 56.0
    spp = 24
    camera_origin = np.array([0.0, 0.1, 0.0], dtype=np.float64)

    rng = np.random.default_rng(20260407)
    objects = build_default_scene()
    tracer = PathTracer(objects=objects, max_depth=6, rr_start_depth=3)

    image_linear = tracer.render(
        width=width,
        height=height,
        fov_deg=fov_deg,
        camera_origin=camera_origin,
        samples_per_pixel=spp,
        rng=rng,
    )
    image_display = tonemap_and_gamma(image_linear)

    output_path = Path(__file__).resolve().parent / "render_path_tracing.ppm"
    save_ppm(output_path, image_display)

    run_sanity_checks(image_linear=image_linear, image_display=image_display, stats=tracer.stats)
    summary = build_image_summary(image_display)

    print("Path tracing MVP completed.")
    print(f"Image file: {output_path}")
    print(f"Resolution: {width}x{height}, FOV(deg): {fov_deg}, SPP: {spp}")
    print("Ray statistics:")
    for key, value in tracer.stats.items():
        print(f"  - {key}: {value}")

    print("\nImage summary (display-space):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
