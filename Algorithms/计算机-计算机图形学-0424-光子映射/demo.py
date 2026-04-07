"""光子映射 (Photon Mapping): minimal runnable MVP.

实现目标:
1. 从点光源发射光子并进行多次漫反射追踪；
2. 在命中点存储光子（位置、入射方向、功率、法线）；
3. 使用 KD-Tree 做 k 近邻收集，估计查询点辐照度；
4. 输出样本表与关键指标，并包含基础质量门禁。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


Array = np.ndarray
EPS = 1e-6


@dataclass
class HitRecord:
    t: float
    position: Array
    normal: Array
    albedo: Array
    object_name: str


@dataclass
class PhotonMap:
    positions: Array  # (M, 3)
    incoming: Array  # (M, 3), direction toward surface
    power: Array  # (M, 3), RGB flux carried by photon
    normals: Array  # (M, 3)


FLOOR_ALBEDO = np.array([0.78, 0.78, 0.78], dtype=float)
WALL_ALBEDO = np.array([0.35, 0.80, 0.35], dtype=float)
SPHERE_ALBEDO = np.array([0.90, 0.25, 0.25], dtype=float)


def normalize(v: Array) -> Array:
    n = float(np.linalg.norm(v))
    if n < EPS:
        return v
    return v / n


def build_onb(normal: Array) -> tuple[Array, Array, Array]:
    """Build an orthonormal basis (tangent, bitangent, normal)."""
    n = normalize(normal)
    if abs(n[2]) < 0.999:
        tangent = normalize(np.cross(n, np.array([0.0, 0.0, 1.0], dtype=float)))
    else:
        tangent = normalize(np.cross(n, np.array([0.0, 1.0, 0.0], dtype=float)))
    bitangent = normalize(np.cross(n, tangent))
    return tangent, bitangent, n


def sample_cosine_hemisphere(normal: Array, rng: np.random.Generator) -> Array:
    """Cosine-weighted hemisphere sampling around the given normal."""
    u1 = float(rng.random())
    u2 = float(rng.random())

    r = math.sqrt(u1)
    phi = 2.0 * math.pi * u2
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    z = math.sqrt(max(0.0, 1.0 - u1))

    t, b, n = build_onb(normal)
    world = x * t + y * b + z * n
    return normalize(world)


def intersect_floor(origin: Array, direction: Array) -> HitRecord | None:
    """Plane y=0, bounded in x/z for a finite scene."""
    if direction[1] >= -EPS:
        return None

    t = -origin[1] / direction[1]
    if t <= EPS:
        return None

    p = origin + t * direction
    if p[0] < -6.0 or p[0] > 6.0 or p[2] < -1.0 or p[2] > 10.0:
        return None

    return HitRecord(
        t=float(t),
        position=p,
        normal=np.array([0.0, 1.0, 0.0], dtype=float),
        albedo=FLOOR_ALBEDO,
        object_name="floor",
    )


def intersect_back_wall(origin: Array, direction: Array) -> HitRecord | None:
    """Plane z=9, inward normal -z."""
    if direction[2] <= EPS:
        return None

    t = (9.0 - origin[2]) / direction[2]
    if t <= EPS:
        return None

    p = origin + t * direction
    if p[0] < -6.0 or p[0] > 6.0 or p[1] < 0.0 or p[1] > 8.0:
        return None

    return HitRecord(
        t=float(t),
        position=p,
        normal=np.array([0.0, 0.0, -1.0], dtype=float),
        albedo=WALL_ALBEDO,
        object_name="back_wall",
    )


def intersect_sphere(origin: Array, direction: Array) -> HitRecord | None:
    center = np.array([-1.2, 1.1, 4.6], dtype=float)
    radius = 1.1

    oc = origin - center
    a = float(np.dot(direction, direction))
    b = 2.0 * float(np.dot(oc, direction))
    c = float(np.dot(oc, oc) - radius * radius)

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None

    sqrt_disc = math.sqrt(disc)
    t0 = (-b - sqrt_disc) / (2.0 * a)
    t1 = (-b + sqrt_disc) / (2.0 * a)

    t = None
    if t0 > EPS:
        t = t0
    elif t1 > EPS:
        t = t1
    if t is None:
        return None

    p = origin + t * direction
    n = normalize(p - center)
    return HitRecord(
        t=float(t),
        position=p,
        normal=n,
        albedo=SPHERE_ALBEDO,
        object_name="sphere",
    )


def intersect_scene(origin: Array, direction: Array) -> HitRecord | None:
    """Return nearest hit among all scene primitives."""
    candidates = [
        intersect_floor(origin, direction),
        intersect_back_wall(origin, direction),
        intersect_sphere(origin, direction),
    ]
    valid = [h for h in candidates if h is not None]
    if not valid:
        return None
    return min(valid, key=lambda h: h.t)


def emit_and_trace_photons(
    n_emit: int,
    max_bounces: int,
    light_pos: Array,
    light_power_rgb: Array,
    rng: np.random.Generator,
) -> PhotonMap:
    """Emit photons and trace diffuse bounces with Russian roulette."""
    if n_emit <= 0:
        raise ValueError("n_emit must be positive")
    if max_bounces <= 0:
        raise ValueError("max_bounces must be positive")

    down = np.array([0.0, -1.0, 0.0], dtype=float)
    power0 = light_power_rgb / float(n_emit)

    pos_buf: list[Array] = []
    incoming_buf: list[Array] = []
    power_buf: list[Array] = []
    normal_buf: list[Array] = []

    for _ in range(n_emit):
        origin = light_pos.copy()
        direction = sample_cosine_hemisphere(down, rng)
        power = power0.copy()

        for _bounce in range(max_bounces):
            hit = intersect_scene(origin, direction)
            if hit is None:
                break

            # Store one photon record at each diffuse hit.
            pos_buf.append(hit.position.copy())
            incoming_buf.append((-direction).copy())
            power_buf.append(power.copy())
            normal_buf.append(hit.normal.copy())

            rr_prob = float(np.clip(hit.albedo.mean(), 0.10, 0.95))
            if rng.random() > rr_prob:
                break

            power = power * hit.albedo / rr_prob
            direction = sample_cosine_hemisphere(hit.normal, rng)
            origin = hit.position + hit.normal * 5e-4

    if not pos_buf:
        raise RuntimeError("No photons were stored; scene/light setup is invalid.")

    return PhotonMap(
        positions=np.asarray(pos_buf, dtype=float),
        incoming=np.asarray(incoming_buf, dtype=float),
        power=np.asarray(power_buf, dtype=float),
        normals=np.asarray(normal_buf, dtype=float),
    )


def estimate_irradiance_knn(
    points: Array,
    normals: Array,
    photon_map: PhotonMap,
    gather_k: int,
    max_radius: float,
) -> tuple[Array, Array, Array]:
    """Estimate irradiance with kNN photon gathering.

    Returns
    -------
    irradiance:
        (N, 3) RGB irradiance estimates.
    used_neighbors:
        (N,) number of neighbors effectively used after radius clipping.
    used_radius:
        (N,) gather radius for each query (k-th neighbor distance clipped).
    """
    pts = np.asarray(points, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be shape (N,3), got {pts.shape}")
    if nrm.shape != pts.shape:
        raise ValueError(f"normals must match points shape, got {nrm.shape} vs {pts.shape}")

    m = photon_map.positions.shape[0]
    if m == 0:
        raise ValueError("photon_map is empty")

    k = int(min(max(1, gather_k), m))
    if max_radius <= 0.0:
        raise ValueError("max_radius must be > 0")

    tree = cKDTree(photon_map.positions)
    dists, idx = tree.query(pts, k=k)

    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]

    n_points = pts.shape[0]
    irr = np.zeros((n_points, 3), dtype=float)
    used_neighbors = np.zeros(n_points, dtype=int)
    used_radius = np.zeros(n_points, dtype=float)

    for i in range(n_points):
        di = dists[i]
        ii = idx[i]
        valid = np.isfinite(di) & (di <= max_radius)
        if not np.any(valid):
            continue

        selected = ii[valid]
        local_dist = di[valid]
        radius = float(max(local_dist.max(), 1e-4))

        wi = photon_map.incoming[selected]  # toward surface
        cos_term = np.clip(np.einsum("ij,j->i", wi, nrm[i]), 0.0, None)

        flux = photon_map.power[selected] * cos_term[:, None]
        irr[i] = flux.sum(axis=0) / (math.pi * radius * radius)

        used_neighbors[i] = selected.shape[0]
        used_radius[i] = radius

    return irr, used_neighbors, used_radius


def build_floor_queries(nx: int, nz: int) -> tuple[Array, Array]:
    xs = np.linspace(-4.5, 4.5, nx)
    zs = np.linspace(0.2, 8.8, nz)
    xx, zz = np.meshgrid(xs, zs, indexing="xy")
    yy = np.full_like(xx, 2e-4)

    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=float), (points.shape[0], 1))
    return points, normals


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    rng = np.random.default_rng(2026)

    n_emit = 26_000
    max_bounces = 4
    gather_k = 60
    max_radius = 1.35

    light_pos = np.array([0.0, 5.8, 1.8], dtype=float)
    light_power_rgb = np.array([220.0, 220.0, 220.0], dtype=float)

    t0 = time.perf_counter()
    photon_map = emit_and_trace_photons(
        n_emit=n_emit,
        max_bounces=max_bounces,
        light_pos=light_pos,
        light_power_rgb=light_power_rgb,
        rng=rng,
    )
    trace_ms = (time.perf_counter() - t0) * 1000.0

    query_points, query_normals = build_floor_queries(nx=28, nz=22)

    t1 = time.perf_counter()
    irradiance, used_neighbors, used_radius = estimate_irradiance_knn(
        points=query_points,
        normals=query_normals,
        photon_map=photon_map,
        gather_k=gather_k,
        max_radius=max_radius,
    )
    gather_ms = (time.perf_counter() - t1) * 1000.0

    if photon_map.positions.shape[0] < n_emit * 0.20:
        raise AssertionError("stored photons too few; likely scene/light configuration issue")

    if np.any(~np.isfinite(irradiance)):
        raise AssertionError("irradiance contains non-finite values")

    if np.any(irradiance < -1e-12):
        raise AssertionError("irradiance should not be negative")

    if float(np.mean(used_neighbors > 0)) < 0.90:
        raise AssertionError("too many query points failed to find usable neighbors")

    # Lambertian outgoing radiance: Lo = albedo/pi * E
    radiance = (FLOOR_ALBEDO[None, :] / math.pi) * irradiance

    brightness = radiance.mean(axis=1)
    sample_n = 10
    sample_table = pd.DataFrame(
        {
            "x": query_points[:sample_n, 0],
            "z": query_points[:sample_n, 2],
            "neighbors": used_neighbors[:sample_n],
            "radius": used_radius[:sample_n],
            "E_r": irradiance[:sample_n, 0],
            "E_g": irradiance[:sample_n, 1],
            "E_b": irradiance[:sample_n, 2],
            "L_mean": brightness[:sample_n],
        }
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "photons_emitted",
                "photons_stored",
                "storage_ratio",
                "trace_ms",
                "gather_ms",
                "query_points",
                "nonzero_neighbor_ratio",
                "avg_neighbors_used",
                "avg_gather_radius",
                "mean_irradiance_rgb",
                "max_irradiance_rgb",
                "mean_radiance",
                "max_radiance",
            ],
            "value": [
                n_emit,
                int(photon_map.positions.shape[0]),
                float(photon_map.positions.shape[0] / n_emit),
                float(trace_ms),
                float(gather_ms),
                int(query_points.shape[0]),
                float(np.mean(used_neighbors > 0)),
                float(np.mean(used_neighbors)),
                float(np.mean(used_radius[used_radius > 0])) if np.any(used_radius > 0) else 0.0,
                np.round(irradiance.mean(axis=0), 6).tolist(),
                np.round(irradiance.max(axis=0), 6).tolist(),
                float(np.mean(brightness)),
                float(np.max(brightness)),
            ],
        }
    )

    print("=== Photon Mapping MVP (Diffuse Global Map) ===")
    print(f"Light position: {light_pos.tolist()}, light power RGB: {light_power_rgb.tolist()}")
    print(f"n_emit={n_emit}, max_bounces={max_bounces}, gather_k={gather_k}, max_radius={max_radius}")
    print()

    print("--- Sample query results (first 10 floor points) ---")
    print(sample_table.to_string(index=False))
    print()

    print("--- Summary metrics ---")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
