"""MVP: geodesic computation on the unit sphere using metric-derived ODEs.

This script implements geodesic integration from first principles:
- metric tensor g_ij(q)
- numerical partial derivatives of metric
- Christoffel symbols Gamma^k_{ij}
- second-order geodesic ODE converted to first-order system
- fixed-step RK4 integrator (no SciPy dependency)

No interactive input is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass
class GeodesicResult:
    name: str
    t: Array
    q: Array
    v: Array
    energy: Array
    energy_rel_drift: float
    plane_residual_max: float
    plane_residual_mean: float
    extra_metric: float | None


def check_vector(name: str, x: Array, dim: int) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def sphere_metric(q: Array) -> Array:
    """Metric tensor on unit sphere in coordinates q=(theta, phi)."""
    theta = float(q[0])
    s = math.sin(theta)
    if abs(s) < 1e-8:
        raise ValueError("theta is too close to coordinate singularity at poles")
    return np.array([[1.0, 0.0], [0.0, s * s]], dtype=float)


def metric_partials(metric_fn, q: Array, h: float = 1e-6) -> Array:
    """Return dg[a, i, j] = partial_{q_a} g_{ij}(q) by central difference."""
    q = np.asarray(q, dtype=float)
    n = q.size
    dg = np.zeros((n, n, n), dtype=float)
    for a in range(n):
        dq = np.zeros(n, dtype=float)
        dq[a] = h
        g_plus = metric_fn(q + dq)
        g_minus = metric_fn(q - dq)
        dg[a] = (g_plus - g_minus) / (2.0 * h)
    return dg


def christoffel_symbols(metric_fn, q: Array, h: float = 1e-6) -> Array:
    """Return Gamma[k, i, j] = Gamma^k_{ij}."""
    g = metric_fn(q)
    g_inv = np.linalg.inv(g)
    dg = metric_partials(metric_fn, q, h=h)

    n = q.size
    gamma = np.zeros((n, n, n), dtype=float)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                accum = 0.0
                for ell in range(n):
                    term = dg[i, j, ell] + dg[j, i, ell] - dg[ell, i, j]
                    accum += g_inv[k, ell] * term
                gamma[k, i, j] = 0.5 * accum
    return gamma


def geodesic_rhs(y: Array, metric_fn, h: float = 1e-6) -> Array:
    """First-order geodesic system.

    y = [q0, q1, v0, v1], with q' = v,
    v'^k = -Gamma^k_{ij}(q) * v^i * v^j.
    """
    n = y.size // 2
    q = y[:n]
    v = y[n:]

    gamma = christoffel_symbols(metric_fn, q, h=h)

    dv = np.zeros(n, dtype=float)
    for k in range(n):
        acc = 0.0
        for i in range(n):
            for j in range(n):
                acc += gamma[k, i, j] * v[i] * v[j]
        dv[k] = -acc

    return np.concatenate([v, dv])


def rk4_integrate(rhs_fn, y0: Array, t_eval: Array) -> Array:
    """Fixed-step RK4 integration on user-provided t_eval grid."""
    y0 = np.asarray(y0, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1 or t_eval.size < 2:
        raise ValueError("t_eval must be 1D with at least 2 points")
    if np.any(np.diff(t_eval) <= 0):
        raise ValueError("t_eval must be strictly increasing")

    y = np.zeros((t_eval.size, y0.size), dtype=float)
    y[0] = y0

    for i in range(t_eval.size - 1):
        h_step = float(t_eval[i + 1] - t_eval[i])
        yi = y[i]

        k1 = rhs_fn(yi)
        k2 = rhs_fn(yi + 0.5 * h_step * k1)
        k3 = rhs_fn(yi + 0.5 * h_step * k2)
        k4 = rhs_fn(yi + h_step * k3)

        y[i + 1] = yi + (h_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not np.all(np.isfinite(y[i + 1])):
            raise RuntimeError("Integration diverged to non-finite state")

    return y


def integrate_geodesic(
    metric_fn,
    q0: Array,
    v0: Array,
    *,
    t_end: float,
    num_points: int = 1600,
    diff_step: float = 1e-6,
) -> tuple[Array, Array, Array]:
    q0 = check_vector("q0", q0, dim=2)
    v0 = check_vector("v0", v0, dim=2)
    if t_end <= 0:
        raise ValueError("t_end must be positive")
    if num_points < 3:
        raise ValueError("num_points must be >= 3")

    y0 = np.concatenate([q0, v0])
    t_eval = np.linspace(0.0, t_end, num_points)

    y = rk4_integrate(
        rhs_fn=lambda state: geodesic_rhs(state, metric_fn=metric_fn, h=diff_step),
        y0=y0,
        t_eval=t_eval,
    )

    q = y[:, :2]
    v = y[:, 2:]
    return t_eval, q, v


def kinetic_energy(metric_fn, q: Array, v: Array) -> float:
    g = metric_fn(q)
    return 0.5 * float(v.T @ g @ v)


def spherical_to_cartesian(q: Array) -> Array:
    theta, phi = float(q[0]), float(q[1])
    st = math.sin(theta)
    ct = math.cos(theta)
    cp = math.cos(phi)
    sp = math.sin(phi)
    return np.array([st * cp, st * sp, ct], dtype=float)


def spherical_velocity_to_cartesian(q: Array, v: Array) -> Array:
    theta, phi = float(q[0]), float(q[1])
    theta_dot, phi_dot = float(v[0]), float(v[1])

    st = math.sin(theta)
    ct = math.cos(theta)
    cp = math.cos(phi)
    sp = math.sin(phi)

    dx = ct * cp * theta_dot - st * sp * phi_dot
    dy = ct * sp * theta_dot + st * cp * phi_dot
    dz = -st * theta_dot
    return np.array([dx, dy, dz], dtype=float)


def great_circle_plane_residual(q_path: Array, v_path: Array) -> tuple[float, float]:
    p0 = spherical_to_cartesian(q_path[0])
    dp0 = spherical_velocity_to_cartesian(q_path[0], v_path[0])

    normal = np.cross(p0, dp0)
    norm_n = float(np.linalg.norm(normal))
    if norm_n < 1e-12:
        raise RuntimeError("initial velocity is degenerate for plane residual check")
    normal = normal / norm_n
    if not np.all(np.isfinite(normal)):
        raise RuntimeError("plane normal contains non-finite values")

    points = np.array([spherical_to_cartesian(q) for q in q_path], dtype=float)
    if not np.all(np.isfinite(points)):
        raise RuntimeError("trajectory contains non-finite Cartesian points")
    residuals = np.abs(np.sum(points * normal[None, :], axis=1))
    return float(np.max(residuals)), float(np.mean(residuals))


def run_case(name: str, q0: Array, v0: Array, *, t_end: float, is_equator: bool) -> GeodesicResult:
    t, q_path, v_path = integrate_geodesic(sphere_metric, q0=q0, v0=v0, t_end=t_end)

    energy = np.array([kinetic_energy(sphere_metric, q, v) for q, v in zip(q_path, v_path)], dtype=float)
    e0 = float(energy[0])
    energy_rel_drift = float(np.max(np.abs(energy - e0)) / max(1.0, abs(e0)))

    plane_max, plane_mean = great_circle_plane_residual(q_path, v_path)

    extra_metric = None
    if is_equator:
        extra_metric = float(np.max(np.abs(q_path[:, 0] - (math.pi / 2.0))))

    return GeodesicResult(
        name=name,
        t=t,
        q=q_path,
        v=v_path,
        energy=energy,
        energy_rel_drift=energy_rel_drift,
        plane_residual_max=plane_max,
        plane_residual_mean=plane_mean,
        extra_metric=extra_metric,
    )


def print_case(result: GeodesicResult) -> None:
    print(f"\n=== {result.name} ===")
    print(
        f"energy drift={result.energy_rel_drift:.3e}, "
        f"plane residual (max/mean)=({result.plane_residual_max:.3e}/{result.plane_residual_mean:.3e})"
    )
    if result.extra_metric is not None:
        print(f"equator theta deviation max={result.extra_metric:.3e}")

    sample_ids = np.linspace(0, result.t.size - 1, 6, dtype=int)
    print("samples: t, theta, phi, theta_dot, phi_dot, energy")
    for idx in sample_ids:
        theta, phi = result.q[idx]
        theta_dot, phi_dot = result.v[idx]
        print(
            f"  {result.t[idx]:7.3f}  {theta: .6f}  {phi: .6f}  "
            f"{theta_dot: .6f}  {phi_dot: .6f}  {result.energy[idx]: .6f}"
        )


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    cases = [
        {
            "name": "equator geodesic",
            "q0": np.array([math.pi / 2.0, 0.0], dtype=float),
            "v0": np.array([0.0, 1.2], dtype=float),
            "t_end": 8.0,
            "is_equator": True,
        },
        {
            "name": "oblique great-circle geodesic",
            "q0": np.array([1.1, 0.3], dtype=float),
            "v0": np.array([0.22, 0.75], dtype=float),
            "t_end": 10.0,
            "is_equator": False,
        },
    ]

    results: list[GeodesicResult] = []
    for cfg in cases:
        result = run_case(
            cfg["name"],
            cfg["q0"],
            cfg["v0"],
            t_end=cfg["t_end"],
            is_equator=cfg["is_equator"],
        )
        print_case(result)
        results.append(result)

    max_energy_drift = max(r.energy_rel_drift for r in results)
    max_plane_residual = max(r.plane_residual_max for r in results)

    print("\n=== summary ===")
    print(f"max energy relative drift: {max_energy_drift:.3e}")
    print(f"max great-circle plane residual: {max_plane_residual:.3e}")

    equator_dev = [r.extra_metric for r in results if r.extra_metric is not None]
    if equator_dev:
        print(f"max equator theta deviation: {max(equator_dev):.3e}")

    # Lightweight, deterministic correctness checks for this MVP.
    if max_energy_drift > 8e-5:
        raise RuntimeError("Energy drift is too large; integration or equations may be incorrect.")
    if max_plane_residual > 3e-4:
        raise RuntimeError("Trajectory is not close to a great circle plane.")
    if equator_dev and max(equator_dev) > 8e-5:
        raise RuntimeError("Equator case deviates too much from theta=pi/2.")

    print("All checks passed.")


if __name__ == "__main__":
    main()
