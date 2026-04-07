"""Minimal runnable MVP for the central force problem.

This script implements a small, source-transparent pipeline for a 2D inverse-square
central force field:
1) Verify the force is radial (parallel to -r).
2) Integrate a bound Kepler orbit using velocity Verlet.
3) Check conservation laws and effective-potential consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class OrbitElements:
    """Classical orbital elements used in this MVP (specific quantities)."""

    specific_energy: float
    specific_angular_momentum: float
    eccentricity: float
    semi_major_axis: float
    semi_latus_rectum: float
    period: float


def central_acceleration(position: np.ndarray, mu: float) -> np.ndarray:
    """Return acceleration a(r) = -mu * r / |r|^3 in 2D."""
    radius = float(np.linalg.norm(position))
    if radius <= 0.0:
        raise ValueError("Position radius must be positive for inverse-square central force.")
    return -mu * position / (radius**3)


def specific_energy(position: np.ndarray, velocity: np.ndarray, mu: float) -> float:
    """Return specific mechanical energy eps = v^2/2 - mu/r."""
    radius = float(np.linalg.norm(position))
    speed_sq = float(np.dot(velocity, velocity))
    return 0.5 * speed_sq - mu / radius


def specific_angular_momentum(position: np.ndarray, velocity: np.ndarray) -> float:
    """Return scalar z-component of specific angular momentum h = (r x v)_z."""
    return float(position[0] * velocity[1] - position[1] * velocity[0])


def estimate_orbit_elements(position: np.ndarray, velocity: np.ndarray, mu: float) -> OrbitElements:
    """Estimate orbit elements from one state vector in an inverse-square field."""
    radius = float(np.linalg.norm(position))
    speed_sq = float(np.dot(velocity, velocity))
    h = specific_angular_momentum(position, velocity)
    eps = 0.5 * speed_sq - mu / radius

    e_vec = ((speed_sq - mu / radius) * position - np.dot(position, velocity) * velocity) / mu
    eccentricity = float(np.linalg.norm(e_vec))

    semi_latus_rectum = h * h / mu
    semi_major_axis = np.inf if abs(eps) < 1e-14 else -mu / (2.0 * eps)

    period = np.nan
    if eccentricity < 1.0 and np.isfinite(semi_major_axis):
        period = 2.0 * np.pi * np.sqrt(semi_major_axis**3 / mu)

    return OrbitElements(
        specific_energy=eps,
        specific_angular_momentum=h,
        eccentricity=eccentricity,
        semi_major_axis=float(semi_major_axis),
        semi_latus_rectum=float(semi_latus_rectum),
        period=float(period),
    )


def velocity_verlet_integrate(
    mu: float,
    position0: np.ndarray,
    velocity0: np.ndarray,
    dt: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate central-force dynamics using velocity Verlet.

    Returns:
        times: shape (steps + 1,)
        positions: shape (steps + 1, 2)
        velocities: shape (steps + 1, 2)
    """
    positions = np.zeros((steps + 1, 2), dtype=np.float64)
    velocities = np.zeros((steps + 1, 2), dtype=np.float64)
    times = np.linspace(0.0, dt * steps, steps + 1, dtype=np.float64)

    r = np.asarray(position0, dtype=np.float64).copy()
    v = np.asarray(velocity0, dtype=np.float64).copy()
    a = central_acceleration(r, mu)

    positions[0] = r
    velocities[0] = v

    for i in range(steps):
        r_next = r + v * dt + 0.5 * a * (dt**2)
        a_next = central_acceleration(r_next, mu)
        v_next = v + 0.5 * (a + a_next) * dt

        positions[i + 1] = r_next
        velocities[i + 1] = v_next

        r, v, a = r_next, v_next, a_next

    return times, positions, velocities


def find_turning_points(radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices of local minima (periapsis) and maxima (apoapsis)."""
    dr = np.diff(radii)
    peri_idx = np.where((dr[:-1] < 0.0) & (dr[1:] >= 0.0))[0] + 1
    apo_idx = np.where((dr[:-1] > 0.0) & (dr[1:] <= 0.0))[0] + 1
    return peri_idx, apo_idx


def check_central_direction(mu: float) -> Dict[str, float]:
    """Check that acceleration is radial and inward for sample points."""
    points = np.array(
        [
            [1.2, 0.3],
            [0.5, -1.5],
            [-2.0, 1.0],
            [1.8, 2.2],
            [-1.1, -0.7],
        ],
        dtype=np.float64,
    )

    cosines = []
    for point in points:
        acc = central_acceleration(point, mu)
        inward = -point
        cosine = float(np.dot(acc, inward) / (np.linalg.norm(acc) * np.linalg.norm(inward)))
        cosines.append(cosine)

    cosines_arr = np.array(cosines, dtype=np.float64)
    return {
        "min_cosine": float(np.min(cosines_arr)),
        "max_cosine": float(np.max(cosines_arr)),
        "mean_cosine": float(np.mean(cosines_arr)),
    }


def run_orbit_demo() -> Dict[str, float]:
    """Run a bound-orbit demo and return scalar metrics for validation."""
    mu = 1.0
    a_true = 1.5
    e_true = 0.4

    r_peri_true = a_true * (1.0 - e_true)
    r_apo_true = a_true * (1.0 + e_true)

    position0 = np.array([r_peri_true, 0.0], dtype=np.float64)
    speed_peri = np.sqrt(mu * (2.0 / r_peri_true - 1.0 / a_true))
    velocity0 = np.array([0.0, speed_peri], dtype=np.float64)

    elements0 = estimate_orbit_elements(position0, velocity0, mu)

    dt = 2.0e-3
    periods = 4.0
    steps = int(np.ceil(periods * elements0.period / dt))

    times, positions, velocities = velocity_verlet_integrate(
        mu=mu,
        position0=position0,
        velocity0=velocity0,
        dt=dt,
        steps=steps,
    )

    radii = np.linalg.norm(positions, axis=1)
    speeds_sq = np.sum(velocities * velocities, axis=1)
    energies = 0.5 * speeds_sq - mu / radii
    h_series = positions[:, 0] * velocities[:, 1] - positions[:, 1] * velocities[:, 0]

    energy0 = float(energies[0])
    h0 = float(h_series[0])

    energy_rel_drift = np.max(np.abs((energies - energy0) / (abs(energy0) + 1e-14)))
    h_rel_drift = np.max(np.abs((h_series - h0) / (abs(h0) + 1e-14)))

    radial_speed = np.sum(positions * velocities, axis=1) / radii
    u_eff = (h0 * h0) / (2.0 * radii * radii) - mu / radii
    radial_energy = 0.5 * radial_speed * radial_speed + u_eff
    radial_identity_residual = np.max(np.abs(radial_energy - energies))

    peri_idx, apo_idx = find_turning_points(radii)
    if len(peri_idx) < 2 or len(apo_idx) < 2:
        raise RuntimeError("Not enough turning points detected. Increase integration length.")

    r_peri_num = float(np.median(radii[peri_idx]))
    r_apo_num = float(np.median(radii[apo_idx]))

    return {
        "mu": mu,
        "a_true": a_true,
        "e_true": e_true,
        "a_est": elements0.semi_major_axis,
        "e_est": elements0.eccentricity,
        "period": elements0.period,
        "time_final": float(times[-1]),
        "energy_rel_drift_max": float(energy_rel_drift),
        "ang_mom_rel_drift_max": float(h_rel_drift),
        "radial_identity_residual_max": float(radial_identity_residual),
        "r_peri_true": r_peri_true,
        "r_apo_true": r_apo_true,
        "r_peri_num": r_peri_num,
        "r_apo_num": r_apo_num,
        "r_peri_abs_err": abs(r_peri_num - r_peri_true),
        "r_apo_abs_err": abs(r_apo_num - r_apo_true),
    }


def main() -> None:
    print("=== Demo A: Central-force direction check ===")
    direction_report = check_central_direction(mu=1.0)
    print(
        "cos(angle(a, -r)) -> min={min_cosine:.12f}, mean={mean_cosine:.12f}, max={max_cosine:.12f}".format(
            **direction_report
        )
    )

    print("\n=== Demo B: Bound-orbit integration under inverse-square central force ===")
    report = run_orbit_demo()

    print("theory: a={:.6f}, e={:.6f}".format(report["a_true"], report["e_true"]))
    print("state->elements: a_est={:.6f}, e_est={:.6f}".format(report["a_est"], report["e_est"]))
    print("period={:.6f}, simulated_time={:.6f}".format(report["period"], report["time_final"]))
    print(
        "drift: energy={:.3e}, angular_momentum={:.3e}, radial_identity={:.3e}".format(
            report["energy_rel_drift_max"],
            report["ang_mom_rel_drift_max"],
            report["radial_identity_residual_max"],
        )
    )
    print(
        "turning points: r_peri true={:.6f}, num={:.6f}, abs_err={:.3e}".format(
            report["r_peri_true"], report["r_peri_num"], report["r_peri_abs_err"]
        )
    )
    print(
        "turning points: r_apo  true={:.6f}, num={:.6f}, abs_err={:.3e}".format(
            report["r_apo_true"], report["r_apo_num"], report["r_apo_abs_err"]
        )
    )

    assert direction_report["min_cosine"] > 1.0 - 1e-12
    assert abs(report["a_est"] - report["a_true"]) < 1e-12
    assert abs(report["e_est"] - report["e_true"]) < 1e-12
    assert report["energy_rel_drift_max"] < 2.5e-4
    assert report["ang_mom_rel_drift_max"] < 2.5e-4
    assert report["radial_identity_residual_max"] < 2.0e-4
    assert report["r_peri_abs_err"] < 1.5e-2
    assert report["r_apo_abs_err"] < 1.5e-2

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
