"""Noether theorem MVP: symmetry probes and conserved quantities.

This script demonstrates the symmetry-conservation correspondence in
small classical-mechanics systems:
- spatial translation -> linear momentum conservation (free particle),
- time translation -> energy conservation (autonomous oscillators),
- rotation -> angular momentum conservation (isotropic oscillator),
- broken rotation symmetry -> non-conservation of angular momentum
  (anisotropic oscillator).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NoetherConfig:
    """Configuration for the Noether-theorem numerical MVP."""

    dt: float = 0.01
    n_steps: int = 4000
    mass: float = 1.0
    k_iso: float = 1.0
    kx_aniso: float = 1.0
    ky_aniso: float = 1.8
    fd_eps: float = 1e-6
    probe_stride: int = 40


def lagrangian_free_1d(x: float, v: float, mass: float) -> float:
    """Lagrangian of a 1D free particle: L = 1/2 m v^2."""

    _ = x  # x does not enter L, which reflects translation symmetry.
    return 0.5 * mass * v * v


def lagrangian_oscillator_2d(q: np.ndarray, v: np.ndarray, mass: float, kx: float, ky: float) -> float:
    """Lagrangian of a 2D (possibly anisotropic) harmonic oscillator."""

    kinetic = 0.5 * mass * float(np.dot(v, v))
    potential = 0.5 * (kx * q[0] * q[0] + ky * q[1] * q[1])
    return kinetic - potential


def rotate_2d(vec: np.ndarray, theta: float) -> np.ndarray:
    """Rotate a 2D vector by angle theta."""

    c = float(np.cos(theta))
    s = float(np.sin(theta))
    x, y = float(vec[0]), float(vec[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=float)


def acceleration_oscillator_2d(q: np.ndarray, mass: float, kx: float, ky: float) -> np.ndarray:
    """Acceleration from V = 1/2(kx x^2 + ky y^2)."""

    return np.array([-kx * q[0] / mass, -ky * q[1] / mass], dtype=float)


def simulate_free_particle(x0: float, v0: float, dt: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Simulate 1D free-particle trajectory with constant velocity."""

    x = np.empty(n_steps + 1, dtype=float)
    v = np.full(n_steps + 1, v0, dtype=float)
    x[0] = x0

    for i in range(n_steps):
        x[i + 1] = x[i] + v[i] * dt

    return x, v


def simulate_oscillator_verlet(
    q0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    n_steps: int,
    mass: float,
    kx: float,
    ky: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity-Verlet integration for 2D oscillator."""

    q = np.empty((n_steps + 1, 2), dtype=float)
    v = np.empty((n_steps + 1, 2), dtype=float)
    q[0] = q0
    v[0] = v0

    a = acceleration_oscillator_2d(q0, mass=mass, kx=kx, ky=ky)
    for i in range(n_steps):
        q_next = q[i] + v[i] * dt + 0.5 * a * dt * dt
        a_next = acceleration_oscillator_2d(q_next, mass=mass, kx=kx, ky=ky)
        v_next = v[i] + 0.5 * (a + a_next) * dt

        q[i + 1] = q_next
        v[i + 1] = v_next
        a = a_next

    return q, v


def translation_symmetry_residual(x: float, v: float, mass: float, eps: float) -> float:
    """Finite-difference residual for x-translation symmetry in free particle."""

    l0 = lagrangian_free_1d(x=x, v=v, mass=mass)
    l1 = lagrangian_free_1d(x=x + eps, v=v, mass=mass)
    return abs((l1 - l0) / eps)


def rotation_symmetry_residual(q: np.ndarray, v: np.ndarray, mass: float, kx: float, ky: float, eps: float) -> float:
    """Finite-difference residual for infinitesimal planar rotation symmetry."""

    q_rot = rotate_2d(q, eps)
    v_rot = rotate_2d(v, eps)
    l0 = lagrangian_oscillator_2d(q=q, v=v, mass=mass, kx=kx, ky=ky)
    l1 = lagrangian_oscillator_2d(q=q_rot, v=v_rot, mass=mass, kx=kx, ky=ky)
    return abs((l1 - l0) / eps)


def momentum_1d(v: np.ndarray, mass: float) -> np.ndarray:
    """Noether charge for translation symmetry in 1D."""

    return mass * v


def energy_2d(q: np.ndarray, v: np.ndarray, mass: float, kx: float, ky: float) -> np.ndarray:
    """Total energy of 2D oscillator."""

    kinetic = 0.5 * mass * np.sum(v * v, axis=1)
    potential = 0.5 * (kx * q[:, 0] ** 2 + ky * q[:, 1] ** 2)
    return kinetic + potential


def angular_momentum_z_2d(q: np.ndarray, v: np.ndarray, mass: float) -> np.ndarray:
    """Noether charge for planar rotation symmetry: Lz = m(x vy - y vx)."""

    return mass * (q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0])


def drift_metrics(name: str, series: np.ndarray, scenario: str) -> dict[str, float | str]:
    """Summarize conservation drift over a time series."""

    s = np.asarray(series, dtype=float)
    drift = s - s[0]
    max_abs = float(np.max(np.abs(drift)))
    denom = max(1e-12, float(np.max(np.abs(s))))
    return {
        "scenario": scenario,
        "quantity": name,
        "initial": float(s[0]),
        "final": float(s[-1]),
        "max_abs_drift": max_abs,
        "relative_max_drift": max_abs / denom,
    }


def symmetry_probe_table(
    cfg: NoetherConfig,
    x_free: np.ndarray,
    v_free: np.ndarray,
    q_iso: np.ndarray,
    v_iso: np.ndarray,
    q_aniso: np.ndarray,
    v_aniso: np.ndarray,
) -> pd.DataFrame:
    """Evaluate symmetry residuals on sampled trajectory points."""

    indices = np.arange(0, cfg.n_steps + 1, cfg.probe_stride, dtype=int)

    translation_res = np.array(
        [translation_symmetry_residual(x_free[i], v_free[i], cfg.mass, cfg.fd_eps) for i in indices],
        dtype=float,
    )
    rotation_iso_res = np.array(
        [
            rotation_symmetry_residual(q_iso[i], v_iso[i], cfg.mass, cfg.k_iso, cfg.k_iso, cfg.fd_eps)
            for i in indices
        ],
        dtype=float,
    )
    rotation_aniso_res = np.array(
        [
            rotation_symmetry_residual(q_aniso[i], v_aniso[i], cfg.mass, cfg.kx_aniso, cfg.ky_aniso, cfg.fd_eps)
            for i in indices
        ],
        dtype=float,
    )

    rows = [
        {
            "scenario": "free_particle",
            "symmetry": "translation_x",
            "residual_mean": float(np.mean(translation_res)),
            "residual_max": float(np.max(translation_res)),
        },
        {
            "scenario": "isotropic_oscillator",
            "symmetry": "rotation_xy",
            "residual_mean": float(np.mean(rotation_iso_res)),
            "residual_max": float(np.max(rotation_iso_res)),
        },
        {
            "scenario": "anisotropic_oscillator",
            "symmetry": "rotation_xy",
            "residual_mean": float(np.mean(rotation_aniso_res)),
            "residual_max": float(np.max(rotation_aniso_res)),
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    cfg = NoetherConfig()

    x0_free, v0_free = 0.35, 1.2
    q0 = np.array([1.0, 0.2], dtype=float)
    v0 = np.array([0.0, 1.1], dtype=float)

    x_free, v_free = simulate_free_particle(x0=x0_free, v0=v0_free, dt=cfg.dt, n_steps=cfg.n_steps)
    q_iso, v_iso = simulate_oscillator_verlet(
        q0=q0,
        v0=v0,
        dt=cfg.dt,
        n_steps=cfg.n_steps,
        mass=cfg.mass,
        kx=cfg.k_iso,
        ky=cfg.k_iso,
    )
    q_aniso, v_aniso = simulate_oscillator_verlet(
        q0=q0,
        v0=v0,
        dt=cfg.dt,
        n_steps=cfg.n_steps,
        mass=cfg.mass,
        kx=cfg.kx_aniso,
        ky=cfg.ky_aniso,
    )

    px_free = momentum_1d(v_free, mass=cfg.mass)
    e_iso = energy_2d(q_iso, v_iso, mass=cfg.mass, kx=cfg.k_iso, ky=cfg.k_iso)
    lz_iso = angular_momentum_z_2d(q_iso, v_iso, mass=cfg.mass)
    e_aniso = energy_2d(q_aniso, v_aniso, mass=cfg.mass, kx=cfg.kx_aniso, ky=cfg.ky_aniso)
    lz_aniso = angular_momentum_z_2d(q_aniso, v_aniso, mass=cfg.mass)

    symmetry_df = symmetry_probe_table(
        cfg=cfg,
        x_free=x_free,
        v_free=v_free,
        q_iso=q_iso,
        v_iso=v_iso,
        q_aniso=q_aniso,
        v_aniso=v_aniso,
    )

    conservation_rows = [
        drift_metrics("p_x", px_free, "free_particle"),
        drift_metrics("E", e_iso, "isotropic_oscillator"),
        drift_metrics("L_z", lz_iso, "isotropic_oscillator"),
        drift_metrics("E", e_aniso, "anisotropic_oscillator"),
        drift_metrics("L_z", lz_aniso, "anisotropic_oscillator"),
    ]
    conservation_df = pd.DataFrame(conservation_rows)

    print("Noether Theorem MVP: Symmetry -> Conserved Quantity")
    print(
        f"dt={cfg.dt}, n_steps={cfg.n_steps}, mass={cfg.mass}, "
        f"k_iso={cfg.k_iso}, k_aniso=({cfg.kx_aniso}, {cfg.ky_aniso}), fd_eps={cfg.fd_eps}"
    )

    print("\n=== symmetry_probe ===")
    print(symmetry_df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))

    print("\n=== conservation_report ===")
    print(conservation_df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))

    sym_free = symmetry_df.loc[
        (symmetry_df["scenario"] == "free_particle") & (symmetry_df["symmetry"] == "translation_x"),
        "residual_mean",
    ].iloc[0]
    sym_iso = symmetry_df.loc[
        (symmetry_df["scenario"] == "isotropic_oscillator") & (symmetry_df["symmetry"] == "rotation_xy"),
        "residual_mean",
    ].iloc[0]
    sym_aniso = symmetry_df.loc[
        (symmetry_df["scenario"] == "anisotropic_oscillator") & (symmetry_df["symmetry"] == "rotation_xy"),
        "residual_mean",
    ].iloc[0]

    drift_px = conservation_df.loc[
        (conservation_df["scenario"] == "free_particle") & (conservation_df["quantity"] == "p_x"),
        "max_abs_drift",
    ].iloc[0]
    drift_e_iso = conservation_df.loc[
        (conservation_df["scenario"] == "isotropic_oscillator") & (conservation_df["quantity"] == "E"),
        "max_abs_drift",
    ].iloc[0]
    drift_lz_iso = conservation_df.loc[
        (conservation_df["scenario"] == "isotropic_oscillator") & (conservation_df["quantity"] == "L_z"),
        "max_abs_drift",
    ].iloc[0]
    drift_e_aniso = conservation_df.loc[
        (conservation_df["scenario"] == "anisotropic_oscillator") & (conservation_df["quantity"] == "E"),
        "max_abs_drift",
    ].iloc[0]
    drift_lz_aniso = conservation_df.loc[
        (conservation_df["scenario"] == "anisotropic_oscillator") & (conservation_df["quantity"] == "L_z"),
        "max_abs_drift",
    ].iloc[0]

    # Symmetry checks
    assert sym_free < 1e-12, f"Translation residual too large for free particle: {sym_free:.3e}"
    assert sym_iso < 1e-5, f"Rotation residual too large for isotropic oscillator: {sym_iso:.3e}"
    assert sym_aniso > 1e-2, f"Rotation residual too small for anisotropic oscillator: {sym_aniso:.3e}"

    # Conservation checks from Noether mapping
    assert drift_px < 1e-12, f"Free-particle momentum drift too large: {drift_px:.3e}"
    assert drift_e_iso < 5e-4, f"Isotropic energy drift too large: {drift_e_iso:.3e}"
    assert drift_lz_iso < 5e-4, f"Isotropic angular-momentum drift too large: {drift_lz_iso:.3e}"
    assert drift_e_aniso < 5e-4, f"Anisotropic energy drift too large: {drift_e_aniso:.3e}"
    assert drift_lz_aniso > 1e-2, f"Anisotropic Lz drift should be visible but got {drift_lz_aniso:.3e}"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
