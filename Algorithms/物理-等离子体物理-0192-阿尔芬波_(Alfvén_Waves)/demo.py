"""Linear 1D Alfvén-wave MVP using Elsasser variables."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

MU0 = 4.0e-7 * math.pi


@dataclass(frozen=True)
class AlfvenConfig:
    """Configuration for the linear Alfvén-wave experiment."""

    background_b0_t: float = 6.0e-4
    mass_density_kg_m3: float = 1.5e-9
    domain_length_m: float = 2.0e5
    n_grid: int = 600
    cfl: float = 0.45
    wave_amplitude_m_s: float = 2.0e3
    n_periods: float = 0.75


def alfven_speed(cfg: AlfvenConfig) -> float:
    return cfg.background_b0_t / math.sqrt(MU0 * cfg.mass_density_kg_m3)


def build_initial_state(cfg: AlfvenConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, cfg.domain_length_m, cfg.n_grid, endpoint=False)
    phase = 2.0 * math.pi * x / cfg.domain_length_m

    # Right-propagating mode: z_minus != 0, z_plus = 0.
    z_plus = np.zeros_like(x)
    z_minus = cfg.wave_amplitude_m_s * np.sin(phase)
    return x, z_plus, z_minus


def lax_wendroff_periodic(u: np.ndarray, c: float, dt: float, dx: float) -> np.ndarray:
    """Second-order Lax-Wendroff update for periodic advection."""
    sigma = c * dt / dx
    u_r = np.roll(u, -1)
    u_l = np.roll(u, 1)
    return u - 0.5 * sigma * (u_r - u_l) + 0.5 * sigma * sigma * (u_r - 2.0 * u + u_l)


def exact_solution(
    x: np.ndarray,
    t: float,
    cfg: AlfvenConfig,
    v_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    phase = 2.0 * math.pi * (x - v_a * t) / cfg.domain_length_m
    z_plus_exact = np.zeros_like(x)
    z_minus_exact = cfg.wave_amplitude_m_s * np.sin(phase)
    return z_plus_exact, z_minus_exact


def to_velocity_and_magnetic_perturbation(
    z_plus: np.ndarray,
    z_minus: np.ndarray,
    cfg: AlfvenConfig,
) -> tuple[np.ndarray, np.ndarray]:
    v_y = 0.5 * (z_plus + z_minus)
    b_bar = 0.5 * (z_plus - z_minus)
    b_y = b_bar * math.sqrt(MU0 * cfg.mass_density_kg_m3)
    return v_y, b_y


def total_perturbation_energy(v_y: np.ndarray, b_y: np.ndarray, cfg: AlfvenConfig, dx: float) -> float:
    kinetic_density = 0.5 * cfg.mass_density_kg_m3 * v_y * v_y
    magnetic_density = 0.5 * b_y * b_y / MU0
    return float(np.sum(kinetic_density + magnetic_density) * dx)


def simulate(cfg: AlfvenConfig) -> dict[str, float | int | np.ndarray]:
    if cfg.n_grid < 16:
        raise ValueError("n_grid must be >= 16.")
    if cfg.cfl <= 0.0 or cfg.cfl > 0.95:
        raise ValueError("cfl must be in (0, 0.95].")
    if cfg.background_b0_t <= 0.0 or cfg.mass_density_kg_m3 <= 0.0:
        raise ValueError("background_b0_t and mass_density_kg_m3 must be positive.")
    if cfg.wave_amplitude_m_s <= 0.0:
        raise ValueError("wave_amplitude_m_s must be positive.")

    x, z_plus, z_minus = build_initial_state(cfg)
    dx = cfg.domain_length_m / cfg.n_grid
    v_a = alfven_speed(cfg)
    t_end = cfg.n_periods * cfg.domain_length_m / v_a
    dt_nominal = cfg.cfl * dx / v_a

    c_plus = -v_a
    c_minus = v_a

    v_y0, b_y0 = to_velocity_and_magnetic_perturbation(z_plus, z_minus, cfg)
    e0 = total_perturbation_energy(v_y0, b_y0, cfg, dx)

    t = 0.0
    steps = 0
    while t < t_end - 1e-15:
        dt = min(dt_nominal, t_end - t)
        z_plus = lax_wendroff_periodic(z_plus, c_plus, dt, dx)
        z_minus = lax_wendroff_periodic(z_minus, c_minus, dt, dx)
        t += dt
        steps += 1

    z_plus_ex, z_minus_ex = exact_solution(x, t_end, cfg, v_a)
    v_num, b_num = to_velocity_and_magnetic_perturbation(z_plus, z_minus, cfg)
    v_ex, b_ex = to_velocity_and_magnetic_perturbation(z_plus_ex, z_minus_ex, cfg)
    e1 = total_perturbation_energy(v_num, b_num, cfg, dx)

    eps = 1e-12
    v_l2_rel = float(np.linalg.norm(v_num - v_ex) / (np.linalg.norm(v_ex) + eps))
    b_l2_rel = float(np.linalg.norm(b_num - b_ex) / (np.linalg.norm(b_ex) + eps))
    energy_rel_drift = abs(e1 - e0) / (abs(e0) + eps)
    max_velocity = float(np.max(np.abs(v_num)))
    mach_like_ratio = max_velocity / v_a

    return {
        "x": x,
        "v_num": v_num,
        "v_ex": v_ex,
        "b_num": b_num,
        "b_ex": b_ex,
        "v_a": v_a,
        "dx": dx,
        "t_end": t_end,
        "steps": steps,
        "v_l2_rel": v_l2_rel,
        "b_l2_rel": b_l2_rel,
        "energy_rel_drift": energy_rel_drift,
        "mach_like_ratio": mach_like_ratio,
    }


def main() -> None:
    cfg = AlfvenConfig()
    out = simulate(cfg)

    x = out["x"]
    v_num = out["v_num"]
    v_ex = out["v_ex"]
    b_num = out["b_num"]
    b_ex = out["b_ex"]

    sample_idx = np.linspace(0, cfg.n_grid - 1, 8, dtype=int)
    sample_df = pd.DataFrame(
        {
            "x_over_L": x[sample_idx] / cfg.domain_length_m,
            "v_num_m_s": v_num[sample_idx],
            "v_exact_m_s": v_ex[sample_idx],
            "b_num_T": b_num[sample_idx],
            "b_exact_T": b_ex[sample_idx],
        }
    )
    sample_df["v_rel_err"] = np.abs(sample_df["v_num_m_s"] - sample_df["v_exact_m_s"]) / (
        np.abs(sample_df["v_exact_m_s"]) + 1e-12
    )
    sample_df["b_rel_err"] = np.abs(sample_df["b_num_T"] - sample_df["b_exact_T"]) / (
        np.abs(sample_df["b_exact_T"]) + 1e-12
    )

    print("=== Linear Alfvén Wave MVP ===")
    print(f"B0 = {cfg.background_b0_t:.3e} T, rho = {cfg.mass_density_kg_m3:.3e} kg/m^3")
    print(f"v_A = {out['v_a']:.3f} m/s")
    print(
        "Grid: "
        f"N={cfg.n_grid}, dx={out['dx']:.3f} m, CFL={cfg.cfl:.2f}, "
        f"steps={out['steps']}, t_end={out['t_end']:.3f} s"
    )
    print(f"L2 relative error (v_y) = {out['v_l2_rel']:.3e}")
    print(f"L2 relative error (b_y) = {out['b_l2_rel']:.3e}")
    print(f"Relative perturbation-energy drift = {out['energy_rel_drift']:.3e}")
    print(f"max(|v_y|)/v_A = {out['mach_like_ratio']:.3e}")
    print("\nSample profile:")
    print(sample_df.to_string(index=False, float_format=lambda z: f"{z: .6e}"))

    checks = [
        (np.isfinite(v_num).all() and np.isfinite(b_num).all(), "finite_output"),
        (out["v_l2_rel"] < 2.5e-2, "velocity_accuracy"),
        (out["b_l2_rel"] < 2.5e-2, "magnetic_accuracy"),
        (out["energy_rel_drift"] < 4.0e-2, "energy_drift"),
        (out["mach_like_ratio"] < 0.3, "linear_small_perturbation"),
    ]
    passed = True
    for ok, name in checks:
        state = "PASS" if ok else "FAIL"
        print(f"{name}: {state}")
        if not ok:
            passed = False

    if not passed:
        raise SystemExit("Validation: FAIL")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
