"""Minimal runnable MVP for Hawking radiation (semiclassical approximation).

The script models a Schwarzschild black hole as a thermal emitter with
an effective emissivity factor, integrates mass loss ODE, and compares
numerical vs analytic mass evolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import constants
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class HawkingConfig:
    """Configuration for the semiclassical evaporation model."""

    emissivity: float = 1.0
    min_mass_kg: float = 1.0
    rtol: float = 1e-8
    atol: float = 1e-10


def hawking_temperature_kelvin(mass_kg: float) -> float:
    """Hawking temperature T_H for a non-rotating, neutral black hole."""
    mass = max(float(mass_kg), 1e-30)
    return constants.hbar * constants.c**3 / (
        8.0 * np.pi * constants.G * mass * constants.k
    )


def schwarzschild_radius_m(mass_kg: float) -> float:
    mass = max(float(mass_kg), 0.0)
    return 2.0 * constants.G * mass / constants.c**2


def horizon_area_m2(mass_kg: float) -> float:
    r_s = schwarzschild_radius_m(mass_kg)
    return 4.0 * np.pi * r_s**2


def hawking_power_w(mass_kg: float, emissivity: float = 1.0) -> float:
    """Approximate Hawking luminosity P ~ 1/M^2.

    This uses the common Schwarzschild scaling form with an extra emissivity
    coefficient to absorb greybody/species simplifications.
    """
    mass = max(float(mass_kg), 1e-30)
    prefactor = constants.hbar * constants.c**6 / (15360.0 * np.pi * constants.G**2)
    return float(emissivity) * prefactor / (mass**2)


def evaporation_constant_k(emissivity: float = 1.0) -> float:
    """K in dM/dt = -K/M^2."""
    return (
        float(emissivity)
        * constants.hbar
        * constants.c**4
        / (15360.0 * np.pi * constants.G**2)
    )


def analytic_mass_profile(m0_kg: float, t_s: np.ndarray, k_const: float) -> np.ndarray:
    mass_cubed = np.maximum(float(m0_kg) ** 3 - 3.0 * k_const * np.asarray(t_s), 0.0)
    return np.cbrt(mass_cubed)


def analytic_lifetime_seconds(m0_kg: float, k_const: float) -> float:
    return float(m0_kg) ** 3 / (3.0 * k_const)


def planck_peak_frequency_hz(temperature_k: float) -> float:
    """Peak frequency of B_nu (Wien displacement in frequency representation)."""
    x_peak = 2.8214393721220787
    return x_peak * constants.k * float(temperature_k) / constants.h


def mass_loss_rhs(
    _t: float,
    y: np.ndarray,
    emissivity: float,
    min_mass_kg: float,
) -> np.ndarray:
    mass = max(float(y[0]), float(min_mass_kg))
    dmdt = -hawking_power_w(mass, emissivity=emissivity) / (constants.c**2)
    return np.array([dmdt], dtype=float)


def integrate_evaporation(
    m0_kg: float,
    cfg: HawkingConfig,
    n_samples: int = 240,
) -> dict[str, np.ndarray | float]:
    """Numerically integrate dM/dt and compare with analytic M(t)."""
    m0 = float(m0_kg)
    k_const = evaporation_constant_k(cfg.emissivity)
    t_life = analytic_lifetime_seconds(m0, k_const)
    t_end = 0.999 * t_life
    t_eval = np.linspace(0.0, t_end, int(n_samples))

    sol = solve_ivp(
        mass_loss_rhs,
        t_span=(0.0, t_end),
        y0=np.array([m0], dtype=float),
        args=(cfg.emissivity, cfg.min_mass_kg),
        t_eval=t_eval,
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=max(t_end / 500.0, 1e-6),
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed for m0={m0:g} kg: {sol.message}")

    m_num = np.maximum(sol.y[0], 0.0)
    m_ana = analytic_mass_profile(m0, sol.t, k_const)
    rel_err = np.abs(m_num - m_ana) / np.maximum(m_ana, 1e-30)

    return {
        "t_s": sol.t,
        "m_numeric_kg": m_num,
        "m_analytic_kg": m_ana,
        "rel_error": rel_err,
        "lifetime_s": t_life,
        "max_rel_error": float(np.max(rel_err)),
    }


def build_summary(initial_masses_kg: np.ndarray, cfg: HawkingConfig) -> pd.DataFrame:
    k_const = evaporation_constant_k(cfg.emissivity)
    rows: list[dict[str, float]] = []

    for mass in np.asarray(initial_masses_kg, dtype=float):
        temp = hawking_temperature_kelvin(mass)
        power = hawking_power_w(mass, emissivity=cfg.emissivity)
        life = analytic_lifetime_seconds(mass, k_const)
        rows.append(
            {
                "M0_kg": mass,
                "T_H_K": temp,
                "P_H_W": power,
                "r_s_m": schwarzschild_radius_m(mass),
                "A_horizon_m2": horizon_area_m2(mass),
                "nu_peak_Hz": planck_peak_frequency_hz(temp),
                "lifetime_s": life,
                "lifetime_years": life / (365.25 * 24.0 * 3600.0),
            }
        )

    return pd.DataFrame(rows)


def sample_trajectory(solution: dict[str, np.ndarray | float], n_rows: int = 8) -> pd.DataFrame:
    t_s = np.asarray(solution["t_s"], dtype=float)
    m_num = np.asarray(solution["m_numeric_kg"], dtype=float)
    m_ana = np.asarray(solution["m_analytic_kg"], dtype=float)
    rel_err = np.asarray(solution["rel_error"], dtype=float)

    idx = np.unique(np.linspace(0, len(t_s) - 1, int(n_rows), dtype=int))
    return pd.DataFrame(
        {
            "time_s": t_s[idx],
            "mass_numeric_kg": m_num[idx],
            "mass_analytic_kg": m_ana[idx],
            "rel_error": rel_err[idx],
        }
    )


def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    cfg = HawkingConfig(emissivity=1.0, min_mass_kg=1.0)
    initial_masses_kg = np.array([2.5e5, 5.0e5, 1.0e6], dtype=float)

    print("=== Hawking Radiation MVP (Schwarzschild, semiclassical) ===")
    print(f"emissivity = {cfg.emissivity:.3f}")
    print(f"evaporation constant K = {evaporation_constant_k(cfg.emissivity):.6e} kg^3/s")
    print()

    summary = build_summary(initial_masses_kg, cfg)
    print("--- Initial Condition Summary ---")
    print(summary.to_string(index=False))
    print()

    print("--- ODE vs Analytic Consistency ---")
    solutions: list[dict[str, np.ndarray | float]] = []
    for mass in initial_masses_kg:
        sol = integrate_evaporation(mass, cfg)
        solutions.append(sol)
        print(
            "M0={:.3e} kg | lifetime={:.6e} s | max_rel_error={:.3e} | M(t_end)={:.3e} kg".format(
                mass,
                float(sol["lifetime_s"]),
                float(sol["max_rel_error"]),
                float(np.asarray(sol["m_numeric_kg"])[-1]),
            )
        )
    print()

    print("--- Sample Trajectory (first mass) ---")
    traj = sample_trajectory(solutions[0], n_rows=10)
    print(traj.to_string(index=False))


if __name__ == "__main__":
    main()
