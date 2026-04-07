"""Minimal MVP for the stellar structure equations.

This script integrates the 1D spherically symmetric stellar structure ODEs:
mass conservation, hydrostatic equilibrium, luminosity generation, and
radiative temperature gradient with an ideal-gas EOS closure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_absolute_error, r2_score

G_CGS = 6.67430e-8
K_B_CGS = 1.380649e-16
M_H_CGS = 1.6735575e-24
C_CGS = 2.99792458e10
A_RAD_CGS = 7.5657e-15

M_SUN_G = 1.98847e33
R_SUN_CM = 6.957e10
L_SUN_ERG_S = 3.828e33


@dataclass(frozen=True)
class StellarConfig:
    mean_molecular_weight: float = 0.61
    hydrogen_fraction: float = 0.70
    opacity_cm2_per_g: float = 0.34
    epsilon0: float = 1.07e-7

    central_density_g_cm3: float = 150.0
    central_temperature_k: float = 1.0e6
    surface_pressure_ratio: float = 1.0e-6

    r0_cm: float = 1.0e6
    r_max_cm: float = 2.0e12
    max_step_cm: float = 2.0e9
    rtol: float = 1.0e-6
    atol: float = 1.0e-8

    primary_method: str = "DOP853"
    secondary_method: str = "RK45"

    @property
    def central_pressure_dyn_cm2(self) -> float:
        return (
            self.central_density_g_cm3
            * K_B_CGS
            * self.central_temperature_k
            / (self.mean_molecular_weight * M_H_CGS)
        )

    @property
    def surface_pressure_dyn_cm2(self) -> float:
        return self.surface_pressure_ratio * self.central_pressure_dyn_cm2


def density_from_pressure_temperature(pressure: float, temperature: float, cfg: StellarConfig) -> float:
    temperature_safe = max(float(temperature), 1.0e-30)
    pressure_safe = max(float(pressure), 0.0)
    return max(
        pressure_safe * cfg.mean_molecular_weight * M_H_CGS / (K_B_CGS * temperature_safe),
        1.0e-30,
    )


def pp_chain_proxy_epsilon(rho_g_cm3: float, temperature_k: float, cfg: StellarConfig) -> float:
    t6 = max(float(temperature_k), 1.0) / 1.0e6
    return cfg.epsilon0 * float(rho_g_cm3) * (cfg.hydrogen_fraction**2) * (t6**4)


def stellar_structure_rhs(radius_cm: float, state: np.ndarray, cfg: StellarConfig) -> list[float]:
    mass_g = float(state[0])
    pressure = float(state[1])
    luminosity = float(state[2])
    temperature = float(state[3])

    if pressure <= 0.0 or temperature <= 0.0:
        return [0.0, 0.0, 0.0, 0.0]

    r2 = max(radius_cm * radius_cm, 1.0e-30)
    rho = density_from_pressure_temperature(pressure, temperature, cfg)
    epsilon = pp_chain_proxy_epsilon(rho, temperature, cfg)

    dm_dr = 4.0 * math.pi * r2 * rho
    dP_dr = -G_CGS * mass_g * rho / r2
    dL_dr = 4.0 * math.pi * r2 * rho * epsilon
    dT_dr = (
        -3.0
        * cfg.opacity_cm2_per_g
        * rho
        * max(luminosity, 1.0e-30)
        / (16.0 * math.pi * A_RAD_CGS * C_CGS * max(temperature**3, 1.0e-30) * r2)
    )

    return [dm_dr, dP_dr, dL_dr, dT_dr]


def make_surface_event(cfg: StellarConfig):
    def event(_radius_cm: float, state: np.ndarray) -> float:
        return float(state[1] - cfg.surface_pressure_dyn_cm2)

    event.terminal = True
    event.direction = -1
    return event


def solve_single_star(cfg: StellarConfig, method: str) -> dict[str, object]:
    rho_c = cfg.central_density_g_cm3
    p_c = cfg.central_pressure_dyn_cm2
    r0 = cfg.r0_cm

    m0 = 4.0 * math.pi * r0**3 * rho_c / 3.0
    eps_c = pp_chain_proxy_epsilon(rho_c, cfg.central_temperature_k, cfg)
    l0 = 4.0 * math.pi * r0**3 * rho_c * eps_c / 3.0
    y0 = [m0, p_c, l0, cfg.central_temperature_k]

    surface_event = make_surface_event(cfg)
    sol = solve_ivp(
        fun=lambda r, y: stellar_structure_rhs(r, y, cfg),
        t_span=(cfg.r0_cm, cfg.r_max_cm),
        y0=y0,
        method=method,
        events=surface_event,
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.max_step_cm,
    )

    reached_surface = len(sol.t_events[0]) > 0
    if reached_surface:
        radius_cm = float(sol.t_events[0][0])
        terminal = sol.y_events[0][0]
    else:
        radius_cm = float(sol.t[-1])
        terminal = sol.y[:, -1]

    mass_g = float(terminal[0])
    pressure = float(terminal[1])
    luminosity = float(terminal[2])
    temperature = float(terminal[3])
    rho = density_from_pressure_temperature(pressure, temperature, cfg)

    return {
        "method": method,
        "solution": sol,
        "reached_surface": reached_surface,
        "surface_radius_cm": radius_cm,
        "surface_mass_g": mass_g,
        "surface_pressure_dyn_cm2": pressure,
        "surface_luminosity_erg_s": luminosity,
        "surface_temperature_k": temperature,
        "surface_density_g_cm3": rho,
    }


def build_profile_dataframe(sol: solve_ivp, cfg: StellarConfig) -> pd.DataFrame:
    radius_cm = sol.t
    mass_g = sol.y[0]
    pressure = sol.y[1]
    luminosity = sol.y[2]
    temperature = sol.y[3]

    rho = np.array(
        [density_from_pressure_temperature(float(p), float(t), cfg) for p, t in zip(pressure, temperature)]
    )

    dP_dr_rhs = -G_CGS * mass_g * rho / np.maximum(radius_cm**2, 1.0e-30)
    dP_dr_numeric = np.gradient(pressure, radius_cm)
    hydrostatic_rel_error = np.abs((dP_dr_numeric - dP_dr_rhs) / np.maximum(np.abs(dP_dr_rhs), 1.0e-30))

    df = pd.DataFrame(
        {
            "radius_cm": radius_cm,
            "radius_rsun": radius_cm / R_SUN_CM,
            "mass_g": mass_g,
            "mass_msun": mass_g / M_SUN_G,
            "pressure_dyn_cm2": pressure,
            "temperature_k": temperature,
            "luminosity_erg_s": luminosity,
            "luminosity_lsun": luminosity / L_SUN_ERG_S,
            "density_g_cm3": rho,
            "hydro_rel_error": hydrostatic_rel_error,
        }
    )
    return df


def compare_solvers(primary_sol: solve_ivp, secondary_sol: solve_ivp) -> dict[str, float]:
    r_lo = max(float(primary_sol.t[0]), float(secondary_sol.t[0]))
    r_hi = min(float(primary_sol.t[-1]), float(secondary_sol.t[-1]))
    grid = np.geomspace(r_lo, r_hi, 120)

    def interp(sol: solve_ivp, idx: int) -> np.ndarray:
        return np.interp(grid, sol.t, sol.y[idx])

    m_primary = interp(primary_sol, 0)
    m_secondary = interp(secondary_sol, 0)
    p_primary = interp(primary_sol, 1)
    p_secondary = interp(secondary_sol, 1)
    t_primary = interp(primary_sol, 3)
    t_secondary = interp(secondary_sol, 3)

    return {
        "mass_mae_msun": float(mean_absolute_error(m_primary, m_secondary) / M_SUN_G),
        "mass_r2": float(r2_score(m_primary, m_secondary)),
        "pressure_mae_fraction": float(mean_absolute_error(p_primary, p_secondary) / p_primary[0]),
        "pressure_r2": float(r2_score(p_primary, p_secondary)),
        "temperature_mae_fraction": float(mean_absolute_error(t_primary, t_secondary) / t_primary[0]),
        "temperature_r2": float(r2_score(t_primary, t_secondary)),
    }


def torch_monotonicity_checks(profile_df: pd.DataFrame) -> dict[str, float]:
    mass = torch.tensor(profile_df["mass_msun"].to_numpy(), dtype=torch.float64)
    pressure = torch.tensor(profile_df["pressure_dyn_cm2"].to_numpy(), dtype=torch.float64)

    dmass = mass[1:] - mass[:-1]
    dpressure = pressure[1:] - pressure[:-1]

    return {
        "mass_decrease_violations": float((dmass < 0.0).sum().item()),
        "pressure_increase_violations": float((dpressure > 0.0).sum().item()),
        "max_dmass": float(dmass.max().item()),
        "min_dpressure": float(dpressure.min().item()),
    }


def print_profile_excerpt(df: pd.DataFrame) -> None:
    keep_cols = [
        "radius_rsun",
        "mass_msun",
        "pressure_dyn_cm2",
        "temperature_k",
        "density_g_cm3",
        "hydro_rel_error",
    ]
    stride = max(len(df) // 12, 1)
    sampled = df.iloc[::stride][keep_cols].copy()
    with pd.option_context("display.precision", 6, "display.width", 160):
        print(sampled.to_string(index=False))


def main() -> None:
    print("Stellar Structure Equations MVP (1D spherical, Newtonian + radiative transport)")

    cfg = StellarConfig()
    primary = solve_single_star(cfg, method=cfg.primary_method)
    secondary = solve_single_star(cfg, method=cfg.secondary_method)

    profile = build_profile_dataframe(primary["solution"], cfg)
    solver_metrics = compare_solvers(primary["solution"], secondary["solution"])
    monotonicity = torch_monotonicity_checks(profile)

    print("\nPrimary solution summary:")
    print(f"  method={primary['method']}, reached_surface={primary['reached_surface']}")
    print(
        f"  R={primary['surface_radius_cm'] / R_SUN_CM:.6f} R_sun, "
        f"M={primary['surface_mass_g'] / M_SUN_G:.6f} M_sun, "
        f"L={primary['surface_luminosity_erg_s'] / L_SUN_ERG_S:.6e} L_sun"
    )
    print(
        f"  P_surface/P_center={primary['surface_pressure_dyn_cm2'] / cfg.central_pressure_dyn_cm2:.6e}, "
        f"T_surface={primary['surface_temperature_k']:.3f} K, "
        f"rho_surface={primary['surface_density_g_cm3']:.3e} g/cm^3"
    )

    print("\nProfile excerpt:")
    print_profile_excerpt(profile)

    print("\nCross-solver consistency:")
    for k, v in solver_metrics.items():
        print(f"  {k}: {v:.6e}")

    print("\nTorch monotonicity diagnostics:")
    for k, v in monotonicity.items():
        print(f"  {k}: {v:.6f}")

    hydro_median = float(profile["hydro_rel_error"].median())
    hydro_p95 = float(profile["hydro_rel_error"].quantile(0.95))
    print(f"\nHydrostatic residual: median={hydro_median:.6e}, p95={hydro_p95:.6e}")

    assert bool(primary["reached_surface"])
    assert bool(secondary["reached_surface"])
    assert primary["surface_radius_cm"] > cfg.r0_cm
    assert primary["surface_mass_g"] > 0.0
    assert primary["surface_temperature_k"] > 0.0

    assert hydro_median < 1.0e-3
    assert hydro_p95 < 1.0e-2

    assert monotonicity["mass_decrease_violations"] <= 0.0
    assert monotonicity["pressure_increase_violations"] <= 0.0

    assert solver_metrics["mass_mae_msun"] < 5.0e-4
    assert solver_metrics["mass_r2"] > 0.999
    assert solver_metrics["pressure_mae_fraction"] < 5.0e-3
    assert solver_metrics["pressure_r2"] > 0.995
    assert solver_metrics["temperature_mae_fraction"] < 2.0e-4

    print("\nChecks passed: stellar structure integration is numerically consistent.")


if __name__ == "__main__":
    main()
