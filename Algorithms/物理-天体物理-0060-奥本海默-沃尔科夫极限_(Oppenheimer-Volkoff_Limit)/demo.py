"""Minimal MVP for the Oppenheimer-Volkoff limit.

This script solves the Tolman-Oppenheimer-Volkoff (TOV) equations for a
relativistic polytropic equation of state (EOS), scans central density,
and estimates the maximum stable neutron-star mass (OV limit).
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
C_CGS = 2.99792458e10
M_SUN_G = 1.98847e33
CM_PER_KM = 1.0e5


@dataclass(frozen=True)
class TOVConfig:
    gamma: float = 2.0
    polytropic_k: float = 1.5e5
    rho_c_min: float = 3.0e14
    rho_c_max: float = 1.0e16
    n_models: int = 24
    rho_surface: float = 1.0e8
    r0_cm: float = 1.0
    r_max_cm: float = 3.0e7
    max_step_cm: float = 2.0e4
    rtol: float = 2.0e-6
    atol: float = 1.0e-6
    primary_method: str = "DOP853"
    secondary_method: str = "RK45"

    @property
    def p_surface(self) -> float:
        return self.polytropic_k * self.rho_surface**self.gamma


@dataclass(frozen=True)
class OVLResult:
    rho_c: float
    mass_msun: float
    radius_km: float
    compactness: float
    model_index: int


def pressure_from_density(rho: float, cfg: TOVConfig) -> float:
    return cfg.polytropic_k * rho**cfg.gamma


def density_from_pressure(pressure: float, cfg: TOVConfig) -> float:
    pressure_safe = max(pressure, 1.0e-60)
    return (pressure_safe / cfg.polytropic_k) ** (1.0 / cfg.gamma)


def tov_rhs(radius_cm: float, state: np.ndarray, cfg: TOVConfig) -> list[float]:
    mass_g, pressure = float(state[0]), float(state[1])
    if pressure <= 0.0:
        return [0.0, 0.0]

    rho = density_from_pressure(pressure, cfg)
    dm_dr = 4.0 * math.pi * radius_cm**2 * rho

    schwarzschild_factor = 1.0 - 2.0 * G_CGS * mass_g / (radius_cm * C_CGS**2)
    if schwarzschild_factor <= 1.0e-10:
        return [0.0, 0.0]

    dP_dr = (
        -G_CGS
        * (rho + pressure / C_CGS**2)
        * (mass_g + 4.0 * math.pi * radius_cm**3 * pressure / C_CGS**2)
        / (radius_cm**2 * schwarzschild_factor)
    )
    return [dm_dr, dP_dr]


def make_surface_event(cfg: TOVConfig):
    def surface_event(_radius_cm: float, state: np.ndarray) -> float:
        return float(state[1] - cfg.p_surface)

    surface_event.terminal = True
    surface_event.direction = -1
    return surface_event


def solve_single_star(rho_c: float, cfg: TOVConfig, method: str) -> dict[str, float]:
    p_c = pressure_from_density(rho_c, cfg)
    m0 = 4.0 * math.pi * cfg.r0_cm**3 * rho_c / 3.0

    surface_event = make_surface_event(cfg)

    sol = solve_ivp(
        fun=lambda r, y: tov_rhs(r, y, cfg),
        t_span=(cfg.r0_cm, cfg.r_max_cm),
        y0=[m0, p_c],
        method=method,
        events=surface_event,
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.max_step_cm,
    )

    if len(sol.t_events[0]) == 0:
        raise RuntimeError(
            f"Surface event not found for rho_c={rho_c:.3e} with method={method}."
        )

    radius_cm = float(sol.t_events[0][0])
    mass_g = float(sol.y_events[0][0][0])
    compactness = 2.0 * G_CGS * mass_g / (radius_cm * C_CGS**2)

    return {
        "rho_c": float(rho_c),
        "p_c": float(p_c),
        "radius_km": radius_cm / CM_PER_KM,
        "mass_msun": mass_g / M_SUN_G,
        "compactness": compactness,
        "steps": float(len(sol.t)),
        "method": method,
    }


def build_mass_radius_sequence(cfg: TOVConfig, method: str) -> pd.DataFrame:
    rho_values = np.geomspace(cfg.rho_c_min, cfg.rho_c_max, cfg.n_models)
    rows = [solve_single_star(float(rho_c), cfg, method=method) for rho_c in rho_values]
    return pd.DataFrame(rows)


def estimate_ov_limit(df: pd.DataFrame) -> OVLResult:
    idx = int(df["mass_msun"].idxmax())
    row = df.loc[idx]
    return OVLResult(
        rho_c=float(row["rho_c"]),
        mass_msun=float(row["mass_msun"]),
        radius_km=float(row["radius_km"]),
        compactness=float(row["compactness"]),
        model_index=idx,
    )


def compare_solvers(primary_df: pd.DataFrame, secondary_df: pd.DataFrame) -> dict[str, float]:
    merged = primary_df.merge(
        secondary_df,
        on="rho_c",
        suffixes=("_primary", "_secondary"),
        how="inner",
    )

    m_primary = merged["mass_msun_primary"].to_numpy()
    m_secondary = merged["mass_msun_secondary"].to_numpy()
    r_primary = merged["radius_km_primary"].to_numpy()
    r_secondary = merged["radius_km_secondary"].to_numpy()

    return {
        "mass_mae_msun": float(mean_absolute_error(m_primary, m_secondary)),
        "mass_r2": float(r2_score(m_primary, m_secondary)),
        "radius_mae_km": float(mean_absolute_error(r_primary, r_secondary)),
        "radius_r2": float(r2_score(r_primary, r_secondary)),
    }


def analyze_turning_point_with_torch(mass_series: np.ndarray) -> dict[str, float]:
    mass_t = torch.tensor(mass_series, dtype=torch.float64)
    dm = mass_t[1:] - mass_t[:-1]

    pos = int((dm > 0).sum().item())
    neg = int((dm < 0).sum().item())

    return {
        "n_positive_deltas": float(pos),
        "n_negative_deltas": float(neg),
        "max_delta": float(dm.max().item()),
        "min_delta": float(dm.min().item()),
    }


def print_compact_table(df: pd.DataFrame, title: str) -> None:
    view = df[["rho_c", "mass_msun", "radius_km", "compactness", "steps"]].copy()
    with pd.option_context("display.precision", 5, "display.width", 140):
        print(title)
        print(view.to_string(index=False))


def main() -> None:
    print("Oppenheimer-Volkoff limit MVP via TOV integration")

    cfg = TOVConfig()
    primary_df = build_mass_radius_sequence(cfg, method=cfg.primary_method)
    secondary_df = build_mass_radius_sequence(cfg, method=cfg.secondary_method)

    ov = estimate_ov_limit(primary_df)
    solver_metrics = compare_solvers(primary_df, secondary_df)
    turning = analyze_turning_point_with_torch(primary_df["mass_msun"].to_numpy())

    print_compact_table(primary_df, "\nPrimary solver mass-radius sequence:")
    print("\nEstimated OV limit:")
    print(
        f"  rho_c*={ov.rho_c:.3e} g/cm^3, M_max={ov.mass_msun:.4f} M_sun, "
        f"R={ov.radius_km:.3f} km, compactness={ov.compactness:.3f}, index={ov.model_index}"
    )

    print("\nCross-solver consistency (primary vs secondary):")
    for k, v in solver_metrics.items():
        print(f"  {k}: {v:.6f}")

    print("\nTurning-point diagnostics (torch finite differences):")
    for k, v in turning.items():
        print(f"  {k}: {v:.6f}")

    masses = primary_df["mass_msun"].to_numpy()
    idx = ov.model_index

    assert 0 < idx < len(primary_df) - 1, "Peak mass must be interior, not boundary."
    assert 1.4 < ov.mass_msun < 2.6
    assert 8.0 < ov.radius_km < 20.0
    assert 0.1 < ov.compactness < 0.7

    stable_diff = np.diff(masses[: idx + 1])
    unstable_diff = np.diff(masses[idx:])
    assert stable_diff.min() > -2.0e-3
    assert unstable_diff.max() < 2.0e-3

    assert solver_metrics["mass_mae_msun"] < 0.02
    assert solver_metrics["mass_r2"] > 0.995
    assert solver_metrics["radius_mae_km"] < 0.2
    assert solver_metrics["radius_r2"] > 0.995

    assert turning["n_positive_deltas"] > 0
    assert turning["n_negative_deltas"] > 0

    print("\nChecks passed: turning-point maximum identified as OV-limit estimate.")


if __name__ == "__main__":
    main()
