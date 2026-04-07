"""Minimal MVP for Chandrasekhar limit estimation.

This script implements two complementary calculations:
1. Solve the n=3 Lane-Emden equation and compute the Chandrasekhar mass
   from the ultra-relativistic electron-degeneracy polytrope.
2. Solve Newtonian white-dwarf structure equations with the full
   zero-temperature electron-degeneracy EOS over a central-density grid,
   then extrapolate the asymptotic mass limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

G_CGS = 6.67430e-8
C_CGS = 2.99792458e10
HBAR_CGS = 1.054571817e-27
M_E_G = 9.1093837015e-28
M_U_G = 1.66053906660e-24
M_SUN_G = 1.98847e33
CM_PER_KM = 1.0e5


@dataclass(frozen=True)
class ChandraConfig:
    mu_e: float = 2.0
    x_c_min: float = 0.3
    x_c_max: float = 100.0
    n_models: int = 24
    x_surface: float = 1.0e-3
    r0_cm: float = 1.0
    r_max_cm: float = 3.0e9
    max_step_cm: float = 1.0e7
    rtol: float = 2.0e-6
    atol: float = 1.0e-8
    lane_emden_xi0: float = 1.0e-8
    lane_emden_xi_max: float = 20.0
    tail_fit_points: int = 8
    primary_method: str = "DOP853"
    secondary_method: str = "RK45"

    @property
    def rho_scale(self) -> float:
        # rho = mu_e m_u n_e, n_e = (m_e c)^3 x^3 / (3 pi^2 hbar^3)
        return self.mu_e * M_U_G * (M_E_G * C_CGS) ** 3 / (3.0 * math.pi**2 * HBAR_CGS**3)


@dataclass(frozen=True)
class LaneEmdenResult:
    xi_1: float
    omega_3: float
    steps: int


def pressure_from_x(x: float) -> float:
    """Cold electron-degeneracy pressure P(x) in cgs."""
    prefactor = M_E_G**4 * C_CGS**5 / (24.0 * math.pi**2 * HBAR_CGS**3)
    term = x * (2.0 * x * x - 3.0) * math.sqrt(1.0 + x * x) + 3.0 * math.asinh(x)
    return prefactor * term


def dpressure_dx(x: float) -> float:
    """Exact derivative dP/dx for the above EOS."""
    prefactor = M_E_G**4 * C_CGS**5 / (3.0 * math.pi**2 * HBAR_CGS**3)
    return prefactor * x**4 / math.sqrt(1.0 + x * x)


def density_from_x(x: float, cfg: ChandraConfig) -> float:
    return cfg.rho_scale * x**3


def lane_emden_rhs(xi: float, state: np.ndarray, n: float = 3.0) -> list[float]:
    theta, dtheta_dxi = float(state[0]), float(state[1])
    d2theta = -2.0 * dtheta_dxi / xi - max(theta, 0.0) ** n
    return [dtheta_dxi, d2theta]


def solve_lane_emden_n3(method: str, cfg: ChandraConfig) -> LaneEmdenResult:
    xi0 = cfg.lane_emden_xi0
    # Series expansion near origin: theta ~ 1 - xi^2/6, theta' ~ -xi/3
    y0 = [1.0 - xi0**2 / 6.0, -xi0 / 3.0]

    def surface_event(_xi: float, state: np.ndarray) -> float:
        return float(state[0])

    surface_event.terminal = True
    surface_event.direction = -1

    sol = solve_ivp(
        fun=lambda xi, y: lane_emden_rhs(xi, y, n=3.0),
        t_span=(xi0, cfg.lane_emden_xi_max),
        y0=y0,
        method=method,
        events=surface_event,
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=0.05,
    )

    if len(sol.t_events[0]) == 0:
        raise RuntimeError(f"Lane-Emden root not found with method={method}.")

    xi_1 = float(sol.t_events[0][0])
    dtheta_surface = float(sol.y_events[0][0][1])
    omega_3 = -xi_1**2 * dtheta_surface
    return LaneEmdenResult(xi_1=xi_1, omega_3=omega_3, steps=len(sol.t))


def relativistic_polytrope_k(mu_e: float) -> float:
    return ((3.0 * math.pi**2) ** (1.0 / 3.0) / 4.0) * HBAR_CGS * C_CGS * (1.0 / (mu_e * M_U_G)) ** (
        4.0 / 3.0
    )


def chandrasekhar_mass_from_lane_emden(omega_3: float, cfg: ChandraConfig) -> float:
    # For n=3: M = 4 pi (K / (pi G))^(3/2) * omega_3
    k_rel = relativistic_polytrope_k(cfg.mu_e)
    mass_g = 4.0 * math.pi * (k_rel / (math.pi * G_CGS)) ** 1.5 * omega_3
    return mass_g / M_SUN_G


def white_dwarf_rhs(radius_cm: float, state: np.ndarray, cfg: ChandraConfig) -> list[float]:
    mass_g, x = float(state[0]), float(state[1])
    if x <= cfg.x_surface:
        return [0.0, 0.0]

    rho = density_from_x(x, cfg)
    dm_dr = 4.0 * math.pi * radius_cm**2 * rho
    dx_dr = -(G_CGS * mass_g * rho) / (radius_cm**2 * dpressure_dx(x))
    return [dm_dr, dx_dr]


def make_x_surface_event(cfg: ChandraConfig):
    def surface_event(_radius_cm: float, state: np.ndarray) -> float:
        return float(state[1] - cfg.x_surface)

    surface_event.terminal = True
    surface_event.direction = -1
    return surface_event


def solve_single_white_dwarf(x_c: float, method: str, cfg: ChandraConfig) -> dict[str, float]:
    rho_c = density_from_x(x_c, cfg)
    m0 = 4.0 * math.pi * cfg.r0_cm**3 * rho_c / 3.0
    y0 = [m0, x_c]

    sol = solve_ivp(
        fun=lambda r, y: white_dwarf_rhs(r, y, cfg),
        t_span=(cfg.r0_cm, cfg.r_max_cm),
        y0=y0,
        method=method,
        events=make_x_surface_event(cfg),
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.max_step_cm,
    )

    if len(sol.t_events[0]) == 0:
        raise RuntimeError(f"Surface event not found for x_c={x_c:.3e}, method={method}.")

    radius_cm = float(sol.t_events[0][0])
    mass_g = float(sol.y_events[0][0][0])

    return {
        "x_c": float(x_c),
        "rho_c": float(rho_c),
        "mass_msun": mass_g / M_SUN_G,
        "radius_km": radius_cm / CM_PER_KM,
        "steps": float(len(sol.t)),
        "method": method,
    }


def build_white_dwarf_sequence(method: str, cfg: ChandraConfig) -> pd.DataFrame:
    x_values = np.geomspace(cfg.x_c_min, cfg.x_c_max, cfg.n_models)
    rows = [solve_single_white_dwarf(float(x_c), method=method, cfg=cfg) for x_c in x_values]
    return pd.DataFrame(rows)


def compare_sequences(primary_df: pd.DataFrame, secondary_df: pd.DataFrame) -> dict[str, float]:
    merged = primary_df.merge(secondary_df, on="x_c", suffixes=("_primary", "_secondary"), how="inner")

    m_p = merged["mass_msun_primary"].to_numpy()
    m_s = merged["mass_msun_secondary"].to_numpy()
    r_p = merged["radius_km_primary"].to_numpy()
    r_s = merged["radius_km_secondary"].to_numpy()

    return {
        "mass_mae_msun": float(mean_absolute_error(m_p, m_s)),
        "mass_r2": float(r2_score(m_p, m_s)),
        "radius_mae_km": float(mean_absolute_error(r_p, r_s)),
        "radius_r2": float(r2_score(r_p, r_s)),
    }


def estimate_limit_with_sklearn(df: pd.DataFrame, tail_points: int) -> dict[str, float]:
    tail = df.tail(tail_points).copy()
    z = (1.0 / tail["x_c"].to_numpy() ** 2).reshape(-1, 1)
    y = tail["mass_msun"].to_numpy()

    model = LinearRegression().fit(z, y)
    y_fit = model.predict(z)

    return {
        "m_limit_msun": float(model.intercept_),
        "slope": float(model.coef_[0]),
        "fit_r2": float(r2_score(y, y_fit)),
        "tail_last_mass_msun": float(y[-1]),
    }


def torch_monotonicity_diagnostics(mass_series: np.ndarray, radius_series: np.ndarray) -> dict[str, float]:
    m = torch.tensor(mass_series, dtype=torch.float64)
    r = torch.tensor(radius_series, dtype=torch.float64)

    dm = m[1:] - m[:-1]
    dr = r[1:] - r[:-1]

    positive_dm = int((dm > 0).sum().item())
    negative_dm = int((dm < 0).sum().item())
    negative_dr = int((dr < 0).sum().item())

    return {
        "n_positive_dm": float(positive_dm),
        "n_negative_dm": float(negative_dm),
        "n_negative_dr": float(negative_dr),
        "first_dm": float(dm[0].item()),
        "last_dm": float(dm[-1].item()),
        "min_dm": float(dm.min().item()),
    }


def print_compact_table(df: pd.DataFrame, title: str) -> None:
    view = df[["x_c", "rho_c", "mass_msun", "radius_km", "steps"]].copy()
    with pd.option_context("display.precision", 6, "display.width", 160):
        print(title)
        print(view.to_string(index=False))


def main() -> None:
    print("Chandrasekhar limit MVP via Lane-Emden + white-dwarf structure scan")

    cfg = ChandraConfig()

    lane_primary = solve_lane_emden_n3(cfg.primary_method, cfg)
    lane_secondary = solve_lane_emden_n3(cfg.secondary_method, cfg)
    m_ch_lane = chandrasekhar_mass_from_lane_emden(lane_primary.omega_3, cfg)

    primary_df = build_white_dwarf_sequence(cfg.primary_method, cfg)
    secondary_df = build_white_dwarf_sequence(cfg.secondary_method, cfg)

    seq_metrics = compare_sequences(primary_df, secondary_df)
    limit_fit = estimate_limit_with_sklearn(primary_df, tail_points=cfg.tail_fit_points)
    torch_diag = torch_monotonicity_diagnostics(
        primary_df["mass_msun"].to_numpy(),
        primary_df["radius_km"].to_numpy(),
    )

    print(
        f"\nLane-Emden n=3 ({cfg.primary_method}): "
        f"xi1={lane_primary.xi_1:.6f}, omega3={lane_primary.omega_3:.6f}, steps={lane_primary.steps}"
    )
    print(
        f"Lane-Emden n=3 ({cfg.secondary_method}): "
        f"xi1={lane_secondary.xi_1:.6f}, omega3={lane_secondary.omega_3:.6f}, steps={lane_secondary.steps}"
    )
    print(f"Chandrasekhar mass from Lane-Emden: M_ch={m_ch_lane:.6f} M_sun")

    print_compact_table(primary_df, "\nPrimary white-dwarf sequence:")

    print("\nAsymptotic-limit fit from high-density tail (sklearn LinearRegression):")
    for k, v in limit_fit.items():
        print(f"  {k}: {v:.6f}")

    print("\nPrimary vs secondary sequence consistency:")
    for k, v in seq_metrics.items():
        print(f"  {k}: {v:.6f}")

    print("\nTorch monotonicity diagnostics:")
    for k, v in torch_diag.items():
        print(f"  {k}: {v:.6f}")

    m_limit = limit_fit["m_limit_msun"]

    assert 6.85 < lane_primary.xi_1 < 6.95
    assert 1.95 < lane_primary.omega_3 < 2.08
    assert 1.35 < m_ch_lane < 1.55

    assert 1.30 < m_limit < 1.60
    assert abs(m_limit - m_ch_lane) < 0.08

    assert seq_metrics["mass_mae_msun"] < 0.01
    assert seq_metrics["mass_r2"] > 0.999
    assert seq_metrics["radius_mae_km"] < 100.0
    assert seq_metrics["radius_r2"] > 0.999

    assert torch_diag["n_negative_dm"] <= 1.0
    assert torch_diag["n_negative_dr"] >= 0.9 * (cfg.n_models - 1)
    assert torch_diag["last_dm"] < torch_diag["first_dm"]
    assert torch_diag["min_dm"] > -5.0e-4

    print("\nChecks passed: Chandrasekhar limit estimate is consistent across methods.")


if __name__ == "__main__":
    main()
