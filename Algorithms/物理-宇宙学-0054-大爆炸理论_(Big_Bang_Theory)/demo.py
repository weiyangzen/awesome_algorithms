"""Big Bang Theory MVP (PHYS-0054).

This script builds a transparent Big Bang timeline under a minimal LambdaCDM
background model. It computes:
- expansion history a(t), H(a), q(a)
- thermal history T(a) = T0 / a
- key cosmic milestones (BBN scale, equality, recombination, acceleration)
- epoch-wise power-law slopes via linear regression

The pipeline is deterministic and non-interactive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression

KM_PER_MPC = 3.0856775814913673e19
SEC_PER_YEAR = 365.25 * 24.0 * 3600.0
SEC_PER_GYR = 1.0e9 * SEC_PER_YEAR


@dataclass(frozen=True)
class BigBangConfig:
    h0_km_s_mpc: float = 67.4
    omega_m: float = 0.315
    omega_r: float = 9.0e-5
    omega_k: float = 0.0
    omega_lambda: Optional[float] = None
    cmb_temperature_k: float = 2.7255

    recombination_z: float = 1089.0
    bbn_temperature_k: float = 1.0e9

    a_min: float = 1.0e-12
    n_grid: int = 120_000


def resolved_densities(cfg: BigBangConfig) -> Tuple[float, float, float, float]:
    omega_lambda = cfg.omega_lambda
    if omega_lambda is None:
        omega_lambda = 1.0 - cfg.omega_m - cfg.omega_r - cfg.omega_k
    return cfg.omega_m, cfg.omega_r, cfg.omega_k, omega_lambda


def h0_si(h0_km_s_mpc: float) -> float:
    """H0 in s^-1."""
    return h0_km_s_mpc / KM_PER_MPC


def e2_of_a(a: np.ndarray, cfg: BigBangConfig) -> np.ndarray:
    """Dimensionless expansion rate E(a)^2 = H(a)^2 / H0^2."""
    omega_m, omega_r, omega_k, omega_lambda = resolved_densities(cfg)
    return omega_r / a**4 + omega_m / a**3 + omega_k / a**2 + omega_lambda


def q_of_a(a: np.ndarray, cfg: BigBangConfig) -> np.ndarray:
    """Deceleration parameter q(a) from Friedmann acceleration equation."""
    omega_m, omega_r, _, omega_lambda = resolved_densities(cfg)
    e2 = e2_of_a(a, cfg)
    num = omega_m / a**3 + 2.0 * omega_r / a**4 - 2.0 * omega_lambda
    return num / (2.0 * e2)


def build_background_history(cfg: BigBangConfig) -> pd.DataFrame:
    if cfg.a_min <= 0.0 or cfg.a_min >= 1.0:
        raise ValueError("a_min must lie in (0, 1)")
    if cfg.n_grid < 2000:
        raise ValueError("n_grid must be >= 2000 for stable timeline integration")

    a = np.logspace(np.log10(cfg.a_min), 0.0, cfg.n_grid)
    e2 = e2_of_a(a, cfg)
    if np.any(e2 <= 0.0):
        raise ValueError("Encountered non-physical E(a)^2 <= 0")

    e = np.sqrt(e2)
    h_km_s_mpc = cfg.h0_km_s_mpc * e
    dt_da = 1.0 / (a * h0_si(cfg.h0_km_s_mpc) * e)
    t_sec = cumulative_trapezoid(dt_da, a, initial=0.0)

    z = 1.0 / a - 1.0
    temperature_k = cfg.cmb_temperature_k / a

    omega_m, omega_r, omega_k, omega_lambda = resolved_densities(cfg)
    contributions = np.column_stack(
        [
            omega_r / a**4,
            omega_m / a**3,
            omega_k / a**2,
            np.full_like(a, omega_lambda),
        ]
    )
    labels = np.array(["radiation", "matter", "curvature", "dark_energy"])
    dominant_component = labels[np.argmax(contributions, axis=1)]

    return pd.DataFrame(
        {
            "a": a,
            "z": z,
            "t_sec": t_sec,
            "temperature_k": temperature_k,
            "H_km_s_Mpc": h_km_s_mpc,
            "q": q_of_a(a, cfg),
            "dominant_component": dominant_component,
        }
    )


def interpolate_by_z(history: pd.DataFrame, z_target: float, col: str) -> float:
    z_desc = history["z"].to_numpy()
    values_desc = history[col].to_numpy()

    z_asc = z_desc[::-1]
    values_asc = values_desc[::-1]

    if z_target < z_asc[0] or z_target > z_asc[-1]:
        raise ValueError(f"z_target={z_target} is outside history range")
    return float(np.interp(z_target, z_asc, values_asc))


def z_from_temperature(temperature_k: float, cfg: BigBangConfig) -> float:
    return temperature_k / cfg.cmb_temperature_k - 1.0


def find_acceleration_transition_z(cfg: BigBangConfig) -> Optional[float]:
    # Need q(0) < 0 and q(high-z) > 0 for a valid transition.
    z_grid = np.concatenate([[0.0], np.logspace(-6, 1.0, 5000)])
    a_grid = 1.0 / (1.0 + z_grid)
    q_grid = q_of_a(a_grid, cfg)

    if q_grid[0] >= 0.0:
        return None

    for i in range(z_grid.size - 1):
        if q_grid[i] <= 0.0 < q_grid[i + 1]:
            z0 = float(z_grid[i])
            z1 = float(z_grid[i + 1])

            def q_of_z(z_val: float) -> float:
                a_val = 1.0 / (1.0 + z_val)
                return float(q_of_a(np.array([a_val]), cfg)[0])

            return float(brentq(q_of_z, z0, z1))
    return None


def build_milestones(cfg: BigBangConfig, history: pd.DataFrame) -> pd.DataFrame:
    omega_m, omega_r, _, _ = resolved_densities(cfg)
    z_eq = omega_m / omega_r - 1.0
    z_bbn = z_from_temperature(cfg.bbn_temperature_k, cfg)
    z_acc = find_acceleration_transition_z(cfg)

    rows = []

    def add_row(name: str, z_val: float, note: str) -> None:
        t_sec = interpolate_by_z(history, z_val, "t_sec")
        rows.append(
            {
                "milestone": name,
                "z": z_val,
                "temperature_k": cfg.cmb_temperature_k * (1.0 + z_val),
                "time_sec": t_sec,
                "time_year": t_sec / SEC_PER_YEAR,
                "note": note,
            }
        )

    add_row("BBN temperature scale", z_bbn, "T ~ 1e9 K")
    add_row("Matter-radiation equality", z_eq, "rho_m = rho_r")
    add_row("Recombination / CMB last scattering", cfg.recombination_z, "z ~ 1089")

    if z_acc is not None:
        add_row("Expansion acceleration transition", z_acc, "q(z)=0")

    add_row("Present day", 0.0, "a=1, z=0")

    milestone_df = pd.DataFrame(rows).sort_values("time_sec", ascending=True).reset_index(drop=True)
    return milestone_df


def fit_epoch_power_laws(history: pd.DataFrame) -> pd.DataFrame:
    """Fit log10(H) = slope * log10(a) + intercept in selected epochs."""
    epoch_spec = [
        ("radiation-dominated", 1.0e5, 1.0e8, -2.0),
        ("matter-dominated", 10.0, 1000.0, -1.5),
        ("dark-energy-dominated", 0.0, 0.5, 0.0),
    ]

    records = []
    for name, z_min, z_max, theory_slope in epoch_spec:
        mask = (history["z"] >= z_min) & (history["z"] <= z_max)
        subset = history.loc[mask]
        if subset.shape[0] < 50:
            raise ValueError(f"Not enough samples for epoch fit: {name}")

        x = np.log10(subset["a"].to_numpy()).reshape(-1, 1)
        y = np.log10(subset["H_km_s_Mpc"].to_numpy())
        model = LinearRegression().fit(x, y)

        records.append(
            {
                "epoch": name,
                "z_range": f"[{z_min:g}, {z_max:g}]",
                "slope_fit": float(model.coef_[0]),
                "slope_theory": theory_slope,
                "r2": float(model.score(x, y)),
            }
        )

    return pd.DataFrame(records)


def torch_age_crosscheck_gyr(history: pd.DataFrame, cfg: BigBangConfig) -> float:
    """Cross-check total age using torch.trapz on the same integrand."""
    a_np = history["a"].to_numpy()
    a = torch.tensor(a_np, dtype=torch.float64)

    omega_m, omega_r, omega_k, omega_lambda = resolved_densities(cfg)
    e2 = omega_r / a**4 + omega_m / a**3 + omega_k / a**2 + omega_lambda
    e = torch.sqrt(e2)

    integrand = 1.0 / (a * h0_si(cfg.h0_km_s_mpc) * e)
    age_sec = torch.trapz(integrand, a).item()
    return float(age_sec / SEC_PER_GYR)


def main() -> None:
    cfg = BigBangConfig()
    history = build_background_history(cfg)
    milestones = build_milestones(cfg, history)
    epoch_fit = fit_epoch_power_laws(history)

    age_gyr_np = float(history["t_sec"].iloc[-1] / SEC_PER_GYR)
    age_gyr_torch = torch_age_crosscheck_gyr(history, cfg)

    print("=== Big Bang Theory MVP (PHYS-0054) ===")
    print(
        "Config: "
        f"H0={cfg.h0_km_s_mpc:.3f} km/s/Mpc, "
        f"Omega_m={cfg.omega_m:.6f}, Omega_r={cfg.omega_r:.6f}, "
        f"Omega_k={cfg.omega_k:.6f}, "
        f"Omega_lambda={'auto' if cfg.omega_lambda is None else f'{cfg.omega_lambda:.6f}'}, "
        f"a_min={cfg.a_min:.1e}, n_grid={cfg.n_grid}"
    )

    sample = pd.concat([history.head(3), history.tail(3)], axis=0)
    print("\nHistory sample (head+tail):")
    print(
        sample[
            ["a", "z", "t_sec", "temperature_k", "H_km_s_Mpc", "q", "dominant_component"]
        ].to_string(index=False)
    )

    print("\nMilestones:")
    print(milestones.to_string(index=False))

    print("\nEpoch power-law fits for H(a):")
    print(epoch_fit.to_string(index=False))

    slope_map: Dict[str, float] = {
        row["epoch"]: float(row["slope_fit"])
        for _, row in epoch_fit.iterrows()
    }
    r2_thresholds = {
        "radiation-dominated": 0.999,
        "matter-dominated": 0.995,
        "dark-energy-dominated": 0.99,
    }
    r2_ok = True
    for _, row in epoch_fit.iterrows():
        epoch_name = str(row["epoch"])
        if float(row["r2"]) < r2_thresholds[epoch_name]:
            r2_ok = False
            break

    t_recomb_yr = float(
        milestones.loc[
            milestones["milestone"] == "Recombination / CMB last scattering", "time_year"
        ].iloc[0]
    )
    z_eq = float(
        milestones.loc[milestones["milestone"] == "Matter-radiation equality", "z"].iloc[0]
    )

    z_acc_rows = milestones.loc[milestones["milestone"] == "Expansion acceleration transition", "z"]
    z_acc = float(z_acc_rows.iloc[0]) if not z_acc_rows.empty else np.nan

    checks = {
        "age in plausible range (13-15 Gyr)": 13.0 < age_gyr_np < 15.0,
        "matter-radiation equality z in plausible range": 2500.0 < z_eq < 4500.0,
        "recombination time in plausible range (2e5-6e5 yr)": 2.0e5 < t_recomb_yr < 6.0e5,
        "acceleration transition redshift in plausible range": 0.4 < z_acc < 1.0,
        "radiation-era slope close to -2": -2.3 < slope_map["radiation-dominated"] < -1.7,
        "matter-era slope close to -1.5": -1.7 < slope_map["matter-dominated"] < -1.3,
        "dark-energy-era slope near 0": -0.8 < slope_map["dark-energy-dominated"] < 0.2,
        "epoch regressions pass R^2 thresholds": r2_ok,
        "torch age cross-check agrees (<0.03 Gyr)": abs(age_gyr_np - age_gyr_torch) < 0.03,
    }

    print("\nConsistency checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    print(
        "\nAge summary: "
        f"numpy={age_gyr_np:.6f} Gyr, torch={age_gyr_torch:.6f} Gyr, "
        f"|delta|={abs(age_gyr_np - age_gyr_torch):.6f} Gyr"
    )

    if all(checks.values()):
        print("\nValidation: PASS")
        return

    print("\nValidation: FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
