"""Minimal runnable MVP for Inertial Confinement Fusion (ICF).

This script implements a transparent 0D hotspot model with:
- Prescribed spherical implosion trajectory R(t)
- DT burn source term using a tabulated-reactivity surrogate
- Alpha self-heating with areal-density-dependent deposition fraction
- Bremsstrahlung radiation loss
- Time integration of species inventories and internal energy

The model is intentionally simplified for algorithm demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Physical constants (SI)
K_B = 1.380649e-23
M_D = 3.343583719e-27
M_T = 5.0073567446e-27
M_ALPHA = 6.644657230e-27
M_AVG = 0.5 * (M_D + M_T)
E_ALPHA_J = 3.5e6 * 1.602176634e-19
E_FUSION_TOTAL_J = 17.6e6 * 1.602176634e-19


@dataclass(frozen=True)
class ICFConfig:
    # Geometry and timing
    radius_initial_m: float = 1.0e-3
    radius_stagnation_m: float = 6.0e-5
    t_stagnation_s: float = 10.0e-9
    t_end_s: float = 16.0e-9
    rebound_speed_m_s: float = 2.0e4

    # Initial plasma state
    rho_initial_kg_m3: float = 6.0
    temperature_initial_keV: float = 0.35

    # Alpha transport and radiation
    alpha_stop_areal_density_kg_m2: float = 3.0
    z_eff: float = 1.0

    # ODE and sampling
    n_eval: int = 700
    rtol: float = 1.0e-6
    atol: float = 1.0e-9
    max_step_s: float = 5.0e-11


@dataclass(frozen=True)
class SimScales:
    n0: float
    e0: float


def validate_config(cfg: ICFConfig) -> None:
    if cfg.radius_initial_m <= cfg.radius_stagnation_m:
        raise ValueError("radius_initial_m must be larger than radius_stagnation_m")
    if cfg.t_stagnation_s <= 0.0 or cfg.t_end_s <= cfg.t_stagnation_s:
        raise ValueError("Require 0 < t_stagnation_s < t_end_s")
    if cfg.rho_initial_kg_m3 <= 0.0:
        raise ValueError("rho_initial_kg_m3 must be positive")
    if cfg.temperature_initial_keV <= 0.0:
        raise ValueError("temperature_initial_keV must be positive")
    if cfg.alpha_stop_areal_density_kg_m2 <= 0.0:
        raise ValueError("alpha_stop_areal_density_kg_m2 must be positive")
    if cfg.n_eval < 20:
        raise ValueError("n_eval must be at least 20")


def radius_and_velocity(t: float, cfg: ICFConfig) -> Tuple[float, float]:
    """Return radius R(t) and dR/dt for a smooth implosion + linear rebound profile."""
    r0 = cfg.radius_initial_m
    rs = cfg.radius_stagnation_m
    ts = cfg.t_stagnation_s

    if t <= ts:
        # Half-cosine compression for smooth acceleration and stagnation velocity.
        phase = np.pi * t / ts
        r = r0 - (r0 - rs) * 0.5 * (1.0 - np.cos(phase))
        dr_dt = -(r0 - rs) * 0.5 * np.sin(phase) * (np.pi / ts)
        return r, dr_dt

    # Post-stagnation: simple linear rebound.
    dt = t - ts
    r = rs + cfg.rebound_speed_m_s * dt
    dr_dt = cfg.rebound_speed_m_s
    return r, dr_dt


def sphere_volume(radius_m: float) -> float:
    return (4.0 / 3.0) * np.pi * radius_m**3


def dt_reactivity_m3_s(temperature_keV: float) -> float:
    """Piecewise-log surrogate of DT reactivity <sigma v> [m^3/s].

    Values are hand-tuned to stay in physically plausible DT ranges for
    algorithmic demonstration (not a precision nuclear-data table).
    """
    t_grid = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0, 100.0])
    sv_grid = np.array([1e-28, 3e-27, 1e-26, 8e-26, 3e-25, 7e-25, 2e-24, 4e-24, 6e-24, 7e-24, 6e-24, 5e-24])

    t = float(np.clip(temperature_keV, t_grid[0], t_grid[-1]))
    log_sv = np.interp(np.log(t), np.log(t_grid), np.log(sv_grid))
    return float(np.exp(log_sv))


def bremsstrahlung_power_density_w_m3(n_e: float, n_i: float, t_keV: float, z_eff: float) -> float:
    """Simple optically-thin bremsstrahlung model.

    p_brem ~ C * Z_eff^2 * n_e * n_i * sqrt(T_e[eV]).
    """
    t_eV = max(1.0, t_keV * 1e3)
    c_brem = 1.69e-38  # SI-like empirical constant for simple MVP scaling
    return c_brem * (z_eff**2) * n_e * n_i * np.sqrt(t_eV)


def unpack_state_scaled(y_scaled: np.ndarray) -> Tuple[float, float, float, float]:
    x_d, x_t, x_alpha, eps = y_scaled
    x_d = float(np.clip(x_d, 0.0, 2.0))
    x_t = float(np.clip(x_t, 0.0, 2.0))
    x_alpha = float(np.clip(x_alpha, 0.0, 4.0))
    eps = float(np.clip(eps, 0.0, 1.0e4))
    return x_d, x_t, x_alpha, eps


def derived_quantities(t: float, y_scaled: np.ndarray, cfg: ICFConfig, scales: SimScales) -> Dict[str, float]:
    x_d, x_t, x_alpha, eps = unpack_state_scaled(y_scaled)
    n_d = x_d * scales.n0
    n_t = x_t * scales.n0
    n_alpha_count = x_alpha * scales.n0
    energy_j = eps * scales.e0

    radius_m, dr_dt = radius_and_velocity(t, cfg)
    radius_m = max(radius_m, 1e-8)
    vol = sphere_volume(radius_m)
    dvol_dt = 4.0 * np.pi * radius_m**2 * dr_dt

    # Number densities
    nd = n_d / vol
    nt = n_t / vol
    n_alpha = n_alpha_count / vol
    n_i = nd + nt + n_alpha
    n_e = nd + nt + 2.0 * n_alpha

    # Single-temperature plasma internal energy: E = 3/2 (n_i + n_e) kT V
    particle_factor = max((n_i + n_e) * vol, 1.0)
    t_kelvin = (2.0 / 3.0) * energy_j / (particle_factor * K_B)
    t_kelvin = max(t_kelvin, 100.0)
    t_keV = t_kelvin * K_B / (1e3 * 1.602176634e-19)

    rho = (n_d * M_D + n_t * M_T + n_alpha_count * M_ALPHA) / vol
    rho_r = rho * radius_m

    sv = dt_reactivity_m3_s(t_keV)
    reaction_rate_raw = (n_d * n_t * sv) / vol  # [1/s]
    # Bound source term by available fuel on a fast but finite timescale.
    max_depletion_rate = min(n_d, n_t) / 1e-12
    reaction_rate = min(reaction_rate_raw, max_depletion_rate)

    f_alpha = 1.0 - np.exp(-rho_r / cfg.alpha_stop_areal_density_kg_m2)
    p_alpha = f_alpha * E_ALPHA_J * reaction_rate

    p_brem_density = bremsstrahlung_power_density_w_m3(n_e=n_e, n_i=n_i, t_keV=t_keV, z_eff=cfg.z_eff)
    p_brem = p_brem_density * vol

    # Pressure from ideal fully ionized plasma with shared T
    pressure = (n_i + n_e) * K_B * t_kelvin
    p_comp = -pressure * dvol_dt  # compression work term

    p_fusion_total = E_FUSION_TOTAL_J * reaction_rate

    return {
        "R_m": radius_m,
        "dR_dt_m_s": dr_dt,
        "V_m3": vol,
        "dV_dt_m3_s": dvol_dt,
        "N_D": n_d,
        "N_T": n_t,
        "N_alpha": n_alpha_count,
        "nD_m3": nd,
        "nT_m3": nt,
        "nI_m3": n_i,
        "nE_m3": n_e,
        "temperature_keV": t_keV,
        "temperature_K": t_kelvin,
        "rho_kg_m3": rho,
        "rhoR_kg_m2": rho_r,
        "reactivity_m3_s": sv,
        "reaction_rate_s": reaction_rate,
        "alpha_dep_frac": f_alpha,
        "p_alpha_W": p_alpha,
        "p_brem_W": p_brem,
        "p_comp_W": p_comp,
        "p_fusion_total_W": p_fusion_total,
    }


def rhs_scaled(t: float, y_scaled: np.ndarray, cfg: ICFConfig, scales: SimScales) -> np.ndarray:
    d = derived_quantities(t, y_scaled, cfg, scales)

    r_f = d["reaction_rate_s"]
    dx_d = -r_f / scales.n0
    dx_t = -r_f / scales.n0
    dx_alpha = r_f / scales.n0

    depsilon = (d["p_comp_W"] + d["p_alpha_W"] - d["p_brem_W"]) / scales.e0

    # Soft floor protection to avoid numerical runaway below zero inventories.
    x_d, x_t, _x_alpha, eps = unpack_state_scaled(y_scaled)
    if x_d <= 0.0 and dx_d < 0.0:
        dx_d = 0.0
    if x_t <= 0.0 and dx_t < 0.0:
        dx_t = 0.0
    if eps <= 0.0 and depsilon < 0.0:
        depsilon = 0.0

    return np.array([dx_d, dx_t, dx_alpha, depsilon], dtype=float)


def run_simulation(cfg: ICFConfig) -> Tuple[np.ndarray, np.ndarray, SimScales]:
    r0 = cfg.radius_initial_m
    v0 = sphere_volume(r0)

    mass0 = cfg.rho_initial_kg_m3 * v0
    n_ions_total = mass0 / M_AVG
    n_d0 = 0.5 * n_ions_total
    n_t0 = 0.5 * n_ions_total
    n_alpha0 = 0.0

    # E0 from ideal plasma, with n_e = n_i for pure DT ions.
    n_i0 = n_ions_total / v0
    n_e0 = n_i0
    t0_k = cfg.temperature_initial_keV * 1e3 * 1.602176634e-19 / K_B
    e0 = 1.5 * (n_i0 + n_e0) * K_B * t0_k * v0

    scales = SimScales(n0=n_ions_total, e0=e0)
    y0_scaled = np.array([n_d0 / scales.n0, n_t0 / scales.n0, n_alpha0 / scales.n0, e0 / scales.e0], dtype=float)

    t_eval = np.linspace(0.0, cfg.t_end_s, cfg.n_eval)
    sol = solve_ivp(
        fun=lambda t, y: rhs_scaled(t, y, cfg, scales),
        t_span=(0.0, cfg.t_end_s),
        y0=y0_scaled,
        method="Radau",
        t_eval=t_eval,
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.max_step_s,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol.t, sol.y, scales


def postprocess(t: np.ndarray, y_scaled: np.ndarray, cfg: ICFConfig, scales: SimScales) -> Tuple[pd.DataFrame, Dict[str, float]]:
    records = []
    for idx in range(t.size):
        yi = y_scaled[:, idx]
        d = derived_quantities(float(t[idx]), yi, cfg, scales)
        records.append(
            {
                "time_ns": t[idx] * 1e9,
                "radius_um": d["R_m"] * 1e6,
                "rho_g_cm3": d["rho_kg_m3"] / 1000.0,
                "temperature_keV": d["temperature_keV"],
                "rhoR_g_cm2": d["rhoR_kg_m2"] / 10.0,
                "alpha_dep_frac": d["alpha_dep_frac"],
                "reaction_rate_1e24_s": d["reaction_rate_s"] / 1e24,
                "P_fusion_GW": d["p_fusion_total_W"] / 1e9,
                "P_alpha_GW": d["p_alpha_W"] / 1e9,
                "P_brem_GW": d["p_brem_W"] / 1e9,
                "N_D_1e18": d["N_D"] / 1e18,
                "N_T_1e18": d["N_T"] / 1e18,
                "N_alpha_1e16": d["N_alpha"] / 1e16,
            }
        )

    df = pd.DataFrame.from_records(records)

    n_d0 = float(max(y_scaled[0, 0], 0.0) * scales.n0)
    n_t0 = float(max(y_scaled[1, 0], 0.0) * scales.n0)
    n_d_final = float(max(y_scaled[0, -1], 0.0) * scales.n0)
    n_t_final = float(max(y_scaled[1, -1], 0.0) * scales.n0)
    n_alpha_final = float(max(y_scaled[2, -1], 0.0) * scales.n0)

    burn_fraction_d = 1.0 - n_d_final / max(n_d0, 1.0)
    burn_fraction_t = 1.0 - n_t_final / max(n_t0, 1.0)
    burn_fraction = 0.5 * (burn_fraction_d + burn_fraction_t)

    yield_j = n_alpha_final * E_FUSION_TOTAL_J

    idx_peak_t = int(df["temperature_keV"].idxmax())
    idx_peak_power = int(df["P_fusion_GW"].idxmax())
    idx_stag = int(np.argmin(np.abs(df["time_ns"].to_numpy() - cfg.t_stagnation_s * 1e9)))

    p_alpha_stag = float(df.loc[idx_stag, "P_alpha_GW"])
    p_brem_stag = float(df.loc[idx_stag, "P_brem_GW"])
    ignition_margin = p_alpha_stag / max(p_brem_stag, 1e-12)

    hot_window = df["temperature_keV"].to_numpy() > 5.0
    tau_hot_s = float(np.trapezoid(hot_window.astype(float), t))

    n_i_peak = float(
        np.max(
            (
                (np.maximum(y_scaled[0], 0.0) + np.maximum(y_scaled[1], 0.0) + np.maximum(y_scaled[2], 0.0))
                * scales.n0
                / np.array([sphere_volume(radius_and_velocity(float(tt), cfg)[0]) for tt in t])
            )
        )
    )
    n_tau = n_i_peak * tau_hot_s

    summary = {
        "burn_fraction": burn_fraction,
        "yield_MJ": yield_j / 1e6,
        "peak_temperature_keV": float(df.loc[idx_peak_t, "temperature_keV"]),
        "peak_rhoR_g_cm2": float(df["rhoR_g_cm2"].max()),
        "peak_fusion_power_GW": float(df.loc[idx_peak_power, "P_fusion_GW"]),
        "time_of_peak_power_ns": float(df.loc[idx_peak_power, "time_ns"]),
        "ignition_margin_at_stag": ignition_margin,
        "hot_tau_ns_above_5keV": tau_hot_s * 1e9,
        "n_tau_m3_s": n_tau,
    }

    return df, summary


def run_checks(df: pd.DataFrame, summary: Dict[str, float]) -> None:
    finite_ok = np.isfinite(df.to_numpy()).all() and np.isfinite(np.array(list(summary.values()))).all()
    if not finite_ok:
        raise RuntimeError("Non-finite values encountered in outputs")

    if not (0.0 <= summary["burn_fraction"] < 1.0):
        raise RuntimeError("burn_fraction outside [0,1)")

    if summary["peak_temperature_keV"] <= 1.0:
        raise RuntimeError("peak_temperature_keV too low for meaningful DT burn window")

    if summary["peak_rhoR_g_cm2"] <= 0.05:
        raise RuntimeError("peak_rhoR_g_cm2 too low for alpha self-heating relevance")

    if summary["yield_MJ"] <= 0.0:
        raise RuntimeError("Fusion yield must be positive")


def main() -> None:
    cfg = ICFConfig()
    validate_config(cfg)

    t, y_scaled, scales = run_simulation(cfg)
    df, summary = postprocess(t=t, y_scaled=y_scaled, cfg=cfg, scales=scales)
    run_checks(df, summary)

    print("=== Inertial Confinement Fusion MVP (0D Hotspot Model) ===")
    print(
        "Config: "
        f"R0={cfg.radius_initial_m*1e6:.1f} um, Rstag={cfg.radius_stagnation_m*1e6:.1f} um, "
        f"t_stag={cfg.t_stagnation_s*1e9:.2f} ns, t_end={cfg.t_end_s*1e9:.2f} ns"
    )
    print(
        "Initial: "
        f"rho0={cfg.rho_initial_kg_m3:.3f} kg/m^3, T0={cfg.temperature_initial_keV:.3f} keV"
    )

    summary_df = pd.DataFrame(
        {
            "metric": list(summary.keys()),
            "value": list(summary.values()),
        }
    )
    print("\n--- Summary Metrics ---")
    print(summary_df.to_string(index=False, justify="left", float_format=lambda x: f"{x:,.6g}"))

    sample_idx = np.linspace(0, len(df) - 1, 10, dtype=int)
    print("\n--- Trajectory Samples ---")
    print(
        df.loc[
            sample_idx,
            [
                "time_ns",
                "radius_um",
                "rho_g_cm3",
                "temperature_keV",
                "rhoR_g_cm2",
                "P_fusion_GW",
                "P_alpha_GW",
                "P_brem_GW",
            ],
        ].to_string(index=False, float_format=lambda x: f"{x:,.5g}")
    )

    print("\nValidation: PASS")


if __name__ == "__main__":
    main()
