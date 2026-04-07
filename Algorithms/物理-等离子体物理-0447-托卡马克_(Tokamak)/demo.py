"""Minimal runnable MVP for Tokamak (PHYS-0427).

This script implements a transparent 0D tokamak power-balance model:
- D-T fusion reactivity surrogate (analytic closed form)
- ITER98(y,2) confinement-time scaling for tau_E
- Bremsstrahlung + transport loss model
- Steady-state temperature roots from net-power balance

The goal is educational algorithm traceability, not reactor-design fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

EV_TO_J = 1.602176634e-19
E_FUSION_DT_J = 17.6e6 * EV_TO_J
ALPHA_FRACTION = 3.5 / 17.6
LAWSON_TRIPLE_PRODUCT_DT = 3.0e21  # keV*s*m^-3, order-of-magnitude target
BREMS_COEFF = 5.35e-39  # W * m^3 / (m^-6 * sqrt(eV)), simplified coefficient
REACTIVITY_SCALE = 6.0e-22  # tunable surrogate scale for educational DT reactivity


@dataclass(frozen=True)
class TokamakConfig:
    # Machine / operation parameters (compact ITER-like educational setting)
    I_p_MA: float = 15.0
    B_t_T: float = 5.3
    R_m: float = 6.2
    a_m: float = 2.0
    kappa: float = 1.7
    isotope_mass_amu: float = 2.5
    H98: float = 1.0

    # Plasma and heating setup
    n20: float = 1.0  # n_e / 1e20 m^-3
    volume_m3: float = 500.0
    Z_eff: float = 1.8
    P_aux_MW: float = 50.0

    # Temperature scan range for equilibrium search
    t_min_keV: float = 2.0
    t_max_keV: float = 30.0
    t_grid_points: int = 240

    @property
    def epsilon(self) -> float:
        return self.a_m / self.R_m

    @property
    def n_e_m3(self) -> float:
        return self.n20 * 1.0e20


def tau_e_ipb98_seconds(cfg: TokamakConfig, heating_power_MW: float | None = None) -> float:
    """ITER98(y,2) confinement scaling.

    tau_E = 0.0562 * H98 * I_p^0.93 * B_t^0.15 * n20^0.41 * P^-0.69
            * R^1.97 * eps^0.58 * kappa^0.78 * M^0.19
    """

    p_mw = cfg.P_aux_MW if heating_power_MW is None else float(heating_power_MW)
    if p_mw <= 0.0:
        raise ValueError("heating_power_MW must be positive")

    tau_e = (
        0.0562
        * cfg.H98
        * (cfg.I_p_MA**0.93)
        * (cfg.B_t_T**0.15)
        * (cfg.n20**0.41)
        * (p_mw**-0.69)
        * (cfg.R_m**1.97)
        * (cfg.epsilon**0.58)
        * (cfg.kappa**0.78)
        * (cfg.isotope_mass_amu**0.19)
    )
    return float(tau_e)


def dt_reactivity_m3_s(temperature_keV: np.ndarray | float) -> np.ndarray:
    """Educational D-T <sigma v> surrogate in m^3/s.

    Surrogate form:
    <sigma v> = A * T^2 / (1 + T/15 + (T/15)^2) * exp(-13 / T^(1/3)),  T in keV
    """

    t = np.asarray(temperature_keV, dtype=float)
    t_safe = np.clip(t, 1e-6, None)
    return (
        REACTIVITY_SCALE
        * (t_safe**2)
        / (1.0 + t_safe / 15.0 + (t_safe / 15.0) ** 2)
        * np.exp(-13.0 / np.cbrt(t_safe))
    )


def power_terms_at_temperature(
    temperature_keV: float,
    cfg: TokamakConfig,
    tau_e_s: float,
) -> dict[str, float]:
    """Compute 0D power terms at a given temperature."""

    if temperature_keV <= 0.0:
        raise ValueError("temperature_keV must be positive")
    if tau_e_s <= 0.0:
        raise ValueError("tau_e_s must be positive")

    n_e = cfg.n_e_m3
    volume = cfg.volume_m3

    sigma_v = float(dt_reactivity_m3_s(temperature_keV))

    # D-T 50/50 mix: n_D = n_T = n_e / 2
    p_fusion_density_w_m3 = 0.25 * n_e * n_e * sigma_v * E_FUSION_DT_J
    p_fusion_mw = p_fusion_density_w_m3 * volume / 1e6
    p_alpha_mw = ALPHA_FRACTION * p_fusion_mw

    t_eV = temperature_keV * 1.0e3
    p_brem_density_w_m3 = BREMS_COEFF * cfg.Z_eff * n_e * n_e * np.sqrt(t_eV)
    p_brem_mw = p_brem_density_w_m3 * volume / 1e6

    # Total thermal energy (electrons + ions): W_th ~= 3 n k_B T * V
    w_thermal_j = 3.0 * n_e * (t_eV * EV_TO_J) * volume
    p_transport_mw = (w_thermal_j / tau_e_s) / 1e6

    p_loss_mw = p_transport_mw + p_brem_mw
    p_net_mw = cfg.P_aux_MW + p_alpha_mw - p_loss_mw

    return {
        "T_keV": float(temperature_keV),
        "sigma_v_m3_s": sigma_v,
        "P_fusion_MW": float(p_fusion_mw),
        "P_alpha_MW": float(p_alpha_mw),
        "P_brem_MW": float(p_brem_mw),
        "P_transport_MW": float(p_transport_mw),
        "P_loss_MW": float(p_loss_mw),
        "P_net_MW": float(p_net_mw),
        "tau_E_s": float(tau_e_s),
    }


def build_power_scan(cfg: TokamakConfig) -> tuple[pd.DataFrame, float]:
    tau_e_s = tau_e_ipb98_seconds(cfg)

    temperature_grid = np.linspace(cfg.t_min_keV, cfg.t_max_keV, cfg.t_grid_points)
    rows = [power_terms_at_temperature(float(t), cfg, tau_e_s) for t in temperature_grid]

    frame = pd.DataFrame(rows)
    frame["nTtau_keV_s_m3"] = cfg.n_e_m3 * frame["T_keV"] * frame["tau_E_s"]
    frame["Lawson_ratio"] = frame["nTtau_keV_s_m3"] / LAWSON_TRIPLE_PRODUCT_DT
    frame["ignition_margin"] = frame["P_alpha_MW"] / frame["P_loss_MW"]

    return frame, tau_e_s


def find_equilibrium_roots(frame: pd.DataFrame, cfg: TokamakConfig, tau_e_s: float) -> list[float]:
    temperatures = frame["T_keV"].to_numpy(dtype=float)
    net_power = frame["P_net_MW"].to_numpy(dtype=float)

    roots: list[float] = []
    for idx in range(len(temperatures) - 1):
        t1 = float(temperatures[idx])
        t2 = float(temperatures[idx + 1])
        f1 = float(net_power[idx])
        f2 = float(net_power[idx + 1])

        if f1 == 0.0:
            roots.append(t1)
            continue

        if f1 * f2 < 0.0:
            root = brentq(
                lambda t: power_terms_at_temperature(float(t), cfg, tau_e_s)["P_net_MW"],
                t1,
                t2,
            )
            roots.append(float(root))

    # Deduplicate possible repeats caused by exact-grid hits.
    roots = sorted(roots)
    deduped: list[float] = []
    for value in roots:
        if not deduped or abs(value - deduped[-1]) > 1e-4:
            deduped.append(value)

    return deduped


def summarize_operating_point(temperature_keV: float, cfg: TokamakConfig, tau_e_s: float) -> dict[str, float]:
    terms = power_terms_at_temperature(temperature_keV, cfg, tau_e_s)
    q_value = terms["P_fusion_MW"] / cfg.P_aux_MW
    triple_product = cfg.n_e_m3 * temperature_keV * tau_e_s
    lawson_ratio = triple_product / LAWSON_TRIPLE_PRODUCT_DT

    out = dict(terms)
    out["Q"] = float(q_value)
    out["nTtau_keV_s_m3"] = float(triple_product)
    out["Lawson_ratio"] = float(lawson_ratio)
    out["ignition_margin"] = float(terms["P_alpha_MW"] / terms["P_loss_MW"])
    return out


def main() -> None:
    cfg = TokamakConfig()
    scan_df, tau_e_s = build_power_scan(cfg)

    roots = find_equilibrium_roots(scan_df, cfg, tau_e_s)
    if not roots:
        raise RuntimeError("No equilibrium root found in the configured temperature range")

    # For tokamak burn analysis we focus on the high-temperature branch.
    selected_temperature = max(roots)
    op = summarize_operating_point(selected_temperature, cfg, tau_e_s)

    sample_indices = np.linspace(0, len(scan_df) - 1, 10, dtype=int)
    sample_df = scan_df.iloc[sample_indices].copy()
    sample_df = sample_df[
        [
            "T_keV",
            "P_fusion_MW",
            "P_alpha_MW",
            "P_loss_MW",
            "P_net_MW",
            "Lawson_ratio",
        ]
    ]

    print("=== Tokamak 0D Power Balance MVP ===")
    print(
        "machine: "
        f"I_p={cfg.I_p_MA:.1f} MA, B_t={cfg.B_t_T:.1f} T, R={cfg.R_m:.1f} m, "
        f"a={cfg.a_m:.1f} m, kappa={cfg.kappa:.2f}, H98={cfg.H98:.2f}"
    )
    print(
        "plasma: "
        f"n={cfg.n_e_m3:.3e} m^-3, V={cfg.volume_m3:.1f} m^3, "
        f"Z_eff={cfg.Z_eff:.2f}, P_aux={cfg.P_aux_MW:.1f} MW"
    )
    print(f"tau_E (ITER98y2, at P_aux) = {tau_e_s:.4f} s")

    print("\nSample scan rows:")
    print(sample_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nEquilibrium roots (P_net = 0) [keV]:")
    print(", ".join(f"{value:.6f}" for value in roots))

    print("\nSelected high-temperature operating point:")
    print(f"T_eq = {op['T_keV']:.6f} keV")
    print(f"P_fusion = {op['P_fusion_MW']:.6f} MW")
    print(f"P_alpha = {op['P_alpha_MW']:.6f} MW")
    print(f"P_loss = {op['P_loss_MW']:.6f} MW")
    print(f"P_net = {op['P_net_MW']:.6e} MW")
    print(f"Q = P_fusion/P_aux = {op['Q']:.6f}")
    print(f"nTtau = {op['nTtau_keV_s_m3']:.6e} keV*s*m^-3")
    print(f"Lawson ratio (nTtau/3e21) = {op['Lawson_ratio']:.6f}")
    print(f"Ignition margin (P_alpha/P_loss) = {op['ignition_margin']:.6f}")

    checks = {
        "finite_scan": bool(np.isfinite(scan_df.to_numpy()).all()),
        "tau_E_positive": bool(tau_e_s > 0.0),
        "roots_found": bool(len(roots) >= 1),
        "two_branches_detected": bool(len(roots) >= 2),
        "root_residual_small": bool(abs(op["P_net_MW"]) < 1e-8),
        "gain_Q_gt_1": bool(op["Q"] > 1.0),
        "lawson_ratio_positive": bool(op["Lawson_ratio"] > 0.0),
    }
    passed = all(checks.values())

    print("\nChecks:")
    for name, flag in checks.items():
        print(f"- {name}: {'PASS' if flag else 'FAIL'}")

    print(f"\nValidation: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
