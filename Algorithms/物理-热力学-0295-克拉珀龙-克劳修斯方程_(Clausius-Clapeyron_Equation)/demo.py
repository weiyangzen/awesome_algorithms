"""Minimal runnable MVP for the Clausius-Clapeyron equation.

Core relation under common assumptions (ideal vapor, negligible liquid volume):
    d ln(P) / d(1/T) = -L / R

Integrated form with approximately constant latent heat L:
    ln(P) = -L/(R*T) + C
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress

R_UNIVERSAL = 8.31446261815324  # J/(mol*K)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for deterministic vapor-pressure experiment."""

    latent_heat_true_j_per_mol: float = 40_700.0
    t_ref_k: float = 373.15
    p_ref_pa: float = 101_325.0
    t_min_k: float = 335.0
    t_max_k: float = 390.0
    n_points: int = 18
    noise_amplitude: float = 0.006
    linear_rel_error_tol: float = 0.012
    pairwise_rel_error_tol: float = 0.015
    latent_heat_rel_tol: float = 0.02

    def validate(self) -> None:
        if self.latent_heat_true_j_per_mol <= 0.0:
            raise ValueError("latent_heat_true_j_per_mol must be positive")
        if self.t_ref_k <= 0.0:
            raise ValueError("t_ref_k must be positive")
        if self.p_ref_pa <= 0.0:
            raise ValueError("p_ref_pa must be positive")
        if self.t_min_k <= 0.0 or self.t_max_k <= 0.0:
            raise ValueError("Temperature bounds must be positive")
        if self.t_max_k <= self.t_min_k:
            raise ValueError("t_max_k must be larger than t_min_k")
        if self.n_points < 6:
            raise ValueError("n_points must be at least 6")
        if not (0.0 <= self.noise_amplitude < 0.2):
            raise ValueError("noise_amplitude should be in [0, 0.2)")


def vapor_pressure_integrated(
    temperature_k: np.ndarray,
    latent_heat_j_per_mol: float,
    p_ref_pa: float,
    t_ref_k: float,
) -> np.ndarray:
    """Integrated Clausius-Clapeyron equation from a reference state."""
    if latent_heat_j_per_mol <= 0.0:
        raise ValueError("latent_heat_j_per_mol must be positive")
    if p_ref_pa <= 0.0:
        raise ValueError("p_ref_pa must be positive")
    if t_ref_k <= 0.0:
        raise ValueError("t_ref_k must be positive")
    if np.any(temperature_k <= 0.0):
        raise ValueError("All temperatures must be positive")

    exponent = -(latent_heat_j_per_mol / R_UNIVERSAL) * (1.0 / temperature_k - 1.0 / t_ref_k)
    return p_ref_pa * np.exp(exponent)


def nonlinear_pressure_model(temperature_k: np.ndarray, latent_heat_j_per_mol: float, c_const: float) -> np.ndarray:
    """Model P(T) = exp(-L/(R*T) + C)."""
    return np.exp(-latent_heat_j_per_mol / (R_UNIVERSAL * temperature_k) + c_const)


def generate_dataset(cfg: ExperimentConfig) -> pd.DataFrame:
    """Generate deterministic synthetic measurements for vapor pressure."""
    cfg.validate()

    temperature_k = np.linspace(cfg.t_min_k, cfg.t_max_k, cfg.n_points)
    p_true_pa = vapor_pressure_integrated(
        temperature_k,
        latent_heat_j_per_mol=cfg.latent_heat_true_j_per_mol,
        p_ref_pa=cfg.p_ref_pa,
        t_ref_k=cfg.t_ref_k,
    )

    phase = np.linspace(0.0, 2.0 * np.pi, cfg.n_points)
    p_measured_pa = p_true_pa * (1.0 + cfg.noise_amplitude * np.sin(phase))

    return pd.DataFrame(
        {
            "T_K": temperature_k,
            "inv_T_1_per_K": 1.0 / temperature_k,
            "P_true_Pa": p_true_pa,
            "P_measured_Pa": p_measured_pa,
            "ln_P_measured": np.log(p_measured_pa),
        }
    )


def fit_linearized_clausius_clapeyron(df: pd.DataFrame) -> dict[str, float]:
    """Fit ln(P) = m*(1/T) + b, where m = -L/R."""
    x = df["inv_T_1_per_K"].to_numpy(dtype=np.float64)
    y = df["ln_P_measured"].to_numpy(dtype=np.float64)

    reg = linregress(x, y)
    latent_heat_est = -reg.slope * R_UNIVERSAL
    return {
        "slope": float(reg.slope),
        "intercept": float(reg.intercept),
        "r2": float(reg.rvalue**2),
        "latent_heat_j_per_mol": float(latent_heat_est),
    }


def fit_nonlinear_clausius_clapeyron(df: pd.DataFrame, init_latent_heat: float, init_c: float) -> dict[str, float]:
    """Nonlinear fit directly in pressure space."""
    xdata = df["T_K"].to_numpy(dtype=np.float64)
    ydata = df["P_measured_Pa"].to_numpy(dtype=np.float64)

    popt, _ = curve_fit(
        nonlinear_pressure_model,
        xdata,
        ydata,
        p0=(init_latent_heat, init_c),
        bounds=([1e3, -200.0], [2e5, 200.0]),
        maxfev=20000,
    )

    return {
        "latent_heat_j_per_mol": float(popt[0]),
        "c_const": float(popt[1]),
    }


def estimate_local_latent_heat(df: pd.DataFrame) -> np.ndarray:
    """Estimate local L(T) from differential form L = -R * dlnP / d(1/T)."""
    x = df["inv_T_1_per_K"].to_numpy(dtype=np.float64)
    y = df["ln_P_measured"].to_numpy(dtype=np.float64)
    dlnp_dinv_t = np.gradient(y, x, edge_order=2)
    return -R_UNIVERSAL * dlnp_dinv_t


def pairwise_prediction_table(df: pd.DataFrame, latent_heat_est: float) -> pd.DataFrame:
    """Check adjacent-pair pressure prediction using the integrated relation."""
    t = df["T_K"].to_numpy(dtype=np.float64)
    p = df["P_measured_Pa"].to_numpy(dtype=np.float64)

    rows: list[dict[str, float | int]] = []
    for i in range(t.size - 1):
        t1 = float(t[i])
        t2 = float(t[i + 1])
        p1 = float(p[i])
        p2_true = float(p[i + 1])

        p2_pred = p1 * np.exp(-(latent_heat_est / R_UNIVERSAL) * (1.0 / t2 - 1.0 / t1))
        rel_error = abs(p2_pred - p2_true) / p2_true

        rows.append(
            {
                "pair_index": i,
                "T1_K": t1,
                "T2_K": t2,
                "P2_true_Pa": p2_true,
                "P2_pred_Pa": float(p2_pred),
                "rel_error": float(rel_error),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    cfg = ExperimentConfig()
    df = generate_dataset(cfg)

    linear_fit = fit_linearized_clausius_clapeyron(df)
    nonlinear_fit = fit_nonlinear_clausius_clapeyron(
        df,
        init_latent_heat=linear_fit["latent_heat_j_per_mol"],
        init_c=linear_fit["intercept"],
    )

    p_fit_linear = np.exp(
        linear_fit["slope"] * df["inv_T_1_per_K"].to_numpy(dtype=np.float64)
        + linear_fit["intercept"]
    )
    df["P_fit_linear_Pa"] = p_fit_linear
    df["linear_rel_error"] = (
        np.abs(df["P_fit_linear_Pa"] - df["P_measured_Pa"]) / df["P_measured_Pa"]
    )

    local_latent_heat = estimate_local_latent_heat(df)
    df["L_local_J_per_mol"] = local_latent_heat

    pair_df = pairwise_prediction_table(df, linear_fit["latent_heat_j_per_mol"])

    summary = {
        "L_true_J_per_mol": cfg.latent_heat_true_j_per_mol,
        "L_linear_J_per_mol": linear_fit["latent_heat_j_per_mol"],
        "L_nonlinear_J_per_mol": nonlinear_fit["latent_heat_j_per_mol"],
        "L_local_mean_J_per_mol": float(np.mean(local_latent_heat)),
        "L_local_std_J_per_mol": float(np.std(local_latent_heat)),
        "linear_r2": linear_fit["r2"],
        "linear_mean_rel_error": float(df["linear_rel_error"].mean()),
        "linear_max_rel_error": float(df["linear_rel_error"].max()),
        "pairwise_max_rel_error": float(pair_df["rel_error"].max()),
    }

    summary_df = pd.DataFrame([summary])

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("Clausius-Clapeyron Equation MVP")
    print("Differential form: d ln(P) / d(1/T) = -L/R")
    print("Integrated form:   ln(P) = -L/(R*T) + C")
    print()

    print("Summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print()

    print("Measurement and fit table:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))
    print()

    print("Adjacent-pair prediction checks:")
    print(pair_df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))

    # Deterministic assertions.
    rel_linear = abs(summary["L_linear_J_per_mol"] - summary["L_true_J_per_mol"]) / summary["L_true_J_per_mol"]
    rel_nonlinear = abs(summary["L_nonlinear_J_per_mol"] - summary["L_true_J_per_mol"]) / summary["L_true_J_per_mol"]
    rel_local_mean = abs(summary["L_local_mean_J_per_mol"] - summary["L_true_J_per_mol"]) / summary["L_true_J_per_mol"]

    assert (df["P_measured_Pa"] > 0.0).all(), "Measured pressures must stay positive"
    assert linear_fit["r2"] > 0.999, f"Linearized fit R^2 too low: {linear_fit['r2']}"
    assert rel_linear < cfg.latent_heat_rel_tol, f"Linear latent heat relative error too large: {rel_linear}"
    assert rel_nonlinear < cfg.latent_heat_rel_tol, f"Nonlinear latent heat relative error too large: {rel_nonlinear}"
    assert rel_local_mean < 0.03, f"Local latent heat mean relative error too large: {rel_local_mean}"
    assert summary["linear_mean_rel_error"] < cfg.linear_rel_error_tol
    assert summary["pairwise_max_rel_error"] < cfg.pairwise_rel_error_tol

    print("All checks passed.")


if __name__ == "__main__":
    main()
