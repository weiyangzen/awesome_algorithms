"""Dark energy MVP: fit flat wCDM parameters from synthetic Type Ia supernova data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

C_KM_S = 299792.458


@dataclass(frozen=True)
class FitResult:
    omega_m: float
    w: float
    chi2: float


def validate_dataframe(df: pd.DataFrame) -> None:
    required = {"z", "mu_obs", "sigma_mu", "mu_true"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if (df["z"] <= 0.0).any():
        raise ValueError("All redshifts z must be > 0 to avoid log(0) in distance modulus.")
    if (df["sigma_mu"] <= 0.0).any():
        raise ValueError("All sigma_mu must be positive.")

    for col in ["z", "mu_obs", "sigma_mu", "mu_true"]:
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Column {col} contains non-finite values.")


def e_of_z(z: np.ndarray, omega_m: float, w: float) -> np.ndarray:
    if not (0.0 < omega_m < 1.0):
        raise ValueError("omega_m must be in (0, 1).")

    zp1 = 1.0 + z
    omega_de = 1.0 - omega_m
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        matter = omega_m * zp1**3
        de = omega_de * zp1 ** (3.0 * (1.0 + w))
        e2 = matter + de
        e = np.sqrt(e2)

    if not np.isfinite(e).all() or (e <= 0.0).any():
        raise RuntimeError("Encountered non-finite or non-positive E(z).")
    return e


def distance_modulus_flat_wcdm(z: np.ndarray, h0: float, omega_m: float, w: float) -> np.ndarray:
    if h0 <= 0.0:
        raise ValueError("h0 must be positive.")

    z = np.asarray(z, dtype=float)
    if z.ndim != 1:
        raise ValueError(f"z must be 1D, got shape={z.shape}.")
    if (z <= 0.0).any():
        raise ValueError("All z must be > 0.")

    order = np.argsort(z)
    z_sorted = z[order]

    # Build a monotonic grid from 0 to max(z) and integrate chi(z)=int dz'/E(z').
    z_grid = np.concatenate(([0.0], z_sorted))
    inv_e = 1.0 / e_of_z(z_grid, omega_m=omega_m, w=w)
    chi_sorted = cumulative_trapezoid(inv_e, z_grid, initial=0.0)[1:]

    d_l_mpc_sorted = (1.0 + z_sorted) * (C_KM_S / h0) * chi_sorted
    if (d_l_mpc_sorted <= 0.0).any() or not np.isfinite(d_l_mpc_sorted).all():
        raise RuntimeError("Luminosity distance became non-finite or non-positive.")

    mu_sorted = 5.0 * np.log10(d_l_mpc_sorted) + 25.0

    mu = np.empty_like(mu_sorted)
    mu[order] = mu_sorted
    return mu


def make_synthetic_snia(
    n_samples: int,
    h0: float,
    omega_m_true: float,
    w_true: float,
    seed: int = 57,
) -> pd.DataFrame:
    if n_samples < 20:
        raise ValueError("n_samples should be >= 20 for stable fitting.")

    rng = np.random.default_rng(seed)

    n_lowz = int(0.4 * n_samples)
    n_highz = n_samples - n_lowz
    z_low = rng.uniform(0.01, 0.25, size=n_lowz)
    z_high = rng.uniform(0.25, 1.5, size=n_highz)
    z = np.sort(np.concatenate([z_low, z_high]))

    # Keep observational noise realistic but moderate, so injected parameters
    # can still be recovered with a simple grid-search MVP.
    sigma_mu = 0.03 + 0.015 * z + rng.uniform(0.0, 0.008, size=n_samples)

    mu_true = distance_modulus_flat_wcdm(
        z=z,
        h0=h0,
        omega_m=omega_m_true,
        w=w_true,
    )
    noise = rng.normal(loc=0.0, scale=sigma_mu)
    mu_obs = mu_true + noise

    return pd.DataFrame(
        {
            "z": z,
            "mu_obs": mu_obs,
            "sigma_mu": sigma_mu,
            "mu_true": mu_true,
        }
    )


def chi2_stat(df: pd.DataFrame, h0: float, omega_m: float, w: float) -> float:
    mu_model = distance_modulus_flat_wcdm(
        z=df["z"].to_numpy(dtype=float),
        h0=h0,
        omega_m=omega_m,
        w=w,
    )
    residual = (df["mu_obs"].to_numpy(dtype=float) - mu_model) / df["sigma_mu"].to_numpy(dtype=float)
    return float(np.dot(residual, residual))


def grid_scan(
    df: pd.DataFrame,
    h0: float,
    omega_range: Tuple[float, float],
    w_range: Tuple[float, float],
    omega_steps: int,
    w_steps: int,
) -> Tuple[FitResult, pd.DataFrame]:
    omega_values = np.linspace(omega_range[0], omega_range[1], omega_steps)
    w_values = np.linspace(w_range[0], w_range[1], w_steps)

    records: List[Tuple[float, float, float]] = []
    best = FitResult(omega_m=np.nan, w=np.nan, chi2=np.inf)

    for omega_m in omega_values:
        for w in w_values:
            chi2 = chi2_stat(df=df, h0=h0, omega_m=float(omega_m), w=float(w))
            records.append((float(omega_m), float(w), chi2))
            if chi2 < best.chi2:
                best = FitResult(omega_m=float(omega_m), w=float(w), chi2=chi2)

    table = pd.DataFrame(records, columns=["omega_m", "w", "chi2"]).sort_values("chi2").reset_index(drop=True)
    return best, table


def clip_interval(center: float, half_width: float, low: float, high: float) -> Tuple[float, float]:
    left = max(low, center - half_width)
    right = min(high, center + half_width)
    if left >= right:
        raise RuntimeError("Invalid local interval after clipping.")
    return left, right


def fit_dark_energy_wcdm(df: pd.DataFrame, h0: float) -> Dict[str, object]:
    stage1_best, stage1_table = grid_scan(
        df=df,
        h0=h0,
        omega_range=(0.10, 0.50),
        w_range=(-1.60, -0.40),
        omega_steps=61,
        w_steps=81,
    )

    om2 = clip_interval(stage1_best.omega_m, half_width=0.06, low=0.05, high=0.70)
    w2 = clip_interval(stage1_best.w, half_width=0.24, low=-2.00, high=-0.20)
    stage2_best, stage2_table = grid_scan(
        df=df,
        h0=h0,
        omega_range=om2,
        w_range=w2,
        omega_steps=91,
        w_steps=91,
    )

    om3 = clip_interval(stage2_best.omega_m, half_width=0.02, low=0.05, high=0.70)
    w3 = clip_interval(stage2_best.w, half_width=0.08, low=-2.00, high=-0.20)
    stage3_best, stage3_table = grid_scan(
        df=df,
        h0=h0,
        omega_range=om3,
        w_range=w3,
        omega_steps=101,
        w_steps=101,
    )

    return {
        "best": stage3_best,
        "stage1_table": stage1_table,
        "stage2_table": stage2_table,
        "stage3_table": stage3_table,
        "stage1_best": stage1_best,
        "stage2_best": stage2_best,
        "stage3_best": stage3_best,
    }


def estimate_1sigma_box(table: pd.DataFrame, chi2_min: float, delta_chi2: float = 2.30) -> Dict[str, Tuple[float, float]]:
    near = table[table["chi2"] <= (chi2_min + delta_chi2)]
    if near.empty:
        top = table.iloc[[0]]
        near = top

    return {
        "omega_m": (float(near["omega_m"].min()), float(near["omega_m"].max())),
        "w": (float(near["w"].min()), float(near["w"].max())),
    }


def main() -> None:
    h0 = 70.0
    omega_m_true = 0.30
    w_true = -1.00

    df = make_synthetic_snia(
        n_samples=180,
        h0=h0,
        omega_m_true=omega_m_true,
        w_true=w_true,
        seed=57,
    )
    validate_dataframe(df)

    fit = fit_dark_energy_wcdm(df=df, h0=h0)
    best: FitResult = fit["best"]  # type: ignore[assignment]
    final_table: pd.DataFrame = fit["stage3_table"]  # type: ignore[assignment]

    mu_fit = distance_modulus_flat_wcdm(
        z=df["z"].to_numpy(dtype=float),
        h0=h0,
        omega_m=best.omega_m,
        w=best.w,
    )

    mae = mean_absolute_error(df["mu_obs"], mu_fit)
    rmse = float(np.sqrt(mean_squared_error(df["mu_obs"], mu_fit)))
    r2 = r2_score(df["mu_obs"], mu_fit)

    n = len(df)
    dof = n - 2
    chi2_red = best.chi2 / dof
    interval = estimate_1sigma_box(final_table, chi2_min=best.chi2, delta_chi2=2.30)

    print("=== Dark Energy (flat wCDM) MVP ===")
    print(f"Synthetic sample size: {n}")
    print(f"Truth: omega_m={omega_m_true:.4f}, w={w_true:.4f}, H0={h0:.1f} km/s/Mpc")
    print(
        "Fitted: "
        f"omega_m={best.omega_m:.4f}, "
        f"w={best.w:.4f}, "
        f"chi2={best.chi2:.3f}, "
        f"chi2/dof={chi2_red:.3f}"
    )

    print(
        "Approx 1-sigma box (delta chi2=2.30): "
        f"omega_m in [{interval['omega_m'][0]:.4f}, {interval['omega_m'][1]:.4f}], "
        f"w in [{interval['w'][0]:.4f}, {interval['w'][1]:.4f}]"
    )

    print(
        "Prediction metrics against noisy observations: "
        f"MAE={mae:.4f} mag, RMSE={rmse:.4f} mag, R2={r2:.4f}"
    )

    print("\nTop-5 candidates from final local scan:")
    print(final_table.head(5).to_string(index=False, justify="center", float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
