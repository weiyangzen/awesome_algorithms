"""Minimal runnable MVP for cosmological power spectrum estimation.

This script builds a synthetic 3D density contrast field in a periodic box,
measures the isotropic power spectrum P(k), and verifies that the recovered
large-scale slope is consistent with the injected spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import linregress


@dataclass(frozen=True)
class PowerSpectrumConfig:
    """Configuration for synthetic-field generation and P(k) estimation."""

    n_grid: int = 64
    box_size_mpc: float = 500.0
    seed: int = 20260407

    # Target input spectrum: P(k) = A * (k / k_pivot)^n * exp(-(k/k_cut)^2)
    amplitude: float = 1.0
    spectral_index: float = -2.0
    k_pivot_mpc_inv: float = 0.1
    k_cut_mpc_inv: float = 0.35

    n_k_bins: int = 28


def make_k_grid(cfg: PowerSpectrumConfig) -> np.ndarray:
    """Return |k| on the full FFT grid in units of 1/Mpc."""
    dx = cfg.box_size_mpc / cfg.n_grid
    k_axis = 2.0 * np.pi * np.fft.fftfreq(cfg.n_grid, d=dx)
    kx, ky, kz = np.meshgrid(k_axis, k_axis, k_axis, indexing="ij")
    return np.sqrt(kx**2 + ky**2 + kz**2)


def fundamental_and_nyquist(cfg: PowerSpectrumConfig) -> tuple[float, float]:
    k_fund = 2.0 * np.pi / cfg.box_size_mpc
    k_nyq = np.pi * cfg.n_grid / cfg.box_size_mpc
    return k_fund, k_nyq


def target_power_spectrum(k_abs: np.ndarray, cfg: PowerSpectrumConfig) -> np.ndarray:
    """Input isotropic spectrum used to color white noise in Fourier space."""
    p = np.zeros_like(k_abs, dtype=float)
    mask = k_abs > 0.0
    k_use = k_abs[mask]

    power_law = (k_use / cfg.k_pivot_mpc_inv) ** cfg.spectral_index
    cutoff = np.exp(-((k_use / cfg.k_cut_mpc_inv) ** 2))
    p[mask] = cfg.amplitude * power_law * cutoff
    return p


def synthesize_density_field(cfg: PowerSpectrumConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a real-space density contrast field with prescribed P(k) shape."""
    rng = np.random.default_rng(cfg.seed)

    # White Gaussian field in real space.
    delta_white = rng.normal(loc=0.0, scale=1.0, size=(cfg.n_grid, cfg.n_grid, cfg.n_grid))

    # Color the field in Fourier space according to sqrt(P_target).
    k_abs = make_k_grid(cfg)
    p_target = target_power_spectrum(k_abs, cfg)
    filter_amp = np.sqrt(p_target)

    dk_white = np.fft.fftn(delta_white)
    dk_colored = dk_white * filter_amp

    delta_colored = np.fft.ifftn(dk_colored).real
    delta_colored -= float(np.mean(delta_colored))
    std = float(np.std(delta_colored))
    if std <= 0.0:
        raise RuntimeError("Generated field has zero variance.")
    delta_colored /= std

    return delta_colored, k_abs, p_target


def estimate_isotropic_power_spectrum(
    delta_x: np.ndarray,
    k_abs: np.ndarray,
    cfg: PowerSpectrumConfig,
) -> pd.DataFrame:
    """Estimate isotropic P(k) by shell averaging in k-space."""
    if delta_x.shape != (cfg.n_grid, cfg.n_grid, cfg.n_grid):
        raise ValueError("delta_x shape does not match configured grid.")

    dk = np.fft.fftn(delta_x)

    # Consistent with FFT convention: delta_k ~= (L/N)^3 * FFT[delta_x].
    norm = cfg.box_size_mpc**3 / (cfg.n_grid**6)
    power_3d = norm * np.abs(dk) ** 2

    k_fund, k_nyq = fundamental_and_nyquist(cfg)
    edges = np.geomspace(k_fund, k_nyq, cfg.n_k_bins + 1)

    k_flat = k_abs.ravel()
    p_flat = power_3d.ravel()

    valid = (k_flat > 0.0) & np.isfinite(p_flat) & (p_flat > 0.0)
    k_flat = k_flat[valid]
    p_flat = p_flat[valid]

    bin_id = np.digitize(k_flat, edges) - 1

    rows: list[dict[str, float]] = []
    for i in range(cfg.n_k_bins):
        m = bin_id == i
        count = int(np.count_nonzero(m))
        if count < 20:
            continue

        k_mean = float(np.mean(k_flat[m]))
        p_mean = float(np.mean(p_flat[m]))

        rows.append(
            {
                "k": k_mean,
                "Pk": p_mean,
                "n_modes": float(count),
            }
        )

    if not rows:
        raise RuntimeError("No valid k-bins were produced for isotropic P(k).")

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def fit_large_scale_power_law(df_pk: pd.DataFrame, cfg: PowerSpectrumConfig) -> dict[str, float]:
    """Fit log P(k) = c + n * log k in the large-scale regime."""
    k_fund, _ = fundamental_and_nyquist(cfg)
    k_min = 2.0 * k_fund
    k_max = min(0.60 * cfg.k_cut_mpc_inv, 0.40 * np.max(df_pk["k"].to_numpy()))

    mask = (df_pk["k"] >= k_min) & (df_pk["k"] <= k_max) & (df_pk["Pk"] > 0.0)
    sub = df_pk.loc[mask]
    if len(sub) < 6:
        raise RuntimeError("Not enough points in fit range for slope recovery.")

    x = np.log(sub["k"].to_numpy())
    y = np.log(sub["Pk"].to_numpy())

    reg = linregress(x, y)
    slope = float(reg.slope)
    intercept = float(reg.intercept)
    r2 = float(reg.rvalue**2)

    return {
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
        "fit_r2": float(r2),
        "fit_k_min": float(np.min(sub["k"])),
        "fit_k_max": float(np.max(sub["k"])),
        "fit_n_points": float(len(sub)),
    }


def evaluate_model_match(df_pk: pd.DataFrame, cfg: PowerSpectrumConfig) -> dict[str, float]:
    """Compare measured P(k) with input model shape up to one amplitude factor."""
    k = df_pk["k"].to_numpy()
    p_est = df_pk["Pk"].to_numpy()

    p_model = target_power_spectrum(k, cfg)
    if np.any(p_model <= 0.0):
        raise RuntimeError("Model power must be positive on fitted bins.")

    # Best scalar amplitude in least-squares sense: alpha = <p_est p_model>/<p_model^2>
    alpha = float(np.dot(p_est, p_model) / np.dot(p_model, p_model))
    p_model_scaled = alpha * p_model

    log_residual = np.log(p_est) - np.log(p_model_scaled)
    log_rmse = float(np.sqrt(np.mean(log_residual**2)))

    ratio = p_est / p_model_scaled
    return {
        "shape_scale_alpha": alpha,
        "shape_log_rmse": log_rmse,
        "ratio_p10": float(np.percentile(ratio, 10)),
        "ratio_p50": float(np.percentile(ratio, 50)),
        "ratio_p90": float(np.percentile(ratio, 90)),
    }


def make_report_table(df_pk: pd.DataFrame, cfg: PowerSpectrumConfig) -> pd.DataFrame:
    """Sample a few k-nodes for deterministic human-readable output."""
    k = df_pk["k"].to_numpy()
    target = target_power_spectrum(k, cfg)

    sample_idx = np.linspace(0, len(df_pk) - 1, 8, dtype=int)
    sampled = df_pk.iloc[sample_idx].copy()
    sampled["P_target(k)"] = target[sample_idx]
    sampled["Pk_over_target"] = sampled["Pk"].to_numpy() / sampled["P_target(k)"].to_numpy()
    return sampled.reset_index(drop=True)


def main() -> None:
    cfg = PowerSpectrumConfig()

    delta_x, k_abs, _ = synthesize_density_field(cfg)
    df_pk = estimate_isotropic_power_spectrum(delta_x, k_abs, cfg)

    fit = fit_large_scale_power_law(df_pk, cfg)
    match = evaluate_model_match(df_pk, cfg)
    table = make_report_table(df_pk, cfg)

    # Basic physical / numerical sanity checks for this MVP.
    slope_err = abs(fit["fit_slope"] - cfg.spectral_index)
    assert slope_err < 0.35, (
        f"Recovered slope mismatch too large: fit={fit['fit_slope']:.3f}, "
        f"target={cfg.spectral_index:.3f}"
    )
    assert fit["fit_r2"] > 0.93, f"Large-scale log-log fit quality too low: R^2={fit['fit_r2']:.4f}"
    assert match["shape_log_rmse"] < 0.65, (
        f"Power-spectrum shape mismatch too large: log-RMSE={match['shape_log_rmse']:.4f}"
    )

    print("Power Spectrum MVP (Cosmology)")
    print("=" * 72)
    print(
        f"Grid={cfg.n_grid}^3, L={cfg.box_size_mpc:.1f} Mpc, seed={cfg.seed}, "
        f"n_in={cfg.spectral_index:.3f}, k_cut={cfg.k_cut_mpc_inv:.3f} 1/Mpc"
    )
    print(
        f"Recovered large-scale slope n_fit={fit['fit_slope']:.4f} "
        f"(target {cfg.spectral_index:.4f}), R^2={fit['fit_r2']:.4f}"
    )
    print(
        f"Model-shape check: alpha={match['shape_scale_alpha']:.4e}, "
        f"log-RMSE={match['shape_log_rmse']:.4f}, "
        f"ratio[p10,p50,p90]=({match['ratio_p10']:.3f}, {match['ratio_p50']:.3f}, {match['ratio_p90']:.3f})"
    )
    print()
    print("Binned isotropic P(k) samples:")
    print(
        table.to_string(
            index=False,
            float_format=lambda x: f"{x:.6e}",
            justify="center",
        )
    )


if __name__ == "__main__":
    main()
