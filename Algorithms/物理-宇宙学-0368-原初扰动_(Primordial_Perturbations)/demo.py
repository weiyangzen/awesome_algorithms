"""Minimal runnable MVP for primordial perturbations in cosmology.

This script synthesizes a primordial curvature perturbation field, estimates
its isotropic power-spectrum tilt, and recovers local-type non-Gaussianity
with explicit source-level steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import normaltest, skew
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class PrimordialConfig:
    """Configuration for synthetic primordial perturbation generation."""

    n_grid: int = 48
    box_size_mpc: float = 1400.0
    seed: int = 20260407

    # Primordial spectrum model:
    # P_zeta(k) = A_s * (k/k0)^(n_s - 1) * exp(-(k/k_cut)^2)
    amplitude: float = 2.1e-9
    spectral_index: float = 0.965
    k_pivot_mpc_inv: float = 0.05
    k_cut_mpc_inv: float = 0.22

    # Local-type non-Gaussianity: zeta = zeta_g + f_NL (zeta_g^2 - <zeta_g^2>)
    f_nl_local: float = 7.0
    sigma_g_target: float = 7.5e-3

    n_k_bins: int = 26
    torch_steps: int = 1000
    torch_lr: float = 0.08


def make_k_grid(cfg: PrimordialConfig) -> np.ndarray:
    """Return |k| over the FFT grid in units of 1/Mpc."""
    dx = cfg.box_size_mpc / cfg.n_grid
    k_axis = 2.0 * np.pi * np.fft.fftfreq(cfg.n_grid, d=dx)
    kx, ky, kz = np.meshgrid(k_axis, k_axis, k_axis, indexing="ij")
    return np.sqrt(kx**2 + ky**2 + kz**2)


def fundamental_and_nyquist(cfg: PrimordialConfig) -> tuple[float, float]:
    """Return fundamental and Nyquist wave numbers."""
    k_fund = 2.0 * np.pi / cfg.box_size_mpc
    k_nyq = np.pi * cfg.n_grid / cfg.box_size_mpc
    return k_fund, k_nyq


def primordial_power_spectrum(k_abs: np.ndarray, cfg: PrimordialConfig) -> np.ndarray:
    """Compute the primordial curvature power spectrum model P_zeta(k)."""
    p = np.zeros_like(k_abs, dtype=float)
    mask = k_abs > 0.0
    k_use = k_abs[mask]

    power_law = np.power(k_use / cfg.k_pivot_mpc_inv, cfg.spectral_index - 1.0)
    cutoff = np.exp(-np.square(k_use / cfg.k_cut_mpc_inv))
    p[mask] = cfg.amplitude * power_law * cutoff
    return p


def synthesize_primordial_field(
    cfg: PrimordialConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate Gaussian and weakly non-Gaussian primordial fields."""
    rng = np.random.default_rng(cfg.seed)

    white = rng.normal(loc=0.0, scale=1.0, size=(cfg.n_grid, cfg.n_grid, cfg.n_grid))
    k_abs = make_k_grid(cfg)
    p_target = primordial_power_spectrum(k_abs, cfg)

    dk_white = np.fft.fftn(white)
    dk_gaussian = dk_white * np.sqrt(p_target)

    zeta_g = np.fft.ifftn(dk_gaussian).real
    zeta_g -= float(np.mean(zeta_g))

    std_g = float(np.std(zeta_g))
    if std_g <= 0.0:
        raise RuntimeError("Generated Gaussian field has zero variance.")
    zeta_g *= cfg.sigma_g_target / std_g

    var_g = float(np.var(zeta_g))
    zeta = zeta_g + cfg.f_nl_local * (np.square(zeta_g) - var_g)
    zeta -= float(np.mean(zeta))

    return zeta, zeta_g, k_abs, p_target


def estimate_isotropic_power_spectrum(
    field: np.ndarray,
    k_abs: np.ndarray,
    cfg: PrimordialConfig,
) -> pd.DataFrame:
    """Estimate isotropic P(k) via shell averaging."""
    if field.shape != (cfg.n_grid, cfg.n_grid, cfg.n_grid):
        raise ValueError("field shape does not match configured grid")

    dk = np.fft.fftn(field)
    norm = cfg.box_size_mpc**3 / (cfg.n_grid**6)
    p_3d = norm * np.abs(dk) ** 2

    k_fund, k_nyq = fundamental_and_nyquist(cfg)
    edges = np.geomspace(k_fund, k_nyq, cfg.n_k_bins + 1)

    k_flat = k_abs.ravel()
    p_flat = p_3d.ravel()

    valid = (k_flat > 0.0) & np.isfinite(p_flat) & (p_flat > 0.0)
    k_flat = k_flat[valid]
    p_flat = p_flat[valid]

    bin_ids = np.digitize(k_flat, edges) - 1

    rows: list[dict[str, float]] = []
    for i in range(cfg.n_k_bins):
        m = bin_ids == i
        count = int(np.count_nonzero(m))
        if count < 24:
            continue

        rows.append(
            {
                "k": float(np.mean(k_flat[m])),
                "Pk": float(np.mean(p_flat[m])),
                "n_modes": float(count),
            }
        )

    if not rows:
        raise RuntimeError("No valid binned spectrum was produced.")

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def fit_spectral_index_sklearn(df_pk: pd.DataFrame, cfg: PrimordialConfig) -> dict[str, float]:
    """Fit n_s using weighted linear regression in log-space."""
    sub = df_pk.copy()
    if len(sub) < 8:
        raise RuntimeError("Not enough spectral bins in fit range.")

    k = sub["k"].to_numpy()
    x = np.log(k / cfg.k_pivot_mpc_inv)
    y_raw = np.log(sub["Pk"].to_numpy())
    # Move the known exponential cutoff term to the left-hand side so that
    # the remaining relation is linear in log(k).
    y = y_raw + np.square(k / cfg.k_cut_mpc_inv)
    w = sub["n_modes"].to_numpy() / np.max(sub["n_modes"].to_numpy())

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y, sample_weight=w)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    n_s_hat = slope + 1.0
    a_hat = float(np.exp(intercept))

    y_pred = model.predict(x.reshape(-1, 1))
    r2 = float(model.score(x.reshape(-1, 1), y, sample_weight=w))

    return {
        "n_s_hat": n_s_hat,
        "A_eff_hat": a_hat,
        "r2": r2,
        "fit_k_min": float(sub["k"].min()),
        "fit_k_max": float(sub["k"].max()),
        "n_points": float(len(sub)),
        "rmse_log": float(np.sqrt(np.mean(np.square(y - y_pred)))),
    }


def fit_spectral_index_torch(df_pk: pd.DataFrame, cfg: PrimordialConfig) -> dict[str, float]:
    """Fit (A_s, n_s) using PyTorch gradient descent on log-spectrum loss."""
    x_np = np.log(df_pk["k"].to_numpy() / cfg.k_pivot_mpc_inv)
    y_np = np.log(df_pk["Pk"].to_numpy())
    k_np = df_pk["k"].to_numpy()
    w_np = df_pk["n_modes"].to_numpy() / np.max(df_pk["n_modes"].to_numpy())

    dtype = torch.float64
    x = torch.tensor(x_np, dtype=dtype)
    y = torch.tensor(y_np, dtype=dtype)
    k = torch.tensor(k_np, dtype=dtype)
    w = torch.tensor(w_np, dtype=dtype)

    ln_a = torch.tensor(float(np.median(y_np)), dtype=dtype, requires_grad=True)
    n_s = torch.tensor(0.95, dtype=dtype, requires_grad=True)

    optimizer = torch.optim.Adam([ln_a, n_s], lr=cfg.torch_lr)

    for _ in range(cfg.torch_steps):
        optimizer.zero_grad()
        y_hat = ln_a + (n_s - 1.0) * x - torch.square(k / cfg.k_cut_mpc_inv)
        loss = torch.sum(w * torch.square(y_hat - y)) / torch.sum(w)
        loss.backward()
        optimizer.step()

    ln_a_hat = float(ln_a.detach().cpu().item())
    n_s_hat = float(n_s.detach().cpu().item())

    y_hat_np = ln_a_hat + (n_s_hat - 1.0) * x_np - np.square(k_np / cfg.k_cut_mpc_inv)
    rmse_log = float(np.sqrt(np.mean(np.square(y_np - y_hat_np))))

    return {
        "n_s_hat": n_s_hat,
        "A_eff_hat": float(np.exp(ln_a_hat)),
        "rmse_log": rmse_log,
    }


def estimate_local_fnl(field: np.ndarray) -> dict[str, float]:
    """Estimate local f_NL from third moment and report normality diagnostics."""
    x = field.ravel()
    x = x - float(np.mean(x))

    sigma2 = float(np.var(x))
    mu3 = float(np.mean(np.power(x, 3)))

    eps = 1e-30
    f_nl_hat = mu3 / (6.0 * sigma2 * sigma2 + eps)

    skewness = float(skew(x, bias=False))
    normal_stat, normal_p = normaltest(x)

    return {
        "sigma": float(np.sqrt(sigma2)),
        "mu3": mu3,
        "skewness": skewness,
        "f_nl_hat": float(f_nl_hat),
        "normal_stat": float(normal_stat),
        "normal_p": float(normal_p),
    }


def make_sample_table(df_pk: pd.DataFrame, cfg: PrimordialConfig) -> pd.DataFrame:
    """Prepare deterministic sampled rows from binned spectrum."""
    k = df_pk["k"].to_numpy()
    p_model = primordial_power_spectrum(k, cfg)

    idx = np.linspace(0, len(df_pk) - 1, 8, dtype=int)
    sampled = df_pk.iloc[idx].copy()
    sampled["P_model"] = p_model[idx]
    sampled["Pk_over_model"] = sampled["Pk"].to_numpy() / sampled["P_model"].to_numpy()
    return sampled.reset_index(drop=True)


def main() -> None:
    cfg = PrimordialConfig()

    zeta, zeta_g, k_abs, _ = synthesize_primordial_field(cfg)
    df_pk = estimate_isotropic_power_spectrum(zeta, k_abs, cfg)

    fit_lr = fit_spectral_index_sklearn(df_pk, cfg)
    fit_torch = fit_spectral_index_torch(df_pk, cfg)

    non_gauss = estimate_local_fnl(zeta)
    gauss_ref = estimate_local_fnl(zeta_g)

    slope_err_lr = abs(fit_lr["n_s_hat"] - cfg.spectral_index)
    slope_err_torch = abs(fit_torch["n_s_hat"] - cfg.spectral_index)
    fnl_err = abs(non_gauss["f_nl_hat"] - cfg.f_nl_local)

    assert slope_err_lr < 0.08, (
        f"sklearn n_s mismatch too large: {fit_lr['n_s_hat']:.4f} vs {cfg.spectral_index:.4f}"
    )
    assert slope_err_torch < 0.08, (
        f"torch n_s mismatch too large: {fit_torch['n_s_hat']:.4f} vs {cfg.spectral_index:.4f}"
    )
    assert fit_lr["r2"] > 0.25, f"Low corrected-log regression quality: R^2={fit_lr['r2']:.4f}"
    assert fnl_err < 3.0, (
        f"f_NL estimator drift too large: {non_gauss['f_nl_hat']:.3f} vs {cfg.f_nl_local:.3f}"
    )
    assert abs(gauss_ref["f_nl_hat"]) < 0.5, (
        f"Gaussian reference should have near-zero f_NL, got {gauss_ref['f_nl_hat']:.3f}"
    )

    df_summary = pd.DataFrame(
        [
            {
                "method": "sklearn_weighted_logfit",
                "n_s_hat": fit_lr["n_s_hat"],
                "A_eff_hat": fit_lr["A_eff_hat"],
                "log_rmse": fit_lr["rmse_log"],
            },
            {
                "method": "torch_weighted_gd",
                "n_s_hat": fit_torch["n_s_hat"],
                "A_eff_hat": fit_torch["A_eff_hat"],
                "log_rmse": fit_torch["rmse_log"],
            },
        ]
    )

    df_non_gauss = pd.DataFrame(
        [
            {
                "field": "non_gaussian_zeta",
                "f_NL_true": cfg.f_nl_local,
                "f_NL_hat": non_gauss["f_nl_hat"],
                "skewness": non_gauss["skewness"],
                "normaltest_p": non_gauss["normal_p"],
            },
            {
                "field": "gaussian_reference",
                "f_NL_true": 0.0,
                "f_NL_hat": gauss_ref["f_nl_hat"],
                "skewness": gauss_ref["skewness"],
                "normaltest_p": gauss_ref["normal_p"],
            },
        ]
    )

    table = make_sample_table(df_pk, cfg)

    print("Primordial Perturbations MVP")
    print("=" * 72)
    print(
        f"Grid={cfg.n_grid}^3, L={cfg.box_size_mpc:.1f} Mpc, seed={cfg.seed}, "
        f"n_s,true={cfg.spectral_index:.4f}, f_NL,true={cfg.f_nl_local:.2f}"
    )
    print(
        f"Recovered n_s (sklearn)={fit_lr['n_s_hat']:.5f}, "
        f"(torch)={fit_torch['n_s_hat']:.5f}, fit R^2={fit_lr['r2']:.4f}"
    )
    print(
        f"Recovered f_NL (moment)={non_gauss['f_nl_hat']:.3f}; "
        f"Gaussian ref f_NL={gauss_ref['f_nl_hat']:.3f}"
    )
    print()
    print("Spectrum fit summary:")
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print()
    print("Non-Gaussianity diagnostics:")
    print(df_non_gauss.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print()
    print("Sampled isotropic P(k) bins:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6e}"))


if __name__ == "__main__":
    main()
