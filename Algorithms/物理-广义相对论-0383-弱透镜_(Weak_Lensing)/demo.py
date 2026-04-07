"""Weak Lensing MVP (PHYS-0364).

This script implements a compact, transparent weak-lensing pipeline:
1) Build a synthetic convergence map kappa(x, y) in the weak regime.
2) Forward model shear (gamma1, gamma2) from kappa in Fourier space.
3) Add shape noise and smooth observed shear.
4) Reconstruct kappa via Kaiser-Squires inversion.
5) Evaluate reconstruction quality with sklearn and PyTorch diagnostics.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


@dataclass(frozen=True)
class WeakLensingConfig:
    """Parameters for a small, reproducible weak-lensing demo."""

    n_grid: int = 128
    pixel_scale_arcmin: float = 0.4
    rng_seed: int = 42
    sigma_e: float = 0.30  # intrinsic ellipticity dispersion per component
    n_gal_per_pix: float = 30.0
    smooth_sigma_pix: float = 1.2


def fourier_kernels(n_grid: int, pixel_scale_arcmin: float) -> tuple[np.ndarray, ...]:
    """Return Fourier coordinates and Kaiser-Squires kernels."""

    freq = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=pixel_scale_arcmin)
    ky, kx = np.meshgrid(freq, freq, indexing="ij")
    k2 = kx**2 + ky**2

    d1 = np.zeros_like(k2)
    d2 = np.zeros_like(k2)
    mask = k2 > 0.0
    d1[mask] = (kx[mask] ** 2 - ky[mask] ** 2) / k2[mask]
    d2[mask] = (2.0 * kx[mask] * ky[mask]) / k2[mask]
    return kx, ky, k2, d1, d2


def synthetic_kappa_map(cfg: WeakLensingConfig) -> np.ndarray:
    """Generate a weak-regime toy convergence map from Gaussian halos."""

    n = cfg.n_grid
    coord = (np.arange(n) - 0.5 * (n - 1)) * cfg.pixel_scale_arcmin
    y, x = np.meshgrid(coord, coord, indexing="ij")

    # (amplitude, x0[arcmin], y0[arcmin], sigma[arcmin])
    halos = [
        (0.030, -8.0, -4.0, 4.5),
        (0.020, 9.0, 7.0, 5.5),
        (-0.012, 2.0, -10.0, 3.0),
        (0.015, -12.0, 10.0, 6.0),
    ]

    kappa = np.zeros((n, n), dtype=float)
    for amp, x0, y0, sigma in halos:
        r2 = (x - x0) ** 2 + (y - y0) ** 2
        kappa += amp * np.exp(-0.5 * r2 / (sigma**2))

    # Add a gentle large-scale mode to mimic line-of-sight structure.
    field_size = n * cfg.pixel_scale_arcmin
    kappa += 0.004 * np.sin(2.0 * np.pi * x / field_size) * np.cos(2.0 * np.pi * y / field_size)
    kappa -= np.mean(kappa)  # zero mode cannot be recovered by KS inversion
    return kappa


def forward_shear_from_kappa(kappa: np.ndarray, pixel_scale_arcmin: float) -> tuple[np.ndarray, np.ndarray]:
    """Map convergence to shear in Fourier space."""

    n = kappa.shape[0]
    kappa_hat = np.fft.fft2(kappa)
    _, _, _, d1, d2 = fourier_kernels(n, pixel_scale_arcmin)

    gamma1_hat = d1 * kappa_hat
    gamma2_hat = d2 * kappa_hat
    gamma1 = np.fft.ifft2(gamma1_hat).real
    gamma2 = np.fft.ifft2(gamma2_hat).real
    return gamma1, gamma2


def add_shape_noise(
    gamma1: np.ndarray,
    gamma2: np.ndarray,
    cfg: WeakLensingConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Add per-pixel shape noise using sigma_e / sqrt(n_gal)."""

    noise_std = cfg.sigma_e / np.sqrt(cfg.n_gal_per_pix)
    noisy_gamma1 = gamma1 + rng.normal(0.0, noise_std, size=gamma1.shape)
    noisy_gamma2 = gamma2 + rng.normal(0.0, noise_std, size=gamma2.shape)
    return noisy_gamma1, noisy_gamma2, float(noise_std)


def kaiser_squires_inversion(gamma1: np.ndarray, gamma2: np.ndarray, pixel_scale_arcmin: float) -> np.ndarray:
    """Reconstruct convergence from shear via Kaiser-Squires."""

    n = gamma1.shape[0]
    gamma1_hat = np.fft.fft2(gamma1)
    gamma2_hat = np.fft.fft2(gamma2)
    _, _, _, d1, d2 = fourier_kernels(n, pixel_scale_arcmin)

    kappa_hat = d1 * gamma1_hat + d2 * gamma2_hat
    kappa_hat[0, 0] = 0.0
    return np.fft.ifft2(kappa_hat).real


def reconstruction_metrics(kappa_true: np.ndarray, kappa_est: np.ndarray) -> dict[str, float]:
    """Compute standard scalar diagnostics for map reconstruction."""

    diff = kappa_est - kappa_true
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))

    true_flat = kappa_true.ravel()
    est_flat = kappa_est.ravel()
    if np.std(est_flat) < 1e-15 or np.std(true_flat) < 1e-15:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(true_flat, est_flat)[0, 1])

    return {"rmse": rmse, "mae": mae, "pearson_corr": corr}


def fit_affine_torch(
    kappa_in: np.ndarray,
    kappa_target: np.ndarray,
    steps: int = 300,
    lr: float = 0.08,
) -> tuple[float, float, float]:
    """Fit affine calibration kappa_out = m * kappa_in + b using PyTorch."""

    x = torch.tensor(kappa_in.ravel(), dtype=torch.float32)
    y = torch.tensor(kappa_target.ravel(), dtype=torch.float32)

    m = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([m, b], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        pred = m * x + b
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_pred = m * x + b
        rmse = torch.sqrt(torch.mean((final_pred - y) ** 2)).item()

    return float(m.item()), float(b.item()), float(rmse)


def radial_power_slope(kappa: np.ndarray, pixel_scale_arcmin: float) -> float:
    """Estimate log-power slope dlogP/dlogk on intermediate scales."""

    n = kappa.shape[0]
    kx, ky, _, _, _ = fourier_kernels(n, pixel_scale_arcmin)
    ell = np.sqrt(kx**2 + ky**2).ravel()
    power = (np.abs(np.fft.fft2(kappa)) ** 2 / kappa.size).ravel()

    mask = ell > 0.0
    ell = ell[mask]
    power = power[mask]

    bins = np.linspace(ell.min(), ell.max(), n // 2 + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    radial = np.full_like(centers, np.nan)
    for i in range(len(centers)):
        bmask = (ell >= bins[i]) & (ell < bins[i + 1])
        if np.any(bmask):
            radial[i] = power[bmask].mean()

    valid = np.isfinite(radial) & (radial > 0.0)
    centers = centers[valid]
    radial = radial[valid]
    if centers.size < 8:
        return float("nan")

    lo = np.quantile(centers, 0.10)
    hi = np.quantile(centers, 0.55)
    fit_mask = (centers >= lo) & (centers <= hi)
    if np.sum(fit_mask) < 5:
        return float("nan")

    reg = LinearRegression().fit(
        np.log10(centers[fit_mask]).reshape(-1, 1),
        np.log10(radial[fit_mask]),
    )
    return float(reg.coef_[0])


def main() -> None:
    cfg = WeakLensingConfig()
    rng = np.random.default_rng(cfg.rng_seed)
    torch.manual_seed(cfg.rng_seed)

    # 1) Truth map and forward model
    kappa_true = synthetic_kappa_map(cfg)
    gamma1_true, gamma2_true = forward_shear_from_kappa(kappa_true, cfg.pixel_scale_arcmin)

    # 2) Add observational noise
    gamma1_obs, gamma2_obs, noise_std = add_shape_noise(gamma1_true, gamma2_true, cfg, rng)

    # 3) Reconstruct with and without smoothing
    kappa_raw = kaiser_squires_inversion(gamma1_obs, gamma2_obs, cfg.pixel_scale_arcmin)
    gamma1_smooth = gaussian_filter(gamma1_obs, sigma=cfg.smooth_sigma_pix, mode="reflect")
    gamma2_smooth = gaussian_filter(gamma2_obs, sigma=cfg.smooth_sigma_pix, mode="reflect")
    kappa_smooth = kaiser_squires_inversion(gamma1_smooth, gamma2_smooth, cfg.pixel_scale_arcmin)

    # 4) sklearn affine calibration
    reg = LinearRegression().fit(kappa_smooth.reshape(-1, 1), kappa_true.ravel())
    m_skl = float(reg.coef_[0])
    c_skl = float(reg.intercept_)
    kappa_skl = reg.predict(kappa_smooth.reshape(-1, 1)).reshape(kappa_smooth.shape)
    r2_skl = float(r2_score(kappa_true.ravel(), kappa_skl.ravel()))

    # 5) PyTorch affine calibration (same target, gradient-based)
    m_torch, b_torch, fit_rmse = fit_affine_torch(kappa_smooth, kappa_true)
    kappa_torch = m_torch * kappa_smooth + b_torch

    # 6) Quantitative diagnostics
    stages = {
        "raw_KS": kappa_raw,
        "smoothed_KS": kappa_smooth,
        "sklearn_calibrated": kappa_skl,
        "torch_calibrated": kappa_torch,
    }
    metrics_rows = []
    for name, km in stages.items():
        stats = reconstruction_metrics(kappa_true, km)
        stats["stage"] = name
        metrics_rows.append(stats)
    metrics_df = pd.DataFrame(metrics_rows)[["stage", "rmse", "mae", "pearson_corr"]]

    slope_true = radial_power_slope(kappa_true, cfg.pixel_scale_arcmin)
    slope_raw = radial_power_slope(kappa_raw, cfg.pixel_scale_arcmin)
    slope_smooth = radial_power_slope(kappa_smooth, cfg.pixel_scale_arcmin)

    signal_norm = np.sqrt(gamma1_true**2 + gamma2_true**2)
    weak_regime_ok = (np.max(np.abs(kappa_true)) < 0.1) and (np.max(signal_norm) < 0.1)

    map_stats_df = pd.DataFrame(
        [
            {"map": "kappa_true", "mean": np.mean(kappa_true), "std": np.std(kappa_true), "max_abs": np.max(np.abs(kappa_true))},
            {"map": "gamma1_true", "mean": np.mean(gamma1_true), "std": np.std(gamma1_true), "max_abs": np.max(np.abs(gamma1_true))},
            {"map": "gamma2_true", "mean": np.mean(gamma2_true), "std": np.std(gamma2_true), "max_abs": np.max(np.abs(gamma2_true))},
            {"map": "gamma1_obs", "mean": np.mean(gamma1_obs), "std": np.std(gamma1_obs), "max_abs": np.max(np.abs(gamma1_obs))},
            {"map": "gamma2_obs", "mean": np.mean(gamma2_obs), "std": np.std(gamma2_obs), "max_abs": np.max(np.abs(gamma2_obs))},
        ]
    )

    slope_df = pd.DataFrame(
        [
            {"field": "kappa_true", "dlogP_dlogk": slope_true},
            {"field": "kappa_raw", "dlogP_dlogk": slope_raw},
            {"field": "kappa_smooth", "dlogP_dlogk": slope_smooth},
        ]
    )

    print("=== Weak Lensing MVP (PHYS-0364) ===")
    print(
        f"grid={cfg.n_grid}x{cfg.n_grid}, pixel_scale={cfg.pixel_scale_arcmin:.3f} arcmin, "
        f"sigma_e={cfg.sigma_e:.3f}, n_gal/pixel={cfg.n_gal_per_pix:.1f}"
    )
    print(f"shape-noise std per shear component = {noise_std:.6f}")
    print(f"weak-regime check (max|kappa|, max|gamma| < 0.1): {weak_regime_ok}")
    print()

    print("[Map Statistics]")
    print(map_stats_df.to_string(index=False, float_format=lambda x: f"{x: .6e}"))
    print()

    print("[Reconstruction Metrics]")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x: .6e}"))
    print()

    print("[Bias Calibration]")
    print(f"sklearn: kappa_cal = {m_skl:.5f} * kappa_smooth + ({c_skl:.5e}), R2={r2_skl:.6f}")
    print(
        "torch  : "
        f"kappa_cal = {m_torch:.5f} * kappa_smooth + ({b_torch:.5e}), "
        f"fit_RMSE={fit_rmse:.6e}"
    )
    print()

    print("[Radial Power Slope]")
    print(slope_df.to_string(index=False, float_format=lambda x: f"{x: .6e}"))


if __name__ == "__main__":
    main()
