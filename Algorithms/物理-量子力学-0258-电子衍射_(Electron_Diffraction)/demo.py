"""Electron diffraction MVP: forward simulation + inverse estimation.

This script models electron diffraction from a finite 1D crystal plane array:
- Forward model:
  1) Relativistic de Broglie wavelength from accelerating voltage
  2) Finite-array interference intensity on an angle grid
  3) Synthetic noisy observation
- Inverse model:
  1) Peak picking + linear regression (sin(theta_m) = m * lambda / d)
  2) Torch-based waveform fitting for lambda refinement
  3) Convert estimated lambda back to accelerating voltage

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

# Physical constants (SI).
H_PLANCK = 6.626_070_15e-34
M_E = 9.109_383_7015e-31
E_CHARGE = 1.602_176_634e-19
C_LIGHT = 2.997_924_58e8


@dataclass
class ElectronDiffractionResult:
    voltage_true_v: float
    lambda_true_m: float
    d_spacing_m: float
    n_planes: int
    theta_deg: np.ndarray
    intensity_clean: np.ndarray
    intensity_noisy: np.ndarray
    peak_table: pd.DataFrame
    lambda_est_regression_m: float
    lambda_est_torch_m: float
    voltage_est_regression_v: float
    voltage_est_torch_v: float
    regression_r2: float
    torch_loss: float


def electron_wavelength_relativistic(voltage_v: float) -> float:
    """Relativistic de Broglie wavelength for electrons accelerated by V."""
    if voltage_v <= 0.0:
        raise ValueError("voltage_v must be positive.")
    kinetic = E_CHARGE * voltage_v
    momentum = np.sqrt(2.0 * M_E * kinetic * (1.0 + kinetic / (2.0 * M_E * C_LIGHT**2)))
    return float(H_PLANCK / momentum)


def finite_array_intensity(
    theta_rad: np.ndarray,
    wavelength_m: float,
    d_spacing_m: float,
    n_planes: int,
    envelope_sigma_deg: float = 26.0,
) -> np.ndarray:
    """Finite-plane interference intensity with a smooth angular envelope."""
    if wavelength_m <= 0.0 or d_spacing_m <= 0.0:
        raise ValueError("wavelength_m and d_spacing_m must be positive.")
    if n_planes < 2:
        raise ValueError("n_planes must be >= 2.")
    if theta_rad.ndim != 1:
        raise ValueError("theta_rad must be a 1D array.")

    phase = 2.0 * np.pi * d_spacing_m * np.sin(theta_rad) / wavelength_m
    half = 0.5 * phase
    numerator = np.sin(n_planes * half)
    denominator = np.sin(half)

    ratio = np.empty_like(theta_rad, dtype=float)
    near_zero = np.abs(denominator) < 1e-12
    ratio[near_zero] = float(n_planes)
    ratio[~near_zero] = numerator[~near_zero] / denominator[~near_zero]

    interference = (ratio / float(n_planes)) ** 2
    envelope = np.exp(-(np.rad2deg(theta_rad) / envelope_sigma_deg) ** 2)
    intensity = interference * envelope

    intensity = np.clip(intensity, 0.0, None)
    max_val = float(np.max(intensity))
    if max_val <= 0.0:
        raise ValueError("Non-positive intensity encountered.")
    intensity /= max_val
    return intensity


def simulate_observation(
    theta_rad: np.ndarray,
    wavelength_m: float,
    d_spacing_m: float,
    n_planes: int,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate clean and noisy normalized diffraction patterns."""
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")
    clean = finite_array_intensity(
        theta_rad=theta_rad,
        wavelength_m=wavelength_m,
        d_spacing_m=d_spacing_m,
        n_planes=n_planes,
    )
    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(loc=0.0, scale=noise_std, size=clean.shape)
    noisy = np.clip(noisy, 0.0, None)
    noisy /= float(np.max(noisy) + 1e-15)
    return clean, noisy


def detect_positive_peaks(
    theta_deg: np.ndarray,
    intensity: np.ndarray,
    max_order: int = 3,
    min_theta_deg: float = 5.0,
) -> pd.DataFrame:
    """Detect diffraction peaks at theta>0 and map them to order m=1..max_order."""
    if theta_deg.shape != intensity.shape or theta_deg.ndim != 1:
        raise ValueError("theta_deg and intensity must be same-shape 1D arrays.")
    if max_order < 1:
        raise ValueError("max_order must be >= 1.")

    positive_mask = theta_deg > min_theta_deg
    theta_pos = theta_deg[positive_mask]
    intensity_pos = intensity[positive_mask]

    peak_idx, props = find_peaks(
        intensity_pos,
        height=0.01,
        prominence=0.01,
        distance=50,
    )
    if peak_idx.size < max_order:
        raise RuntimeError(
            f"Only detected {peak_idx.size} positive-side peaks; need at least {max_order}."
        )

    # Keep the most prominent peaks (principal diffraction maxima),
    # then sort by angle and map to orders m=1..max_order.
    prominence = props["prominences"]
    top_by_prominence = np.argsort(prominence)[-max_order:]
    selected = peak_idx[top_by_prominence]
    selected = selected[np.argsort(theta_pos[selected])]
    selected_theta_deg = theta_pos[selected]
    selected_theta_rad = np.deg2rad(selected_theta_deg)
    selected_height = intensity_pos[selected]

    orders = np.arange(1, max_order + 1, dtype=int)
    table = pd.DataFrame(
        {
            "order_m": orders,
            "theta_deg": selected_theta_deg,
            "sin_theta": np.sin(selected_theta_rad),
            "peak_intensity": selected_height,
        }
    )
    return table


def estimate_wavelength_from_peaks(
    peak_table: pd.DataFrame,
    d_spacing_m: float,
) -> tuple[float, float]:
    """Use sin(theta_m)=m*lambda/d linear fit through origin to estimate lambda."""
    if d_spacing_m <= 0.0:
        raise ValueError("d_spacing_m must be positive.")
    required_cols = {"order_m", "sin_theta"}
    if not required_cols.issubset(set(peak_table.columns)):
        raise ValueError(f"peak_table must contain columns: {required_cols}")

    m = peak_table["order_m"].to_numpy(dtype=float).reshape(-1, 1)
    y = peak_table["sin_theta"].to_numpy(dtype=float)

    model = LinearRegression(fit_intercept=False)
    model.fit(m, y)
    slope = float(model.coef_[0])  # slope ~= lambda / d
    y_hat = model.predict(m)

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-15)

    lambda_est = slope * d_spacing_m
    return float(lambda_est), r2


def voltage_from_wavelength(target_lambda_m: float) -> float:
    """Invert relativistic de Broglie relation via scalar root finding."""
    if target_lambda_m <= 0.0:
        raise ValueError("target_lambda_m must be positive.")

    def objective(v: float) -> float:
        return electron_wavelength_relativistic(v) - target_lambda_m

    # Wide bracket for practical electron diffraction ranges.
    v_min, v_max = 1.0, 5.0e5
    f_min = objective(v_min)
    f_max = objective(v_max)
    if f_min * f_max > 0.0:
        raise RuntimeError("Failed to bracket voltage root for target wavelength.")
    return float(brentq(objective, v_min, v_max, xtol=1e-12, rtol=1e-10, maxiter=200))


def finite_array_intensity_torch(
    theta_rad: torch.Tensor,
    wavelength_m: torch.Tensor,
    d_spacing_m: float,
    n_planes: int,
    envelope_sigma_deg: float = 26.0,
) -> torch.Tensor:
    """Torch version of finite-array intensity, used for fitting lambda."""
    phase = 2.0 * torch.pi * d_spacing_m * torch.sin(theta_rad) / wavelength_m
    half = 0.5 * phase
    numerator = torch.sin(float(n_planes) * half)
    denominator = torch.sin(half)

    ratio_raw = numerator / (denominator + 1e-12)
    ratio = torch.where(
        torch.abs(denominator) < 1e-8,
        torch.full_like(ratio_raw, float(n_planes)),
        ratio_raw,
    )

    interference = (ratio / float(n_planes)) ** 2
    envelope = torch.exp(-(torch.rad2deg(theta_rad) / envelope_sigma_deg) ** 2)
    intensity = torch.clamp(interference * envelope, min=0.0)
    intensity = intensity / (torch.max(intensity) + 1e-15)
    return intensity


def refine_wavelength_with_torch(
    theta_rad: np.ndarray,
    observed_intensity: np.ndarray,
    d_spacing_m: float,
    n_planes: int,
    init_lambda_m: float,
    steps: int = 700,
    lr: float = 4e-4,
) -> tuple[float, float]:
    """Fit lambda by minimizing MSE between observed and modeled diffraction curves."""
    if init_lambda_m <= 0.0:
        raise ValueError("init_lambda_m must be positive.")

    theta_t = torch.tensor(theta_rad, dtype=torch.float64)
    target_t = torch.tensor(observed_intensity, dtype=torch.float64)

    # Softplus parameterization to enforce positive wavelength.
    init_raw = np.log(np.exp(init_lambda_m) - 1.0)
    raw = torch.tensor(init_raw, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        lambda_t = torch.nn.functional.softplus(raw) + 1e-15
        pred_t = finite_array_intensity_torch(
            theta_rad=theta_t,
            wavelength_m=lambda_t,
            d_spacing_m=d_spacing_m,
            n_planes=n_planes,
        )
        loss = torch.mean((pred_t - target_t) ** 2)
        loss.backward()
        optimizer.step()

    lambda_est = float((torch.nn.functional.softplus(raw) + 1e-15).item())
    final_loss = float(loss.item())
    return lambda_est, final_loss


def run_electron_diffraction_mvp() -> ElectronDiffractionResult:
    """Full pipeline: simulate pattern, estimate lambda, recover voltage."""
    voltage_true_v = 3_000.0
    d_spacing_m = 0.123e-9
    n_planes = 8
    theta_deg = np.linspace(-45.0, 45.0, 3601, dtype=float)
    theta_rad = np.deg2rad(theta_deg)

    lambda_true_m = electron_wavelength_relativistic(voltage_true_v)
    intensity_clean, intensity_noisy = simulate_observation(
        theta_rad=theta_rad,
        wavelength_m=lambda_true_m,
        d_spacing_m=d_spacing_m,
        n_planes=n_planes,
        noise_std=0.02,
        seed=20260407,
    )

    peak_table = detect_positive_peaks(theta_deg=theta_deg, intensity=intensity_noisy, max_order=3)
    lambda_est_regression_m, regression_r2 = estimate_wavelength_from_peaks(
        peak_table=peak_table,
        d_spacing_m=d_spacing_m,
    )
    voltage_est_regression_v = voltage_from_wavelength(lambda_est_regression_m)

    lambda_est_torch_m, torch_loss = refine_wavelength_with_torch(
        theta_rad=theta_rad,
        observed_intensity=intensity_noisy,
        d_spacing_m=d_spacing_m,
        n_planes=n_planes,
        init_lambda_m=lambda_est_regression_m,
    )
    voltage_est_torch_v = voltage_from_wavelength(lambda_est_torch_m)

    return ElectronDiffractionResult(
        voltage_true_v=voltage_true_v,
        lambda_true_m=lambda_true_m,
        d_spacing_m=d_spacing_m,
        n_planes=n_planes,
        theta_deg=theta_deg,
        intensity_clean=intensity_clean,
        intensity_noisy=intensity_noisy,
        peak_table=peak_table,
        lambda_est_regression_m=lambda_est_regression_m,
        lambda_est_torch_m=lambda_est_torch_m,
        voltage_est_regression_v=voltage_est_regression_v,
        voltage_est_torch_v=voltage_est_torch_v,
        regression_r2=regression_r2,
        torch_loss=torch_loss,
    )


def run_checks(result: ElectronDiffractionResult) -> None:
    """Sanity checks for the MVP output."""
    rel_err_reg = abs(result.lambda_est_regression_m - result.lambda_true_m) / result.lambda_true_m
    rel_err_torch = abs(result.lambda_est_torch_m - result.lambda_true_m) / result.lambda_true_m

    if rel_err_reg > 0.20:
        raise AssertionError(f"Regression wavelength relative error too large: {rel_err_reg:.3f}")
    if rel_err_torch > 0.20:
        raise AssertionError(f"Torch wavelength relative error too large: {rel_err_torch:.3f}")
    if result.regression_r2 < 0.995:
        raise AssertionError(f"Peak regression R^2 too low: {result.regression_r2:.6f}")
    if not np.all(np.diff(result.peak_table["theta_deg"].to_numpy(dtype=float)) > 0.0):
        raise AssertionError("Detected peak angles are not strictly increasing.")
    if result.torch_loss > 0.03:
        raise AssertionError(f"Torch fit loss unexpectedly high: {result.torch_loss:.6f}")


def main() -> None:
    result = run_electron_diffraction_mvp()
    run_checks(result)

    pm = 1e12  # meter -> picometer
    print("Electron Diffraction MVP (finite crystal interference + inverse estimation)")
    print(f"True accelerating voltage: {result.voltage_true_v:.2f} V")
    print(f"Assumed plane spacing d: {result.d_spacing_m * 1e10:.4f} Angstrom")
    print(f"Finite plane count N: {result.n_planes}")
    print()
    print(f"True wavelength:         {result.lambda_true_m * pm:.6f} pm")
    print(f"Regression wavelength:   {result.lambda_est_regression_m * pm:.6f} pm")
    print(f"Torch-fit wavelength:    {result.lambda_est_torch_m * pm:.6f} pm")
    print(f"Regression-estimated V:  {result.voltage_est_regression_v:.3f} V")
    print(f"Torch-fit-estimated V:   {result.voltage_est_torch_v:.3f} V")
    print(f"Peak-fit R^2:            {result.regression_r2:.8f}")
    print(f"Torch curve-fit MSE:     {result.torch_loss:.8e}")
    print()
    print("Detected positive-side diffraction peaks:")
    print(result.peak_table.to_string(index=False, float_format=lambda x: f"{x:.8f}"))


if __name__ == "__main__":
    main()
