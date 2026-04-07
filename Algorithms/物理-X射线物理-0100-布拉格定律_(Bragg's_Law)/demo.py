"""Bragg's law MVP: generate a synthetic XRD pattern and recover d-spacing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass(frozen=True)
class BraggConfig:
    """Configuration for synthetic Bragg diffraction and inversion."""

    wavelength_angstrom: float = 1.5406  # Cu K-alpha, Angstrom
    d_spacing_angstrom: float = 3.6
    requested_max_order: int = 4
    two_theta_min_deg: float = 10.0
    two_theta_max_deg: float = 125.0
    n_points: int = 5000
    peak_sigma_deg: float = 0.22
    intensity_decay: float = 0.30
    baseline: float = 0.03
    noise_std: float = 0.008
    seed: int = 7
    min_peak_separation_deg: float = 2.0
    min_prominence: float = 0.03

    @property
    def max_physical_order(self) -> int:
        return int(np.floor(2.0 * self.d_spacing_angstrom / self.wavelength_angstrom))

    @property
    def usable_order_count(self) -> int:
        return max(1, min(self.requested_max_order, self.max_physical_order))


def bragg_two_theta_deg(
    order_n: np.ndarray, wavelength_angstrom: float, d_spacing_angstrom: float
) -> np.ndarray:
    """Convert Bragg order n to diffraction angle 2theta (degrees)."""
    ratio = order_n * wavelength_angstrom / (2.0 * d_spacing_angstrom)
    ratio = np.clip(ratio, -1.0, 1.0)
    theta_rad = np.arcsin(ratio)
    return np.rad2deg(2.0 * theta_rad)


def simulate_xrd_pattern(
    config: BraggConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic 1D powder XRD profile with Gaussian peaks."""
    orders = np.arange(1, config.usable_order_count + 1, dtype=int)
    expected_two_theta = bragg_two_theta_deg(
        orders.astype(float), config.wavelength_angstrom, config.d_spacing_angstrom
    )

    two_theta = np.linspace(config.two_theta_min_deg, config.two_theta_max_deg, config.n_points)
    intensity = np.full_like(two_theta, fill_value=config.baseline, dtype=float)
    for n, center in zip(orders, expected_two_theta):
        amplitude = np.exp(-config.intensity_decay * (n - 1))
        intensity += amplitude * np.exp(-0.5 * ((two_theta - center) / config.peak_sigma_deg) ** 2)

    rng = np.random.default_rng(config.seed)
    intensity += rng.normal(loc=0.0, scale=config.noise_std, size=two_theta.size)
    intensity = np.clip(intensity, a_min=0.0, a_max=None)
    return two_theta, intensity, orders, expected_two_theta


def detect_peak_positions(
    config: BraggConfig, two_theta: np.ndarray, intensity: np.ndarray, expected_count: int
) -> np.ndarray:
    """Detect strongest diffraction peaks and return their 2theta positions."""
    step = float(two_theta[1] - two_theta[0])
    min_distance = max(1, int(round(config.min_peak_separation_deg / step)))
    primary_prominence = max(config.min_prominence, 0.8 * float(np.std(intensity)))

    peak_idx, props = find_peaks(intensity, prominence=primary_prominence, distance=min_distance)
    if peak_idx.size < expected_count:
        relaxed_distance = max(1, min_distance // 2)
        peak_idx, props = find_peaks(
            intensity,
            prominence=0.5 * config.min_prominence,
            distance=relaxed_distance,
        )
    if peak_idx.size < expected_count:
        raise RuntimeError(
            f"Expected at least {expected_count} peaks, but detected {peak_idx.size}."
        )

    prominences = props.get("prominences")
    if prominences is None:
        prominences = np.ones_like(peak_idx, dtype=float)

    top_local = np.argsort(prominences)[-expected_count:]
    selected_idx = np.sort(peak_idx[top_local])
    return two_theta[selected_idx]


def estimate_d_spacing_from_peaks(
    orders: np.ndarray, detected_two_theta_deg: np.ndarray, wavelength_angstrom: float
) -> tuple[np.ndarray, float]:
    """Recover per-order d estimates using n*lambda = 2*d*sin(theta)."""
    theta_rad = np.deg2rad(detected_two_theta_deg / 2.0)
    sin_theta = np.sin(theta_rad)
    d_each = orders.astype(float) * wavelength_angstrom / (2.0 * sin_theta)
    return d_each, float(np.mean(d_each))


def fit_bragg_linear_relation(
    orders: np.ndarray, detected_two_theta_deg: np.ndarray, wavelength_angstrom: float
) -> tuple[float, float, float, np.ndarray]:
    """Fit n = m * sin(theta) (through origin), where m = 2d/lambda."""
    theta_rad = np.deg2rad(detected_two_theta_deg / 2.0)
    x = np.sin(theta_rad).reshape(-1, 1)
    y = orders.astype(float)

    model = LinearRegression(fit_intercept=False)
    model.fit(x, y)
    y_pred = model.predict(x)

    slope = float(model.coef_[0])
    d_regression = 0.5 * wavelength_angstrom * slope
    fit_r2 = float(r2_score(y, y_pred))
    fit_mae = float(mean_absolute_error(y, y_pred))
    return d_regression, fit_r2, fit_mae, y_pred


def build_result_table(
    orders: np.ndarray,
    expected_two_theta_deg: np.ndarray,
    detected_two_theta_deg: np.ndarray,
    d_each: np.ndarray,
    d_reference: float,
) -> pd.DataFrame:
    """Build a readable per-peak report table."""
    theta_rad = np.deg2rad(detected_two_theta_deg / 2.0)
    table = pd.DataFrame(
        {
            "order_n": orders,
            "two_theta_expected_deg": expected_two_theta_deg,
            "two_theta_detected_deg": detected_two_theta_deg,
            "theta_detected_deg": detected_two_theta_deg / 2.0,
            "sin_theta": np.sin(theta_rad),
            "d_estimated_angstrom": d_each,
            "d_error_pct": (d_each - d_reference) / d_reference * 100.0,
        }
    )
    return table


def evaluate_with_torch(d_each: np.ndarray, d_true: float) -> tuple[float, float]:
    """Use torch for independent tensor-based error statistics."""
    d_tensor = torch.tensor(d_each, dtype=torch.float64)
    d_true_tensor = torch.tensor(d_true, dtype=torch.float64)
    rel_l1 = torch.mean(torch.abs((d_tensor - d_true_tensor) / d_true_tensor)).item()
    rmse = torch.sqrt(torch.mean((d_tensor - d_true_tensor) ** 2)).item()
    return float(rel_l1), float(rmse)


def main() -> None:
    config = BraggConfig()

    two_theta, intensity, orders, expected_two_theta = simulate_xrd_pattern(config)
    detected_two_theta = detect_peak_positions(
        config=config,
        two_theta=two_theta,
        intensity=intensity,
        expected_count=orders.size,
    )

    d_each, d_mean = estimate_d_spacing_from_peaks(
        orders=orders,
        detected_two_theta_deg=detected_two_theta,
        wavelength_angstrom=config.wavelength_angstrom,
    )
    d_regression, fit_r2, fit_mae, order_pred = fit_bragg_linear_relation(
        orders=orders,
        detected_two_theta_deg=detected_two_theta,
        wavelength_angstrom=config.wavelength_angstrom,
    )
    torch_rel_l1, torch_rmse = evaluate_with_torch(d_each=d_each, d_true=config.d_spacing_angstrom)

    theta_rad = np.deg2rad(detected_two_theta / 2.0)
    bragg_residual = (
        orders.astype(float) * config.wavelength_angstrom
        - 2.0 * d_mean * np.sin(theta_rad)
    )
    max_abs_bragg_residual = float(np.max(np.abs(bragg_residual)))
    rel_error_mean = abs(d_mean - config.d_spacing_angstrom) / config.d_spacing_angstrom
    rel_error_reg = abs(d_regression - config.d_spacing_angstrom) / config.d_spacing_angstrom

    table = build_result_table(
        orders=orders,
        expected_two_theta_deg=expected_two_theta,
        detected_two_theta_deg=detected_two_theta,
        d_each=d_each,
        d_reference=config.d_spacing_angstrom,
    )

    summary = pd.DataFrame(
        [
            {
                "d_true_angstrom": config.d_spacing_angstrom,
                "d_mean_angstrom": d_mean,
                "d_reg_angstrom": d_regression,
                "d_mean_rel_error": rel_error_mean,
                "d_reg_rel_error": rel_error_reg,
                "fit_r2_n_vs_sin_theta": fit_r2,
                "fit_mae_n_vs_sin_theta": fit_mae,
                "max_abs_bragg_residual": max_abs_bragg_residual,
                "torch_rel_l1": torch_rel_l1,
                "torch_rmse_angstrom": torch_rmse,
            }
        ]
    )

    print("=== Bragg's Law MVP (Synthetic XRD) ===")
    print(
        f"wavelength={config.wavelength_angstrom:.4f} A, "
        f"d_true={config.d_spacing_angstrom:.4f} A, orders={orders.tolist()}"
    )
    print("\nDetected peak table:")
    print(table.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    print("\nLinear fit check (n vs sin(theta)):")
    for n_i, pred_i in zip(orders, order_pred):
        print(f"  n={n_i:>2d}, predicted={pred_i: .6f}, abs_err={abs(pred_i-n_i): .6f}")
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x: .6e}"))

    checks = {
        "d_mean_rel_error < 0.5%": rel_error_mean < 5e-3,
        "d_reg_rel_error < 0.5%": rel_error_reg < 5e-3,
        "fit_r2 > 0.999": fit_r2 > 0.999,
        "max_abs_bragg_residual < 0.015 A": max_abs_bragg_residual < 0.015,
        "torch_rel_l1 < 0.5%": torch_rel_l1 < 5e-3,
    }

    print("\nValidation checks:")
    for key, ok in checks.items():
        print(f"  {key}: {'PASS' if ok else 'FAIL'}")
    overall_pass = all(checks.values())
    print(f"\nValidation: {'PASS' if overall_pass else 'FAIL'}")
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
