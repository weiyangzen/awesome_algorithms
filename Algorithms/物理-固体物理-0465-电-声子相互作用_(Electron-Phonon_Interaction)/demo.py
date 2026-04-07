"""Electron-phonon interaction MVP in normal state.

This script builds a normalized Eliashberg spectral function alpha^2F(omega),
computes the electron self-energy ImSigma/ReSigma on a real-frequency grid,
and reports coupling/scattering diagnostics across temperature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.integrate import simpson
from sklearn.linear_model import LinearRegression

K_B_EV = 8.617333262e-5  # eV / K


@dataclass(frozen=True)
class ElectronPhononParams:
    """Physical and numerical settings for the MVP."""

    lambda_target: float = 0.85
    omega0_eV: float = 0.020
    sigma_eV: float = 0.004
    omega_ph_max_eV: float = 0.100
    n_omega_ph: int = 1201
    omega_el_max_eV: float = 0.250
    n_omega_el: int = 801
    fit_window_eV: float = 0.008
    temperatures_K: tuple[float, ...] = (20.0, 100.0, 300.0)


def check_params(params: ElectronPhononParams) -> None:
    """Validate model parameters."""

    if params.lambda_target <= 0:
        raise ValueError("lambda_target must be positive")
    if params.omega0_eV <= 0 or params.sigma_eV <= 0:
        raise ValueError("omega0_eV and sigma_eV must be positive")
    if params.omega_ph_max_eV <= params.omega0_eV:
        raise ValueError("omega_ph_max_eV must be larger than omega0_eV")
    if params.omega_el_max_eV <= 0 or params.fit_window_eV <= 0:
        raise ValueError("omega_el_max_eV and fit_window_eV must be positive")
    if params.n_omega_ph < 101 or params.n_omega_el < 101:
        raise ValueError("frequency grids are too coarse")
    if params.n_omega_el % 2 == 0:
        raise ValueError("n_omega_el must be odd so that omega=0 is included")
    if any(t <= 0 for t in params.temperatures_K):
        raise ValueError("all temperatures must be positive")


def fermi_dirac(energy_eV: np.ndarray, temperature_K: float) -> np.ndarray:
    """Stable Fermi-Dirac distribution f(E,T)."""

    kbt = K_B_EV * temperature_K
    x = np.clip(energy_eV / kbt, -80.0, 80.0)
    return 1.0 / (np.exp(x) + 1.0)


def bose_einstein(energy_eV: np.ndarray, temperature_K: float) -> np.ndarray:
    """Stable Bose-Einstein distribution n_B(E,T) for E>0."""

    kbt = K_B_EV * temperature_K
    x = np.clip(energy_eV / kbt, 1e-12, 80.0)
    return 1.0 / np.expm1(x)


def build_frequency_grids(params: ElectronPhononParams) -> tuple[np.ndarray, np.ndarray]:
    """Construct electron and phonon frequency grids."""

    omega_el = np.linspace(-params.omega_el_max_eV, params.omega_el_max_eV, params.n_omega_el)
    omega_ph = np.linspace(1e-5, params.omega_ph_max_eV, params.n_omega_ph)
    return omega_el, omega_ph


def normalized_alpha2f(params: ElectronPhononParams, omega_ph: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Build alpha^2F(omega) and normalize it so lambda matches lambda_target.

    We enforce:
        lambda = 2 * integral_0^inf [alpha^2F(omega) / omega] d omega.
    """

    g = np.exp(-0.5 * ((omega_ph - params.omega0_eV) / params.sigma_eV) ** 2)
    denom = 2.0 * simpson(g / omega_ph, x=omega_ph)
    amplitude = params.lambda_target / denom
    alpha2f = amplitude * g
    lambda_numeric = 2.0 * simpson(alpha2f / omega_ph, x=omega_ph)
    return alpha2f, float(lambda_numeric), float(amplitude)


def imag_self_energy(
    omega_el: np.ndarray,
    omega_ph: np.ndarray,
    alpha2f: np.ndarray,
    temperature_K: float,
) -> np.ndarray:
    """Compute Im Sigma(omega,T) using a normal-state Eliashberg kernel."""

    w = omega_el[:, None]
    ph = omega_ph[None, :]

    n_b = bose_einstein(omega_ph, temperature_K)[None, :]
    f_minus = fermi_dirac(ph - w, temperature_K)
    f_plus = fermi_dirac(ph + w, temperature_K)

    kernel = 2.0 * n_b + f_minus + f_plus
    integrand = alpha2f[None, :] * kernel
    return -np.pi * simpson(integrand, x=omega_ph, axis=1)


def real_self_energy_kramers_kronig(omega_el: np.ndarray, imag_sigma: np.ndarray) -> np.ndarray:
    """Discrete principal-value Kramers-Kronig transform."""

    dw = omega_el[1] - omega_el[0]
    weights = np.full_like(omega_el, dw)
    weights[0] = 0.5 * dw
    weights[-1] = 0.5 * dw

    diff = omega_el[None, :] - omega_el[:, None]
    kernel = np.divide(
        1.0,
        diff,
        out=np.zeros_like(diff),
        where=~np.eye(omega_el.size, dtype=bool),
    )

    return (1.0 / np.pi) * (kernel @ (imag_sigma * weights))


def estimate_lambda_from_slope(
    omega_el: np.ndarray,
    real_sigma: np.ndarray,
    fit_window_eV: float,
) -> tuple[float, float]:
    """Estimate lambda from slope: lambda = -d ReSigma / d omega |_{omega->0}."""

    mask = np.abs(omega_el) <= fit_window_eV
    x = omega_el[mask].reshape(-1, 1)
    y = real_sigma[mask]

    model = LinearRegression()
    model.fit(x, y)
    slope = float(model.coef_[0])
    r2 = float(model.score(x, y))
    return -slope, r2


def gamma_at_fermi_torch(omega_ph: np.ndarray, alpha2f: np.ndarray, temperature_K: float) -> float:
    """Compute Gamma(E_F,T) with torch as a cross-check path."""

    omega_t = torch.tensor(omega_ph, dtype=torch.float64)
    alpha_t = torch.tensor(alpha2f, dtype=torch.float64)

    kbt = K_B_EV * temperature_K
    y = torch.clamp(omega_t / kbt, min=1e-12, max=80.0)

    n_b = 1.0 / torch.expm1(y)
    f = 1.0 / (torch.exp(torch.clamp(omega_t / kbt, min=-80.0, max=80.0)) + 1.0)

    kernel = 2.0 * n_b + 2.0 * f
    imag_sigma_0 = -np.pi * torch.trapz(alpha_t * kernel, omega_t).item()
    return float(-2.0 * imag_sigma_0)


def build_temperature_table(
    params: ElectronPhononParams,
    omega_el: np.ndarray,
    omega_ph: np.ndarray,
    alpha2f: np.ndarray,
) -> tuple[pd.DataFrame, dict[float, np.ndarray], dict[float, np.ndarray]]:
    """Compute per-temperature diagnostics and spectra."""

    idx0 = int(np.argmin(np.abs(omega_el)))
    rows: list[dict[str, float]] = []
    imag_map: dict[float, np.ndarray] = {}
    real_map: dict[float, np.ndarray] = {}

    for temp in params.temperatures_K:
        imag_sigma = imag_self_energy(omega_el, omega_ph, alpha2f, temp)
        real_sigma = real_self_energy_kramers_kronig(omega_el, imag_sigma)

        lambda_slope, slope_r2 = estimate_lambda_from_slope(omega_el, real_sigma, params.fit_window_eV)
        gamma_ef_np = -2.0 * float(imag_sigma[idx0])
        gamma_ef_torch = gamma_at_fermi_torch(omega_ph, alpha2f, temp)
        torch_rel_err = abs(gamma_ef_torch - gamma_ef_np) / max(1e-12, abs(gamma_ef_np))

        rows.append(
            {
                "temperature_K": float(temp),
                "lambda_from_slope": float(lambda_slope),
                "slope_fit_r2": float(slope_r2),
                "gamma_EF_meV": float(gamma_ef_np * 1e3),
                "torch_gamma_rel_error": float(torch_rel_err),
                "max_abs_ReSigma_meV": float(np.max(np.abs(real_sigma)) * 1e3),
            }
        )

        imag_map[temp] = imag_sigma
        real_map[temp] = real_sigma

    return pd.DataFrame(rows), imag_map, real_map


def build_summary_table(
    params: ElectronPhononParams,
    lambda_numeric: float,
    amplitude: float,
    temperature_table: pd.DataFrame,
) -> pd.DataFrame:
    """Collect headline metrics for quick validation."""

    low_t_row = temperature_table.iloc[0]
    high_t_row = temperature_table.iloc[-1]

    lambda_low_t = float(low_t_row["lambda_from_slope"])
    gamma_ratio = float(high_t_row["gamma_EF_meV"]) / max(1e-12, float(low_t_row["gamma_EF_meV"]))

    return pd.DataFrame(
        [
            {"metric": "lambda_target", "value": f"{params.lambda_target:.6f}"},
            {"metric": "lambda_from_alpha2F", "value": f"{lambda_numeric:.6f}"},
            {"metric": "alpha2F_amplitude", "value": f"{amplitude:.6e}"},
            {"metric": "lambda_from_slope_lowT", "value": f"{lambda_low_t:.6f}"},
            {
                "metric": "lambda_slope_rel_error",
                "value": f"{abs(lambda_low_t - params.lambda_target) / params.lambda_target:.3e}",
            },
            {"metric": "mstar_over_m_lowT", "value": f"{1.0 + lambda_low_t:.6f}"},
            {
                "metric": "gamma_ratio_highT_over_lowT",
                "value": f"{gamma_ratio:.6f}",
            },
            {
                "metric": "max_torch_gamma_rel_error",
                "value": f"{temperature_table['torch_gamma_rel_error'].max():.3e}",
            },
        ]
    )


def main() -> None:
    params = ElectronPhononParams()
    check_params(params)

    omega_el, omega_ph = build_frequency_grids(params)
    alpha2f, lambda_numeric, amplitude = normalized_alpha2f(params, omega_ph)

    temp_table, _imag_map, _real_map = build_temperature_table(params, omega_el, omega_ph, alpha2f)
    summary = build_summary_table(params, lambda_numeric, amplitude, temp_table)

    print("=== Electron-Phonon Interaction MVP (Normal State) ===")
    print(
        "params:",
        {
            "lambda_target": params.lambda_target,
            "omega0_eV": params.omega0_eV,
            "sigma_eV": params.sigma_eV,
            "omega_ph_max_eV": params.omega_ph_max_eV,
            "omega_el_max_eV": params.omega_el_max_eV,
            "n_omega_ph": params.n_omega_ph,
            "n_omega_el": params.n_omega_el,
            "fit_window_eV": params.fit_window_eV,
            "temperatures_K": params.temperatures_K,
        },
    )
    print("\n[summary]")
    print(summary.to_string(index=False))
    print("\n[temperature_scan]")
    print(temp_table.to_string(index=False))

    # Minimal quality gates for automated validation.
    if abs(lambda_numeric - params.lambda_target) > 5e-3:
        raise AssertionError("alpha^2F normalization failed to reproduce target lambda")

    low_t_lambda = float(temp_table.iloc[0]["lambda_from_slope"])
    if abs(low_t_lambda - params.lambda_target) > 0.20:
        raise AssertionError("low-temperature lambda estimate is far from target")

    gammas = temp_table["gamma_EF_meV"].to_numpy()
    if np.any(np.diff(gammas) <= 0.0):
        raise AssertionError("Gamma(E_F,T) should increase with temperature in this setup")

    if float(temp_table["torch_gamma_rel_error"].max()) > 2e-2:
        raise AssertionError("torch/numpy gamma cross-check mismatch is too large")


if __name__ == "__main__":
    main()
