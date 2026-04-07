"""MVP: Electrodynamics of superconductors.

This script demonstrates:
1) Meissner magnetic-field decay and penetration-depth fitting.
2) Two-fluid complex conductivity versus temperature.
3) Surface impedance computed from complex conductivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.constants import e, m_e, mu_0, pi
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def meissner_profile(x: np.ndarray, b0: float, lambda_l: float) -> np.ndarray:
    """Magnetic field profile in a 1D semi-infinite superconductor."""
    return b0 * np.exp(-x / lambda_l)


def fit_penetration_depth(x: np.ndarray, b_obs: np.ndarray, b0: float) -> tuple[float, float]:
    """Fit lambda_L from noisy B(x) data with bounded nonlinear least squares."""

    def model(x_val: np.ndarray, lambda_l: float) -> np.ndarray:
        return meissner_profile(x_val, b0=b0, lambda_l=lambda_l)

    params, covariance = curve_fit(
        model,
        x,
        b_obs,
        p0=[80e-9],
        bounds=(1e-9, 1e-5),
        maxfev=20000,
    )
    lambda_fit = float(params[0])
    if covariance.size == 0:
        lambda_std = float("nan")
    else:
        lambda_std = float(np.sqrt(np.diag(covariance))[0])
    return lambda_fit, lambda_std


def superfluid_fraction(t: np.ndarray, tc: float) -> np.ndarray:
    """Simple two-fluid model: f_s = 1 - (T/Tc)^4 for T < Tc."""
    return np.clip(1.0 - (t / tc) ** 4, 0.0, 1.0)


def two_fluid_conductivity(
    t: np.ndarray,
    omega: float,
    tc: float,
    n_total: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sigma1, sigma2, f_s, f_n."""
    f_s = superfluid_fraction(t, tc)
    f_n = 1.0 - f_s
    prefactor = n_total * e * e / m_e

    sigma1 = f_n * prefactor * tau / (1.0 + (omega * tau) ** 2)
    sigma2 = f_s * prefactor / omega
    return sigma1, sigma2, f_s, f_n


def surface_impedance(sigma1: np.ndarray, sigma2: np.ndarray, omega: float) -> np.ndarray:
    """Compute Zs = sqrt(i * omega * mu0 / (sigma1 - i*sigma2))."""
    sigma_tilde = sigma1 - 1j * sigma2
    return np.sqrt(1j * omega * mu_0 / sigma_tilde)


def main() -> None:
    rng = np.random.default_rng(seed=20260407)

    # Synthetic Meissner measurement (Nb-like scale).
    lambda_true = 85e-9  # m
    b0 = 1.2e-2  # T
    x = np.linspace(0.0, 600e-9, 80)
    b_clean = meissner_profile(x, b0=b0, lambda_l=lambda_true)
    b_obs = b_clean + rng.normal(loc=0.0, scale=1.5e-4, size=x.size)

    lambda_fit, lambda_std = fit_penetration_depth(x, b_obs, b0=b0)
    b_fit = meissner_profile(x, b0=b0, lambda_l=lambda_fit)
    fit_r2 = r2_score(b_obs, b_fit)

    # Temperature sweep electrodynamics.
    tc = 9.2  # K
    t = np.linspace(2.0, 9.5, 8)
    f = 10e9  # Hz
    omega = 2.0 * pi * f
    n_total = 8.5e28  # 1/m^3
    tau = 2.0e-14  # s

    sigma1, sigma2, f_s, f_n = two_fluid_conductivity(t, omega, tc, n_total, tau)
    zs = surface_impedance(sigma1, sigma2, omega)

    # Effective penetration depth from superfluid fraction (for intuition).
    f_s_safe = np.maximum(f_s, 1e-12)
    lambda_eff = lambda_true / np.sqrt(f_s_safe)

    df = pd.DataFrame(
        {
            "T_K": t,
            "f_s": f_s,
            "f_n": f_n,
            "lambda_eff_nm": lambda_eff * 1e9,
            "sigma1_S_per_m": sigma1,
            "sigma2_S_per_m": sigma2,
            "Re_Zs_mOhm": np.real(zs) * 1e3,
            "Im_Zs_mOhm": np.imag(zs) * 1e3,
        }
    )

    print("=== Superconductor Electrodynamics MVP ===")
    print(f"True lambda_L: {lambda_true * 1e9:.3f} nm")
    print(f"Fitted lambda_L: {lambda_fit * 1e9:.3f} nm")
    print(f"Std(lambda_L): {lambda_std * 1e9:.3f} nm")
    print(f"Meissner fit R^2: {fit_r2:.6f}")
    print("\nTemperature sweep results:")
    print(df.to_string(index=False, justify='center', float_format=lambda v: f"{v:.6e}"))


if __name__ == "__main__":
    main()
