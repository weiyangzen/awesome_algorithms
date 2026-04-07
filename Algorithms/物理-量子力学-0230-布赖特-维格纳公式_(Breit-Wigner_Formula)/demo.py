"""Breit-Wigner Formula MVP.

This script builds a minimal, auditable pipeline:
1) generate synthetic resonance data,
2) fit Breit-Wigner parameters with weighted nonlinear least squares,
3) report recovery quality and diagnostics.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score


@dataclass(frozen=True)
class ResonanceParams:
    """Container for Breit-Wigner model parameters."""

    e_r: float
    gamma: float
    amplitude: float
    background: float


def breit_wigner_cross_section(
    energy: np.ndarray,
    e_r: float,
    gamma: float,
    amplitude: float,
    background: float,
) -> np.ndarray:
    """Lorentzian-form Breit-Wigner cross section.

    sigma(E) = background + amplitude * (Gamma/2)^2 / ((E - E_r)^2 + (Gamma/2)^2)
    """
    g = max(float(gamma), 1e-12)
    half_width_sq = (0.5 * g) ** 2
    denom = (energy - e_r) ** 2 + half_width_sq
    return background + amplitude * half_width_sq / denom


def build_synthetic_dataset(
    rng: np.random.Generator,
    n_points: int = 260,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ResonanceParams]:
    """Generate noisy resonance data for demonstration."""
    energy = np.linspace(0.6, 1.4, n_points)
    true_params = ResonanceParams(
        e_r=1.01,
        gamma=0.085,
        amplitude=8.5,
        background=0.45,
    )

    clean = breit_wigner_cross_section(
        energy,
        true_params.e_r,
        true_params.gamma,
        true_params.amplitude,
        true_params.background,
    )

    noise_std = 0.05 + 0.015 * np.sqrt(np.maximum(clean, 0.0))
    noisy = clean + rng.normal(0.0, noise_std, size=energy.size)
    return energy, noisy, clean, noise_std, true_params


def residual_vector(
    theta: np.ndarray,
    energy: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Weighted residuals used by least squares."""
    model = breit_wigner_cross_section(energy, *theta)
    return (model - observed) / sigma


def residual_jacobian(
    theta: np.ndarray,
    energy: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Analytical Jacobian of weighted residuals wrt [e_r, gamma, amplitude, background].

    The `observed` argument is unused here but kept for SciPy arg signature compatibility."""
    e_r, gamma, amplitude, _background = theta
    g = max(float(gamma), 1e-12)

    dx = energy - e_r
    g2 = (0.5 * g) ** 2
    denom = dx**2 + g2
    denom_sq = denom**2

    lorentz = g2 / denom
    d_lorentz_de = 2.0 * g2 * dx / denom_sq
    d_lorentz_dg = 0.5 * g * (dx**2) / denom_sq

    jac = np.empty((energy.size, 4), dtype=float)
    jac[:, 0] = amplitude * d_lorentz_de
    jac[:, 1] = amplitude * d_lorentz_dg
    jac[:, 2] = lorentz
    jac[:, 3] = 1.0

    return jac / sigma[:, None]


def initial_guess(energy: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Construct a simple deterministic starting point."""
    idx_peak = int(np.argmax(observed))
    e0 = float(energy[idx_peak])
    background0 = float(np.percentile(observed, 15))
    amplitude0 = max(float(observed[idx_peak] - background0), 1e-3)
    gamma0 = 0.12 * float(energy.max() - energy.min())
    return np.array([e0, gamma0, amplitude0, background0], dtype=float)


def fit_breit_wigner(
    energy: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
) -> tuple[ResonanceParams, np.ndarray, np.ndarray, np.ndarray, int]:
    """Fit parameters and return fit diagnostics.

    Returns:
        fitted_params, fitted_curve, standard_error, raw_theta, nfev
    """
    theta0 = initial_guess(energy, observed)

    span = float(energy.max() - energy.min())
    lower = np.array([energy.min(), 1e-6, 1e-6, -np.inf], dtype=float)
    upper = np.array([energy.max(), 2.0 * span, np.inf, np.inf], dtype=float)

    result = least_squares(
        residual_vector,
        x0=theta0,
        jac=residual_jacobian,
        bounds=(lower, upper),
        method="trf",
        args=(energy, observed, sigma),
        max_nfev=2000,
    )

    if not result.success:
        raise RuntimeError(f"least_squares failed: {result.message}")

    fitted = result.x
    fitted_curve = breit_wigner_cross_section(energy, *fitted)

    # Approximate covariance from local linearization: cov ~= s^2 * (J^T J)^-1.
    j = result.jac
    dof = max(1, energy.size - fitted.size)
    rss = 2.0 * result.cost
    s2 = rss / dof
    jtj = j.T @ j
    try:
        cov = s2 * np.linalg.inv(jtj)
        stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        stderr = np.full(fitted.shape, np.nan)

    params = ResonanceParams(
        e_r=float(fitted[0]),
        gamma=float(fitted[1]),
        amplitude=float(fitted[2]),
        background=float(fitted[3]),
    )

    return params, fitted_curve, stderr, fitted, int(result.nfev)


def print_report(
    true_params: ResonanceParams,
    fitted_params: ResonanceParams,
    stderr: np.ndarray,
    energy: np.ndarray,
    observed: np.ndarray,
    fitted_curve: np.ndarray,
    sigma: np.ndarray,
    nfev: int,
) -> None:
    """Print a compact textual report."""
    rmse = float(np.sqrt(mean_squared_error(observed, fitted_curve)))
    r2 = float(r2_score(observed, fitted_curve))
    wrmse = float(np.sqrt(np.mean(((fitted_curve - observed) / sigma) ** 2)))

    summary = pd.DataFrame(
        {
            "parameter": ["E_r", "Gamma", "Amplitude", "Background"],
            "true": [
                true_params.e_r,
                true_params.gamma,
                true_params.amplitude,
                true_params.background,
            ],
            "fit": [
                fitted_params.e_r,
                fitted_params.gamma,
                fitted_params.amplitude,
                fitted_params.background,
            ],
            "stderr_approx": stderr,
        }
    )
    summary["abs_error"] = np.abs(summary["fit"] - summary["true"])

    print("=== Breit-Wigner Formula MVP ===")
    print(f"data_points={energy.size}, nfev={nfev}")
    print(f"RMSE={rmse:.6f}, weighted_RMSE={wrmse:.6f}, R2={r2:.6f}")
    print("\nParameter recovery:")
    print(summary.to_string(index=False, justify="center", float_format=lambda x: f"{x: .6f}"))

    sample_idx = np.linspace(0, energy.size - 1, 8, dtype=int)
    check = pd.DataFrame(
        {
            "E": energy[sample_idx],
            "obs": observed[sample_idx],
            "fit": fitted_curve[sample_idx],
            "residual": observed[sample_idx] - fitted_curve[sample_idx],
        }
    )
    print("\nSample predictions:")
    print(check.to_string(index=False, justify="center", float_format=lambda x: f"{x: .6f}"))


def main() -> None:
    rng = np.random.default_rng(20260407)

    energy, observed, _clean, sigma, true_params = build_synthetic_dataset(rng=rng)
    fitted_params, fitted_curve, stderr, _theta, nfev = fit_breit_wigner(
        energy=energy,
        observed=observed,
        sigma=sigma,
    )

    print_report(
        true_params=true_params,
        fitted_params=fitted_params,
        stderr=stderr,
        energy=energy,
        observed=observed,
        fitted_curve=fitted_curve,
        sigma=sigma,
        nfev=nfev,
    )


if __name__ == "__main__":
    main()
