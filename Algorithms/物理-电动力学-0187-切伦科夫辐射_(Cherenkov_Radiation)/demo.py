"""Minimal runnable MVP for Cherenkov radiation in electrodynamics.

The script implements source-level Frank-Tamm photon-yield calculations,
checks threshold behavior, verifies a constant-index analytical integral,
and reports wavelength-resolved Cherenkov angles in a weakly dispersive medium.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


ALPHA = 1.0 / 137.035999084  # Fine-structure constant


@dataclass(frozen=True)
class CherenkovConfig:
    """Configuration for Cherenkov-radiation MVP experiments."""

    lambda_min_nm: float = 250.0
    lambda_max_nm: float = 700.0
    n_lambda_samples: int = 6001

    # Weak-dispersion Cauchy-like refractive-index model n(lambda_um)=n0+b/lambda_um^2
    n0: float = 1.322
    cauchy_b_um2: float = 0.0030

    # Beta values used in monotonic-yield checks and reports.
    betas_for_sweep: tuple[float, ...] = (0.76, 0.82, 0.90, 0.98)


def build_wavelength_grid(config: CherenkovConfig) -> np.ndarray:
    """Return wavelength grid in meters."""
    if config.lambda_min_nm <= 0.0 or config.lambda_max_nm <= config.lambda_min_nm:
        raise ValueError("Invalid wavelength range")
    if config.n_lambda_samples < 100:
        raise ValueError("n_lambda_samples must be >= 100 for stable integration")

    return np.linspace(
        config.lambda_min_nm * 1e-9,
        config.lambda_max_nm * 1e-9,
        config.n_lambda_samples,
        dtype=np.float64,
    )


def refractive_index_cauchy(lambda_m: np.ndarray, n0: float, b_um2: float) -> np.ndarray:
    """Cauchy-like refractive index model, lambda in meters."""
    if np.any(lambda_m <= 0.0):
        raise ValueError("Wavelengths must be positive")

    lambda_um = lambda_m * 1e6
    n = n0 + b_um2 / (lambda_um * lambda_um)
    return n.astype(np.float64)


def cherenkov_condition(beta: float, n_lambda: np.ndarray) -> np.ndarray:
    """Boolean mask of wavelengths satisfying beta*n(lambda) > 1."""
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0,1)")
    return beta * n_lambda > 1.0


def cherenkov_angle_rad(beta: float, n_lambda: np.ndarray) -> np.ndarray:
    """Cherenkov angle theta(lambda)=arccos(1/(beta*n(lambda))) where allowed."""
    mask = cherenkov_condition(beta, n_lambda)
    out = np.full_like(n_lambda, np.nan, dtype=np.float64)
    if np.any(mask):
        arg = 1.0 / (beta * n_lambda[mask])
        arg = np.clip(arg, -1.0, 1.0)
        out[mask] = np.arccos(arg)
    return out


def frank_tamm_photon_density(beta: float, n_lambda: np.ndarray, lambda_m: np.ndarray) -> np.ndarray:
    """Compute d^2N/(dx d lambda) from Frank-Tamm formula.

    Formula used:
        d^2N/(dx d lambda) = 2*pi*alpha*(1/lambda^2)*(1 - 1/(beta^2*n(lambda)^2))
    and is clamped to zero where beta*n(lambda) <= 1.
    """
    if lambda_m.shape != n_lambda.shape:
        raise ValueError("lambda_m and n_lambda must have the same shape")

    spectrum = np.zeros_like(lambda_m, dtype=np.float64)
    mask = cherenkov_condition(beta, n_lambda)
    if np.any(mask):
        factor = 1.0 - 1.0 / ((beta * n_lambda[mask]) ** 2)
        spectrum[mask] = 2.0 * math.pi * ALPHA * factor / (lambda_m[mask] ** 2)
    return spectrum


def integrate_photons_per_meter(spectrum: np.ndarray, lambda_m: np.ndarray) -> float:
    """Integrate d^2N/(dx d lambda) over wavelength, giving dN/dx."""
    return float(np.trapezoid(spectrum, lambda_m))


def analytic_constant_index_yield(
    beta: float,
    n_const: float,
    lambda_min_m: float,
    lambda_max_m: float,
) -> float:
    """Closed-form Frank-Tamm integral for constant refractive index."""
    if beta * n_const <= 1.0:
        return 0.0

    prefactor = 2.0 * math.pi * ALPHA * (1.0 - 1.0 / ((beta * n_const) ** 2))
    return prefactor * (1.0 / lambda_min_m - 1.0 / lambda_max_m)


def run_demo(config: CherenkovConfig) -> Dict[str, pd.DataFrame]:
    """Run deterministic checks and return report tables."""
    lambda_m = build_wavelength_grid(config)
    n_lambda = refractive_index_cauchy(lambda_m, config.n0, config.cauchy_b_um2)

    # 1) Threshold checks in dispersive medium.
    n_min = float(np.min(n_lambda))
    n_max = float(np.max(n_lambda))

    beta_no_emit = 0.98 / n_max
    spec_no_emit = frank_tamm_photon_density(beta_no_emit, n_lambda, lambda_m)
    assert beta_no_emit * n_max < 1.0, "beta_no_emit is not actually below threshold"
    assert np.max(spec_no_emit) == 0.0, "Below-threshold beta unexpectedly produced emission"

    beta_all_emit = min(0.99, 1.02 / n_min)
    assert beta_all_emit * n_min > 1.0, "beta_all_emit is not above full-band threshold"
    mask_all_emit = cherenkov_condition(beta_all_emit, n_lambda)
    assert bool(np.all(mask_all_emit)), "Expected full-band emission for beta_all_emit"

    # 2) Constant-index numerical integration vs analytical integral.
    beta_analytic = 0.92
    n_const = 1.333
    n_const_array = np.full_like(lambda_m, n_const)
    spectrum_const = frank_tamm_photon_density(beta_analytic, n_const_array, lambda_m)
    numeric_const = integrate_photons_per_meter(spectrum_const, lambda_m)
    analytic_const = analytic_constant_index_yield(
        beta_analytic,
        n_const,
        float(lambda_m[0]),
        float(lambda_m[-1]),
    )
    const_rel_err = abs(numeric_const - analytic_const) / max(1e-15, abs(analytic_const))
    assert const_rel_err < 5.0e-5, (
        "Constant-index Frank-Tamm integral mismatch too large: "
        f"{const_rel_err:.3e}"
    )

    # 3) Monotonic photon yield for increasing beta.
    sweep_rows = []
    total_yields = []
    for beta in config.betas_for_sweep:
        spectrum = frank_tamm_photon_density(beta, n_lambda, lambda_m)
        total = integrate_photons_per_meter(spectrum, lambda_m)
        angles = cherenkov_angle_rad(beta, n_lambda)
        mean_angle_deg = float(np.degrees(np.nanmean(angles)))
        max_angle_deg = float(np.degrees(np.nanmax(angles)))
        active_fraction = float(np.mean(np.isfinite(angles)))

        total_yields.append(total)
        sweep_rows.append(
            {
                "beta": float(beta),
                "dN_dx_per_m": float(total),
                "mean_theta_deg": mean_angle_deg,
                "max_theta_deg": max_angle_deg,
                "active_wavelength_fraction": active_fraction,
            }
        )

    total_yields_arr = np.asarray(total_yields, dtype=np.float64)
    monotonic_steps = np.diff(total_yields_arr)
    assert bool(np.all(monotonic_steps > 0.0)), "Total Cherenkov yield should increase with beta"

    # 4) Angle sanity bounds for high-beta case.
    high_beta = config.betas_for_sweep[-1]
    high_angles = cherenkov_angle_rad(high_beta, n_lambda)
    finite_angles = high_angles[np.isfinite(high_angles)]
    assert finite_angles.size > 0, "No finite Cherenkov angle for high-beta case"
    assert float(np.min(finite_angles)) > 0.0, "Cherenkov angle must be positive"
    assert float(np.max(finite_angles)) < (0.5 * math.pi), "Cherenkov angle must be < 90 deg"

    checks_df = pd.DataFrame(
        [
            {
                "check": "below_threshold_zero_emission",
                "value": float(np.max(spec_no_emit)),
                "note": "expected exactly 0",
            },
            {
                "check": "full_band_emission_fraction",
                "value": float(np.mean(mask_all_emit)),
                "note": "expected 1",
            },
            {
                "check": "const_n_numeric_vs_analytic_rel_err",
                "value": const_rel_err,
                "note": "expected < 5e-5",
            },
            {
                "check": "monotonic_min_step_dNdx",
                "value": float(np.min(monotonic_steps)),
                "note": "expected > 0",
            },
            {
                "check": "n_lambda_min",
                "value": n_min,
                "note": "dispersion diagnostic",
            },
            {
                "check": "n_lambda_max",
                "value": n_max,
                "note": "dispersion diagnostic",
            },
            {
                "check": "beta_no_emit",
                "value": beta_no_emit,
                "note": "computed below threshold",
            },
            {
                "check": "beta_all_emit",
                "value": beta_all_emit,
                "note": "computed above full-band threshold",
            },
        ]
    )

    sweep_df = pd.DataFrame(sweep_rows)
    return {"checks": checks_df, "sweep": sweep_df}


def main() -> None:
    config = CherenkovConfig()
    reports = run_demo(config)

    checks = reports["checks"]
    sweep = reports["sweep"]

    print("=== Cherenkov Radiation MVP (Frank-Tamm + Threshold + Angle) ===")
    print(
        "wavelength range: "
        f"{config.lambda_min_nm:.1f}-{config.lambda_max_nm:.1f} nm, "
        f"samples={config.n_lambda_samples}"
    )
    print(
        "dispersion model: "
        f"n(lambda_um)= {config.n0:.6f} + {config.cauchy_b_um2:.6f}/lambda_um^2"
    )

    print("\nchecks:")
    for row in checks.itertuples(index=False):
        print(f"  {row.check:38s} : {row.value:.6e}   ({row.note})")

    print("\nbeta sweep:")
    for row in sweep.itertuples(index=False):
        print(
            "  "
            f"beta={row.beta:.3f}, dN/dx={row.dN_dx_per_m:.6e} 1/m, "
            f"mean_theta={row.mean_theta_deg:.3f} deg, "
            f"max_theta={row.max_theta_deg:.3f} deg, "
            f"active_frac={row.active_wavelength_fraction:.3f}"
        )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
