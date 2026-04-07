"""Aharonov-Bohm effect MVP using a two-path interference model.

The script models electron-like double-slit interference in the Fraunhofer regime:
I(x, Phi) = 1 + V cos(beta*x + Delta_phi)
where Delta_phi = sign * 2*pi*(Phi/Phi0) is the Aharonov-Bohm phase shift.

It validates three key properties:
1) Flux periodicity: I(Phi + Phi0) = I(Phi)
2) Phase linearity: estimated phase is linear in flux
3) Fringe translation: x_shift = -Delta_phi / beta
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import e, h, pi


@dataclass(frozen=True)
class ABConfig:
    """Configuration for one Aharonov-Bohm interference simulation."""

    wavelength_m: float = 50e-12
    slit_separation_m: float = 2e-6
    screen_distance_m: float = 1.0
    screen_half_width_m: float = 5e-3
    n_screen_points: int = 4001
    visibility: float = 0.95
    orientation_sign: float = 1.0
    flux_scan: tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.5)


def magnetic_flux_quantum(abs_charge_coulomb: float = e) -> float:
    """Return magnetic flux quantum Phi0 = h/|q| in Weber."""
    if abs_charge_coulomb <= 0.0:
        raise ValueError("|q| must be positive.")
    return h / abs_charge_coulomb


def fringe_wave_number(config: ABConfig) -> float:
    """Return beta = k*d/L, the fringe spatial phase coefficient in rad/m."""
    if config.wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive.")
    if config.slit_separation_m <= 0.0:
        raise ValueError("slit_separation_m must be positive.")
    if config.screen_distance_m <= 0.0:
        raise ValueError("screen_distance_m must be positive.")
    k = 2.0 * pi / config.wavelength_m
    return k * config.slit_separation_m / config.screen_distance_m


def ab_phase_shift(flux_fraction: float, orientation_sign: float) -> float:
    """Return AB phase shift Delta_phi = sign * 2*pi*(Phi/Phi0)."""
    return orientation_sign * 2.0 * pi * flux_fraction


def interference_intensity(screen_x_m: np.ndarray, flux_fraction: float, config: ABConfig) -> np.ndarray:
    """Compute normalized interference intensity on the screen."""
    beta = fringe_wave_number(config)
    delta_phi = ab_phase_shift(flux_fraction, config.orientation_sign)
    phase = beta * screen_x_m + delta_phi
    intensity = 1.0 + config.visibility * np.cos(phase)
    return intensity


def wrapped_phase_difference(a: float, b: float) -> float:
    """Shortest phase difference a-b mapped to [-pi, pi]."""
    return float(np.angle(np.exp(1j * (a - b))))


def estimate_phase_shift(screen_x_m: np.ndarray, intensity: np.ndarray, beta: float) -> float:
    """Estimate phase offset from a one-frequency projection of fringe signal."""
    centered = intensity - np.mean(intensity)
    projector = np.exp(-1j * beta * screen_x_m)
    coefficient = np.sum(centered * projector)
    if np.abs(coefficient) < 1e-12:
        raise ValueError("Projection magnitude too small; cannot estimate phase.")
    return float(np.angle(coefficient))


def run_ab_effect_mvp(config: ABConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run the AB simulation, return result table and diagnostics."""
    if config.n_screen_points < 1001:
        raise ValueError("n_screen_points should be >= 1001 for stable phase estimation.")
    if config.screen_half_width_m <= 0.0:
        raise ValueError("screen_half_width_m must be positive.")
    if not (0.0 <= config.visibility <= 1.0):
        raise ValueError("visibility must be in [0, 1].")

    beta = fringe_wave_number(config)
    x = np.linspace(-config.screen_half_width_m, config.screen_half_width_m, config.n_screen_points)

    rows: list[dict[str, float]] = []
    for flux_fraction in config.flux_scan:
        pattern = interference_intensity(x, flux_fraction, config)
        phase_pred = ab_phase_shift(flux_fraction, config.orientation_sign)
        phase_est = estimate_phase_shift(x, pattern, beta)
        phase_err = wrapped_phase_difference(phase_est, phase_pred)

        shift_pred = -phase_pred / beta
        shift_est = -phase_est / beta
        center_intensity = float(pattern[config.n_screen_points // 2])

        rows.append(
            {
                "flux_over_flux0": float(flux_fraction),
                "phase_pred_rad": float(phase_pred),
                "phase_est_rad": float(phase_est),
                "phase_err_rad": float(phase_err),
                "shift_pred_um": float(shift_pred * 1e6),
                "shift_est_um": float(shift_est * 1e6),
                "center_intensity": center_intensity,
            }
        )

    report = pd.DataFrame(rows)

    # Numerical periodicity test: Phi and Phi + Phi0 should produce identical fringes.
    base_flux = 0.25
    pattern_base = interference_intensity(x, base_flux, config)
    pattern_periodic = interference_intensity(x, base_flux + 1.0, config)
    periodic_rmse = float(np.sqrt(np.mean((pattern_base - pattern_periodic) ** 2)))

    # Linearity test on principal branch [-0.5, 0.5].
    x_fit = report["flux_over_flux0"].to_numpy(dtype=np.float64)
    y_fit_wrapped = report["phase_est_rad"].to_numpy(dtype=np.float64)
    y_fit = np.unwrap(y_fit_wrapped)
    slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
    y_pred = slope * x_fit + intercept
    ss_res = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    diagnostics = {
        "beta_rad_per_m": float(beta),
        "fringe_period_um": float((2.0 * pi / beta) * 1e6),
        "phase_mae_rad": float(np.mean(np.abs(report["phase_err_rad"].to_numpy(dtype=np.float64)))),
        "periodicity_rmse": periodic_rmse,
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
        "fit_r2": float(r2),
        "expected_slope": float(config.orientation_sign * 2.0 * pi),
    }
    return report, diagnostics


def run_checks(report: pd.DataFrame, diagnostics: dict[str, float], config: ABConfig) -> None:
    """Automatic validation checks for MVP correctness."""
    expected_slope = diagnostics["expected_slope"]
    assert diagnostics["periodicity_rmse"] < 1e-10, "Flux periodicity check failed."
    assert diagnostics["phase_mae_rad"] < 5e-3, "Phase estimation error is too large."
    assert abs(diagnostics["fit_slope"] - expected_slope) < 3e-2, "Phase-flux slope mismatch."
    assert diagnostics["fit_r2"] > 0.9999, "Phase-flux relation is not sufficiently linear."

    shift_err_um = (
        np.abs(report["phase_err_rad"].to_numpy(dtype=np.float64))
        / diagnostics["beta_rad_per_m"]
        * 1e6
    )
    assert float(np.max(shift_err_um)) < 0.2, "Fringe shift mismatch is too large (um scale)."

    assert len(config.flux_scan) >= 3, "Need at least 3 flux points for meaningful regression."


def main() -> None:
    config = ABConfig()
    phi0 = magnetic_flux_quantum()

    report, diagnostics = run_ab_effect_mvp(config)
    run_checks(report, diagnostics, config)

    print("Aharonov-Bohm effect MVP (two-path interference model)")
    print(f"Flux quantum Phi0 = {phi0:.6e} Wb")
    print(f"beta = {diagnostics['beta_rad_per_m']:.6e} rad/m")
    print(f"fringe period = {diagnostics['fringe_period_um']:.3f} um")
    print()
    print(report.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()
    print("Diagnostics:")
    print(f"  periodicity_rmse : {diagnostics['periodicity_rmse']:.3e}")
    print(f"  phase_mae_rad    : {diagnostics['phase_mae_rad']:.3e}")
    print(f"  fit_slope        : {diagnostics['fit_slope']:.6f}")
    print(f"  expected_slope   : {diagnostics['expected_slope']:.6f}")
    print(f"  fit_r2           : {diagnostics['fit_r2']:.6f}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
