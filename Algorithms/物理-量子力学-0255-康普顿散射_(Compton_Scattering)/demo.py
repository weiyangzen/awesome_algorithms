"""Compton Scattering MVP.

This script demonstrates core Compton-scattering computations:
1) Scattered photon energy vs. angle.
2) Wavelength shift consistency with the Compton formula.
3) Klein-Nishina differential cross section.
4) Low-energy consistency with the Thomson limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# SI constants (exact or CODATA values).
PLANCK_J_S = 6.626_070_15e-34
LIGHT_SPEED_M_S = 299_792_458.0
ELECTRON_MASS_KG = 9.109_383_701_5e-31
EV_TO_J = 1.602_176_634e-19
KEV_TO_J = 1.0e3 * EV_TO_J
CLASSICAL_ELECTRON_RADIUS_M = 2.817_940_326_2e-15
BARN_M2 = 1.0e-28


@dataclass(frozen=True)
class ComptonConfig:
    incident_energy_kev: float = 661.7
    low_energy_check_kev: float = 1.0
    n_theta: int = 181
    shift_abs_tol_m: float = 1.0e-20
    shift_rel_tol: float = 1.0e-10
    low_energy_point_rel_tol: float = 2.0e-2
    low_energy_total_rel_tol: float = 1.0e-2


def electron_rest_energy_kev() -> float:
    return ELECTRON_MASS_KG * LIGHT_SPEED_M_S**2 / KEV_TO_J


def compton_wavelength_m() -> float:
    return PLANCK_J_S / (ELECTRON_MASS_KG * LIGHT_SPEED_M_S)


def wavelength_from_energy_m(energy_kev: np.ndarray | float) -> np.ndarray:
    energy_j = np.asarray(energy_kev, dtype=float) * KEV_TO_J
    return (PLANCK_J_S * LIGHT_SPEED_M_S) / energy_j


def scattered_photon_energy_kev(
    incident_energy_kev: float,
    theta_rad: np.ndarray,
    rest_energy_kev: float,
) -> np.ndarray:
    if incident_energy_kev <= 0.0:
        raise ValueError("incident_energy_kev must be positive.")

    alpha = incident_energy_kev / rest_energy_kev
    return incident_energy_kev / (1.0 + alpha * (1.0 - np.cos(theta_rad)))


def compton_shift_predicted_m(theta_rad: np.ndarray, lambda_c_m: float) -> np.ndarray:
    return lambda_c_m * (1.0 - np.cos(theta_rad))


def klein_nishina_differential_cross_section(
    theta_rad: np.ndarray,
    incident_energy_kev: float,
    rest_energy_kev: float,
) -> np.ndarray:
    e_prime = scattered_photon_energy_kev(incident_energy_kev, theta_rad, rest_energy_kev)
    ratio = e_prime / incident_energy_kev
    sin2 = np.sin(theta_rad) ** 2

    return 0.5 * (CLASSICAL_ELECTRON_RADIUS_M**2) * ratio**2 * (ratio + 1.0 / ratio - sin2)


def thomson_differential_cross_section(theta_rad: np.ndarray) -> np.ndarray:
    return 0.5 * (CLASSICAL_ELECTRON_RADIUS_M**2) * (1.0 + np.cos(theta_rad) ** 2)


def total_cross_section(theta_rad: np.ndarray, d_sigma_d_omega: np.ndarray) -> float:
    # Integrate over azimuth first: dOmega = 2*pi*sin(theta) dtheta
    integrand = d_sigma_d_omega * np.sin(theta_rad)
    return 2.0 * np.pi * float(np.trapezoid(integrand, theta_rad))


def relative_error(x: np.ndarray, y: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    denom = np.maximum(np.abs(y), eps)
    return np.abs(x - y) / denom


def build_report_table(
    theta_deg: np.ndarray,
    scattered_energy_kev: np.ndarray,
    lambda_prime_m: np.ndarray,
    delta_lambda_m: np.ndarray,
    delta_lambda_pred_m: np.ndarray,
    dsigma_kn_m2_sr: np.ndarray,
    dsigma_th_m2_sr: np.ndarray,
) -> pd.DataFrame:
    ratio = dsigma_kn_m2_sr / np.maximum(dsigma_th_m2_sr, 1e-40)

    return pd.DataFrame(
        {
            "theta_deg": theta_deg,
            "E_prime_kev": scattered_energy_kev,
            "lambda_prime_pm": lambda_prime_m * 1e12,
            "delta_lambda_pm": delta_lambda_m * 1e12,
            "delta_lambda_pred_pm": delta_lambda_pred_m * 1e12,
            "dsigma_kn_barn_sr": dsigma_kn_m2_sr / BARN_M2,
            "dsigma_th_barn_sr": dsigma_th_m2_sr / BARN_M2,
            "kn_over_th": ratio,
        }
    )


def validate(
    theta_rad: np.ndarray,
    delta_lambda_m: np.ndarray,
    delta_lambda_pred_m: np.ndarray,
    dsigma_kn_m2_sr: np.ndarray,
    cfg: ComptonConfig,
    rest_energy_kev: float,
) -> tuple[bool, dict[str, float]]:
    abs_shift_err = np.abs(delta_lambda_m - delta_lambda_pred_m)
    max_abs_shift_err = float(np.max(abs_shift_err))

    nonzero_mask = np.abs(delta_lambda_pred_m) > 1e-18
    if np.any(nonzero_mask):
        rel_shift_err = abs_shift_err[nonzero_mask] / np.abs(delta_lambda_pred_m[nonzero_mask])
        max_rel_shift_err = float(np.max(rel_shift_err))
    else:
        max_rel_shift_err = 0.0

    dsigma_kn_low = klein_nishina_differential_cross_section(
        theta_rad=theta_rad,
        incident_energy_kev=cfg.low_energy_check_kev,
        rest_energy_kev=rest_energy_kev,
    )
    dsigma_th = thomson_differential_cross_section(theta_rad)

    low_energy_point_rel = relative_error(dsigma_kn_low, dsigma_th)
    max_low_energy_point_rel = float(np.max(low_energy_point_rel))

    sigma_kn_low = total_cross_section(theta_rad, dsigma_kn_low)
    sigma_th = (8.0 * np.pi / 3.0) * CLASSICAL_ELECTRON_RADIUS_M**2
    low_energy_total_rel = float(abs(sigma_kn_low - sigma_th) / sigma_th)

    finite = bool(np.isfinite(delta_lambda_m).all() and np.isfinite(dsigma_kn_m2_sr).all())
    non_negative = bool(np.all(dsigma_kn_m2_sr >= 0.0))

    passed = (
        finite
        and non_negative
        and max_abs_shift_err <= cfg.shift_abs_tol_m
        and max_rel_shift_err <= cfg.shift_rel_tol
        and max_low_energy_point_rel <= cfg.low_energy_point_rel_tol
        and low_energy_total_rel <= cfg.low_energy_total_rel_tol
    )

    metrics = {
        "max_abs_shift_err_m": max_abs_shift_err,
        "max_rel_shift_err": max_rel_shift_err,
        "max_low_energy_point_rel": max_low_energy_point_rel,
        "low_energy_total_rel": low_energy_total_rel,
        "finite": float(finite),
        "non_negative": float(non_negative),
        "sigma_kn_low_m2": sigma_kn_low,
        "sigma_thomson_m2": sigma_th,
    }
    return passed, metrics


def main() -> None:
    cfg = ComptonConfig()
    rest_energy_kev = electron_rest_energy_kev()
    lambda_c_m = compton_wavelength_m()

    theta_rad = np.linspace(0.0, np.pi, cfg.n_theta)
    theta_deg = np.degrees(theta_rad)

    e_prime_kev = scattered_photon_energy_kev(
        incident_energy_kev=cfg.incident_energy_kev,
        theta_rad=theta_rad,
        rest_energy_kev=rest_energy_kev,
    )

    lambda_0_m = float(wavelength_from_energy_m(cfg.incident_energy_kev))
    lambda_prime_m = wavelength_from_energy_m(e_prime_kev)
    delta_lambda_m = lambda_prime_m - lambda_0_m
    delta_lambda_pred_m = compton_shift_predicted_m(theta_rad, lambda_c_m)

    dsigma_kn_m2_sr = klein_nishina_differential_cross_section(
        theta_rad=theta_rad,
        incident_energy_kev=cfg.incident_energy_kev,
        rest_energy_kev=rest_energy_kev,
    )
    dsigma_th_m2_sr = thomson_differential_cross_section(theta_rad)

    sigma_kn_incident = total_cross_section(theta_rad, dsigma_kn_m2_sr)

    table = build_report_table(
        theta_deg=theta_deg,
        scattered_energy_kev=e_prime_kev,
        lambda_prime_m=lambda_prime_m,
        delta_lambda_m=delta_lambda_m,
        delta_lambda_pred_m=delta_lambda_pred_m,
        dsigma_kn_m2_sr=dsigma_kn_m2_sr,
        dsigma_th_m2_sr=dsigma_th_m2_sr,
    )

    passed, metrics = validate(
        theta_rad=theta_rad,
        delta_lambda_m=delta_lambda_m,
        delta_lambda_pred_m=delta_lambda_pred_m,
        dsigma_kn_m2_sr=dsigma_kn_m2_sr,
        cfg=cfg,
        rest_energy_kev=rest_energy_kev,
    )

    sample_idx = np.linspace(0, len(table) - 1, 10, dtype=int)
    sample = table.iloc[sample_idx]

    print("=== Compton Scattering MVP ===")
    print(
        f"E0={cfg.incident_energy_kev:.3f} keV, "
        f"m_ec2={rest_energy_kev:.6f} keV, "
        f"lambda_C={lambda_c_m * 1e12:.6f} pm"
    )
    print(
        f"theta points={cfg.n_theta}, "
        f"lambda0={lambda_0_m * 1e12:.6f} pm, "
        f"sigma_KN(E0)={sigma_kn_incident / BARN_M2:.6f} barn"
    )
    print()
    print("Sampled angle table:")
    print(sample.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print()
    print("Validation metrics:")
    print(f"max_abs_shift_err_m    = {metrics['max_abs_shift_err_m']:.6e} (tol={cfg.shift_abs_tol_m:.1e})")
    print(f"max_rel_shift_err      = {metrics['max_rel_shift_err']:.6e} (tol={cfg.shift_rel_tol:.1e})")
    print(
        f"max_low_energy_point_rel = {metrics['max_low_energy_point_rel']:.6e} "
        f"(tol={cfg.low_energy_point_rel_tol:.1e})"
    )
    print(
        f"low_energy_total_rel   = {metrics['low_energy_total_rel']:.6e} "
        f"(tol={cfg.low_energy_total_rel_tol:.1e})"
    )
    print(f"sigma_kn_low_m2        = {metrics['sigma_kn_low_m2']:.6e}")
    print(f"sigma_thomson_m2       = {metrics['sigma_thomson_m2']:.6e}")
    print(f"finite_check           = {bool(metrics['finite'])}")
    print(f"non_negative_check     = {bool(metrics['non_negative'])}")
    print(f"Validation: {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
