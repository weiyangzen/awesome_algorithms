"""Integer Quantum Hall Effect (IQHE) MVP.

This script builds a minimal, inspectable simulation pipeline for a 2D electron gas
under strong magnetic field:
1) continuous filling factor nu(B) = n_e h / (e B)
2) smooth quantization into plateaus (integer nu) using broadened Landau transitions
3) transport tensor (sigma_xx, sigma_xy) -> (rho_xx, rho_xy)
4) automatic checks for plateau quantization, monotonic Hall response, and
   transition locations near half-integer filling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import e, h, hbar, m_e
from scipy.signal import find_peaks
from scipy.special import expit


@dataclass(frozen=True)
class QHEConfig:
    """Configuration for the integer QHE toy model."""

    electron_density_m2: float = 3.0e15
    b_min_t: float = 2.0
    b_max_t: float = 12.0
    n_b_points: int = 600

    max_level: int = 10
    transition_width_nu: float = 0.035
    sigma_xx_peak_scale: float = 0.12  # dimensionless in units of e^2/h

    effective_mass_ratio: float = 0.067  # GaAs-like m*/m_e
    peak_height_threshold: float = 0.01


def validate_config(config: QHEConfig) -> None:
    """Validate numerical and physical ranges."""
    if config.electron_density_m2 <= 0.0:
        raise ValueError("electron_density_m2 must be positive.")
    if config.b_min_t <= 0.0 or config.b_max_t <= 0.0:
        raise ValueError("Magnetic field bounds must be positive.")
    if config.b_min_t >= config.b_max_t:
        raise ValueError("b_min_t must be smaller than b_max_t.")
    if config.n_b_points < 200:
        raise ValueError("n_b_points should be >= 200 for stable diagnostics.")
    if config.max_level < 2:
        raise ValueError("max_level should be >= 2.")
    if config.transition_width_nu <= 0.0:
        raise ValueError("transition_width_nu must be positive.")
    if config.sigma_xx_peak_scale <= 0.0:
        raise ValueError("sigma_xx_peak_scale must be positive.")
    if config.effective_mass_ratio <= 0.0:
        raise ValueError("effective_mass_ratio must be positive.")


def magnetic_field_grid(config: QHEConfig) -> np.ndarray:
    """Return magnetic field grid (Tesla)."""
    return np.linspace(config.b_min_t, config.b_max_t, config.n_b_points)


def filling_factor(electron_density_m2: float, b_field_t: np.ndarray) -> np.ndarray:
    """Continuous filling factor nu = n_e h / (e B)."""
    return electron_density_m2 * h / (e * b_field_t)


def landau_level_energies_mev(b_field_t: np.ndarray, max_level: int, m_ratio: float) -> np.ndarray:
    """Landau levels E_n = (n+1/2) hbar omega_c, returned in meV.

    Shape: (n_B, max_level+1)
    """
    n = np.arange(max_level + 1, dtype=np.float64)
    m_eff = m_ratio * m_e
    omega_c = e * b_field_t[:, None] / m_eff
    energies_joule = (n[None, :] + 0.5) * hbar * omega_c
    return energies_joule / e * 1e3


def quantized_filling_and_sigma_xx(
    nu_cont: np.ndarray,
    max_level: int,
    transition_width_nu: float,
    sigma_xx_peak_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return smooth integer-plateau filling and sigma_xx in units of e^2/h.

    We model each Landau level occupancy with a sigmoid centered at half-integer filling:
      occ_n(nu) = sigmoid((nu - (n + 1/2)) / width)
      nu_q = sum_n occ_n

    sigma_xx is modeled as a sum of transition peaks occ_n*(1-occ_n), maximal near
    half-integer filling where states are extended.
    """
    levels = np.arange(max_level + 1, dtype=np.float64)
    x = (nu_cont[:, None] - (levels[None, :] + 0.5)) / transition_width_nu
    occ = expit(x)

    nu_q = np.sum(occ, axis=1)
    sigma_xx_e2_over_h = sigma_xx_peak_scale * np.sum(occ * (1.0 - occ), axis=1)
    return nu_q, sigma_xx_e2_over_h


def conductivity_tensor(nu_q: np.ndarray, sigma_xx_e2_over_h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sigma_xy, sigma_xx) in SI units (Siemens)."""
    unit = e**2 / h
    sigma_xy = nu_q * unit
    sigma_xx = sigma_xx_e2_over_h * unit
    return sigma_xy, sigma_xx


def resistivity_tensor(sigma_xy: np.ndarray, sigma_xx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert 2x2 isotropic Hall conductivity tensor to get (rho_xy, rho_xx)."""
    denom = sigma_xx**2 + sigma_xy**2
    rho_xy = sigma_xy / denom
    rho_xx = sigma_xx / denom
    return rho_xy, rho_xx


def run_qhe_mvp(config: QHEConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    """Execute the full integer QHE MVP and return pointwise table + diagnostics."""
    validate_config(config)

    b = magnetic_field_grid(config)
    nu_cont = filling_factor(config.electron_density_m2, b)
    nu_q, sigma_xx_e2_over_h = quantized_filling_and_sigma_xx(
        nu_cont=nu_cont,
        max_level=config.max_level,
        transition_width_nu=config.transition_width_nu,
        sigma_xx_peak_scale=config.sigma_xx_peak_scale,
    )

    sigma_xy, sigma_xx = conductivity_tensor(nu_q=nu_q, sigma_xx_e2_over_h=sigma_xx_e2_over_h)
    rho_xy, rho_xx = resistivity_tensor(sigma_xy=sigma_xy, sigma_xx=sigma_xx)

    # Integer reference for plateau evaluation.
    nu_int = np.clip(np.rint(nu_q), 1.0, None)
    rho_xy_quantized = h / (nu_int * e**2)

    # Landau-level energy reference at each B.
    energies_mev = landau_level_energies_mev(
        b_field_t=b,
        max_level=min(config.max_level, 2),
        m_ratio=config.effective_mass_ratio,
    )

    report = pd.DataFrame(
        {
            "B_T": b,
            "nu_cont": nu_cont,
            "nu_quantized": nu_q,
            "nu_integer_ref": nu_int,
            "sigma_xy_e2_over_h": nu_q,
            "sigma_xx_e2_over_h": sigma_xx_e2_over_h,
            "rho_xy_ohm": rho_xy,
            "rho_xx_ohm": rho_xx,
            "rho_xy_quantized_ref_ohm": rho_xy_quantized,
            "E0_meV": energies_mev[:, 0],
            "E1_meV": energies_mev[:, 1],
            "E2_meV": energies_mev[:, 2],
        }
    )

    plateau_mask = sigma_xx_e2_over_h < 0.01
    rel_err = np.abs(rho_xy - rho_xy_quantized) / rho_xy_quantized
    plateau_rel_err = rel_err[plateau_mask]

    # Hall conductivity should be monotonic non-increasing with B in this setup.
    monotonic_violation = float(np.max(np.diff(nu_q)))

    # Longitudinal conductivity peaks should occur near half-integer filling.
    peaks, _ = find_peaks(
        sigma_xx_e2_over_h,
        height=config.peak_height_threshold,
        distance=max(10, config.n_b_points // 25),
    )
    peak_nu = nu_cont[peaks]
    half_integer_distance = np.abs(peak_nu - (np.floor(peak_nu) + 0.5))

    diagnostics = {
        "b_min_t": float(b[0]),
        "b_max_t": float(b[-1]),
        "nu_min": float(np.min(nu_cont)),
        "nu_max": float(np.max(nu_cont)),
        "plateau_point_fraction": float(np.mean(plateau_mask)),
        "plateau_rel_err_median": float(np.median(plateau_rel_err)) if plateau_rel_err.size else float("nan"),
        "plateau_rel_err_max": float(np.max(plateau_rel_err)) if plateau_rel_err.size else float("nan"),
        "monotonic_violation": monotonic_violation,
        "n_transition_peaks": int(peaks.size),
        "half_integer_peak_mean_abs_dev": float(np.mean(half_integer_distance)) if peaks.size else float("nan"),
    }

    return report, diagnostics


def run_checks(report: pd.DataFrame, diagnostics: dict[str, float]) -> None:
    """Automated validity checks for this toy IQHE model."""
    assert diagnostics["nu_max"] > 5.5 and diagnostics["nu_min"] < 1.2, "B-range does not cover enough filling factors."
    assert diagnostics["plateau_point_fraction"] > 0.7, "Too few plateau points for robust quantization diagnostics."
    assert diagnostics["plateau_rel_err_median"] < 0.01, "Median plateau Hall-resistance error is too large."
    assert diagnostics["plateau_rel_err_max"] < 0.10, "Worst plateau Hall-resistance error is too large."

    # For non-increasing trend, max(diff) should stay near zero (no positive jumps).
    assert diagnostics["monotonic_violation"] < 1e-8, "Hall conductivity is not monotonic in B."

    # We expect several transitions (nu ~ 5.5, 4.5, ..., 1.5) in chosen B window.
    assert diagnostics["n_transition_peaks"] >= 4, "Not enough longitudinal conductivity peaks detected."
    assert diagnostics["half_integer_peak_mean_abs_dev"] < 0.03, "Transition peaks are not aligned with half-integer filling."

    assert not report.isna().any().any(), "Report contains NaN values."


def summarize_plateaus(report: pd.DataFrame) -> pd.DataFrame:
    """Return representative points closest to integer fillings."""
    reps: list[pd.Series] = []
    max_integer = int(np.floor(report["nu_quantized"].max()))
    for n in range(1, max_integer + 1):
        idx = (report["nu_quantized"] - n).abs().idxmin()
        reps.append(report.loc[idx])

    cols = [
        "B_T",
        "nu_cont",
        "nu_quantized",
        "sigma_xx_e2_over_h",
        "rho_xy_ohm",
        "rho_xy_quantized_ref_ohm",
        "rho_xx_ohm",
    ]
    return pd.DataFrame(reps)[cols].sort_values("nu_quantized", ascending=False).reset_index(drop=True)


def main() -> None:
    config = QHEConfig()
    report, diagnostics = run_qhe_mvp(config)
    run_checks(report, diagnostics)

    plateau_table = summarize_plateaus(report)

    print("Integer Quantum Hall Effect MVP (2DEG toy model)")
    print(f"B range: [{diagnostics['b_min_t']:.2f}, {diagnostics['b_max_t']:.2f}] T")
    print(f"Continuous filling range nu: [{diagnostics['nu_min']:.3f}, {diagnostics['nu_max']:.3f}]")
    print()
    print("Representative plateau points:")
    print(plateau_table.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()
    print("Diagnostics:")
    print(f"  plateau_point_fraction           : {diagnostics['plateau_point_fraction']:.3f}")
    print(f"  plateau_rel_err_median           : {diagnostics['plateau_rel_err_median']:.3e}")
    print(f"  plateau_rel_err_max              : {diagnostics['plateau_rel_err_max']:.3e}")
    print(f"  monotonic_violation              : {diagnostics['monotonic_violation']:.3e}")
    print(f"  n_transition_peaks               : {diagnostics['n_transition_peaks']}")
    print(f"  half_integer_peak_mean_abs_dev   : {diagnostics['half_integer_peak_mean_abs_dev']:.3e}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
