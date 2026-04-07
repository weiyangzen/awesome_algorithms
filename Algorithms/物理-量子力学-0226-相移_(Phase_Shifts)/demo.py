"""Phase Shifts MVP for quantum scattering.

This script computes partial-wave phase shifts for a spherical square-well
potential in two ways:
1) analytic boundary matching,
2) numerical radial integration (Numerov) + asymptotic matching.

It compares both results across an energy grid and validates agreement using
phase-error and total-cross-section error thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import spherical_jn, spherical_yn


@dataclass(frozen=True)
class PhaseShiftConfig:
    mass: float = 1.0
    hbar: float = 1.0
    well_depth: float = 8.0
    well_radius: float = 1.0

    energy_min: float = 0.5
    energy_max: float = 6.0
    n_energy: int = 12
    l_max: int = 2

    r_min: float = 1.0e-4
    r_max: float = 35.0
    n_r: int = 55_000
    match_r1: float = 18.0
    match_r2: float = 26.0

    phase_abs_tol: float = 6.0e-2
    sigma_rel_tol: float = 2.0e-1


def validate_config(cfg: PhaseShiftConfig) -> None:
    if cfg.mass <= 0.0 or cfg.hbar <= 0.0:
        raise ValueError("mass and hbar must be positive.")
    if cfg.well_depth <= 0.0 or cfg.well_radius <= 0.0:
        raise ValueError("well_depth and well_radius must be positive.")
    if cfg.energy_min <= 0.0 or cfg.energy_max <= cfg.energy_min:
        raise ValueError("energy range must satisfy 0 < energy_min < energy_max.")
    if cfg.n_energy < 2:
        raise ValueError("n_energy must be >= 2.")
    if cfg.l_max < 0:
        raise ValueError("l_max must be >= 0.")
    if cfg.n_r < 1000:
        raise ValueError("n_r must be >= 1000 for stable Numerov integration.")
    if not (0.0 < cfg.r_min < cfg.well_radius < cfg.match_r1 < cfg.match_r2 < cfg.r_max):
        raise ValueError("Require r_min < a < match_r1 < match_r2 < r_max.")


def wave_number(energy: float, mass: float, hbar: float) -> float:
    return float(np.sqrt(2.0 * mass * energy) / hbar)


def inner_wave_number(energy: float, well_depth: float, mass: float, hbar: float) -> float:
    inner_energy = energy + well_depth
    if inner_energy <= 0.0:
        raise ValueError("energy + well_depth must be positive for this attractive square well MVP.")
    return float(np.sqrt(2.0 * mass * inner_energy) / hbar)


def phase_shift_analytic_square_well(energy: float, ell: int, cfg: PhaseShiftConfig) -> float:
    """Analytic phase shift from matching at r=a for a spherical square well."""

    k = wave_number(energy, cfg.mass, cfg.hbar)
    q = inner_wave_number(energy, cfg.well_depth, cfg.mass, cfg.hbar)
    a = cfg.well_radius

    ka = k * a
    qa = q * a

    j_ka = spherical_jn(ell, ka)
    j_qa = spherical_jn(ell, qa)
    jp_ka = spherical_jn(ell, ka, derivative=True)
    jp_qa = spherical_jn(ell, qa, derivative=True)

    y_ka = spherical_yn(ell, ka)
    yp_ka = spherical_yn(ell, ka, derivative=True)

    numerator = q * j_ka * jp_qa - k * j_qa * jp_ka
    denominator = q * y_ka * jp_qa - k * yp_ka * j_qa

    return float(np.arctan2(numerator, denominator))


def potential_square_well(r: np.ndarray, well_depth: float, well_radius: float) -> np.ndarray:
    return np.where(r < well_radius, -well_depth, 0.0)


def radial_g_function(
    r: np.ndarray,
    energy: float,
    ell: int,
    mass: float,
    hbar: float,
    potential: np.ndarray,
) -> np.ndarray:
    kinetic_term = 2.0 * mass * (energy - potential) / (hbar**2)
    centrifugal_term = ell * (ell + 1.0) / (r**2)
    return kinetic_term - centrifugal_term


def numerov_radial_solution(energy: float, ell: int, cfg: PhaseShiftConfig) -> tuple[np.ndarray, np.ndarray]:
    """Solve u'' + g(r)u = 0 with Numerov for reduced radial wavefunction u(r)."""

    r = np.linspace(cfg.r_min, cfg.r_max, cfg.n_r)
    h = r[1] - r[0]

    potential = potential_square_well(r, cfg.well_depth, cfg.well_radius)
    g = radial_g_function(r, energy, ell, cfg.mass, cfg.hbar, potential)

    u = np.zeros_like(r)
    u[0] = r[0] ** (ell + 1)
    u[1] = r[1] ** (ell + 1)

    h2_over_12 = (h * h) / 12.0

    for i in range(1, len(r) - 1):
        c_prev = 1.0 + h2_over_12 * g[i - 1]
        c_curr = 1.0 - 5.0 * h2_over_12 * g[i]
        c_next = 1.0 + h2_over_12 * g[i + 1]

        u[i + 1] = (2.0 * c_curr * u[i] - c_prev * u[i - 1]) / c_next

        if i % 2000 == 0:
            scale = max(abs(u[i - 1]), abs(u[i]), abs(u[i + 1]))
            if scale > 1.0e100:
                u[: i + 2] /= scale

    return r, u


def phase_shift_from_asymptotic_log_derivative(
    r: np.ndarray,
    u: np.ndarray,
    energy: float,
    ell: int,
    cfg: PhaseShiftConfig,
) -> float:
    """Estimate phase shift from asymptotic logarithmic-derivative matching."""

    k = wave_number(energy, cfg.mass, cfg.hbar)

    idx_start = int(np.searchsorted(r, cfg.match_r1))
    idx_end = int(np.searchsorted(r, cfg.match_r2))
    idx_end = min(idx_end, len(r) - 2)

    if idx_end - idx_start < 5:
        raise ValueError("Asymptotic matching window is too small.")

    # Avoid matching exactly at a node by selecting the largest-|u| point in window.
    window = np.abs(u[idx_start:idx_end])
    idx = idx_start + int(np.argmax(window))
    idx = min(max(idx, 1), len(r) - 2)

    dr = r[1] - r[0]
    u_prime = (u[idx + 1] - u[idx - 1]) / (2.0 * dr)
    beta = u_prime / (k * u[idx])

    x = k * r[idx]
    j = spherical_jn(ell, x)
    y = spherical_yn(ell, x)
    j_prime = spherical_jn(ell, x, derivative=True)
    y_prime = spherical_yn(ell, x, derivative=True)

    numerator = beta * j - j_prime
    denominator = beta * y - y_prime

    return float(np.arctan2(numerator, denominator))


def phase_shift_numeric(energy: float, ell: int, cfg: PhaseShiftConfig) -> float:
    r, u = numerov_radial_solution(energy, ell, cfg)
    return phase_shift_from_asymptotic_log_derivative(r, u, energy, ell, cfg)


def phase_error_mod_pi(delta_num: float, delta_ref: float) -> float:
    """Phase shifts are equivalent modulo pi. Return minimal absolute difference."""

    diff = delta_num - delta_ref
    wrapped = (diff + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    return float(abs(wrapped))


def total_cross_section(k: float, deltas: np.ndarray) -> float:
    ell = np.arange(len(deltas))
    prefactor = 4.0 * np.pi / (k * k)
    partial_sum = np.sum((2 * ell + 1) * np.sin(deltas) ** 2)
    return float(prefactor * partial_sum)


def run_phase_shift_mvp(cfg: PhaseShiftConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    validate_config(cfg)

    energies = np.linspace(cfg.energy_min, cfg.energy_max, cfg.n_energy)

    detail_rows: list[dict[str, float]] = []
    sigma_rows: list[dict[str, float]] = []

    for energy in energies:
        deltas_analytic = []
        deltas_numeric = []

        for ell in range(cfg.l_max + 1):
            delta_a = phase_shift_analytic_square_well(energy, ell, cfg)
            delta_n = phase_shift_numeric(energy, ell, cfg)
            err = phase_error_mod_pi(delta_n, delta_a)

            detail_rows.append(
                {
                    "energy": energy,
                    "ell": float(ell),
                    "delta_analytic_rad": delta_a,
                    "delta_numeric_rad": delta_n,
                    "delta_analytic_deg": np.degrees(delta_a),
                    "delta_numeric_deg": np.degrees(delta_n),
                    "phase_abs_err": err,
                }
            )

            deltas_analytic.append(delta_a)
            deltas_numeric.append(delta_n)

        k = wave_number(energy, cfg.mass, cfg.hbar)
        sigma_analytic = total_cross_section(k, np.asarray(deltas_analytic))
        sigma_numeric = total_cross_section(k, np.asarray(deltas_numeric))
        sigma_rel_err = abs(sigma_numeric - sigma_analytic) / max(abs(sigma_analytic), 1.0e-12)

        sigma_rows.append(
            {
                "energy": energy,
                "k": k,
                "sigma_analytic": sigma_analytic,
                "sigma_numeric": sigma_numeric,
                "sigma_rel_err": sigma_rel_err,
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    sigma_df = pd.DataFrame(sigma_rows)

    max_phase_err = float(detail_df["phase_abs_err"].max())
    mean_phase_err = float(detail_df["phase_abs_err"].mean())
    max_sigma_rel_err = float(sigma_df["sigma_rel_err"].max())

    finite_ok = bool(
        np.isfinite(detail_df.to_numpy(dtype=float)).all()
        and np.isfinite(sigma_df.to_numpy(dtype=float)).all()
    )

    passed = (
        finite_ok
        and max_phase_err <= cfg.phase_abs_tol
        and max_sigma_rel_err <= cfg.sigma_rel_tol
    )

    metrics = {
        "max_phase_abs_err": max_phase_err,
        "mean_phase_abs_err": mean_phase_err,
        "max_sigma_rel_err": max_sigma_rel_err,
        "finite_ok": float(finite_ok),
        "passed": float(passed),
    }

    return detail_df, sigma_df, metrics


def main() -> None:
    cfg = PhaseShiftConfig()
    detail_df, sigma_df, metrics = run_phase_shift_mvp(cfg)

    print("=== Quantum Phase Shifts MVP (Spherical Square Well) ===")
    print(
        f"m={cfg.mass}, hbar={cfg.hbar}, V0={cfg.well_depth}, a={cfg.well_radius}, "
        f"l_max={cfg.l_max}, energies=[{cfg.energy_min}, {cfg.energy_max}], n_energy={cfg.n_energy}"
    )
    print(f"Numerov grid: r in [{cfg.r_min}, {cfg.r_max}], n_r={cfg.n_r}")
    print()

    print("Phase-shift detail table (all rows):")
    print(detail_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print()

    print("Total cross-section comparison:")
    print(sigma_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print()

    print("Validation metrics:")
    print(f"max_phase_abs_err = {metrics['max_phase_abs_err']:.6e} (tol={cfg.phase_abs_tol:.2e})")
    print(f"mean_phase_abs_err= {metrics['mean_phase_abs_err']:.6e}")
    print(f"max_sigma_rel_err = {metrics['max_sigma_rel_err']:.6e} (tol={cfg.sigma_rel_tol:.2e})")
    print(f"finite_ok         = {bool(metrics['finite_ok'])}")
    print(f"Validation: {'PASS' if metrics['passed'] else 'FAIL'}")

    if not metrics["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
