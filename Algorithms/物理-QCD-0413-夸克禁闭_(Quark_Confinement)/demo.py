"""Minimal runnable MVP for quark confinement via the Cornell potential.

This script builds a transparent confinement pipeline:
1) define the Cornell static potential V(r) = -kappa/r + sigma*r + c0,
2) solve the radial Schrodinger equation (s-wave) with finite differences,
3) extract low-lying bound states and spatial scales,
4) verify confinement signatures (positive string tension, large-r force plateau).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg as spla


@dataclass(frozen=True)
class ConfinementConfig:
    hbar_c_gev_fm: float = 0.1973269804
    kappa_gev_fm: float = 0.25
    sigma_gev_per_fm: float = 0.90
    c0_gev: float = -0.35
    quark_mass_gev: float = 1.50
    r_max_fm: float = 3.00
    grid_points: int = 1000
    n_levels: int = 5
    tail_fit_min_fm: float = 1.50
    sample_r_fm: tuple[float, ...] = (0.30, 0.60, 1.00, 1.50, 2.00, 2.50)


def validate_config(config: ConfinementConfig) -> None:
    if config.hbar_c_gev_fm <= 0.0:
        raise ValueError("hbar_c_gev_fm must be positive.")
    if config.kappa_gev_fm <= 0.0:
        raise ValueError("kappa_gev_fm must be positive.")
    if config.sigma_gev_per_fm <= 0.0:
        raise ValueError("sigma_gev_per_fm must be positive.")
    if config.quark_mass_gev <= 0.0:
        raise ValueError("quark_mass_gev must be positive.")
    if config.r_max_fm <= 1.0:
        raise ValueError("r_max_fm must be > 1.0 fm for a meaningful tail fit.")
    if config.grid_points < 200:
        raise ValueError("grid_points must be >= 200.")
    if config.n_levels < 2 or config.n_levels >= config.grid_points - 2:
        raise ValueError("n_levels must be in [2, grid_points-3].")
    if not (0.5 < config.tail_fit_min_fm < config.r_max_fm):
        raise ValueError("tail_fit_min_fm must be in (0.5, r_max_fm).")
    if not config.sample_r_fm:
        raise ValueError("sample_r_fm cannot be empty.")
    if min(config.sample_r_fm) <= 0.0 or max(config.sample_r_fm) >= config.r_max_fm:
        raise ValueError("sample_r_fm values must lie in (0, r_max_fm).")


def cornell_potential(r_fm: np.ndarray, config: ConfinementConfig) -> np.ndarray:
    r = np.asarray(r_fm, dtype=float)
    return -config.kappa_gev_fm / r + config.sigma_gev_per_fm * r + config.c0_gev


def confinement_force(r_fm: np.ndarray, config: ConfinementConfig) -> np.ndarray:
    r = np.asarray(r_fm, dtype=float)
    return config.kappa_gev_fm / (r**2) + config.sigma_gev_per_fm


def build_radial_grid(config: ConfinementConfig) -> tuple[np.ndarray, float]:
    dr = config.r_max_fm / (config.grid_points + 1)
    r = dr * np.arange(1, config.grid_points + 1, dtype=float)
    return r, dr


def build_radial_hamiltonian(r_fm: np.ndarray, dr_fm: float, config: ConfinementConfig) -> sparse.csc_matrix:
    mu = 0.5 * config.quark_mass_gev
    kinetic_prefactor = (config.hbar_c_gev_fm**2) / (2.0 * mu)

    potential = cornell_potential(r_fm, config)
    main_diag = 2.0 * kinetic_prefactor / (dr_fm**2) + potential
    off_diag = -kinetic_prefactor / (dr_fm**2) * np.ones(r_fm.size - 1, dtype=float)

    hamiltonian = sparse.diags(
        diagonals=[off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        format="csc",
    )
    return hamiltonian


def solve_s_wave_levels(config: ConfinementConfig) -> tuple[pd.DataFrame, np.ndarray, float]:
    r_fm, dr_fm = build_radial_grid(config)
    hamiltonian = build_radial_hamiltonian(r_fm, dr_fm, config)

    evals, evecs = spla.eigsh(hamiltonian, k=config.n_levels, which="SA")
    order = np.argsort(evals)
    evals = np.asarray(evals[order], dtype=float)
    evecs = np.asarray(evecs[:, order], dtype=float)

    rows: list[dict[str, float | str]] = []
    for idx, energy in enumerate(evals, start=1):
        u = evecs[:, idx - 1]
        norm = np.sqrt(np.sum(np.abs(u) ** 2) * dr_fm)
        u = u / norm

        prob = np.abs(u) ** 2
        mean_r = float(np.sum(prob * r_fm) * dr_fm)
        rms_r = float(np.sqrt(np.sum(prob * (r_fm**2)) * dr_fm))
        recovered_norm = float(np.sum(prob) * dr_fm)

        rows.append(
            {
                "state": f"{idx}S",
                "energy_gev": float(energy),
                "mean_r_fm": mean_r,
                "rms_r_fm": rms_r,
                "norm": recovered_norm,
            }
        )

    levels_df = pd.DataFrame(rows)
    return levels_df, r_fm, dr_fm


def fit_large_r_string_tension(r_fm: np.ndarray, config: ConfinementConfig) -> tuple[float, float, float]:
    potential = cornell_potential(r_fm, config)
    mask = r_fm >= config.tail_fit_min_fm
    slope, intercept = np.polyfit(r_fm[mask], potential[mask], deg=1)

    pred = slope * r_fm[mask] + intercept
    ss_res = float(np.sum((potential[mask] - pred) ** 2))
    ss_tot = float(np.sum((potential[mask] - np.mean(potential[mask])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
    return float(slope), float(intercept), float(r2)


def build_potential_report(config: ConfinementConfig) -> pd.DataFrame:
    sample_r = np.asarray(config.sample_r_fm, dtype=float)
    potential = cornell_potential(sample_r, config)
    force = confinement_force(sample_r, config)

    coulomb_only = -config.kappa_gev_fm / sample_r + config.c0_gev
    linear_only = config.sigma_gev_per_fm * sample_r + config.c0_gev

    return pd.DataFrame(
        {
            "r_fm": sample_r,
            "V_total_gev": potential,
            "F_total_gev_per_fm": force,
            "V_coulomb_only_gev": coulomb_only,
            "V_linear_only_gev": linear_only,
        }
    )


def run_quality_checks(
    levels_df: pd.DataFrame,
    potential_df: pd.DataFrame,
    sigma_fit: float,
    sigma_fit_r2: float,
    config: ConfinementConfig,
) -> None:
    energies = levels_df["energy_gev"].to_numpy(dtype=float)
    if not np.all(np.isfinite(energies)):
        raise AssertionError("Non-finite eigen energies found.")
    if not np.all(np.diff(energies) > 0.0):
        raise AssertionError("Energy levels must be strictly increasing for ordered bound states.")

    norms = levels_df["norm"].to_numpy(dtype=float)
    if not np.all(np.abs(norms - 1.0) < 5e-3):
        raise AssertionError("Wavefunction normalization drift exceeds tolerance.")

    if not (0.60 * config.sigma_gev_per_fm <= sigma_fit <= 1.40 * config.sigma_gev_per_fm):
        raise AssertionError("Large-r fitted string tension is out of expected confinement range.")
    if sigma_fit_r2 < 0.98:
        raise AssertionError("Large-r linear fit quality is too weak for a confinement signal.")

    force = potential_df["F_total_gev_per_fm"].to_numpy(dtype=float)
    if not np.all(np.diff(force) < 0.0):
        raise AssertionError("Confining force should monotonically decrease toward sigma at large r.")

    force_tail = float(force[-1])
    if abs(force_tail - config.sigma_gev_per_fm) > 0.08 * config.sigma_gev_per_fm:
        raise AssertionError("Large-r force is not close enough to the string-tension plateau.")

    v_total = potential_df["V_total_gev"].to_numpy(dtype=float)
    if not (v_total[-1] - v_total[1] > 1.0):
        raise AssertionError("Energy cost for large separation is too small to indicate confinement.")


def main() -> None:
    config = ConfinementConfig()
    validate_config(config)

    levels_df, dense_r, _ = solve_s_wave_levels(config)
    potential_df = build_potential_report(config)

    sigma_fit, intercept_fit, sigma_fit_r2 = fit_large_r_string_tension(dense_r, config)
    run_quality_checks(levels_df, potential_df, sigma_fit, sigma_fit_r2, config)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print("=== Quark Confinement MVP (Cornell Potential) ===")
    print(
        "config:",
        {
            "kappa_gev_fm": config.kappa_gev_fm,
            "sigma_gev_per_fm": config.sigma_gev_per_fm,
            "c0_gev": config.c0_gev,
            "quark_mass_gev": config.quark_mass_gev,
            "grid_points": config.grid_points,
            "n_levels": config.n_levels,
            "tail_fit_min_fm": config.tail_fit_min_fm,
        },
    )

    print("\nPotential / Force samples:")
    print(potential_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    print("\nLow-lying s-wave levels from finite-difference Schrodinger solve:")
    print(levels_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    print("\nLarge-r linear fit V(r) ~= sigma*r + b:")
    print(
        {
            "sigma_fit_gev_per_fm": round(sigma_fit, 6),
            "intercept_fit_gev": round(intercept_fit, 6),
            "fit_r2": round(sigma_fit_r2, 6),
            "target_sigma_gev_per_fm": config.sigma_gev_per_fm,
        }
    )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
