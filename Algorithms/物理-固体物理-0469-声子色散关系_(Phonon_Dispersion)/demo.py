"""Minimal runnable MVP for phonon dispersion in solid-state physics.

This script computes 1D phonon dispersion relations with explicit formulas:
- monatomic chain: one acoustic branch,
- diatomic chain: acoustic + optical branches.

No black-box phonon package is used; all physics formulas and checks are
implemented directly in source code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class DispersionConfig:
    """Physical and numerical parameters for the MVP."""

    lattice_constant_m: float = 5.43e-10
    spring_constant_n_per_m: float = 18.0

    # Monatomic chain mass (kg)
    mono_mass_kg: float = 4.663_706_6e-26

    # Diatomic chain masses (kg): choose different values to show a branch gap.
    mass_a_kg: float = 4.663_706_6e-26
    mass_b_kg: float = 1.162_377_3e-25

    n_q: int = 801
    small_q_fraction_for_linear_check: float = 0.06


def validate_config(cfg: DispersionConfig) -> None:
    if cfg.lattice_constant_m <= 0.0:
        raise ValueError("lattice_constant_m must be positive")
    if cfg.spring_constant_n_per_m <= 0.0:
        raise ValueError("spring_constant_n_per_m must be positive")
    if cfg.mono_mass_kg <= 0.0 or cfg.mass_a_kg <= 0.0 or cfg.mass_b_kg <= 0.0:
        raise ValueError("all masses must be positive")
    if cfg.n_q < 64:
        raise ValueError("n_q must be >= 64 for stable checks")
    if not (0.0 < cfg.small_q_fraction_for_linear_check < 0.2):
        raise ValueError("small_q_fraction_for_linear_check should be in (0, 0.2)")


def build_q_grid(cfg: DispersionConfig) -> np.ndarray:
    """Build q grid in first Brillouin zone [0, pi/a]."""
    q_max = math.pi / cfg.lattice_constant_m
    return np.linspace(0.0, q_max, cfg.n_q)


def monatomic_dispersion_1d(q: np.ndarray, cfg: DispersionConfig) -> np.ndarray:
    """Monatomic chain dispersion: omega = 2*sqrt(K/M)*|sin(qa/2)|."""
    prefactor = 2.0 * math.sqrt(cfg.spring_constant_n_per_m / cfg.mono_mass_kg)
    return prefactor * np.abs(np.sin(0.5 * q * cfg.lattice_constant_m))


def monatomic_group_velocity_1d(q: np.ndarray, cfg: DispersionConfig) -> np.ndarray:
    """Group velocity for q in [0, pi/a]: d omega/d q = a*sqrt(K/M)*cos(qa/2)."""
    vs = cfg.lattice_constant_m * math.sqrt(cfg.spring_constant_n_per_m / cfg.mono_mass_kg)
    return vs * np.cos(0.5 * q * cfg.lattice_constant_m)


def diatomic_dispersion_1d(q: np.ndarray, cfg: DispersionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Diatomic chain acoustic/optical branches with nearest-neighbor springs.

    Formula:
      omega^2_{+/-}(q) = A +/- sqrt(A^2 - B * sin^2(qa/2))
    where
      A = K*(1/m1 + 1/m2), B = 4*K^2/(m1*m2)

    '-' gives the acoustic branch and '+' gives the optical branch.
    """
    k = cfg.spring_constant_n_per_m
    m1 = cfg.mass_a_kg
    m2 = cfg.mass_b_kg

    a_term = k * (1.0 / m1 + 1.0 / m2)
    b_term = 4.0 * (k**2) / (m1 * m2)
    s2 = np.sin(0.5 * q * cfg.lattice_constant_m) ** 2

    radicand = np.maximum(a_term * a_term - b_term * s2, 0.0)
    root = np.sqrt(radicand)

    omega2_acoustic = np.maximum(a_term - root, 0.0)
    omega2_optical = np.maximum(a_term + root, 0.0)

    return np.sqrt(omega2_acoustic), np.sqrt(omega2_optical)


def angular_to_thz(omega: np.ndarray) -> np.ndarray:
    return np.asarray(omega, dtype=float) / (TWO_PI * 1.0e12)


def build_sample_tables(
    q: np.ndarray,
    omega_mono: np.ndarray,
    v_mono: np.ndarray,
    omega_ac: np.ndarray,
    omega_op: np.ndarray,
    cfg: DispersionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.linspace(0, q.size - 1, 9, dtype=int)
    q_norm = q[idx] / (math.pi / cfg.lattice_constant_m)

    mono_df = pd.DataFrame(
        {
            "q_over_pi_per_a": q_norm,
            "omega_acoustic_THz": angular_to_thz(omega_mono[idx]),
            "v_group_km_per_s": v_mono[idx] / 1.0e3,
        }
    )

    di_df = pd.DataFrame(
        {
            "q_over_pi_per_a": q_norm,
            "omega_acoustic_THz": angular_to_thz(omega_ac[idx]),
            "omega_optical_THz": angular_to_thz(omega_op[idx]),
            "branch_gap_THz": angular_to_thz(omega_op[idx] - omega_ac[idx]),
        }
    )
    return mono_df, di_df


def run_sanity_checks(
    q: np.ndarray,
    omega_mono: np.ndarray,
    omega_ac: np.ndarray,
    omega_op: np.ndarray,
    cfg: DispersionConfig,
) -> dict[str, float]:
    # 1) Monatomic long-wave linearity: omega ~ v_s * q.
    v_sound = cfg.lattice_constant_m * math.sqrt(cfg.spring_constant_n_per_m / cfg.mono_mass_kg)
    q_cut = cfg.small_q_fraction_for_linear_check * (math.pi / cfg.lattice_constant_m)
    mask = q <= q_cut
    rel_linear_err = float(
        np.max(np.abs(omega_mono[mask] - v_sound * q[mask]) / np.maximum(omega_mono[mask], 1.0e-30))
    )

    # 2) Monatomic zone-boundary frequency.
    omega_boundary = float(omega_mono[-1])
    omega_boundary_expected = 2.0 * math.sqrt(cfg.spring_constant_n_per_m / cfg.mono_mass_kg)
    boundary_rel_err = abs(omega_boundary - omega_boundary_expected) / omega_boundary_expected

    # 3) Diatomic acoustic branch at Gamma should be ~0.
    acoustic_gamma = float(omega_ac[0])

    # 4) Diatomic optical Gamma frequency exact value.
    optical_gamma = float(omega_op[0])
    optical_gamma_expected = math.sqrt(
        2.0
        * cfg.spring_constant_n_per_m
        * (1.0 / cfg.mass_a_kg + 1.0 / cfg.mass_b_kg)
    )
    optical_gamma_rel_err = abs(optical_gamma - optical_gamma_expected) / optical_gamma_expected

    # 5) Acoustic branch must stay below optical branch.
    min_branch_gap = float(np.min(omega_op - omega_ac))

    assert rel_linear_err < 2.5e-3, f"small-q linearity failed: {rel_linear_err:.3e}"
    assert boundary_rel_err < 1.0e-12, f"zone-boundary mismatch: {boundary_rel_err:.3e}"
    assert acoustic_gamma < 1.0e-10, f"acoustic Gamma should be 0, got {acoustic_gamma:.3e}"
    assert optical_gamma_rel_err < 1.0e-12, f"optical Gamma mismatch: {optical_gamma_rel_err:.3e}"
    assert min_branch_gap > 0.0, "acoustic branch should stay below optical branch"

    return {
        "sound_velocity_m_per_s": v_sound,
        "small_q_linear_rel_err": rel_linear_err,
        "zone_boundary_rel_err": float(boundary_rel_err),
        "acoustic_gamma_rad_per_s": acoustic_gamma,
        "optical_gamma_rel_err": float(optical_gamma_rel_err),
        "min_branch_gap_THz": float(min_branch_gap / (TWO_PI * 1.0e12)),
    }


def main() -> None:
    cfg = DispersionConfig()
    validate_config(cfg)

    q = build_q_grid(cfg)

    omega_mono = monatomic_dispersion_1d(q, cfg)
    v_mono = monatomic_group_velocity_1d(q, cfg)

    omega_ac, omega_op = diatomic_dispersion_1d(q, cfg)

    mono_df, di_df = build_sample_tables(q, omega_mono, v_mono, omega_ac, omega_op, cfg)
    checks = run_sanity_checks(q, omega_mono, omega_ac, omega_op, cfg)

    print("=== Phonon Dispersion MVP (1D Harmonic Chains) ===")
    print(
        {
            "lattice_constant_m": cfg.lattice_constant_m,
            "spring_constant_N_per_m": cfg.spring_constant_n_per_m,
            "mono_mass_kg": cfg.mono_mass_kg,
            "mass_a_kg": cfg.mass_a_kg,
            "mass_b_kg": cfg.mass_b_kg,
            "n_q": cfg.n_q,
            "q_max_pi_over_a": 1.0,
            "mono_omega_max_THz": float(angular_to_thz(np.array([omega_mono.max()]))[0]),
            "di_optical_gamma_THz": float(angular_to_thz(np.array([omega_op[0]]))[0]),
        }
    )

    print("\n[monatomic_samples]")
    with pd.option_context("display.width", 140, "display.max_rows", 20):
        print(mono_df.to_string(index=False, float_format=lambda x: f"{x:12.6f}"))

    print("\n[diatomic_samples]")
    with pd.option_context("display.width", 140, "display.max_rows", 20):
        print(di_df.to_string(index=False, float_format=lambda x: f"{x:12.6f}"))

    print("\n[sanity_checks]")
    for key, value in checks.items():
        print(f"- {key}: {value:.6e}")


if __name__ == "__main__":
    main()
