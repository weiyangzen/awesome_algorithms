"""Phonon MVP: 1D monatomic lattice dispersion and thermodynamics.

This script implements a transparent, non-black-box phonon calculation pipeline:
1) build first-Brillouin-zone q-grid,
2) compute acoustic-branch dispersion omega(q),
3) evaluate Bose occupation and mode heat capacity,
4) aggregate per-atom thermodynamic curves versus temperature,
5) run physics sanity checks.

Model scope: harmonic nearest-neighbor monatomic chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

HBAR = 1.054_571_817e-34  # J*s
K_B = 1.380_649e-23  # J/K
TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class PhononConfig:
    """Physical and numerical settings for the MVP."""

    spring_constant_n_per_m: float = 20.0
    atomic_mass_kg: float = 4.663_706_6e-26
    lattice_constant_m: float = 5.43e-10

    n_q: int = 4096

    t_min_k: float = 5.0
    t_max_k: float = 1200.0
    n_temps: int = 90

    low_t1_k: float = 6.0
    low_t2_k: float = 12.0
    high_t_check_k: float = 5000.0

    small_q_fraction_of_bz: float = 0.05


def validate_config(cfg: PhononConfig) -> None:
    if cfg.spring_constant_n_per_m <= 0.0:
        raise ValueError("spring_constant_n_per_m must be positive")
    if cfg.atomic_mass_kg <= 0.0:
        raise ValueError("atomic_mass_kg must be positive")
    if cfg.lattice_constant_m <= 0.0:
        raise ValueError("lattice_constant_m must be positive")
    if cfg.n_q < 256:
        raise ValueError("n_q is too small for stable Brillouin-zone averaging")
    if cfg.t_min_k <= 0.0 or cfg.t_max_k <= cfg.t_min_k:
        raise ValueError("temperature range must satisfy 0 < t_min_k < t_max_k")
    if cfg.n_temps < 8:
        raise ValueError("n_temps must be at least 8")
    if cfg.low_t1_k <= 0.0 or cfg.low_t2_k <= cfg.low_t1_k:
        raise ValueError("low_t points must satisfy 0 < low_t1_k < low_t2_k")
    if cfg.high_t_check_k <= cfg.low_t2_k:
        raise ValueError("high_t_check_k must be larger than low-temperature checks")
    if not (0.0 < cfg.small_q_fraction_of_bz < 0.2):
        raise ValueError("small_q_fraction_of_bz should be in (0, 0.2)")


def sound_velocity(cfg: PhononConfig) -> float:
    """Long-wavelength acoustic speed v_s = a * sqrt(K/M)."""
    return cfg.lattice_constant_m * math.sqrt(cfg.spring_constant_n_per_m / cfg.atomic_mass_kg)


def omega_max(cfg: PhononConfig) -> float:
    """Upper edge of acoustic branch: omega_max = 2 * sqrt(K/M)."""
    return 2.0 * math.sqrt(cfg.spring_constant_n_per_m / cfg.atomic_mass_kg)


def effective_debye_temperature(cfg: PhononConfig) -> float:
    return HBAR * omega_max(cfg) / K_B


def build_q_grid(cfg: PhononConfig) -> np.ndarray:
    """Midpoint grid over the first Brillouin zone [-pi/a, pi/a).

    Midpoint sampling avoids exactly hitting q=0 where the strict harmonic
    oscillator formula degenerates for the translational mode.
    """
    q_min = -math.pi / cfg.lattice_constant_m
    q_max = math.pi / cfg.lattice_constant_m
    dq = (q_max - q_min) / cfg.n_q
    return q_min + (np.arange(cfg.n_q) + 0.5) * dq


def phonon_dispersion_monatomic_1d(q: np.ndarray, cfg: PhononConfig) -> np.ndarray:
    """Acoustic branch of a 1D nearest-neighbor monatomic chain.

    omega(q) = 2 * sqrt(K/M) * |sin(qa/2)|
    """
    prefactor = 2.0 * math.sqrt(cfg.spring_constant_n_per_m / cfg.atomic_mass_kg)
    return prefactor * np.abs(np.sin(0.5 * q * cfg.lattice_constant_m))


def group_velocity_monatomic_1d(q: np.ndarray, cfg: PhononConfig) -> np.ndarray:
    """Analytic group velocity d omega / d q for the acoustic branch."""
    prefactor = cfg.lattice_constant_m * math.sqrt(cfg.spring_constant_n_per_m / cfg.atomic_mass_kg)
    phase = 0.5 * q * cfg.lattice_constant_m
    return prefactor * np.cos(phase) * np.sign(np.sin(phase))


def _heat_capacity_kernel(x: np.ndarray) -> np.ndarray:
    """Dimensionless kernel g(x)=x^2 exp(x)/(exp(x)-1)^2 with stable numerics."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    small = x < 1.0e-6
    out[small] = 1.0 - (x[small] ** 2) / 12.0

    if np.any(~small):
        xm = x[~small]
        exp_minus = np.exp(-np.clip(xm, 0.0, 120.0))
        denom = -np.expm1(-xm)  # 1 - exp(-x), stable near zero
        out[~small] = (xm * xm) * exp_minus / (denom * denom)

    return out


def phonon_thermo_curves(
    omega_q: np.ndarray,
    temperatures_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute per-atom thermal energy and heat capacity curves.

    Returns
    -------
    thermal_energy : np.ndarray
        U_th(T) per atom in J.
    heat_capacity : np.ndarray
        C(T) per atom in J/K.
    zero_point_energy : float
        U_zp per atom in J.
    """
    omega = np.asarray(omega_q, dtype=float)
    temps = np.asarray(temperatures_k, dtype=float)

    if np.any(omega < 0.0):
        raise ValueError("omega_q must be non-negative")
    if np.any(temps <= 0.0):
        raise ValueError("temperatures must be positive")

    x = (HBAR * omega[None, :]) / (K_B * temps[:, None])

    bose = 1.0 / np.expm1(np.clip(x, 1.0e-12, 120.0))
    thermal_energy = np.mean(HBAR * omega[None, :] * bose, axis=1)

    kernel = _heat_capacity_kernel(x)
    heat_capacity = K_B * np.mean(kernel, axis=1)

    zero_point_energy = 0.5 * float(np.mean(HBAR * omega))
    return thermal_energy, heat_capacity, zero_point_energy


def build_dispersion_sample_table(q: np.ndarray, omega_q: np.ndarray, v_q: np.ndarray, cfg: PhononConfig) -> pd.DataFrame:
    idx = np.linspace(0, q.size - 1, 9, dtype=int)
    return pd.DataFrame(
        {
            "q_over_pi_per_a": q[idx] / (math.pi / cfg.lattice_constant_m),
            "omega_THz": omega_q[idx] / (TWO_PI * 1e12),
            "v_group_km_per_s": v_q[idx] / 1e3,
        }
    )


def build_thermo_table(cfg: PhononConfig, omega_q: np.ndarray) -> pd.DataFrame:
    temperatures = np.geomspace(cfg.t_min_k, cfg.t_max_k, cfg.n_temps)
    u_th, c_t, u_zp = phonon_thermo_curves(omega_q, temperatures)

    return pd.DataFrame(
        {
            "T_K": temperatures,
            "U_th_meV_per_atom": (u_th / 1.602176634e-22),
            "C_over_kB": c_t / K_B,
            "C_1e23J_per_atomK": c_t * 1.0e23,
            "U_zp_meV_per_atom": np.full_like(temperatures, u_zp / 1.602176634e-22),
        }
    )


def run_sanity_checks(cfg: PhononConfig, q: np.ndarray, omega_q: np.ndarray, thermo_df: pd.DataFrame) -> dict[str, float]:
    vs = sound_velocity(cfg)

    # 1) Long-wavelength linear dispersion check: omega ~= v_s * |q|.
    q_cut = cfg.small_q_fraction_of_bz * (math.pi / cfg.lattice_constant_m)
    small_mask = np.abs(q) <= q_cut
    omega_small = omega_q[small_mask]
    omega_linear = vs * np.abs(q[small_mask])
    rel_lin = float(
        np.max(
            np.abs(omega_small - omega_linear)
            / np.maximum(omega_small, 1.0e-30)
        )
    )

    # 2) Low-T scaling in 1D acoustic branch: C(T) ~ T.
    low_t = np.array([cfg.low_t1_k, cfg.low_t2_k], dtype=float)
    _, c_low, _ = phonon_thermo_curves(omega_q, low_t)
    ratio_c_over_t = float((c_low[1] / low_t[1]) / (c_low[0] / low_t[0]))

    # 3) High-T equipartition: C -> k_B per atom for one acoustic branch in 1D.
    _, c_high_arr, _ = phonon_thermo_curves(omega_q, np.array([cfg.high_t_check_k], dtype=float))
    c_high_over_kb = float(c_high_arr[0] / K_B)

    # 4) Positivity in report range.
    min_c_over_kb = float(thermo_df["C_over_kB"].min())

    assert rel_lin < 2.0e-3, f"small-q linear dispersion error too large: {rel_lin:.3e}"
    assert abs(ratio_c_over_t - 1.0) < 0.12, f"low-T C(T)~T check failed: ratio={ratio_c_over_t:.5f}"
    assert abs(c_high_over_kb - 1.0) < 0.03, f"high-T equipartition failed: C/kB={c_high_over_kb:.5f}"
    assert min_c_over_kb > 0.0, "heat capacity must stay positive"

    return {
        "small_q_linear_rel_err": rel_lin,
        "low_t_ratio_of_C_over_T": ratio_c_over_t,
        "high_t_C_over_kB": c_high_over_kb,
        "min_C_over_kB": min_c_over_kb,
    }


def main() -> None:
    cfg = PhononConfig()
    validate_config(cfg)

    q = build_q_grid(cfg)
    omega_q = phonon_dispersion_monatomic_1d(q, cfg)
    v_q = group_velocity_monatomic_1d(q, cfg)

    dispersion_table = build_dispersion_sample_table(q, omega_q, v_q, cfg)
    thermo_df = build_thermo_table(cfg, omega_q)

    checks = run_sanity_checks(cfg, q, omega_q, thermo_df)

    print("=== Phonon MVP: 1D Monatomic Lattice ===")
    print(
        {
            "spring_constant_N_per_m": cfg.spring_constant_n_per_m,
            "atomic_mass_kg": cfg.atomic_mass_kg,
            "lattice_constant_m": cfg.lattice_constant_m,
            "n_q": cfg.n_q,
            "temperature_range_K": [cfg.t_min_k, cfg.t_max_k],
            "n_temps": cfg.n_temps,
            "omega_max_THz": omega_max(cfg) / (TWO_PI * 1e12),
            "effective_theta_D_K": effective_debye_temperature(cfg),
            "sound_velocity_m_per_s": sound_velocity(cfg),
        }
    )

    print("\n[dispersion_samples]")
    with pd.option_context("display.width", 120, "display.max_rows", 20):
        print(dispersion_table.to_string(index=False, float_format=lambda x: f"{x:10.6f}"))

    print("\n[thermo_samples]")
    idx = np.linspace(0, len(thermo_df) - 1, 10, dtype=int)
    sample_df = thermo_df.iloc[idx].copy()
    with pd.option_context("display.width", 140, "display.max_rows", 20):
        print(sample_df.to_string(index=False, float_format=lambda x: f"{x:12.6f}"))

    print("\n[sanity_checks]")
    for key, value in checks.items():
        print(f"- {key}: {value:.6e}")


if __name__ == "__main__":
    main()
