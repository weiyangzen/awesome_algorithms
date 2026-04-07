"""Minimal runnable MVP for Bethe-Salpeter Equation (PHYS-0216).

This demo implements a pedagogical excitonic Bethe-Salpeter solver in the
Tamm-Dancoff approximation (TDA) on a 1D k-grid:
1) build independent quasiparticle transition energies DeltaE(k)
2) build direct attractive + exchange repulsive electron-hole kernel
3) assemble H_BSE = diag(DeltaE) + K and solve exciton eigenmodes
4) compute oscillator strengths and compare interacting vs independent spectra
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh


@dataclass(frozen=True)
class BSEConfig:
    """Configuration for a minimal excitonic BSE model."""

    nk: int = 61
    k_max: float = 0.35

    qp_gap: float = 2.20
    alpha_c: float = 10.0
    alpha_v: float = 8.0

    direct_strength: float = 1.20
    exchange_strength: float = 0.18
    sigma_direct: float = 0.07
    sigma_exchange: float = 0.16

    dipole_width: float = 0.12
    dipole_scale: float = 1.0

    broadening: float = 0.03
    omega_min: float = 1.80
    omega_max: float = 2.70
    n_omega: int = 1200

    n_report_states: int = 8


def build_k_grid(cfg: BSEConfig) -> np.ndarray:
    if cfg.nk < 9:
        raise ValueError("nk must be >= 9")
    if cfg.nk % 2 == 0:
        raise ValueError("nk must be odd for symmetric k-grid around 0")
    if cfg.k_max <= 0.0:
        raise ValueError("k_max must be positive")
    return np.linspace(-cfg.k_max, cfg.k_max, cfg.nk, dtype=float)


def quasiparticle_transitions(k_grid: np.ndarray, cfg: BSEConfig) -> np.ndarray:
    if cfg.qp_gap <= 0.0:
        raise ValueError("qp_gap must be positive")
    if cfg.alpha_c < 0.0 or cfg.alpha_v < 0.0:
        raise ValueError("alpha_c and alpha_v must be non-negative")
    curvature = cfg.alpha_c + cfg.alpha_v
    return cfg.qp_gap + curvature * k_grid * k_grid


def dipole_profile(k_grid: np.ndarray, cfg: BSEConfig) -> np.ndarray:
    if cfg.dipole_width <= 0.0:
        raise ValueError("dipole_width must be positive")
    profile = np.exp(-0.5 * (k_grid / cfg.dipole_width) ** 2)
    return cfg.dipole_scale * profile


def interaction_kernel(k_grid: np.ndarray, cfg: BSEConfig) -> np.ndarray:
    if cfg.direct_strength <= 0.0:
        raise ValueError("direct_strength must be positive")
    if cfg.exchange_strength < 0.0:
        raise ValueError("exchange_strength must be non-negative")
    if cfg.sigma_direct <= 0.0 or cfg.sigma_exchange <= 0.0:
        raise ValueError("Kernel widths must be positive")

    delta_k = k_grid[:, None] - k_grid[None, :]

    direct = -cfg.direct_strength * np.exp(-0.5 * (delta_k / cfg.sigma_direct) ** 2)
    exchange = cfg.exchange_strength * np.exp(-0.5 * (delta_k / cfg.sigma_exchange) ** 2)

    return (direct + exchange) / float(len(k_grid))


def build_bse_hamiltonian(transitions: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if transitions.ndim != 1:
        raise ValueError("transitions must be 1D")
    if kernel.shape != (transitions.size, transitions.size):
        raise ValueError("Kernel shape mismatch")
    return np.diag(transitions) + kernel


def solve_bse(hamiltonian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.all(np.isfinite(hamiltonian)):
        raise ValueError("Hamiltonian contains non-finite values")
    eigvals, eigvecs = eigh(hamiltonian)
    return eigvals, eigvecs


def oscillator_strengths(dipole: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    amplitudes = dipole @ eigvecs
    return np.abs(amplitudes) ** 2 / float(dipole.size)


def lorentzian_spectrum(
    omega_grid: np.ndarray,
    energies: np.ndarray,
    weights: np.ndarray,
    gamma: float,
) -> np.ndarray:
    if gamma <= 0.0:
        raise ValueError("broadening gamma must be positive")
    delta = omega_grid[:, None] - energies[None, :]
    kernel = gamma / np.pi / (delta * delta + gamma * gamma)
    return kernel @ weights


def run_bse(cfg: BSEConfig) -> dict[str, object]:
    k_grid = build_k_grid(cfg)
    transitions = quasiparticle_transitions(k_grid, cfg)
    dipole = dipole_profile(k_grid, cfg)

    kernel = interaction_kernel(k_grid, cfg)
    h_bse = build_bse_hamiltonian(transitions, kernel)
    exciton_energies, exciton_vectors = solve_bse(h_bse)
    osc = oscillator_strengths(dipole, exciton_vectors)

    omega_grid = np.linspace(cfg.omega_min, cfg.omega_max, cfg.n_omega, dtype=float)
    interacting_absorption = lorentzian_spectrum(
        omega_grid=omega_grid,
        energies=exciton_energies,
        weights=osc,
        gamma=cfg.broadening,
    )

    independent_weights = (dipole * dipole) / float(cfg.nk)
    independent_absorption = lorentzian_spectrum(
        omega_grid=omega_grid,
        energies=transitions,
        weights=independent_weights,
        gamma=cfg.broadening,
    )

    qp_edge = float(np.min(transitions))
    bright_index = int(np.argmax(osc))
    bright_energy = float(exciton_energies[bright_index])
    bright_binding = float(qp_edge - bright_energy)

    interacting_peak_energy = float(omega_grid[int(np.argmax(interacting_absorption))])
    independent_peak_energy = float(omega_grid[int(np.argmax(independent_absorption))])
    redshift = float(independent_peak_energy - interacting_peak_energy)

    report_n = min(cfg.n_report_states, cfg.nk)
    report_rows = []
    for idx in range(report_n):
        report_rows.append(
            {
                "state": idx,
                "Omega_exciton": float(exciton_energies[idx]),
                "binding_vs_qp_edge": float(qp_edge - exciton_energies[idx]),
                "oscillator_strength": float(osc[idx]),
                "is_bright": bool(osc[idx] > 1.0e-3),
            }
        )
    exciton_table = pd.DataFrame(report_rows)

    symmetry_error = float(np.max(np.abs(h_bse - h_bse.T)))
    norm_error = float(np.max(np.abs(np.sum(exciton_vectors * exciton_vectors, axis=0) - 1.0)))
    sum_rule_error = float(abs(np.sum(osc) - np.sum(independent_weights)))

    summary = {
        "qp_edge": qp_edge,
        "lowest_exciton": float(exciton_energies[0]),
        "lowest_binding": float(qp_edge - exciton_energies[0]),
        "bright_state": bright_index,
        "bright_energy": bright_energy,
        "bright_binding": bright_binding,
        "interacting_peak_energy": interacting_peak_energy,
        "independent_peak_energy": independent_peak_energy,
        "peak_redshift": redshift,
        "max_hamiltonian_asymmetry": symmetry_error,
        "max_eigenvector_norm_error": norm_error,
        "oscillator_sum_rule_error": sum_rule_error,
    }

    return {
        "cfg": cfg,
        "k_grid": k_grid,
        "transitions": transitions,
        "dipole": dipole,
        "kernel": kernel,
        "hamiltonian": h_bse,
        "exciton_energies": exciton_energies,
        "exciton_vectors": exciton_vectors,
        "oscillator_strengths": osc,
        "omega_grid": omega_grid,
        "interacting_absorption": interacting_absorption,
        "independent_absorption": independent_absorption,
        "exciton_table": exciton_table,
        "summary": summary,
    }


def main() -> None:
    cfg = BSEConfig()
    result = run_bse(cfg)
    summary = result["summary"]
    exciton_table = result["exciton_table"]

    checks = {
        "Hamiltonian symmetry error < 1e-12": summary["max_hamiltonian_asymmetry"] < 1.0e-12,
        "Eigenvector normalization error < 1e-10": summary["max_eigenvector_norm_error"] < 1.0e-10,
        "Oscillator sum-rule error < 1e-10": summary["oscillator_sum_rule_error"] < 1.0e-10,
        "Lowest exciton below QP edge": summary["lowest_exciton"] < summary["qp_edge"],
        "Bright exciton binding in (0.05, 0.50) eV": 0.05 < summary["bright_binding"] < 0.50,
        "Interacting peak redshift > 0": summary["peak_redshift"] > 0.0,
        "Absorption arrays are finite": bool(
            np.isfinite(result["interacting_absorption"]).all() and np.isfinite(result["independent_absorption"]).all()
        ),
    }

    pd.set_option("display.float_format", lambda x: f"{x:.8f}")

    print("=== Bethe-Salpeter Equation MVP (PHYS-0216) ===")
    print(
        f"nk={cfg.nk}, k in [-{cfg.k_max:.3f}, {cfg.k_max:.3f}], "
        f"QP gap={cfg.qp_gap:.4f} eV, alpha_c+alpha_v={cfg.alpha_c + cfg.alpha_v:.4f}"
    )
    print(
        f"kernel: direct={cfg.direct_strength:.4f}, exchange={cfg.exchange_strength:.4f}, "
        f"sigma_d={cfg.sigma_direct:.4f}, sigma_x={cfg.sigma_exchange:.4f}"
    )

    print("\nLowest exciton states:")
    print(exciton_table.to_string(index=False))

    summary_table = pd.DataFrame(
        {
            "quantity": [
                "QP edge",
                "lowest exciton",
                "lowest binding",
                "bright state index",
                "bright exciton energy",
                "bright binding",
                "independent absorption peak",
                "interacting absorption peak",
                "peak redshift (independent - interacting)",
                "max Hamiltonian asymmetry",
                "max eigenvector norm error",
                "oscillator sum-rule error",
            ],
            "value": [
                summary["qp_edge"],
                summary["lowest_exciton"],
                summary["lowest_binding"],
                float(summary["bright_state"]),
                summary["bright_energy"],
                summary["bright_binding"],
                summary["independent_peak_energy"],
                summary["interacting_peak_energy"],
                summary["peak_redshift"],
                summary["max_hamiltonian_asymmetry"],
                summary["max_eigenvector_norm_error"],
                summary["oscillator_sum_rule_error"],
            ],
        }
    )

    print("\nBSE summary:")
    print(summary_table.to_string(index=False))

    print("\nThreshold checks:")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(checks.values())
    print(f"\nValidation: {'PASS' if all_passed else 'FAIL'}")

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
