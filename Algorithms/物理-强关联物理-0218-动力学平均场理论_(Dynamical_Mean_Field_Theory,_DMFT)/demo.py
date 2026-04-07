"""Minimal, runnable DMFT MVP (single-band Hubbard model on Bethe lattice).

This script implements a pedagogical DMFT loop at half filling:
1) fit a one-pole bath to current hybridization,
2) solve a tiny Anderson impurity model by exact diagonalization,
3) update self-energy and lattice Green's function,
4) iterate to self-consistency.

No interactive input is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ComplexArray = np.ndarray


@dataclass
class DMFTConfig:
    """Control parameters for the pedagogical DMFT run."""

    U: float = 2.2
    beta: float = 30.0
    n_iw: int = 96
    half_bandwidth: float = 1.0
    mix: float = 0.55
    tol: float = 1e-4
    max_iter: int = 60
    n_fit: int = 40


def matsubara_frequencies(beta: float, n_iw: int) -> ComplexArray:
    n = np.arange(n_iw)
    return 1j * (2 * n + 1) * math.pi / beta


def _fermion_sign(state: int, orbital: int) -> float:
    lower = state & ((1 << orbital) - 1)
    return -1.0 if (lower.bit_count() % 2) else 1.0


def build_fermion_operators(num_orbitals: int = 4) -> tuple[list[ComplexArray], list[ComplexArray], list[ComplexArray], ComplexArray]:
    """Build creation/annihilation/number operators in Fock basis."""

    dim = 1 << num_orbitals
    creation: list[ComplexArray] = []

    for orb in range(num_orbitals):
        op = np.zeros((dim, dim), dtype=np.complex128)
        for state in range(dim):
            if ((state >> orb) & 1) == 0:
                new_state = state | (1 << orb)
                op[new_state, state] = _fermion_sign(state, orb)
        creation.append(op)

    annihilation = [c_dag.conj().T for c_dag in creation]
    number = [creation[i] @ annihilation[i] for i in range(num_orbitals)]
    identity = np.eye(dim, dtype=np.complex128)
    return creation, annihilation, number, identity


def build_hamiltonian(
    U: float,
    eps_b: float,
    V: float,
    operators: tuple[list[ComplexArray], list[ComplexArray], list[ComplexArray], ComplexArray],
) -> ComplexArray:
    """Two-site Anderson impurity Hamiltonian (impurity + one bath level)."""

    c_dag, c, n, identity = operators

    h_int = U * (n[0] - 0.5 * identity) @ (n[1] - 0.5 * identity)
    h_bath = eps_b * (n[2] + n[3])
    h_hyb = V * (c_dag[0] @ c[2] + c_dag[2] @ c[0] + c_dag[1] @ c[3] + c_dag[3] @ c[1])

    h_total = h_int + h_bath + h_hyb
    return 0.5 * (h_total + h_total.conj().T)


def thermal_expectation(
    evals: ComplexArray,
    evecs: ComplexArray,
    operator: ComplexArray,
    beta: float,
) -> float:
    e0 = float(np.min(evals).real)
    boltz = np.exp(-beta * (evals.real - e0))
    z_part = float(np.sum(boltz))

    op_eig = evecs.conj().T @ operator @ evecs
    exp_val = float(np.sum(boltz * np.real(np.diag(op_eig))) / z_part)
    return exp_val


def impurity_green_function(
    evals: ComplexArray,
    evecs: ComplexArray,
    annihilation_op: ComplexArray,
    beta: float,
    iwn: ComplexArray,
) -> ComplexArray:
    """Lehmann representation for Matsubara Green's function."""

    e0 = float(np.min(evals).real)
    boltz = np.exp(-beta * (evals.real - e0))
    z_part = float(np.sum(boltz))

    c_eig = evecs.conj().T @ annihilation_op @ evecs
    dE = evals.real[:, None] - evals.real[None, :]
    spectral_weight = np.abs(c_eig) ** 2 * (boltz[:, None] + boltz[None, :])

    denom = iwn[:, None, None] + dE[None, :, :]
    return np.sum(spectral_weight[None, :, :] / denom, axis=(1, 2)) / z_part


def fit_single_pole_hybridization(
    delta_target: ComplexArray,
    iwn: ComplexArray,
    eps_init: float,
    V_init: float,
    half_bandwidth: float,
    n_fit: int,
) -> tuple[float, float, bool, float]:
    """Fit Delta(iwn) ≈ V^2 / (iwn - eps_b) over low Matsubara frequencies."""

    n_use = min(n_fit, len(iwn))
    iwn_fit = iwn[:n_use]
    target_fit = delta_target[:n_use]
    weights = 1.0 / (1.0 + np.arange(n_use))

    def objective(x: np.ndarray) -> float:
        eps_b, V = float(x[0]), float(x[1])
        if V <= 1e-8:
            return 1e9
        model = (V * V) / (iwn_fit - eps_b)
        diff = target_fit - model
        return float(np.mean(weights * np.abs(diff) ** 2))

    bounds = [(-2.0 * half_bandwidth, 2.0 * half_bandwidth), (1e-6, 2.5 * half_bandwidth)]
    x0 = np.array([eps_init, max(V_init, 1e-4)], dtype=float)

    result = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds)
    if result.success:
        eps_b, V = float(result.x[0]), float(result.x[1])
    else:
        eps_b, V = float(eps_init), float(V_init)

    return eps_b, V, bool(result.success), objective(np.array([eps_b, V]))


def solve_impurity_ed(
    U: float,
    eps_b: float,
    V: float,
    beta: float,
    iwn: ComplexArray,
    operators: tuple[list[ComplexArray], list[ComplexArray], list[ComplexArray], ComplexArray],
) -> tuple[ComplexArray, float]:
    """Solve impurity by exact diagonalization of a 16x16 Hamiltonian."""

    c_dag, c, n, _ = operators
    hamiltonian = build_hamiltonian(U, eps_b, V, operators)
    evals, evecs = np.linalg.eigh(hamiltonian)

    g_up = impurity_green_function(evals, evecs, c[0], beta, iwn)
    g_dn = impurity_green_function(evals, evecs, c[1], beta, iwn)
    g_imp = 0.5 * (g_up + g_dn)

    double_occ = thermal_expectation(evals, evecs, n[0] @ n[1], beta)
    return g_imp, double_occ


def bethe_lattice_green(iwn: ComplexArray, sigma_iw: ComplexArray, hopping_t: float) -> ComplexArray:
    """Compute local lattice Green's function for Bethe lattice via quadratic equation."""

    z = iwn - sigma_iw
    root = np.sqrt(z * z - (2.0 * hopping_t) ** 2)

    g1 = (z - root) / (2.0 * hopping_t * hopping_t)
    g2 = (z + root) / (2.0 * hopping_t * hopping_t)

    high_freq = 1.0 / z
    choose_g1 = np.abs(g1 - high_freq) <= np.abs(g2 - high_freq)
    return np.where(choose_g1, g1, g2)


def run_dmft(config: DMFTConfig) -> dict[str, object]:
    """Run the DMFT self-consistency loop and return diagnostics."""

    operators = build_fermion_operators(num_orbitals=4)
    iwn = matsubara_frequencies(beta=config.beta, n_iw=config.n_iw)

    hopping_t = config.half_bandwidth / 2.0
    delta_iw = (hopping_t * hopping_t) / iwn

    eps_b, V = 0.0, hopping_t
    converged = False
    fit_success_count = 0
    err = float("inf")

    for it in range(1, config.max_iter + 1):
        eps_b, V, fit_ok, fit_loss = fit_single_pole_hybridization(
            delta_target=delta_iw,
            iwn=iwn,
            eps_init=eps_b,
            V_init=V,
            half_bandwidth=config.half_bandwidth,
            n_fit=config.n_fit,
        )
        fit_success_count += int(fit_ok)

        g_imp, double_occ = solve_impurity_ed(
            U=config.U,
            eps_b=eps_b,
            V=V,
            beta=config.beta,
            iwn=iwn,
            operators=operators,
        )

        sigma_iw = iwn - delta_iw - 1.0 / g_imp
        g_loc = bethe_lattice_green(iwn=iwn, sigma_iw=sigma_iw, hopping_t=hopping_t)
        delta_next = (hopping_t * hopping_t) * g_loc

        err = float(np.max(np.abs(delta_next - delta_iw)))
        delta_iw = config.mix * delta_next + (1.0 - config.mix) * delta_iw

        if err < config.tol:
            converged = True
            break

    omega = iwn.imag
    slope = float((sigma_iw.imag[1] - sigma_iw.imag[0]) / (omega[1] - omega[0]))
    z_qp = float(1.0 / (1.0 - slope)) if abs(1.0 - slope) > 1e-12 else float("inf")

    return {
        "config": config,
        "iterations": it,
        "converged": converged,
        "max_delta_error": err,
        "fit_success_ratio": fit_success_count / it,
        "fit_loss_last": fit_loss,
        "eps_b": eps_b,
        "V": V,
        "double_occupancy": double_occ,
        "Z_est": z_qp,
        "iwn": iwn,
        "sigma_iw": sigma_iw,
        "g_imp": g_imp,
        "delta_iw": delta_iw,
    }


def main() -> None:
    config = DMFTConfig()
    result = run_dmft(config)

    iwn = result["iwn"]
    sigma_iw = result["sigma_iw"]
    g_imp = result["g_imp"]

    rows = 8
    report = pd.DataFrame(
        {
            "n": np.arange(rows),
            "omega_n": iwn.imag[:rows],
            "Im_G_imp": g_imp.imag[:rows],
            "Im_Sigma": sigma_iw.imag[:rows],
        }
    )

    print("=== DMFT MVP (Bethe lattice, one-bath-site ED solver) ===")
    print(f"U={config.U:.3f}, beta={config.beta:.2f}, n_iw={config.n_iw}")
    print(
        "converged=",
        result["converged"],
        f"iterations={result['iterations']}",
        f"max|Δ_new-Δ_old|={result['max_delta_error']:.3e}",
    )
    print(
        f"fitted bath: eps_b={result['eps_b']:.5f}, V={result['V']:.5f}, "
        f"fit_loss={result['fit_loss_last']:.3e}, fit_success_ratio={result['fit_success_ratio']:.2f}"
    )
    print(
        f"double occupancy=<n_up n_dn>={result['double_occupancy']:.6f}, "
        f"quasiparticle Z_est={result['Z_est']:.6f}"
    )
    print("\nLow-frequency Matsubara summary:")
    print(report.to_string(index=False, justify="right", float_format=lambda x: f"{x: .6f}"))

    assert np.isfinite(result["max_delta_error"])
    assert np.all(np.isfinite(g_imp.real)) and np.all(np.isfinite(g_imp.imag))
    assert np.all(np.isfinite(sigma_iw.real)) and np.all(np.isfinite(sigma_iw.imag))


if __name__ == "__main__":
    main()
