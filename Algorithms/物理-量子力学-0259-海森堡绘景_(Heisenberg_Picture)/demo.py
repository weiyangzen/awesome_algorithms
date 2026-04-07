"""Minimal runnable MVP for Heisenberg Picture.

This script demonstrates a spin-1/2 system with Hamiltonian
    H = (omega / 2) * sigma_z
and validates three facts:
1) Heisenberg-picture and Schrodinger-picture expectations are identical.
2) Numerical expectations match analytic solution (cos/sin precession).
3) Heisenberg equation dA/dt = i[H, A] holds numerically.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def pauli_matrices() -> Dict[str, np.ndarray]:
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    eye = np.eye(2, dtype=np.complex128)
    return {"sx": sx, "sy": sy, "sz": sz, "I": eye}


def unitary_from_hamiltonian(H: np.ndarray, t: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(H)
    phases = np.exp(-1.0j * eigvals * t)
    U = eigvecs @ np.diag(phases) @ eigvecs.conj().T
    return U


def heisenberg_operator(A: np.ndarray, H: np.ndarray, t: float) -> np.ndarray:
    U = unitary_from_hamiltonian(H, t)
    return U.conj().T @ A @ U


def schrodinger_density(rho0: np.ndarray, H: np.ndarray, t: float) -> np.ndarray:
    U = unitary_from_hamiltonian(H, t)
    return U @ rho0 @ U.conj().T


def expectation(rho: np.ndarray, A: np.ndarray) -> float:
    value = np.trace(rho @ A)
    return float(np.real(value))


def max_unitarity_error(H: np.ndarray, time_grid: np.ndarray) -> float:
    errors = []
    I = np.eye(H.shape[0], dtype=np.complex128)
    for t in time_grid:
        U = unitary_from_hamiltonian(H, float(t))
        errors.append(float(np.linalg.norm(U.conj().T @ U - I, ord="fro")))
    return max(errors)


def heisenberg_equation_residual(
    A: np.ndarray,
    H: np.ndarray,
    t0: float,
    dt: float,
) -> float:
    A_plus = heisenberg_operator(A, H, t0 + dt)
    A_minus = heisenberg_operator(A, H, t0 - dt)
    lhs = (A_plus - A_minus) / (2.0 * dt)

    A_t = heisenberg_operator(A, H, t0)
    rhs = 1.0j * (H @ A_t - A_t @ H)
    return float(np.linalg.norm(lhs - rhs, ord="fro"))


def analytic_expectations(omega: float, t: float) -> Tuple[float, float, float]:
    return math.cos(omega * t), math.sin(omega * t), 0.0


def main() -> None:
    mats = pauli_matrices()
    sx, sy, sz = mats["sx"], mats["sy"], mats["sz"]

    omega = 1.7
    H = 0.5 * omega * sz

    ket_plus_x = (1.0 / math.sqrt(2.0)) * np.array([[1.0], [1.0]], dtype=np.complex128)
    rho0 = ket_plus_x @ ket_plus_x.conj().T

    n_steps = 9
    t_final = 2.0 * math.pi / omega
    time_grid = np.linspace(0.0, t_final, n_steps)

    print("Heisenberg Picture MVP: spin-1/2 precession")
    print(f"omega = {omega:.6f}, time range = [0, {t_final:.6f}], steps = {n_steps}")

    unitary_err = max_unitarity_error(H, time_grid)
    print(f"max unitarity error ||U^dagger U - I||_F = {unitary_err:.3e}")

    header = (
        " t"
        " | <sx>_H  <sx>_S  <sx>_ana"
        " | <sy>_H  <sy>_S  <sy>_ana"
        " | <sz>_H  <sz>_S  <sz>_ana"
    )
    print(header)
    print("-" * len(header))

    max_picture_gap = 0.0
    max_analytic_gap = 0.0

    for t in time_grid:
        t = float(t)

        sx_h = heisenberg_operator(sx, H, t)
        sy_h = heisenberg_operator(sy, H, t)
        sz_h = heisenberg_operator(sz, H, t)

        rho_t = schrodinger_density(rho0, H, t)

        ex_h = expectation(rho0, sx_h)
        ey_h = expectation(rho0, sy_h)
        ez_h = expectation(rho0, sz_h)

        ex_s = expectation(rho_t, sx)
        ey_s = expectation(rho_t, sy)
        ez_s = expectation(rho_t, sz)

        ex_a, ey_a, ez_a = analytic_expectations(omega, t)

        picture_gap = max(abs(ex_h - ex_s), abs(ey_h - ey_s), abs(ez_h - ez_s))
        analytic_gap = max(abs(ex_h - ex_a), abs(ey_h - ey_a), abs(ez_h - ez_a))

        max_picture_gap = max(max_picture_gap, picture_gap)
        max_analytic_gap = max(max_analytic_gap, analytic_gap)

        print(
            f"{t: .3f}"
            f" | {ex_h: .6f} {ex_s: .6f} {ex_a: .6f}"
            f" | {ey_h: .6f} {ey_s: .6f} {ey_a: .6f}"
            f" | {ez_h: .6f} {ez_s: .6f} {ez_a: .6f}"
        )

    fd_dt = 1e-6
    t_probe = 0.37
    heis_residual = heisenberg_equation_residual(sx, H, t_probe, fd_dt)

    print("\nValidation summary")
    print(f"max picture equivalence gap = {max_picture_gap:.3e}")
    print(f"max analytic gap           = {max_analytic_gap:.3e}")
    print(f"Heisenberg equation residual (sigma_x) = {heis_residual:.3e}")

    assert unitary_err < 1e-12, "Unitary check failed."
    assert max_picture_gap < 1e-10, "Heisenberg vs Schrodinger mismatch too large."
    assert max_analytic_gap < 1e-10, "Deviation from analytic precession too large."
    assert heis_residual < 1e-5, "Heisenberg equation residual too large."


if __name__ == "__main__":
    main()
