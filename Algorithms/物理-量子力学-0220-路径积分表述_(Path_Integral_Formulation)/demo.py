"""Minimal runnable MVP for the Path Integral Formulation.

This demo computes the Euclidean-time propagator of the 1D harmonic oscillator
by discretizing the path integral and evaluating the resulting Gaussian integral.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def build_euclidean_system(
    n_slices: int,
    beta: float,
    mass: float,
    omega: float,
    x_initial: float,
    x_final: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build quadratic form S_E = 0.5 x^T A x - h^T x + const.

    Discretization uses symmetric potential splitting:
    S_E = sum_k [ m/(2*dt) (x_{k+1}-x_k)^2 + dt/2 * (V_k + V_{k+1}) ],
    with V(x) = 0.5 * m * omega^2 * x^2.
    Endpoints x_0 and x_N are fixed; integration variables are x_1..x_{N-1}.
    """
    if n_slices < 2:
        raise ValueError("n_slices must be >= 2")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if mass <= 0.0:
        raise ValueError("mass must be positive")

    dt = beta / float(n_slices)
    dim = n_slices - 1

    kinetic_diag = 2.0 * mass / dt
    potential_diag = mass * (omega**2) * dt
    diag = np.full(dim, kinetic_diag + potential_diag, dtype=float)
    off = np.full(dim - 1, -mass / dt, dtype=float)

    matrix_a = np.diag(diag)
    if dim > 1:
        matrix_a += np.diag(off, k=1) + np.diag(off, k=-1)

    vector_h = np.zeros(dim, dtype=float)
    vector_h[0] += mass * x_initial / dt
    vector_h[-1] += mass * x_final / dt

    const_term = (
        (mass / (2.0 * dt) + 0.25 * mass * (omega**2) * dt)
        * (x_initial**2 + x_final**2)
    )
    return matrix_a, vector_h, const_term, dt


def discrete_euclidean_propagator(
    n_slices: int,
    beta: float,
    mass: float,
    omega: float,
    hbar: float,
    x_initial: float,
    x_final: float,
) -> float:
    """Approximate K_E(x_f, beta; x_i, 0) with discretized path integral."""
    if hbar <= 0.0:
        raise ValueError("hbar must be positive")

    matrix_a, vector_h, const_term, dt = build_euclidean_system(
        n_slices=n_slices,
        beta=beta,
        mass=mass,
        omega=omega,
        x_initial=x_initial,
        x_final=x_final,
    )

    # SPD linear solve for A^{-1} h and log(det(A)) via Cholesky.
    chol, lower = cho_factor(matrix_a, lower=True, check_finite=True)
    a_inv_h = cho_solve((chol, lower), vector_h, check_finite=True)
    log_det_a = 2.0 * float(np.sum(np.log(np.diag(chol))))

    dim = n_slices - 1
    log_prefactor = 0.5 * n_slices * np.log(mass / (2.0 * np.pi * hbar * dt))
    log_gaussian = 0.5 * dim * np.log(2.0 * np.pi * hbar) - 0.5 * log_det_a
    exponent = (-const_term + 0.5 * float(vector_h @ a_inv_h)) / hbar

    log_kernel = log_prefactor + log_gaussian + exponent
    return float(np.exp(log_kernel))


def exact_euclidean_propagator(
    beta: float,
    mass: float,
    omega: float,
    hbar: float,
    x_initial: float,
    x_final: float,
) -> float:
    """Analytic Euclidean propagator for a 1D harmonic oscillator."""
    if beta <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("beta, mass, and hbar must be positive")

    if np.isclose(omega, 0.0):
        prefactor = np.sqrt(mass / (2.0 * np.pi * hbar * beta))
        exponent = -mass * (x_final - x_initial) ** 2 / (2.0 * hbar * beta)
        return float(prefactor * np.exp(exponent))

    sinh_term = np.sinh(omega * beta)
    cosh_term = np.cosh(omega * beta)

    prefactor = np.sqrt(mass * omega / (2.0 * np.pi * hbar * sinh_term))
    exponent = (
        -mass
        * omega
        / (2.0 * hbar * sinh_term)
        * ((x_final**2 + x_initial**2) * cosh_term - 2.0 * x_initial * x_final)
    )
    return float(prefactor * np.exp(exponent))


def run_convergence_experiment() -> None:
    """Run a small deterministic experiment and print convergence diagnostics."""
    hbar = 1.0
    mass = 1.0
    omega = 1.2
    beta = 1.5
    x_initial = -0.4
    x_final = 0.7

    n_slices_grid = [4, 8, 16, 32, 64, 128]

    exact = exact_euclidean_propagator(
        beta=beta,
        mass=mass,
        omega=omega,
        hbar=hbar,
        x_initial=x_initial,
        x_final=x_final,
    )

    print("Path Integral Formulation demo (Euclidean harmonic oscillator)")
    print(
        "params: "
        f"hbar={hbar:.3f}, m={mass:.3f}, omega={omega:.3f}, beta={beta:.3f}, "
        f"x_i={x_initial:.3f}, x_f={x_final:.3f}"
    )
    print(f"exact K_E = {exact:.10e}")
    print()
    print("N_slices   K_discrete         rel_error")

    for n_slices in n_slices_grid:
        approx = discrete_euclidean_propagator(
            n_slices=n_slices,
            beta=beta,
            mass=mass,
            omega=omega,
            hbar=hbar,
            x_initial=x_initial,
            x_final=x_final,
        )
        rel_error = abs(approx - exact) / max(abs(exact), 1e-15)
        print(f"{n_slices:8d}  {approx: .10e}  {rel_error: .3e}")


def main() -> None:
    run_convergence_experiment()


if __name__ == "__main__":
    main()
