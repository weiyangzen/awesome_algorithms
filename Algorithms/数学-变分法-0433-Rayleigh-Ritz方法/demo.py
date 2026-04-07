"""Rayleigh-Ritz method MVP.

We solve the model eigenvalue problem:
    -u'' = lambda * u,  x in (0, 1)
    u(0) = u(1) = 0

Its exact smallest eigenvalue is pi^2.
The Rayleigh-Ritz approximation uses basis:
    phi_k(x) = x(1-x)x^k, k = 0,1,...,N-1
and solves the generalized eigenproblem:
    K c = lambda M c
where
    K_ij = integral(phi_i' * phi_j')
    M_ij = integral(phi_i  * phi_j)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


@dataclass
class RitzResult:
    basis_size: int
    lambda_1: float
    rel_eig_error: float
    l2_func_error: float


def gauss_legendre_on_unit_interval(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return quadrature points/weights on [0, 1]."""
    x_ref, w_ref = np.polynomial.legendre.leggauss(num_points)
    x = 0.5 * (x_ref + 1.0)
    w = 0.5 * w_ref
    return x, w


def basis_and_derivative(x: np.ndarray, basis_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute phi_k(x) and phi_k'(x) for k=0..basis_size-1.

    Returns arrays with shape (basis_size, len(x)).
    """
    phi = np.empty((basis_size, x.size), dtype=float)
    dphi = np.empty((basis_size, x.size), dtype=float)

    one_minus_x = 1.0 - x
    for k in range(basis_size):
        xk = x**k
        xk1 = xk * x
        phi[k] = xk1 * one_minus_x
        # d/dx [x^(k+1) (1-x)] = (k+1)x^k - (k+2)x^(k+1)
        dphi[k] = (k + 1) * xk - (k + 2) * xk1

    return phi, dphi


def assemble_matrices(basis_size: int, quad_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Assemble stiffness matrix K and mass matrix M."""
    x, w = gauss_legendre_on_unit_interval(quad_points)
    phi, dphi = basis_and_derivative(x, basis_size)

    weighted_phi = phi * w[np.newaxis, :]
    weighted_dphi = dphi * w[np.newaxis, :]

    m_mat = weighted_phi @ phi.T
    k_mat = weighted_dphi @ dphi.T
    return k_mat, m_mat


def rayleigh_ritz_smallest_eigenpair(basis_size: int) -> tuple[float, np.ndarray]:
    """Compute smallest eigenvalue/eigenvector of Kc=lambda Mc."""
    k_mat, m_mat = assemble_matrices(basis_size=basis_size)
    eigvals, eigvecs = eigh(k_mat, m_mat)
    return float(eigvals[0]), eigvecs[:, 0]


def reconstruct_function(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate u_N(x)=sum c_k phi_k(x)."""
    basis_size = coeffs.size
    phi, _ = basis_and_derivative(x, basis_size)
    return coeffs @ phi


def l2_error_to_exact_mode(coeffs: np.ndarray, grid_size: int = 4000) -> float:
    """Return L2 error between normalized Ritz mode and exact first mode."""
    x = np.linspace(0.0, 1.0, grid_size)
    u = reconstruct_function(coeffs, x)

    # Remove arbitrary scaling/sign of eigenvectors by L2 normalization + sign alignment.
    u_norm = math.sqrt(np.trapezoid(u * u, x))
    u = u / u_norm

    u_exact = math.sqrt(2.0) * np.sin(math.pi * x)
    if np.trapezoid(u * u_exact, x) < 0.0:
        u = -u

    err = np.trapezoid((u - u_exact) ** 2, x)
    return math.sqrt(err)


def run_experiment(basis_sizes: list[int]) -> list[RitzResult]:
    """Run Rayleigh-Ritz for several basis sizes and collect metrics."""
    exact_lambda_1 = math.pi**2
    results: list[RitzResult] = []

    for n in basis_sizes:
        lam, coeffs = rayleigh_ritz_smallest_eigenpair(n)
        rel_eig_error = abs(lam - exact_lambda_1) / exact_lambda_1
        l2_func_error = l2_error_to_exact_mode(coeffs)
        results.append(
            RitzResult(
                basis_size=n,
                lambda_1=lam,
                rel_eig_error=rel_eig_error,
                l2_func_error=l2_func_error,
            )
        )

    return results


def print_results(results: list[RitzResult]) -> None:
    """Pretty-print experiment results."""
    print("Rayleigh-Ritz MVP: -u''=lambda u on (0,1), u(0)=u(1)=0")
    print(f"Exact smallest eigenvalue: pi^2 = {math.pi**2:.12f}\n")
    header = f"{'N':>4} | {'lambda_1':>15} | {'rel_err(lambda)':>15} | {'L2_err(mode)':>12}"
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row.basis_size:4d} | {row.lambda_1:15.10f} | "
            f"{row.rel_eig_error:15.6e} | {row.l2_func_error:12.6e}"
        )


def main() -> None:
    basis_sizes = [2, 3, 4, 6, 8, 10]
    results = run_experiment(basis_sizes)
    print_results(results)


if __name__ == "__main__":
    main()
