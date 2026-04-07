"""Effective Mass Approximation: minimal runnable MVP.

This script builds synthetic E(k) data around a band extremum, fits a full
quadratic model with least squares, and recovers the effective mass tensor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import constants, linalg
from sklearn.metrics import r2_score


def rotation_matrix_xyz(ax: float, ay: float, az: float) -> np.ndarray:
    """Return a 3D rotation matrix from XYZ Euler angles (radians)."""
    sx, cx = np.sin(ax), np.cos(ax)
    sy, cy = np.sin(ay), np.cos(ay)
    sz, cz = np.sin(az), np.cos(az)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def design_matrix(k: np.ndarray) -> np.ndarray:
    """Build linear system features for full 3D quadratic model."""
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    return np.column_stack(
        [
            np.ones(k.shape[0]),
            kx,
            ky,
            kz,
            kx * kx,
            ky * ky,
            kz * kz,
            kx * ky,
            kx * kz,
            ky * kz,
        ]
    )


def unpack_quadratic_coeffs(theta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Map fitted theta to scalar c, vector b, and symmetric matrix A."""
    c = float(theta[0])
    b = theta[1:4]
    a_xx, a_yy, a_zz = theta[4], theta[5], theta[6]
    a_xy, a_xz, a_yz = theta[7], theta[8], theta[9]

    # E(k) = c + b^T k + k^T A k, where cross terms in X use k_i k_j.
    A = np.array(
        [
            [a_xx, 0.5 * a_xy, 0.5 * a_xz],
            [0.5 * a_xy, a_yy, 0.5 * a_yz],
            [0.5 * a_xz, 0.5 * a_yz, a_zz],
        ]
    )
    return c, b, A


def main() -> None:
    rng = np.random.default_rng(20260407)

    hbar = constants.hbar
    q_e = constants.e
    m0 = constants.m_e

    # Ground-truth anisotropic effective mass tensor (kg), built from
    # principal masses and a fixed rotation.
    principal_masses_over_m0 = np.array([0.20, 0.32, 0.95])
    principal_masses = principal_masses_over_m0 * m0
    R = rotation_matrix_xyz(
        np.deg2rad(18.0),
        np.deg2rad(-12.0),
        np.deg2rad(9.0),
    )
    M_true = R @ np.diag(principal_masses) @ R.T
    invM_true = np.linalg.inv(M_true)

    # Synthetic band minimum location and reference energy.
    k0_true = np.array([0.7e9, -0.5e9, 0.35e9])  # 1/m
    E0_true_eV = 1.120

    n_samples = 900
    q = rng.uniform(-1.2e9, 1.2e9, size=(n_samples, 3))
    k = k0_true + q

    delta_E_J = 0.5 * hbar**2 * np.einsum("ni,ij,nj->n", q, invM_true, q)
    noise_eV = rng.normal(0.0, 5e-5, size=n_samples)  # 0.05 meV std
    E_eV = E0_true_eV + delta_E_J / q_e + noise_eV

    # Numerical conditioning: fit in scaled coordinates.
    k_scale = 1.0e9
    x = k / k_scale
    X = design_matrix(x)
    theta, *_ = linalg.lstsq(X, E_eV)
    E_pred = X @ theta

    c_eV, b_x, A_x = unpack_quadratic_coeffs(theta)
    H_x = 2.0 * A_x
    x0_est = -0.5 * np.linalg.solve(A_x, b_x)
    k0_est = x0_est * k_scale
    E0_est_eV = c_eV + b_x @ x0_est + x0_est @ A_x @ x0_est

    # Chain rule from x-space Hessian to k-space Hessian:
    # d^2E/dk^2 = (d^2E/dx^2) / k_scale^2.
    H_k_J = (H_x * q_e) / (k_scale**2)
    invM_est = H_k_J / (hbar**2)
    M_est = np.linalg.inv(invM_est)

    evals_true, _ = np.linalg.eigh(M_true)
    evals_est, _ = np.linalg.eigh(M_est)
    evals_true = np.sort(evals_true)
    evals_est = np.sort(evals_est)
    rel_mass_err = np.abs(evals_est - evals_true) / evals_true

    # PyTorch check: Hessian from autograd should match analytic H.
    A_t = torch.tensor(A_x, dtype=torch.float64)
    b_t = torch.tensor(b_x, dtype=torch.float64)
    c_t = torch.tensor(c_eV, dtype=torch.float64)
    x0_t = torch.tensor(x0_est, dtype=torch.float64, requires_grad=True)

    def quad_energy(x: torch.Tensor) -> torch.Tensor:
        return c_t + torch.dot(b_t, x) + x @ A_t @ x

    H_torch = torch.autograd.functional.hessian(quad_energy, x0_t).detach().numpy()
    hessian_check = np.max(np.abs(H_torch - H_x))

    rmse_meV = np.sqrt(np.mean((E_pred - E_eV) ** 2)) * 1e3
    r2 = r2_score(E_eV, E_pred)
    k0_err = np.linalg.norm(k0_est - k0_true) / np.linalg.norm(k0_true)
    E0_err_meV = np.abs(E0_est_eV - E0_true_eV) * 1e3

    summary = pd.DataFrame(
        {
            "principal_axis": ["1", "2", "3"],
            "m_true/m0": evals_true / m0,
            "m_est/m0": evals_est / m0,
            "rel_err_%": rel_mass_err * 100.0,
        }
    )

    print("=== Effective Mass Approximation MVP ===")
    print(f"samples: {n_samples}")
    print(f"fit RMSE: {rmse_meV:.6f} meV")
    print(f"fit R^2 : {r2:.8f}")
    print(f"k0 relative error: {k0_err:.6e}")
    print(f"E0 absolute error: {E0_err_meV:.6f} meV")
    print(f"max|H_torch - H_analytic|: {hessian_check:.3e}\n")

    print("k0_true (1/m):", np.array2string(k0_true, precision=3))
    print("k0_est  (1/m):", np.array2string(k0_est, precision=3))
    print("\nPrincipal effective masses:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
