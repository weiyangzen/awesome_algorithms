"""Minimal runnable MVP for CKM matrix parameterization and fitting."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def ckm_standard(theta12: float, theta13: float, theta23: float, delta: float) -> np.ndarray:
    """Build CKM matrix using the PDG standard parameterization."""
    s12, c12 = np.sin(theta12), np.cos(theta12)
    s13, c13 = np.sin(theta13), np.cos(theta13)
    s23, c23 = np.sin(theta23), np.cos(theta23)

    e_minus = np.exp(-1j * delta)
    e_plus = np.exp(1j * delta)

    return np.array(
        [
            [c12 * c13, s12 * c13, s13 * e_minus],
            [
                -s12 * c23 - c12 * s23 * s13 * e_plus,
                c12 * c23 - s12 * s23 * s13 * e_plus,
                s23 * c13,
            ],
            [
                s12 * s23 - c12 * c23 * s13 * e_plus,
                -c12 * s23 - s12 * c23 * s13 * e_plus,
                c23 * c13,
            ],
        ],
        dtype=np.complex128,
    )


def unitarity_residual(v_ckm: np.ndarray) -> float:
    """Return Frobenius norm of VV^dagger - I."""
    identity = np.eye(3, dtype=np.complex128)
    residual = v_ckm @ v_ckm.conj().T - identity
    return float(np.linalg.norm(residual, ord="fro"))


def jarlskog_from_matrix(v_ckm: np.ndarray) -> float:
    """Compute Jarlskog invariant from matrix elements."""
    return float(np.imag(v_ckm[0, 0] * v_ckm[1, 1] * np.conj(v_ckm[0, 1]) * np.conj(v_ckm[1, 0])))


def jarlskog_from_angles(theta12: float, theta13: float, theta23: float, delta: float) -> float:
    """Compute Jarlskog invariant from PDG angles and phase."""
    s12, c12 = np.sin(theta12), np.cos(theta12)
    s13, c13 = np.sin(theta13), np.cos(theta13)
    s23, c23 = np.sin(theta23), np.cos(theta23)
    return float(c12 * c23 * (c13**2) * s12 * s23 * s13 * np.sin(delta))


def angles_to_wolfenstein(theta12: float, theta13: float, theta23: float, delta: float) -> tuple[float, float, float, float]:
    """Convert standard angles/phases to leading-order Wolfenstein parameters."""
    lam = float(np.sin(theta12))
    a_param = float(np.sin(theta23) / (lam**2))
    rho = float(np.sin(theta13) * np.cos(delta) / (a_param * lam**3))
    eta = float(np.sin(theta13) * np.sin(delta) / (a_param * lam**3))
    return lam, a_param, rho, eta


def ckm_wolfenstein_o3(lam: float, a_param: float, rho: float, eta: float) -> np.ndarray:
    """Build CKM matrix with Wolfenstein expansion up to O(lambda^3)."""
    return np.array(
        [
            [1.0 - 0.5 * lam**2, lam, a_param * lam**3 * (rho - 1j * eta)],
            [-lam, 1.0 - 0.5 * lam**2, a_param * lam**2],
            [a_param * lam**3 * (1.0 - rho - 1j * eta), -a_param * lam**2, 1.0],
        ],
        dtype=np.complex128,
    )


def observables_from_params(params: np.ndarray) -> np.ndarray:
    """Return observables: 9 |V_ij| values + Jarlskog invariant."""
    v_ckm = ckm_standard(*params)
    abs_entries = np.abs(v_ckm).reshape(-1)
    jarlskog = jarlskog_from_matrix(v_ckm)
    return np.concatenate([abs_entries, np.array([jarlskog])])


def weighted_residuals(params: np.ndarray, measured: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Residuals normalized by per-observable uncertainty."""
    pred = observables_from_params(params)
    return (pred - measured) / sigma


def format_deg(angle_rad: float) -> str:
    """Format radians to degrees string."""
    return f"{np.degrees(angle_rad):.6f} deg"


def main() -> None:
    # Ground-truth CKM parameters (PDG-like central values, approximate)
    theta12_true = np.deg2rad(13.04)
    theta13_true = np.deg2rad(0.201)
    theta23_true = np.deg2rad(2.38)
    delta_true = np.deg2rad(68.8)
    true_params = np.array([theta12_true, theta13_true, theta23_true, delta_true], dtype=float)

    v_true = ckm_standard(*true_params)
    unitary_err = unitarity_residual(v_true)
    j_matrix = jarlskog_from_matrix(v_true)
    j_angles = jarlskog_from_angles(*true_params)

    lam, a_param, rho, eta = angles_to_wolfenstein(*true_params)
    v_wolf_o3 = ckm_wolfenstein_o3(lam, a_param, rho, eta)
    wolf_error = float(np.linalg.norm(v_true - v_wolf_o3, ord="fro"))

    # Synthetic measurement (absolute CKM entries + J) and weighted fit
    rng = np.random.default_rng(69)
    sigma = np.array([2e-4] * 9 + [8e-7], dtype=float)
    measured = observables_from_params(true_params) + rng.normal(0.0, sigma)

    x0 = np.array(
        [
            np.deg2rad(12.5),
            np.deg2rad(0.25),
            np.deg2rad(2.0),
            np.deg2rad(120.0),
        ],
        dtype=float,
    )
    lower = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    upper = np.array([np.pi / 2, 0.3, np.pi / 2, 2 * np.pi], dtype=float)

    fit = least_squares(
        fun=weighted_residuals,
        x0=x0,
        bounds=(lower, upper),
        args=(measured, sigma),
        method="trf",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=2000,
    )
    if not fit.success:
        raise RuntimeError(f"Least-squares fit failed: {fit.message}")

    fitted_params = fit.x
    predicted = observables_from_params(fitted_params)
    chi2 = float(np.sum(((predicted - measured) / sigma) ** 2))
    dof = measured.size - fitted_params.size

    print("=== CKM Matrix MVP ===")
    print("Standard-parameterization CKM matrix V:")
    print(np.array2string(v_true, precision=6, suppress_small=False))
    print()

    print(f"Unitarity residual ||V V^dagger - I||_F = {unitary_err:.3e}")
    print(f"Jarlskog J (from matrix) = {j_matrix:.6e}")
    print(f"Jarlskog J (from angles) = {j_angles:.6e}")
    print(f"|J_matrix - J_angles| = {abs(j_matrix - j_angles):.3e}")
    print()

    print("Wolfenstein (O(lambda^3)) parameters estimated from true angles:")
    print(f"lambda = {lam:.6f}, A = {a_param:.6f}, rho = {rho:.6f}, eta = {eta:.6f}")
    print(f"Frobenius error ||V_exact - V_wolf_O3||_F = {wolf_error:.3e}")
    print()

    print("Fit results from noisy synthetic observables (|V_ij| + J):")
    names = ["theta12", "theta13", "theta23", "delta"]
    for idx, name in enumerate(names):
        truth = true_params[idx]
        est = fitted_params[idx]
        err = est - truth
        print(
            f"  {name:7s}: true={format_deg(truth):>14s} | fit={format_deg(est):>14s} | "
            f"delta={np.degrees(err):+.4e} deg"
        )

    print(f"chi2/dof = {chi2:.3f}/{dof} = {chi2 / max(dof, 1):.3f}")
    print(f"solver status: {fit.status}, nfev={fit.nfev}, cost={fit.cost:.6f}")


if __name__ == "__main__":
    main()
