"""Minimal runnable MVP for normal mode analysis in classical mechanics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class ModeAnalysisResult:
    masses: np.ndarray
    springs: np.ndarray
    M: np.ndarray
    K: np.ndarray
    omega: np.ndarray
    eigenvalues: np.ndarray
    phi: np.ndarray
    mass_metric: np.ndarray
    stiffness_metric: np.ndarray
    eigen_residuals: np.ndarray


def build_chain_matrices(masses: np.ndarray, springs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build M and K for a 1D spring-mass chain with wall springs.

    masses shape: (n,)
    springs shape: (n+1,), where
      springs[0] couples left wall-mass0,
      springs[n] couples mass(n-1)-right wall,
      springs[i] (1<=i<=n-1) couples mass(i-1)-mass(i).
    """
    n = masses.size
    if springs.size != n + 1:
        raise ValueError("springs must have length n+1")
    if np.any(masses <= 0):
        raise ValueError("all masses must be > 0")
    if np.any(springs < 0):
        raise ValueError("all spring constants must be >= 0")

    M = np.diag(masses)
    K = np.zeros((n, n), dtype=float)

    for j, k in enumerate(springs):
        if j == 0:
            K[0, 0] += k
        elif j == n:
            K[n - 1, n - 1] += k
        else:
            left = j - 1
            right = j
            K[left, left] += k
            K[right, right] += k
            K[left, right] -= k
            K[right, left] -= k

    return M, K


def normal_mode_analysis(masses: np.ndarray, springs: np.ndarray) -> ModeAnalysisResult:
    """Solve K phi = lambda M phi using mass scaling and symmetric eigendecomposition."""
    M, K = build_chain_matrices(masses, springs)

    inv_sqrt_m = np.diag(1.0 / np.sqrt(masses))
    A = inv_sqrt_m @ K @ inv_sqrt_m

    eigenvalues, U = np.linalg.eigh(A)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    omega = np.sqrt(eigenvalues)

    phi = inv_sqrt_m @ U

    mass_metric = phi.T @ M @ phi
    stiffness_metric = phi.T @ K @ phi

    # Residual per mode for K*phi_i = lambda_i*M*phi_i.
    residuals = []
    for i in range(phi.shape[1]):
        lhs = K @ phi[:, i]
        rhs = eigenvalues[i] * (M @ phi[:, i])
        residuals.append(float(np.linalg.norm(lhs - rhs)))

    return ModeAnalysisResult(
        masses=masses,
        springs=springs,
        M=M,
        K=K,
        omega=omega,
        eigenvalues=eigenvalues,
        phi=phi,
        mass_metric=mass_metric,
        stiffness_metric=stiffness_metric,
        eigen_residuals=np.array(residuals),
    )


def modal_initial_conditions(
    phi: np.ndarray,
    M: np.ndarray,
    x0: np.ndarray,
    v0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project initial displacement/velocity into modal coordinates."""
    q0 = phi.T @ M @ x0
    qdot0 = phi.T @ M @ v0
    return q0, qdot0


def modal_time_response(
    omega: np.ndarray,
    q0: np.ndarray,
    qdot0: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return q(t), qdot(t) for each mode and time sample."""
    n = omega.size
    nt = t.size
    q = np.zeros((n, nt), dtype=float)
    qdot = np.zeros((n, nt), dtype=float)

    for i in range(n):
        w = float(omega[i])
        if w > EPS:
            wt = w * t
            cos_wt = np.cos(wt)
            sin_wt = np.sin(wt)
            b = qdot0[i] / w
            q[i, :] = q0[i] * cos_wt + b * sin_wt
            qdot[i, :] = -q0[i] * w * sin_wt + b * w * cos_wt
        else:
            # Zero-frequency mode: q(t) = q0 + qdot0 * t
            q[i, :] = q0[i] + qdot0[i] * t
            qdot[i, :] = qdot0[i]

    return q, qdot


def reconstruct_physical_response(phi: np.ndarray, q: np.ndarray, qdot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map modal response back to physical coordinates."""
    x = phi @ q
    v = phi @ qdot
    return x, v


def total_energy(M: np.ndarray, K: np.ndarray, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute E(t)=0.5*(v^T M v + x^T K x) for each time sample."""
    kinetic = 0.5 * np.sum(v * (M @ v), axis=0)
    potential = 0.5 * np.sum(x * (K @ x), axis=0)
    return kinetic + potential


def max_offdiag_abs(a: np.ndarray) -> float:
    """Maximum absolute value of off-diagonal entries."""
    b = a.copy()
    np.fill_diagonal(b, 0.0)
    return float(np.max(np.abs(b)))


def build_mode_table(result: ModeAnalysisResult) -> pd.DataFrame:
    """Human-readable mode summary."""
    period = np.where(result.omega > EPS, 2.0 * np.pi / result.omega, np.inf)
    return pd.DataFrame(
        {
            "mode": np.arange(1, result.omega.size + 1, dtype=int),
            "omega(rad/s)": result.omega,
            "period(s)": period,
            "eig_residual": result.eigen_residuals,
        }
    )


def main() -> None:
    # A non-uniform 4-DOF chain: enough structure to show coupled modes.
    masses = np.array([1.0, 1.2, 0.9, 1.4], dtype=float)
    springs = np.array([80.0, 120.0, 100.0, 110.0, 90.0], dtype=float)

    result = normal_mode_analysis(masses, springs)

    x0 = np.array([0.06, -0.02, 0.03, -0.01], dtype=float)
    v0 = np.array([0.00, 0.04, -0.03, 0.02], dtype=float)

    t = np.linspace(0.0, 12.0, 1201)
    q0, qdot0 = modal_initial_conditions(result.phi, result.M, x0, v0)
    q, qdot = modal_time_response(result.omega, q0, qdot0, t)
    x, v = reconstruct_physical_response(result.phi, q, qdot)

    energy = total_energy(result.M, result.K, x, v)
    relative_energy_drift = float(np.max(np.abs(energy - energy[0])) / max(abs(energy[0]), EPS))

    x0_err = float(np.max(np.abs(x[:, 0] - x0)))
    v0_err = float(np.max(np.abs(v[:, 0] - v0)))
    init_recon_err = max(x0_err, v0_err)

    mass_offdiag = max_offdiag_abs(result.mass_metric)
    stiff_offdiag = max_offdiag_abs(result.stiffness_metric)
    max_residual = float(np.max(result.eigen_residuals))

    # Conservative tolerances for float64 MVP.
    checks = {
        "max eigen residual < 1e-10": max_residual < 1e-10,
        "mass offdiag < 1e-12": mass_offdiag < 1e-12,
        "stiffness offdiag < 1e-10": stiff_offdiag < 1e-10,
        "initial reconstruction error < 1e-12": init_recon_err < 1e-12,
        "relative energy drift < 1e-11": relative_energy_drift < 1e-11,
    }

    table = build_mode_table(result)
    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    print("=== Normal Mode Analysis MVP (PHYS-0132) ===")
    print("Masses:", masses)
    print("Springs:", springs)
    print("\nMode summary:")
    print(table.to_string(index=False))

    print("\nOrthogonality / diagonalization checks:")
    print(f"max |offdiag(Phi^T M Phi)| = {mass_offdiag:.3e}")
    print(f"max |offdiag(Phi^T K Phi)| = {stiff_offdiag:.3e}")

    print("\nReconstruction / conservation checks:")
    print(f"max initial reconstruction error = {init_recon_err:.3e}")
    print(f"relative energy drift = {relative_energy_drift:.3e}")

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
