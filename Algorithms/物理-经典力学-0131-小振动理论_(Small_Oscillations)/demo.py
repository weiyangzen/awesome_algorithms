"""Small oscillations MVP (2-DOF coupled system).

Model near stable equilibrium (linearized):
    M q_ddot + K q = 0

where:
- q in R^n is small generalized displacement around equilibrium,
- M is symmetric positive-definite mass matrix,
- K is symmetric positive-definite stiffness (Hessian) matrix.

This script demonstrates:
1) normal-mode extraction via generalized eigenproblem K phi = w^2 M phi,
2) modal analytic reconstruction for undamped motion,
3) full ODE integration cross-check,
4) diagnostics: orthogonality, energy drift, reconstruction error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import eigh


@dataclass(frozen=True)
class SmallOscillationConfig:
    """Physical and numerical setup for a 2-DOF coupled oscillator."""

    m1: float = 1.0
    m2: float = 1.4
    k1: float = 42.0
    k2: float = 55.0
    kc: float = 18.0
    q0: tuple[float, float] = (0.08, -0.03)
    v0: tuple[float, float] = (0.00, 0.05)
    t_start: float = 0.0
    t_end: float = 20.0
    num_points: int = 2400
    rtol: float = 1e-9
    atol: float = 1e-11


def build_mass_matrix(cfg: SmallOscillationConfig) -> np.ndarray:
    """Return mass matrix M."""

    return np.array([[cfg.m1, 0.0], [0.0, cfg.m2]], dtype=float)


def build_stiffness_matrix(cfg: SmallOscillationConfig) -> np.ndarray:
    """Return stiffness matrix K for two masses with coupling spring."""

    return np.array(
        [
            [cfg.k1 + cfg.kc, -cfg.kc],
            [-cfg.kc, cfg.k2 + cfg.kc],
        ],
        dtype=float,
    )


def validate_inputs(cfg: SmallOscillationConfig, m: np.ndarray, k: np.ndarray) -> None:
    """Validate physical and numerical parameters."""

    if cfg.m1 <= 0.0 or cfg.m2 <= 0.0:
        raise ValueError("Masses must be positive.")
    if cfg.k1 <= 0.0 or cfg.k2 <= 0.0 or cfg.kc < 0.0:
        raise ValueError("Stiffness values must satisfy k1>0, k2>0, kc>=0.")
    if cfg.num_points < 10:
        raise ValueError("num_points must be >= 10.")
    if cfg.t_end <= cfg.t_start:
        raise ValueError("Require t_end > t_start.")

    q0 = np.array(cfg.q0, dtype=float)
    v0 = np.array(cfg.v0, dtype=float)
    if q0.shape != (2,) or v0.shape != (2,):
        raise ValueError("q0 and v0 must be length-2 vectors.")
    if not np.all(np.isfinite(q0)) or not np.all(np.isfinite(v0)):
        raise ValueError("q0 and v0 must be finite.")

    if not np.allclose(m, m.T, atol=1e-12):
        raise ValueError("Mass matrix must be symmetric.")
    if not np.allclose(k, k.T, atol=1e-12):
        raise ValueError("Stiffness matrix must be symmetric.")


def modal_decomposition(m: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve K phi = w^2 M phi and return (omega, Phi).

    scipy.linalg.eigh(K, M) returns eigenvectors that are M-orthonormal.
    """

    eigvals, phi = eigh(k, m)
    if np.any(eigvals <= 0.0):
        raise ValueError("Expected strictly positive eigenvalues for stable small oscillations.")
    omega = np.sqrt(eigvals)
    return omega, phi


def modal_time_solution(
    t: np.ndarray,
    m: np.ndarray,
    phi: np.ndarray,
    omega: np.ndarray,
    q0: np.ndarray,
    v0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic modal solution for undamped linear system.

    Returns:
    - eta(t), eta_dot(t), q(t), v(t)
    with shapes (N, 2), (N, 2), (N, 2), (N, 2).
    """

    eta0 = phi.T @ m @ q0
    eta_dot0 = phi.T @ m @ v0

    wt = np.outer(t, omega)
    cos_wt = np.cos(wt)
    sin_wt = np.sin(wt)

    eta = eta0 * cos_wt + (eta_dot0 / omega) * sin_wt
    eta_dot = -eta0 * omega * sin_wt + eta_dot0 * cos_wt

    q = eta @ phi.T
    v = eta_dot @ phi.T
    return eta, eta_dot, q, v


def full_rhs(_t: float, y: np.ndarray, inv_m: np.ndarray, k: np.ndarray) -> np.ndarray:
    """RHS for first-order state y=[q1,q2,v1,v2]."""

    q = y[:2]
    v = y[2:]
    a = -inv_m @ (k @ q)
    return np.array([v[0], v[1], a[0], a[1]], dtype=float)


def integrate_full_ode(
    cfg: SmallOscillationConfig,
    m: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically integrate full system and return (q, v)."""

    inv_m = np.linalg.inv(m)
    y0 = np.array([cfg.q0[0], cfg.q0[1], cfg.v0[0], cfg.v0[1]], dtype=float)

    sol = solve_ivp(
        fun=lambda tt, yy: full_rhs(tt, yy, inv_m, k),
        t_span=(cfg.t_start, cfg.t_end),
        y0=y0,
        t_eval=t,
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    q = sol.y[:2].T
    v = sol.y[2:].T
    return q, v


def total_energy(q: np.ndarray, v: np.ndarray, m: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Compute total mechanical energy E = 0.5 v^T M v + 0.5 q^T K q."""

    kinetic = 0.5 * np.einsum("bi,ij,bj->b", v, m, v)
    potential = 0.5 * np.einsum("bi,ij,bj->b", q, k, q)
    return kinetic + potential


def simulate(cfg: SmallOscillationConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run modal + ODE simulation and return trajectory + summary."""

    m = build_mass_matrix(cfg)
    k = build_stiffness_matrix(cfg)
    validate_inputs(cfg, m, k)

    omega, phi = modal_decomposition(m, k)
    t = np.linspace(cfg.t_start, cfg.t_end, cfg.num_points)
    q0 = np.array(cfg.q0, dtype=float)
    v0 = np.array(cfg.v0, dtype=float)

    eta, eta_dot, q_modal, v_modal = modal_time_solution(t, m, phi, omega, q0, v0)
    q_ode, v_ode = integrate_full_ode(cfg, m, k, t)

    e_ode = total_energy(q_ode, v_ode, m, k)
    e0 = float(e_ode[0])
    max_rel_energy_drift = float(np.max(np.abs((e_ode - e0) / max(abs(e0), 1e-12))))

    max_q_error = float(np.max(np.linalg.norm(q_modal - q_ode, axis=1)))
    max_v_error = float(np.max(np.linalg.norm(v_modal - v_ode, axis=1)))

    m_ortho_err = float(np.max(np.abs(phi.T @ m @ phi - np.eye(2))))
    k_diag_err = float(np.max(np.abs(phi.T @ k @ phi - np.diag(omega**2))))

    freq_hz = omega / (2.0 * np.pi)
    q_norm = np.linalg.norm(q_ode, axis=1)

    traj = pd.DataFrame(
        {
            "t": t,
            "q1_ode": q_ode[:, 0],
            "q2_ode": q_ode[:, 1],
            "q1_modal": q_modal[:, 0],
            "q2_modal": q_modal[:, 1],
            "v1_ode": v_ode[:, 0],
            "v2_ode": v_ode[:, 1],
            "eta1": eta[:, 0],
            "eta2": eta[:, 1],
            "energy": e_ode,
            "q_norm": q_norm,
        }
    )

    summary = {
        "omega1_rad_s": float(omega[0]),
        "omega2_rad_s": float(omega[1]),
        "freq1_hz": float(freq_hz[0]),
        "freq2_hz": float(freq_hz[1]),
        "mode_shape_phi11": float(phi[0, 0]),
        "mode_shape_phi21": float(phi[1, 0]),
        "mode_shape_phi12": float(phi[0, 1]),
        "mode_shape_phi22": float(phi[1, 1]),
        "m_orthogonality_max_abs_err": m_ortho_err,
        "k_diagonalization_max_abs_err": k_diag_err,
        "max_q_modal_vs_ode_l2_err": max_q_error,
        "max_v_modal_vs_ode_l2_err": max_v_error,
        "max_rel_energy_drift": max_rel_energy_drift,
        "max_q_norm": float(np.max(q_norm)),
    }
    return traj, summary


def main() -> None:
    cfg = SmallOscillationConfig()
    traj, summary = simulate(cfg)

    print("Small Oscillations MVP (2-DOF)")
    print(
        "params: "
        f"m1={cfg.m1}, m2={cfg.m2}, k1={cfg.k1}, k2={cfg.k2}, kc={cfg.kc}, "
        f"q0={cfg.q0}, v0={cfg.v0}"
    )
    print(
        f"time_span=[{cfg.t_start}, {cfg.t_end}], num_points={cfg.num_points}, "
        f"rtol={cfg.rtol}, atol={cfg.atol}"
    )

    summary_df = pd.DataFrame(
        [
            {"metric": "omega1_rad_s", "value": f"{summary['omega1_rad_s']:.8f}"},
            {"metric": "omega2_rad_s", "value": f"{summary['omega2_rad_s']:.8f}"},
            {"metric": "freq1_hz", "value": f"{summary['freq1_hz']:.8f}"},
            {"metric": "freq2_hz", "value": f"{summary['freq2_hz']:.8f}"},
            {
                "metric": "m_orthogonality_max_abs_err",
                "value": f"{summary['m_orthogonality_max_abs_err']:.3e}",
            },
            {
                "metric": "k_diagonalization_max_abs_err",
                "value": f"{summary['k_diagonalization_max_abs_err']:.3e}",
            },
            {
                "metric": "max_q_modal_vs_ode_l2_err",
                "value": f"{summary['max_q_modal_vs_ode_l2_err']:.3e}",
            },
            {
                "metric": "max_v_modal_vs_ode_l2_err",
                "value": f"{summary['max_v_modal_vs_ode_l2_err']:.3e}",
            },
            {"metric": "max_rel_energy_drift", "value": f"{summary['max_rel_energy_drift']:.3e}"},
            {"metric": "max_q_norm", "value": f"{summary['max_q_norm']:.6f}"},
        ]
    )

    print("\nsummary:")
    print(summary_df.to_string(index=False))

    print("\ntrajectory_head:")
    print(traj.head(5).to_string(index=False))
    print("\ntrajectory_tail:")
    print(traj.tail(5).to_string(index=False))

    if summary["m_orthogonality_max_abs_err"] > 1e-10:
        raise AssertionError("Mode orthogonality check failed.")
    if summary["k_diagonalization_max_abs_err"] > 1e-9:
        raise AssertionError("K diagonalization check failed.")
    if summary["max_q_modal_vs_ode_l2_err"] > 2e-7:
        raise AssertionError("Modal displacement reconstruction mismatch is too large.")
    if summary["max_v_modal_vs_ode_l2_err"] > 3e-7:
        raise AssertionError("Modal velocity reconstruction mismatch is too large.")
    if summary["max_rel_energy_drift"] > 1e-6:
        raise AssertionError("Energy drift too large; verify integration tolerance.")


if __name__ == "__main__":
    main()
