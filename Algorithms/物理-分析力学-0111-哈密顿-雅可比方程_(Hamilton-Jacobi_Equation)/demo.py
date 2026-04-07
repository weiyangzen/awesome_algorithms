"""Hamilton-Jacobi Equation MVP for a 1D harmonic oscillator.

We solve the time-independent Hamilton-Jacobi equation:
    H(q, dW/dq) = E
for
    H(q, p) = p^2/(2m) + 0.5*m*omega^2*q^2.

The reduced action W(q;E) on the positive-momentum branch is computed in two ways:
1) numerical quadrature of p(q) = dW/dq;
2) closed-form expression for cross-check.

Then we verify trajectory reconstruction from HJ theory against a high-accuracy ODE solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, solve_ivp


@dataclass(frozen=True)
class HJParams:
    """Physical and numerical parameters for the HJ MVP."""

    mass: float = 1.0
    omega: float = 2.0
    energy: float = 2.0
    q_fraction_max: float = 0.98
    n_q: int = 1500
    t_end: float = 6.0
    n_t: int = 1200


def check_params(params: HJParams) -> None:
    """Validate parameters and raise ValueError on invalid setup."""

    if params.mass <= 0 or params.omega <= 0 or params.energy <= 0:
        raise ValueError("mass, omega, energy must be positive")
    if not (0.0 < params.q_fraction_max < 1.0):
        raise ValueError("q_fraction_max must be in (0, 1)")
    if params.n_q < 32 or params.n_t < 32:
        raise ValueError("n_q and n_t must be at least 32")
    if params.t_end <= 0:
        raise ValueError("t_end must be positive")


def turning_point_amplitude(params: HJParams) -> float:
    """Return classical turning-point amplitude A = sqrt(2E/(m omega^2))."""

    return float(np.sqrt(2.0 * params.energy / (params.mass * params.omega * params.omega)))


def potential(q: np.ndarray | float, params: HJParams) -> np.ndarray | float:
    """Harmonic potential V(q) = 0.5*m*omega^2*q^2."""

    return 0.5 * params.mass * params.omega * params.omega * (q * q)


def momentum_positive_branch(q: np.ndarray, params: HJParams, tol: float = 1e-12) -> np.ndarray:
    """Compute positive branch p(q)=sqrt(2m(E-V(q))) in classically allowed region."""

    inside = 2.0 * params.mass * (params.energy - potential(q, params))
    if np.min(inside) < -tol:
        raise ValueError("q grid exceeds classically allowed region for selected energy")
    inside = np.clip(inside, 0.0, None)
    return np.sqrt(inside)


def reduced_action_numeric(q: np.ndarray, params: HJParams) -> tuple[np.ndarray, np.ndarray]:
    """Numerically compute W(q)=int_0^q p(q')dq' using cumulative trapezoid."""

    p = momentum_positive_branch(q, params)
    w = cumulative_trapezoid(p, q, initial=0.0)
    return w, p


def reduced_action_closed_form(q: np.ndarray, params: HJParams) -> np.ndarray:
    """Closed-form W(q) on positive branch with W(0)=0."""

    a = turning_point_amplitude(params)
    q_over_a = np.clip(q / a, -1.0, 1.0)
    root = np.sqrt(np.maximum(a * a - q * q, 0.0))
    pref = 0.5 * params.mass * params.omega
    return pref * (q * root + a * a * np.arcsin(q_over_a))


def hj_residual_from_w(q: np.ndarray, w: np.ndarray, params: HJParams) -> np.ndarray:
    """Compute residual R=(dW/dq)^2/(2m)+V-E for the time-independent HJ equation."""

    dw_dq = np.gradient(w, q, edge_order=2)
    return (dw_dq * dw_dq) / (2.0 * params.mass) + potential(q, params) - params.energy


def canonical_rhs(_t: float, y: np.ndarray, params: HJParams) -> np.ndarray:
    """Canonical equations for harmonic oscillator in (q,p)."""

    q, p = y
    dqdt = p / params.mass
    dpdt = -(params.mass * params.omega * params.omega) * q
    return np.array([dqdt, dpdt], dtype=float)


def hj_trajectory_from_constant_beta(t: np.ndarray, params: HJParams) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct trajectory from HJ complete integral with q(0)=0, p(0)>0."""

    a = turning_point_amplitude(params)
    q = a * np.sin(params.omega * t)
    p = params.mass * a * params.omega * np.cos(params.omega * t)
    return q, p


def scipy_reference_trajectory(t: np.ndarray, params: HJParams) -> tuple[np.ndarray, np.ndarray]:
    """High-accuracy ODE trajectory used as reference."""

    y0 = np.array([0.0, np.sqrt(2.0 * params.mass * params.energy)], dtype=float)
    sol = solve_ivp(
        fun=lambda tau, y: canonical_rhs(tau, y, params),
        t_span=(0.0, float(t[-1])),
        y0=y0,
        t_eval=t,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.y[0], sol.y[1]


def build_report(params: HJParams) -> pd.DataFrame:
    """Run all checks and return a compact diagnostics table."""

    a = turning_point_amplitude(params)
    q_max = params.q_fraction_max * a
    q_grid = np.linspace(0.0, q_max, params.n_q, dtype=float)

    w_num, p_grid = reduced_action_numeric(q_grid, params)
    w_closed = reduced_action_closed_form(q_grid, params)

    dw_dq_num = np.gradient(w_num, q_grid, edge_order=2)
    residual = hj_residual_from_w(q_grid, w_num, params)
    sl = slice(4, -4)

    max_abs_w_diff = float(np.max(np.abs(w_num[sl] - w_closed[sl])))
    max_abs_grad_diff = float(np.max(np.abs(dw_dq_num[sl] - p_grid[sl])))
    max_abs_residual = float(np.max(np.abs(residual[sl])))

    t = np.linspace(0.0, params.t_end, params.n_t, dtype=float)
    q_hj, p_hj = hj_trajectory_from_constant_beta(t, params)
    q_ref, p_ref = scipy_reference_trajectory(t, params)

    q_err = float(np.max(np.abs(q_hj - q_ref)))
    p_err = float(np.max(np.abs(p_hj - p_ref)))

    energy_hj = (p_hj * p_hj) / (2.0 * params.mass) + potential(q_hj, params)
    energy_ref = (p_ref * p_ref) / (2.0 * params.mass) + potential(q_ref, params)
    e0 = max(1.0, abs(float(energy_hj[0])))
    energy_drift_hj = float(np.max(np.abs(energy_hj - energy_hj[0])) / e0)
    energy_drift_ref = float(np.max(np.abs(energy_ref - energy_ref[0])) / e0)

    period = 2.0 * np.pi / params.omega

    return pd.DataFrame(
        [
            {"metric": "turning_point_amplitude_A", "value": f"{a:.6f}"},
            {"metric": "max_abs_W_numeric_minus_closed", "value": f"{max_abs_w_diff:.3e}"},
            {"metric": "max_abs_dW_dq_minus_p", "value": f"{max_abs_grad_diff:.3e}"},
            {"metric": "max_abs_HJ_residual", "value": f"{max_abs_residual:.3e}"},
            {"metric": "max_abs_q_error_vs_scipy", "value": f"{q_err:.3e}"},
            {"metric": "max_abs_p_error_vs_scipy", "value": f"{p_err:.3e}"},
            {"metric": "energy_rel_drift_HJ_traj", "value": f"{energy_drift_hj:.3e}"},
            {"metric": "energy_rel_drift_scipy_ref", "value": f"{energy_drift_ref:.3e}"},
            {"metric": "analytic_period", "value": f"{period:.6f}"},
            {"metric": "simulated_t_end", "value": f"{params.t_end:.6f}"},
        ]
    )


def main() -> None:
    params = HJParams()
    check_params(params)

    report = build_report(params)

    print("=== Hamilton-Jacobi Equation MVP (1D Harmonic Oscillator) ===")
    print(
        "params:",
        {
            "mass": params.mass,
            "omega": params.omega,
            "energy": params.energy,
            "q_fraction_max": params.q_fraction_max,
            "n_q": params.n_q,
            "n_t": params.n_t,
            "t_end": params.t_end,
        },
    )
    print(report.to_string(index=False))

    # Minimal quality gates for this MVP.
    metrics = {row.metric: float(row.value) for row in report.itertuples(index=False)}
    if metrics["max_abs_HJ_residual"] > 1e-2:
        raise AssertionError("HJ PDE residual is unexpectedly large.")
    if metrics["max_abs_q_error_vs_scipy"] > 1e-8:
        raise AssertionError("HJ trajectory does not match high-accuracy ODE reference.")
    if metrics["max_abs_p_error_vs_scipy"] > 1e-8:
        raise AssertionError("HJ momentum trajectory does not match reference.")


if __name__ == "__main__":
    main()
