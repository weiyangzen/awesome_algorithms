"""Hamiltonian mechanics MVP: 1D simple pendulum in canonical variables.

We model the system with generalized coordinate q (angle) and conjugate momentum p:
    H(q, p) = p^2 / (2 m l^2) + m g l (1 - cos q)
The canonical equations are:
    dq/dt = +dH/dp
    dp/dt = -dH/dq
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class HamiltonianParams:
    """Physical and numerical parameters for the pendulum Hamiltonian system."""

    mass: float = 1.0
    length: float = 1.0
    gravity: float = 9.81
    q0: float = 0.9
    p0: float = 0.0
    dt: float = 0.01
    t_end: float = 40.0


def check_params(params: HamiltonianParams) -> None:
    """Validate scalar parameters and raise ValueError on invalid setup."""

    if params.mass <= 0 or params.length <= 0 or params.gravity <= 0:
        raise ValueError("mass/length/gravity must be positive")
    if params.dt <= 0 or params.t_end <= 0:
        raise ValueError("dt and t_end must be positive")
    for name in ("q0", "p0"):
        value = getattr(params, name)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")


def hamiltonian(q: np.ndarray | float, p: np.ndarray | float, params: HamiltonianParams) -> np.ndarray | float:
    """Compute Hamiltonian H(q,p) for a simple pendulum."""

    ml2 = params.mass * (params.length**2)
    kinetic = (p * p) / (2.0 * ml2)
    potential = params.mass * params.gravity * params.length * (1.0 - np.cos(q))
    return kinetic + potential


def canonical_rhs(_t: float, y: np.ndarray, params: HamiltonianParams) -> np.ndarray:
    """Return [dq/dt, dp/dt] from Hamilton canonical equations."""

    q, p = y
    ml2 = params.mass * (params.length**2)
    dqdt = p / ml2
    dpdt = -params.mass * params.gravity * params.length * np.sin(q)
    return np.array([dqdt, dpdt], dtype=float)


def symplectic_euler(params: HamiltonianParams) -> dict[str, np.ndarray]:
    """Integrate with kick-drift symplectic Euler scheme."""

    n_steps = int(np.round(params.t_end / params.dt))
    t = np.linspace(0.0, n_steps * params.dt, n_steps + 1)
    q = np.empty(n_steps + 1, dtype=float)
    p = np.empty(n_steps + 1, dtype=float)

    q[0] = params.q0
    p[0] = params.p0

    mgl = params.mass * params.gravity * params.length
    ml2 = params.mass * (params.length**2)

    for i in range(n_steps):
        p_next = p[i] - params.dt * mgl * np.sin(q[i])
        q_next = q[i] + params.dt * (p_next / ml2)
        p[i + 1] = p_next
        q[i + 1] = q_next

    e = hamiltonian(q, p, params)
    return {"t": t, "q": q, "p": p, "energy": e}


def scipy_reference(params: HamiltonianParams, t_eval: np.ndarray) -> dict[str, np.ndarray]:
    """High-accuracy SciPy trajectory for comparison."""

    y0 = np.array([params.q0, params.p0], dtype=float)
    sol = solve_ivp(
        fun=lambda t, y: canonical_rhs(t, y, params),
        t_span=(0.0, float(t_eval[-1])),
        y0=y0,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    if not sol.success:
        raise RuntimeError(f"scipy solver failed: {sol.message}")

    q = sol.y[0]
    p = sol.y[1]
    e = hamiltonian(q, p, params)
    return {"t": sol.t, "q": q, "p": p, "energy": e}


def energy_relative_drift(energy: np.ndarray) -> float:
    """Maximum relative drift with baseline stabilization."""

    baseline = max(1.0, abs(float(energy[0])))
    return float(np.max(np.abs(energy - energy[0])) / baseline)


def estimate_period_from_crossings(t: np.ndarray, q: np.ndarray) -> float:
    """Estimate oscillation period using upward zero crossings with interpolation."""

    crossings: list[float] = []
    for i in range(len(q) - 1):
        q0 = q[i]
        q1 = q[i + 1]
        if q0 < 0.0 <= q1 and q1 != q0:
            alpha = -q0 / (q1 - q0)
            crossings.append(float(t[i] + alpha * (t[i + 1] - t[i])))

    if len(crossings) < 2:
        return float("nan")

    periods = np.diff(np.array(crossings, dtype=float))
    return float(np.mean(periods))


def divergence_numeric(q: float, p: float, params: HamiltonianParams, eps: float = 1e-6) -> float:
    """Finite-difference estimate of divergence of Hamiltonian vector field."""

    y_qp = np.array([q + eps, p], dtype=float)
    y_qm = np.array([q - eps, p], dtype=float)
    y_pp = np.array([q, p + eps], dtype=float)
    y_pm = np.array([q, p - eps], dtype=float)

    dqdt_qp = canonical_rhs(0.0, y_qp, params)[0]
    dqdt_qm = canonical_rhs(0.0, y_qm, params)[0]
    dpdt_pp = canonical_rhs(0.0, y_pp, params)[1]
    dpdt_pm = canonical_rhs(0.0, y_pm, params)[1]

    d_dq_dqdt = (dqdt_qp - dqdt_qm) / (2.0 * eps)
    d_dp_dpdt = (dpdt_pp - dpdt_pm) / (2.0 * eps)
    return float(d_dq_dqdt + d_dp_dpdt)


def build_report(sym: dict[str, np.ndarray], ref: dict[str, np.ndarray], params: HamiltonianParams) -> pd.DataFrame:
    """Collect concise diagnostics table."""

    sym_drift = energy_relative_drift(sym["energy"])
    ref_drift = energy_relative_drift(ref["energy"])

    q_diff = np.max(np.abs(sym["q"] - ref["q"]))
    p_diff = np.max(np.abs(sym["p"] - ref["p"]))

    period_sym = estimate_period_from_crossings(sym["t"], sym["q"])
    period_ref = estimate_period_from_crossings(ref["t"], ref["q"])
    period_rel_err = abs(period_sym - period_ref) / max(1.0, abs(period_ref))

    # Liouville check at several points: divergence should be ~0 in canonical coordinates.
    sample_points = [(-1.2, -2.0), (-0.4, 0.0), (0.7, 1.8), (1.1, -1.3)]
    div_vals = np.array([divergence_numeric(q, p, params) for q, p in sample_points], dtype=float)

    return pd.DataFrame(
        [
            {"metric": "symplectic_energy_rel_drift", "value": f"{sym_drift:.2e}"},
            {"metric": "scipy_energy_rel_drift", "value": f"{ref_drift:.2e}"},
            {"metric": "max_abs_q_difference", "value": f"{q_diff:.3e}"},
            {"metric": "max_abs_p_difference", "value": f"{p_diff:.3e}"},
            {"metric": "period_symplectic_s", "value": f"{period_sym:.6f}"},
            {"metric": "period_scipy_s", "value": f"{period_ref:.6f}"},
            {"metric": "period_relative_error", "value": f"{period_rel_err:.3e}"},
            {"metric": "max_abs_divergence", "value": f"{np.max(np.abs(div_vals)):.3e}"},
            {"metric": "final_q_rad", "value": f"{sym['q'][-1]:.6f}"},
            {"metric": "final_p", "value": f"{sym['p'][-1]:.6f}"},
        ]
    )


def main() -> None:
    params = HamiltonianParams()
    check_params(params)

    sym = symplectic_euler(params)
    ref = scipy_reference(params, sym["t"])
    report = build_report(sym, ref, params)

    print("=== Hamiltonian Mechanics MVP (Simple Pendulum, Canonical Form) ===")
    print(
        "params:",
        {
            "mass": params.mass,
            "length": params.length,
            "gravity": params.gravity,
            "q0_rad": params.q0,
            "p0": params.p0,
            "dt": params.dt,
            "t_end": params.t_end,
        },
    )
    print(report.to_string(index=False))

    # Minimal quality gates.
    sym_drift = energy_relative_drift(sym["energy"])
    ref_drift = energy_relative_drift(ref["energy"])
    if sym_drift > 5e-2:
        raise AssertionError("Symplectic Euler drift is unexpectedly large.")
    if ref_drift > 1e-6:
        raise AssertionError("High-accuracy SciPy reference drift is unexpectedly large.")


if __name__ == "__main__":
    main()
