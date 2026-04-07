"""Minimal runnable MVP for Leapfrog Algorithm (PHYS-0324).

This script demonstrates leapfrog (kick-drift-kick) integration on a 1D
harmonic oscillator and validates three core properties:
1) second-order convergence,
2) long-time energy behavior vs explicit Euler,
3) time reversibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OscillatorParams:
    """Physical parameters for 1D harmonic oscillator."""

    mass: float = 1.0
    spring_k: float = 1.0


@dataclass
class Trajectory:
    """Discrete trajectory and diagnostics."""

    t: np.ndarray
    q: np.ndarray
    p: np.ndarray
    energy: np.ndarray
    nfev: int


class HarmonicForce:
    """Force field F(q) = -k*q with evaluation counter."""

    def __init__(self, spring_k: float) -> None:
        self.spring_k = float(spring_k)
        self.nfev = 0

    def force(self, q: float) -> float:
        self.nfev += 1
        return -self.spring_k * q


def validate_inputs(q0: float, p0: float, dt: float, n_steps: int, params: OscillatorParams) -> None:
    """Basic guard rails for deterministic numerical integration."""

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if not math.isfinite(dt) or dt == 0.0:
        raise ValueError("dt must be finite and non-zero")
    if not math.isfinite(q0) or not math.isfinite(p0):
        raise ValueError("q0 and p0 must be finite")
    if not math.isfinite(params.mass) or params.mass <= 0.0:
        raise ValueError("mass must be finite and positive")
    if not math.isfinite(params.spring_k) or params.spring_k <= 0.0:
        raise ValueError("spring_k must be finite and positive")


def exact_solution(
    t: np.ndarray,
    q0: float,
    p0: float,
    params: OscillatorParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic solution of m*q'' + k*q = 0 under (q0, p0)."""

    omega = math.sqrt(params.spring_k / params.mass)
    wt = omega * t

    q = q0 * np.cos(wt) + (p0 / (params.mass * omega)) * np.sin(wt)
    p = p0 * np.cos(wt) - (params.mass * omega * q0) * np.sin(wt)
    return q, p


def leapfrog_solve(
    q0: float,
    p0: float,
    dt: float,
    n_steps: int,
    params: OscillatorParams,
) -> Trajectory:
    """Kick-drift-kick leapfrog for separable Hamiltonian H = p^2/(2m) + kq^2/2."""

    validate_inputs(q0=q0, p0=p0, dt=dt, n_steps=n_steps, params=params)

    field = HarmonicForce(params.spring_k)
    t = dt * np.arange(n_steps + 1, dtype=float)
    q = np.empty(n_steps + 1, dtype=float)
    p = np.empty(n_steps + 1, dtype=float)

    q[0] = q0
    p[0] = p0

    for n in range(n_steps):
        f_n = field.force(float(q[n]))
        p_half = p[n] + 0.5 * dt * f_n

        q[n + 1] = q[n] + dt * (p_half / params.mass)

        f_np1 = field.force(float(q[n + 1]))
        p[n + 1] = p_half + 0.5 * dt * f_np1

        if not math.isfinite(float(q[n + 1])) or not math.isfinite(float(p[n + 1])):
            raise RuntimeError("non-finite state encountered during leapfrog integration")

    energy = 0.5 * (p * p) / params.mass + 0.5 * params.spring_k * (q * q)
    return Trajectory(t=t, q=q, p=p, energy=energy, nfev=field.nfev)


def explicit_euler_solve(
    q0: float,
    p0: float,
    dt: float,
    n_steps: int,
    params: OscillatorParams,
) -> Trajectory:
    """Explicit Euler baseline for comparison."""

    validate_inputs(q0=q0, p0=p0, dt=dt, n_steps=n_steps, params=params)

    field = HarmonicForce(params.spring_k)
    t = dt * np.arange(n_steps + 1, dtype=float)
    q = np.empty(n_steps + 1, dtype=float)
    p = np.empty(n_steps + 1, dtype=float)

    q[0] = q0
    p[0] = p0

    for n in range(n_steps):
        f_n = field.force(float(q[n]))

        q[n + 1] = q[n] + dt * (p[n] / params.mass)
        p[n + 1] = p[n] + dt * f_n

        if not math.isfinite(float(q[n + 1])) or not math.isfinite(float(p[n + 1])):
            raise RuntimeError("non-finite state encountered during Euler integration")

    energy = 0.5 * (p * p) / params.mass + 0.5 * params.spring_k * (q * q)
    return Trajectory(t=t, q=q, p=p, energy=energy, nfev=field.nfev)


def phase_error(q_num: float, p_num: float, q_ref: float, p_ref: float) -> float:
    """Euclidean phase-space error at one time point."""

    dq = q_num - q_ref
    dp = p_num - p_ref
    return float(math.sqrt(dq * dq + dp * dp))


def run_convergence_demo(params: OscillatorParams) -> pd.DataFrame:
    """Empirically show second-order convergence of leapfrog."""

    print("Convergence demo (harmonic oscillator, T=2)")

    t_end = 2.0
    h_values = [0.20, 0.10, 0.05, 0.025]

    rows: list[dict[str, float | int | str]] = []
    errors: list[float] = []

    for i, h in enumerate(h_values):
        n_steps = int(round(t_end / h))
        traj = leapfrog_solve(q0=1.0, p0=0.0, dt=h, n_steps=n_steps, params=params)

        q_ref, p_ref = exact_solution(
            t=np.array([traj.t[-1]], dtype=float),
            q0=1.0,
            p0=0.0,
            params=params,
        )
        err = phase_error(
            q_num=float(traj.q[-1]),
            p_num=float(traj.p[-1]),
            q_ref=float(q_ref[0]),
            p_ref=float(p_ref[0]),
        )
        errors.append(err)

        ratio_text = "-"
        if i > 0 and err > 0.0:
            ratio_text = f"{errors[i - 1] / err:.3f}"

        rows.append(
            {
                "h": h,
                "steps": n_steps,
                "phase_error_T": err,
                "prev_over_cur": ratio_text,
            }
        )

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    last_ratio = errors[-2] / errors[-1]
    assert last_ratio > 3.5, f"Expected ~2nd-order trend, got ratio={last_ratio:.3f}"
    return table


def run_long_time_energy_demo(params: OscillatorParams) -> pd.DataFrame:
    """Compare long-time energy drift: leapfrog vs explicit Euler."""

    print("\nLong-time energy demo (T=200, h=0.1)")

    dt = 0.1
    t_end = 200.0
    n_steps = int(round(t_end / dt))

    leap = leapfrog_solve(q0=1.0, p0=0.0, dt=dt, n_steps=n_steps, params=params)
    euler = explicit_euler_solve(q0=1.0, p0=0.0, dt=dt, n_steps=n_steps, params=params)

    leap_drift = float(np.max(np.abs(leap.energy - leap.energy[0])))
    euler_drift = float(np.max(np.abs(euler.energy - euler.energy[0])))
    drift_ratio = euler_drift / leap_drift

    table = pd.DataFrame(
        [
            {
                "method": "Leapfrog",
                "max_abs_energy_drift": leap_drift,
                "nfev": leap.nfev,
            },
            {
                "method": "ExplicitEuler",
                "max_abs_energy_drift": euler_drift,
                "nfev": euler.nfev,
            },
        ]
    )

    print(table.to_string(index=False))
    print(f"drift_ratio (Euler / Leapfrog) = {drift_ratio:.3e}")

    assert leap_drift < 5e-3, f"Leapfrog drift too large: {leap_drift:.6e}"
    assert euler_drift > 1.0, "Euler drift unexpectedly small for this setup"
    assert drift_ratio > 1_000.0, "Leapfrog should preserve energy much better than Euler"
    return table


def run_reversibility_demo(params: OscillatorParams) -> float:
    """Check time reversibility by integrating forward then backward."""

    print("\nReversibility demo (forward + backward integration)")

    q0 = 0.7
    p0 = -0.2
    dt = 0.05
    n_steps = 800

    forward = leapfrog_solve(q0=q0, p0=p0, dt=dt, n_steps=n_steps, params=params)
    backward = leapfrog_solve(
        q0=float(forward.q[-1]),
        p0=float(forward.p[-1]),
        dt=-dt,
        n_steps=n_steps,
        params=params,
    )

    err = phase_error(
        q_num=float(backward.q[-1]),
        p_num=float(backward.p[-1]),
        q_ref=q0,
        p_ref=p0,
    )
    print(f"round_trip_phase_error = {err:.6e}")

    assert err < 1e-10, f"Reversibility error too large: {err:.6e}"
    return err


def main() -> None:
    params = OscillatorParams(mass=1.0, spring_k=1.0)

    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    print("Leapfrog Algorithm MVP (PHYS-0324)")
    print("=" * 72)

    run_convergence_demo(params)
    run_long_time_energy_demo(params)
    run_reversibility_demo(params)

    print("=" * 72)
    print("All checks passed.")


if __name__ == "__main__":
    main()
