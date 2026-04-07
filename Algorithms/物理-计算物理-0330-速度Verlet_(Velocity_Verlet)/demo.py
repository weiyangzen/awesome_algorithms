"""Minimal runnable MVP for Velocity Verlet (PHYS-0325).

This script implements velocity-Verlet integration on a 1D harmonic oscillator,
then validates three expected properties:
1) second-order convergence,
2) bounded long-time energy drift (vs explicit Euler baseline),
3) time reversibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OscillatorParams:
    """Physical parameters for a 1D harmonic oscillator."""

    mass: float = 1.0
    spring_k: float = 1.0


@dataclass(frozen=True)
class Trajectory:
    """Discrete trajectory and diagnostics."""

    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    energy: np.ndarray
    n_force_evals: int


class HarmonicAcceleration:
    """Acceleration model a(q) = -(k/m) * q with evaluation counting."""

    def __init__(self, params: OscillatorParams) -> None:
        self.params = params
        self.n_evals = 0

    def __call__(self, q: float) -> float:
        self.n_evals += 1
        return -(self.params.spring_k / self.params.mass) * q


def validate_inputs(
    q0: float,
    v0: float,
    dt: float,
    n_steps: int,
    params: OscillatorParams,
) -> None:
    """Fail fast on invalid integration settings."""

    if not math.isfinite(q0) or not math.isfinite(v0):
        raise ValueError("q0 and v0 must be finite")
    if not math.isfinite(dt) or dt == 0.0:
        raise ValueError("dt must be finite and non-zero")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if not math.isfinite(params.mass) or params.mass <= 0.0:
        raise ValueError("mass must be finite and positive")
    if not math.isfinite(params.spring_k) or params.spring_k <= 0.0:
        raise ValueError("spring_k must be finite and positive")


def steps_for_duration(t_end: float, dt: float) -> int:
    """Convert duration and step size into an integer number of steps."""

    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if dt == 0.0:
        raise ValueError("dt must be non-zero")

    raw = t_end / abs(dt)
    n_steps = int(round(raw))
    if n_steps <= 0:
        raise ValueError("invalid n_steps derived from t_end and dt")

    if not math.isclose(raw, float(n_steps), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("t_end must be an integer multiple of |dt| for this MVP")
    return n_steps


def exact_state(t: np.ndarray, q0: float, v0: float, params: OscillatorParams) -> tuple[np.ndarray, np.ndarray]:
    """Analytic solution of m*q'' + k*q = 0."""

    omega = math.sqrt(params.spring_k / params.mass)
    wt = omega * t
    q = q0 * np.cos(wt) + (v0 / omega) * np.sin(wt)
    v = -q0 * omega * np.sin(wt) + v0 * np.cos(wt)
    return q, v


def velocity_verlet_integrate(
    q0: float,
    v0: float,
    dt: float,
    n_steps: int,
    params: OscillatorParams,
) -> Trajectory:
    """Velocity-Verlet integration for 1D separable Hamiltonian."""

    validate_inputs(q0=q0, v0=v0, dt=dt, n_steps=n_steps, params=params)
    accel = HarmonicAcceleration(params)

    t = dt * np.arange(n_steps + 1, dtype=float)
    q = np.empty(n_steps + 1, dtype=float)
    v = np.empty(n_steps + 1, dtype=float)
    energy = np.empty(n_steps + 1, dtype=float)

    q[0] = q0
    v[0] = v0
    energy[0] = 0.5 * params.mass * v0 * v0 + 0.5 * params.spring_k * q0 * q0

    a_n = accel(float(q[0]))
    for n in range(n_steps):
        q_np1 = q[n] + v[n] * dt + 0.5 * a_n * dt * dt
        a_np1 = accel(float(q_np1))
        v_np1 = v[n] + 0.5 * (a_n + a_np1) * dt

        if not math.isfinite(q_np1) or not math.isfinite(v_np1):
            raise RuntimeError("non-finite state encountered in velocity-Verlet integration")

        q[n + 1] = q_np1
        v[n + 1] = v_np1
        energy[n + 1] = 0.5 * params.mass * v_np1 * v_np1 + 0.5 * params.spring_k * q_np1 * q_np1
        a_n = a_np1

    return Trajectory(t=t, q=q, v=v, energy=energy, n_force_evals=accel.n_evals)


def explicit_euler_integrate(
    q0: float,
    v0: float,
    dt: float,
    n_steps: int,
    params: OscillatorParams,
) -> Trajectory:
    """Explicit Euler baseline integrator."""

    validate_inputs(q0=q0, v0=v0, dt=dt, n_steps=n_steps, params=params)
    accel = HarmonicAcceleration(params)

    t = dt * np.arange(n_steps + 1, dtype=float)
    q = np.empty(n_steps + 1, dtype=float)
    v = np.empty(n_steps + 1, dtype=float)
    energy = np.empty(n_steps + 1, dtype=float)

    q[0] = q0
    v[0] = v0
    energy[0] = 0.5 * params.mass * v0 * v0 + 0.5 * params.spring_k * q0 * q0

    for n in range(n_steps):
        a_n = accel(float(q[n]))
        q_np1 = q[n] + v[n] * dt
        v_np1 = v[n] + a_n * dt

        if not math.isfinite(q_np1) or not math.isfinite(v_np1):
            raise RuntimeError("non-finite state encountered in Euler integration")

        q[n + 1] = q_np1
        v[n + 1] = v_np1
        energy[n + 1] = 0.5 * params.mass * v_np1 * v_np1 + 0.5 * params.spring_k * q_np1 * q_np1

    return Trajectory(t=t, q=q, v=v, energy=energy, n_force_evals=accel.n_evals)


def phase_error(q_num: float, v_num: float, q_ref: float, v_ref: float) -> float:
    """Euclidean state error in (q, v) phase space."""

    dq = q_num - q_ref
    dv = v_num - v_ref
    return float(math.sqrt(dq * dq + dv * dv))


def run_convergence_study(params: OscillatorParams) -> pd.DataFrame:
    """Show second-order global error of velocity-Verlet."""

    print("Convergence study (velocity-Verlet, harmonic oscillator)")

    q0 = 1.0
    v0 = 0.0
    t_end = 2.0
    h_values = [0.2, 0.1, 0.05, 0.025]

    rows: list[dict[str, float | int | str]] = []
    errors: list[float] = []

    for idx, h in enumerate(h_values):
        n_steps = steps_for_duration(t_end=t_end, dt=h)
        traj = velocity_verlet_integrate(q0=q0, v0=v0, dt=h, n_steps=n_steps, params=params)

        q_ref, v_ref = exact_state(np.array([t_end], dtype=float), q0=q0, v0=v0, params=params)
        err = phase_error(
            q_num=float(traj.q[-1]),
            v_num=float(traj.v[-1]),
            q_ref=float(q_ref[0]),
            v_ref=float(v_ref[0]),
        )
        errors.append(err)

        ratio_text = "-"
        if idx > 0 and err > 0.0:
            ratio_text = f"{errors[idx - 1] / err:.3f}"

        rows.append(
            {
                "h": h,
                "steps": n_steps,
                "phase_error": err,
                "prev_over_cur": ratio_text,
                "n_force_evals": traj.n_force_evals,
            }
        )

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    last_ratio = errors[-2] / errors[-1]
    assert last_ratio > 3.9, f"Expected near 2nd-order behavior, got ratio={last_ratio:.3f}"
    return table


def max_abs_energy_drift(energy: np.ndarray) -> float:
    """Maximum absolute energy drift with respect to initial energy."""

    return float(np.max(np.abs(energy - energy[0])))


def run_energy_benchmark(params: OscillatorParams) -> pd.DataFrame:
    """Compare long-time energy behavior: velocity-Verlet vs explicit Euler."""

    print("\nLong-time energy benchmark")

    q0 = 1.0
    v0 = 0.0
    dt = 0.1
    t_end = 200.0
    n_steps = steps_for_duration(t_end=t_end, dt=dt)

    vv = velocity_verlet_integrate(q0=q0, v0=v0, dt=dt, n_steps=n_steps, params=params)
    eu = explicit_euler_integrate(q0=q0, v0=v0, dt=dt, n_steps=n_steps, params=params)

    vv_drift = max_abs_energy_drift(vv.energy)
    eu_drift = max_abs_energy_drift(eu.energy)
    drift_ratio = eu_drift / vv_drift

    table = pd.DataFrame(
        [
            {
                "method": "VelocityVerlet",
                "dt": dt,
                "steps": n_steps,
                "max_abs_energy_drift": vv_drift,
                "n_force_evals": vv.n_force_evals,
            },
            {
                "method": "ExplicitEuler",
                "dt": dt,
                "steps": n_steps,
                "max_abs_energy_drift": eu_drift,
                "n_force_evals": eu.n_force_evals,
            },
        ]
    )

    print(table.to_string(index=False))
    print(f"drift_ratio (Euler / VelocityVerlet) = {drift_ratio:.3e}")

    assert vv_drift < 2e-3, f"Velocity-Verlet energy drift too large: {vv_drift:.6e}"
    assert eu_drift > 1.0, "Euler drift unexpectedly small for this setup"
    assert drift_ratio > 1e6, "Expected Velocity-Verlet to outperform Euler in long-time energy behavior"
    return table


def run_reversibility_check(params: OscillatorParams) -> float:
    """Forward then backward integration should return to initial state."""

    print("\nTime reversibility check")

    q0 = 0.7
    v0 = -0.2
    dt = 0.05
    n_steps = 800

    forward = velocity_verlet_integrate(q0=q0, v0=v0, dt=dt, n_steps=n_steps, params=params)
    backward = velocity_verlet_integrate(
        q0=float(forward.q[-1]),
        v0=float(forward.v[-1]),
        dt=-dt,
        n_steps=n_steps,
        params=params,
    )

    round_trip = phase_error(
        q_num=float(backward.q[-1]),
        v_num=float(backward.v[-1]),
        q_ref=q0,
        v_ref=v0,
    )
    print(f"round_trip_phase_error = {round_trip:.3e}")

    assert round_trip < 1e-10, f"Round-trip error too large: {round_trip:.3e}"
    return round_trip


def main() -> None:
    params = OscillatorParams(mass=1.0, spring_k=1.0)

    convergence_table = run_convergence_study(params)
    energy_table = run_energy_benchmark(params)
    round_trip_error = run_reversibility_check(params)

    if convergence_table.empty or energy_table.empty:
        raise RuntimeError("unexpected empty result table")
    if round_trip_error <= 0.0:
        # Very tiny positive error is expected from floating-point arithmetic.
        raise RuntimeError("unexpected non-positive round-trip error")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
