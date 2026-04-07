"""Minimal runnable MVP for two coupled oscillators.

Model:
    m1*x1'' + (k1+kc)*x1 - kc*x2 = 0
    m2*x2'' + (k2+kc)*x2 - kc*x1 = 0
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh


@dataclass(frozen=True)
class CoupledOscillatorParams:
    m1: float
    m2: float
    k1: float
    k2: float
    kc: float
    dt: float
    steps: int


def validate_params(p: CoupledOscillatorParams) -> None:
    values = {
        "m1": p.m1,
        "m2": p.m2,
        "k1": p.k1,
        "k2": p.k2,
        "kc": p.kc,
        "dt": p.dt,
    }
    for key, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite, got {value!r}")
    if p.m1 <= 0.0 or p.m2 <= 0.0:
        raise ValueError("m1 and m2 must be positive")
    if p.k1 < 0.0 or p.k2 < 0.0 or p.kc < 0.0:
        raise ValueError("k1, k2, kc must be non-negative")
    if p.dt <= 0.0:
        raise ValueError("dt must be positive")
    if p.steps <= 0:
        raise ValueError("steps must be positive")


def mass_matrix(p: CoupledOscillatorParams) -> np.ndarray:
    return np.array([[p.m1, 0.0], [0.0, p.m2]], dtype=float)


def stiffness_matrix(p: CoupledOscillatorParams) -> np.ndarray:
    return np.array([[p.k1 + p.kc, -p.kc], [-p.kc, p.k2 + p.kc]], dtype=float)


def normal_modes(m_mat: np.ndarray, k_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve K*phi = lambda*M*phi with lambda=omega^2.

    Returns:
        omega: natural angular frequencies, shape (2,)
        phi: mode matrix with mass-orthonormal columns, shape (2, 2)
    """
    eigvals, eigvecs = eigh(k_mat, m_mat)
    omega = np.sqrt(np.maximum(eigvals, 0.0))
    return omega, eigvecs


def rhs(state: np.ndarray, minv: np.ndarray, k_mat: np.ndarray) -> np.ndarray:
    x = state[:2]
    v = state[2:]
    a = -minv @ (k_mat @ x)
    out = np.empty_like(state)
    out[:2] = v
    out[2:] = a
    return out


def rk4_step(state: np.ndarray, dt: float, minv: np.ndarray, k_mat: np.ndarray) -> np.ndarray:
    k1 = rhs(state, minv=minv, k_mat=k_mat)
    k2 = rhs(state + 0.5 * dt * k1, minv=minv, k_mat=k_mat)
    k3 = rhs(state + 0.5 * dt * k2, minv=minv, k_mat=k_mat)
    k4 = rhs(state + dt * k3, minv=minv, k_mat=k_mat)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_rk4(
    p: CoupledOscillatorParams,
    x0: np.ndarray,
    v0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_params(p)
    x0 = np.asarray(x0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    if x0.shape != (2,) or v0.shape != (2,):
        raise ValueError("x0 and v0 must be shape (2,)")
    if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(v0)):
        raise ValueError("x0 and v0 must be finite")

    m_mat = mass_matrix(p)
    k_mat = stiffness_matrix(p)
    minv = np.linalg.inv(m_mat)

    times = np.linspace(0.0, p.dt * p.steps, p.steps + 1, dtype=float)
    states = np.empty((p.steps + 1, 4), dtype=float)
    states[0, :2] = x0
    states[0, 2:] = v0

    for i in range(p.steps):
        states[i + 1] = rk4_step(states[i], dt=p.dt, minv=minv, k_mat=k_mat)
        if not np.all(np.isfinite(states[i + 1])):
            raise RuntimeError("non-finite state encountered")

    return times, states[:, :2], states[:, 2:]


def analytic_solution(
    times: np.ndarray,
    p: CoupledOscillatorParams,
    x0: np.ndarray,
    v0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m_mat = mass_matrix(p)
    k_mat = stiffness_matrix(p)
    omega, phi = normal_modes(m_mat, k_mat)

    x0 = np.asarray(x0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    q0 = phi.T @ (m_mat @ x0)
    qd0 = phi.T @ (m_mat @ v0)

    q_t = np.empty((times.shape[0], 2), dtype=float)
    qd_t = np.empty((times.shape[0], 2), dtype=float)

    for i, t in enumerate(times):
        for j in range(2):
            w = omega[j]
            if w <= 1e-14:
                q_t[i, j] = q0[j] + qd0[j] * t
                qd_t[i, j] = qd0[j]
            else:
                c = math.cos(w * t)
                s = math.sin(w * t)
                q_t[i, j] = q0[j] * c + (qd0[j] / w) * s
                qd_t[i, j] = -q0[j] * w * s + qd0[j] * c

    x_t = q_t @ phi.T
    v_t = qd_t @ phi.T
    return omega, x_t, v_t


def total_energy(x: np.ndarray, v: np.ndarray, m_mat: np.ndarray, k_mat: np.ndarray) -> np.ndarray:
    kinetic = 0.5 * np.einsum("bi,ij,bj->b", v, m_mat, v)
    potential = 0.5 * np.einsum("bi,ij,bj->b", x, k_mat, x)
    return kinetic + potential


def build_sample_table(times: np.ndarray, x_num: np.ndarray, v_num: np.ndarray, rows: int = 8) -> pd.DataFrame:
    n = min(rows, times.shape[0])
    return pd.DataFrame(
        {
            "t_s": times[:n],
            "x1_m": x_num[:n, 0],
            "x2_m": x_num[:n, 1],
            "v1_mps": v_num[:n, 0],
            "v2_mps": v_num[:n, 1],
        }
    )


def main() -> None:
    params = CoupledOscillatorParams(
        m1=1.0,
        m2=1.2,
        k1=12.0,
        k2=10.0,
        kc=4.0,
        dt=0.002,
        steps=6000,
    )
    x0 = np.array([0.08, -0.03], dtype=float)
    v0 = np.array([0.0, 0.12], dtype=float)

    times, x_num, v_num = simulate_rk4(params, x0=x0, v0=v0)
    omega, x_exact, v_exact = analytic_solution(times, params, x0=x0, v0=v0)

    pos_err = np.max(np.abs(x_num - x_exact), axis=0)
    vel_err = np.max(np.abs(v_num - v_exact), axis=0)
    max_pos_err = float(np.max(pos_err))
    max_vel_err = float(np.max(vel_err))

    m_mat = mass_matrix(params)
    k_mat = stiffness_matrix(params)
    e = total_energy(x_num, v_num, m_mat, k_mat)
    rel_energy_drift = float(np.max(np.abs(e - e[0])) / max(abs(e[0]), 1e-12))

    summary = pd.DataFrame(
        {
            "metric": [
                "omega1_rad_s",
                "omega2_rad_s",
                "freq1_hz",
                "freq2_hz",
                "max_abs_position_error_m",
                "max_abs_velocity_error_mps",
                "relative_energy_drift",
            ],
            "value": [
                float(omega[0]),
                float(omega[1]),
                float(omega[0] / (2.0 * math.pi)),
                float(omega[1] / (2.0 * math.pi)),
                max_pos_err,
                max_vel_err,
                rel_energy_drift,
            ],
        }
    )

    print("Coupled oscillators MVP (2-DOF, undamped, unforced)")
    print("Parameters:")
    print(params)
    print("\nSummary metrics:")
    print(summary.to_string(index=False, justify="left", float_format=lambda x: f"{x:.8e}"))
    print("\nTrajectory sample (RK4):")
    print(build_sample_table(times, x_num, v_num).to_string(index=False, float_format=lambda x: f"{x:.8e}"))

    # Automatic checks for reproducible non-interactive validation.
    assert max_pos_err < 2e-6, f"position error too large: {max_pos_err}"
    assert max_vel_err < 2e-5, f"velocity error too large: {max_vel_err}"
    assert rel_energy_drift < 2e-9, f"energy drift too large: {rel_energy_drift}"


if __name__ == "__main__":
    main()
