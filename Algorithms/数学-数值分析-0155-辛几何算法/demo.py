"""Minimal runnable MVP for symplectic algorithms.

The demo integrates a 1D harmonic oscillator Hamiltonian system with:
1) Explicit Euler (non-symplectic baseline),
2) Symplectic Euler,
3) Velocity Verlet (second-order symplectic).

It reports long-time energy behavior and a numerical Jacobian determinant of
one-step maps to highlight symplectic structure preservation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math

import numpy as np


StepFn = Callable[[float, float, float, float], tuple[float, float]]


@dataclass
class SimulationResult:
    method: str
    time: np.ndarray
    q: np.ndarray
    p: np.ndarray
    energy: np.ndarray


def hamiltonian(q: np.ndarray, p: np.ndarray, omega: float) -> np.ndarray:
    """Hamiltonian H(q,p)=0.5*(p^2 + omega^2*q^2)."""
    return 0.5 * (p * p + (omega * q) * (omega * q))


def exact_solution(
    t: np.ndarray,
    q0: float,
    p0: float,
    omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form solution for the 1D harmonic oscillator."""
    wt = omega * t
    q = q0 * np.cos(wt) + (p0 / omega) * np.sin(wt)
    p = -q0 * omega * np.sin(wt) + p0 * np.cos(wt)
    return q, p


def explicit_euler_step(q: float, p: float, h: float, omega: float) -> tuple[float, float]:
    """Non-symplectic explicit Euler (baseline)."""
    q_next = q + h * p
    p_next = p - h * (omega * omega) * q
    return q_next, p_next


def symplectic_euler_step(q: float, p: float, h: float, omega: float) -> tuple[float, float]:
    """Symplectic Euler (kick-drift form)."""
    p_next = p - h * (omega * omega) * q
    q_next = q + h * p_next
    return q_next, p_next


def velocity_verlet_step(q: float, p: float, h: float, omega: float) -> tuple[float, float]:
    """Velocity Verlet (Stoermer-Verlet), second-order symplectic."""
    a_n = -(omega * omega) * q
    p_half = p + 0.5 * h * a_n
    q_next = q + h * p_half
    a_next = -(omega * omega) * q_next
    p_next = p_half + 0.5 * h * a_next
    return q_next, p_next


def integrate(
    step_fn: StepFn,
    method_name: str,
    q0: float,
    p0: float,
    h: float,
    steps: int,
    omega: float,
) -> SimulationResult:
    """Run a full trajectory with a fixed-step integrator."""
    if h <= 0.0:
        raise ValueError(f"step size h must be positive, got {h}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    q = np.empty(steps + 1, dtype=float)
    p = np.empty(steps + 1, dtype=float)
    t = h * np.arange(steps + 1, dtype=float)

    q[0] = float(q0)
    p[0] = float(p0)

    for i in range(steps):
        q[i + 1], p[i + 1] = step_fn(float(q[i]), float(p[i]), h, omega)
        if not (math.isfinite(float(q[i + 1])) and math.isfinite(float(p[i + 1]))):
            raise RuntimeError(f"{method_name} produced non-finite values at step {i + 1}")

    energy = hamiltonian(q, p, omega)
    return SimulationResult(method=method_name, time=t, q=q, p=p, energy=energy)


def step_jacobian_determinant(
    step_fn: StepFn,
    q: float,
    p: float,
    h: float,
    omega: float,
    eps: float = 1e-7,
) -> float:
    """Numerically estimate det(dPhi_h/d(q,p)) for one-step map Phi_h."""
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    q_plus, p_plus = step_fn(q + eps, p, h, omega)
    q_minus, p_minus = step_fn(q - eps, p, h, omega)
    dphi_dq = np.array(
        [(q_plus - q_minus) / (2.0 * eps), (p_plus - p_minus) / (2.0 * eps)],
        dtype=float,
    )

    q_plus, p_plus = step_fn(q, p + eps, h, omega)
    q_minus, p_minus = step_fn(q, p - eps, h, omega)
    dphi_dp = np.array(
        [(q_plus - q_minus) / (2.0 * eps), (p_plus - p_minus) / (2.0 * eps)],
        dtype=float,
    )

    jacobian = np.column_stack([dphi_dq, dphi_dp])
    return float(np.linalg.det(jacobian))


def summarize_result(
    result: SimulationResult,
    q_exact: np.ndarray,
    p_exact: np.ndarray,
    energy0: float,
) -> dict[str, float | str]:
    """Build scalar metrics for compact comparison."""
    energy_error = result.energy - energy0
    state_error = np.sqrt((result.q - q_exact) ** 2 + (result.p - p_exact) ** 2)

    return {
        "method": result.method,
        "final_energy_drift": float(energy_error[-1]),
        "max_abs_energy_drift": float(np.max(np.abs(energy_error))),
        "rms_state_error": float(np.sqrt(np.mean(state_error**2))),
        "final_state_error": float(state_error[-1]),
    }


def print_metrics_table(rows: list[dict[str, float | str]]) -> None:
    """Pretty-print metrics for all methods."""
    header = (
        "Method             | final dH         | max |dH|         | "
        "RMS state err     | final state err   | det(J_step)"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{str(row['method']):<18s} | "
            f"{float(row['final_energy_drift']):>15.6e} | "
            f"{float(row['max_abs_energy_drift']):>15.6e} | "
            f"{float(row['rms_state_error']):>15.6e} | "
            f"{float(row['final_state_error']):>15.6e} | "
            f"{float(row['jacobian_det']):>10.7f}"
        )


def main() -> None:
    omega = 1.0
    q0 = 1.0
    p0 = 0.0
    h = 0.1
    periods = 80
    total_time = periods * 2.0 * math.pi / omega
    steps = int(total_time / h)

    print("=" * 90)
    print("Symplectic Algorithm MVP: Harmonic Oscillator Long-Time Integration")
    print("=" * 90)
    print(f"omega={omega}, q0={q0}, p0={p0}, h={h}, periods={periods}, steps={steps}")

    integrators: list[tuple[str, StepFn]] = [
        ("Explicit Euler", explicit_euler_step),
        ("Symplectic Euler", symplectic_euler_step),
        ("Velocity Verlet", velocity_verlet_step),
    ]

    t = h * np.arange(steps + 1, dtype=float)
    q_exact, p_exact = exact_solution(t=t, q0=q0, p0=p0, omega=omega)
    energy0 = float(hamiltonian(np.array([q0]), np.array([p0]), omega)[0])

    summary_rows: list[dict[str, float | str]] = []
    for method_name, step_fn in integrators:
        result = integrate(
            step_fn=step_fn,
            method_name=method_name,
            q0=q0,
            p0=p0,
            h=h,
            steps=steps,
            omega=omega,
        )
        row = summarize_result(result=result, q_exact=q_exact, p_exact=p_exact, energy0=energy0)
        row["jacobian_det"] = step_jacobian_determinant(
            step_fn=step_fn,
            q=0.3,
            p=-0.7,
            h=h,
            omega=omega,
            eps=1e-7,
        )
        summary_rows.append(row)

    print_metrics_table(summary_rows)
    print("\nInterpretation:")
    print("- Symplectic methods keep det(J_step)~=1 and bound long-time energy error.")
    print("- Explicit Euler has det(J_step)>1 and accumulates strong energy drift.")


if __name__ == "__main__":
    main()
