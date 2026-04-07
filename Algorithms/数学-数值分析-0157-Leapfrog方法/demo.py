"""Minimal runnable MVP for Leapfrog method (MATH-0157)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np


@dataclass
class HarmonicOscillator:
    """a(q) = -omega^2 * q with evaluation counter."""

    omega: float = 1.0
    nfev: int = 0

    def acceleration(self, q: float) -> float:
        self.nfev += 1
        return -(self.omega * self.omega) * q


@dataclass
class Trajectory:
    t: np.ndarray
    q: np.ndarray
    p: np.ndarray
    energy: np.ndarray
    nfev: int


def validate_inputs(q0: float, p0: float, h: float, n_steps: int, omega: float) -> None:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("h must be finite and positive")
    if not math.isfinite(q0) or not math.isfinite(p0):
        raise ValueError("q0 and p0 must be finite")
    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and positive")


def exact_harmonic(t: np.ndarray, omega: float, q0: float, p0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Exact solution for q'' + omega^2 q = 0."""
    wt = omega * t
    q = q0 * np.cos(wt) + (p0 / omega) * np.sin(wt)
    p = -q0 * omega * np.sin(wt) + p0 * np.cos(wt)
    return q, p


def leapfrog_solve(q0: float, p0: float, h: float, n_steps: int, omega: float = 1.0) -> Trajectory:
    """
    Leapfrog (kick-drift-kick) for separable Hamiltonian system:
        q' = p
        p' = -omega^2 q
    """
    validate_inputs(q0=q0, p0=p0, h=h, n_steps=n_steps, omega=omega)

    osc = HarmonicOscillator(omega=omega)
    t = h * np.arange(n_steps + 1, dtype=float)
    q = np.empty(n_steps + 1, dtype=float)
    p = np.empty(n_steps + 1, dtype=float)

    q[0] = q0
    p[0] = p0

    for n in range(n_steps):
        a_n = osc.acceleration(float(q[n]))
        p_half = p[n] + 0.5 * h * a_n
        q[n + 1] = q[n] + h * p_half
        a_np1 = osc.acceleration(float(q[n + 1]))
        p[n + 1] = p_half + 0.5 * h * a_np1

        if not math.isfinite(float(q[n + 1])) or not math.isfinite(float(p[n + 1])):
            raise RuntimeError("non-finite value encountered in leapfrog integration")

    energy = 0.5 * (p * p + (omega * q) * (omega * q))
    return Trajectory(t=t, q=q, p=p, energy=energy, nfev=osc.nfev)


def explicit_euler_solve(q0: float, p0: float, h: float, n_steps: int, omega: float = 1.0) -> Trajectory:
    """Explicit Euler baseline for comparison on the same Hamiltonian system."""
    validate_inputs(q0=q0, p0=p0, h=h, n_steps=n_steps, omega=omega)

    osc = HarmonicOscillator(omega=omega)
    t = h * np.arange(n_steps + 1, dtype=float)
    q = np.empty(n_steps + 1, dtype=float)
    p = np.empty(n_steps + 1, dtype=float)
    q[0] = q0
    p[0] = p0

    for n in range(n_steps):
        a_n = osc.acceleration(float(q[n]))
        q[n + 1] = q[n] + h * p[n]
        p[n + 1] = p[n] + h * a_n

        if not math.isfinite(float(q[n + 1])) or not math.isfinite(float(p[n + 1])):
            raise RuntimeError("non-finite value encountered in Euler integration")

    energy = 0.5 * (p * p + (omega * q) * (omega * q))
    return Trajectory(t=t, q=q, p=p, energy=energy, nfev=osc.nfev)


def phase_error(q_num: float, p_num: float, q_exact: float, p_exact: float) -> float:
    """Euclidean error on (q, p) phase plane at one time point."""
    dq = q_num - q_exact
    dp = p_num - p_exact
    return float(math.sqrt(dq * dq + dp * dp))


def run_convergence_demo() -> Tuple[List[float], List[float]]:
    """Show second-order convergence of Leapfrog on a short horizon."""
    print("Convergence demo (Leapfrog on q''+q=0, T=2)")
    print("h        steps   phase_error(T)        prev/cur")

    h_values = [0.2, 0.1, 0.05, 0.025]
    errors: List[float] = []
    t_end = 2.0

    for i, h in enumerate(h_values):
        n_steps = int(round(t_end / h))
        traj = leapfrog_solve(q0=1.0, p0=0.0, h=h, n_steps=n_steps, omega=1.0)
        q_e, p_e = exact_harmonic(np.array([t_end]), omega=1.0, q0=1.0, p0=0.0)
        err = phase_error(
            q_num=float(traj.q[-1]),
            p_num=float(traj.p[-1]),
            q_exact=float(q_e[0]),
            p_exact=float(p_e[0]),
        )
        errors.append(err)

        ratio_text = "-"
        if i > 0 and err > 0.0:
            ratio_text = f"{errors[i - 1] / err:.3f}"

        print(f"{h:<8.3f} {n_steps:<7d} {err:<20.6e} {ratio_text}")

    last_ratio = errors[-2] / errors[-1]
    assert last_ratio > 3.5, f"Expected second-order trend, got ratio={last_ratio:.3f}"
    return h_values, errors


def run_energy_demo() -> None:
    """Compare long-time energy drift between Leapfrog and explicit Euler."""
    print("\nLong-time energy drift demo (T=200, h=0.1)")
    h = 0.1
    t_end = 200.0
    n_steps = int(round(t_end / h))

    leap = leapfrog_solve(q0=1.0, p0=0.0, h=h, n_steps=n_steps, omega=1.0)
    euler = explicit_euler_solve(q0=1.0, p0=0.0, h=h, n_steps=n_steps, omega=1.0)

    leap_drift = float(np.max(np.abs(leap.energy - leap.energy[0])))
    euler_drift = float(np.max(np.abs(euler.energy - euler.energy[0])))

    print(f"Leapfrog: max |H-H0| = {leap_drift:.6e}, nfev={leap.nfev}")
    print(f"Euler   : max |H-H0| = {euler_drift:.6e}, nfev={euler.nfev}")
    print(f"drift_ratio (Euler / Leapfrog) = {euler_drift / leap_drift:.3e}")

    assert leap_drift < 5e-3, f"Leapfrog energy drift too large: {leap_drift}"
    assert euler_drift > 1.0, "Euler drift unexpectedly small on this setup"
    assert euler_drift > 1_000.0 * leap_drift, "Leapfrog should preserve energy much better"


def main() -> None:
    print("Leapfrog Method MVP (MATH-0157)")
    print("=" * 72)
    run_convergence_demo()
    run_energy_demo()
    print("=" * 72)
    print("All checks passed.")


if __name__ == "__main__":
    main()
