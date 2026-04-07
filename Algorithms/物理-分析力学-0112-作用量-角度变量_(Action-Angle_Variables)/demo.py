"""Minimal runnable MVP for Action-Angle Variables (PHYS-0112).

Model:
    1D harmonic oscillator with Hamiltonian
        H(q, p) = p^2 / (2m) + (m * omega^2 * q^2) / 2

Action-angle transform:
    J = E / omega
    q = sqrt(2J / (m * omega)) * sin(theta)
    p = sqrt(2m * omega * J) * cos(theta)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def qp_from_action_angle(
    action: np.ndarray | float,
    angle: np.ndarray | float,
    mass: float,
    omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map action-angle coordinates (J, theta) to canonical coordinates (q, p)."""
    j = np.asarray(action, dtype=float)
    theta = np.asarray(angle, dtype=float)
    if np.any(j < 0.0):
        raise ValueError("action J must be non-negative")
    if mass <= 0.0 or omega <= 0.0:
        raise ValueError("mass and omega must be positive")

    q = np.sqrt(2.0 * j / (mass * omega)) * np.sin(theta)
    p = np.sqrt(2.0 * mass * omega * j) * np.cos(theta)
    return q, p


def action_from_qp(
    q: np.ndarray | float,
    p: np.ndarray | float,
    mass: float,
    omega: float,
) -> np.ndarray:
    """Compute action J from (q, p) for harmonic oscillator."""
    q_arr = np.asarray(q, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    if mass <= 0.0 or omega <= 0.0:
        raise ValueError("mass and omega must be positive")
    return 0.5 * (p_arr**2 / (mass * omega) + mass * omega * q_arr**2)


def angle_from_qp(
    q: np.ndarray | float,
    p: np.ndarray | float,
    mass: float,
    omega: float,
) -> np.ndarray:
    """Recover angle variable theta from (q, p) using atan2(m*omega*q, p)."""
    q_arr = np.asarray(q, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    if mass <= 0.0 or omega <= 0.0:
        raise ValueError("mass and omega must be positive")
    return np.arctan2(mass * omega * q_arr, p_arr)


def velocity_verlet_harmonic(
    q0: float,
    p0: float,
    mass: float,
    omega: float,
    dt: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate harmonic Hamiltonian dynamics via velocity Verlet.

    Hamilton equations:
        q_dot = p / m
        p_dot = -m * omega^2 * q
    """
    if mass <= 0.0 or omega <= 0.0:
        raise ValueError("mass and omega must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    q = np.empty(n_steps + 1, dtype=float)
    p = np.empty(n_steps + 1, dtype=float)
    t = np.linspace(0.0, dt * n_steps, n_steps + 1)

    q[0] = q0
    p[0] = p0
    k = mass * omega**2

    for i in range(n_steps):
        p_half = p[i] - 0.5 * dt * k * q[i]
        q[i + 1] = q[i] + dt * p_half / mass
        p[i + 1] = p_half - 0.5 * dt * k * q[i + 1]

    return t, q, p


def main() -> None:
    mass = 1.3
    omega = 2.4
    j0 = 1.1
    theta0 = 0.35
    dt = 0.0015
    n_steps = 8000

    q0, p0 = qp_from_action_angle(j0, theta0, mass=mass, omega=omega)
    q0_s = float(np.asarray(q0))
    p0_s = float(np.asarray(p0))

    t, q, p = velocity_verlet_harmonic(
        q0=q0_s,
        p0=p0_s,
        mass=mass,
        omega=omega,
        dt=dt,
        n_steps=n_steps,
    )

    action_t = action_from_qp(q, p, mass=mass, omega=omega)
    theta_t = np.unwrap(angle_from_qp(q, p, mass=mass, omega=omega))
    theta_expected = theta0 + omega * t

    # Frequency estimate from least-squares line fit theta(t) ~ a*t + b.
    slope, intercept = np.polyfit(t, theta_t, deg=1)

    # Round-trip consistency (q,p) -> (J,theta) -> (q,p).
    q_recon, p_recon = qp_from_action_angle(action_t, theta_t, mass=mass, omega=omega)
    max_recon_q = float(np.max(np.abs(q_recon - q)))
    max_recon_p = float(np.max(np.abs(p_recon - p)))

    max_action_abs_dev = float(np.max(np.abs(action_t - j0)))
    max_action_rel_dev = max_action_abs_dev / j0
    rms_theta_err = float(np.sqrt(np.mean((theta_t - theta_expected) ** 2)))
    omega_rel_err = float(abs((slope - omega) / omega))

    sample_idx = np.linspace(0, n_steps, 8, dtype=int)
    df_sample = pd.DataFrame(
        {
            "t": t[sample_idx],
            "q": q[sample_idx],
            "p": p[sample_idx],
            "J_from_qp": action_t[sample_idx],
            "theta_recovered": theta_t[sample_idx],
            "theta_expected": theta_expected[sample_idx],
        }
    )

    print("Action-Angle Variables MVP (PHYS-0112)")
    print("=" * 78)
    print(
        "Parameters: "
        f"m={mass:.3f}, omega={omega:.3f}, J0={j0:.3f}, "
        f"theta0={theta0:.3f}, dt={dt:.4f}, n_steps={n_steps}"
    )
    print("\nSample trajectory states:")
    print(df_sample.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nQuality checks:")
    print(f"- max |J(t)-J0| / J0       = {max_action_rel_dev:.6e}")
    print(f"- RMS angle tracking error = {rms_theta_err:.6e}")
    print(f"- relative freq error      = {omega_rel_err:.6e}")
    print(f"- max round-trip |dq|      = {max_recon_q:.6e}")
    print(f"- max round-trip |dp|      = {max_recon_p:.6e}")

    assert max_action_rel_dev < 1.0e-5, (
        f"Action invariance too poor: rel dev {max_action_rel_dev:.3e}"
    )
    assert rms_theta_err < 3.0e-3, f"Angle linearity mismatch too large: {rms_theta_err:.3e}"
    assert omega_rel_err < 8.0e-5, f"Frequency estimate mismatch: {omega_rel_err:.3e}"
    assert max_recon_q < 1.0e-12, f"q round-trip mismatch too large: {max_recon_q:.3e}"
    assert max_recon_p < 1.0e-12, f"p round-trip mismatch too large: {max_recon_p:.3e}"

    print("=" * 78)
    print("All checks passed.")
    print(f"Estimated theta(t) ≈ {slope:.8f} * t + {intercept:.8f}")


if __name__ == "__main__":
    main()
