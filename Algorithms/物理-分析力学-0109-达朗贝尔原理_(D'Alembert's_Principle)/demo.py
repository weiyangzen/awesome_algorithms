"""MVP demo for PHYS-0109: D'Alembert's Principle.

Model:
- Constrained pendulum (mass m, length l) with gravity and time-varying tangential force.
- Dynamics are derived from D'Alembert's principle in generalized coordinate theta.
- Along each simulated state, virtual-work decomposition verifies
  (W_gravity + W_applied + W_constraint + W_inertia) ~= 0,
  while ideal-constraint virtual work stays near zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class PendulumParams:
    """Physical parameters for the constrained pendulum."""

    mass: float
    length: float
    gravity: float
    force_amp: float
    force_freq: float


def basis_vectors(theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return tangent and inward-radial unit vectors for angle theta."""
    tangent_unit = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    inward_radial_unit = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    return tangent_unit, inward_radial_unit


def applied_tangential_force(t: float, params: PendulumParams) -> float:
    """External tangential drive (N), positive along tangent direction."""
    return params.force_amp * np.cos(params.force_freq * t)


def dynamics(t: float, y: np.ndarray, params: PendulumParams) -> np.ndarray:
    """Pendulum ODE from D'Alembert principle in generalized coordinate theta."""
    theta, omega = float(y[0]), float(y[1])
    force_t = applied_tangential_force(t, params)

    alpha = force_t / (params.mass * params.length) - (params.gravity / params.length) * np.sin(theta)
    return np.array([omega, alpha], dtype=float)


def acceleration_vector(theta: float, omega: float, alpha: float, length: float) -> np.ndarray:
    """Cartesian acceleration using theta, omega, alpha."""
    tangent_unit, inward_radial_unit = basis_vectors(theta)
    return length * alpha * tangent_unit + length * omega * omega * inward_radial_unit


def tension_force(theta: float, omega: float, params: PendulumParams) -> float:
    """Constraint force magnitude along inward radial direction."""
    return params.mass * params.length * omega * omega + params.mass * params.gravity * np.cos(theta)


def virtual_work_terms(
    t: float,
    theta: float,
    omega: float,
    alpha: float,
    params: PendulumParams,
    delta_theta: float,
) -> dict[str, float]:
    """Compute virtual-work contributions for one instantaneous state."""
    tangent_unit, inward_radial_unit = basis_vectors(theta)
    dr_dtheta = params.length * tangent_unit
    delta_r = dr_dtheta * delta_theta

    force_t = applied_tangential_force(t, params)
    tension = tension_force(theta, omega, params)

    force_gravity = np.array([0.0, -params.mass * params.gravity], dtype=float)
    force_applied = force_t * tangent_unit
    force_constraint = tension * inward_radial_unit

    accel = acceleration_vector(theta, omega, alpha, params.length)
    force_inertia = -params.mass * accel

    work_gravity = float(np.dot(force_gravity, delta_r))
    work_applied = float(np.dot(force_applied, delta_r))
    work_constraint = float(np.dot(force_constraint, delta_r))
    work_inertia = float(np.dot(force_inertia, delta_r))
    work_total = work_gravity + work_applied + work_constraint + work_inertia

    generalized_residual = float(np.dot(force_gravity + force_applied - params.mass * accel, dr_dtheta))

    return {
        "W_gravity": work_gravity,
        "W_applied": work_applied,
        "W_constraint": work_constraint,
        "W_inertia": work_inertia,
        "W_total": work_total,
        "generalized_residual": generalized_residual,
    }


def run_demo() -> None:
    """Simulate one trajectory and print D'Alembert consistency checks."""
    params = PendulumParams(
        mass=1.0,
        length=0.9,
        gravity=9.81,
        force_amp=2.5,
        force_freq=1.7,
    )

    t0, t1 = 0.0, 8.0
    t_eval = np.linspace(t0, t1, 401)
    y0 = np.array([0.35, -0.2], dtype=float)

    solution = solve_ivp(
        fun=lambda t, y: dynamics(t, y, params),
        t_span=(t0, t1),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")

    delta_theta = 1e-5
    total_virtual_works: list[float] = []
    constraint_virtual_works: list[float] = []
    generalized_residuals: list[float] = []

    for idx, t in enumerate(solution.t):
        theta = float(solution.y[0, idx])
        omega = float(solution.y[1, idx])
        alpha = float(dynamics(t, np.array([theta, omega], dtype=float), params)[1])

        terms = virtual_work_terms(t, theta, omega, alpha, params, delta_theta)
        total_virtual_works.append(terms["W_total"])
        constraint_virtual_works.append(terms["W_constraint"])
        generalized_residuals.append(terms["generalized_residual"])

    arr_total = np.abs(np.array(total_virtual_works, dtype=float))
    arr_constraint = np.abs(np.array(constraint_virtual_works, dtype=float))
    arr_residual = np.abs(np.array(generalized_residuals, dtype=float))

    print("D'Alembert principle MVP check (driven constrained pendulum)")
    print(f"samples                     : {len(solution.t)}")
    print(f"max |W_total|               : {arr_total.max():.3e}")
    print(f"rms |W_total|               : {np.sqrt(np.mean(arr_total**2)):.3e}")
    print(f"max |W_constraint|          : {arr_constraint.max():.3e}")
    print(f"max |generalized residual|  : {arr_residual.max():.3e}")

    tol_total = 1e-8
    tol_constraint = 1e-10
    passed = bool(arr_total.max() < tol_total and arr_constraint.max() < tol_constraint)
    print(f"pass (tol_total={tol_total:.0e}, tol_constraint={tol_constraint:.0e}): {passed}")

    probe_indices = [0, len(solution.t) // 2, len(solution.t) - 1]
    print("\nProbe samples:")
    for idx in probe_indices:
        t = float(solution.t[idx])
        theta = float(solution.y[0, idx])
        omega = float(solution.y[1, idx])
        alpha = float(dynamics(t, np.array([theta, omega], dtype=float), params)[1])
        terms = virtual_work_terms(t, theta, omega, alpha, params, delta_theta)

        print(
            f"t={t:5.2f}, theta={theta:+.6f}, omega={omega:+.6f}, "
            f"W_total={terms['W_total']:+.3e}, W_constraint={terms['W_constraint']:+.3e}"
        )


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
