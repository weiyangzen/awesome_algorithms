"""MVP demo for PHYS-0110: Principle of Virtual Work.

Model:
- A pendulum bob (mass m, rod length l) under gravity and a horizontal force Fx.
- Constraint force (tension) is ideal and should do zero virtual work.
- Static equilibrium is solved via generalized force Q(theta) = 0.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import root_scalar


def pendulum_kinematics(theta: float, length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return position r(theta), tangent dr/dtheta, and inward radial unit vector."""
    r = np.array([length * np.sin(theta), -length * np.cos(theta)], dtype=float)
    tangent = np.array([length * np.cos(theta), length * np.sin(theta)], dtype=float)
    inward_radial_unit = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    return r, tangent, inward_radial_unit


def generalized_force(theta: float, mass: float, gravity: float, length: float, force_x: float) -> float:
    """Generalized force Q_theta from non-constraint forces."""
    return length * (force_x * np.cos(theta) - mass * gravity * np.sin(theta))


def virtual_work_breakdown(
    theta: float,
    delta_theta: float,
    mass: float,
    gravity: float,
    length: float,
    force_x: float,
    tension: float,
) -> dict[str, float]:
    """Compute virtual work contributions from gravity, horizontal force, and tension."""
    _, tangent, inward_radial_unit = pendulum_kinematics(theta, length)
    delta_r = tangent * delta_theta

    force_gravity = np.array([0.0, -mass * gravity], dtype=float)
    force_horizontal = np.array([force_x, 0.0], dtype=float)
    force_tension = tension * inward_radial_unit

    work_gravity = float(np.dot(force_gravity, delta_r))
    work_horizontal = float(np.dot(force_horizontal, delta_r))
    work_tension = float(np.dot(force_tension, delta_r))

    return {
        "W_gravity": work_gravity,
        "W_horizontal": work_horizontal,
        "W_tension": work_tension,
        "W_total": work_gravity + work_horizontal + work_tension,
    }


def solve_equilibrium_theta(mass: float, gravity: float, length: float, force_x: float) -> float:
    """Solve Q_theta(theta)=0 using a secant root solver."""

    def q(theta: float) -> float:
        return generalized_force(theta, mass, gravity, length, force_x)

    theta_guess = float(np.arctan2(force_x, mass * gravity))
    solution = root_scalar(
        q,
        method="secant",
        x0=theta_guess - 0.15,
        x1=theta_guess + 0.15,
        xtol=1e-12,
        rtol=1e-10,
        maxiter=100,
    )
    if not solution.converged:
        raise RuntimeError("Root solver did not converge for equilibrium angle.")

    return float(solution.root)


def run_case(mass: float, gravity: float, length: float, force_x: float) -> None:
    """Run one scenario and print numerical checks for virtual work principle."""
    theta_eq = solve_equilibrium_theta(mass, gravity, length, force_x)
    theta_analytic = float(np.arctan2(force_x, mass * gravity))

    q_eq = generalized_force(theta_eq, mass, gravity, length, force_x)

    delta_theta = 1e-4
    # Any finite tension works for virtual-work orthogonality check.
    assumed_tension = 17.5

    work_eq = virtual_work_breakdown(
        theta_eq,
        delta_theta,
        mass,
        gravity,
        length,
        force_x,
        assumed_tension,
    )

    theta_perturbed = theta_eq + 0.15
    work_perturbed = virtual_work_breakdown(
        theta_perturbed,
        delta_theta,
        mass,
        gravity,
        length,
        force_x,
        assumed_tension,
    )

    print(f"Case Fx={force_x:.3f} N")
    print(f"  theta_numeric   = {theta_eq:+.12f} rad")
    print(f"  theta_analytic  = {theta_analytic:+.12f} rad")
    print(f"  abs_error       = {abs(theta_eq - theta_analytic):.3e}")
    print(f"  Q(theta_eq)     = {q_eq:+.3e}")
    print(
        "  Virtual work @ equilibrium: "
        f"Wg={work_eq['W_gravity']:+.3e}, "
        f"Wx={work_eq['W_horizontal']:+.3e}, "
        f"Wt={work_eq['W_tension']:+.3e}, "
        f"Wtotal={work_eq['W_total']:+.3e}"
    )
    print(
        "  Virtual work @ perturbed (+0.15 rad): "
        f"Wtotal={work_perturbed['W_total']:+.3e}"
    )
    print()


def main() -> None:
    mass = 1.2
    gravity = 9.81
    length = 0.8

    for force_x in (-8.0, 0.0, 6.0, 15.0):
        run_case(mass=mass, gravity=gravity, length=length, force_x=force_x)


if __name__ == "__main__":
    main()
