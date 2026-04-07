"""Minimal runnable MVP for centrifugal force in a rotating frame.

This script demonstrates two things:
1) Vector-form centrifugal force: F_cf = m * (-Omega x (Omega x r)).
2) Rotating spring equilibrium and time-domain verification via explicit Euler.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def centrifugal_acceleration(omega_vec: np.ndarray, position_vec: np.ndarray) -> np.ndarray:
    """Return centrifugal acceleration a_cf = -Omega x (Omega x r)."""
    return -np.cross(omega_vec, np.cross(omega_vec, position_vec))


def centrifugal_force(mass: float, omega_vec: np.ndarray, position_vec: np.ndarray) -> np.ndarray:
    """Return vector centrifugal force."""
    return mass * centrifugal_acceleration(omega_vec, position_vec)


def centrifugal_force_magnitude(mass: float, omega: float, radius: float) -> float:
    """Return scalar centrifugal force magnitude m * omega^2 * r."""
    return mass * omega * omega * radius


def spring_equilibrium_radius(mass: float, k: float, l0: float, omega: float) -> float:
    """Analytical equilibrium radius for rotating radial spring model."""
    denom = k - mass * omega * omega
    if denom <= 0.0:
        raise ValueError(
            "No bounded static equilibrium because k <= m*omega^2. "
            f"Got k={k:.6f}, m*omega^2={mass * omega * omega:.6f}."
        )
    return k * l0 / denom


def effective_potential_gradient(r: float, mass: float, k: float, l0: float, omega: float) -> float:
    """dU_eff/dr = k(r-l0) - m*omega^2*r."""
    return k * (r - l0) - mass * omega * omega * r


def simulate_radial_dynamics(
    mass: float,
    k: float,
    l0: float,
    omega: float,
    r0: float,
    v0: float,
    dt: float,
    steps: int,
    damping: float,
) -> np.ndarray:
    """Simulate m*r_ddot = -k(r-l0) + m*omega^2*r - c*r_dot using explicit Euler.

    Returns an array with columns: [time, radius, radial_velocity].
    """
    r = float(r0)
    v = float(v0)
    history = np.zeros((steps, 3), dtype=np.float64)

    for i in range(steps):
        spring_force = -k * (r - l0)
        centrifugal = mass * omega * omega * r
        damping_force = -damping * v

        a = (spring_force + centrifugal + damping_force) / mass
        v += a * dt
        r += v * dt

        if r < 0.0:
            r = 0.0
            v = 0.0

        history[i, 0] = (i + 1) * dt
        history[i, 1] = r
        history[i, 2] = v

    return history


def run_vector_consistency_demo() -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
    """Validate vector formula against scalar magnitude for several points."""
    mass = 2.0
    omega = 3.0
    omega_vec = np.array([0.0, 0.0, omega], dtype=np.float64)
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )

    rows: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
    for pos in positions:
        f_vec = centrifugal_force(mass, omega_vec, pos)
        radius = float(np.linalg.norm(pos[:2]))
        expected_mag = centrifugal_force_magnitude(mass, omega, radius)
        actual_mag = float(np.linalg.norm(f_vec[:2]))

        radial_xy = pos[:2]
        radial_norm = float(np.linalg.norm(radial_xy))
        if radial_norm > 0.0:
            outward_dot = float(np.dot(f_vec[:2], radial_xy / radial_norm))
            assert outward_dot > 0.0, "Centrifugal force should point outward in xy-plane."

        assert abs(actual_mag - expected_mag) < 1e-10, "Vector/scalar centrifugal force mismatch."
        rows.append((pos, f_vec, expected_mag, actual_mag))

    return rows


def run_rotating_spring_demo() -> Dict[str, np.ndarray]:
    """Compare analytical equilibrium radius and simulated steady-state radius."""
    mass = 1.0
    k = 50.0
    l0 = 0.2
    omegas = np.array([0.0, 1.0, 2.0, 3.0, 3.5], dtype=np.float64)

    dt = 1e-3
    steps = 50000
    damping = 3.0

    analytical = np.zeros_like(omegas)
    simulated = np.zeros_like(omegas)
    abs_error = np.zeros_like(omegas)

    for i, omega in enumerate(omegas):
        r_eq = spring_equilibrium_radius(mass, k, l0, float(omega))
        analytical[i] = r_eq

        grad = effective_potential_gradient(r_eq, mass, k, l0, float(omega))
        assert abs(grad) < 1e-10, "Analytical equilibrium should zero the effective potential gradient."

        history = simulate_radial_dynamics(
            mass=mass,
            k=k,
            l0=l0,
            omega=float(omega),
            r0=l0 * 0.5,
            v0=0.0,
            dt=dt,
            steps=steps,
            damping=damping,
        )
        r_sim = float(history[-1, 1])
        simulated[i] = r_sim
        abs_error[i] = abs(r_sim - r_eq)

    assert np.all(np.diff(analytical) > 0.0), "Equilibrium radius should increase with omega in this regime."
    assert float(np.max(abs_error)) < 2.5e-3, "Simulation did not converge close enough to analytical equilibrium."

    return {
        "omegas": omegas,
        "analytical": analytical,
        "simulated": simulated,
        "abs_error": abs_error,
    }


def main() -> None:
    print("=== Demo A: Vector centrifugal force consistency ===")
    vector_rows = run_vector_consistency_demo()
    for pos, f_vec, expected_mag, actual_mag in vector_rows:
        print(
            "pos={} -> F_cf={} | expected_mag={:.6f}, actual_mag={:.6f}".format(
                np.array2string(pos, precision=3),
                np.array2string(f_vec, precision=3),
                expected_mag,
                actual_mag,
            )
        )

    print("\n=== Demo B: Rotating spring equilibrium (analytical vs simulation) ===")
    report = run_rotating_spring_demo()
    for omega, ra, rs, err in zip(
        report["omegas"],
        report["analytical"],
        report["simulated"],
        report["abs_error"],
    ):
        print(
            "omega={:.2f} rad/s | r_eq={:.6f} m | r_sim={:.6f} m | abs_err={:.6e}".format(
                float(omega), float(ra), float(rs), float(err)
            )
        )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
