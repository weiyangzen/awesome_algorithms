"""Minimal runnable MVP for Berry phase of a spin-1/2 system.

The demo computes the geometric phase on a closed loop in parameter space:
- Hamiltonian: H(\\theta, \\phi) = - n(\\theta, \\phi) · sigma
- Loop: fixed polar angle \\theta, azimuth \\phi in [0, 2\\pi)
- Discrete Berry phase: gamma = -arg(prod_j <u_j | u_{j+1}>)

This is gauge-invariant and suitable for a compact numerical MVP.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]

SIGMA_X: ComplexArray = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_Y: ComplexArray = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_Z: ComplexArray = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def wrap_to_pi(angle: float) -> float:
    """Map an angle to (-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def bloch_vector(theta: float, phi: float) -> RealArray:
    """Unit vector on the Bloch sphere."""
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=np.float64,
    )


def hamiltonian(theta: float, phi: float) -> ComplexArray:
    """2x2 Hermitian Hamiltonian H = - n · sigma."""
    n = bloch_vector(theta, phi)
    return -(n[0] * SIGMA_X + n[1] * SIGMA_Y + n[2] * SIGMA_Z)


def ground_state(theta: float, phi: float) -> ComplexArray:
    """Ground-state eigenvector of H(theta, phi)."""
    evals, evecs = np.linalg.eigh(hamiltonian(theta, phi))
    index = int(np.argmin(evals))
    return evecs[:, index]


def sample_ground_states(theta: float, n_steps: int) -> ComplexArray:
    """Sample ground states on a closed phi-loop with uniform discretization."""
    phis = np.linspace(0.0, 2.0 * np.pi, n_steps, endpoint=False)
    states = np.array([ground_state(theta, phi) for phi in phis], dtype=np.complex128)
    return states


def berry_phase_discrete(states: ComplexArray) -> float:
    """Gauge-invariant discrete Berry phase for a closed loop."""
    shifted = np.roll(states, shift=-1, axis=0)
    overlaps = np.sum(np.conjugate(states) * shifted, axis=1)

    # Keep only phase factors to improve numerical robustness.
    denom = np.maximum(np.abs(overlaps), 1e-15)
    phase_factors = overlaps / denom

    gamma = -np.angle(np.prod(phase_factors))
    return wrap_to_pi(float(gamma))


def apply_random_gauge(states: ComplexArray, rng: np.random.Generator) -> ComplexArray:
    """Multiply each state by an independent random U(1) phase."""
    phases = rng.uniform(0.0, 2.0 * np.pi, size=states.shape[0])
    return states * np.exp(1j * phases)[:, None]


def analytic_berry_phase(theta: float) -> float:
    """Analytical result for ground state of H = -n·sigma on fixed-theta loop."""
    solid_angle = 2.0 * np.pi * (1.0 - np.cos(theta))
    gamma = -0.5 * solid_angle
    return wrap_to_pi(float(gamma))


def main() -> None:
    theta_deg = 60.0
    theta = np.deg2rad(theta_deg)
    step_grid = [8, 16, 32, 64, 128, 256, 512]

    analytic = analytic_berry_phase(theta)
    rng = np.random.default_rng(20260407)

    print("Berry phase demo (spin-1/2, closed azimuth loop)")
    print(f"theta = {theta_deg:.1f} deg")
    print(f"analytical gamma = {analytic:+.12f} rad")
    print()
    print(
        "{:<8s} {:>18s} {:>18s} {:>18s}".format(
            "N", "gamma_numeric(rad)", "error(rad)", "gauge_diff(rad)"
        )
    )

    max_abs_gauge_diff = 0.0
    for n_steps in step_grid:
        states = sample_ground_states(theta, n_steps)
        gamma_numeric = berry_phase_discrete(states)

        gauged_states = apply_random_gauge(states, rng)
        gamma_gauged = berry_phase_discrete(gauged_states)

        error = wrap_to_pi(gamma_numeric - analytic)
        gauge_diff = wrap_to_pi(gamma_numeric - gamma_gauged)
        max_abs_gauge_diff = max(max_abs_gauge_diff, abs(gauge_diff))

        print(
            f"{n_steps:<8d} {gamma_numeric:+18.12f} {error:+18.12f} {gauge_diff:+18.12e}"
        )

    print()
    print(f"max |gauge_diff| = {max_abs_gauge_diff:.3e} rad")


if __name__ == "__main__":
    main()
