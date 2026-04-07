"""Minimal runnable MVP for the Quantum Zeno Effect.

Model:
- Two-level system with Hamiltonian H = (omega / 2) * sigma_x, setting hbar = 1.
- Initial state |0>.
- Total evolution time T is divided into N equal intervals.
- At the end of each interval, perform projective measurement onto |0><0|.

Goal:
Show that the survival probability increases with measurement frequency N.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]

SIGMA_X: ComplexArray = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def hamiltonian(omega: float) -> ComplexArray:
    """Return H = (omega / 2) * sigma_x."""
    return 0.5 * omega * SIGMA_X


def unitary_from_hermitian(h: ComplexArray, dt: float) -> ComplexArray:
    """Compute U(dt)=exp(-i h dt) via eigen-decomposition of Hermitian h."""
    evals, evecs = np.linalg.eigh(h)
    phase = np.exp(-1j * evals * dt)
    return (evecs * phase) @ evecs.conj().T


def free_survival_probability(omega: float, total_time: float) -> float:
    """Survival probability without intermediate measurements."""
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    proj0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    u_total = unitary_from_hermitian(hamiltonian(omega), total_time)
    psi_t = u_total @ psi0
    amp_sq = np.vdot(psi_t, proj0 @ psi_t).real
    return float(np.clip(amp_sq, 0.0, 1.0))


def zeno_survival_simulation(omega: float, total_time: float, n_measure: int) -> float:
    """Numerically simulate repeated projective measurements onto |0><0|."""
    if n_measure <= 0:
        raise ValueError("n_measure must be positive")

    psi = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    proj0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)

    dt = total_time / n_measure
    u_step = unitary_from_hermitian(hamiltonian(omega), dt)

    survival = 1.0
    for _ in range(n_measure):
        psi = u_step @ psi
        psi_keep = proj0 @ psi
        p_step = float(np.clip(np.vdot(psi_keep, psi_keep).real, 0.0, 1.0))
        survival *= p_step
        if p_step < 1e-15:
            return 0.0
        psi = psi_keep / np.sqrt(p_step)

    return float(np.clip(survival, 0.0, 1.0))


def zeno_survival_analytic(omega: float, total_time: float, n_measure: int) -> float:
    """Analytic result: [cos^2(omega*T/(2N))]^N, computed stably."""
    if n_measure <= 0:
        raise ValueError("n_measure must be positive")

    x = omega * total_time / (2.0 * n_measure)
    cos_sq = float(np.cos(x) ** 2)
    if cos_sq <= 0.0:
        return 0.0
    log_p = n_measure * np.log(cos_sq)
    return float(np.exp(log_p))


def zeno_short_time_approx(omega: float, total_time: float, n_measure: int) -> float:
    """Large-N approximation: exp(-(omega^2 * T^2) / (4N))."""
    return float(np.exp(-(omega * omega * total_time * total_time) / (4.0 * n_measure)))


def main() -> None:
    omega = 1.0
    total_time = np.pi
    measurement_grid = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    p_free = free_survival_probability(omega, total_time)

    print("Quantum Zeno Effect demo (two-level Rabi model)")
    print(f"omega = {omega:.3f}, T = {total_time:.6f}")
    print(f"free evolution survival P_free(T) = {p_free:.12f}")
    print()
    print(
        "{:<8s} {:>16s} {:>16s} {:>12s} {:>18s}".format(
            "N", "P_sim", "P_analytic", "abs_err", "short_time_approx"
        )
    )

    max_abs_err = 0.0
    for n_measure in measurement_grid:
        p_sim = zeno_survival_simulation(omega, total_time, n_measure)
        p_ana = zeno_survival_analytic(omega, total_time, n_measure)
        p_approx = zeno_short_time_approx(omega, total_time, n_measure)
        abs_err = abs(p_sim - p_ana)
        max_abs_err = max(max_abs_err, abs_err)

        print(
            f"{n_measure:<8d} {p_sim:16.12f} {p_ana:16.12f} "
            f"{abs_err:12.3e} {p_approx:18.12f}"
        )

    print()
    print(f"max_abs_err(sim vs analytic) = {max_abs_err:.3e}")


if __name__ == "__main__":
    main()
