"""Minimal runnable MVP for the Schrödinger equation."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def build_hamiltonian_1d(
    x: np.ndarray,
    mass: float,
    hbar: float,
    potential_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Build 1D finite-difference Hamiltonian H = -(hbar^2/2m) d2/dx2 + V(x)."""
    if x.ndim != 1 or x.size < 3:
        raise ValueError("x must be a 1D grid with at least 3 points")

    dx = float(x[1] - x[0])
    if not np.allclose(np.diff(x), dx):
        raise ValueError("x grid must be uniform")

    n = x.size
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    laplacian = (
        np.diag(main)
        + np.diag(off, k=1)
        + np.diag(off, k=-1)
    ) / (dx * dx)

    kinetic = -(hbar * hbar) / (2.0 * mass) * laplacian
    potential = np.diag(potential_fn(x).astype(float))
    return kinetic + potential


def validate_hamiltonian(hamiltonian: np.ndarray, atol: float = 1e-10) -> None:
    """Validate Hamiltonian structure for closed-system evolution."""
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be a square matrix")

    if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=atol):
        raise ValueError("hamiltonian must be Hermitian")


def normalize_state(psi: np.ndarray, dx: float) -> np.ndarray:
    """Normalize wavefunction so that sum |psi|^2 dx = 1."""
    norm = float(np.sqrt(np.sum(np.abs(psi) ** 2) * dx))
    if norm <= 0.0:
        raise ValueError("state norm must be positive")
    return psi / norm


def pack_complex_vector(psi: np.ndarray) -> np.ndarray:
    """Pack complex vector into real vector [Re(psi), Im(psi)]."""
    return np.concatenate([psi.real, psi.imag])


def unpack_real_vector(y: np.ndarray, n: int) -> np.ndarray:
    """Unpack [Re(psi), Im(psi)] back to complex vector."""
    real = y[:n]
    imag = y[n:]
    return real + 1j * imag


def unitary_evolution(
    psi0: np.ndarray,
    hamiltonian: np.ndarray,
    t: float,
    hbar: float = 1.0,
) -> np.ndarray:
    """Compute psi(t) = exp(-i H t / hbar) psi(0) for time-independent H."""
    propagator = expm(-1j * hamiltonian * t / hbar)
    return propagator @ psi0


def schrodinger_rhs_real(
    _t: float,
    y: np.ndarray,
    hamiltonian: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Real-valued RHS wrapper for d psi / dt = -(i/hbar) H psi."""
    n = hamiltonian.shape[0]
    psi = unpack_real_vector(y, n)
    dpsi = -(1j / hbar) * (hamiltonian @ psi)
    return pack_complex_vector(dpsi)


def integrate_schrodinger(
    psi0: np.ndarray,
    hamiltonian: np.ndarray,
    t_eval: np.ndarray,
    hbar: float = 1.0,
) -> list[np.ndarray]:
    """Integrate Schrödinger ODE on a given time grid."""
    y0 = pack_complex_vector(psi0)
    solution = solve_ivp(
        fun=lambda t, y: schrodinger_rhs_real(t, y, hamiltonian, hbar),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    n = hamiltonian.shape[0]
    return [unpack_real_vector(solution.y[:, i], n) for i in range(solution.y.shape[1])]


def wave_diagnostics(
    psi: np.ndarray,
    hamiltonian: np.ndarray,
    x: np.ndarray,
    dx: float,
) -> tuple[float, float, float]:
    """Return norm, <x>, and <H>."""
    prob = np.abs(psi) ** 2
    norm = float(np.sum(prob) * dx)

    if norm <= 0.0:
        raise ValueError("wavefunction norm became non-positive")

    x_mean = float(np.sum(prob * x) * dx / norm)
    energy = float(np.real(np.vdot(psi, hamiltonian @ psi) * dx / norm))
    return norm, x_mean, energy


def l2_error(psi_a: np.ndarray, psi_b: np.ndarray, dx: float) -> float:
    """Compute L2 error under grid quadrature."""
    return float(np.sqrt(np.sum(np.abs(psi_a - psi_b) ** 2) * dx))


def main() -> None:
    hbar = 1.0
    mass = 1.0
    omega = 1.0

    x = np.linspace(-8.0, 8.0, 120)
    dx = float(x[1] - x[0])

    potential_fn = lambda arr: 0.5 * mass * (omega**2) * (arr**2)
    hamiltonian = build_hamiltonian_1d(x, mass, hbar, potential_fn)
    validate_hamiltonian(hamiltonian)

    x0 = -1.5
    sigma = 0.8
    k0 = 1.2
    psi0 = np.exp(-((x - x0) ** 2) / (2.0 * sigma * sigma)) * np.exp(1j * k0 * x)
    psi0 = normalize_state(psi0, dx)

    t_eval = np.linspace(0.0, 2.0, 9)

    psi_unitary = [unitary_evolution(psi0, hamiltonian, t, hbar=hbar) for t in t_eval]
    psi_ode = integrate_schrodinger(psi0, hamiltonian, t_eval, hbar=hbar)

    point_errors = [l2_error(ua, ob, dx) for ua, ob in zip(psi_unitary, psi_ode)]
    max_diff = max(point_errors)

    ode_norm_dev = max(abs(wave_diagnostics(psi, hamiltonian, x, dx)[0] - 1.0) for psi in psi_ode)

    print("Schrodinger equation demo (1D harmonic potential)")
    print("H = -(hbar^2/2m) d2/dx2 + 0.5*m*omega^2*x^2")
    print(f"grid_points={x.size}, dx={dx:.4f}, mass={mass:.3f}, omega={omega:.3f}, hbar={hbar:.3f}")
    print()
    print("time    norm(unitary)    <x>(unitary)    E(unitary)    L2(unitary, ode)")

    for t, psi_u, err in zip(t_eval, psi_unitary, point_errors):
        norm_u, x_mean_u, energy_u = wave_diagnostics(psi_u, hamiltonian, x, dx)
        print(f"{t:4.2f}   {norm_u:12.6f}   {x_mean_u:12.6f}   {energy_u:10.6f}   {err:14.3e}")

    print()
    print(f"max L2(unitary, ode) = {max_diff:.3e}")
    print(f"max |norm(ode)-1|    = {ode_norm_dev:.3e}")
    print("psi(x, t_final) first 6 values:")
    print(np.array2string(psi_unitary[-1][:6], precision=5, suppress_small=True))


if __name__ == "__main__":
    main()
