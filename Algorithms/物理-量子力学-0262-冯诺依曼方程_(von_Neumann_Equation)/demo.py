"""Minimal runnable MVP for the von Neumann equation."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return [a, b] = ab - ba."""
    return a @ b - b @ a


def validate_density_matrix(rho: np.ndarray, atol: float = 1e-10) -> None:
    """Validate that rho is a physical density matrix."""
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix")

    if not np.allclose(rho, rho.conj().T, atol=atol):
        raise ValueError("rho must be Hermitian")

    if not np.isclose(np.trace(rho), 1.0, atol=atol):
        raise ValueError("trace(rho) must be 1")

    eigvals = np.linalg.eigvalsh(rho)
    if np.min(eigvals) < -atol:
        raise ValueError("rho must be positive semidefinite")


def pack_complex_matrix(rho: np.ndarray) -> np.ndarray:
    """Pack complex matrix into a real vector [Re(rho), Im(rho)]."""
    return np.concatenate([rho.real.ravel(), rho.imag.ravel()])


def unpack_real_vector(y: np.ndarray, n: int) -> np.ndarray:
    """Unpack real vector back to an n x n complex matrix."""
    real = y[: n * n].reshape((n, n))
    imag = y[n * n :].reshape((n, n))
    return real + 1j * imag


def unitary_evolution(
    rho0: np.ndarray,
    hamiltonian: np.ndarray,
    t: float,
    hbar: float = 1.0,
) -> np.ndarray:
    """Closed-form evolution rho(t) = U rho(0) U^† for time-independent H."""
    unitary = expm(-1j * hamiltonian * t / hbar)
    return unitary @ rho0 @ unitary.conj().T


def von_neumann_rhs_real(
    _t: float,
    y: np.ndarray,
    hamiltonian: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Real-valued RHS wrapper for d rho / dt = -(i/hbar)[H, rho]."""
    n = hamiltonian.shape[0]
    rho = unpack_real_vector(y, n)
    drho = -(1j / hbar) * commutator(hamiltonian, rho)
    return pack_complex_matrix(drho)


def integrate_von_neumann(
    rho0: np.ndarray,
    hamiltonian: np.ndarray,
    t_eval: np.ndarray,
    hbar: float = 1.0,
) -> list[np.ndarray]:
    """Numerically integrate the von Neumann equation over t_eval."""
    y0 = pack_complex_matrix(rho0)
    solution = solve_ivp(
        fun=lambda t, y: von_neumann_rhs_real(t, y, hamiltonian, hbar),
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


def density_diagnostics(rho: np.ndarray) -> tuple[float, float, float, float]:
    """Return trace deviation, Hermiticity error, min eigenvalue, and purity."""
    trace_dev = float(abs(np.trace(rho) - 1.0))
    herm_err = float(np.linalg.norm(rho - rho.conj().T, ord="fro"))
    sym_rho = 0.5 * (rho + rho.conj().T)
    min_eig = float(np.min(np.linalg.eigvalsh(sym_rho)).real)
    purity = float(np.real(np.trace(rho @ rho)))
    return trace_dev, herm_err, min_eig, purity


def main() -> None:
    hbar = 1.0

    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    omega = 1.8
    detuning = 0.6
    hamiltonian = 0.5 * omega * sigma_x + 0.5 * detuning * sigma_z

    rho0 = np.array(
        [[0.7, 0.2 + 0.1j], [0.2 - 0.1j, 0.3]],
        dtype=complex,
    )
    validate_density_matrix(rho0)

    t_eval = np.linspace(0.0, 6.0, 9)

    rho_unitary = [unitary_evolution(rho0, hamiltonian, t, hbar=hbar) for t in t_eval]
    rho_ode = integrate_von_neumann(rho0, hamiltonian, t_eval, hbar=hbar)

    max_diff = max(
        float(np.linalg.norm(rho_u - rho_o, ord="fro"))
        for rho_u, rho_o in zip(rho_unitary, rho_ode)
    )

    print("von Neumann equation demo (two-level closed quantum system)")
    print("H = (omega/2) * sigma_x + (detuning/2) * sigma_z")
    print(f"omega={omega:.3f}, detuning={detuning:.3f}, hbar={hbar:.3f}")
    print()
    print("time    trace_dev      herm_err       min_eig        purity")

    for t, rho in zip(t_eval, rho_unitary):
        trace_dev, herm_err, min_eig, purity = density_diagnostics(rho)
        print(
            f"{t:4.1f}  {trace_dev:10.3e}  {herm_err:10.3e}  "
            f"{min_eig:10.6f}  {purity:10.6f}"
        )

    print()
    print(f"max ||rho_unitary - rho_ode||_F = {max_diff:.3e}")
    print("rho(t_final) =")
    print(np.array2string(rho_unitary[-1], precision=6, suppress_small=True))


if __name__ == "__main__":
    main()
