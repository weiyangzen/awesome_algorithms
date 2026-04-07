"""Density matrix MVP for a single-qubit quantum system.

This script demonstrates:
1) constructing density matrices from ensembles,
2) checking physical validity (Hermitian, trace=1, PSD),
3) unitary evolution under a Hamiltonian,
4) open-system phase-flip noise via Kraus operators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import expm


Array = np.ndarray


def dagger(matrix: Array) -> Array:
    """Conjugate transpose."""
    return matrix.conj().T


def normalize_state(state: Array) -> Array:
    """Normalize a state vector under l2 norm."""
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(state)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("State norm must be positive and finite.")
    return state / norm


def density_from_ensemble(probabilities: Array, states: list[Array]) -> Array:
    """Build rho = sum_i p_i |psi_i><psi_i| from a classical ensemble."""
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if p.size == 0:
        raise ValueError("Probabilities cannot be empty.")
    if len(states) != p.size:
        raise ValueError("Number of states must match number of probabilities.")
    if np.any(p < 0.0):
        raise ValueError("Probabilities must be non-negative.")
    total = float(np.sum(p))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Probability sum must be positive and finite.")
    p = p / total

    normalized_states = [normalize_state(s) for s in states]
    dim = normalized_states[0].size
    if any(s.size != dim for s in normalized_states):
        raise ValueError("All states in ensemble must have the same dimension.")

    rho = np.zeros((dim, dim), dtype=np.complex128)
    for pi, psi in zip(p, normalized_states):
        rho += pi * np.outer(psi, psi.conj())
    return rho


def is_hermitian(matrix: Array, atol: float = 1e-10) -> bool:
    return np.allclose(matrix, dagger(matrix), atol=atol, rtol=0.0)


def trace_close_to_one(matrix: Array, atol: float = 1e-10) -> bool:
    return np.isclose(np.trace(matrix).real, 1.0, atol=atol, rtol=0.0)


def min_eigenvalue_hermitian(matrix: Array) -> float:
    herm = 0.5 * (matrix + dagger(matrix))
    return float(np.linalg.eigvalsh(herm).min().real)


def validate_density_matrix(rho: Array, atol: float = 1e-10) -> None:
    """Raise error if rho is not a valid density matrix."""
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("Density matrix must be square.")
    if not is_hermitian(rho, atol=atol):
        raise ValueError("Density matrix must be Hermitian.")
    if not trace_close_to_one(rho, atol=atol):
        raise ValueError("Density matrix trace must equal 1.")
    if min_eigenvalue_hermitian(rho) < -1e-9:
        raise ValueError("Density matrix must be positive semidefinite.")


def purity(rho: Array) -> float:
    """Tr(rho^2): 1 for pure states, <1 for mixed states."""
    return float(np.trace(rho @ rho).real)


def von_neumann_entropy(rho: Array, base: float = 2.0) -> float:
    """S(rho) = -Tr(rho log rho), evaluated from eigenvalues."""
    evals = np.linalg.eigvalsh(0.5 * (rho + dagger(rho))).real
    evals = np.clip(evals, 0.0, 1.0)
    mask = evals > 1e-14
    if not np.any(mask):
        return 0.0
    log_base = np.log(base)
    return float(-np.sum(evals[mask] * (np.log(evals[mask]) / log_base)))


def expectation(rho: Array, observable: Array) -> float:
    """Expectation value Tr(rho A), returning real part."""
    return float(np.trace(rho @ observable).real)


def unitary_from_hamiltonian(hamiltonian: Array, time: float, hbar: float = 1.0) -> Array:
    """U(t) = exp(-i H t / hbar)."""
    if hbar <= 0.0:
        raise ValueError("hbar must be positive.")
    return expm(-1j * hamiltonian * (time / hbar))


def evolve_unitary(rho: Array, hamiltonian: Array, time: float, hbar: float = 1.0) -> Array:
    """rho(t) = U rho U^dagger."""
    unitary = unitary_from_hamiltonian(hamiltonian, time=time, hbar=hbar)
    return unitary @ rho @ dagger(unitary)


def phase_flip_channel(rho: Array, probability: float) -> Array:
    """Single-qubit phase-flip channel with Kraus operators.

    E(rho) = (1-p) rho + p Z rho Z
    """
    p = float(probability)
    if p < 0.0 or p > 1.0:
        raise ValueError("Noise probability must be in [0, 1].")
    identity = np.eye(2, dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    k0 = np.sqrt(1.0 - p) * identity
    k1 = np.sqrt(p) * sigma_z
    return k0 @ rho @ dagger(k0) + k1 @ rho @ dagger(k1)


def bloch_vector(rho: Array) -> tuple[float, float, float]:
    """Return single-qubit Bloch components (rx, ry, rz)."""
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return (
        expectation(rho, sigma_x),
        expectation(rho, sigma_y),
        expectation(rho, sigma_z),
    )


@dataclass(frozen=True)
class StepMetrics:
    time: float
    purity_unitary: float
    entropy_unitary_bits: float
    min_eig_unitary: float
    purity_noisy: float
    entropy_noisy_bits: float
    min_eig_noisy: float
    bloch_norm_unitary: float
    bloch_norm_noisy: float


def run_density_matrix_mvp() -> tuple[Array, Array, pd.DataFrame]:
    """Run one reproducible density-matrix experiment."""
    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    ket1 = np.array([0.0, 1.0], dtype=np.complex128)
    ket_plus = normalize_state(ket0 + ket1)

    # Nontrivial mixed initial state from an ensemble.
    rho0 = density_from_ensemble(
        probabilities=np.array([0.55, 0.45], dtype=np.float64),
        states=[ket0, ket_plus],
    )
    validate_density_matrix(rho0)

    # Hamiltonian H = (omega/2) * sigma_x + (delta/2) * sigma_z
    omega = 1.4
    delta = 0.7
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    hamiltonian = 0.5 * omega * sigma_x + 0.5 * delta * sigma_z
    validate_density_matrix(
        density_from_ensemble(
            probabilities=np.array([1.0, 0.0]),
            states=[ket0, ket1],
        )
    )

    times = np.linspace(0.0, 5.0, 8)
    p_noise = 0.22
    rows: list[StepMetrics] = []

    for t in times:
        rho_u = evolve_unitary(rho0, hamiltonian, time=float(t), hbar=1.0)
        rho_u = 0.5 * (rho_u + dagger(rho_u))
        rho_n = phase_flip_channel(rho_u, probability=p_noise)

        validate_density_matrix(rho_u)
        validate_density_matrix(rho_n)

        rux, ruy, ruz = bloch_vector(rho_u)
        rnx, rny, rnz = bloch_vector(rho_n)

        rows.append(
            StepMetrics(
                time=float(t),
                purity_unitary=purity(rho_u),
                entropy_unitary_bits=von_neumann_entropy(rho_u, base=2.0),
                min_eig_unitary=min_eigenvalue_hermitian(rho_u),
                purity_noisy=purity(rho_n),
                entropy_noisy_bits=von_neumann_entropy(rho_n, base=2.0),
                min_eig_noisy=min_eigenvalue_hermitian(rho_n),
                bloch_norm_unitary=float(np.sqrt(rux**2 + ruy**2 + ruz**2)),
                bloch_norm_noisy=float(np.sqrt(rnx**2 + rny**2 + rnz**2)),
            )
        )

    report = pd.DataFrame([r.__dict__ for r in rows])
    return rho0, hamiltonian, report


def run_checks(rho0: Array, report: pd.DataFrame) -> None:
    """Basic physical and numerical sanity checks."""
    p0 = purity(rho0)
    if not (0.5 <= p0 <= 1.0 + 1e-12):
        raise AssertionError("Initial purity must lie in [1/d, 1].")

    # Unitary evolution preserves purity; noisy channel should not increase it.
    pu = report["purity_unitary"].to_numpy()
    pn = report["purity_noisy"].to_numpy()
    if not np.allclose(pu, pu[0], atol=1e-10, rtol=0.0):
        raise AssertionError("Purity under unitary evolution should be constant.")
    if np.any(pn - pu > 1e-10):
        raise AssertionError("Noisy purity should not exceed unitary purity.")

    # Entropy should stay constant under unitary evolution.
    su = report["entropy_unitary_bits"].to_numpy()
    if not np.allclose(su, su[0], atol=1e-10, rtol=0.0):
        raise AssertionError("Unitary entropy should be constant.")

    # PSD guard via minimum eigenvalues.
    if np.any(report["min_eig_unitary"].to_numpy() < -1e-9):
        raise AssertionError("Unitary state lost positive semidefiniteness.")
    if np.any(report["min_eig_noisy"].to_numpy() < -1e-9):
        raise AssertionError("Noisy state lost positive semidefiniteness.")


def main() -> None:
    rho0, hamiltonian, report = run_density_matrix_mvp()
    run_checks(rho0, report)

    np.set_printoptions(precision=5, suppress=True)
    print("Initial density matrix rho0:")
    print(rho0)
    print("\nHamiltonian H:")
    print(hamiltonian)
    print("\nTime-series diagnostics:")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
