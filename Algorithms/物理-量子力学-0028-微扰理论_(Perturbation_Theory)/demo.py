"""Minimal runnable MVP for non-degenerate stationary perturbation theory."""

from __future__ import annotations

import numpy as np


def build_problem() -> tuple[np.ndarray, np.ndarray]:
    """Build a small non-degenerate model H = H0 + lambda * V."""
    h0 = np.diag([0.0, 1.0, 2.3, 3.7]).astype(float)
    v = np.array(
        [
            [0.30, 0.12, -0.08, 0.00],
            [0.12, -0.15, 0.10, 0.06],
            [-0.08, 0.10, 0.20, -0.09],
            [0.00, 0.06, -0.09, -0.05],
        ],
        dtype=float,
    )
    return h0, v


def assert_hermitian(matrix: np.ndarray, atol: float = 1e-12) -> None:
    """Validate Hermitian property."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not np.allclose(matrix, matrix.conj().T, atol=atol):
        raise ValueError("matrix must be Hermitian")


def unperturbed_spectrum(h0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return unperturbed eigenvalues/eigenvectors and minimal level spacing."""
    e0, u0 = np.linalg.eigh(h0)
    if e0.size < 2:
        min_gap = np.inf
    else:
        min_gap = float(np.min(np.diff(e0)))
    return e0, u0, min_gap


def first_order_energy_shift(v0: np.ndarray) -> np.ndarray:
    """First-order energy coefficient dE^(1)."""
    return np.real(np.diag(v0))


def second_order_energy_shift(
    e0: np.ndarray,
    v0: np.ndarray,
    atol: float = 1e-12,
) -> np.ndarray:
    """Second-order energy coefficient dE^(2) for non-degenerate case."""
    n = e0.size
    d2 = np.zeros(n, dtype=float)
    for i in range(n):
        for m in range(n):
            if i == m:
                continue
            denom = float(e0[i] - e0[m])
            if abs(denom) < atol:
                raise ValueError(
                    "near-degenerate denominator encountered; "
                    "use degenerate perturbation theory"
                )
            d2[i] += float(abs(v0[m, i]) ** 2 / denom)
    return d2


def perturbative_energies(
    e0: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    lambda_: float,
    order: int,
) -> np.ndarray:
    """Compute first- or second-order perturbative energies."""
    if order == 1:
        return e0 + lambda_ * d1
    if order == 2:
        return e0 + lambda_ * d1 + (lambda_ ** 2) * d2
    raise ValueError("order must be 1 or 2")


def exact_eigensystem(
    h0: np.ndarray,
    v: np.ndarray,
    lambda_: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize H(lambda) exactly."""
    h = h0 + lambda_ * v
    e_exact, u_exact = np.linalg.eigh(h)
    return e_exact, u_exact


def first_order_state_vector(
    state_index: int,
    e0: np.ndarray,
    v0: np.ndarray,
    lambda_: float,
    atol: float = 1e-12,
) -> np.ndarray:
    """Approximate |psi_n> ~= |n0> + lambda |n1>, normalized."""
    n = e0.size
    psi = np.zeros(n, dtype=complex)
    psi[state_index] = 1.0

    for m in range(n):
        if m == state_index:
            continue
        denom = float(e0[state_index] - e0[m])
        if abs(denom) < atol:
            raise ValueError(
                "near-degenerate denominator in state correction; "
                "use degenerate perturbation theory"
            )
        psi[m] += lambda_ * v0[m, state_index] / denom

    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("invalid zero norm in state correction")
    return psi / norm


def state_fidelity(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """Return fidelity |<a|b>|^2."""
    return float(abs(np.vdot(psi_a, psi_b)) ** 2)


def run_demo() -> None:
    """Execute perturbation-theory MVP and print diagnostics."""
    h0, v = build_problem()
    assert_hermitian(h0)
    assert_hermitian(v)

    e0, u0, min_gap = unperturbed_spectrum(h0)
    if min_gap < 1e-6:
        raise ValueError(
            f"unperturbed spectrum is (near-)degenerate, min_gap={min_gap:.3e}"
        )

    v0 = u0.conj().T @ v @ u0
    d1 = first_order_energy_shift(v0)
    d2 = second_order_energy_shift(e0, v0)

    lambdas = np.array([0.00, 0.02, 0.05, 0.08, 0.12], dtype=float)

    print("Perturbation Theory demo (non-degenerate, time-independent)")
    print(f"dimension={h0.shape[0]}, min unperturbed level gap={min_gap:.3f}")
    print()
    print("lambda   max|E1-Eexact|   max|E2-Eexact|")

    for lambda_ in lambdas:
        e_exact, _ = exact_eigensystem(h0, v, float(lambda_))
        e_1st = perturbative_energies(e0, d1, d2, float(lambda_), order=1)
        e_2nd = perturbative_energies(e0, d1, d2, float(lambda_), order=2)

        err_1st = float(np.max(np.abs(e_1st - e_exact)))
        err_2nd = float(np.max(np.abs(e_2nd - e_exact)))
        print(f"{lambda_:>5.2f}    {err_1st:>13.6e}   {err_2nd:>13.6e}")

    lambda_state = 0.08
    e_exact, u_exact = exact_eigensystem(h0, v, lambda_state)
    coeff_exact = u0.conj().T @ u_exact

    print()
    print(f"State fidelity at lambda={lambda_state:.2f} (1st-order state vs exact)")
    print("state   fidelity")

    for n in range(e0.size):
        psi_approx = first_order_state_vector(n, e0, v0, lambda_state)
        psi_exact = coeff_exact[:, n]
        fidelity = state_fidelity(psi_approx, psi_exact)
        print(f"{n:>2d}      {fidelity:.8f}")

    print()
    print("Unperturbed energies E0:")
    print(np.array2string(e0, precision=6, suppress_small=True))
    print("Exact energies at lambda=0.12:")
    print(np.array2string(exact_eigensystem(h0, v, 0.12)[0], precision=6, suppress_small=True))


if __name__ == "__main__":
    run_demo()
