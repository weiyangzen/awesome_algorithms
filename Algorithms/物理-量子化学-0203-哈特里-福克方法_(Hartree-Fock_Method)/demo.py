"""Minimal runnable MVP for the Hartree-Fock method.

This demo implements closed-shell RHF SCF for H2 in a minimal STO-3G-like
2-basis setup using fixed one- and two-electron integrals.
"""

from __future__ import annotations

import numpy as np


def set_eri_with_permutational_symmetry(
    eri: np.ndarray,
    mu: int,
    nu: int,
    lam: int,
    sig: int,
    value: float,
) -> None:
    """Fill an ERI value using chemist-notation 8-fold symmetry."""
    indices = {
        (mu, nu, lam, sig),
        (nu, mu, lam, sig),
        (mu, nu, sig, lam),
        (nu, mu, sig, lam),
        (lam, sig, mu, nu),
        (sig, lam, mu, nu),
        (lam, sig, nu, mu),
        (sig, lam, nu, mu),
    }
    for a, b, c, d in indices:
        eri[a, b, c, d] = value


def build_h2_sto3g_problem() -> dict[str, np.ndarray | float | int]:
    """Return a tiny RHF benchmark problem for H2 at R=1.4 bohr.

    Values are standard teaching integrals for a minimal 2-function basis.
    """
    overlap = np.array(
        [
            [1.0, 0.6593],
            [0.6593, 1.0],
        ],
        dtype=float,
    )

    h_core = np.array(
        [
            [-1.120409, -0.958379],
            [-0.958379, -1.120409],
        ],
        dtype=float,
    )

    eri = np.zeros((2, 2, 2, 2), dtype=float)
    set_eri_with_permutational_symmetry(eri, 0, 0, 0, 0, 0.774605)
    set_eri_with_permutational_symmetry(eri, 0, 0, 0, 1, 0.444108)
    set_eri_with_permutational_symmetry(eri, 0, 0, 1, 1, 0.569700)
    set_eri_with_permutational_symmetry(eri, 0, 1, 0, 1, 0.297029)
    set_eri_with_permutational_symmetry(eri, 0, 1, 1, 1, 0.444108)
    set_eri_with_permutational_symmetry(eri, 1, 1, 1, 1, 0.774605)

    return {
        "S": overlap,
        "H": h_core,
        "ERI": eri,
        "E_nuc": 0.714286,  # 1 / R with R=1.4 bohr
        "n_electrons": 2,
    }


def symmetric_orthogonalization(overlap: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return X = S^{-1/2} for symmetric orthogonalization."""
    evals, evecs = np.linalg.eigh(overlap)
    if float(np.min(evals)) < tol:
        raise ValueError("Overlap matrix is singular or ill-conditioned")
    inv_sqrt = np.diag(evals ** -0.5)
    return evecs @ inv_sqrt @ evecs.T


def build_fock(density: np.ndarray, h_core: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Build RHF Fock matrix F = H + J - 1/2 K."""
    coulomb = np.einsum("ls,mnls->mn", density, eri, optimize=True)
    exchange = np.einsum("ls,mlns->mn", density, eri, optimize=True)
    return h_core + coulomb - 0.5 * exchange


def solve_roothaan(fock: np.ndarray, overlap: np.ndarray, n_occ: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve FC = SCE and return orbital energies, coefficients, and density."""
    x = symmetric_orthogonalization(overlap)
    f_orth = x.T @ fock @ x
    eps, c_tilde = np.linalg.eigh(f_orth)
    coeff = x @ c_tilde
    c_occ = coeff[:, :n_occ]
    density = 2.0 * (c_occ @ c_occ.T)
    return eps, coeff, density


def electronic_energy(density: np.ndarray, h_core: np.ndarray, fock: np.ndarray) -> float:
    """Return RHF electronic energy in Hartree."""
    return 0.5 * float(np.sum(density * (h_core + fock)))


def scf_rhf(
    overlap: np.ndarray,
    h_core: np.ndarray,
    eri: np.ndarray,
    n_electrons: int,
    e_nuc: float,
    max_iter: int = 50,
    e_tol: float = 1e-10,
    p_tol: float = 1e-8,
) -> dict[str, object]:
    """Run a minimal RHF self-consistent field loop."""
    nbf = overlap.shape[0]
    if n_electrons % 2 != 0:
        raise ValueError("This MVP only supports closed-shell (even-electron) RHF")

    n_occ = n_electrons // 2
    density = np.zeros((nbf, nbf), dtype=float)
    e_prev = np.inf
    history: list[tuple[int, float, float, float]] = []

    converged = False
    eps = np.zeros(nbf, dtype=float)
    coeff = np.eye(nbf)
    fock = h_core.copy()

    for it in range(1, max_iter + 1):
        fock = build_fock(density, h_core, eri)
        eps, coeff, density_new = solve_roothaan(fock, overlap, n_occ=n_occ)

        fock_new = build_fock(density_new, h_core, eri)
        e_elec = electronic_energy(density_new, h_core, fock_new)
        e_total = e_elec + e_nuc

        d_energy = abs(e_total - e_prev)
        d_density = float(np.linalg.norm(density_new - density, ord="fro"))
        history.append((it, e_total, d_energy, d_density))

        density = density_new
        e_prev = e_total

        if d_energy < e_tol and d_density < p_tol:
            converged = True
            fock = fock_new
            break

    return {
        "converged": converged,
        "iterations": history,
        "E_total": e_prev,
        "E_elec": e_prev - e_nuc,
        "eps": eps,
        "C": coeff,
        "P": density,
        "F": fock,
    }


def commutator_residual(fock: np.ndarray, density: np.ndarray, overlap: np.ndarray) -> float:
    """Return ||F P S - S P F||_F as an SCF stationary-condition diagnostic."""
    residual = fock @ density @ overlap - overlap @ density @ fock
    return float(np.linalg.norm(residual, ord="fro"))


def main() -> None:
    problem = build_h2_sto3g_problem()

    overlap = problem["S"]
    h_core = problem["H"]
    eri = problem["ERI"]
    e_nuc = float(problem["E_nuc"])
    n_electrons = int(problem["n_electrons"])

    result = scf_rhf(
        overlap=overlap,
        h_core=h_core,
        eri=eri,
        n_electrons=n_electrons,
        e_nuc=e_nuc,
        max_iter=50,
        e_tol=1e-12,
        p_tol=1e-10,
    )

    converged = bool(result["converged"])
    iterations = result["iterations"]
    e_total = float(result["E_total"])
    e_elec = float(result["E_elec"])
    eps = np.asarray(result["eps"], dtype=float)
    coeff = np.asarray(result["C"], dtype=float)
    density = np.asarray(result["P"], dtype=float)
    fock = np.asarray(result["F"], dtype=float)

    electrons_from_density = float(np.trace(density @ overlap))
    ortho_error = float(np.linalg.norm(coeff.T @ overlap @ coeff - np.eye(coeff.shape[1]), ord="fro"))
    stationarity = commutator_residual(fock, density, overlap)

    print("Hartree-Fock RHF demo: H2 / STO-3G(minimal) / R=1.4 bohr")
    print("iter    E_total(Ha)         dE               dP")
    for it, e_i, de_i, dp_i in iterations:
        print(f"{it:3d}   {e_i: .12f}   {de_i: .3e}   {dp_i: .3e}")

    print()
    print(f"Converged: {converged}")
    print(f"Electronic energy (Ha): {e_elec:.12f}")
    print(f"Total energy (Ha):      {e_total:.12f}")
    print(f"MO energies (Ha):       {np.array2string(eps, precision=9, suppress_small=True)}")
    print(f"Tr(P S) electrons:      {electrons_from_density:.12f}")
    print(f"Orthonormality error:   {ortho_error:.3e}")
    print(f"SCF commutator residual:{stationarity:.3e}")

    # Deterministic sanity checks for this educational benchmark.
    assert converged, "SCF did not converge"
    assert -1.2 < e_total < -1.0, "Total energy out of expected RHF range"
    assert abs(electrons_from_density - n_electrons) < 1e-8, "Electron count mismatch"
    assert ortho_error < 1e-8, "MO coefficients violate C^T S C = I"

    print("All checks passed.")


if __name__ == "__main__":
    main()
