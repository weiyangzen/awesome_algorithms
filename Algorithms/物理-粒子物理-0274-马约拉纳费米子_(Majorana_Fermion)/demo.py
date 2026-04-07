"""Minimal runnable MVP for Majorana zero modes in a Kitaev chain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KitaevParams:
    """Parameter set for the open-boundary Kitaev chain."""

    n_sites: int
    hopping: float
    pairing: float
    chemical_potential: float
    zero_tol: float = 1e-2
    edge_sites: int = 3


def build_kitaev_bdg(params: KitaevParams) -> np.ndarray:
    """Build BdG Hamiltonian H = [[A, B], [-B, -A]] for a 1D Kitaev chain."""
    if params.n_sites < 3:
        raise ValueError("n_sites must be >= 3")
    if abs(params.hopping) < 1e-12:
        raise ValueError("hopping must be non-zero")
    if abs(params.pairing) < 1e-12:
        raise ValueError("pairing must be non-zero")

    n = params.n_sites
    a_block = np.zeros((n, n), dtype=float)
    b_block = np.zeros((n, n), dtype=float)

    np.fill_diagonal(a_block, -params.chemical_potential)

    for i in range(n - 1):
        a_block[i, i + 1] = -params.hopping
        a_block[i + 1, i] = -params.hopping

        # Real p-wave pairing -> antisymmetric block B.
        b_block[i, i + 1] = params.pairing
        b_block[i + 1, i] = -params.pairing

    return np.block(
        [
            [a_block, b_block],
            [-b_block, -a_block],
        ]
    )


def validate_bdg_hamiltonian(hamiltonian: np.ndarray, atol: float = 1e-10) -> None:
    """Validate matrix shape and Hermiticity."""
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("Hamiltonian must be a square matrix")

    dim = hamiltonian.shape[0]
    if dim % 2 != 0:
        raise ValueError("BdG Hamiltonian dimension must be even")

    if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=atol):
        raise ValueError("BdG Hamiltonian must be Hermitian")


def particle_hole_symmetry_error(hamiltonian: np.ndarray) -> float:
    """Return relative Frobenius residual of tau_x H* tau_x + H."""
    dim = hamiltonian.shape[0]
    n = dim // 2

    identity = np.eye(n, dtype=float)
    zeros = np.zeros((n, n), dtype=float)
    tau_x = np.block([[zeros, identity], [identity, zeros]])

    residual = tau_x @ hamiltonian.conj() @ tau_x + hamiltonian
    numerator = np.linalg.norm(residual, ord="fro")
    denominator = max(np.linalg.norm(hamiltonian, ord="fro"), 1e-12)
    return float(numerator / denominator)


def diagonalize_bdg(hamiltonian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize Hermitian BdG Hamiltonian with dense symmetric eigensolver."""
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return eigenvalues, eigenvectors


def edge_localization_score(
    eigenvector: np.ndarray,
    n_sites: int,
    edge_sites: int,
) -> float:
    """Compute probability weight near both edges, including particle and hole parts."""
    if edge_sites < 1:
        raise ValueError("edge_sites must be >= 1")
    if edge_sites > n_sites:
        raise ValueError("edge_sites cannot exceed n_sites")

    particle_prob = np.abs(eigenvector[:n_sites]) ** 2
    hole_prob = np.abs(eigenvector[n_sites:]) ** 2
    local_density = particle_prob + hole_prob

    total = float(np.sum(local_density))
    if total <= 0.0:
        raise ValueError("invalid eigenvector normalization")

    left = float(np.sum(local_density[:edge_sites]))
    right = float(np.sum(local_density[-edge_sites:]))
    return (left + right) / total


def analyze_phase(label: str, params: KitaevParams) -> dict[str, Any]:
    """Analyze one parameter point and summarize low-energy Majorana signatures."""
    hamiltonian = build_kitaev_bdg(params)
    validate_bdg_hamiltonian(hamiltonian)

    phs_residual = particle_hole_symmetry_error(hamiltonian)
    eigenvalues, eigenvectors = diagonalize_bdg(hamiltonian)

    order = np.argsort(np.abs(eigenvalues))
    lowest_indices = order[:8]
    lowest_energies = eigenvalues[lowest_indices]

    zero_mode_indices = np.where(np.abs(eigenvalues) < params.zero_tol)[0]
    edge_scores = [
        edge_localization_score(eigenvectors[:, idx], params.n_sites, params.edge_sites)
        for idx in order[:2]
    ]

    return {
        "label": label,
        "params": params,
        "phs_residual": phs_residual,
        "min_abs_energy": float(np.min(np.abs(eigenvalues))),
        "zero_mode_count": int(zero_mode_indices.size),
        "lowest_energies": lowest_energies,
        "edge_scores": edge_scores,
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Pretty-print one phase summary."""
    params: KitaevParams = summary["params"]

    print(f"[{summary['label']}]")
    print(
        "n_sites={n}, t={t:.3f}, Delta={d:.3f}, mu={mu:.3f}, "
        "zero_tol={tol:.1e}, edge_sites={e}".format(
            n=params.n_sites,
            t=params.hopping,
            d=params.pairing,
            mu=params.chemical_potential,
            tol=params.zero_tol,
            e=params.edge_sites,
        )
    )
    print(f"particle-hole residual    = {summary['phs_residual']:.3e}")
    print(f"min |E|                  = {summary['min_abs_energy']:.3e}")
    print(f"near-zero mode count     = {summary['zero_mode_count']}")
    print(
        "lowest |E| candidates    = "
        + np.array2string(summary["lowest_energies"], precision=6, suppress_small=False)
    )
    print(
        "edge localization scores = "
        + np.array2string(np.asarray(summary["edge_scores"]), precision=6, suppress_small=False)
    )
    print()


def main() -> None:
    topological = KitaevParams(
        n_sites=60,
        hopping=1.0,
        pairing=0.8,
        chemical_potential=0.0,
        zero_tol=1e-2,
        edge_sites=3,
    )
    trivial = KitaevParams(
        n_sites=60,
        hopping=1.0,
        pairing=0.8,
        chemical_potential=3.0,
        zero_tol=1e-2,
        edge_sites=3,
    )

    topo_summary = analyze_phase("Topological regime |mu| < 2|t|", topological)
    trivial_summary = analyze_phase("Trivial regime |mu| > 2|t|", trivial)

    print("Majorana fermion MVP via Kitaev chain BdG diagonalization")
    print("Model: open chain, real p-wave pairing, dense Hermitian eigensolver")
    print()

    print_summary(topo_summary)
    print_summary(trivial_summary)

    if topo_summary["zero_mode_count"] < 2:
        raise RuntimeError("Expected at least two near-zero modes in topological regime")
    if topo_summary["edge_scores"][0] < 0.5:
        raise RuntimeError("Expected strong edge localization for topological near-zero mode")
    if trivial_summary["zero_mode_count"] != 0:
        raise RuntimeError("Expected no near-zero mode in trivial regime")

    print("MVP checks passed: topology contrast and edge localization are both observed.")


if __name__ == "__main__":
    main()
