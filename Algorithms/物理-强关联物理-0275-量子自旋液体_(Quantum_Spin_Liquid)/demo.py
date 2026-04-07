"""Minimal runnable MVP for Quantum Spin Liquid via toric code ED.

The script builds a 2D toric-code Hamiltonian on a small torus (Lx=Ly=2),
performs exact diagonalization, and reports topological signatures:
- fourfold ground-state manifold,
- finite excitation gap,
- Wilson-loop sector labels.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity, kron


ComplexArray = np.ndarray


@dataclass
class ToricCodeConfig:
    """Configuration for the minimal toric-code spin-liquid demo."""

    L: int = 2
    Je: float = 1.0
    Jm: float = 1.0
    degeneracy_tol: float = 1e-8


@dataclass
class ToricCodeResult:
    """Container for solved spectrum and diagnostics."""

    energies: ComplexArray
    eigenvectors: ComplexArray
    ground_degeneracy: int
    gap: float
    As_mean: float
    Bp_mean: float
    sector_labels: list[tuple[float, float, float]]


def h_edge(x: int, y: int, L: int) -> int:
    """Horizontal edge index at lattice coordinate (x, y)."""

    return y * L + x


def v_edge(x: int, y: int, L: int) -> int:
    """Vertical edge index at lattice coordinate (x, y)."""

    return L * L + y * L + x


def star_edges(x: int, y: int, L: int) -> list[int]:
    """Edges touching vertex (x, y) on a periodic LxL torus."""

    return [
        h_edge(x, y, L),
        h_edge((x - 1) % L, y, L),
        v_edge(x, y, L),
        v_edge(x, (y - 1) % L, L),
    ]


def plaquette_edges(x: int, y: int, L: int) -> list[int]:
    """Boundary edges of plaquette (x, y) on a periodic LxL torus."""

    return [
        h_edge(x, y, L),
        v_edge((x + 1) % L, y, L),
        h_edge(x, (y + 1) % L, L),
        v_edge(x, y, L),
    ]


def kron_chain(ops: list[csr_matrix]) -> csr_matrix:
    """Kronecker product of a sequence of sparse operators."""

    out = ops[0]
    for op in ops[1:]:
        out = kron(out, op, format="csr")
    return out


def build_pauli_tables(n_qubits: int) -> tuple[list[csr_matrix], list[csr_matrix]]:
    """Build full-space X_i and Z_i sparse operators for each qubit i."""

    sx = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128))
    sz = csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128))
    id2 = identity(2, format="csr", dtype=np.complex128)

    x_ops: list[csr_matrix] = []
    z_ops: list[csr_matrix] = []

    for i in range(n_qubits):
        factors_x = [id2 for _ in range(n_qubits)]
        factors_z = [id2 for _ in range(n_qubits)]
        factors_x[i] = sx
        factors_z[i] = sz
        x_ops.append(kron_chain(factors_x))
        z_ops.append(kron_chain(factors_z))

    return x_ops, z_ops


def multiply_ops(ops: list[csr_matrix]) -> csr_matrix:
    """Multiply sparse operators in order."""

    out = ops[0]
    for op in ops[1:]:
        out = out @ op
    return out


def build_toric_code_hamiltonian(config: ToricCodeConfig) -> tuple[csr_matrix, list[csr_matrix], list[csr_matrix], list[csr_matrix], list[csr_matrix]]:
    """Build H, star operators, plaquette operators, and local X/Z tables."""

    L = config.L
    n_qubits = 2 * L * L
    dim = 1 << n_qubits

    x_ops, z_ops = build_pauli_tables(n_qubits)

    A_ops: list[csr_matrix] = []
    B_ops: list[csr_matrix] = []

    for y in range(L):
        for x in range(L):
            A_ops.append(multiply_ops([x_ops[idx] for idx in star_edges(x, y, L)]))
            B_ops.append(multiply_ops([z_ops[idx] for idx in plaquette_edges(x, y, L)]))

    H = csr_matrix((dim, dim), dtype=np.complex128)
    for A in A_ops:
        H = H - config.Je * A
    for B in B_ops:
        H = H - config.Jm * B

    H = 0.5 * (H + H.getH())
    return H, A_ops, B_ops, x_ops, z_ops


def expectation(vec: ComplexArray, op: csr_matrix) -> float:
    """Return real expectation value <vec|op|vec>."""

    val = np.vdot(vec, op @ vec)
    return float(np.real(val))


def build_wilson_loops(L: int, z_ops: list[csr_matrix]) -> tuple[csr_matrix, csr_matrix]:
    """Construct two non-contractible Z Wilson loops on the torus."""

    loop_x_edges = [h_edge(x, 0, L) for x in range(L)]
    loop_y_edges = [v_edge(0, y, L) for y in range(L)]

    Wx = multiply_ops([z_ops[idx] for idx in loop_x_edges])
    Wy = multiply_ops([z_ops[idx] for idx in loop_y_edges])
    return Wx, Wy


def project_operator_to_subspace(op: csr_matrix, basis: ComplexArray) -> ComplexArray:
    """Project operator into the subspace spanned by column vectors in basis."""

    dense_op = op.toarray()
    return basis.conj().T @ dense_op @ basis


def classify_ground_sectors(Wx: csr_matrix, Wy: csr_matrix, ground_basis: ComplexArray) -> list[tuple[float, float, float]]:
    """Label ground states by projected Wilson-loop eigenvalues.

    Returns rows of (energy_shift_in_subspace, <Wx>, <Wy>).
    """

    Mx = project_operator_to_subspace(Wx, ground_basis)
    My = project_operator_to_subspace(Wy, ground_basis)

    eval_x, U = np.linalg.eigh(0.5 * (Mx + Mx.conj().T))
    My_in_x_basis = U.conj().T @ My @ U

    labels: list[tuple[float, float, float]] = []
    tol = 1e-9
    n = len(eval_x)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(eval_x[j] - eval_x[i]) <= tol:
            j += 1

        # Resolve residual mixing inside degenerate Wx sectors by diagonalizing Wy blocks.
        block = 0.5 * (My_in_x_basis[i:j, i:j] + My_in_x_basis[i:j, i:j].conj().T)
        eval_y, _ = np.linalg.eigh(block)
        wx = float(np.real(np.mean(eval_x[i:j])))
        for wy in eval_y:
            labels.append((0.0, wx, float(np.real(wy))))

        i = j

    labels.sort(key=lambda t: (round(t[1], 6), round(t[2], 6)))
    return labels


def solve_toric_code(config: ToricCodeConfig) -> ToricCodeResult:
    """Solve toric-code model by exact diagonalization and diagnostics."""

    H, A_ops, B_ops, _, z_ops = build_toric_code_hamiltonian(config)
    dense_H = H.toarray()

    energies, eigenvectors = np.linalg.eigh(dense_H)
    energies = np.real_if_close(energies)

    E0 = float(energies[0])
    ground_mask = np.abs(energies - E0) <= config.degeneracy_tol
    ground_indices = np.where(ground_mask)[0]
    ground_degeneracy = int(len(ground_indices))

    if ground_degeneracy >= len(energies):
        gap = 0.0
    else:
        gap = float(energies[ground_degeneracy] - E0)

    gs = eigenvectors[:, ground_indices[0]]
    As_mean = float(np.mean([expectation(gs, A) for A in A_ops]))
    Bp_mean = float(np.mean([expectation(gs, B) for B in B_ops]))

    ground_basis = eigenvectors[:, ground_indices]
    Wx, Wy = build_wilson_loops(config.L, z_ops)
    sector_labels = classify_ground_sectors(Wx, Wy, ground_basis)

    return ToricCodeResult(
        energies=np.asarray(energies, dtype=np.float64),
        eigenvectors=eigenvectors,
        ground_degeneracy=ground_degeneracy,
        gap=gap,
        As_mean=As_mean,
        Bp_mean=Bp_mean,
        sector_labels=sector_labels,
    )


def main() -> None:
    config = ToricCodeConfig()
    result = solve_toric_code(config)

    n_show = min(10, len(result.energies))
    spectrum_df = pd.DataFrame(
        {
            "level": np.arange(n_show),
            "energy": result.energies[:n_show],
            "deltaE": result.energies[:n_show] - result.energies[0],
        }
    )

    sector_df = pd.DataFrame(result.sector_labels, columns=["dE_subspace", "Wx", "Wy"])

    print("=== Quantum Spin Liquid MVP: Toric Code on L=2 Torus ===")
    print(f"L={config.L}, qubits={2 * config.L * config.L}, Je={config.Je:.3f}, Jm={config.Jm:.3f}")
    print(f"ground_degeneracy={result.ground_degeneracy}, gap={result.gap:.6f}")
    print(f"<A_s>_mean={result.As_mean:.6f}, <B_p>_mean={result.Bp_mean:.6f}")

    print("\nLowest spectrum:")
    print(spectrum_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    print("\nWilson-loop labels inside ground manifold:")
    print(sector_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    assert result.ground_degeneracy >= 4, "Expected at least four ground states on torus."
    assert result.gap > 0.0, "Expected a positive excitation gap."
    assert np.isfinite(result.As_mean) and np.isfinite(result.Bp_mean)
    assert abs(result.As_mean) <= 1.000001 and abs(result.Bp_mean) <= 1.000001
    assert np.all(np.isfinite(result.energies))


if __name__ == "__main__":
    main()
