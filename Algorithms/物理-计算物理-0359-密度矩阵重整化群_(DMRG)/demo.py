"""Minimal runnable MVP for density-matrix renormalization group (DMRG).

This script implements an infinite-system DMRG baseline for the spin-1/2
antiferromagnetic Heisenberg chain with open boundaries:

    H = J * sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})

To keep the implementation auditable:
- Superblock ground state is solved with scipy.sparse.linalg.eigsh.
- Reduced density matrix is built explicitly and truncated by largest eigenvalues.
- A small-system exact diagonalization benchmark is included.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import eigsh


# Single-site spin-1/2 operators (real representation via S+, S-, Sz)
ID2 = csr_matrix(np.eye(2, dtype=float))
SZ = csr_matrix(np.array([[0.5, 0.0], [0.0, -0.5]], dtype=float))
SP = csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float))
SM = csr_matrix(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float))


@dataclass
class Block:
    """Renormalized block used by infinite-system DMRG."""

    length: int
    basis_dim: int
    h: csr_matrix
    sz_edge: csr_matrix
    sp_edge: csr_matrix
    sm_edge: csr_matrix


def heisenberg_bond(op_left_sp: csr_matrix, op_left_sm: csr_matrix, op_left_sz: csr_matrix,
                    op_right_sp: csr_matrix, op_right_sm: csr_matrix, op_right_sz: csr_matrix,
                    j: float) -> csr_matrix:
    """Return J*(SxSx + SySy + SzSz) using S+, S-, Sz operators."""
    flip_flop = 0.5 * (kron(op_left_sp, op_right_sm, format="csr") + kron(op_left_sm, op_right_sp, format="csr"))
    zz_term = kron(op_left_sz, op_right_sz, format="csr")
    return j * (flip_flop + zz_term)


def initial_block() -> Block:
    """Start from one physical site as the initial block."""
    h0 = csr_matrix((2, 2), dtype=float)
    return Block(length=1, basis_dim=2, h=h0, sz_edge=SZ, sp_edge=SP, sm_edge=SM)


def enlarge_block(block: Block, j: float) -> Block:
    """Add one site to the right side of a block."""
    id_block = eye(block.basis_dim, format="csr")
    h_enlarged = kron(block.h, ID2, format="csr")
    h_enlarged = h_enlarged + heisenberg_bond(block.sp_edge, block.sm_edge, block.sz_edge, SP, SM, SZ, j)

    # New edge operator is on the appended site.
    sz_edge_new = kron(id_block, SZ, format="csr")
    sp_edge_new = kron(id_block, SP, format="csr")
    sm_edge_new = kron(id_block, SM, format="csr")

    return Block(
        length=block.length + 1,
        basis_dim=block.basis_dim * 2,
        h=h_enlarged,
        sz_edge=sz_edge_new,
        sp_edge=sp_edge_new,
        sm_edge=sm_edge_new,
    )


def superblock_hamiltonian(left: Block, right: Block, j: float) -> csr_matrix:
    """Construct H_super = H_left + H_right + center bond."""
    id_left = eye(left.basis_dim, format="csr")
    id_right = eye(right.basis_dim, format="csr")

    h_lr = kron(left.h, id_right, format="csr") + kron(id_left, right.h, format="csr")
    h_center = heisenberg_bond(left.sp_edge, left.sm_edge, left.sz_edge,
                               right.sp_edge, right.sm_edge, right.sz_edge, j)
    return h_lr + h_center


def ground_state(h: csr_matrix) -> tuple[float, np.ndarray]:
    """Lowest-energy eigenpair of sparse Hermitian matrix."""
    n = h.shape[0]
    if n <= 8:
        dense = h.toarray()
        evals, evecs = np.linalg.eigh(dense)
        e0 = float(evals[0])
        psi0 = evecs[:, 0]
    else:
        evals, evecs = eigsh(h, k=1, which="SA", tol=1e-10, maxiter=300000)
        e0 = float(np.real(evals[0]))
        psi0 = np.real(evecs[:, 0])

    norm = np.linalg.norm(psi0)
    if norm <= 0.0:
        raise RuntimeError("Ground-state vector has zero norm.")
    psi0 = psi0 / norm
    return e0, psi0


def project_operator(op: csr_matrix, basis: np.ndarray) -> csr_matrix:
    """Project operator to truncated basis: O' = U^T O U."""
    op_dense = op.toarray()
    projected = basis.T @ op_dense @ basis
    return csr_matrix(projected)


def truncate_left_block(enlarged_left: Block, psi_super: np.ndarray, right_dim: int, m_keep: int) -> tuple[Block, float]:
    """Density-matrix truncation on left enlarged block."""
    left_dim = enlarged_left.basis_dim
    psi_matrix = psi_super.reshape(left_dim, right_dim)

    rho_left = psi_matrix @ psi_matrix.T
    rho_left = 0.5 * (rho_left + rho_left.T)  # enforce symmetry numerically

    eigvals, eigvecs = np.linalg.eigh(rho_left)
    order = np.argsort(eigvals)[::-1]
    keep = min(m_keep, left_dim)
    keep_idx = order[:keep]

    kept_weights = eigvals[keep_idx]
    basis = eigvecs[:, keep_idx]
    truncation_error = float(max(0.0, 1.0 - np.sum(kept_weights)))

    truncated = Block(
        length=enlarged_left.length,
        basis_dim=keep,
        h=project_operator(enlarged_left.h, basis),
        sz_edge=project_operator(enlarged_left.sz_edge, basis),
        sp_edge=project_operator(enlarged_left.sp_edge, basis),
        sm_edge=project_operator(enlarged_left.sm_edge, basis),
    )
    return truncated, truncation_error


def infinite_dmrg(length: int, m_keep: int, j: float = 1.0) -> pd.DataFrame:
    """Run infinite-system DMRG up to the target chain length (even length)."""
    if length < 4 or length % 2 != 0:
        raise ValueError("length must be even and >= 4")
    if m_keep < 2:
        raise ValueError("m_keep must be >= 2")

    block = initial_block()
    rows: list[dict[str, float]] = []

    while 2 * block.length < length:
        left_enlarged = enlarge_block(block, j)
        right_enlarged = enlarge_block(block, j)  # mirrored by construction for this MVP

        h_super = superblock_hamiltonian(left_enlarged, right_enlarged, j)
        e0, psi0 = ground_state(h_super)

        block, trunc_err = truncate_left_block(left_enlarged, psi0, right_enlarged.basis_dim, m_keep)
        full_length = 2 * block.length

        rows.append(
            {
                "step": len(rows) + 1,
                "length": full_length,
                "block_dim": block.basis_dim,
                "super_dim": float(h_super.shape[0]),
                "ground_energy": e0,
                "energy_per_site": e0 / full_length,
                "truncation_error": trunc_err,
            }
        )

    return pd.DataFrame(rows)


def kron_all(ops: list[csr_matrix]) -> csr_matrix:
    """Kronecker product of an operator list."""
    out = ops[0]
    for op in ops[1:]:
        out = kron(out, op, format="csr")
    return out


def exact_heisenberg_energy(length: int, j: float = 1.0) -> float:
    """Exact diagonalization ground energy for a small open chain."""
    if length < 2:
        raise ValueError("length must be >= 2")

    dim = 2 ** length
    h_total = csr_matrix((dim, dim), dtype=float)

    for site in range(length - 1):
        base_ops = [ID2 for _ in range(length)]

        ops_sp_sm = base_ops.copy()
        ops_sp_sm[site] = SP
        ops_sp_sm[site + 1] = SM

        ops_sm_sp = base_ops.copy()
        ops_sm_sp[site] = SM
        ops_sm_sp[site + 1] = SP

        ops_sz_sz = base_ops.copy()
        ops_sz_sz[site] = SZ
        ops_sz_sz[site + 1] = SZ

        h_total = h_total + j * (
            0.5 * (kron_all(ops_sp_sm) + kron_all(ops_sm_sp))
            + kron_all(ops_sz_sz)
        )

    e0, _ = ground_state(h_total)
    return e0


def main() -> None:
    target_length = 20
    m_keep = 24
    coupling_j = 1.0

    history = infinite_dmrg(length=target_length, m_keep=m_keep, j=coupling_j)
    if history.empty:
        raise RuntimeError("DMRG history is empty; configuration is invalid.")

    final = history.iloc[-1]

    # Small-size cross-check: compare DMRG vs exact diagonalization at L=10.
    small_length = 10
    small_dmrg = infinite_dmrg(length=small_length, m_keep=m_keep, j=coupling_j)
    e_dmrg_small = float(small_dmrg.iloc[-1]["ground_energy"])
    e_exact_small = exact_heisenberg_energy(length=small_length, j=coupling_j)
    abs_err_small = abs(e_dmrg_small - e_exact_small)

    print("DMRG (infinite-system) for 1D spin-1/2 Heisenberg chain")
    print(f"target_length={target_length}, m_keep={m_keep}, J={coupling_j:.2f}")
    print()
    print("Last 5 growth steps:")
    print(history.tail(5).to_string(index=False))
    print()
    print(f"Final total energy at L={int(final['length'])}: {float(final['ground_energy']):.10f}")
    print(f"Final energy per site: {float(final['energy_per_site']):.10f}")
    print(f"Final truncation error: {float(final['truncation_error']):.3e}")
    print()
    print("Small-system validation (L=10):")
    print(f"DMRG energy:  {e_dmrg_small:.10f}")
    print(f"Exact energy: {e_exact_small:.10f}")
    print(f"Absolute error: {abs_err_small:.3e}")

    # A pragmatic tolerance for this lightweight infinite-system MVP.
    if abs_err_small > 2e-2:
        raise AssertionError(
            f"DMRG small-system error too large: {abs_err_small:.3e} > 2e-2"
        )

    print("All checks passed.")


if __name__ == "__main__":
    main()
