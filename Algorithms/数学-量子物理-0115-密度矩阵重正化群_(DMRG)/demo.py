"""Minimal runnable MVP for Density Matrix Renormalization Group (DMRG).

This script implements an infinite-system DMRG toy solver for the
spin-1/2 antiferromagnetic Heisenberg chain with open boundaries:

    H = sum_{i=1}^{N-1} (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})

No external DMRG framework is used; all core steps are explicitly coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


DTYPE = np.complex128


@dataclass
class Block:
    """Renormalized block used in infinite-system DMRG."""

    length: int
    h: np.ndarray
    sx_edge: np.ndarray
    sy_edge: np.ndarray
    sz_edge: np.ndarray
    trunc_error: float = 0.0



def spin_operators() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return spin-1/2 operators (Pauli/2)."""
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=DTYPE)
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=DTYPE)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=DTYPE)
    return sx, sy, sz



def init_single_site_block() -> Block:
    """Create length-1 block with zero on-site Hamiltonian."""
    sx, sy, sz = spin_operators()
    h0 = np.zeros((2, 2), dtype=DTYPE)
    return Block(length=1, h=h0, sx_edge=sx, sy_edge=sy, sz_edge=sz)



def enlarge_block(block: Block, sx: np.ndarray, sy: np.ndarray, sz: np.ndarray) -> Block:
    """Enlarge block by adding one site on the right."""
    dim = block.h.shape[0]
    eye_dim = np.eye(dim, dtype=DTYPE)
    eye_site = np.eye(2, dtype=DTYPE)

    h_new = np.kron(block.h, eye_site)
    h_new += np.kron(block.sx_edge, sx)
    h_new += np.kron(block.sy_edge, sy)
    h_new += np.kron(block.sz_edge, sz)

    sx_new = np.kron(eye_dim, sx)
    sy_new = np.kron(eye_dim, sy)
    sz_new = np.kron(eye_dim, sz)

    return Block(
        length=block.length + 1,
        h=h_new,
        sx_edge=sx_new,
        sy_edge=sy_new,
        sz_edge=sz_new,
    )



def build_superblock_hamiltonian(system: Block, env: Block) -> np.ndarray:
    """Build superblock Hamiltonian H_sys + H_env + boundary coupling."""
    d_sys = system.h.shape[0]
    d_env = env.h.shape[0]

    eye_sys = np.eye(d_sys, dtype=DTYPE)
    eye_env = np.eye(d_env, dtype=DTYPE)

    h_super = np.kron(system.h, eye_env)
    h_super += np.kron(eye_sys, env.h)
    h_super += np.kron(system.sx_edge, env.sx_edge)
    h_super += np.kron(system.sy_edge, env.sy_edge)
    h_super += np.kron(system.sz_edge, env.sz_edge)
    return h_super



def ground_state_dense(h: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute lowest eigenpair of a Hermitian matrix."""
    evals, evecs = np.linalg.eigh(h)
    e0 = float(np.real(evals[0]))
    psi0 = evecs[:, 0]
    return e0, psi0



def truncate_system_block(system: Block, psi0: np.ndarray, m_keep: int) -> Block:
    """Density-matrix truncation on system block basis."""
    d_sys = system.h.shape[0]
    d_env = psi0.size // d_sys
    psi_mat = psi0.reshape(d_sys, d_env)

    rho_sys = psi_mat @ psi_mat.conj().T
    rho_sys = 0.5 * (rho_sys + rho_sys.conj().T)

    evals, evecs = np.linalg.eigh(rho_sys)
    order = np.argsort(evals)[::-1]

    m_eff = min(m_keep, d_sys)
    keep = order[:m_eff]
    kept_weights = np.clip(np.real(evals[keep]), 0.0, None)
    trunc_error = float(max(0.0, 1.0 - np.sum(kept_weights)))

    basis = evecs[:, keep]

    h_trunc = basis.conj().T @ system.h @ basis
    sx_trunc = basis.conj().T @ system.sx_edge @ basis
    sy_trunc = basis.conj().T @ system.sy_edge @ basis
    sz_trunc = basis.conj().T @ system.sz_edge @ basis

    h_trunc = 0.5 * (h_trunc + h_trunc.conj().T)

    return Block(
        length=system.length,
        h=h_trunc,
        sx_edge=sx_trunc,
        sy_edge=sy_trunc,
        sz_edge=sz_trunc,
        trunc_error=trunc_error,
    )



def infinite_dmrg(target_length: int, m_keep: int) -> List[Dict[str, float]]:
    """Run infinite-system DMRG until reaching target even length."""
    if target_length < 4 or target_length % 2 != 0:
        raise ValueError("target_length must be an even integer >= 4")
    if m_keep < 2:
        raise ValueError("m_keep must be >= 2")

    sx, sy, sz = spin_operators()
    block = init_single_site_block()

    history: List[Dict[str, float]] = []

    while True:
        system = enlarge_block(block, sx, sy, sz)
        env = system  # mirror block in infinite-system setup

        h_super = build_superblock_hamiltonian(system, env)
        e0, psi0 = ground_state_dense(h_super)

        chain_length = 2 * system.length

        record: Dict[str, float] = {
            "length": float(chain_length),
            "energy": e0,
            "energy_per_site": e0 / chain_length,
            "system_dim_before_trunc": float(system.h.shape[0]),
            "trunc_error": float("nan"),
        }

        if chain_length >= target_length:
            history.append(record)
            break

        block = truncate_system_block(system, psi0, m_keep=m_keep)
        record["trunc_error"] = block.trunc_error
        record["system_dim_after_trunc"] = float(block.h.shape[0])
        history.append(record)

    return history



def kron_n_dense(ops: List[np.ndarray]) -> np.ndarray:
    """Kronecker product of a list of dense operators."""
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out



def two_site_term_dense(
    n_sites: int,
    i: int,
    op_a: np.ndarray,
    op_b: np.ndarray,
    eye_2: np.ndarray,
) -> np.ndarray:
    """Embed two-site operator acting on (i, i+1)."""
    ops: List[np.ndarray] = [eye_2 for _ in range(n_sites)]
    ops[i] = op_a
    ops[i + 1] = op_b
    return kron_n_dense(ops)



def exact_ground_energy_heisenberg(n_sites: int) -> float:
    """Exact ground energy by dense diagonalization (small N only)."""
    if n_sites < 2:
        raise ValueError("n_sites must be >= 2")

    sx, sy, sz = spin_operators()
    eye_2 = np.eye(2, dtype=DTYPE)

    dim = 2**n_sites
    h = np.zeros((dim, dim), dtype=DTYPE)

    for i in range(n_sites - 1):
        h += two_site_term_dense(n_sites, i, sx, sx, eye_2)
        h += two_site_term_dense(n_sites, i, sy, sy, eye_2)
        h += two_site_term_dense(n_sites, i, sz, sz, eye_2)

    eigvals = np.linalg.eigvalsh(h)
    return float(np.real(eigvals[0]))



def print_history(history: List[Dict[str, float]]) -> None:
    """Pretty-print DMRG growth history."""
    print("DMRG growth history (infinite-system):")
    print(
        "  length | total_energy | energy/site | dim_before | dim_after | trunc_error"
    )
    for rec in history:
        length = int(rec["length"])
        e = rec["energy"]
        e_site = rec["energy_per_site"]
        dim_before = int(rec["system_dim_before_trunc"])
        dim_after = rec.get("system_dim_after_trunc", float("nan"))
        trunc_error = rec["trunc_error"]

        dim_after_str = "-" if np.isnan(dim_after) else str(int(dim_after))
        trunc_str = "-" if np.isnan(trunc_error) else f"{trunc_error:.3e}"

        print(
            f"  {length:6d} | {e:12.8f} | {e_site:11.8f} |"
            f" {dim_before:10d} | {dim_after_str:8s} | {trunc_str}"
        )



def main() -> None:
    target_length = 8
    m_keep = 8

    history = infinite_dmrg(target_length=target_length, m_keep=m_keep)
    print_history(history)

    dmrg_e = history[-1]["energy"]
    exact_e = exact_ground_energy_heisenberg(target_length)

    abs_err = abs(dmrg_e - exact_e)
    per_site_err = abs_err / target_length

    print("\nCross-check with exact diagonalization:")
    print(f"  N = {target_length}")
    print(f"  DMRG energy      = {dmrg_e:.10f}")
    print(f"  Exact energy     = {exact_e:.10f}")
    print(f"  Absolute error   = {abs_err:.6e}")
    print(f"  Per-site error   = {per_site_err:.6e}")

    if not np.isfinite(dmrg_e):
        raise RuntimeError("DMRG returned non-finite energy")
    if per_site_err > 7e-2:
        raise RuntimeError(
            "DMRG MVP quality check failed: per-site error is larger than 7e-2"
        )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
