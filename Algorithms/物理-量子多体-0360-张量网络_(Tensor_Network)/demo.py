"""Minimal runnable MVP for Tensor Network in quantum many-body physics.

This demo builds the ground state of a 1D transverse-field Ising chain,
then compresses that many-body wavefunction into an MPS (matrix product state)
using sequential SVD (TT-SVD style).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.sparse import csr_matrix, kron
from scipy.sparse.linalg import eigsh


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the Tensor Network MVP."""

    n_sites: int = 10
    coupling_j: float = 1.0
    field_hx: float = 1.1
    chi_grid: tuple[int, ...] = (2, 4, 8, 16, 32)
    svd_cutoff: float = 1e-12

    def validate(self) -> None:
        if self.n_sites < 4:
            raise ValueError("n_sites must be >= 4")
        if any(chi <= 0 for chi in self.chi_grid):
            raise ValueError("all chi values must be positive")
        if sorted(self.chi_grid) != list(self.chi_grid):
            raise ValueError("chi_grid must be sorted ascending")
        if self.svd_cutoff < 0.0:
            raise ValueError("svd_cutoff must be non-negative")


@dataclass
class MPSDecomposition:
    """Container for one MPS decomposition result."""

    cores: list[np.ndarray]
    singular_values: list[np.ndarray]


ID2 = csr_matrix(np.eye(2, dtype=float))
X = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float))
Z = csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float))


def kron_all(ops: Sequence[csr_matrix]) -> csr_matrix:
    """Kronecker product of an operator list."""
    out = ops[0]
    for op in ops[1:]:
        out = kron(out, op, format="csr")
    return out


def build_tfim_hamiltonian(n_sites: int, coupling_j: float, field_hx: float) -> csr_matrix:
    """Build 1D open-boundary transverse-field Ising Hamiltonian.

    H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i
    """
    if n_sites < 2:
        raise ValueError("n_sites must be >= 2")

    dim = 2**n_sites
    hamiltonian = csr_matrix((dim, dim), dtype=float)

    for site in range(n_sites - 1):
        ops = [ID2 for _ in range(n_sites)]
        ops[site] = Z
        ops[site + 1] = Z
        hamiltonian = hamiltonian - coupling_j * kron_all(ops)

    for site in range(n_sites):
        ops = [ID2 for _ in range(n_sites)]
        ops[site] = X
        hamiltonian = hamiltonian - field_hx * kron_all(ops)

    return hamiltonian


def lowest_eigenpair(hamiltonian: csr_matrix) -> tuple[float, np.ndarray]:
    """Return the ground-state energy/vector of a sparse Hermitian matrix."""
    evals, evecs = eigsh(hamiltonian, k=1, which="SA", tol=1e-10, maxiter=300000)
    energy0 = float(np.real(evals[0]))
    psi0 = np.asarray(evecs[:, 0], dtype=np.complex128)
    norm = np.linalg.norm(psi0)
    if norm <= 0.0:
        raise RuntimeError("ground-state vector has zero norm")
    psi0 /= norm
    return energy0, psi0


def state_to_tensor(state: np.ndarray, n_sites: int) -> np.ndarray:
    """Reshape |psi> to a rank-n tensor with physical dimension 2 per site."""
    expected = 2**n_sites
    if state.ndim != 1 or state.size != expected:
        raise ValueError(f"state must be 1D with size {expected}")
    return state.reshape((2,) * n_sites)


def _choose_rank(singular_values: np.ndarray, max_bond: int, cutoff: float) -> int:
    """Select kept bond dimension from singular values."""
    keep = int(np.sum(singular_values > cutoff)) if cutoff > 0.0 else singular_values.size
    keep = max(1, keep)
    keep = min(keep, max_bond)
    return keep


def mps_from_state_by_svd(state_tensor: np.ndarray, max_bond: int, cutoff: float) -> MPSDecomposition:
    """Sequential SVD decomposition from full state tensor to open-boundary MPS."""
    if state_tensor.ndim < 2:
        raise ValueError("state_tensor must have at least 2 modes")
    if any(dim != 2 for dim in state_tensor.shape):
        raise ValueError("this MVP assumes qubit chain with physical dim=2")

    n_sites = state_tensor.ndim
    residual = state_tensor.astype(np.complex128, copy=True)

    cores: list[np.ndarray] = []
    singular_spectrum: list[np.ndarray] = []
    left_rank = 1

    for site in range(n_sites - 1):
        residual = residual.reshape(left_rank * 2, -1)
        u, s, vh = svd(residual, full_matrices=False)
        keep = _choose_rank(s, max_bond=max_bond, cutoff=cutoff)

        u = u[:, :keep]
        s = s[:keep]
        vh = vh[:keep, :]

        cores.append(u.reshape(left_rank, 2, keep))
        singular_spectrum.append(s.copy())

        residual = s[:, None] * vh
        left_rank = keep

    cores.append(residual.reshape(left_rank, 2, 1))
    return MPSDecomposition(cores=cores, singular_values=singular_spectrum)


def mps_to_state(cores: Sequence[np.ndarray]) -> np.ndarray:
    """Reconstruct dense state vector from MPS cores."""
    if not cores:
        raise ValueError("cores must be non-empty")
    result = cores[0]
    for core in cores[1:]:
        result = np.tensordot(result, core, axes=([-1], [0]))
    result = np.squeeze(result, axis=(0, -1))
    return result.reshape(-1)


def align_phase(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Remove global phase from candidate so comparison with reference is meaningful."""
    overlap = np.vdot(reference, candidate)
    if abs(overlap) < 1e-15:
        return candidate
    phase = overlap / abs(overlap)
    return candidate * np.conjugate(phase)


def normalized_state(state: np.ndarray) -> np.ndarray:
    """Return normalized copy of a state vector."""
    norm = np.linalg.norm(state)
    if norm <= 0.0:
        raise ValueError("state norm must be positive")
    return state / norm


def mps_norm(cores: Sequence[np.ndarray]) -> float:
    """Compute <psi|psi> using transfer-matrix contraction."""
    identity = np.eye(2, dtype=np.complex128)
    env = np.array([1.0 + 0.0j], dtype=np.complex128)
    for core in cores:
        env = np.einsum("i,ij->j", env, single_site_transfer(core, identity), optimize=True)
    return float(np.real_if_close(env.item()))


def normalize_mps_cores(cores: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Scale first core so that MPS has unit norm."""
    copied = [core.copy() for core in cores]
    norm_sq = mps_norm(copied)
    if norm_sq <= 0.0:
        raise ValueError("cannot normalize zero-norm MPS")
    copied[0] /= np.sqrt(norm_sq)
    return copied


def single_site_transfer(core: np.ndarray, operator: np.ndarray) -> np.ndarray:
    """Construct one transfer matrix E(O) for a given local operator O."""
    left_rank, phys_dim, right_rank = core.shape
    transfer = np.zeros((left_rank * left_rank, right_rank * right_rank), dtype=np.complex128)
    for s in range(phys_dim):
        for t in range(phys_dim):
            coeff = operator[s, t]
            if abs(coeff) < 1e-15:
                continue
            transfer += coeff * np.kron(np.conjugate(core[:, s, :]), core[:, t, :])
    return transfer


def mps_local_z_expectation(cores: Sequence[np.ndarray], site: int) -> float:
    """Compute <Z_site> from MPS via transfer contractions."""
    n_sites = len(cores)
    if not (0 <= site < n_sites):
        raise ValueError("site index out of range")

    identity = np.eye(2, dtype=np.complex128)
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    env = np.array([1.0 + 0.0j], dtype=np.complex128)
    for idx, core in enumerate(cores):
        operator = pauli_z if idx == site else identity
        env = np.einsum("i,ij->j", env, single_site_transfer(core, operator), optimize=True)
    return float(np.real_if_close(env.item()))


def exact_local_z_expectation(state: np.ndarray, n_sites: int, site: int) -> float:
    """Compute exact <Z_site> from dense state tensor."""
    tensor = state.reshape((2,) * n_sites)
    moved = np.moveaxis(tensor, site, 0).reshape(2, -1)
    probs = np.sum(np.abs(moved) ** 2, axis=1)
    return float(probs[0] - probs[1])


def entanglement_entropy(singular_values: np.ndarray) -> float:
    """Von Neumann entropy from Schmidt singular values across one bond."""
    probs = np.square(np.abs(singular_values))
    total = probs.sum()
    if total <= 0.0:
        return 0.0
    probs = probs / total
    probs = probs[probs > 1e-15]
    return float(-np.sum(probs * np.log2(probs)))


def compression_report(
    exact_state: np.ndarray,
    state_tensor: np.ndarray,
    hamiltonian: csr_matrix,
    exact_energy: float,
    chi_grid: Sequence[int],
    cutoff: float,
) -> pd.DataFrame:
    """Run MPS compression for each chi and summarize quality/cost metrics."""
    n_sites = state_tensor.ndim
    center_site = n_sites // 2
    exact_z_center = exact_local_z_expectation(exact_state, n_sites, center_site)

    rows: list[dict[str, float]] = []
    for chi in chi_grid:
        decomp = mps_from_state_by_svd(state_tensor, max_bond=chi, cutoff=cutoff)
        recon = normalized_state(mps_to_state(decomp.cores))
        recon = align_phase(exact_state, recon)

        relative_error = float(np.linalg.norm(recon - exact_state) / np.linalg.norm(exact_state))
        fidelity = float(abs(np.vdot(exact_state, recon)) ** 2)

        energy_recon = float(np.real(np.vdot(recon, hamiltonian @ recon)))
        energy_abs_error = abs(energy_recon - exact_energy)

        normalized_cores = normalize_mps_cores(decomp.cores)
        z_center_mps = mps_local_z_expectation(normalized_cores, center_site)
        z_center_abs_error = abs(z_center_mps - exact_z_center)

        observed_max_bond = max(core.shape[2] for core in decomp.cores[:-1]) if n_sites > 1 else 1
        mps_params = int(sum(core.size for core in decomp.cores))
        full_params = int(exact_state.size)

        middle_bond = (n_sites // 2) - 1
        center_entropy = entanglement_entropy(decomp.singular_values[middle_bond])

        rows.append(
            {
                "chi_cap": int(chi),
                "max_observed_bond": int(observed_max_bond),
                "mps_params": mps_params,
                "compression_ratio(full/mps)": full_params / mps_params,
                "relative_state_error": relative_error,
                "fidelity": fidelity,
                "energy_abs_error": energy_abs_error,
                "center_site_z_abs_error": z_center_abs_error,
                "center_bond_entropy": center_entropy,
            }
        )

    return pd.DataFrame(rows)


def run_checks(report: pd.DataFrame) -> None:
    """Assert minimal correctness properties for this MVP."""
    rel_errors = report["relative_state_error"].to_numpy(dtype=float)
    fidelities = report["fidelity"].to_numpy(dtype=float)
    energy_errors = report["energy_abs_error"].to_numpy(dtype=float)
    z_errors = report["center_site_z_abs_error"].to_numpy(dtype=float)

    if not np.all(np.diff(rel_errors) <= 1e-10):
        raise AssertionError("relative_state_error should be non-increasing as chi grows")

    if not np.all(np.diff(fidelities) >= -1e-10):
        raise AssertionError("fidelity should be non-decreasing as chi grows")

    if fidelities[-1] < 1.0 - 1e-10:
        raise AssertionError("largest chi should recover near-exact state")

    if energy_errors[-1] > 1e-9:
        raise AssertionError("largest chi energy error is too large")

    if z_errors[-1] > 1e-9:
        raise AssertionError("largest chi local observable error is too large")

    if rel_errors[2] > 0.2:
        raise AssertionError("chi=8 should provide a useful approximation in this setup")


def main() -> None:
    cfg = ExperimentConfig()
    cfg.validate()

    hamiltonian = build_tfim_hamiltonian(
        n_sites=cfg.n_sites,
        coupling_j=cfg.coupling_j,
        field_hx=cfg.field_hx,
    )
    exact_energy, exact_state = lowest_eigenpair(hamiltonian)
    state_tensor = state_to_tensor(exact_state, n_sites=cfg.n_sites)

    report = compression_report(
        exact_state=exact_state,
        state_tensor=state_tensor,
        hamiltonian=hamiltonian,
        exact_energy=exact_energy,
        chi_grid=cfg.chi_grid,
        cutoff=cfg.svd_cutoff,
    )

    print("=== Tensor Network MVP: MPS compression of TFIM ground state ===")
    print(
        f"n_sites={cfg.n_sites}, J={cfg.coupling_j:.3f}, h={cfg.field_hx:.3f}, "
        f"hilbert_dim={2**cfg.n_sites}"
    )
    print(f"exact_ground_energy={exact_energy:.12f}")
    print()
    print(report.to_string(index=False))

    run_checks(report)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
