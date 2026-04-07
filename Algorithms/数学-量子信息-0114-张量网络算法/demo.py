"""Minimal runnable MVP for tensor-network algorithm (MPS / TT-SVD)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

try:
    import scipy.linalg as sp_linalg

    HAVE_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    sp_linalg = None
    HAVE_SCIPY = False

try:
    import pandas as pd

    HAVE_PANDAS = True
except Exception:  # pragma: no cover - optional dependency
    pd = None
    HAVE_PANDAS = False

try:
    import torch

    HAVE_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    torch = None
    HAVE_TORCH = False


@dataclass
class TTDecomposition:
    """Container for one MPS decomposition result."""

    cores: List[np.ndarray]
    singular_values: List[np.ndarray]


def _svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD with SciPy fallback to NumPy."""
    if HAVE_SCIPY:
        return sp_linalg.svd(matrix, full_matrices=False)
    return np.linalg.svd(matrix, full_matrices=False)


def random_quantum_state(n_qubits: int, seed: int = 7, intrinsic_bond: int = 12) -> np.ndarray:
    """Create a reproducible medium-entanglement random state via random MPS cores."""
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    rng = np.random.default_rng(seed)
    if intrinsic_bond <= 0:
        raise ValueError("intrinsic_bond must be positive")

    # Choose valid internal ranks with both physical and intrinsic caps.
    ranks = [1]
    for cut in range(1, n_qubits):
        left_max = 2**cut
        right_max = 2 ** (n_qubits - cut)
        ranks.append(min(intrinsic_bond, left_max, right_max))
    ranks.append(1)

    cores = []
    for i in range(n_qubits):
        left_rank = ranks[i]
        right_rank = ranks[i + 1]
        core = rng.normal(size=(left_rank, 2, right_rank)) + 1j * rng.normal(
            size=(left_rank, 2, right_rank)
        )
        cores.append(core.astype(np.complex128))

    state_tensor = cores[0]
    for i in range(1, n_qubits):
        state_tensor = np.tensordot(state_tensor, cores[i], axes=([-1], [0]))
    state = np.squeeze(state_tensor, axis=(0, -1)).reshape(-1)
    state /= np.linalg.norm(state)
    return state


def vector_to_tensor(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """Reshape a state vector into a rank-n tensor with physical dimension 2."""
    expected_dim = 2**n_qubits
    if state.ndim != 1 or state.size != expected_dim:
        raise ValueError(f"state must be 1D with size {expected_dim}")
    return state.reshape((2,) * n_qubits)


def _truncation_rank(s: np.ndarray, max_bond: int | None, cutoff: float) -> int:
    keep = s.size
    if cutoff > 0.0:
        keep = int(np.sum(s > cutoff))
    keep = max(1, keep)
    if max_bond is not None:
        keep = min(keep, max_bond)
    return keep


def tt_svd(state_tensor: np.ndarray, max_bond: int | None, cutoff: float = 0.0) -> TTDecomposition:
    """Perform TT-SVD to obtain an open-boundary MPS.

    Args:
        state_tensor: Tensor of shape (2, 2, ..., 2).
        max_bond: Optional bond-dimension cap.
        cutoff: Singular-value truncation threshold.
    """
    if state_tensor.ndim < 2:
        raise ValueError("state_tensor must have at least 2 dimensions")
    if any(dim != 2 for dim in state_tensor.shape):
        raise ValueError("this MVP assumes qubits, so each mode must have size 2")

    n_qubits = state_tensor.ndim
    cores: List[np.ndarray] = []
    singular_values: List[np.ndarray] = []

    residual = state_tensor.astype(np.complex128, copy=True)
    left_rank = 1

    for site in range(n_qubits - 1):
        residual = residual.reshape(left_rank * 2, -1)
        u, s, vh = _svd(residual)
        keep = _truncation_rank(s, max_bond=max_bond, cutoff=cutoff)

        u = u[:, :keep]
        s = s[:keep]
        vh = vh[:keep, :]

        core = u.reshape(left_rank, 2, keep)
        cores.append(core)
        singular_values.append(s.copy())

        residual = s[:, None] * vh
        left_rank = keep

    final_core = residual.reshape(left_rank, 2, 1)
    cores.append(final_core)
    return TTDecomposition(cores=cores, singular_values=singular_values)


def mps_to_state(cores: Sequence[np.ndarray]) -> np.ndarray:
    """Reconstruct full state vector from MPS cores."""
    if not cores:
        raise ValueError("cores must be non-empty")

    result = cores[0]
    for idx in range(1, len(cores)):
        result = np.tensordot(result, cores[idx], axes=([-1], [0]))

    result = np.squeeze(result, axis=(0, -1))
    return result.reshape(-1)


def normalize_mps(cores: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Return a copied MPS whose global norm is 1 by scaling the first core."""
    copied = [core.copy() for core in cores]
    norm = np.linalg.norm(mps_to_state(copied))
    if norm <= 0.0:
        raise ValueError("cannot normalize zero-norm MPS")
    copied[0] = copied[0] / norm
    return copied


def mps_amplitude(cores: Sequence[np.ndarray], bitstring: Sequence[int]) -> complex:
    """Return amplitude <bitstring|psi> via sequential matrix-chain contraction."""
    if len(bitstring) != len(cores):
        raise ValueError("bitstring length must match number of qubits")
    vec = np.array([1.0 + 0.0j], dtype=np.complex128)
    for bit, core in zip(bitstring, cores):
        if bit not in (0, 1):
            raise ValueError("bitstring entries must be 0 or 1")
        vec = np.einsum("i,ij->j", vec, core[:, bit, :], optimize=True)
    return complex(vec.item())


def _single_site_transfer(core: np.ndarray, operator: np.ndarray) -> np.ndarray:
    """Build one transfer matrix E = sum_{s,t} O_{s,t} conj(A_s) \otimes A_t."""
    left_rank, phys_dim, right_rank = core.shape
    transfer = np.zeros((left_rank * left_rank, right_rank * right_rank), dtype=np.complex128)

    for s in range(phys_dim):
        for t in range(phys_dim):
            coeff = operator[s, t]
            if abs(coeff) < 1e-15:
                continue
            transfer += coeff * np.kron(np.conjugate(core[:, s, :]), core[:, t, :])

    return transfer


def mps_norm(cores: Sequence[np.ndarray]) -> float:
    """Compute <psi|psi> via transfer matrices."""
    identity = np.eye(2, dtype=np.complex128)
    env = np.array([1.0 + 0.0j], dtype=np.complex128)
    for core in cores:
        env = np.einsum("i,ij->j", env, _single_site_transfer(core, identity), optimize=True)
    return float(np.real_if_close(env.item()))


def mps_local_z_expectation(cores: Sequence[np.ndarray], site: int) -> float:
    """Compute <psi|Z_site|psi> without reconstructing the full state."""
    n_qubits = len(cores)
    if not (0 <= site < n_qubits):
        raise ValueError("site out of range")

    identity = np.eye(2, dtype=np.complex128)
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    env = np.array([1.0 + 0.0j], dtype=np.complex128)
    for idx, core in enumerate(cores):
        operator = pauli_z if idx == site else identity
        env = np.einsum("i,ij->j", env, _single_site_transfer(core, operator), optimize=True)
    return float(np.real_if_close(env.item()))


def exact_local_z_expectation(state: np.ndarray, n_qubits: int, site: int) -> float:
    """Ground-truth local Z expectation from full state tensor."""
    tensor = state.reshape((2,) * n_qubits)
    moved = np.moveaxis(tensor, site, 0).reshape(2, -1)
    probs = np.sum(np.abs(moved) ** 2, axis=1)
    return float(probs[0] - probs[1])


def bitstring_to_index(bits: Sequence[int]) -> int:
    """Convert bits like [1,0,1] to basis index 5."""
    idx = 0
    for bit in bits:
        idx = (idx << 1) | int(bit)
    return idx


def entanglement_entropy_from_singular_values(s: np.ndarray) -> float:
    """Von Neumann entropy S = -sum p_i log2 p_i for Schmidt probabilities p_i."""
    probs = np.square(np.abs(s))
    total = probs.sum()
    if total <= 0:
        return 0.0
    probs = probs / total
    probs = probs[probs > 1e-15]
    return float(-np.sum(probs * np.log2(probs)))


def summarize_compression(state: np.ndarray, state_tensor: np.ndarray, chi_grid: Sequence[int]) -> list[dict]:
    """Run TT-SVD at different bond caps and summarize compression quality."""
    n_qubits = state_tensor.ndim
    full_params = state.size
    rows: list[dict] = []

    for chi in chi_grid:
        decomp = tt_svd(state_tensor, max_bond=chi, cutoff=1e-12)
        recon = mps_to_state(decomp.cores)
        recon /= np.linalg.norm(recon)

        rel_error = float(np.linalg.norm(recon - state) / np.linalg.norm(state))
        fidelity = float(np.abs(np.vdot(state, recon)) ** 2)
        mps_params = int(sum(core.size for core in decomp.cores))
        max_observed_bond = int(max(core.shape[2] for core in decomp.cores[:-1])) if n_qubits > 1 else 1

        middle_bond = (n_qubits // 2) - 1
        middle_entropy = entanglement_entropy_from_singular_values(decomp.singular_values[middle_bond])

        rows.append(
            {
                "chi_cap": int(chi),
                "max_observed_bond": max_observed_bond,
                "mps_params": mps_params,
                "compression_ratio(full/mps)": full_params / mps_params,
                "relative_error": rel_error,
                "fidelity": fidelity,
                "middle_bond_entropy": middle_entropy,
            }
        )

    return rows


def _torch_amplitude_cross_check(cores: Sequence[np.ndarray], bitstring: Sequence[int]) -> complex | None:
    """Optional cross-check: compute the same amplitude path in PyTorch."""
    if not HAVE_TORCH:
        return None
    vec = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    for bit, core in zip(bitstring, cores):
        mat = torch.from_numpy(core[:, bit, :])
        vec = vec @ mat
    return complex(vec.item())


def main() -> None:
    n_qubits = 10
    seed = 11
    chi_exact = 32  # for n=10, exact rank upper bound at the center cut is 2^(n/2)=32
    chi_approx = 8

    state = random_quantum_state(n_qubits=n_qubits, seed=seed)
    state_tensor = vector_to_tensor(state, n_qubits=n_qubits)

    chi_grid = [2, 4, 8, 16, 32]
    rows = summarize_compression(state=state, state_tensor=state_tensor, chi_grid=chi_grid)

    print("=== MPS Compression Summary (TT-SVD) ===")
    if HAVE_PANDAS:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    else:
        for row in rows:
            print(row)

    decomp_exact = tt_svd(state_tensor, max_bond=chi_exact, cutoff=1e-12)
    decomp_approx = tt_svd(state_tensor, max_bond=chi_approx, cutoff=1e-12)
    cores_exact = normalize_mps(decomp_exact.cores)
    cores_approx = normalize_mps(decomp_approx.cores)

    recon_exact = mps_to_state(cores_exact)
    recon_exact /= np.linalg.norm(recon_exact)
    recon_approx = mps_to_state(cores_approx)
    recon_approx /= np.linalg.norm(recon_approx)

    rel_error_exact = float(np.linalg.norm(recon_exact - state) / np.linalg.norm(state))
    rel_error_approx = float(np.linalg.norm(recon_approx - state) / np.linalg.norm(state))

    # Bitstring amplitude checks.
    test_bitstrings = [
        [0] * n_qubits,
        [1] * n_qubits,
        [0, 1] * (n_qubits // 2),
        [1, 0] * (n_qubits // 2),
    ]

    print("\n=== Basis Amplitude Checks ===")
    amp_errors_exact = []
    amp_errors_approx = []
    for bits in test_bitstrings:
        idx = bitstring_to_index(bits)
        amp_true = state[idx]
        amp_exact = mps_amplitude(cores_exact, bits)
        amp_approx = mps_amplitude(cores_approx, bits)

        err_exact = abs(amp_exact - amp_true)
        err_approx = abs(amp_approx - amp_true)
        amp_errors_exact.append(err_exact)
        amp_errors_approx.append(err_approx)

        print(
            f"bits={''.join(str(b) for b in bits)} "
            f"|true|={abs(amp_true):.6e} "
            f"err_exact={err_exact:.3e} err_chi{chi_approx}={err_approx:.3e}"
        )

    # Optional torch consistency check on one amplitude path.
    torch_amp = _torch_amplitude_cross_check(cores_exact, test_bitstrings[0])
    if torch_amp is not None:
        np_amp = mps_amplitude(cores_exact, test_bitstrings[0])
        print(f"\nPyTorch cross-check (first bitstring): |delta|={abs(torch_amp - np_amp):.3e}")

    # Local observable checks: <Z_i> using full vector and MPS transfer contraction.
    z_true = np.array([exact_local_z_expectation(state, n_qubits, i) for i in range(n_qubits)])
    z_exact_mps = np.array([mps_local_z_expectation(cores_exact, i) for i in range(n_qubits)])
    z_approx_mps = np.array([mps_local_z_expectation(cores_approx, i) for i in range(n_qubits)])

    mean_abs_z_err_exact = float(np.mean(np.abs(z_exact_mps - z_true)))
    mean_abs_z_err_approx = float(np.mean(np.abs(z_approx_mps - z_true)))

    print("\n=== Local Z Expectation Error ===")
    print(f"mean_abs_error(chi={chi_exact}) = {mean_abs_z_err_exact:.3e}")
    print(f"mean_abs_error(chi={chi_approx}) = {mean_abs_z_err_approx:.3e}")

    # Norm checks from transfer matrices.
    norm_exact = mps_norm(cores_exact)
    norm_approx = mps_norm(cores_approx)
    print("\n=== Norm Checks ===")
    print(f"<psi_exact|psi_exact> from MPS transfer = {norm_exact:.12f}")
    print(f"<psi_approx|psi_approx> from MPS transfer = {norm_approx:.12f}")

    # Regression-style sanity assertions for this MVP.
    rel_errors = [row["relative_error"] for row in rows]
    for i in range(1, len(rel_errors)):
        if rel_errors[i] > rel_errors[i - 1] + 1e-10:
            raise AssertionError("relative error should be non-increasing with larger chi")

    if rel_error_exact > 1e-10:
        raise AssertionError(f"exact-rank reconstruction too large: {rel_error_exact}")
    if max(amp_errors_exact) > 1e-10:
        raise AssertionError("exact-rank MPS amplitude mismatch is too large")
    if mean_abs_z_err_exact > 1e-10:
        raise AssertionError("exact-rank MPS local-Z mismatch is too large")
    if abs(norm_exact - 1.0) > 1e-10:
        raise AssertionError("exact-rank MPS norm should be 1")
    if abs(norm_approx - 1.0) > 1e-8:
        raise AssertionError("approx MPS norm drift is unexpectedly large")

    # The compressed case should be clearly approximate but still useful.
    if not (rel_error_approx < 0.40 and mean_abs_z_err_approx < 0.20):
        raise AssertionError("compressed MPS quality fell outside expected MVP range")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
