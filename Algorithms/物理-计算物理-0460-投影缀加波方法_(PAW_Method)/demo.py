"""PAW (Projector Augmented-Wave) minimal runnable MVP.

This demo builds a 1D toy atom:
1) Solve pseudo-wave and all-electron reference states on a finite-difference grid.
2) Build PAW partial waves/projectors in an augmentation region.
3) Reconstruct an all-electron-like wavefunction from the pseudo wavefunction.
4) Report quantitative improvements and run automatic checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh_tridiagonal
from sklearn.metrics import mean_squared_error


@dataclass
class PawResult:
    x: np.ndarray
    dx: float
    potential_pseudo: np.ndarray
    potential_all_electron: np.ndarray
    psi_tilde: np.ndarray
    psi_ae_ref: np.ndarray
    psi_paw: np.ndarray
    psi_paw_tuned: np.ndarray
    projector_matrix: np.ndarray
    dual_overlap: np.ndarray
    coeffs_paw: np.ndarray
    coeffs_tuned: np.ndarray
    core_mask: np.ndarray
    metrics: pd.DataFrame
    energies_pseudo: np.ndarray
    energies_all_electron: np.ndarray


def normalize_l2(vec: np.ndarray, dx: float) -> np.ndarray:
    norm = np.sqrt(np.sum(vec * vec) * dx)
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Wavefunction norm is non-positive or non-finite.")
    return vec / norm


def align_phase(vec: np.ndarray) -> np.ndarray:
    idx = int(np.argmax(np.abs(vec)))
    return -vec if vec[idx] < 0.0 else vec


def align_to_reference(vec: np.ndarray, ref: np.ndarray, dx: float) -> np.ndarray:
    overlap = float(np.sum(vec * ref) * dx)
    return -vec if overlap < 0.0 else vec


def smooth_cutoff(x: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    return np.exp(-((np.abs(x) / radius) ** 8))


def solve_lowest_states(
    potential: np.ndarray, dx: float, num_states: int
) -> tuple[np.ndarray, np.ndarray]:
    """Solve low-lying states for 1D Schrödinger equation with Dirichlet boundaries."""
    if potential.ndim != 1:
        raise ValueError("potential must be a 1D array.")
    if num_states < 1:
        raise ValueError("num_states must be >= 1.")
    if potential.size < 5:
        raise ValueError("Grid is too small.")

    n = potential.size
    interior = n - 2
    if num_states > interior:
        raise ValueError("num_states exceeds interior grid size.")

    diag = np.full(interior, 1.0 / dx**2) + potential[1:-1]
    off = np.full(interior - 1, -0.5 / dx**2)

    eigvals, eigvecs = eigh_tridiagonal(
        diag,
        off,
        select="i",
        select_range=(0, num_states - 1),
        lapack_driver="stemr",
    )

    states = np.zeros((num_states, n), dtype=float)
    for i in range(num_states):
        states[i, 1:-1] = eigvecs[:, i]
        states[i] = normalize_l2(states[i], dx)
        states[i] = align_phase(states[i])
    return eigvals, states


def build_projectors(
    phi_tilde: np.ndarray, cutoff: np.ndarray, dx: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build dual projectors p_i from localized basis using PAW duality condition."""
    if phi_tilde.ndim != 2:
        raise ValueError("phi_tilde must be 2D: [n_partial, n_grid].")
    if cutoff.ndim != 1:
        raise ValueError("cutoff must be 1D.")
    if phi_tilde.shape[1] != cutoff.size:
        raise ValueError("Grid sizes of phi_tilde and cutoff do not match.")

    basis = phi_tilde * cutoff[None, :]
    overlap = np.einsum("ig,jg->ij", basis, phi_tilde) * dx
    if np.linalg.cond(overlap) > 1e10:
        raise ValueError("Projector overlap matrix is ill-conditioned.")

    transform = np.linalg.solve(overlap, np.eye(overlap.shape[0]))
    projectors = transform @ basis
    dual = np.einsum("ig,jg->ij", projectors, phi_tilde) * dx
    return projectors, dual


def projector_coefficients(
    projectors: np.ndarray, psi_tilde: np.ndarray, dx: float
) -> np.ndarray:
    return np.einsum("ig,g->i", projectors, psi_tilde) * dx


def reconstruct_paw(
    psi_tilde: np.ndarray, coeffs: np.ndarray, delta_phi: np.ndarray, dx: float
) -> np.ndarray:
    psi = psi_tilde + np.einsum("i,ig->g", coeffs, delta_phi)
    return normalize_l2(psi, dx)


def expectation_energy(psi: np.ndarray, potential: np.ndarray, dx: float) -> float:
    lap = np.zeros_like(psi)
    lap[1:-1] = (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / dx**2
    h_psi = -0.5 * lap + potential * psi
    return float(np.sum(psi * h_psi) * dx)


def refine_coefficients_torch(
    coeffs_init: np.ndarray,
    psi_tilde: np.ndarray,
    delta_phi: np.ndarray,
    psi_target: np.ndarray,
    core_mask: np.ndarray,
    steps: int = 400,
    lr: float = 0.03,
) -> np.ndarray:
    """Use Torch autograd to refine PAW coefficients with core-weighted loss."""
    if steps < 1:
        raise ValueError("steps must be >= 1.")

    dtype = torch.float64
    coeff_ref = torch.tensor(coeffs_init, dtype=dtype)
    coeffs = torch.tensor(coeffs_init, dtype=dtype, requires_grad=True)
    psi_t = torch.tensor(psi_tilde, dtype=dtype)
    delta_t = torch.tensor(delta_phi, dtype=dtype)
    target_t = torch.tensor(psi_target, dtype=dtype)
    core_t = torch.tensor(core_mask.astype(float), dtype=dtype)
    core_t = core_t / core_t.mean()

    optimizer = torch.optim.Adam([coeffs], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        pred = psi_t + torch.einsum("i,ig->g", coeffs, delta_t)
        loss_core = torch.mean(core_t * (pred - target_t) ** 2)
        loss_global = torch.mean((pred - target_t) ** 2)
        reg = torch.sum((coeffs - coeff_ref) ** 2)
        loss = loss_core + 0.2 * loss_global + 1e-3 * reg
        loss.backward()
        optimizer.step()
    return coeffs.detach().cpu().numpy()


def build_metrics_table(
    psi_ae_ref: np.ndarray,
    psi_tilde: np.ndarray,
    psi_paw: np.ndarray,
    psi_paw_tuned: np.ndarray,
    core_mask: np.ndarray,
    dx: float,
    v_ae: np.ndarray,
) -> pd.DataFrame:
    models = {
        "pseudo": psi_tilde,
        "paw": psi_paw,
        "paw_tuned": psi_paw_tuned,
    }
    rows: list[dict[str, float | str]] = []
    for name, psi in models.items():
        rmse_global = float(np.sqrt(mean_squared_error(psi_ae_ref, psi)))
        rmse_core = float(
            np.sqrt(mean_squared_error(psi_ae_ref[core_mask], psi[core_mask]))
        )
        overlap = float(np.sum(psi_ae_ref * psi) * dx)
        energy_ae = expectation_energy(psi, v_ae, dx)
        rows.append(
            {
                "model": name,
                "rmse_global": rmse_global,
                "rmse_core": rmse_core,
                "overlap_with_ae": overlap,
                "energy_on_ae_hamiltonian": energy_ae,
            }
        )
    return pd.DataFrame(rows)


def run_paw_mvp() -> PawResult:
    torch.manual_seed(0)

    n_grid = 801
    x = np.linspace(-8.0, 8.0, n_grid)
    dx = float(x[1] - x[0])

    v_pseudo = -1.0 / np.sqrt(x**2 + 0.45**2)
    v_all_electron = v_pseudo - 1.6 * np.exp(-(x / 0.11) ** 2)

    energies_pseudo, states_pseudo = solve_lowest_states(
        potential=v_pseudo, dx=dx, num_states=4
    )
    energies_all_electron, states_all_electron = solve_lowest_states(
        potential=v_all_electron, dx=dx, num_states=4
    )

    psi_tilde = states_pseudo[0].copy()
    psi_ae_ref = align_to_reference(states_all_electron[0].copy(), psi_tilde, dx)

    augmentation_radius = 1.2
    cutoff = smooth_cutoff(x, augmentation_radius)
    core_mask = np.abs(x) <= augmentation_radius

    partial_indices = np.array([0, 2], dtype=int)
    phi_tilde = states_pseudo[partial_indices].copy()
    phi_ae = phi_tilde + cutoff[None, :] * (
        states_all_electron[partial_indices] - states_pseudo[partial_indices]
    )
    for i in range(phi_ae.shape[0]):
        phi_ae[i] = align_to_reference(phi_ae[i], phi_tilde[i], dx)

    projectors, dual = build_projectors(phi_tilde=phi_tilde, cutoff=cutoff, dx=dx)
    coeffs_paw = projector_coefficients(projectors, psi_tilde, dx)
    delta_phi = phi_ae - phi_tilde
    psi_paw = reconstruct_paw(psi_tilde, coeffs_paw, delta_phi, dx)
    psi_paw = align_to_reference(psi_paw, psi_ae_ref, dx)

    coeffs_tuned = refine_coefficients_torch(
        coeffs_init=coeffs_paw,
        psi_tilde=psi_tilde,
        delta_phi=delta_phi,
        psi_target=psi_ae_ref,
        core_mask=core_mask,
    )
    psi_paw_tuned = reconstruct_paw(psi_tilde, coeffs_tuned, delta_phi, dx)
    psi_paw_tuned = align_to_reference(psi_paw_tuned, psi_ae_ref, dx)

    metrics = build_metrics_table(
        psi_ae_ref=psi_ae_ref,
        psi_tilde=psi_tilde,
        psi_paw=psi_paw,
        psi_paw_tuned=psi_paw_tuned,
        core_mask=core_mask,
        dx=dx,
        v_ae=v_all_electron,
    )

    return PawResult(
        x=x,
        dx=dx,
        potential_pseudo=v_pseudo,
        potential_all_electron=v_all_electron,
        psi_tilde=psi_tilde,
        psi_ae_ref=psi_ae_ref,
        psi_paw=psi_paw,
        psi_paw_tuned=psi_paw_tuned,
        projector_matrix=projectors,
        dual_overlap=dual,
        coeffs_paw=coeffs_paw,
        coeffs_tuned=coeffs_tuned,
        core_mask=core_mask,
        metrics=metrics,
        energies_pseudo=energies_pseudo,
        energies_all_electron=energies_all_electron,
    )


def run_checks(result: PawResult) -> None:
    if not np.all(np.isfinite(result.metrics.select_dtypes(include=[np.number]).values)):
        raise AssertionError("Metrics contain non-finite values.")

    dual_err = float(np.max(np.abs(result.dual_overlap - np.eye(result.dual_overlap.shape[0]))))
    if dual_err > 1e-8:
        raise AssertionError(f"Projector duality error too large: {dual_err:.3e}")

    metrics = result.metrics.set_index("model")
    rmse_core_pseudo = float(metrics.loc["pseudo", "rmse_core"])
    rmse_core_paw = float(metrics.loc["paw", "rmse_core"])
    rmse_core_tuned = float(metrics.loc["paw_tuned", "rmse_core"])
    overlap_pseudo = float(metrics.loc["pseudo", "overlap_with_ae"])
    overlap_paw = float(metrics.loc["paw", "overlap_with_ae"])
    overlap_tuned = float(metrics.loc["paw_tuned", "overlap_with_ae"])

    if rmse_core_paw >= rmse_core_pseudo:
        raise AssertionError("PAW reconstruction did not improve core RMSE.")
    if overlap_paw <= overlap_pseudo:
        raise AssertionError("PAW reconstruction did not improve overlap.")
    if rmse_core_tuned > rmse_core_paw + 5e-4:
        raise AssertionError("Torch-tuned coefficients degraded core RMSE too much.")
    if overlap_tuned < overlap_paw - 5e-4:
        raise AssertionError("Torch-tuned coefficients degraded overlap too much.")


def main() -> None:
    result = run_paw_mvp()
    run_checks(result)

    with np.printoptions(precision=6, suppress=True):
        print("=== PAW Method MVP (1D Toy Model) ===")
        print(f"Grid points: {result.x.size}, dx={result.dx:.5f}")
        print(
            "Lowest pseudo energies:",
            np.array2string(result.energies_pseudo, separator=", "),
        )
        print(
            "Lowest all-electron energies:",
            np.array2string(result.energies_all_electron, separator=", "),
        )
        print("PAW coefficients (projector):", np.array2string(result.coeffs_paw, separator=", "))
        print("PAW coefficients (torch tuned):", np.array2string(result.coeffs_tuned, separator=", "))
        print("Dual overlap matrix <p_i|phi_tilde_j>:")
        print(result.dual_overlap)

    print("\nMetrics against all-electron reference:")
    print(result.metrics.to_string(index=False))
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
