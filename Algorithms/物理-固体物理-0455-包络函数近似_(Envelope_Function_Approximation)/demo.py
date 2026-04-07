"""Envelope Function Approximation (EFA) MVP for a 1D semiconductor quantum well.

The script solves the BenDaniel-Duke effective-mass envelope equation:

    [-(hbar^2/2) d/dz (1/m*(z)) d/dz + V(z)] psi_n(z) = E_n psi_n(z)

for a finite square quantum well heterostructure with position-dependent mass.
It includes:
1) explicit finite-difference assembly of the tridiagonal Hamiltonian,
2) SciPy eigen-solve + PyTorch backend cross-check,
3) bound-state diagnostics and physical sanity checks,
4) E1-vs-(1/L^2) regression for width-scaling behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
import torch
from scipy import constants, linalg
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class EnvelopeConfig:
    """Model and numerics configuration for 1D EFA well states."""

    half_domain_nm: float = 25.0
    well_width_nm: float = 10.0
    barrier_height_ev: float = 0.30
    m_well_over_m0: float = 0.067
    m_barrier_over_m0: float = 0.092
    n_grid: int = 801  # including two Dirichlet boundaries
    n_report_states: int = 3
    scan_lowest_eig_count: int = 18


def validate_config(cfg: EnvelopeConfig) -> None:
    """Validate basic configuration constraints."""
    if cfg.n_grid < 11 or cfg.n_grid % 2 == 0:
        raise ValueError("n_grid must be odd and >= 11 for symmetric setup.")
    if cfg.well_width_nm <= 0.0:
        raise ValueError("well_width_nm must be positive.")
    if cfg.well_width_nm >= 2.0 * cfg.half_domain_nm:
        raise ValueError("well_width must be strictly smaller than full domain size.")
    if cfg.barrier_height_ev <= 0.0:
        raise ValueError("barrier_height_ev must be positive.")
    if cfg.m_well_over_m0 <= 0.0 or cfg.m_barrier_over_m0 <= 0.0:
        raise ValueError("effective masses must be positive.")


def build_grid(cfg: EnvelopeConfig) -> tuple[np.ndarray, float]:
    """Return z-grid (meters) and spacing."""
    z_m = np.linspace(
        -cfg.half_domain_nm * 1e-9,
        cfg.half_domain_nm * 1e-9,
        cfg.n_grid,
        dtype=np.float64,
    )
    dz = float(z_m[1] - z_m[0])
    return z_m, dz


def build_profiles(z_m: np.ndarray, cfg: EnvelopeConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build piecewise-constant potential and effective-mass profiles."""
    half_well_m = 0.5 * cfg.well_width_nm * 1e-9
    well_mask = np.abs(z_m) <= half_well_m

    v_ev = np.where(well_mask, 0.0, cfg.barrier_height_ev).astype(np.float64)
    m_eff = np.where(well_mask, cfg.m_well_over_m0, cfg.m_barrier_over_m0).astype(np.float64) * constants.m_e
    return v_ev, m_eff, well_mask


def build_bdd_tridiagonal(v_ev: np.ndarray, m_eff_kg: np.ndarray, dz: float) -> tuple[np.ndarray, np.ndarray]:
    """Assemble BenDaniel-Duke Hamiltonian as a tridiagonal matrix.

    Dirichlet boundaries are imposed by removing the two edge points.
    """
    v_j = v_ev * constants.e
    inv_m = 1.0 / m_eff_kg

    inv_left = 0.5 * (inv_m[1:-1] + inv_m[:-2])  # 1/m_{i-1/2}
    inv_right = 0.5 * (inv_m[1:-1] + inv_m[2:])  # 1/m_{i+1/2}

    pref = constants.hbar**2 / (2.0 * dz * dz)
    diag = v_j[1:-1] + pref * (inv_left + inv_right)
    off = -pref * inv_right[:-1]
    return diag.astype(np.float64), off.astype(np.float64)


def solve_tridiagonal(diag: np.ndarray, off: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve full eigen-spectrum and eigenvectors for the tridiagonal Hamiltonian."""
    evals_j, evecs_interior = linalg.eigh_tridiagonal(diag, off)
    return evals_j, evecs_interior


def normalize_state(z_m: np.ndarray, psi_interior: np.ndarray) -> np.ndarray:
    """Embed interior eigenvector into full grid and normalize in real space."""
    psi = np.zeros(z_m.size, dtype=np.float64)
    psi[1:-1] = psi_interior

    norm = float(np.sqrt(np.trapezoid(psi * psi, x=z_m)))
    if norm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    psi /= norm

    center_idx = int(np.argmin(np.abs(z_m)))
    if psi[center_idx] < 0.0:
        psi *= -1.0
    return psi


def residual_relative_norm(diag: np.ndarray, off: np.ndarray, psi_interior: np.ndarray, energy_j: float) -> float:
    """Relative infinity-norm residual for H psi = E psi on interior grid."""
    hpsi = diag * psi_interior
    hpsi[:-1] += off * psi_interior[1:]
    hpsi[1:] += off * psi_interior[:-1]
    resid = hpsi - energy_j * psi_interior
    denom = max(float(np.max(np.abs(energy_j * psi_interior))), 1e-30)
    return float(np.max(np.abs(resid)) / denom)


def infinite_well_energy_ev(n: int, cfg: EnvelopeConfig) -> float:
    """Reference E_n for infinite square well with well-region effective mass."""
    length_m = cfg.well_width_nm * 1e-9
    m_well = cfg.m_well_over_m0 * constants.m_e
    energy_j = (n * np.pi * constants.hbar) ** 2 / (2.0 * m_well * length_m * length_m)
    return float(energy_j / constants.e)


def max_scipy_torch_spectrum_diff(diag: np.ndarray, off: np.ndarray, evals_scipy_j: np.ndarray) -> float:
    """Cross-check SciPy spectrum using dense PyTorch eigensolver."""
    h_dense = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    evals_torch_j = torch.linalg.eigvalsh(torch.tensor(h_dense, dtype=torch.float64)).cpu().numpy()
    return float(np.max(np.abs(evals_torch_j - evals_scipy_j)))


def scan_ground_state_vs_width(cfg: EnvelopeConfig, widths_nm: np.ndarray) -> tuple[pd.DataFrame, float, float, float]:
    """Scan E1 vs well width and fit E1 ~ a*(1/L^2) + b."""
    rows: list[dict[str, float]] = []
    for width_nm in widths_nm:
        cfg_i = replace(cfg, well_width_nm=float(width_nm))
        z_i, dz_i = build_grid(cfg_i)
        v_ev_i, m_i, _ = build_profiles(z_i, cfg_i)
        diag_i, off_i = build_bdd_tridiagonal(v_ev_i, m_i, dz_i)
        evals_i_j = linalg.eigh_tridiagonal(
            diag_i,
            off_i,
            eigvals_only=True,
            select="i",
            select_range=(0, cfg_i.scan_lowest_eig_count - 1),
        )
        evals_i_ev = evals_i_j / constants.e
        bound = evals_i_ev[evals_i_ev < cfg_i.barrier_height_ev - 1e-6]
        if bound.size == 0:
            continue

        rows.append(
            {
                "well_width_nm": float(width_nm),
                "inv_width2_nm^-2": 1.0 / float(width_nm * width_nm),
                "E1_ev": float(bound[0]),
            }
        )

    df = pd.DataFrame(rows).sort_values("well_width_nm", ignore_index=True)
    reg = LinearRegression()
    x = df[["inv_width2_nm^-2"]].to_numpy()
    y = df["E1_ev"].to_numpy()
    reg.fit(x, y)
    slope = float(reg.coef_[0])
    intercept = float(reg.intercept_)
    r2 = float(reg.score(x, y))
    return df, slope, intercept, r2


def main() -> None:
    cfg = EnvelopeConfig()
    validate_config(cfg)

    z_m, dz = build_grid(cfg)
    z_nm = z_m * 1e9
    v_ev, m_eff_kg, well_mask = build_profiles(z_m, cfg)
    diag, off = build_bdd_tridiagonal(v_ev, m_eff_kg, dz)

    evals_j, evecs_interior = solve_tridiagonal(diag, off)
    evals_ev = evals_j / constants.e
    bound_indices = np.where(evals_ev < cfg.barrier_height_ev - 1e-6)[0]

    if bound_indices.size == 0:
        raise RuntimeError("No bound states found under current configuration.")

    report_indices = bound_indices[: cfg.n_report_states]
    state_rows: list[dict[str, float]] = []
    psi_list: list[np.ndarray] = []

    for n_level, eig_idx in enumerate(report_indices, start=1):
        psi = normalize_state(z_m, evecs_interior[:, eig_idx])
        psi_list.append(psi)
        prob_in_well = float(np.trapezoid(psi[well_mask] ** 2, x=z_m[well_mask]))
        res_rel = residual_relative_norm(diag, off, evecs_interior[:, eig_idx], evals_j[eig_idx])
        e_inf = infinite_well_energy_ev(n_level, cfg)
        state_rows.append(
            {
                "state_n": float(n_level),
                "E_bound_ev": float(evals_ev[eig_idx]),
                "E_infinite_well_ev": e_inf,
                "E_bound / E_inf": float(evals_ev[eig_idx] / e_inf),
                "P_in_well": prob_in_well,
                "relative_residual": res_rel,
            }
        )

    state_df = pd.DataFrame(state_rows)
    max_backend_diff_j = max_scipy_torch_spectrum_diff(diag, off, evals_j)

    overlap_offdiag = 0.0
    if len(psi_list) > 1:
        overlap = np.zeros((len(psi_list), len(psi_list)), dtype=np.float64)
        for i in range(len(psi_list)):
            for j in range(len(psi_list)):
                overlap[i, j] = np.trapezoid(psi_list[i] * psi_list[j], x=z_m)
        overlap_offdiag = float(np.max(np.abs(overlap - np.eye(len(psi_list)))))

    scan_widths = np.array([6.0, 7.0, 8.0, 10.0, 12.0, 14.0], dtype=np.float64)
    scan_df, slope, intercept, r2 = scan_ground_state_vs_width(cfg, scan_widths)

    sample_points_nm = np.array([-12.5, -5.0, -2.5, 0.0, 2.5, 5.0, 12.5], dtype=np.float64)
    profile_df = pd.DataFrame(
        {
            "z_nm": sample_points_nm,
            "V_ev": np.interp(sample_points_nm, z_nm, v_ev),
            "m_eff_over_m0": np.interp(sample_points_nm, z_nm, m_eff_kg / constants.m_e),
        }
    )

    print("=== Envelope Function Approximation MVP (1D Quantum Well) ===")
    print("\nConfig:")
    print(pd.Series(asdict(cfg)).to_string())
    print(f"dz = {dz * 1e9:.5f} nm")
    print(f"grid interior size = {diag.size}")
    print(f"bound-state count (E < V0): {bound_indices.size}")

    print("\nProfile sample:")
    print(profile_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    print("\nLowest bound states:")
    print(state_df.to_string(index=False, float_format=lambda x: f"{x: .8f}"))

    print("\nBackend and quality checks:")
    print(f"max|E_scipy - E_torch| (J) = {max_backend_diff_j:.3e}")
    print(f"max orthonormality error   = {overlap_offdiag:.3e}")

    print("\nWidth scan and scaling fit: E1 ~ slope*(1/L^2) + intercept")
    print(scan_df.to_string(index=False, float_format=lambda x: f"{x: .8f}"))
    print(f"slope (eV*nm^2) = {slope:.8f}")
    print(f"intercept (eV)  = {intercept:.8f}")
    print(f"R^2             = {r2:.8f}")

    assert bound_indices.size >= 2, "Expected at least two bound states."
    assert max_backend_diff_j < 1e-9 * constants.e, "SciPy and PyTorch spectra disagree too much."
    assert float(state_df["relative_residual"].max()) < 2e-10, "Eigen residual is too large."
    assert float(state_df.loc[0, "E_bound_ev"]) > 0.0, "Ground-state energy should be positive."
    assert float(state_df.loc[0, "E_bound_ev"]) < cfg.barrier_height_ev, "Ground state must be below barrier."
    assert bool(np.all(state_df["E_bound / E_inf"].to_numpy() < 1.0)), "Finite-well bound states should be below infinite-well levels."
    assert float(state_df.loc[0, "P_in_well"]) > 0.80, "Ground state should be well localized in well."
    assert overlap_offdiag < 5e-3, "Reported states are not sufficiently orthonormal."
    assert slope > 0.0, "E1 should increase with 1/L^2."
    assert r2 > 0.97, "Width-scaling linearity is weaker than expected for this setup."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
