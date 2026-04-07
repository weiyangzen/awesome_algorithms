"""One-dimensional Nearly Free Electron Model (NFEM) MVP.

This script implements a compact, explicit NFEM workflow in reduced units:
- constructs a truncated plane-wave Hamiltonian for a weak periodic potential,
- computes bands in the first Brillouin zone via SciPy eigensolver,
- cross-checks eigenvalues with PyTorch,
- compares full numerical bands with 2x2 anti-crossing formula near zone edge,
- verifies second-order perturbation away from degeneracy,
- fits the first band gap versus potential Fourier amplitude using scikit-learn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class NFEMConfig:
    """Parameters for a 1D NFEM with V(x)=2*V1*cos(Gx)."""

    lattice_constant: float = 1.0
    kinetic_prefactor: float = 1.0  # Reduced units: E_free(k)=kinetic_prefactor*k^2
    fourier_potential_v1: float = 0.08
    plane_wave_cutoff: int = 3  # Basis m in [-M, ..., M]
    n_k: int = 241  # Sample points in first BZ
    two_level_window_ratio: float = 0.22  # Window near +G/2 for 2x2 comparison
    perturbation_exclusion_ratio: float = 0.20  # Exclude edge neighborhood for PT


def reciprocal_lattice_vector(cfg: NFEMConfig) -> float:
    """Return G = 2*pi/a for a 1D lattice."""
    return 2.0 * np.pi / cfg.lattice_constant


def k_grid_first_bz(cfg: NFEMConfig) -> np.ndarray:
    """Uniform k-grid in first Brillouin zone [-G/2, G/2]."""
    g = reciprocal_lattice_vector(cfg)
    return np.linspace(-0.5 * g, 0.5 * g, cfg.n_k, dtype=np.float64)


def plane_wave_indices(cfg: NFEMConfig) -> np.ndarray:
    """Plane-wave index set m in [-M, ..., M]."""
    return np.arange(-cfg.plane_wave_cutoff, cfg.plane_wave_cutoff + 1, dtype=np.int64)


def build_hamiltonian_k(k: float, cfg: NFEMConfig) -> np.ndarray:
    """Build truncated plane-wave Hamiltonian H(k).

    Basis: |k+mG>, m in [-M,...,M]
    Diagonal: E_free(k+mG)
    Off-diagonal nearest in m: V1 (from Fourier components at +-G).
    """
    m = plane_wave_indices(cfg)
    g = reciprocal_lattice_vector(cfg)
    q = k + m * g

    n = m.size
    h = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(h, cfg.kinetic_prefactor * (q**2))

    for i in range(n - 1):
        h[i, i + 1] = cfg.fourier_potential_v1
        h[i + 1, i] = cfg.fourier_potential_v1

    return h


def solve_bands_scipy(cfg: NFEMConfig) -> tuple[np.ndarray, np.ndarray]:
    """Solve full NFEM bands with SciPy on the k-grid."""
    ks = k_grid_first_bz(cfg)
    n_basis = 2 * cfg.plane_wave_cutoff + 1
    bands = np.empty((ks.size, n_basis), dtype=np.float64)

    for i, k in enumerate(ks):
        bands[i] = eigh(build_hamiltonian_k(float(k), cfg), eigvals_only=True)

    return ks, bands


def max_scipy_torch_eigen_diff(cfg: NFEMConfig, ks: np.ndarray, bands_scipy: np.ndarray) -> float:
    """Compute max absolute eigenspectrum difference between SciPy and PyTorch."""
    max_diff = 0.0
    for i, k in enumerate(ks):
        h_torch = torch.tensor(build_hamiltonian_k(float(k), cfg), dtype=torch.float64)
        evals_torch = np.sort(torch.linalg.eigvalsh(h_torch).cpu().numpy())
        diff = float(np.max(np.abs(evals_torch - bands_scipy[i])))
        if diff > max_diff:
            max_diff = diff
    return max_diff


def two_level_near_boundary(k: float, cfg: NFEMConfig) -> tuple[float, float]:
    """Two-level analytic energies for coupling between |k> and |k-G|.

    E± = (Ek + Ekg)/2 ± sqrt(((Ek - Ekg)/2)^2 + V1^2)
    """
    g = reciprocal_lattice_vector(cfg)
    ek = cfg.kinetic_prefactor * (k**2)
    ekg = cfg.kinetic_prefactor * ((k - g) ** 2)

    center = 0.5 * (ek + ekg)
    split = np.sqrt((0.5 * (ek - ekg)) ** 2 + cfg.fourier_potential_v1**2)
    return float(center - split), float(center + split)


def compare_two_level_model(ks: np.ndarray, bands: np.ndarray, cfg: NFEMConfig) -> tuple[float, np.ndarray]:
    """Compare full numerical first two bands with 2x2 formula near +G/2."""
    g = reciprocal_lattice_vector(cfg)
    window = cfg.two_level_window_ratio * g
    mask = ks >= (0.5 * g - window)

    analytic = np.empty((int(np.sum(mask)), 2), dtype=np.float64)
    idx = 0
    for k in ks[mask]:
        e_minus, e_plus = two_level_near_boundary(float(k), cfg)
        analytic[idx, 0] = e_minus
        analytic[idx, 1] = e_plus
        idx += 1

    err = float(np.max(np.abs(bands[mask, :2] - analytic)))
    return err, mask


def perturbative_first_band(ks: np.ndarray, cfg: NFEMConfig) -> np.ndarray:
    """Second-order perturbative correction away from zone-edge degeneracy."""
    g = reciprocal_lattice_vector(cfg)
    e0 = cfg.kinetic_prefactor * (ks**2)
    e_plus = cfg.kinetic_prefactor * ((ks + g) ** 2)
    e_minus = cfg.kinetic_prefactor * ((ks - g) ** 2)

    corr = (cfg.fourier_potential_v1**2) / (e0 - e_plus) + (cfg.fourier_potential_v1**2) / (e0 - e_minus)
    return e0 + corr


def compare_perturbation(ks: np.ndarray, bands: np.ndarray, cfg: NFEMConfig) -> tuple[float, np.ndarray]:
    """Compare full first band with perturbation result away from |k|=G/2."""
    g = reciprocal_lattice_vector(cfg)
    margin = cfg.perturbation_exclusion_ratio * g
    boundary_distance = np.abs(np.abs(ks) - 0.5 * g)
    mask = boundary_distance >= margin

    e_pt = perturbative_first_band(ks[mask], cfg)
    err = float(np.max(np.abs(bands[mask, 0] - e_pt)))
    return err, mask


def first_direct_gap(ks: np.ndarray, bands: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return k location and value of min direct gap between band1 and band2."""
    gaps = bands[:, 1] - bands[:, 0]
    idx = int(np.argmin(gaps))
    return float(ks[idx]), float(gaps[idx]), gaps


def fit_gap_vs_potential(cfg: NFEMConfig, potential_values: np.ndarray) -> tuple[pd.DataFrame, float, float, float]:
    """Fit Egap ~ slope * V1 + intercept for weak-potential regime."""
    rows: list[dict[str, float]] = []
    for v1 in potential_values:
        cfg_i = replace(cfg, fourier_potential_v1=float(v1))
        ks_i, bands_i = solve_bands_scipy(cfg_i)
        k_gap_i, gap_i, _ = first_direct_gap(ks_i, bands_i)
        rows.append({"v1": float(v1), "k_gap": k_gap_i, "gap": gap_i})

    df = pd.DataFrame(rows)

    reg = LinearRegression()
    x = df[["v1"]].to_numpy()
    y = df["gap"].to_numpy()
    reg.fit(x, y)

    slope = float(reg.coef_[0])
    intercept = float(reg.intercept_)
    r2 = float(reg.score(x, y))
    return df, slope, intercept, r2


def build_band_sample_table(ks: np.ndarray, bands: np.ndarray) -> pd.DataFrame:
    """Create a compact sample table from first two bands."""
    n = ks.size
    sample_idx = np.array([0, n // 8, n // 4, n // 2, 3 * n // 4, 7 * n // 8, n - 1], dtype=int)
    sample_idx = np.unique(sample_idx)

    df = pd.DataFrame(
        {
            "k": ks[sample_idx],
            "band1": bands[sample_idx, 0],
            "band2": bands[sample_idx, 1],
            "gap12": bands[sample_idx, 1] - bands[sample_idx, 0],
        }
    )
    return df


def main() -> None:
    cfg = NFEMConfig()

    ks, bands = solve_bands_scipy(cfg)
    g = reciprocal_lattice_vector(cfg)

    backend_diff = max_scipy_torch_eigen_diff(cfg, ks, bands)
    two_level_err, two_level_mask = compare_two_level_model(ks, bands, cfg)
    perturb_err, perturb_mask = compare_perturbation(ks, bands, cfg)

    k_gap, min_gap, gaps = first_direct_gap(ks, bands)
    expected_gap = 2.0 * abs(cfg.fourier_potential_v1)
    gap_rel_err = abs(min_gap - expected_gap) / expected_gap

    potential_values = np.linspace(0.02, 0.12, 6)
    gap_scan_df, slope, intercept, r2 = fit_gap_vs_potential(cfg, potential_values)

    band_sample_df = build_band_sample_table(ks, bands)

    print("=== Nearly Free Electron Model (1D) MVP ===")
    print("\nConfig:")
    print(pd.Series(asdict(cfg)).to_string())

    print("\nBand sample (first two bands):")
    print(band_sample_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    print("\nCore checks:")
    print(f"max|E_scipy - E_torch|                  = {backend_diff:.3e}")
    print(f"2x2 model max error (near +G/2)         = {two_level_err:.3e}")
    print(f"2nd-order PT max error (away from edge) = {perturb_err:.3e}")
    print(f"min direct gap Egap                      = {min_gap:.6f}")
    print(f"expected weak-potential 2|V1|            = {expected_gap:.6f}")
    print(f"relative gap error                       = {gap_rel_err:.3%}")
    print(f"k at min gap                             = {k_gap:.6f}")
    print(f"Brillouin boundary |G/2|                 = {0.5 * g:.6f}")

    print("\nGap-vs-potential linear fit (Egap ~ slope*V1 + intercept):")
    print(gap_scan_df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    print(f"slope     = {slope:.6f}")
    print(f"intercept = {intercept:.6f}")
    print(f"R^2       = {r2:.6f}")

    assert backend_diff < 1e-10, "SciPy and PyTorch eigensolvers disagree too much."
    assert two_level_err < 6e-3, "Two-level anti-crossing approximation mismatch is too large."
    assert perturb_err < 2e-3, "Second-order perturbation mismatch is too large away from edge."
    assert gap_rel_err < 0.03, "First band gap deviates too much from 2|V1| scaling."
    assert abs(abs(k_gap) - 0.5 * g) <= (g / (cfg.n_k - 1)), "Gap minimum is not at the zone boundary grid point."
    assert abs(slope - 2.0) < 0.05, "Fitted Egap-V1 slope is not close to 2."
    assert abs(intercept) < 0.01, "Fitted Egap-V1 intercept should stay near zero."
    assert r2 > 0.999, "Gap-vs-potential linear relation is weaker than expected."
    assert int(np.sum(two_level_mask)) > 0, "Two-level window produced no samples."
    assert int(np.sum(perturb_mask)) > 0, "Perturbative mask produced no samples."
    assert float(np.min(gaps)) > 0.0, "Band gap should remain positive for this weak potential setup."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
