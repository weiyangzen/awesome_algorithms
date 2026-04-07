"""One-dimensional nearest-neighbor tight-binding model MVP.

This script is intentionally compact but complete:
- builds the real-space Hamiltonian with periodic boundary conditions,
- compares numerical eigenvalues with analytic E(k),
- cross-checks SciPy vs PyTorch eigensolvers,
- estimates effective mass near k=0 via scikit-learn regression.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class TightBindingConfig:
    """Model parameters for a 1D monoatomic tight-binding chain."""

    n_sites: int = 64
    lattice_constant: float = 1.0
    onsite_energy: float = 0.5
    hopping: float = 1.2
    fit_window: float = 0.25  # Use |k| <= fit_window for effective-mass fit.


def build_real_space_hamiltonian(cfg: TightBindingConfig) -> np.ndarray:
    """Construct the N x N real-space Hamiltonian with periodic boundaries."""
    n = cfg.n_sites
    h = np.eye(n, dtype=np.float64) * cfg.onsite_energy
    for i in range(n - 1):
        h[i, i + 1] = -cfg.hopping
        h[i + 1, i] = -cfg.hopping
    h[0, n - 1] = -cfg.hopping
    h[n - 1, 0] = -cfg.hopping
    return h


def allowed_k_points(cfg: TightBindingConfig) -> np.ndarray:
    """Discrete k points for a periodic chain."""
    n = np.arange(cfg.n_sites, dtype=np.float64) - (cfg.n_sites // 2)
    return 2.0 * np.pi * n / (cfg.n_sites * cfg.lattice_constant)


def analytic_band_energy(k_points: np.ndarray, cfg: TightBindingConfig) -> np.ndarray:
    """Dispersion relation E(k)=eps0-2t cos(ka)."""
    return cfg.onsite_energy - 2.0 * cfg.hopping * np.cos(k_points * cfg.lattice_constant)


def compute_band_table(k_points: np.ndarray, energies: np.ndarray, cfg: TightBindingConfig) -> pd.DataFrame:
    """Build a compact band table with first and second k-derivatives."""
    velocity = 2.0 * cfg.hopping * cfg.lattice_constant * np.sin(k_points * cfg.lattice_constant)
    curvature = 2.0 * cfg.hopping * (cfg.lattice_constant**2) * np.cos(k_points * cfg.lattice_constant)
    table = pd.DataFrame(
        {
            "k": k_points,
            "energy": energies,
            "group_velocity": velocity,
            "curvature": curvature,
        }
    )
    return table.sort_values("k", ignore_index=True)


def fit_effective_mass(k_points: np.ndarray, energies: np.ndarray, cfg: TightBindingConfig) -> dict[str, float]:
    """Fit E ~ E0 + alpha*k^2 near k=0 and infer effective mass m*=1/(2*alpha)."""
    mask = np.abs(k_points) <= cfg.fit_window
    x = (k_points[mask] ** 2).reshape(-1, 1)
    y = energies[mask]

    reg = LinearRegression()
    reg.fit(x, y)

    alpha = float(reg.coef_[0])
    fitted_e0 = float(reg.intercept_)
    fitted_m = 1.0 / (2.0 * alpha)

    theoretical_alpha = cfg.hopping * (cfg.lattice_constant**2)
    theoretical_m = 1.0 / (2.0 * theoretical_alpha)
    relative_error = abs(fitted_m - theoretical_m) / theoretical_m

    return {
        "fitted_e0": fitted_e0,
        "fitted_alpha": alpha,
        "fitted_mass": fitted_m,
        "theoretical_mass": theoretical_m,
        "relative_mass_error": relative_error,
    }


def main() -> None:
    cfg = TightBindingConfig()

    h_real = build_real_space_hamiltonian(cfg)
    evals_scipy = np.sort(eigh(h_real, eigvals_only=True))

    k_points = allowed_k_points(cfg)
    evals_analytic = np.sort(analytic_band_energy(k_points, cfg))

    h_torch = torch.tensor(h_real, dtype=torch.float64)
    evals_torch = np.sort(torch.linalg.eigvalsh(h_torch).cpu().numpy())

    max_real_vs_analytic = float(np.max(np.abs(evals_scipy - evals_analytic)))
    max_scipy_vs_torch = float(np.max(np.abs(evals_scipy - evals_torch)))

    band_table = compute_band_table(k_points, analytic_band_energy(k_points, cfg), cfg)
    fit_stats = fit_effective_mass(k_points, analytic_band_energy(k_points, cfg), cfg)

    print("=== Tight-Binding 1D MVP ===")
    print("\nConfig:")
    print(pd.Series(asdict(cfg)).to_string())

    print("\nBand sample (first 8 rows sorted by k):")
    print(band_table.head(8).to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    print("\nConsistency metrics:")
    print(f"max|E_real - E_analytic| = {max_real_vs_analytic:.3e}")
    print(f"max|E_scipy - E_torch|   = {max_scipy_vs_torch:.3e}")

    print("\nEffective-mass fit near k=0:")
    print(f"fitted E0         = {fit_stats['fitted_e0']:.6f}")
    print(f"fitted alpha      = {fit_stats['fitted_alpha']:.6f}")
    print(f"fitted mass       = {fit_stats['fitted_mass']:.6f}")
    print(f"theoretical mass  = {fit_stats['theoretical_mass']:.6f}")
    print(f"relative mass err = {fit_stats['relative_mass_error']:.3%}")

    assert max_real_vs_analytic < 1e-10, "Real-space and analytic spectra mismatch is too large."
    assert max_scipy_vs_torch < 1e-10, "SciPy and PyTorch eigensolvers disagree too much."
    assert fit_stats["relative_mass_error"] < 0.10, "Effective-mass fit error is larger than expected."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
