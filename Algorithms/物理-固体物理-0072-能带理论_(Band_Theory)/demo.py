"""Band Theory MVP: 1D periodic potential via plane-wave Bloch Hamiltonian.

Model (dimensionless units, equivalent to hbar^2/(2m)=1):
    H = -d^2/dx^2 + V(x)
    V(x) = 2*V1*cos(G0*x) + 2*V2*cos(2*G0*x),  G0 = 2*pi/a

In reciprocal basis |k+G_n>, the Hamiltonian is:
    H_nm(k) = (k + G_n)^2 * delta_nm + V_{n-m}
where V_{±1}=V1, V_{±2}=V2.

This script computes E_n(k), estimates the first band gap at BZ boundary,
and validates key band-theory properties with automated checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh
from sklearn.metrics import mean_squared_error


@dataclass(frozen=True)
class BandConfig:
    lattice_constant: float = 1.0
    v1: float = 0.15
    v2: float = 0.04
    n_harmonics: int = 4
    n_kpoints: int = 241
    n_bands_report: int = 4
    center_window_fraction: float = 0.35
    fit_window_fraction: float = 0.20
    torch_steps: int = 600
    torch_lr: float = 0.05


@dataclass(frozen=True)
class BandResult:
    config: BandConfig
    k_grid: np.ndarray
    bands: np.ndarray
    band_gap_zone_boundary: float
    predicted_gap_2v1: float
    gap_relative_error: float
    center_rmse_vs_free: float
    inversion_symmetry_error: float
    alpha_quadratic_fit: float
    e0_quadratic_fit: float
    effective_mass_ratio: float
    torch_fit_loss: float


def validate_config(config: BandConfig) -> None:
    if config.lattice_constant <= 0.0:
        raise ValueError("lattice_constant must be positive.")
    if config.n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1.")
    if config.n_kpoints < 5:
        raise ValueError("n_kpoints must be >= 5.")
    if config.n_kpoints % 2 == 0:
        raise ValueError("n_kpoints should be odd so k=0 is sampled exactly.")
    if config.n_bands_report < 2:
        raise ValueError("n_bands_report must be >= 2 for a band-gap calculation.")

    basis_dim = 2 * config.n_harmonics + 1
    if config.n_bands_report > basis_dim:
        raise ValueError("n_bands_report cannot exceed reciprocal basis dimension.")
    if not (0.0 < config.center_window_fraction <= 1.0):
        raise ValueError("center_window_fraction must be in (0, 1].")
    if not (0.0 < config.fit_window_fraction <= config.center_window_fraction):
        raise ValueError("fit_window_fraction must be in (0, center_window_fraction].")
    if config.torch_steps <= 0:
        raise ValueError("torch_steps must be positive.")
    if config.torch_lr <= 0.0:
        raise ValueError("torch_lr must be positive.")


def build_fourier_coefficients(v1: float, v2: float) -> dict[int, float]:
    """Fourier coefficients in integer G-index units.

    q-index meaning: q=1 corresponds to +G0, q=2 corresponds to +2G0.
    """
    return {
        -2: float(v2),
        -1: float(v1),
        0: 0.0,
        1: float(v1),
        2: float(v2),
    }


def build_bloch_hamiltonian(
    k_value: float,
    g0: float,
    n_indices: np.ndarray,
    coeffs: dict[int, float],
) -> np.ndarray:
    """Construct Hermitian Bloch Hamiltonian H(k) in plane-wave basis."""
    if n_indices.ndim != 1:
        raise ValueError("n_indices must be a 1D array.")

    kinetic = (k_value + g0 * n_indices.astype(np.float64)) ** 2
    hamiltonian = np.diag(kinetic)

    for i, ni in enumerate(n_indices):
        for j, nj in enumerate(n_indices):
            hamiltonian[i, j] += coeffs.get(int(ni - nj), 0.0)

    # Numerical symmetry guard.
    return 0.5 * (hamiltonian + hamiltonian.T)


def fit_effective_mass_torch(
    k_values: np.ndarray,
    band_values: np.ndarray,
    steps: int,
    lr: float,
) -> tuple[float, float, float]:
    """Fit E(k) = E0 + alpha*k^2 with torch autograd, return (E0, alpha, loss)."""
    if k_values.ndim != 1 or band_values.ndim != 1:
        raise ValueError("k_values and band_values must be 1D arrays.")
    if k_values.size != band_values.size:
        raise ValueError("k_values and band_values must have the same length.")
    if k_values.size < 5:
        raise ValueError("Need at least 5 samples for robust quadratic fit.")

    torch.manual_seed(2026)
    dtype = torch.float64

    k_t = torch.tensor(k_values, dtype=dtype)
    e_t = torch.tensor(band_values, dtype=dtype)

    e0 = torch.nn.Parameter(torch.tensor(float(np.min(band_values)), dtype=dtype))
    raw_alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype))

    optimizer = torch.optim.Adam([e0, raw_alpha], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        alpha = torch.nn.functional.softplus(raw_alpha) + 1e-10
        pred = e0 + alpha * (k_t**2)
        loss = torch.mean((pred - e_t) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        alpha = torch.nn.functional.softplus(raw_alpha) + 1e-10
        pred = e0 + alpha * (k_t**2)
        loss = torch.mean((pred - e_t) ** 2)

    return float(e0.item()), float(alpha.item()), float(loss.item())


def solve_band_structure(config: BandConfig) -> BandResult:
    validate_config(config)

    a = config.lattice_constant
    g0 = 2.0 * np.pi / a
    k_grid = np.linspace(-np.pi / a, np.pi / a, config.n_kpoints, dtype=np.float64)
    n_indices = np.arange(-config.n_harmonics, config.n_harmonics + 1, dtype=np.int64)
    coeffs = build_fourier_coefficients(config.v1, config.v2)

    bands = np.empty((config.n_kpoints, config.n_bands_report), dtype=np.float64)

    for ik, kval in enumerate(k_grid):
        hk = build_bloch_hamiltonian(kval, g0, n_indices, coeffs)
        eigvals = eigh(hk, eigvals_only=True, check_finite=True)
        bands[ik, :] = eigvals[: config.n_bands_report]

    edge_idx = int(np.argmin(np.abs(k_grid - np.pi / a)))
    band_gap = float(bands[edge_idx, 1] - bands[edge_idx, 0])

    predicted_gap = float(2.0 * abs(config.v1))
    gap_relative_error = (
        float(abs(band_gap - predicted_gap) / predicted_gap)
        if predicted_gap > 0.0
        else 0.0
    )

    symmetry_error = float(np.max(np.abs(bands - bands[::-1, :])))

    center_k_cut = config.center_window_fraction * (np.pi / a)
    center_mask = np.abs(k_grid) <= center_k_cut
    free_center = k_grid[center_mask] ** 2
    center_rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true=free_center,
                y_pred=bands[center_mask, 0],
            )
        )
    )

    fit_k_cut = config.fit_window_fraction * (np.pi / a)
    fit_mask = np.abs(k_grid) <= fit_k_cut
    e0_fit, alpha_fit, fit_loss = fit_effective_mass_torch(
        k_values=k_grid[fit_mask],
        band_values=bands[fit_mask, 0],
        steps=config.torch_steps,
        lr=config.torch_lr,
    )
    effective_mass_ratio = float(1.0 / alpha_fit)

    return BandResult(
        config=config,
        k_grid=k_grid,
        bands=bands,
        band_gap_zone_boundary=band_gap,
        predicted_gap_2v1=predicted_gap,
        gap_relative_error=gap_relative_error,
        center_rmse_vs_free=center_rmse,
        inversion_symmetry_error=symmetry_error,
        alpha_quadratic_fit=alpha_fit,
        e0_quadratic_fit=e0_fit,
        effective_mass_ratio=effective_mass_ratio,
        torch_fit_loss=fit_loss,
    )


def build_report_tables(result: BandResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    k = result.k_grid
    bands = result.bands
    n_bands = bands.shape[1]

    sample_indices = [0, len(k) // 4, len(k) // 2, 3 * len(k) // 4, len(k) - 1]
    sample_rows: list[dict[str, float]] = []
    for idx in sample_indices:
        row: dict[str, float] = {
            "k": float(k[idx]),
        }
        for band_idx in range(n_bands):
            row[f"E{band_idx + 1}"] = float(bands[idx, band_idx])
        sample_rows.append(row)

    bands_table = pd.DataFrame(sample_rows)

    metrics_table = pd.DataFrame(
        [
            {"metric": "band_gap_zone_boundary", "value": result.band_gap_zone_boundary},
            {"metric": "predicted_gap_2|V1|", "value": result.predicted_gap_2v1},
            {"metric": "gap_relative_error", "value": result.gap_relative_error},
            {"metric": "center_rmse_vs_free", "value": result.center_rmse_vs_free},
            {
                "metric": "inversion_symmetry_max_error",
                "value": result.inversion_symmetry_error,
            },
            {"metric": "alpha_quadratic_fit", "value": result.alpha_quadratic_fit},
            {
                "metric": "effective_mass_ratio_m*/m_free",
                "value": result.effective_mass_ratio,
            },
            {"metric": "torch_fit_loss", "value": result.torch_fit_loss},
        ]
    )
    return bands_table, metrics_table


def run_checks(result: BandResult) -> None:
    if not np.isfinite(result.bands).all():
        raise AssertionError("Band energies contain non-finite values.")

    if result.band_gap_zone_boundary <= 0.05:
        raise AssertionError(
            f"Band gap is too small or non-positive: {result.band_gap_zone_boundary:.6f}"
        )

    if result.inversion_symmetry_error > 1e-10:
        raise AssertionError(
            "Inversion symmetry violated: "
            f"max |E(k)-E(-k)| = {result.inversion_symmetry_error:.3e}"
        )

    if result.gap_relative_error > 0.25:
        raise AssertionError(
            "Zone-boundary gap deviates too much from weak-potential estimate: "
            f"rel_error={result.gap_relative_error:.3f}"
        )

    if result.center_rmse_vs_free > 0.20:
        raise AssertionError(
            "First band deviates too much from free-electron parabola near k=0: "
            f"rmse={result.center_rmse_vs_free:.3e}"
        )

    if result.torch_fit_loss > 2e-4:
        raise AssertionError(
            f"Torch quadratic fit did not converge sufficiently: loss={result.torch_fit_loss:.3e}"
        )

    if not (0.3 <= result.effective_mass_ratio <= 3.0):
        raise AssertionError(
            "Effective mass ratio is outside sanity range: "
            f"m*/m_free={result.effective_mass_ratio:.3f}"
        )


def main() -> None:
    config = BandConfig()
    result = solve_band_structure(config)
    run_checks(result)

    bands_table, metrics_table = build_report_tables(result)

    print("Band Theory MVP (1D periodic potential, plane-wave Bloch Hamiltonian)")
    print(
        "config:",
        {
            "a": config.lattice_constant,
            "v1": config.v1,
            "v2": config.v2,
            "n_harmonics": config.n_harmonics,
            "n_kpoints": config.n_kpoints,
            "n_bands_report": config.n_bands_report,
        },
    )
    print()
    print("Sampled bands:")
    print(bands_table.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()
    print("Metrics:")
    print(metrics_table.to_string(index=False, float_format=lambda v: f"{v: .6e}"))
    print("All checks passed.")


if __name__ == "__main__":
    main()
