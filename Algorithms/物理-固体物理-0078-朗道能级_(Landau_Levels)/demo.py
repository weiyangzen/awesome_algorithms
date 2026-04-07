"""Minimal runnable MVP for Landau levels (PHYS-0078).

We solve the Landau-gauge 1D reduced Hamiltonian with finite differences:

H_k = p_x^2 / (2m) + (hbar*k - q*B*x)^2 / (2m)

For each k_y = k, this is a shifted harmonic oscillator. The spectrum should be
independent of k and approach analytic Landau levels:

E_n = hbar * omega_c * (n + 1/2), omega_c = |q|B/m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal


@dataclass(frozen=True)
class LandauConfig:
    """Configuration for finite-difference Landau-level simulation."""

    hbar: float = 1.0
    mass: float = 1.0
    charge_abs: float = 1.0
    magnetic_field: float = 1.0
    x_min: float = -8.0
    x_max: float = 8.0
    n_grid: int = 801
    n_levels: int = 5
    k_values: tuple[float, ...] = (-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5)

    def validate(self) -> None:
        if self.hbar <= 0 or self.mass <= 0:
            raise ValueError("hbar and mass must be positive")
        if self.charge_abs <= 0 or self.magnetic_field <= 0:
            raise ValueError("charge_abs and magnetic_field must be positive")
        if self.n_grid < 101:
            raise ValueError("n_grid should be >= 101 for stable discretization")
        if self.n_levels < 1 or self.n_levels >= self.n_grid - 2:
            raise ValueError("n_levels must be in [1, n_grid-3]")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")


def build_tridiagonal_kinetic(n: int, dx: float, hbar: float, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Return diagonal/off-diagonal arrays for -hbar^2/(2m) d^2/dx^2."""
    coef = hbar * hbar / (mass * dx * dx)
    diag = np.full(n, coef, dtype=np.float64)
    off = np.full(n - 1, -0.5 * coef, dtype=np.float64)
    return diag, off


def solve_reduced_hamiltonian(
    x: np.ndarray,
    kinetic_diag: np.ndarray,
    kinetic_off: np.ndarray,
    *,
    hbar: float,
    mass: float,
    charge_abs: float,
    magnetic_field: float,
    k_value: float,
    n_levels: int,
) -> np.ndarray:
    """Solve lowest eigenvalues for one k in Landau gauge A=(0,Bx,0)."""
    potential = 0.5 / mass * (hbar * k_value - charge_abs * magnetic_field * x) ** 2
    h_diag = kinetic_diag + potential

    eigvals = eigh_tridiagonal(
        h_diag,
        kinetic_off,
        select="i",
        select_range=(0, n_levels - 1),
        check_finite=False,
    )[0]
    return eigvals


def analytic_landau_levels(cfg: LandauConfig) -> np.ndarray:
    """Analytic Landau levels E_n = hbar*omega_c*(n+1/2)."""
    omega_c = cfg.charge_abs * cfg.magnetic_field / cfg.mass
    n = np.arange(cfg.n_levels, dtype=np.float64)
    return cfg.hbar * omega_c * (n + 0.5)


def main() -> None:
    cfg = LandauConfig()
    cfg.validate()

    x = np.linspace(cfg.x_min, cfg.x_max, cfg.n_grid, dtype=np.float64)
    dx = float(x[1] - x[0])

    kinetic_diag, kinetic_off = build_tridiagonal_kinetic(cfg.n_grid, dx, cfg.hbar, cfg.mass)

    numeric_levels = []
    for k_value in cfg.k_values:
        eigvals = solve_reduced_hamiltonian(
            x,
            kinetic_diag,
            kinetic_off,
            hbar=cfg.hbar,
            mass=cfg.mass,
            charge_abs=cfg.charge_abs,
            magnetic_field=cfg.magnetic_field,
            k_value=k_value,
            n_levels=cfg.n_levels,
        )
        numeric_levels.append(eigvals)

    energies = np.asarray(numeric_levels, dtype=np.float64)  # shape: [n_k, n_levels]
    analytic = analytic_landau_levels(cfg)

    mean_numeric = energies.mean(axis=0)
    spread_across_k = energies.max(axis=0) - energies.min(axis=0)
    abs_error = np.abs(mean_numeric - analytic)
    rel_error = abs_error / np.maximum(np.abs(analytic), 1e-12)

    table = pd.DataFrame(
        {
            "n": np.arange(cfg.n_levels, dtype=int),
            "E_analytic": analytic,
            "E_numeric_mean": mean_numeric,
            "abs_error": abs_error,
            "rel_error": rel_error,
            "k_degeneracy_spread": spread_across_k,
        }
    )

    pd.set_option("display.width", 120)
    pd.set_option("display.precision", 8)

    print("=== Landau Levels MVP (finite-difference in Landau gauge) ===")
    print(f"grid={cfg.n_grid}, x_range=[{cfg.x_min}, {cfg.x_max}], k_samples={len(cfg.k_values)}")
    print(table.to_string(index=False))

    # Basic acceptance checks for this MVP.
    assert np.all(rel_error < 2.5e-2), "Relative error too large for one or more levels"
    assert np.all(spread_across_k < 4e-2), "k-independence (Landau degeneracy) is too weak"
    assert np.isfinite(energies).all(), "Non-finite eigenvalues encountered"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
