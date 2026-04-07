"""Minimal runnable MVP for the Born-Oppenheimer approximation (H2+ toy model).

This demo separates electronic and nuclear motion in three explicit stages:
1) Electronic problem at fixed internuclear distance R (LCAO with 1s basis).
2) Born-Oppenheimer potential energy surface V_BO(R).
3) 1D nuclear vibrational problem on V_BO(R).

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


@dataclass(frozen=True)
class BOConfig:
    r_min: float = 0.5
    r_max: float = 8.0
    n_grid: int = 1200
    optimize_left: float = 0.5
    optimize_right: float = 8.0
    proton_mass_au: float = 1836.15267343
    n_vib_levels: int = 5
    fit_window: float = 0.30


def overlap_1s(r: np.ndarray | float) -> np.ndarray | float:
    """Overlap S(R) between two normalized 1s orbitals on separated centers."""

    r_arr = np.asarray(r, dtype=float)
    return np.exp(-r_arr) * (1.0 + r_arr + r_arr**2 / 3.0)


def h_aa(r: np.ndarray | float) -> np.ndarray | float:
    """Diagonal electronic Hamiltonian element <1s_A|H_e|1s_A> for fixed R."""

    r_arr = np.asarray(r, dtype=float)
    return -0.5 - 1.0 / r_arr + (1.0 + 1.0 / r_arr) * np.exp(-2.0 * r_arr)


def h_ab(r: np.ndarray | float) -> np.ndarray | float:
    """Off-diagonal electronic Hamiltonian element <1s_A|H_e|1s_B>."""

    r_arr = np.asarray(r, dtype=float)
    s_val = overlap_1s(r_arr)
    return -0.5 * s_val - np.exp(-r_arr) * (1.0 + r_arr)


def electronic_energy_sigma_g(r: np.ndarray | float) -> np.ndarray | float:
    """Bonding electronic eigenvalue E_e(R) for H2+ in a minimal LCAO basis."""

    r_arr = np.asarray(r, dtype=float)
    s_val = overlap_1s(r_arr)
    return (h_aa(r_arr) + h_ab(r_arr)) / (1.0 + s_val)


def bo_potential(r: np.ndarray | float) -> np.ndarray | float:
    """Born-Oppenheimer PES V_BO(R) = E_e(R) + 1/R in atomic units."""

    r_arr = np.asarray(r, dtype=float)
    return electronic_energy_sigma_g(r_arr) + 1.0 / r_arr


def find_equilibrium(cfg: BOConfig) -> dict[str, float]:
    """Locate equilibrium geometry by scalar minimization of V_BO(R)."""

    res = minimize_scalar(
        lambda x: float(bo_potential(x)),
        bounds=(cfg.optimize_left, cfg.optimize_right),
        method="bounded",
    )
    if not res.success:
        raise RuntimeError(f"Failed to minimize BO potential: {res.message}")

    r_eq = float(res.x)
    return {
        "r_eq": r_eq,
        "v_min": float(res.fun),
        "e_electronic_at_eq": float(electronic_energy_sigma_g(r_eq)),
    }


def solve_nuclear_motion(cfg: BOConfig) -> dict[str, np.ndarray | float]:
    """Solve 1D nuclear Schrodinger equation on the BO surface via finite differences."""

    r_grid = np.linspace(cfg.r_min, cfg.r_max, cfg.n_grid, dtype=float)
    dr = float(r_grid[1] - r_grid[0])

    mu = cfg.proton_mass_au / 2.0
    v_grid = bo_potential(r_grid)

    # Tridiagonal form for T + V with Dirichlet boundaries on the chosen finite box.
    diag = 1.0 / (mu * dr * dr) + v_grid
    offdiag = np.full(r_grid.size - 1, -0.5 / (mu * dr * dr), dtype=float)

    evals, evecs = eigh_tridiagonal(
        diag,
        offdiag,
        select="i",
        select_range=(0, cfg.n_vib_levels - 1),
    )

    psi0 = evecs[:, 0].copy()
    psi0 /= np.sqrt(np.sum(psi0**2) * dr)
    mean_r = float(np.sum(r_grid * psi0**2) * dr)

    return {
        "r_grid": r_grid,
        "dr": dr,
        "mu": mu,
        "v_grid": v_grid,
        "levels": evals,
        "wavefunctions": evecs,
        "mean_r_ground": mean_r,
    }


def fit_local_quadratic(r_grid: np.ndarray, v_grid: np.ndarray, r_eq: float, window: float) -> dict[str, float]:
    """Fit V(R) around equilibrium to estimate local curvature / force constant."""

    mask = np.abs(r_grid - r_eq) <= window
    if np.count_nonzero(mask) < 6:
        raise ValueError("Too few points in local fitting window.")

    x = (r_grid[mask] - r_eq).reshape(-1, 1)
    y = v_grid[mask]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = poly.fit_transform(x)
    reg = LinearRegression().fit(x_poly, y)

    slope = float(reg.coef_[0])
    quad = float(reg.coef_[1])
    force_constant = 2.0 * quad

    return {
        "slope": slope,
        "quad": quad,
        "force_constant": force_constant,
        "r2": float(reg.score(x_poly, y)),
    }


def torch_force_and_curvature(r_eq: float) -> tuple[float, float]:
    """Use autograd to compute dV/dR and d2V/dR2 at equilibrium candidate."""

    r = torch.tensor(r_eq, dtype=torch.float64, requires_grad=True)

    s_val = torch.exp(-r) * (1.0 + r + r**2 / 3.0)
    h_diag = -0.5 - 1.0 / r + (1.0 + 1.0 / r) * torch.exp(-2.0 * r)
    h_off = -0.5 * s_val - torch.exp(-r) * (1.0 + r)
    v = (h_diag + h_off) / (1.0 + s_val) + 1.0 / r

    grad = torch.autograd.grad(v, r, create_graph=True)[0]
    hess = torch.autograd.grad(grad, r)[0]

    force = -float(grad.detach().cpu().item())
    curvature = float(hess.detach().cpu().item())
    return force, curvature


def harmonic_observables(force_constant: float, mu: float) -> tuple[float, float]:
    """Approximate vibrational frequency and ZPE from harmonic curvature."""

    if force_constant <= 0.0:
        return float("nan"), float("nan")

    omega = float(np.sqrt(force_constant / mu))
    zpe = 0.5 * omega
    return omega, zpe


def validate_results(
    eq_data: dict[str, float],
    nuclear_data: dict[str, np.ndarray | float],
    fit_data: dict[str, float],
    force_at_eq: float,
    autograd_curvature: float,
) -> list[tuple[str, bool]]:
    """Basic numerical sanity checks for the MVP."""

    levels = np.asarray(nuclear_data["levels"], dtype=float)
    checks: list[tuple[str, bool]] = []

    checks.append(("Equilibrium inside search interval", 0.5 < eq_data["r_eq"] < 8.0))
    checks.append(("BO minimum below dissociation limit (-0.5 Ha)", eq_data["v_min"] < -0.5))
    checks.append(("Autograd force near zero at R_eq", abs(force_at_eq) < 1e-5))
    checks.append(("Quadratic curvature positive", fit_data["force_constant"] > 0.0))
    checks.append(("Autograd curvature positive", autograd_curvature > 0.0))
    checks.append(("Vibrational levels strictly increasing", bool(np.all(np.diff(levels) > 0.0))))
    checks.append(("Ground vib level above BO minimum", float(levels[0]) > eq_data["v_min"]))

    return checks


def main() -> None:
    cfg = BOConfig()

    eq_data = find_equilibrium(cfg)
    nuclear_data = solve_nuclear_motion(cfg)

    fit_data = fit_local_quadratic(
        r_grid=np.asarray(nuclear_data["r_grid"], dtype=float),
        v_grid=np.asarray(nuclear_data["v_grid"], dtype=float),
        r_eq=eq_data["r_eq"],
        window=cfg.fit_window,
    )

    force_at_eq, autograd_curvature = torch_force_and_curvature(eq_data["r_eq"])
    omega, zpe = harmonic_observables(fit_data["force_constant"], float(nuclear_data["mu"]))

    summary = pd.DataFrame(
        [
            {
                "R_eq_bohr": eq_data["r_eq"],
                "E_electronic_eq_hartree": eq_data["e_electronic_at_eq"],
                "V_BO_min_hartree": eq_data["v_min"],
                "force_at_eq_hartree_per_bohr": force_at_eq,
                "k_fit_hartree_per_bohr2": fit_data["force_constant"],
                "k_autograd_hartree_per_bohr2": autograd_curvature,
                "omega_harmonic_au": omega,
                "zpe_harmonic_hartree": zpe,
                "E_vib0_total_hartree": float(np.asarray(nuclear_data["levels"])[0]),
                "mean_R_vib0_bohr": float(nuclear_data["mean_r_ground"]),
                "local_fit_r2": fit_data["r2"],
            }
        ]
    )

    levels_df = pd.DataFrame(
        {
            "v": np.arange(np.asarray(nuclear_data["levels"]).size, dtype=int),
            "E_total_hartree": np.asarray(nuclear_data["levels"], dtype=float),
        }
    )

    pes_df = pd.DataFrame(
        {
            "R_bohr": np.asarray(nuclear_data["r_grid"], dtype=float),
            "V_BO_hartree": np.asarray(nuclear_data["v_grid"], dtype=float),
        }
    )

    print("=== Born-Oppenheimer MVP (H2+ minimal-basis toy) ===")
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    print("\n=== Lowest Vibrational Levels (on BO PES) ===")
    print(levels_df.to_string(index=False))

    print("\n=== PES Sample Points ===")
    print(pes_df.iloc[::200, :].to_string(index=False))

    checks = validate_results(eq_data, nuclear_data, fit_data, force_at_eq, autograd_curvature)
    print("\n=== Validation Checks ===")
    all_pass = True
    for name, ok in checks:
        state = "PASS" if ok else "FAIL"
        print(f"[{state}] {name}")
        all_pass = all_pass and ok

    print(f"\nValidation: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
