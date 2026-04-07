"""Minimal runnable MVP for Debye shielding in plasma physics.

This script demonstrates Debye shielding with two explicit parts:
1) Compute Debye length from a two-species plasma (electrons + ions).
2) Solve the linearized spherical Poisson-Boltzmann equation by
   finite differences for u(r)=r*phi(r):
       u'' - u/lambda_D^2 = 0
   with Dirichlet boundaries from the analytic screened potential.

It prints a compact table and runs physical/numerical sanity checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPSILON_0 = 8.8541878128e-12  # F/m
ELEMENTARY_CHARGE = 1.602176634e-19  # C
PI4_EPS0 = 4.0 * math.pi * EPSILON_0


@dataclass(frozen=True)
class DebyeConfig:
    # Quasi-neutral reference plasma
    n_e_m3: float = 1.0e17
    n_i_m3: float = 1.0e17
    z_i: int = 1

    # Temperatures in eV (k_B T = e * T_eV)
    t_e_eV: float = 2.0
    t_i_eV: float = 0.2

    # Test charge and radial solver grid
    test_charge_c: float = ELEMENTARY_CHARGE
    r_min_over_lambda: float = 0.08
    r_max_over_lambda: float = 8.0
    n_grid: int = 800

    # A second scenario for scaling sanity check (lambda_D ~ 1/sqrt(n))
    density_multiplier_probe: float = 4.0


def debye_length_two_species(
    n_e_m3: float,
    t_e_eV: float,
    n_i_m3: float,
    t_i_eV: float,
    z_i: int,
) -> tuple[float, dict[str, float]]:
    """Return Debye length for electron+ion plasma in SI.

    Linearized Poisson-Boltzmann gives:
        1/lambda_D^2 = sum_s [ n_s q_s^2 / (epsilon_0 k_B T_s) ]
    With temperature in eV: k_B T_s = e * T_s(eV).
    """
    if min(n_e_m3, n_i_m3, t_e_eV, t_i_eV) <= 0.0:
        raise ValueError("Densities and temperatures must be positive.")
    if z_i <= 0:
        raise ValueError("Ion charge state z_i must be positive.")

    q_e = ELEMENTARY_CHARGE
    q_i = z_i * ELEMENTARY_CHARGE

    inv_lambda2_e = n_e_m3 * (q_e * q_e) / (EPSILON_0 * (ELEMENTARY_CHARGE * t_e_eV))
    inv_lambda2_i = n_i_m3 * (q_i * q_i) / (EPSILON_0 * (ELEMENTARY_CHARGE * t_i_eV))
    inv_lambda2_total = inv_lambda2_e + inv_lambda2_i

    lambda_d = math.sqrt(1.0 / inv_lambda2_total)
    parts = {
        "inv_lambda2_e": inv_lambda2_e,
        "inv_lambda2_i": inv_lambda2_i,
        "inv_lambda2_total": inv_lambda2_total,
    }
    return lambda_d, parts


def analytic_screened_potential(r: np.ndarray, q_c: float, lambda_d: float) -> np.ndarray:
    """Yukawa / Debye-Huckel potential for a point test charge."""
    rr = np.asarray(r, dtype=float)
    return (q_c / (PI4_EPS0 * rr)) * np.exp(-rr / lambda_d)


def coulomb_potential(r: np.ndarray, q_c: float) -> np.ndarray:
    rr = np.asarray(r, dtype=float)
    return q_c / (PI4_EPS0 * rr)


def thomas_solve_tridiagonal(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve tridiagonal Ax=b using Thomas algorithm.

    Parameters
    ----------
    lower : (n-1,) subdiagonal
    diag  : (n,)   main diagonal
    upper : (n-1,) superdiagonal
    rhs   : (n,)   right-hand side
    """
    n = diag.size
    if n == 0:
        return np.array([], dtype=float)
    if lower.size != n - 1 or upper.size != n - 1 or rhs.size != n:
        raise ValueError("Invalid tridiagonal dimensions.")

    c_prime = np.zeros(n - 1, dtype=float)
    d_prime = np.zeros(n, dtype=float)

    denom = diag[0]
    if abs(denom) < 1e-18:
        raise ValueError("Singular tridiagonal system at first pivot.")

    if n > 1:
        c_prime[0] = upper[0] / denom
    d_prime[0] = rhs[0] / denom

    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        if abs(denom) < 1e-18:
            raise ValueError(f"Singular tridiagonal system at pivot {i}.")
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    x = np.zeros(n, dtype=float)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def solve_potential_fd(
    r_grid: np.ndarray,
    lambda_d: float,
    q_c: float,
) -> np.ndarray:
    """Finite-difference solver for u'' - u/lambda^2 = 0, u=r*phi.

    Boundary values are taken from analytic screened potential at both ends.
    This keeps the solver focused on the interior discretization behavior.
    """
    r = np.asarray(r_grid, dtype=float)
    if r.ndim != 1 or r.size < 3:
        raise ValueError("r_grid must be 1D with at least 3 points.")

    dr = r[1] - r[0]
    if not np.allclose(np.diff(r), dr, rtol=1e-10, atol=1e-14):
        raise ValueError("r_grid must be uniformly spaced.")
    if dr <= 0.0:
        raise ValueError("r_grid spacing must be positive.")

    n = r.size
    alpha = (dr * dr) / (lambda_d * lambda_d)

    # Unknowns are interior u[1:n-1]
    m = n - 2
    lower = -np.ones(m - 1, dtype=float)
    diag = (2.0 + alpha) * np.ones(m, dtype=float)
    upper = -np.ones(m - 1, dtype=float)
    rhs = np.zeros(m, dtype=float)

    u_left = (q_c / PI4_EPS0) * math.exp(-r[0] / lambda_d)
    u_right = (q_c / PI4_EPS0) * math.exp(-r[-1] / lambda_d)

    rhs[0] += u_left
    rhs[-1] += u_right

    u_inner = thomas_solve_tridiagonal(lower, diag, upper, rhs)

    u = np.empty(n, dtype=float)
    u[0] = u_left
    u[-1] = u_right
    u[1:-1] = u_inner

    phi = u / r
    return phi


def build_profiles(cfg: DebyeConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    lambda_d, parts = debye_length_two_species(
        n_e_m3=cfg.n_e_m3,
        t_e_eV=cfg.t_e_eV,
        n_i_m3=cfg.n_i_m3,
        t_i_eV=cfg.t_i_eV,
        z_i=cfg.z_i,
    )

    r_min = cfg.r_min_over_lambda * lambda_d
    r_max = cfg.r_max_over_lambda * lambda_d
    r_grid = np.linspace(r_min, r_max, cfg.n_grid)

    phi_num = solve_potential_fd(r_grid, lambda_d, cfg.test_charge_c)
    phi_ana = analytic_screened_potential(r_grid, cfg.test_charge_c, lambda_d)
    phi_coulomb = coulomb_potential(r_grid, cfg.test_charge_c)

    rel_err = np.abs(phi_num - phi_ana) / np.maximum(np.abs(phi_ana), 1e-30)
    shield_factor_num = phi_num / phi_coulomb

    df = pd.DataFrame(
        {
            "r_m": r_grid,
            "r_over_lambda": r_grid / lambda_d,
            "phi_num_V": phi_num,
            "phi_analytic_V": phi_ana,
            "phi_coulomb_V": phi_coulomb,
            "shield_factor_num": shield_factor_num,
            "rel_err": rel_err,
        }
    )

    metrics = {
        "lambda_d_m": lambda_d,
        "inv_lambda2_e": parts["inv_lambda2_e"],
        "inv_lambda2_i": parts["inv_lambda2_i"],
        "max_rel_err": float(np.max(rel_err[1:-1])),
        "mean_rel_err": float(np.mean(rel_err[1:-1])),
    }
    return df, metrics


def run_sanity_checks(cfg: DebyeConfig, df: pd.DataFrame, metrics: dict[str, float]) -> dict[str, float]:
    lambda_d = metrics["lambda_d_m"]

    # 1) Numerical solution should match analytic profile in the interior.
    assert metrics["max_rel_err"] < 2.0e-3, (
        f"Finite-difference error too large: max_rel_err={metrics['max_rel_err']:.3e}"
    )

    # 2) Shielding factor at r=lambda_D should be ~exp(-1).
    idx = int(np.argmin(np.abs(df["r_over_lambda"].to_numpy() - 1.0)))
    shield_at_1lambda = float(df.iloc[idx]["shield_factor_num"])
    expected = math.exp(-1.0)
    rel_shield_err = abs(shield_at_1lambda - expected) / expected
    assert rel_shield_err < 0.03, (
        "Shielding factor near r=lambda_D deviates too much: "
        f"{shield_at_1lambda:.6f} vs exp(-1)={expected:.6f}"
    )

    # 3) Potential should decrease monotonically away from test charge.
    phi_num = df["phi_num_V"].to_numpy()
    monotonic_violations = int(np.sum(np.diff(phi_num) > 0.0))
    assert monotonic_violations == 0, "Potential is not monotone decreasing."

    # 4) Density scaling sanity: n -> 4n gives lambda_D -> lambda_D/2.
    lambda_scaled, _ = debye_length_two_species(
        n_e_m3=cfg.n_e_m3 * cfg.density_multiplier_probe,
        t_e_eV=cfg.t_e_eV,
        n_i_m3=cfg.n_i_m3 * cfg.density_multiplier_probe,
        t_i_eV=cfg.t_i_eV,
        z_i=cfg.z_i,
    )
    ratio = lambda_scaled / lambda_d
    expected_ratio = 1.0 / math.sqrt(cfg.density_multiplier_probe)
    ratio_err = abs(ratio - expected_ratio) / expected_ratio
    assert ratio_err < 1e-12

    return {
        "shield_at_1lambda": shield_at_1lambda,
        "shield_expected": expected,
        "shield_rel_err": rel_shield_err,
        "monotonic_violations": float(monotonic_violations),
        "lambda_scaled_ratio": ratio,
        "lambda_scaled_expected_ratio": expected_ratio,
        "lambda_scaled_ratio_rel_err": ratio_err,
    }


def main() -> None:
    cfg = DebyeConfig()
    df, metrics = build_profiles(cfg)
    checks = run_sanity_checks(cfg, df, metrics)

    print("=== Debye Shielding MVP (Linearized Poisson-Boltzmann) ===")
    print(
        "Plasma params: "
        f"n_e={cfg.n_e_m3:.3e} m^-3, n_i={cfg.n_i_m3:.3e} m^-3, "
        f"T_e={cfg.t_e_eV:.3f} eV, T_i={cfg.t_i_eV:.3f} eV, Z_i={cfg.z_i}"
    )
    print(
        "Debye contributions: "
        f"inv_lambda^2(e)={metrics['inv_lambda2_e']:.3e}, "
        f"inv_lambda^2(i)={metrics['inv_lambda2_i']:.3e}, "
        f"lambda_D={metrics['lambda_d_m']:.6e} m"
    )

    sample_idx = np.linspace(0, len(df) - 1, 8, dtype=int)
    sample_df = df.iloc[sample_idx].copy()
    cols = [
        "r_over_lambda",
        "phi_num_V",
        "phi_analytic_V",
        "phi_coulomb_V",
        "shield_factor_num",
        "rel_err",
    ]

    print("\nSample radial profile points:")
    with pd.option_context("display.max_rows", 20, "display.width", 160):
        print(sample_df[cols].to_string(index=False, float_format=lambda x: f"{x:11.6e}"))

    print("\nChecks:")
    print(f"- Interior max relative error: {metrics['max_rel_err']:.6e}")
    print(f"- Interior mean relative error: {metrics['mean_rel_err']:.6e}")
    print(
        "- Shielding at r=lambda_D: "
        f"{checks['shield_at_1lambda']:.6e} (target exp(-1)={checks['shield_expected']:.6e}, "
        f"rel_err={checks['shield_rel_err']:.2%})"
    )
    print(
        "- Density scaling lambda_D(4n)/lambda_D(n): "
        f"{checks['lambda_scaled_ratio']:.6e} "
        f"(expected={checks['lambda_scaled_expected_ratio']:.6e})"
    )
    print("- Monotonicity violations in phi(r): 0")


if __name__ == "__main__":
    main()
