"""Minimal runnable MVP for classical scattering theory (PHYS-0128)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq


@dataclass
class ScatteringConfig:
    energy: float = 3.0
    kappa: float = 1.2
    b_min: float = 0.08
    b_max: float = 3.6
    num_b: int = 160
    quad_epsabs: float = 1e-10
    quad_epsrel: float = 1e-10


@dataclass
class AngleValidationReport:
    max_abs_error: float
    mean_abs_error: float
    monotonic_decreasing: bool


@dataclass
class CrossSectionValidationReport:
    median_relative_error: float
    p90_relative_error: float
    num_effective_points: int


def radial_equation_residual(r: float, b: float, energy: float, kappa: float) -> float:
    """Residual of 1 - b^2/r^2 - V(r)/E = 0 for repulsive Coulomb V(r)=kappa/r."""
    return 1.0 - (b * b) / (r * r) - (kappa / (energy * r))


def find_turning_point(b: float, energy: float, kappa: float) -> float:
    """Find r_min solving radial_equation_residual(r)=0 with safe bracketing."""
    low = 1e-12
    high = max(1.0, b + (kappa / energy) + 1.0)

    f_low = radial_equation_residual(low, b, energy, kappa)
    f_high = radial_equation_residual(high, b, energy, kappa)

    if not (f_low < 0.0):
        raise RuntimeError("Low bracket must be in forbidden region (residual < 0).")

    while f_high <= 0.0:
        high *= 2.0
        f_high = radial_equation_residual(high, b, energy, kappa)
        if high > 1e8:
            raise RuntimeError("Failed to bracket turning point.")

    return float(brentq(radial_equation_residual, low, high, args=(b, energy, kappa), maxiter=200))


def deflection_angle_analytic(b: np.ndarray, energy: float, kappa: float) -> np.ndarray:
    """Rutherford analytical angle for repulsive Coulomb scattering."""
    return 2.0 * np.arctan(kappa / (2.0 * energy * b))


def deflection_angle_numeric_single(
    b: float,
    energy: float,
    kappa: float,
    epsabs: float,
    epsrel: float,
) -> tuple[float, float]:
    """
    Numerically evaluate classical deflection angle:
      theta = pi - 2 * integral_{r_min}^{inf} [ b / (r^2 * sqrt(1 - b^2/r^2 - kappa/(E r))) ] dr
    with a transformed finite integral to remove endpoint singularity.
    """
    r_min = find_turning_point(b, energy, kappa)
    u_max = 1.0 / r_min
    a = kappa / energy
    slope_at_root = -(a + 2.0 * b * b * u_max)  # d/du [1 - a*u - b^2*u^2] at u=u_max
    if slope_at_root >= 0.0:
        raise RuntimeError("Unexpected non-negative slope near turning point.")

    def transformed_integrand(t: float) -> float:
        # u = u_max * (1 - t^2), t in [0,1], removes sqrt singularity at u=u_max.
        if t < 1e-12:
            return 2.0 * b * np.sqrt(u_max / (-slope_at_root))

        u = u_max * (1.0 - t * t)
        q = 1.0 - a * u - (b * b) * (u * u)
        if q <= 0.0:
            q = 1e-30
        return 2.0 * b * u_max * t / np.sqrt(q)

    integral_value, _ = quad(
        transformed_integrand,
        0.0,
        1.0,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=300,
    )

    theta = float(np.pi - 2.0 * integral_value)
    return theta, r_min


def compute_angle_table(config: ScatteringConfig) -> pd.DataFrame:
    """Compute b -> theta map numerically and analytically."""
    b_grid = np.linspace(config.b_min, config.b_max, config.num_b)
    theta_num = np.zeros_like(b_grid)
    r_min = np.zeros_like(b_grid)

    for i, b in enumerate(b_grid):
        theta_i, rmin_i = deflection_angle_numeric_single(
            float(b),
            config.energy,
            config.kappa,
            config.quad_epsabs,
            config.quad_epsrel,
        )
        theta_num[i] = theta_i
        r_min[i] = rmin_i

    theta_ana = deflection_angle_analytic(b_grid, config.energy, config.kappa)
    abs_error = np.abs(theta_num - theta_ana)

    return pd.DataFrame(
        {
            "b": b_grid,
            "r_min": r_min,
            "theta_numeric_rad": theta_num,
            "theta_analytic_rad": theta_ana,
            "abs_error": abs_error,
        }
    )


def numerical_dsigma_domega(b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute dσ/dΩ from b(theta): dσ/dΩ = b/sin(theta) * |db/dtheta|."""
    db_dtheta = np.gradient(b, theta, edge_order=2)
    sin_theta = np.sin(theta)
    dsigma = (b / sin_theta) * np.abs(db_dtheta)
    return dsigma


def rutherford_dsigma_domega(theta: np.ndarray, energy: float, kappa: float) -> np.ndarray:
    """Analytical Rutherford differential cross section."""
    coeff = (kappa / (4.0 * energy)) ** 2
    return coeff / (np.sin(theta / 2.0) ** 4)


def compute_cross_section_table(angle_df: pd.DataFrame, energy: float, kappa: float) -> pd.DataFrame:
    """
    Build cross-section comparison table.
    We trim endpoint angles where finite-difference derivative is less stable.
    """
    theta = angle_df["theta_numeric_rad"].to_numpy()
    b = angle_df["b"].to_numpy()

    dsigma_num = numerical_dsigma_domega(b, theta)
    dsigma_ana = rutherford_dsigma_domega(theta, energy, kappa)

    stable_mask = (theta > 0.18) & (theta < 2.8)
    stable_mask[[0, 1, -2, -1]] = False

    rel_error = np.abs(dsigma_num - dsigma_ana) / dsigma_ana

    return pd.DataFrame(
        {
            "b": b[stable_mask],
            "theta_rad": theta[stable_mask],
            "dsigma_num": dsigma_num[stable_mask],
            "dsigma_analytic": dsigma_ana[stable_mask],
            "relative_error": rel_error[stable_mask],
        }
    )


def build_angle_validation(angle_df: pd.DataFrame) -> AngleValidationReport:
    theta_num = angle_df["theta_numeric_rad"].to_numpy()
    abs_error = angle_df["abs_error"].to_numpy()
    monotonic = bool(np.all(np.diff(theta_num) < 0.0))
    return AngleValidationReport(
        max_abs_error=float(np.max(abs_error)),
        mean_abs_error=float(np.mean(abs_error)),
        monotonic_decreasing=monotonic,
    )


def build_cross_validation(cross_df: pd.DataFrame) -> CrossSectionValidationReport:
    rel = cross_df["relative_error"].to_numpy()
    return CrossSectionValidationReport(
        median_relative_error=float(np.median(rel)),
        p90_relative_error=float(np.quantile(rel, 0.90)),
        num_effective_points=int(rel.size),
    )


def main() -> None:
    config = ScatteringConfig()
    angle_df = compute_angle_table(config)
    cross_df = compute_cross_section_table(angle_df, config.energy, config.kappa)

    angle_report = build_angle_validation(angle_df)
    cross_report = build_cross_validation(cross_df)

    checks = {
        "theta(b) monotonic decreasing": angle_report.monotonic_decreasing,
        "max abs angle error < 2e-6": angle_report.max_abs_error < 2e-6,
        "cross-section median relative error < 2%": cross_report.median_relative_error < 0.02,
        "cross-section p90 relative error < 8%": cross_report.p90_relative_error < 0.08,
    }

    print("=== Classical Scattering Theory MVP (PHYS-0128) ===")
    print(
        "Model: repulsive Coulomb potential V(r)=kappa/r, E={E:.3f}, kappa={K:.3f}".format(
            E=config.energy,
            K=config.kappa,
        )
    )
    print(
        "Impact parameter grid: [{bmin:.3f}, {bmax:.3f}] with {n} samples".format(
            bmin=config.b_min,
            bmax=config.b_max,
            n=config.num_b,
        )
    )

    print("\n[Angle comparison]")
    print(
        "max_abs_error = {mx:.3e}, mean_abs_error = {mn:.3e}, monotonic = {mono}".format(
            mx=angle_report.max_abs_error,
            mn=angle_report.mean_abs_error,
            mono=angle_report.monotonic_decreasing,
        )
    )
    print("sample rows:")
    print(angle_df.head(8).to_string(index=False))

    print("\n[Cross-section comparison]")
    print(
        "effective_points = {n}, median_rel_error = {m:.3e}, p90_rel_error = {p90:.3e}".format(
            n=cross_report.num_effective_points,
            m=cross_report.median_relative_error,
            p90=cross_report.p90_relative_error,
        )
    )
    print("sample rows:")
    print(cross_df.head(8).to_string(index=False))

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
