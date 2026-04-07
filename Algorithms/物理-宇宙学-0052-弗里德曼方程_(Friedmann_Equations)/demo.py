"""Friedmann equations MVP demo for background cosmology evolution.

This script implements a transparent, non-interactive pipeline for a homogeneous
and isotropic FRW universe:
- H(a), H(z)
- cosmic age
- comoving/luminosity/angular-diameter distances
- deceleration parameter and acceleration-transition redshift
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

C_KM_PER_S = 299792.458
KM_PER_MPC = 3.0856775814913673e19
SEC_PER_GYR = 365.25 * 24.0 * 3600.0 * 1.0e9


def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Compat wrapper for NumPy trapezoid integration API differences."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    # Fallback: manual composite trapezoid rule.
    dx = np.diff(x)
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * dx))


@dataclass(frozen=True)
class CosmologyParams:
    """Background cosmological parameters at z=0."""

    name: str
    h0_km_s_mpc: float
    omega_m: float
    omega_r: float
    omega_lambda: float
    omega_k: Optional[float] = None

    def with_derived_omega_k(self) -> "CosmologyParams":
        if self.omega_k is None:
            omega_k = 1.0 - self.omega_m - self.omega_r - self.omega_lambda
        else:
            omega_k = self.omega_k
        return CosmologyParams(
            name=self.name,
            h0_km_s_mpc=self.h0_km_s_mpc,
            omega_m=self.omega_m,
            omega_r=self.omega_r,
            omega_lambda=self.omega_lambda,
            omega_k=omega_k,
        )


def validate_params(p: CosmologyParams) -> None:
    if p.h0_km_s_mpc <= 0.0:
        raise ValueError("H0 must be positive.")
    for key, value in {
        "omega_m": p.omega_m,
        "omega_r": p.omega_r,
        "omega_lambda": p.omega_lambda,
        "omega_k": p.omega_k,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{key} is non-finite.")

    # Ensure expansion rate is real and positive over the sampled range.
    z_grid = np.linspace(0.0, 10.0, 2001)
    a_grid = 1.0 / (1.0 + z_grid)
    e2 = e2_of_a(a_grid, p)
    if np.any(e2 <= 0.0):
        raise ValueError(
            f"E(a)^2 <= 0 encountered for cosmology '{p.name}'. "
            "Choose physically valid density parameters."
        )


def h0_si(p: CosmologyParams) -> float:
    """Hubble constant H0 in s^-1."""
    return p.h0_km_s_mpc / KM_PER_MPC


def e2_of_a(a: np.ndarray, p: CosmologyParams) -> np.ndarray:
    """Dimensionless Hubble-rate squared E(a)^2 = H(a)^2 / H0^2."""
    return (
        p.omega_r / a**4
        + p.omega_m / a**3
        + p.omega_k / a**2
        + p.omega_lambda
    )


def e_of_a(a: np.ndarray, p: CosmologyParams) -> np.ndarray:
    return np.sqrt(e2_of_a(a, p))


def h_of_z(z: np.ndarray, p: CosmologyParams) -> np.ndarray:
    a = 1.0 / (1.0 + z)
    return p.h0_km_s_mpc * e_of_a(a, p)


def deceleration_parameter_of_z(z: np.ndarray, p: CosmologyParams) -> np.ndarray:
    """q(z) = -a a_ddot / a_dot^2 using the 2nd Friedmann equation."""
    a = 1.0 / (1.0 + z)
    numerator = p.omega_m / a**3 + 2.0 * p.omega_r / a**4 - 2.0 * p.omega_lambda
    return numerator / (2.0 * e2_of_a(a, p))


def age_of_universe_gyr(
    p: CosmologyParams,
    n_grid: int = 200_000,
    a_min: float = 1.0e-6,
) -> float:
    """t0 = ∫ da / (a H(a)) from a=0 to 1, truncated at tiny a_min."""
    if n_grid < 1000:
        raise ValueError("n_grid must be >= 1000 for stable integration.")
    if not (0.0 < a_min < 1.0):
        raise ValueError("a_min must be in (0, 1).")

    a_grid = np.linspace(a_min, 1.0, n_grid)
    integrand = 1.0 / (a_grid * h0_si(p) * e_of_a(a_grid, p))
    age_seconds = integrate_trapezoid(integrand, a_grid)
    return age_seconds / SEC_PER_GYR


def lookback_time_gyr(z: float, p: CosmologyParams, n_grid: int = 50_000) -> float:
    if z < 0.0:
        raise ValueError("z must be non-negative.")
    a_start = 1.0 / (1.0 + z)
    a_grid = np.linspace(a_start, 1.0, n_grid)
    integrand = 1.0 / (a_grid * h0_si(p) * e_of_a(a_grid, p))
    t_seconds = integrate_trapezoid(integrand, a_grid)
    return t_seconds / SEC_PER_GYR


def line_of_sight_comoving_distance_mpc(
    z: float,
    p: CosmologyParams,
    n_grid: int = 80_000,
) -> float:
    if z < 0.0:
        raise ValueError("z must be non-negative.")
    z_grid = np.linspace(0.0, z, n_grid)
    integrand = 1.0 / e_of_a(1.0 / (1.0 + z_grid), p)
    d_h = C_KM_PER_S / p.h0_km_s_mpc  # Hubble distance in Mpc
    return float(d_h * integrate_trapezoid(integrand, z_grid))


def transverse_comoving_distance_mpc(z: float, p: CosmologyParams) -> float:
    d_c = line_of_sight_comoving_distance_mpc(z, p)
    d_h = C_KM_PER_S / p.h0_km_s_mpc

    if abs(p.omega_k) < 1.0e-12:
        return d_c
    if p.omega_k > 0.0:
        sqrt_ok = np.sqrt(p.omega_k)
        return float((d_h / sqrt_ok) * np.sinh(sqrt_ok * d_c / d_h))

    sqrt_abs_ok = np.sqrt(-p.omega_k)
    return float((d_h / sqrt_abs_ok) * np.sin(sqrt_abs_ok * d_c / d_h))


def luminosity_distance_mpc(z: float, p: CosmologyParams) -> float:
    return (1.0 + z) * transverse_comoving_distance_mpc(z, p)


def angular_diameter_distance_mpc(z: float, p: CosmologyParams) -> float:
    return transverse_comoving_distance_mpc(z, p) / (1.0 + z)


def find_acceleration_transition_z(
    p: CosmologyParams,
    z_max: float = 5.0,
    n_grid: int = 100_000,
) -> Optional[float]:
    """Solve q(z)=0 by sign-change interpolation over a dense grid."""
    z_grid = np.linspace(0.0, z_max, n_grid)
    q_grid = deceleration_parameter_of_z(z_grid, p)

    # Need q(0) < 0 and q(large z) > 0 for a transition.
    if q_grid[0] >= 0.0:
        return None

    for i in range(n_grid - 1):
        q0, q1 = q_grid[i], q_grid[i + 1]
        if q0 <= 0.0 < q1:
            z0, z1 = z_grid[i], z_grid[i + 1]
            # Linear interpolation for q(z)=0 between (z0,q0) and (z1,q1)
            return float(z0 - q0 * (z1 - z0) / (q1 - q0))
    return None


def run_case(p_raw: CosmologyParams) -> Dict[str, float]:
    p = p_raw.with_derived_omega_k()
    validate_params(p)

    age_gyr = age_of_universe_gyr(p)
    q0 = float(deceleration_parameter_of_z(np.array([0.0]), p)[0])
    z_acc = find_acceleration_transition_z(p)
    h_z1 = float(h_of_z(np.array([1.0]), p)[0])
    h_z3 = float(h_of_z(np.array([3.0]), p)[0])

    dl_05 = luminosity_distance_mpc(0.5, p)
    dl_10 = luminosity_distance_mpc(1.0, p)
    da_10 = angular_diameter_distance_mpc(1.0, p)
    t_lb_10 = lookback_time_gyr(1.0, p)

    print(f"\n=== Case: {p.name} ===")
    print(f"H0 [km/s/Mpc]              : {p.h0_km_s_mpc:.4f}")
    print(
        "Omega(m, r, k, Lambda)     : "
        f"({p.omega_m:.6f}, {p.omega_r:.6f}, {p.omega_k:.6f}, {p.omega_lambda:.6f})"
    )
    print(f"Age t0 [Gyr]               : {age_gyr:.6f}")
    print(f"q0                          : {q0:.6f}")
    print(
        "z_acc (q=0 transition)      : "
        f"{z_acc:.6f}" if z_acc is not None else "z_acc (q=0 transition)      : None"
    )
    print(f"H(z=1) [km/s/Mpc]          : {h_z1:.6f}")
    print(f"H(z=3) [km/s/Mpc]          : {h_z3:.6f}")
    print(f"D_L(z=0.5) [Mpc]           : {dl_05:.6f}")
    print(f"D_L(z=1.0) [Mpc]           : {dl_10:.6f}")
    print(f"D_A(z=1.0) [Mpc]           : {da_10:.6f}")
    print(f"Lookback t(z=1) [Gyr]      : {t_lb_10:.6f}")

    return {
        "omega_sum": p.omega_m + p.omega_r + p.omega_k + p.omega_lambda,
        "age_gyr": age_gyr,
        "q0": q0,
        "z_acc": -1.0 if z_acc is None else z_acc,
        "h_z1": h_z1,
        "h_z3": h_z3,
        "dl_05": dl_05,
        "dl_10": dl_10,
        "da_10": da_10,
        "lookback_1": t_lb_10,
    }


def main() -> None:
    # Case A: near-flat LambdaCDM (Planck-like)
    omega_m = 0.315
    omega_r = 9.0e-5
    omega_lambda = 1.0 - omega_m - omega_r
    lcdm = CosmologyParams(
        name="Flat-LambdaCDM",
        h0_km_s_mpc=67.4,
        omega_m=omega_m,
        omega_r=omega_r,
        omega_lambda=omega_lambda,
        omega_k=0.0,
    )

    # Case B: Einstein-de Sitter universe
    eds = CosmologyParams(
        name="Einstein-de Sitter",
        h0_km_s_mpc=67.4,
        omega_m=1.0,
        omega_r=0.0,
        omega_lambda=0.0,
        omega_k=0.0,
    )

    # Case C: open matter-curvature universe (no dark energy)
    open_mk = CosmologyParams(
        name="Open-Matter+Curvature",
        h0_km_s_mpc=67.4,
        omega_m=0.3,
        omega_r=0.0,
        omega_lambda=0.0,
        omega_k=0.7,
    )

    results = {
        "lcdm": run_case(lcdm),
        "eds": run_case(eds),
        "open_mk": run_case(open_mk),
    }

    # Consistency checks (MVP acceptance)
    assert abs(results["lcdm"]["omega_sum"] - 1.0) < 1e-12
    assert 13.0 < results["lcdm"]["age_gyr"] < 15.0
    assert results["lcdm"]["q0"] < 0.0
    assert 0.4 < results["lcdm"]["z_acc"] < 1.0
    assert results["lcdm"]["h_z3"] > results["lcdm"]["h_z1"] > lcdm.h0_km_s_mpc
    assert results["lcdm"]["dl_10"] > results["lcdm"]["dl_05"] > 0.0

    # Einstein-de Sitter age has analytic solution t0 = 2/(3H0)
    h0_s_inv = h0_si(eds)
    t0_eds_exact_gyr = (2.0 / (3.0 * h0_s_inv)) / SEC_PER_GYR
    assert abs(results["eds"]["age_gyr"] - t0_eds_exact_gyr) < 0.02
    assert abs(results["eds"]["q0"] - 0.5) < 1e-6
    assert results["eds"]["z_acc"] < 0.0  # encoded None as -1.0

    # Open matter+curvature universe should still decelerate today (q0 > 0)
    assert results["open_mk"]["q0"] > 0.0

    print("\n=== Summary ===")
    print("All physical sanity checks passed.")


if __name__ == "__main__":
    main()
