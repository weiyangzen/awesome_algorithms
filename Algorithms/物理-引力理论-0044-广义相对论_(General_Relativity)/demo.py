"""Minimal runnable MVP for General Relativity (PHYS-0044).

This demo computes light deflection in Schwarzschild spacetime and compares
numerical exact bending angles against weak-field approximations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq


@dataclass
class GRConfig:
    mass: float = 1.0
    b_min: float = 6.0
    b_max: float = 40.0
    num_b: int = 28
    quad_epsabs: float = 1e-10
    quad_epsrel: float = 1e-10


@dataclass
class ValidationReport:
    monotonic_decreasing: bool
    all_positive: bool
    far_weak_max_rel_error: float
    far_pn2_max_rel_error: float
    far_weak_mean_rel_error: float
    far_pn2_mean_rel_error: float
    num_far_points: int


def critical_impact_parameter(mass: float) -> float:
    """Critical impact parameter for null geodesics in Schwarzschild spacetime."""
    return 3.0 * np.sqrt(3.0) * mass


def radial_polynomial_u(u: float, b: float, mass: float) -> float:
    """F(u) = 1/b^2 - u^2 + 2 M u^3, where u = 1/r."""
    return 1.0 / (b * b) - (u * u) + 2.0 * mass * (u * u * u)


def find_turning_point_u(b: float, mass: float) -> float:
    """Find turning point u0 in (0, 1/(3M)) for scattering trajectories."""
    b_crit = critical_impact_parameter(mass)
    if b <= b_crit:
        raise ValueError(
            f"b={b:.6f} is not in scattering regime; must be > b_crit={b_crit:.6f}."
        )

    low = 1e-15
    high = (1.0 / (3.0 * mass)) - 1e-15

    f_low = radial_polynomial_u(low, b, mass)
    f_high = radial_polynomial_u(high, b, mass)
    if not (f_low > 0.0 and f_high < 0.0):
        raise RuntimeError("Failed to bracket turning-point root in expected interval.")

    return float(brentq(radial_polynomial_u, low, high, args=(b, mass), maxiter=200))


def deflection_angle_exact_single(
    b: float,
    mass: float,
    epsabs: float,
    epsrel: float,
) -> tuple[float, float]:
    """
    Compute exact Schwarzschild light deflection angle:
      alpha = 2 * integral_0^{u0} du / sqrt(1/b^2 - u^2 + 2 M u^3) - pi
    using u = u0 * (1 - t^2), t in [0, 1].
    """
    u0 = find_turning_point_u(b, mass)

    # dF/du at u0. For scattering roots u0 < 1/(3M), this derivative is negative.
    slope = -2.0 * u0 + 6.0 * mass * (u0 * u0)
    if slope >= 0.0:
        raise RuntimeError("Unexpected non-negative slope at turning point.")

    def transformed_integrand(t: float) -> float:
        # u = u0 * (1 - t^2), du = -2*u0*t dt, singularity removed at t=0.
        if t < 1e-12:
            return 2.0 * np.sqrt(u0 / (-slope))

        u = u0 * (1.0 - t * t)
        q = radial_polynomial_u(u, b, mass)
        if q <= 0.0:
            q = 1e-30
        return 2.0 * u0 * t / np.sqrt(q)

    integral_value, _ = quad(
        transformed_integrand,
        0.0,
        1.0,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=300,
    )

    alpha = float(2.0 * integral_value - np.pi)
    r0 = 1.0 / u0
    return alpha, r0


def weak_field_first_order(b: np.ndarray, mass: float) -> np.ndarray:
    """Einstein weak-field first-order light-bending angle."""
    return 4.0 * mass / b


def weak_field_second_order(b: np.ndarray, mass: float) -> np.ndarray:
    """Second-order post-Newtonian expansion for Schwarzschild light bending."""
    x = mass / b
    return 4.0 * x + (15.0 * np.pi / 4.0) * (x * x)


def build_result_table(config: GRConfig) -> pd.DataFrame:
    b_grid = np.linspace(config.b_min, config.b_max, config.num_b)
    alpha_exact = np.zeros_like(b_grid)
    r0 = np.zeros_like(b_grid)

    for i, b in enumerate(b_grid):
        alpha_i, r0_i = deflection_angle_exact_single(
            float(b),
            config.mass,
            config.quad_epsabs,
            config.quad_epsrel,
        )
        alpha_exact[i] = alpha_i
        r0[i] = r0_i

    alpha_weak = weak_field_first_order(b_grid, config.mass)
    alpha_pn2 = weak_field_second_order(b_grid, config.mass)

    weak_rel_error = np.abs(alpha_weak - alpha_exact) / alpha_exact
    pn2_rel_error = np.abs(alpha_pn2 - alpha_exact) / alpha_exact

    return pd.DataFrame(
        {
            "b": b_grid,
            "r0": r0,
            "alpha_exact_rad": alpha_exact,
            "alpha_weak_rad": alpha_weak,
            "alpha_pn2_rad": alpha_pn2,
            "weak_rel_error": weak_rel_error,
            "pn2_rel_error": pn2_rel_error,
        }
    )


def build_validation_report(df: pd.DataFrame, far_field_b_threshold: float = 25.0) -> ValidationReport:
    alpha = df["alpha_exact_rad"].to_numpy()
    b = df["b"].to_numpy()
    weak_rel = df["weak_rel_error"].to_numpy()
    pn2_rel = df["pn2_rel_error"].to_numpy()

    monotonic = bool(np.all(np.diff(alpha) < 0.0))
    all_positive = bool(np.all(alpha > 0.0))

    far_mask = b >= far_field_b_threshold
    if np.count_nonzero(far_mask) == 0:
        raise RuntimeError("No far-field points available for validation.")

    weak_far = weak_rel[far_mask]
    pn2_far = pn2_rel[far_mask]

    return ValidationReport(
        monotonic_decreasing=monotonic,
        all_positive=all_positive,
        far_weak_max_rel_error=float(np.max(weak_far)),
        far_pn2_max_rel_error=float(np.max(pn2_far)),
        far_weak_mean_rel_error=float(np.mean(weak_far)),
        far_pn2_mean_rel_error=float(np.mean(pn2_far)),
        num_far_points=int(np.count_nonzero(far_mask)),
    )


def main() -> None:
    config = GRConfig()
    b_crit = critical_impact_parameter(config.mass)
    if config.b_min <= b_crit:
        raise SystemExit(
            "Invalid config: b_min must be strictly larger than critical impact parameter "
            f"b_crit={b_crit:.6f}."
        )

    df = build_result_table(config)
    report = build_validation_report(df)

    checks = {
        "alpha(b) monotonic decreasing": report.monotonic_decreasing,
        "all alpha > 0": report.all_positive,
        "far-field max weak relative error < 13%": report.far_weak_max_rel_error < 0.13,
        "far-field max PN2 relative error < 1.7%": report.far_pn2_max_rel_error < 0.017,
        "far-field PN2 mean error < weak mean error": (
            report.far_pn2_mean_rel_error < report.far_weak_mean_rel_error
        ),
    }

    print("=== General Relativity MVP (PHYS-0044) ===")
    print("Scenario: Schwarzschild null-geodesic light deflection")
    print(
        "Parameters: M={M:.3f}, b in [{bmin:.3f}, {bmax:.3f}], samples={n}, b_crit={bc:.6f}".format(
            M=config.mass,
            bmin=config.b_min,
            bmax=config.b_max,
            n=config.num_b,
            bc=b_crit,
        )
    )

    print("\n[Result table sample]")
    print(df.head(10).to_string(index=False))

    print("\n[Far-field error summary (b >= 25M)]")
    print(
        "points={n}, weak_max={wmax:.3e}, pn2_max={pmax:.3e}, weak_mean={wmean:.3e}, pn2_mean={pmean:.3e}".format(
            n=report.num_far_points,
            wmax=report.far_weak_max_rel_error,
            pmax=report.far_pn2_max_rel_error,
            wmean=report.far_weak_mean_rel_error,
            pmean=report.far_pn2_mean_rel_error,
        )
    )

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
