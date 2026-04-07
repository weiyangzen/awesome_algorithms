"""Minimal runnable MVP for light deflection in Schwarzschild spacetime.

This script computes gravitational light bending by integrating the
null-geodesic equation in dimensionless form:

    d2w/dphi2 + w = 3 * epsilon * w^2

where:
    w = b / r
    epsilon = (G*M/c^2) / b

It compares numeric deflection with the weak-field Einstein formula:

    alpha_weak = 4 * G*M / (c^2 * b)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# SI constants
G = 6.674_30e-11
C = 299_792_458.0
M_SUN = 1.988_47e30
R_SUN = 6.963_4e8
ARCSEC_PER_RAD = 206_264.806_247_096_36
SCATTERING_LIMIT = 1.0 / (3.0 * np.sqrt(3.0))  # epsilon must be below this.


@dataclass(frozen=True)
class DeflectionConfig:
    mass_kg: float = M_SUN
    impact_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
    phi_max: float = 2.2
    rtol: float = 1e-10
    atol: float = 1e-12
    max_step: float = 0.01
    sample_points: int = 1600


@dataclass(frozen=True)
class CaseResult:
    b_m: float
    epsilon: float
    phi_infinity: float
    alpha_numeric_rad: float
    alpha_weak_rad: float
    weak_relative_error: float
    invariant_max_abs_error: float
    ode_residual_rms: float


def gravitational_radius_m(mass_kg: float) -> float:
    """Return GM/c^2 in meters."""
    return G * mass_kg / (C * C)


def periapsis_equation(w0: float, epsilon: float) -> float:
    """Equation at closest approach: w0^2 - 2*epsilon*w0^3 - 1 = 0."""
    return (w0 * w0) - 2.0 * epsilon * (w0**3) - 1.0


def solve_w0(epsilon: float) -> float:
    """Solve the closest-approach value w0 using robust bracketing."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if epsilon >= SCATTERING_LIMIT:
        raise ValueError(
            f"epsilon={epsilon:.6f} exceeds scattering limit {SCATTERING_LIMIT:.6f}"
        )

    low = 1.0
    high = 2.0
    f_low = periapsis_equation(low, epsilon)
    f_high = periapsis_equation(high, epsilon)
    while f_high <= 0.0:
        high *= 1.5
        f_high = periapsis_equation(high, epsilon)
        if high > 1e6:
            raise RuntimeError("Failed to bracket closest approach root for w0.")

    if f_low >= 0.0:
        raise RuntimeError("Unexpected bracket: f(low) must be negative.")

    return float(brentq(periapsis_equation, low, high, args=(epsilon,), maxiter=200))


def rhs(phi: float, y: np.ndarray, epsilon: float) -> np.ndarray:
    """State equation for y=[w, dw/dphi]."""
    del phi
    w, w_prime = y
    w_double_prime = 3.0 * epsilon * (w * w) - w
    return np.array([w_prime, w_double_prime], dtype=float)


def build_escape_event():
    """Event function: ray reaches infinity when w=b/r crosses zero."""

    def event(phi: float, y: np.ndarray) -> float:
        del phi
        return float(y[0])

    event.terminal = True
    event.direction = -1.0
    return event


def integrate_case(b_m: float, m_geo_m: float, cfg: DeflectionConfig) -> CaseResult:
    """Integrate one impact parameter and return physical/diagnostic metrics."""
    epsilon = m_geo_m / b_m
    w0 = solve_w0(epsilon)

    y0 = np.array([w0, 0.0], dtype=float)
    event = build_escape_event()
    sol = solve_ivp(
        fun=lambda phi, y: rhs(phi, y, epsilon),
        t_span=(0.0, cfg.phi_max),
        y0=y0,
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.max_step,
        dense_output=True,
        events=event,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    if len(sol.t_events) == 0 or len(sol.t_events[0]) == 0:
        raise RuntimeError("Ray did not reach infinity within phi_max.")

    phi_inf = float(sol.t_events[0][0])
    alpha_numeric = 2.0 * phi_inf - np.pi
    alpha_weak = 4.0 * epsilon
    weak_relative_error = abs(alpha_numeric - alpha_weak) / alpha_numeric

    # Diagnostics sampled on [0, phi_inf].
    phi_grid = np.linspace(0.0, phi_inf, cfg.sample_points)
    sampled = sol.sol(phi_grid)
    w = sampled[0]
    w_prime = sampled[1]

    invariant = (w_prime * w_prime) + (w * w) - 2.0 * epsilon * (w**3)
    invariant_max_abs_error = float(np.max(np.abs(invariant - 1.0)))

    # Numeric ODE residual from finite-difference second derivative.
    w_double_prime_fd = np.gradient(w_prime, phi_grid, edge_order=2)
    residual = w_double_prime_fd - (3.0 * epsilon * (w * w) - w)
    ode_residual_rms = float(np.sqrt(np.mean(residual[2:-2] ** 2)))

    return CaseResult(
        b_m=b_m,
        epsilon=epsilon,
        phi_infinity=phi_inf,
        alpha_numeric_rad=float(alpha_numeric),
        alpha_weak_rad=float(alpha_weak),
        weak_relative_error=float(weak_relative_error),
        invariant_max_abs_error=invariant_max_abs_error,
        ode_residual_rms=ode_residual_rms,
    )


def run_experiment(cfg: DeflectionConfig) -> tuple[pd.DataFrame, list[CaseResult]]:
    """Evaluate multiple impact parameters and build a summary table."""
    m_geo_m = gravitational_radius_m(cfg.mass_kg)
    results: list[CaseResult] = []

    for k in cfg.impact_multipliers:
        b_m = k * R_SUN
        results.append(integrate_case(b_m=b_m, m_geo_m=m_geo_m, cfg=cfg))

    rows = []
    for k, r in zip(cfg.impact_multipliers, results):
        rows.append(
            {
                "b_over_Rsun": k,
                "b_m": r.b_m,
                "epsilon": r.epsilon,
                "phi_infinity_rad": r.phi_infinity,
                "alpha_numeric_arcsec": r.alpha_numeric_rad * ARCSEC_PER_RAD,
                "alpha_weak_arcsec": r.alpha_weak_rad * ARCSEC_PER_RAD,
                "weak_relative_error": r.weak_relative_error,
                "invariant_max_abs_error": r.invariant_max_abs_error,
                "ode_residual_rms": r.ode_residual_rms,
            }
        )

    return pd.DataFrame(rows), results


def validate(results: list[CaseResult]) -> dict[str, bool]:
    """Return threshold checks for physics and numerics."""
    b_values = np.array([r.b_m for r in results], dtype=float)
    alpha_values = np.array([r.alpha_numeric_rad for r in results], dtype=float)
    weak_rel_err = np.array([r.weak_relative_error for r in results], dtype=float)
    invariant_err = np.array([r.invariant_max_abs_error for r in results], dtype=float)
    residual_rms = np.array([r.ode_residual_rms for r in results], dtype=float)

    order = np.argsort(b_values)
    alpha_sorted = alpha_values[order]
    weak_err_sorted = weak_rel_err[order]

    far_field = b_values >= (2.0 * R_SUN)
    checks = {
        "alpha positive": bool(np.all(alpha_values > 0.0)),
        "alpha decreases with larger b": bool(np.all(np.diff(alpha_sorted) < 0.0)),
        "weak-field error (<2Rsun excluded) < 0.2%": bool(
            np.all(weak_rel_err[far_field] < 2.0e-3)
        ),
        "max invariant error < 1e-8": bool(np.max(invariant_err) < 1.0e-8),
        "max ODE residual RMS < 3e-5": bool(np.max(residual_rms) < 3.0e-5),
        "weak-field error monotonic in b-order": bool(np.all(np.diff(weak_err_sorted) <= 1e-12)),
    }
    return checks


def main() -> None:
    cfg = DeflectionConfig()
    table, results = run_experiment(cfg)
    checks = validate(results)
    m_geo_m = gravitational_radius_m(cfg.mass_kg)

    print("=== Light Deflection MVP (PHYS-0368) ===")
    print("Model: Schwarzschild null geodesic in equatorial plane")
    print(f"Mass M = {cfg.mass_kg:.6e} kg")
    print(f"Gravitational radius GM/c^2 = {m_geo_m:.6f} m")
    print(f"Impact multipliers over R_sun: {cfg.impact_multipliers}")
    print()
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

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
