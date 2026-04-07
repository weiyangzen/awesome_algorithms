"""Minimal runnable MVP for Lagrange points in the circular restricted three-body problem.

This script computes all five Lagrange points (L1-L5) in normalized CR3BP coordinates,
then reports Jacobi constants and linearized stability diagnostics.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar


@dataclass
class CR3BPConfig:
    m1: float
    m2: float
    bracket_eps: float = 1e-6
    root_xtol: float = 1e-13
    root_rtol: float = 1e-11


@dataclass
class LagrangePointResult:
    name: str
    x: float
    y: float
    jacobi_constant: float
    max_real_eigenvalue: float
    stability: str
    iterations: int


def validate_config(config: CR3BPConfig) -> None:
    if not np.isfinite(config.m1) or not np.isfinite(config.m2):
        raise ValueError("m1 and m2 must be finite.")
    if config.m1 <= 0.0 or config.m2 <= 0.0:
        raise ValueError("m1 and m2 must be positive.")
    if config.bracket_eps <= 0.0:
        raise ValueError("bracket_eps must be > 0.")
    if config.root_xtol <= 0.0 or config.root_rtol <= 0.0:
        raise ValueError("root tolerances must be > 0.")


def mass_ratio(m1: float, m2: float) -> float:
    return float(m2 / (m1 + m2))


def collinear_equation(x: float, mu: float) -> float:
    """Equation for collinear Lagrange points in normalized rotating frame.

    Primaries are at x=-mu and x=1-mu.
    Equilibrium on x-axis satisfies dOmega/dx = 0.
    """
    term1 = (1.0 - mu) * (x + mu) / abs(x + mu) ** 3
    term2 = mu * (x - 1.0 + mu) / abs(x - 1.0 + mu) ** 3
    return x - term1 - term2


def solve_brent_root(
    func: Callable[[float], float],
    bracket: Tuple[float, float],
    xtol: float,
    rtol: float,
) -> Tuple[float, int]:
    a, b = bracket
    fa = func(a)
    fb = func(b)
    if not np.isfinite(fa) or not np.isfinite(fb):
        raise ValueError(f"Non-finite function value at bracket endpoints: f({a})={fa}, f({b})={fb}")
    if fa * fb > 0.0:
        raise ValueError(
            "Bracket does not contain sign change. "
            f"a={a}, b={b}, f(a)={fa}, f(b)={fb}"
        )

    result = root_scalar(func, bracket=(a, b), method="brentq", xtol=xtol, rtol=rtol, maxiter=200)
    if not result.converged:
        raise RuntimeError(f"Brent solver failed to converge for bracket {bracket}.")

    iterations = 0 if result.iterations is None else int(result.iterations)
    return float(result.root), iterations


def effective_potential(x: float, y: float, mu: float) -> float:
    r1 = np.hypot(x + mu, y)
    r2 = np.hypot(x - 1.0 + mu, y)
    return (1.0 - mu) / r1 + mu / r2 + 0.5 * (x * x + y * y)


def jacobi_constant_at_rest(x: float, y: float, mu: float) -> float:
    # At equilibrium with zero velocity in rotating frame: C = 2 * Omega
    return 2.0 * effective_potential(x=x, y=y, mu=mu)


def second_derivatives_omega(x: float, y: float, mu: float) -> Tuple[float, float, float]:
    r1_sq = (x + mu) ** 2 + y * y
    r2_sq = (x - 1.0 + mu) ** 2 + y * y
    r1 = np.sqrt(r1_sq)
    r2 = np.sqrt(r2_sq)

    r1_3 = r1_sq * r1
    r2_3 = r2_sq * r2
    r1_5 = r1_3 * r1_sq
    r2_5 = r2_3 * r2_sq

    omega_xx = (
        1.0
        - (1.0 - mu) / r1_3
        - mu / r2_3
        + 3.0 * (1.0 - mu) * (x + mu) ** 2 / r1_5
        + 3.0 * mu * (x - 1.0 + mu) ** 2 / r2_5
    )
    omega_yy = (
        1.0
        - (1.0 - mu) / r1_3
        - mu / r2_3
        + 3.0 * (1.0 - mu) * y * y / r1_5
        + 3.0 * mu * y * y / r2_5
    )
    omega_xy = 3.0 * (1.0 - mu) * (x + mu) * y / r1_5 + 3.0 * mu * (x - 1.0 + mu) * y / r2_5
    return float(omega_xx), float(omega_yy), float(omega_xy)


def planar_linearization_eigenvalues(x: float, y: float, mu: float) -> np.ndarray:
    omega_xx, omega_yy, omega_xy = second_derivatives_omega(x=x, y=y, mu=mu)

    a = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [omega_xx, omega_xy, 0.0, 2.0],
            [omega_xy, omega_yy, -2.0, 0.0],
        ],
        dtype=float,
    )
    return np.linalg.eigvals(a)


def classify_stability(eigenvalues: Sequence[complex], tol: float = 1e-9) -> Tuple[str, float]:
    max_real = float(np.max(np.real(eigenvalues)))
    if max_real < tol:
        return "linearly stable", max_real
    return "linearly unstable", max_real


def solve_lagrange_points(mu: float, config: CR3BPConfig) -> List[LagrangePointResult]:
    eps = config.bracket_eps
    f = lambda x: collinear_equation(x=x, mu=mu)

    # Brackets from CR3BP geometry:
    # L1 in (-mu, 1-mu), L2 in (1-mu, +inf), L3 in (-inf, -mu)
    x_l1, it_l1 = solve_brent_root(
        func=f,
        bracket=(-mu + eps, 1.0 - mu - eps),
        xtol=config.root_xtol,
        rtol=config.root_rtol,
    )
    x_l2, it_l2 = solve_brent_root(
        func=f,
        bracket=(1.0 - mu + eps, 2.0),
        xtol=config.root_xtol,
        rtol=config.root_rtol,
    )
    x_l3, it_l3 = solve_brent_root(
        func=f,
        bracket=(-2.0, -mu - eps),
        xtol=config.root_xtol,
        rtol=config.root_rtol,
    )

    x_l4 = 0.5 - mu
    y_l4 = np.sqrt(3.0) / 2.0
    x_l5 = x_l4
    y_l5 = -np.sqrt(3.0) / 2.0

    points = [
        ("L1", x_l1, 0.0, it_l1),
        ("L2", x_l2, 0.0, it_l2),
        ("L3", x_l3, 0.0, it_l3),
        ("L4", x_l4, y_l4, 0),
        ("L5", x_l5, y_l5, 0),
    ]

    results: List[LagrangePointResult] = []
    for name, x, y, iterations in points:
        jacobi = jacobi_constant_at_rest(x=x, y=y, mu=mu)
        eigvals = planar_linearization_eigenvalues(x=x, y=y, mu=mu)
        stability, max_real = classify_stability(eigenvalues=eigvals)
        results.append(
            LagrangePointResult(
                name=name,
                x=float(x),
                y=float(y),
                jacobi_constant=float(jacobi),
                max_real_eigenvalue=max_real,
                stability=stability,
                iterations=int(iterations),
            )
        )

    return results


def to_table(results: Sequence[LagrangePointResult]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for item in results:
        records.append(
            {
                "point": item.name,
                "x": item.x,
                "y": item.y,
                "Jacobi_C": item.jacobi_constant,
                "max_real_eig": item.max_real_eigenvalue,
                "stability": item.stability,
                "root_iters": item.iterations,
            }
        )
    df = pd.DataFrame.from_records(records)
    return df


def main() -> None:
    # Earth-Moon mass example (kg)
    config = CR3BPConfig(m1=5.972e24, m2=7.348e22)
    validate_config(config)

    mu = mass_ratio(config.m1, config.m2)
    if not (0.0 < mu < 0.5):
        raise ValueError(f"Mass ratio mu must be in (0, 0.5), got {mu}.")

    results = solve_lagrange_points(mu=mu, config=config)
    df = to_table(results)

    pd.set_option("display.float_format", lambda v: f"{v: .10f}")

    print("=== Lagrange Points in Circular Restricted Three-Body Problem ===")
    print(f"m1={config.m1:.6e}, m2={config.m2:.6e}, mu=m2/(m1+m2)={mu:.10f}")
    print("normalized frame: primary1 at x=-mu, primary2 at x=1-mu")
    print()
    print(df.to_string(index=False))

    mu_crit = 0.0385208965
    print("\nTriangular-point stability threshold (Routh):")
    print(f"  mu_crit ~= {mu_crit:.10f}, current mu={mu:.10f}")
    if mu < mu_crit:
        print("  expected: L4/L5 linearly stable in planar CR3BP (small perturbations).")
    else:
        print("  expected: L4/L5 not linearly stable in planar CR3BP.")


if __name__ == "__main__":
    main()
