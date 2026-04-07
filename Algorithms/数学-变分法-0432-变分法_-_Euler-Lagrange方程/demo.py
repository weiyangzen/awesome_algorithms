"""Euler-Lagrange equation MVP for a 1D fixed-endpoint variational problem.

We solve:
    minimize J[y] = integral_0^1 (0.5 * y'(x)^2 + 0.5 * lam * y(x)^2) dx
subject to:
    y(0) = y0, y(1) = y1.

Euler-Lagrange equation:
    y''(x) - lam * y(x) = 0

The demo discretizes this boundary-value problem with finite differences,
solves the resulting tridiagonal linear system, and compares against
the closed-form analytic solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class CaseConfig:
    name: str
    lam: float
    y0: float
    y1: float
    num_points: int


def validate_config(cfg: CaseConfig) -> None:
    if cfg.num_points < 3:
        raise ValueError("num_points must be >= 3 (including two boundary points).")
    if cfg.lam < 0.0:
        raise ValueError("lam must be >= 0 for this MVP.")
    if not np.isfinite(cfg.lam) or not np.isfinite(cfg.y0) or not np.isfinite(cfg.y1):
        raise ValueError("lam, y0, y1 must be finite.")


def build_linear_system(
    num_points: int,
    lam: float,
    y0: float,
    y1: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build A * y_interior = rhs from finite-difference Euler-Lagrange equation."""
    n_interior = num_points - 2
    h = 1.0 / float(num_points - 1)

    diag = (2.0 + lam * h * h) * np.ones(n_interior, dtype=float)
    off = -1.0 * np.ones(n_interior - 1, dtype=float)

    a = np.diag(diag)
    if n_interior > 1:
        a += np.diag(off, k=1)
        a += np.diag(off, k=-1)

    rhs = np.zeros(n_interior, dtype=float)
    rhs[0] += y0
    rhs[-1] += y1
    return a, rhs


def solve_discrete_euler_lagrange(cfg: CaseConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_grid, y_numeric) using linear solve on interior unknowns."""
    validate_config(cfg)
    x = np.linspace(0.0, 1.0, cfg.num_points, dtype=float)
    a, rhs = build_linear_system(
        num_points=cfg.num_points,
        lam=cfg.lam,
        y0=cfg.y0,
        y1=cfg.y1,
    )
    y_interior = np.linalg.solve(a, rhs)

    y = np.empty(cfg.num_points, dtype=float)
    y[0] = cfg.y0
    y[-1] = cfg.y1
    y[1:-1] = y_interior
    return x, y


def analytic_solution(x: np.ndarray, lam: float, y0: float, y1: float) -> np.ndarray:
    """Closed-form solution of y'' - lam*y = 0 with y(0)=y0, y(1)=y1."""
    if lam < 1e-12:
        return y0 + (y1 - y0) * x

    mu = float(np.sqrt(lam))
    sinh_mu = float(np.sinh(mu))
    if abs(sinh_mu) < 1e-15:
        return y0 + (y1 - y0) * x

    c2 = y0
    c1 = (y1 - y0 * np.cosh(mu)) / sinh_mu
    return c1 * np.sinh(mu * x) + c2 * np.cosh(mu * x)


def action_value(x: np.ndarray, y: np.ndarray, lam: float) -> float:
    """Discrete approximation of the functional J[y]."""
    h = float(x[1] - x[0])
    dy = np.diff(y) / h
    y_mid = 0.5 * (y[:-1] + y[1:])
    integrand = 0.5 * dy * dy + 0.5 * lam * y_mid * y_mid
    return float(h * np.sum(integrand))


def euler_lagrange_residual(y: np.ndarray, h: float, lam: float) -> np.ndarray:
    """Residual r_i = y''(x_i) - lam*y(x_i) on interior points."""
    second_diff = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (h * h)
    return second_diff - lam * y[1:-1]


def run_case(cfg: CaseConfig) -> dict:
    print(f"\n=== Case: {cfg.name} ===")
    print(
        "lambda={:.6g}, y(0)={:.6g}, y(1)={:.6g}, num_points={}".format(
            cfg.lam, cfg.y0, cfg.y1, cfg.num_points
        )
    )

    x, y_num = solve_discrete_euler_lagrange(cfg)
    y_ref = analytic_solution(x, cfg.lam, cfg.y0, cfg.y1)

    h = float(x[1] - x[0])
    residual = euler_lagrange_residual(y_num, h, cfg.lam)
    err = y_num - y_ref

    max_abs_err = float(np.max(np.abs(err)))
    l2_err = float(np.sqrt(h * np.sum(err * err)))
    residual_inf = float(np.max(np.abs(residual)))

    j_num = action_value(x, y_num, cfg.lam)
    j_ref = action_value(x, y_ref, cfg.lam)

    # Second-order finite-difference consistency target.
    expected_scale = h * h * (1.0 + abs(cfg.y0) + abs(cfg.y1))
    pass_flag = max_abs_err <= 8.0 * expected_scale and residual_inf <= 1e-10

    print(f"grid step h:                 {h:.9e}")
    print(f"max |y_num - y_ref|:         {max_abs_err:.9e}")
    print(f"L2 error:                    {l2_err:.9e}")
    print(f"max interior EL residual:    {residual_inf:.9e}")
    print(f"J[y_num]:                    {j_num:.9e}")
    print(f"J[y_ref]:                    {j_ref:.9e}")
    print(f"|J[y_num]-J[y_ref]|:         {abs(j_num - j_ref):.9e}")
    print(f"pass finite-diff check:      {pass_flag}")

    return {
        "max_abs_err": max_abs_err,
        "l2_err": l2_err,
        "residual_inf": residual_inf,
        "action_gap": float(abs(j_num - j_ref)),
        "pass_flag": float(pass_flag),
    }


def main() -> None:
    cases: List[CaseConfig] = [
        CaseConfig(
            name="Baseline (lambda=2, homogeneous left boundary)",
            lam=2.0,
            y0=0.0,
            y1=1.0,
            num_points=161,
        ),
        CaseConfig(
            name="Stronger potential (lambda=10)",
            lam=10.0,
            y0=1.0,
            y1=-0.5,
            num_points=201,
        ),
        CaseConfig(
            name="Near free particle (lambda~0)",
            lam=1e-8,
            y0=-1.0,
            y1=2.0,
            num_points=121,
        ),
    ]

    results = [run_case(cfg) for cfg in cases]

    max_abs_err = max(item["max_abs_err"] for item in results)
    max_residual = max(item["residual_inf"] for item in results)
    max_action_gap = max(item["action_gap"] for item in results)
    all_pass = all(bool(item["pass_flag"]) for item in results)

    print("\n=== Summary ===")
    print(f"max absolute solution error: {max_abs_err:.9e}")
    print(f"max EL residual infinity:    {max_residual:.9e}")
    print(f"max action gap:              {max_action_gap:.9e}")
    print(f"all cases pass:              {all_pass}")


if __name__ == "__main__":
    main()
