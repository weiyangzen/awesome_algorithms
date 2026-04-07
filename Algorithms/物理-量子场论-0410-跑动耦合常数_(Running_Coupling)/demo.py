"""Running coupling MVP (one-loop RG) for QFT-like toy/phenomenology cases.

This script demonstrates the one-loop renormalization group equation

    d alpha / d ln(mu) = c * alpha^2,

with:
- analytic closed-form running coupling,
- explicit RK4 integration on a logarithmic energy grid,
- deterministic test cases for QED-like and QCD-like flows.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class CaseConfig:
    name: str
    mu0: float
    alpha0: float
    mu_end: float
    num_points: int
    c: float


@dataclass(frozen=True)
class CaseResult:
    name: str
    c: float
    mu0: float
    mu_end: float
    alpha0: float
    alpha_end_numeric: float
    alpha_end_analytic: float
    max_rel_error: float
    monotonic_ok: bool
    pole_scale: float


def check_positive_finite_scalar(name: str, x: float) -> None:
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x}.")


def check_mu_grid(mu_grid: np.ndarray) -> None:
    if mu_grid.ndim != 1:
        raise ValueError(f"mu_grid must be 1D, got shape={mu_grid.shape}.")
    if mu_grid.size < 2:
        raise ValueError("mu_grid must contain at least 2 points.")
    if not np.all(np.isfinite(mu_grid)):
        raise ValueError("mu_grid contains non-finite values.")
    if not np.all(mu_grid > 0.0):
        raise ValueError("mu_grid must be strictly positive.")

    diff = np.diff(mu_grid)
    strictly_increasing = np.all(diff > 0.0)
    strictly_decreasing = np.all(diff < 0.0)
    if not (strictly_increasing or strictly_decreasing):
        raise ValueError("mu_grid must be strictly monotonic.")


def beta_one_loop(alpha: float, c: float) -> float:
    return c * alpha * alpha


def running_coupling_analytic(
    mu: np.ndarray,
    mu0: float,
    alpha0: float,
    c: float,
) -> np.ndarray:
    check_positive_finite_scalar("mu0", mu0)
    check_positive_finite_scalar("alpha0", alpha0)

    if mu.ndim != 1:
        raise ValueError(f"mu must be 1D, got shape={mu.shape}.")
    if not np.all(np.isfinite(mu)) or not np.all(mu > 0.0):
        raise ValueError("mu must be positive and finite.")

    log_ratio = np.log(mu / mu0)
    denominator = 1.0 - c * alpha0 * log_ratio
    if np.any(denominator <= 0.0):
        raise ValueError(
            "Requested mu range crosses a one-loop Landau pole; "
            "analytic coupling becomes non-physical in this toy model."
        )

    return alpha0 / denominator


def running_coupling_rk4(
    mu_grid: np.ndarray,
    alpha0: float,
    c: float,
) -> np.ndarray:
    check_mu_grid(mu_grid)
    check_positive_finite_scalar("alpha0", alpha0)

    t_grid = np.log(mu_grid)
    alpha = np.empty_like(mu_grid, dtype=float)
    alpha[0] = alpha0

    for i in range(1, mu_grid.size):
        dt = float(t_grid[i] - t_grid[i - 1])
        a = float(alpha[i - 1])

        k1 = beta_one_loop(a, c)
        k2 = beta_one_loop(a + 0.5 * dt * k1, c)
        k3 = beta_one_loop(a + 0.5 * dt * k2, c)
        k4 = beta_one_loop(a + dt * k3, c)

        a_next = a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(a_next) or a_next <= 0.0:
            raise RuntimeError(
                "RK4 integration produced non-finite/non-positive coupling; "
                "likely too close to Landau pole."
            )
        alpha[i] = a_next

    return alpha


def relative_error(x: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return np.abs(x - y) / (np.abs(y) + eps)


def beta_coeff_qed(nf: int) -> float:
    if nf <= 0:
        raise ValueError("QED nf must be positive.")
    return 2.0 * float(nf) / (3.0 * np.pi)


def beta_coeff_qcd(nf: int) -> float:
    if nf <= 0:
        raise ValueError("QCD nf must be positive.")
    beta0 = 11.0 - 2.0 * float(nf) / 3.0
    return -beta0 / (2.0 * np.pi)


def estimate_one_loop_pole_scale(mu0: float, alpha0: float, c: float) -> float:
    check_positive_finite_scalar("mu0", mu0)
    check_positive_finite_scalar("alpha0", alpha0)

    denom = c * alpha0
    if denom == 0.0:
        return float("inf")

    exponent = 1.0 / denom
    if exponent > 700.0:
        return float("inf")
    if exponent < -700.0:
        return 0.0

    return float(mu0 * np.exp(exponent))


def monotonicity_check(alpha: np.ndarray, mu0: float, mu_end: float, c: float) -> bool:
    diff = np.diff(alpha)
    if np.allclose(diff, 0.0, atol=1e-14, rtol=0.0):
        return True

    t_direction = np.sign(np.log(mu_end / mu0))
    expected_sign = np.sign(c) * t_direction

    tol = 1e-12
    if expected_sign > 0.0:
        return bool(np.all(diff >= -tol))
    if expected_sign < 0.0:
        return bool(np.all(diff <= tol))
    return bool(np.all(np.abs(diff) <= tol))


def run_case(cfg: CaseConfig) -> CaseResult:
    mu_grid = np.geomspace(cfg.mu0, cfg.mu_end, cfg.num_points)

    alpha_numeric = running_coupling_rk4(mu_grid=mu_grid, alpha0=cfg.alpha0, c=cfg.c)
    alpha_exact = running_coupling_analytic(mu=mu_grid, mu0=cfg.mu0, alpha0=cfg.alpha0, c=cfg.c)

    rel = relative_error(alpha_numeric, alpha_exact)
    max_rel = float(np.max(rel))

    pole_scale = estimate_one_loop_pole_scale(mu0=cfg.mu0, alpha0=cfg.alpha0, c=cfg.c)
    monotonic_ok = monotonicity_check(alpha=alpha_numeric, mu0=cfg.mu0, mu_end=cfg.mu_end, c=cfg.c)

    print(f"\n=== {cfg.name} ===")
    print(f"c (beta coefficient in dα/dlnμ = c α²): {cfg.c:.9e}")
    print(f"mu range: [{cfg.mu0:.6g}, {cfg.mu_end:.6g}] GeV with {cfg.num_points} log-points")
    print(f"alpha(mu0): {cfg.alpha0:.9e}")
    print(f"alpha(mu_end) numeric : {alpha_numeric[-1]:.9e}")
    print(f"alpha(mu_end) analytic: {alpha_exact[-1]:.9e}")
    print(f"max relative error (RK4 vs analytic): {max_rel:.9e}")
    print(f"monotonic trend check: {monotonic_ok}")
    if np.isfinite(pole_scale):
        print(f"one-loop pole scale estimate: {pole_scale:.9e} GeV")
    else:
        print("one-loop pole scale estimate: +inf (outside practical range)")

    return CaseResult(
        name=cfg.name,
        c=cfg.c,
        mu0=cfg.mu0,
        mu_end=cfg.mu_end,
        alpha0=cfg.alpha0,
        alpha_end_numeric=float(alpha_numeric[-1]),
        alpha_end_analytic=float(alpha_exact[-1]),
        max_rel_error=max_rel,
        monotonic_ok=monotonic_ok,
        pole_scale=pole_scale,
    )


def print_summary(results: List[CaseResult]) -> None:
    print("\n=== Summary ===")
    print(
        "case | alpha_end_numeric | alpha_end_analytic | max_rel_error | monotonic_ok | pole_scale"
    )
    print("-" * 108)
    for r in results:
        pole_text = f"{r.pole_scale:.6e}" if np.isfinite(r.pole_scale) else "inf"
        print(
            f"{r.name:20s} | {r.alpha_end_numeric:17.9e} | {r.alpha_end_analytic:17.9e} "
            f"| {r.max_rel_error:12.3e} | {str(r.monotonic_ok):12s} | {pole_text}"
        )

    worst_err = max(item.max_rel_error for item in results)
    all_monotonic = all(item.monotonic_ok for item in results)
    pass_flag = (worst_err < 1e-8) and all_monotonic

    print("\nGlobal checks:")
    print(f"worst max_rel_error < 1e-8 : {worst_err < 1e-8} (value={worst_err:.3e})")
    print(f"all monotonic checks passed: {all_monotonic}")
    print(f"overall pass flag: {pass_flag}")


def main() -> None:
    c_qed_nf1 = beta_coeff_qed(nf=1)
    c_qcd_nf5 = beta_coeff_qcd(nf=5)

    cases = [
        CaseConfig(
            name="QED-like UV run",
            mu0=1.0,
            alpha0=1.0 / 137.0,
            mu_end=1.0e6,
            num_points=2200,
            c=c_qed_nf1,
        ),
        CaseConfig(
            name="QCD-like UV run",
            mu0=91.1876,
            alpha0=0.1181,
            mu_end=1.0e4,
            num_points=1800,
            c=c_qcd_nf5,
        ),
        CaseConfig(
            name="QCD-like IR run",
            mu0=91.1876,
            alpha0=0.1181,
            mu_end=1.0,
            num_points=2200,
            c=c_qcd_nf5,
        ),
    ]

    results = [run_case(cfg) for cfg in cases]
    print_summary(results)


if __name__ == "__main__":
    main()
