"""Minimal runnable MVP for LOD (Locally One-Dimensional) method.

We solve the 2D heat equation on (0,1)^2 with homogeneous Dirichlet boundary:
    u_t = kappa * (u_xx + u_yy)

Using Lie-type LOD splitting per time step:
    (I - dt*kappa*Dxx) u*     = u^n
    (I - dt*kappa*Dyy) u^(n+1) = u*

Each 1D implicit sweep is solved by a Thomas tridiagonal solver implemented in
this file (no PDE black-box solver is used).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np


@dataclass
class LODConfig:
    """Configuration for a deterministic LOD diffusion test."""

    n: int = 40
    kappa: float = 1.0
    dt: float = 0.001
    t_end: float = 0.2
    report_every: int = 20

    def validate(self) -> None:
        if self.n < 3:
            raise ValueError("n must be >= 3")
        if not math.isfinite(self.kappa) or self.kappa <= 0.0:
            raise ValueError("kappa must be finite and positive")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(self.t_end) or self.t_end <= 0.0:
            raise ValueError("t_end must be finite and positive")
        if self.report_every <= 0:
            raise ValueError("report_every must be positive")
        steps_real = self.t_end / self.dt
        steps_int = round(steps_real)
        if abs(steps_real - steps_int) > 1e-12:
            raise ValueError("t_end/dt must be an integer for this demo")


@dataclass
class TridiagonalFactor:
    """Factorized tridiagonal data for repeated Thomas solves."""

    lower: np.ndarray
    c_prime: np.ndarray
    inv_denom: np.ndarray


@dataclass
class SimulationResult:
    """Recorded diagnostics and final state."""

    times: np.ndarray
    rel_errors: np.ndarray
    l2_norms: np.ndarray
    final_u: np.ndarray


def build_grid(n: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return interior meshgrid and spacing h for unit square with Dirichlet boundaries."""
    h = 1.0 / (n + 1)
    pts = np.linspace(h, 1.0 - h, n)
    xg, yg = np.meshgrid(pts, pts, indexing="ij")
    return xg, yg, h


def exact_solution(xg: np.ndarray, yg: np.ndarray, t: float, kappa: float) -> np.ndarray:
    """Analytic solution for u0=sin(pi x)sin(pi y) under homogeneous diffusion."""
    decay = math.exp(-2.0 * math.pi * math.pi * kappa * t)
    return decay * np.sin(math.pi * xg) * np.sin(math.pi * yg)


def factorize_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray) -> TridiagonalFactor:
    """Pre-factor tridiagonal matrix for repeated Thomas solves."""
    n = diag.size
    if lower.size != n - 1 or upper.size != n - 1:
        raise ValueError("invalid tridiagonal sizes")

    c_prime = np.zeros(max(n - 1, 0), dtype=float)
    inv_denom = np.zeros(n, dtype=float)

    inv_denom[0] = 1.0 / diag[0]
    if n > 1:
        c_prime[0] = upper[0] * inv_denom[0]
        for i in range(1, n - 1):
            denom = diag[i] - lower[i - 1] * c_prime[i - 1]
            inv_denom[i] = 1.0 / denom
            c_prime[i] = upper[i] * inv_denom[i]
        denom = diag[n - 1] - lower[n - 2] * c_prime[n - 2]
        inv_denom[n - 1] = 1.0 / denom

    return TridiagonalFactor(lower=lower.copy(), c_prime=c_prime, inv_denom=inv_denom)


def solve_tridiagonal_factored(fac: TridiagonalFactor, rhs: np.ndarray) -> np.ndarray:
    """Solve Ax=rhs using pre-factorized Thomas coefficients."""
    n = rhs.size
    if fac.inv_denom.size != n:
        raise ValueError("rhs size mismatch")

    d_prime = np.empty(n, dtype=float)
    d_prime[0] = rhs[0] * fac.inv_denom[0]

    for i in range(1, n):
        d_prime[i] = (rhs[i] - fac.lower[i - 1] * d_prime[i - 1]) * fac.inv_denom[i]

    x = np.empty(n, dtype=float)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - fac.c_prime[i] * x[i + 1]

    return x


def lod_step(u: np.ndarray, fac: TridiagonalFactor) -> np.ndarray:
    """One LOD step: implicit x-sweep then implicit y-sweep."""
    n = u.shape[0]
    if u.shape[1] != n:
        raise ValueError("u must be square")

    u_star = np.empty_like(u)
    for j in range(n):
        u_star[:, j] = solve_tridiagonal_factored(fac, u[:, j])

    u_next = np.empty_like(u)
    for i in range(n):
        u_next[i, :] = solve_tridiagonal_factored(fac, u_star[i, :])

    return u_next


def run_lod_simulation(cfg: LODConfig) -> SimulationResult:
    """Run deterministic LOD simulation and record diagnostics."""
    cfg.validate()

    xg, yg, h = build_grid(cfg.n)
    u = exact_solution(xg, yg, t=0.0, kappa=cfg.kappa)

    n_steps = int(round(cfg.t_end / cfg.dt))
    r = cfg.kappa * cfg.dt / (h * h)

    lower = -r * np.ones(cfg.n - 1, dtype=float)
    diag = (1.0 + 2.0 * r) * np.ones(cfg.n, dtype=float)
    upper = -r * np.ones(cfg.n - 1, dtype=float)
    fac = factorize_tridiagonal(lower, diag, upper)

    time_log: List[float] = []
    rel_error_log: List[float] = []
    l2_log: List[float] = []

    for step in range(n_steps + 1):
        t = step * cfg.dt
        if (step % cfg.report_every == 0) or (step == n_steps):
            u_ex = exact_solution(xg, yg, t=t, kappa=cfg.kappa)
            rel_err = np.linalg.norm((u - u_ex).ravel()) / np.linalg.norm(u_ex.ravel())
            l2 = np.linalg.norm(u.ravel()) * h

            time_log.append(t)
            rel_error_log.append(float(rel_err))
            l2_log.append(float(l2))

        if step == n_steps:
            break

        u = lod_step(u, fac)
        if not np.all(np.isfinite(u)):
            raise RuntimeError("non-finite value encountered")

    return SimulationResult(
        times=np.array(time_log, dtype=float),
        rel_errors=np.array(rel_error_log, dtype=float),
        l2_norms=np.array(l2_log, dtype=float),
        final_u=u,
    )


def print_report(result: SimulationResult) -> None:
    """Print compact table for non-interactive validation."""
    print("time      relative_l2_error    l2_norm")
    for t, e, nrm in zip(result.times, result.rel_errors, result.l2_norms):
        print(f"{t:0.4f}    {e:0.8f}          {nrm:0.8f}")

    print("\nfinal_field_stats")
    print(f"  min={float(np.min(result.final_u)):.8f}")
    print(f"  max={float(np.max(result.final_u)):.8f}")
    print(f"  mean={float(np.mean(result.final_u)):.8f}")


def run_checks(result: SimulationResult) -> None:
    """Basic correctness checks for this MVP setup."""
    if result.times.size < 2:
        raise AssertionError("not enough report points")

    assert np.all(np.isfinite(result.final_u)), "final field contains non-finite values"
    assert result.rel_errors[-1] < 0.03, "final relative error is too large"

    l2_diffs = np.diff(result.l2_norms)
    assert np.all(l2_diffs <= 1e-12), "l2 norm should be non-increasing for heat equation"


def main() -> None:
    print("LOD Method MVP (MATH-0455)")
    print("Model: 2D heat equation solved by Locally One-Dimensional splitting")
    print("=" * 72)

    cfg = LODConfig()
    result = run_lod_simulation(cfg)
    print_report(result)
    run_checks(result)

    print("=" * 72)
    print("All checks passed.")


if __name__ == "__main__":
    main()
