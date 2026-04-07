"""Crank-Nicolson MVP for 1D heat equation with zero Dirichlet boundaries.

Model:
    u_t = alpha * u_xx, x in [0, 1], t > 0
    u(0, t) = u(1, t) = 0
    u(x, 0) = sin(pi x)

Exact solution:
    u(x, t) = exp(-alpha * pi^2 * t) * sin(pi x)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np


@dataclass
class RunResult:
    nx: int
    nt: int
    dx: float
    dt: float
    r: float
    l2_error: float
    max_error: float
    snapshots: List[Tuple[int, float, float]]


def initial_condition(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x)


def exact_solution(x: np.ndarray, t: float, alpha: float) -> np.ndarray:
    return np.exp(-alpha * (np.pi**2) * t) * np.sin(np.pi * x)


def l2_error(u: np.ndarray, v: np.ndarray, dx: float) -> float:
    return float(np.sqrt(dx * np.sum((u - v) ** 2)))


def solve_tridiagonal(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Thomas algorithm for tridiagonal linear systems."""
    n = diag.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return rhs / diag

    c_prime = np.empty(n - 1, dtype=float)
    d_prime = np.empty(n, dtype=float)

    denom = diag[0]
    c_prime[0] = upper[0] / denom
    d_prime[0] = rhs[0] / denom

    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    x = np.empty(n, dtype=float)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def crank_nicolson_heat_1d(alpha: float, t_final: float, nx: int, nt: int) -> RunResult:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if t_final <= 0.0:
        raise ValueError("t_final must be positive.")
    if nx < 2:
        raise ValueError("nx must be >= 2.")
    if nt < 1:
        raise ValueError("nt must be >= 1.")

    x = np.linspace(0.0, 1.0, nx + 1, dtype=float)
    dx = 1.0 / nx
    dt = t_final / nt
    r = alpha * dt / (dx * dx)

    # Interior nodes only (Dirichlet boundaries are always zero).
    u = initial_condition(x[1:-1])
    m = u.size

    # Left matrix: (I - 0.5*r*L)
    lower_l = -0.5 * r * np.ones(m - 1, dtype=float)
    diag_l = (1.0 + r) * np.ones(m, dtype=float)
    upper_l = -0.5 * r * np.ones(m - 1, dtype=float)

    # Right matrix: (I + 0.5*r*L)
    lower_r = 0.5 * r * np.ones(m - 1, dtype=float)
    diag_r = (1.0 - r) * np.ones(m, dtype=float)
    upper_r = 0.5 * r * np.ones(m - 1, dtype=float)

    snapshot_steps = {0, nt // 2, nt}
    snapshots: List[Tuple[int, float, float]] = []

    u_full_0 = np.zeros(nx + 1, dtype=float)
    u_full_0[1:-1] = u
    err0 = l2_error(u_full_0, exact_solution(x, 0.0, alpha), dx)
    snapshots.append((0, 0.0, err0))

    for n in range(1, nt + 1):
        rhs = diag_r * u
        if m > 1:
            rhs[1:] += lower_r * u[:-1]
            rhs[:-1] += upper_r * u[1:]

        u = solve_tridiagonal(lower_l, diag_l, upper_l, rhs)

        if n in snapshot_steps:
            t_n = n * dt
            u_full = np.zeros(nx + 1, dtype=float)
            u_full[1:-1] = u
            ref = exact_solution(x, t_n, alpha)
            snapshots.append((n, t_n, l2_error(u_full, ref, dx)))

    u_final = np.zeros(nx + 1, dtype=float)
    u_final[1:-1] = u
    u_exact = exact_solution(x, t_final, alpha)

    return RunResult(
        nx=nx,
        nt=nt,
        dx=dx,
        dt=dt,
        r=r,
        l2_error=l2_error(u_final, u_exact, dx),
        max_error=float(np.max(np.abs(u_final - u_exact))),
        snapshots=sorted(snapshots, key=lambda x: x[0]),
    )


def print_convergence_table(results: List[RunResult]) -> None:
    print("Crank-Nicolson 1D Heat Equation Convergence")
    print("=" * 72)
    print(
        f"{'nx':>6} {'nt':>6} {'dx':>10} {'dt':>10} {'r':>10} "
        f"{'L2 error':>14} {'Linf error':>14}"
    )
    print("-" * 72)
    for item in results:
        print(
            f"{item.nx:6d} {item.nt:6d} {item.dx:10.4e} {item.dt:10.4e} "
            f"{item.r:10.4e} {item.l2_error:14.6e} {item.max_error:14.6e}"
        )
    print("-" * 72)
    print("Observed order (using successive L2 errors):")
    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]
        order = math.log(prev.l2_error / curr.l2_error) / math.log(prev.dx / curr.dx)
        print(
            f"  ({prev.nx:>3d}->{curr.nx:>3d}) "
            f"order ≈ {order:.3f}"
        )


def print_snapshots(result: RunResult) -> None:
    print("\nSelected snapshot L2 errors:")
    for step, t_now, err in result.snapshots:
        print(f"  step={step:5d}, t={t_now:9.5f}, L2 error={err:.6e}")


def main() -> None:
    alpha = 1.0
    t_final = 0.5
    grid_cases = [(40, 400), (80, 800), (160, 1600)]

    results: List[RunResult] = []
    for nx, nt in grid_cases:
        results.append(crank_nicolson_heat_1d(alpha=alpha, t_final=t_final, nx=nx, nt=nt))

    print_convergence_table(results)
    print_snapshots(results[-1])


if __name__ == "__main__":
    main()
