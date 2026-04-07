"""Implicit Euler MVP for a PDE (1D heat equation with Dirichlet BCs).

We solve:
    u_t = alpha * u_xx,  x in (0, 1), t > 0
    u(0, t) = u(1, t) = 0
    u(x, 0) = sin(pi x)

Spatial discretization uses second-order central difference on a uniform grid.
Time integration uses implicit Euler:
    (I - r * L) u^{n+1} = u^n,
where r = alpha * dt / dx^2 and L is the 1D Laplacian stencil.

Because this leads to a tridiagonal linear system, we solve each step
with a source-visible Thomas algorithm (no black-box PDE solver).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np


@dataclass
class Snapshot:
    step: int
    time: float
    l2_error: float
    linf_error: float
    discrete_energy: float
    max_abs_u: float


def make_grid(nx: int) -> Tuple[np.ndarray, float]:
    """Return uniform 1D grid in [0, 1] and spacing dx."""
    if nx < 3:
        raise ValueError("nx must be >= 3")
    x = np.linspace(0.0, 1.0, nx)
    dx = 1.0 / float(nx - 1)
    return x, dx


def initial_condition(x: np.ndarray) -> np.ndarray:
    """Sine initial condition compatible with homogeneous Dirichlet BCs."""
    return np.sin(math.pi * x)


def exact_solution(x: np.ndarray, t: float, alpha: float) -> np.ndarray:
    """Analytic solution for the selected IC/BC pair."""
    return math.exp(-alpha * math.pi * math.pi * t) * np.sin(math.pi * x)


def assemble_tridiagonal(nx: int, r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build tridiagonal coefficients for interior unknowns."""
    interior_n = nx - 2
    if interior_n < 1:
        raise ValueError("need at least one interior point")
    lower = np.full(interior_n - 1, -r, dtype=float)
    diag = np.full(interior_n, 1.0 + 2.0 * r, dtype=float)
    upper = np.full(interior_n - 1, -r, dtype=float)
    return lower, diag, upper


def thomas_solve(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal linear system using the Thomas algorithm."""
    n = diag.size
    if rhs.size != n:
        raise ValueError("rhs size does not match diagonal size")
    if lower.size != max(0, n - 1) or upper.size != max(0, n - 1):
        raise ValueError("tridiagonal coefficient sizes are inconsistent")

    a = lower.astype(float, copy=True)
    b = diag.astype(float, copy=True)
    c = upper.astype(float, copy=True)
    d = rhs.astype(float, copy=True)

    for i in range(1, n):
        pivot = b[i - 1]
        if abs(pivot) < 1e-15:
            raise RuntimeError(f"zero pivot in Thomas forward pass at i={i}")
        m = a[i - 1] / pivot
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]

    if abs(b[-1]) < 1e-15:
        raise RuntimeError("zero pivot in Thomas backward start")
    x = np.empty(n, dtype=float)
    x[-1] = d[-1] / b[-1]

    for i in range(n - 2, -1, -1):
        if abs(b[i]) < 1e-15:
            raise RuntimeError(f"zero pivot in Thomas backward pass at i={i}")
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def collect_snapshot(
    u: np.ndarray,
    x: np.ndarray,
    t: float,
    alpha: float,
    dx: float,
    step: int,
) -> Snapshot:
    """Compute deterministic diagnostics against exact solution."""
    u_exact = exact_solution(x, t, alpha)
    err = u - u_exact
    l2_error = float(np.sqrt(np.mean(err * err)))
    linf_error = float(np.max(np.abs(err)))
    discrete_energy = float(dx * np.sum(u * u))
    max_abs_u = float(np.max(np.abs(u)))
    return Snapshot(
        step=step,
        time=t,
        l2_error=l2_error,
        linf_error=linf_error,
        discrete_energy=discrete_energy,
        max_abs_u=max_abs_u,
    )


def implicit_euler_heat(
    nx: int,
    alpha: float,
    final_time: float,
    dt_target: float,
    snapshot_count: int = 8,
) -> Tuple[np.ndarray, np.ndarray, List[Snapshot], float, int, float]:
    """Run implicit Euler for 1D heat equation and return diagnostics."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if final_time < 0.0:
        raise ValueError("final_time must be non-negative")
    if dt_target <= 0.0:
        raise ValueError("dt_target must be positive")
    if snapshot_count < 1:
        raise ValueError("snapshot_count must be >= 1")

    x, dx = make_grid(nx)
    u = initial_condition(x)
    u[0] = 0.0
    u[-1] = 0.0

    if final_time == 0.0:
        steps = 1
        dt = 0.0
        r = 0.0
    else:
        steps = max(1, int(math.ceil(final_time / dt_target)))
        dt = final_time / float(steps)
        r = alpha * dt / (dx * dx)

    lower, diag, upper = assemble_tridiagonal(nx, r)

    snapshots: List[Snapshot] = [collect_snapshot(u, x, 0.0, alpha, dx, step=0)]
    stride = max(1, steps // snapshot_count)

    for step in range(1, steps + 1):
        rhs = u[1:-1]
        u_inner_new = thomas_solve(lower, diag, upper, rhs)

        u[1:-1] = u_inner_new
        u[0] = 0.0
        u[-1] = 0.0

        if not np.all(np.isfinite(u)):
            raise RuntimeError(f"non-finite solution encountered at step={step}")

        if (step % stride == 0) or (step == steps):
            t = step * dt
            snapshots.append(collect_snapshot(u, x, t, alpha, dx, step))

    return x, u, snapshots, dt, steps, r


def run_checks(snapshots: List[Snapshot]) -> None:
    """Basic self-checks for monotone diffusion behavior and accuracy sanity."""
    if len(snapshots) < 2:
        raise RuntimeError("need at least two snapshots for checks")

    energies = [s.discrete_energy for s in snapshots]
    for i in range(len(energies) - 1):
        if energies[i + 1] > energies[i] + 1e-10:
            raise RuntimeError(
                "discrete energy is not non-increasing: "
                f"E[{i}]={energies[i]:.8e}, E[{i+1}]={energies[i+1]:.8e}"
            )

    final_l2 = snapshots[-1].l2_error
    if final_l2 > 2.5e-2:
        raise RuntimeError(f"final L2 error too large for MVP baseline: {final_l2:.6e}")


def print_report(
    nx: int,
    alpha: float,
    final_time: float,
    dt: float,
    steps: int,
    r: float,
    snapshots: List[Snapshot],
) -> None:
    """Print deterministic report (no interactive input)."""
    print("Implicit Euler PDE MVP (1D heat equation)")
    print(
        f"nx={nx}, alpha={alpha:.6f}, final_time={final_time:.6f}, "
        f"dt={dt:.6f}, steps={steps}, r=alpha*dt/dx^2={r:.6f}"
    )
    print(
        "{:<8s} {:<10s} {:<14s} {:<14s} {:<14s} {:<14s}".format(
            "step", "time", "L2_error", "Linf_error", "energy", "max|u|"
        )
    )

    for s in snapshots:
        print(
            f"{s.step:<8d} {s.time:<10.5f} {s.l2_error:<14.6e} "
            f"{s.linf_error:<14.6e} {s.discrete_energy:<14.6e} {s.max_abs_u:<14.6e}"
        )

    print(f"Final L2 error: {snapshots[-1].l2_error:.6e}")
    print(f"Final Linf error: {snapshots[-1].linf_error:.6e}")
    print("All checks passed.")


def main() -> None:
    nx = 101
    alpha = 1.0
    final_time = 0.2
    dt_target = 0.002

    _, _, snapshots, dt, steps, r = implicit_euler_heat(
        nx=nx,
        alpha=alpha,
        final_time=final_time,
        dt_target=dt_target,
        snapshot_count=8,
    )

    run_checks(snapshots)
    print_report(
        nx=nx,
        alpha=alpha,
        final_time=final_time,
        dt=dt,
        steps=steps,
        r=r,
        snapshots=snapshots,
    )


if __name__ == "__main__":
    main()
