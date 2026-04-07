"""Minimal runnable MVP for ADI (Alternating Direction Implicit) method.

This script solves the 2D heat equation on [0,1]x[0,1] with homogeneous
Dirichlet boundary condition:
    u_t = alpha * (u_xx + u_yy)
using the Peaceman-Rachford ADI splitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.linalg import solve_banded as scipy_solve_banded
except Exception:  # pragma: no cover - fallback for environments without scipy
    scipy_solve_banded = None


@dataclass
class StepRecord:
    """Diagnostics snapshot for one full ADI time step."""

    step: int
    time: float
    l2_error: float
    linf_error: float
    interior_energy: float


def build_tridiagonal_banded(n: int, r: float) -> np.ndarray:
    """Build banded matrix for I - r*T where T is 1D second-difference stencil."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if r < 0:
        raise ValueError(f"r must be non-negative, got {r}")

    ab = np.zeros((3, n), dtype=np.float64)
    ab[0, 1:] = -r
    ab[1, :] = 1.0 + 2.0 * r
    ab[2, :-1] = -r
    return ab


def thomas_solve_banded(ab: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve tridiagonal system(s) using Thomas algorithm.

    Supports b with shape (n,) or (n, m), solving A X = b where A is encoded by ab.
    """
    n = ab.shape[1]
    if b.ndim == 1:
        b2 = b.reshape(-1, 1).astype(np.float64, copy=True)
        squeeze = True
    elif b.ndim == 2:
        b2 = b.astype(np.float64, copy=True)
        squeeze = False
    else:
        raise ValueError(f"b must be 1D or 2D, got ndim={b.ndim}")

    if b2.shape[0] != n:
        raise ValueError(f"b row count {b2.shape[0]} does not match matrix size {n}")

    upper = ab[0, 1:].copy()
    diag = ab[1, :].copy()
    lower = ab[2, :-1].copy()

    for i in range(1, n):
        w = lower[i - 1] / diag[i - 1]
        diag[i] -= w * upper[i - 1]
        b2[i, :] -= w * b2[i - 1, :]

    x = np.empty_like(b2)
    x[-1, :] = b2[-1, :] / diag[-1]
    for i in range(n - 2, -1, -1):
        x[i, :] = (b2[i, :] - upper[i] * x[i + 1, :]) / diag[i]

    if squeeze:
        return x[:, 0]
    return x


def solve_tridiagonal(ab: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dispatch to SciPy's banded solver if available, otherwise Thomas fallback."""
    if scipy_solve_banded is not None:
        return scipy_solve_banded((1, 1), ab, b, check_finite=False)
    return thomas_solve_banded(ab, b)


def laplacian_x(u: np.ndarray) -> np.ndarray:
    """Apply 1D second-difference along x on an interior grid (zero BC outside)."""
    out = -2.0 * u.copy()
    out[1:, :] += u[:-1, :]
    out[:-1, :] += u[1:, :]
    return out


def laplacian_y(u: np.ndarray) -> np.ndarray:
    """Apply 1D second-difference along y on an interior grid (zero BC outside)."""
    out = -2.0 * u.copy()
    out[:, 1:] += u[:, :-1]
    out[:, :-1] += u[:, 1:]
    return out


def exact_solution(x: np.ndarray, y: np.ndarray, t: float, alpha: float) -> np.ndarray:
    """Analytical solution for u0=sin(pi x)sin(pi y) with zero Dirichlet BC."""
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mode = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    decay = np.exp(-2.0 * (np.pi**2) * alpha * t)
    return decay * mode


def one_adi_step(u: np.ndarray, rx: float, ry: float, abx: np.ndarray, aby: np.ndarray) -> np.ndarray:
    """Run one Peaceman-Rachford ADI full step."""
    # Half-step: implicit in x, explicit in y.
    rhs_half = u + ry * laplacian_y(u)
    u_half = solve_tridiagonal(abx, rhs_half)

    # Full-step: implicit in y, explicit in x.
    rhs_full = u_half + rx * laplacian_x(u_half)
    u_next_t = solve_tridiagonal(aby, rhs_full.T)
    u_next = u_next_t.T
    return u_next


def run_adi_heat(
    n_points: int,
    alpha: float,
    dt: float,
    n_steps: int,
) -> tuple[np.ndarray, list[StepRecord], float, float]:
    """Solve 2D heat equation and return full-grid solution and diagnostics."""
    if n_points < 3:
        raise ValueError("n_points must be >= 3")
    if alpha <= 0 or dt <= 0 or n_steps <= 0:
        raise ValueError("alpha, dt, n_steps must be positive")

    x = np.linspace(0.0, 1.0, n_points)
    y = np.linspace(0.0, 1.0, n_points)
    h = 1.0 / (n_points - 1)

    nx = n_points - 2
    ny = n_points - 2

    rx = alpha * dt / (2.0 * h * h)
    ry = alpha * dt / (2.0 * h * h)

    abx = build_tridiagonal_banded(nx, rx)
    aby = build_tridiagonal_banded(ny, ry)

    u_full0 = exact_solution(x, y, t=0.0, alpha=alpha)
    u = u_full0[1:-1, 1:-1].copy()

    records: list[StepRecord] = []
    for k in range(1, n_steps + 1):
        t_now = k * dt
        u = one_adi_step(u=u, rx=rx, ry=ry, abx=abx, aby=aby)

        u_full = np.zeros((n_points, n_points), dtype=np.float64)
        u_full[1:-1, 1:-1] = u

        u_exact = exact_solution(x, y, t=t_now, alpha=alpha)
        err = u_full - u_exact

        l2_error = float(np.sqrt(np.mean(err * err)))
        linf_error = float(np.max(np.abs(err)))
        interior_energy = float(np.mean(u * u))

        records.append(
            StepRecord(
                step=k,
                time=t_now,
                l2_error=l2_error,
                linf_error=linf_error,
                interior_energy=interior_energy,
            )
        )

    u_full_final = np.zeros((n_points, n_points), dtype=np.float64)
    u_full_final[1:-1, 1:-1] = u
    return u_full_final, records, rx, ry


def print_trace(records: list[StepRecord], limit: int = 8) -> None:
    """Print compact step diagnostics."""
    print("step | time     | l2_error   | linf_error | interior_energy")
    print("-----+----------+------------+------------+----------------")

    if len(records) <= limit:
        rows = records
    else:
        head = max(1, limit // 2)
        tail = max(1, limit - head)
        rows = records[:head] + records[-tail:]

    shown_steps = {r.step for r in rows}
    for row in rows:
        print(
            f"{row.step:4d} | {row.time: .4f} | {row.l2_error: .3e} |"
            f" {row.linf_error: .3e} | {row.interior_energy: .3e}"
        )

    if len(shown_steps) < len(records):
        print(" ...  (middle steps omitted)")


def main() -> None:
    """Run deterministic ADI MVP without interactive input."""
    n_points = 41
    alpha = 0.1
    dt = 0.002
    n_steps = 50

    print("=== ADI Method MVP: 2D Heat Equation ===")
    print(f"grid: {n_points} x {n_points}")
    print(f"alpha={alpha}, dt={dt}, n_steps={n_steps}, final_time={n_steps * dt:.4f}")

    u_final, records, rx, ry = run_adi_heat(
        n_points=n_points,
        alpha=alpha,
        dt=dt,
        n_steps=n_steps,
    )

    final = records[-1]

    print(f"rx={rx:.6f}, ry={ry:.6f}")
    print(f"final l2 error:   {final.l2_error:.6e}")
    print(f"final linf error: {final.linf_error:.6e}")
    print(f"center value u(0.5,0.5) ~= {u_final[n_points // 2, n_points // 2]:.8f}")

    print("\nTrace:")
    print_trace(records, limit=10)

    # A modest threshold for this coarse-grid MVP demonstration.
    if final.linf_error > 1.0e-2:
        raise AssertionError(f"ADI MVP accuracy check failed: linf={final.linf_error:.3e}")

    print("\nADI MVP run finished successfully.")


if __name__ == "__main__":
    main()
