"""Multigrid MVP for 2D Poisson equation on a unit square.

We solve:
    -Laplace(u) = f, in (0,1)^2
    u = 0, on boundary

Discretization:
- 5-point finite difference stencil on interior points only.
- V-cycle multigrid with weighted Jacobi smoother.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def apply_operator(u: np.ndarray, h: float) -> np.ndarray:
    """Apply 5-point Laplacian operator A to interior field u: A u = f."""
    up = np.pad(u, 1, mode="constant", constant_values=0.0)
    return (
        4.0 * up[1:-1, 1:-1]
        - up[:-2, 1:-1]
        - up[2:, 1:-1]
        - up[1:-1, :-2]
        - up[1:-1, 2:]
    ) / (h * h)


def weighted_jacobi(
    u: np.ndarray,
    f: np.ndarray,
    h: float,
    omega: float = 2.0 / 3.0,
    sweeps: int = 3,
) -> np.ndarray:
    """Weighted Jacobi smoother for A u = f."""
    out = u.copy()
    h2 = h * h
    for _ in range(sweeps):
        up = np.pad(out, 1, mode="constant", constant_values=0.0)
        neighbor_sum = (
            up[:-2, 1:-1] + up[2:, 1:-1] + up[1:-1, :-2] + up[1:-1, 2:]
        )
        jacobi_update = 0.25 * (neighbor_sum + h2 * f)
        out = (1.0 - omega) * out + omega * jacobi_update
    return out


def compute_residual(u: np.ndarray, f: np.ndarray, h: float) -> np.ndarray:
    """Residual r = f - A u."""
    return f - apply_operator(u, h)


def restrict_full_weighting(rf: np.ndarray) -> np.ndarray:
    """Full-weighting restriction from fine grid (2n+1) to coarse grid (n)."""
    nf = rf.shape[0]
    if nf % 2 == 0 or rf.shape[1] != nf:
        raise ValueError("Fine residual must be square with odd size.")

    nc = (nf - 1) // 2
    rc = np.zeros((nc, nc), dtype=rf.dtype)

    for ic in range(nc):
        i = 2 * ic + 1
        for jc in range(nc):
            j = 2 * jc + 1
            center = 4.0 * rf[i, j]
            axial = 2.0 * (
                rf[i - 1, j] + rf[i + 1, j] + rf[i, j - 1] + rf[i, j + 1]
            )
            diag = rf[i - 1, j - 1] + rf[i - 1, j + 1] + rf[i + 1, j - 1] + rf[i + 1, j + 1]
            rc[ic, jc] = (center + axial + diag) / 16.0
    return rc


def prolong_bilinear(ec: np.ndarray) -> np.ndarray:
    """Bilinear prolongation from coarse grid (n) to fine grid (2n+1)."""
    nc = ec.shape[0]
    if ec.shape[1] != nc:
        raise ValueError("Coarse correction must be square.")

    nf = 2 * nc + 1
    ef = np.zeros((nf, nf), dtype=ec.dtype)

    for ic in range(nc):
        i = 2 * ic + 1
        for jc in range(nc):
            j = 2 * jc + 1
            val = ec[ic, jc]

            ef[i, j] += val

            if i - 1 >= 0:
                ef[i - 1, j] += 0.5 * val
            if i + 1 < nf:
                ef[i + 1, j] += 0.5 * val
            if j - 1 >= 0:
                ef[i, j - 1] += 0.5 * val
            if j + 1 < nf:
                ef[i, j + 1] += 0.5 * val

            if i - 1 >= 0 and j - 1 >= 0:
                ef[i - 1, j - 1] += 0.25 * val
            if i - 1 >= 0 and j + 1 < nf:
                ef[i - 1, j + 1] += 0.25 * val
            if i + 1 < nf and j - 1 >= 0:
                ef[i + 1, j - 1] += 0.25 * val
            if i + 1 < nf and j + 1 < nf:
                ef[i + 1, j + 1] += 0.25 * val

    return ef


def direct_solve_coarsest(f: np.ndarray, h: float) -> np.ndarray:
    """Direct solve on a tiny coarse grid using dense linear algebra."""
    n = f.shape[0]
    if f.shape[1] != n:
        raise ValueError("f must be square")

    size = n * n
    a = np.zeros((size, size), dtype=float)
    b = f.reshape(-1).copy()
    inv_h2 = 1.0 / (h * h)

    def idx(i: int, j: int) -> int:
        return i * n + j

    for i in range(n):
        for j in range(n):
            row = idx(i, j)
            a[row, row] = 4.0 * inv_h2
            if i > 0:
                a[row, idx(i - 1, j)] = -inv_h2
            if i < n - 1:
                a[row, idx(i + 1, j)] = -inv_h2
            if j > 0:
                a[row, idx(i, j - 1)] = -inv_h2
            if j < n - 1:
                a[row, idx(i, j + 1)] = -inv_h2

    x = np.linalg.solve(a, b)
    return x.reshape(n, n)


def v_cycle(
    u: np.ndarray,
    f: np.ndarray,
    h: float,
    pre_sweeps: int = 3,
    post_sweeps: int = 3,
    omega: float = 2.0 / 3.0,
) -> np.ndarray:
    """One recursive V-cycle."""
    n = u.shape[0]
    if u.shape != f.shape or u.shape[1] != n:
        raise ValueError("u and f must be square and same shape")

    if n <= 3:
        return direct_solve_coarsest(f, h)

    u = weighted_jacobi(u, f, h, omega=omega, sweeps=pre_sweeps)

    r = compute_residual(u, f, h)
    rc = restrict_full_weighting(r)

    ec0 = np.zeros_like(rc)
    ec = v_cycle(ec0, rc, 2.0 * h, pre_sweeps, post_sweeps, omega)

    u += prolong_bilinear(ec)

    u = weighted_jacobi(u, f, h, omega=omega, sweeps=post_sweeps)
    return u


def l2_norm(arr: np.ndarray) -> float:
    return float(np.linalg.norm(arr.ravel(), ord=2))


def setup_poisson_problem(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build test problem with known exact solution u*=sin(pi x)sin(pi y)."""
    h = 1.0 / (n + 1)
    xs = np.linspace(h, 1.0 - h, n)
    ys = np.linspace(h, 1.0 - h, n)
    xg, yg = np.meshgrid(xs, ys, indexing="ij")

    u_true = np.sin(math.pi * xg) * np.sin(math.pi * yg)
    f = 2.0 * math.pi**2 * u_true
    u0 = np.zeros_like(u_true)
    return u0, f, u_true, h


def main() -> None:
    n = 63  # must be 2^k - 1 for this simple restriction/prolongation pair
    cycles = 10

    u, f, u_true, h = setup_poisson_problem(n)

    print(f"Multigrid MVP: n={n}x{n}, h={h:.6f}, cycles={cycles}")

    initial_residual = l2_norm(compute_residual(u, f, h))
    print(f"Initial residual L2: {initial_residual:.6e}")

    for k in range(1, cycles + 1):
        u = v_cycle(u, f, h)
        residual = l2_norm(compute_residual(u, f, h))
        rel = residual / initial_residual
        err = l2_norm(u - u_true) / l2_norm(u_true)
        print(
            f"Cycle {k:02d}: residual={residual:.6e}, "
            f"residual_ratio={rel:.6e}, relative_error={err:.6e}"
        )


if __name__ == "__main__":
    main()
