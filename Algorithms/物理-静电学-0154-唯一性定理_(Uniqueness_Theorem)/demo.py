"""Numerical MVP for the electrostatic Uniqueness Theorem.

Problem setup:
- 2D square domain [0, 1] x [0, 1]
- Dirichlet boundary condition:
  * V(x, 0) = 0
  * V(0, y) = 0
  * V(1, y) = 0
  * V(x, 1) = sin(pi x)

We solve Laplace's equation ∇²V = 0 twice with different initial guesses.
The uniqueness theorem predicts both converged solutions must be identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SolveResult:
    potential: np.ndarray
    iterations: int
    final_delta: float


def build_dirichlet_boundary(n: int) -> np.ndarray:
    """Create boundary values for the square domain."""
    if n < 5:
        raise ValueError("n must be >= 5")

    v = np.zeros((n, n), dtype=float)
    x = np.linspace(0.0, 1.0, n)
    v[-1, :] = np.sin(math.pi * x)  # top edge: y = 1
    return v


def analytic_solution(n: int) -> np.ndarray:
    """Closed-form harmonic solution for the chosen boundary condition."""
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y)
    return np.sin(math.pi * xx) * np.sinh(math.pi * yy) / np.sinh(math.pi)


def solve_laplace_jacobi(
    boundary_values: np.ndarray,
    *,
    init_mode: str,
    max_iter: int = 40_000,
    tol: float = 1e-12,
) -> SolveResult:
    """Solve ∇²V=0 with fixed Dirichlet boundary via Jacobi iteration."""
    n = boundary_values.shape[0]
    if boundary_values.shape != (n, n):
        raise ValueError("boundary_values must be square")

    if init_mode == "zeros":
        v = np.zeros_like(boundary_values)
    elif init_mode == "random":
        rng = np.random.default_rng(42)
        v = rng.uniform(-0.8, 0.8, size=boundary_values.shape)
    else:
        raise ValueError("init_mode must be 'zeros' or 'random'")

    # Enforce Dirichlet boundaries from the start.
    v[0, :] = boundary_values[0, :]
    v[-1, :] = boundary_values[-1, :]
    v[:, 0] = boundary_values[:, 0]
    v[:, -1] = boundary_values[:, -1]

    for it in range(1, max_iter + 1):
        vn = v.copy()
        vn[1:-1, 1:-1] = 0.25 * (
            v[:-2, 1:-1] + v[2:, 1:-1] + v[1:-1, :-2] + v[1:-1, 2:]
        )

        # Re-pin boundaries each iteration.
        vn[0, :] = boundary_values[0, :]
        vn[-1, :] = boundary_values[-1, :]
        vn[:, 0] = boundary_values[:, 0]
        vn[:, -1] = boundary_values[:, -1]

        delta = float(np.max(np.abs(vn - v)))
        v = vn
        if delta < tol:
            return SolveResult(potential=v, iterations=it, final_delta=delta)

    return SolveResult(potential=v, iterations=max_iter, final_delta=delta)


def main() -> None:
    n = 81
    boundary = build_dirichlet_boundary(n)
    exact = analytic_solution(n)

    result_a = solve_laplace_jacobi(boundary, init_mode="zeros")
    result_b = solve_laplace_jacobi(boundary, init_mode="random")

    # Difference between two converged numerical solutions.
    uniqueness_gap = float(np.max(np.abs(result_a.potential - result_b.potential)))

    # Error against analytic harmonic solution.
    err_a = float(np.max(np.abs(result_a.potential - exact)))
    err_b = float(np.max(np.abs(result_b.potential - exact)))

    print("=== Uniqueness Theorem Numerical Check ===")
    print(f"Grid size: {n}x{n}")
    print(f"Run A (zeros init): iterations={result_a.iterations}, final_delta={result_a.final_delta:.3e}")
    print(f"Run B (random init): iterations={result_b.iterations}, final_delta={result_b.final_delta:.3e}")
    print(f"Max |V_A - V_B| (same boundary): {uniqueness_gap:.3e}")
    print(f"Max |V_A - V_exact|: {err_a:.3e}")
    print(f"Max |V_B - V_exact|: {err_b:.3e}")

    verdict = "PASS" if uniqueness_gap < 2e-8 else "CHECK"
    print(f"Uniqueness empirical verdict: {verdict}")


if __name__ == "__main__":
    main()
