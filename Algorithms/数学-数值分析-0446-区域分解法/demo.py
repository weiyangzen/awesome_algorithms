"""Domain Decomposition Method MVP (overlapping Schwarz iteration).

Model problem (1D Poisson):
    -u''(x) = pi^2 sin(pi x), x in (0, 1)
    u(0) = u(1) = 0
Exact solution:
    u(x) = sin(pi x)

We discretize the PDE on interior grid points and solve the linear system with
an overlapping two-subdomain multiplicative Schwarz iteration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DDConfig:
    n_interior: int = 120
    overlap: int = 8
    max_iterations: int = 80
    tolerance: float = 1e-10

    def validate(self) -> None:
        if self.n_interior < 8:
            raise ValueError("n_interior must be >= 8 for this MVP.")
        if self.overlap < 1:
            raise ValueError("overlap must be >= 1.")
        if self.overlap >= self.n_interior // 2:
            raise ValueError("overlap is too large; require overlap < n_interior/2.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")


@dataclass
class IterationStat:
    iteration: int
    relative_residual: float
    relative_error_to_discrete_solution: float
    relative_error_to_exact_solution: float


@dataclass
class DDResult:
    config: DDConfig
    h: float
    x: np.ndarray
    subdomain_1: tuple[int, int]
    subdomain_2: tuple[int, int]
    u: np.ndarray
    u_discrete_exact: np.ndarray
    u_exact_continuous: np.ndarray
    history: list[IterationStat]


def build_problem(n_interior: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    h = 1.0 / (n_interior + 1)
    x = np.linspace(h, 1.0 - h, n_interior)
    rhs = (math.pi**2) * np.sin(math.pi * x)
    u_exact = np.sin(math.pi * x)
    return h, x, rhs, u_exact


def solve_tridiagonal(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Thomas algorithm for tridiagonal linear systems."""
    n = diag.size
    if n == 0:
        raise ValueError("diag must be non-empty")
    if rhs.size != n:
        raise ValueError("rhs size mismatch")
    if n == 1:
        if abs(diag[0]) < 1e-15:
            raise ValueError("singular 1x1 system")
        return rhs / diag
    if lower.size != n - 1 or upper.size != n - 1:
        raise ValueError("lower/upper size mismatch")

    c_prime = np.empty(n - 1, dtype=float)
    d_prime = np.empty(n, dtype=float)

    denom = float(diag[0])
    if abs(denom) < 1e-15:
        raise ValueError("zero pivot at row 0")
    c_prime[0] = upper[0] / denom
    d_prime[0] = rhs[0] / denom

    for i in range(1, n - 1):
        denom = float(diag[i] - lower[i - 1] * c_prime[i - 1])
        if abs(denom) < 1e-15:
            raise ValueError(f"zero pivot at row {i}")
        c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    denom = float(diag[n - 1] - lower[n - 2] * c_prime[n - 2])
    if abs(denom) < 1e-15:
        raise ValueError(f"zero pivot at row {n - 1}")
    d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / denom

    x = np.empty(n, dtype=float)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def solve_discrete_poisson_exact(rhs: np.ndarray, h: float) -> np.ndarray:
    n = rhs.size
    inv_h2 = 1.0 / (h * h)
    lower = np.full(n - 1, -inv_h2, dtype=float)
    diag = np.full(n, 2.0 * inv_h2, dtype=float)
    upper = np.full(n - 1, -inv_h2, dtype=float)
    return solve_tridiagonal(lower, diag, upper, rhs)


def split_into_two_overlapping_subdomains(
    n_interior: int,
    overlap: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    mid = n_interior // 2
    left_end = min(n_interior - 1, mid + overlap - 1)
    right_start = max(0, mid - overlap)

    sub_1 = (0, left_end)
    sub_2 = (right_start, n_interior - 1)

    if sub_1[1] < sub_2[0]:
        raise ValueError("subdomains do not overlap")
    return sub_1, sub_2


def solve_local_subproblem(
    u_global: np.ndarray,
    rhs_global: np.ndarray,
    h: float,
    start: int,
    end: int,
) -> np.ndarray:
    """Solve a subdomain Poisson system with interface Dirichlet values."""
    n = u_global.size
    if not (0 <= start <= end < n):
        raise ValueError("invalid subdomain range")

    m = end - start + 1
    inv_h2 = 1.0 / (h * h)

    rhs_local = rhs_global[start : end + 1].copy()
    left_bc = 0.0 if start == 0 else float(u_global[start - 1])
    right_bc = 0.0 if end == n - 1 else float(u_global[end + 1])

    rhs_local[0] += left_bc * inv_h2
    rhs_local[-1] += right_bc * inv_h2

    lower = np.full(m - 1, -inv_h2, dtype=float)
    diag = np.full(m, 2.0 * inv_h2, dtype=float)
    upper = np.full(m - 1, -inv_h2, dtype=float)

    return solve_tridiagonal(lower, diag, upper, rhs_local)


def relative_residual(u: np.ndarray, rhs: np.ndarray, h: float) -> float:
    inv_h2 = 1.0 / (h * h)
    au = 2.0 * inv_h2 * u
    au[:-1] += -inv_h2 * u[1:]
    au[1:] += -inv_h2 * u[:-1]

    r = au - rhs
    denom = np.linalg.norm(rhs)
    if denom == 0.0:
        return float(np.linalg.norm(r))
    return float(np.linalg.norm(r) / denom)


def run_multiplicative_schwarz(config: DDConfig) -> DDResult:
    config.validate()

    h, x, rhs, u_exact = build_problem(config.n_interior)
    sub_1, sub_2 = split_into_two_overlapping_subdomains(
        config.n_interior,
        config.overlap,
    )

    u_discrete_exact = solve_discrete_poisson_exact(rhs, h)
    u = np.zeros_like(rhs)

    norm_discrete = float(np.linalg.norm(u_discrete_exact))
    norm_exact = float(np.linalg.norm(u_exact))

    history: list[IterationStat] = []

    for k in range(1, config.max_iterations + 1):
        local_1 = solve_local_subproblem(u, rhs, h, sub_1[0], sub_1[1])
        u[sub_1[0] : sub_1[1] + 1] = local_1

        local_2 = solve_local_subproblem(u, rhs, h, sub_2[0], sub_2[1])
        u[sub_2[0] : sub_2[1] + 1] = local_2

        residual = relative_residual(u, rhs, h)
        error_to_discrete = float(np.linalg.norm(u - u_discrete_exact) / norm_discrete)
        error_to_exact = float(np.linalg.norm(u - u_exact) / norm_exact)

        history.append(
            IterationStat(
                iteration=k,
                relative_residual=residual,
                relative_error_to_discrete_solution=error_to_discrete,
                relative_error_to_exact_solution=error_to_exact,
            )
        )

        if residual <= config.tolerance:
            break

    return DDResult(
        config=config,
        h=h,
        x=x,
        subdomain_1=sub_1,
        subdomain_2=sub_2,
        u=u,
        u_discrete_exact=u_discrete_exact,
        u_exact_continuous=u_exact,
        history=history,
    )


def print_report(result: DDResult) -> None:
    c = result.config
    print("Domain Decomposition (Multiplicative Schwarz) demo")
    print(
        f"n_interior={c.n_interior}, overlap={c.overlap}, "
        f"max_iterations={c.max_iterations}, tolerance={c.tolerance:.1e}"
    )
    print(
        f"subdomain_1={result.subdomain_1}, subdomain_2={result.subdomain_2}, h={result.h:.6f}"
    )
    print()

    print(
        f"{'iter':>6} {'rel_residual':>16} {'rel_err_discrete':>18} {'rel_err_exact':>16}"
    )
    for row in result.history:
        print(
            f"{row.iteration:6d} {row.relative_residual:16.6e} "
            f"{row.relative_error_to_discrete_solution:18.6e} "
            f"{row.relative_error_to_exact_solution:16.6e}"
        )

    final = result.history[-1]
    print()
    print(
        "Final summary: "
        f"iterations={final.iteration}, "
        f"residual={final.relative_residual:.6e}, "
        f"error_to_discrete={final.relative_error_to_discrete_solution:.6e}, "
        f"error_to_exact={final.relative_error_to_exact_solution:.6e}"
    )


def run_checks(result: DDResult) -> None:
    if not result.history:
        raise RuntimeError("No iterations were executed.")

    residuals = np.array([h.relative_residual for h in result.history], dtype=float)
    discrete_errors = np.array(
        [h.relative_error_to_discrete_solution for h in result.history],
        dtype=float,
    )

    if not np.all(np.isfinite(residuals)):
        raise AssertionError("Residual sequence contains non-finite values.")
    if not np.all(np.isfinite(discrete_errors)):
        raise AssertionError("Error sequence contains non-finite values.")

    final = result.history[-1]
    if final.relative_residual > 1e-9:
        raise AssertionError(
            f"Residual is too large at termination: {final.relative_residual:.3e}"
        )
    if final.relative_error_to_discrete_solution > 1e-8:
        raise AssertionError(
            "Schwarz iteration did not approach the discrete reference solution "
            f"enough: {final.relative_error_to_discrete_solution:.3e}"
        )
    if final.relative_residual >= residuals[0]:
        raise AssertionError("Residual did not improve.")


def main() -> None:
    config = DDConfig(
        n_interior=120,
        overlap=8,
        max_iterations=80,
        tolerance=1e-10,
    )
    result = run_multiplicative_schwarz(config)
    print_report(result)
    run_checks(result)
    print("All checks passed.")


if __name__ == "__main__":
    main()
