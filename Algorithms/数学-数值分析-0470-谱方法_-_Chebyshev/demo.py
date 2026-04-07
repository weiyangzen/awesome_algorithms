"""Minimal runnable MVP for Chebyshev spectral method (MATH-0470)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class SolveResult:
    """Container for one Chebyshev-collocation solve."""

    n: int
    x: np.ndarray
    u_num: np.ndarray
    u_exact: np.ndarray
    error_inf: float


def chebyshev_gauss_lobatto_nodes(n: int) -> np.ndarray:
    """Return x_j = cos(pi*j/n), j=0..n, on [-1, 1]."""
    if n < 2:
        raise ValueError("n must be >= 2")
    j = np.arange(n + 1, dtype=float)
    return np.cos(np.pi * j / n)


def chebyshev_diff_matrix(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Construct first-order Chebyshev differentiation matrix on CGL nodes."""
    x = chebyshev_gauss_lobatto_nodes(n)

    k = np.arange(n + 1)
    c = np.ones(n + 1, dtype=float)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** k)

    dx = x[:, None] - x[None, :]
    d = (np.outer(c, 1.0 / c)) / (dx + np.eye(n + 1))
    d = d - np.diag(np.sum(d, axis=1))

    if not np.all(np.isfinite(d)):
        raise RuntimeError("differentiation matrix contains non-finite values")
    return x, d


def exact_solution(x: np.ndarray) -> np.ndarray:
    """Manufactured exact solution for verification."""
    return np.exp(x) * np.sin(3.0 * x)


def rhs_for_poisson(x: np.ndarray) -> np.ndarray:
    """Right-hand side for u'' = f corresponding to exact_solution."""
    # u(x) = e^x sin(3x) => u''(x) = e^x * (6cos(3x) - 8sin(3x))
    return np.exp(x) * (6.0 * np.cos(3.0 * x) - 8.0 * np.sin(3.0 * x))


def solve_poisson_dirichlet_chebyshev(n: int) -> SolveResult:
    """Solve u''(x)=f(x), x in [-1,1], with Dirichlet values from exact solution."""
    x, d1 = chebyshev_diff_matrix(n)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        d2 = d1 @ d1

    if not np.all(np.isfinite(d2)):
        raise RuntimeError("second-derivative matrix contains non-finite values")

    u_exact = exact_solution(x)
    f = rhs_for_poisson(x)

    # Interior unknowns u[1:n]: D2_ii * u_i = f_i - D2_ib * u_b
    a = d2[1:n, 1:n].copy()
    rhs = f[1:n].copy()
    rhs -= d2[1:n, 0] * u_exact[0]
    rhs -= d2[1:n, n] * u_exact[n]

    u_num = np.empty(n + 1, dtype=float)
    u_num[0] = u_exact[0]
    u_num[n] = u_exact[n]
    u_num[1:n] = np.linalg.solve(a, rhs)

    error_inf = float(np.max(np.abs(u_num - u_exact)))
    return SolveResult(n=n, x=x, u_num=u_num, u_exact=u_exact, error_inf=error_inf)


def polynomial_derivative_selfcheck() -> None:
    """For polynomial degree <= n, Dp should match p' up to roundoff."""
    n = 12
    x, d1 = chebyshev_diff_matrix(n)
    p = x**4 - 2.0 * x**3 + x
    dp_exact = 4.0 * x**3 - 6.0 * x**2 + 1.0
    dp_num = d1 @ p
    err = float(np.max(np.abs(dp_num - dp_exact)))
    print(f"Polynomial derivative self-check (N={n}): inf-norm error = {err:.3e}")
    assert err < 1e-10, f"polynomial derivative check failed: {err}"


def run_convergence_demo() -> Tuple[List[int], List[float]]:
    """Show spectral convergence on a smooth manufactured solution."""
    n_list = [8, 10, 12, 14, 16]
    errs: List[float] = []

    print("\nChebyshev collocation for Poisson BVP")
    print("N      inf-norm error        prev/cur")
    for i, n in enumerate(n_list):
        result = solve_poisson_dirichlet_chebyshev(n)
        errs.append(result.error_inf)
        ratio = "-"
        if i > 0 and result.error_inf > 0.0:
            ratio = f"{errs[i - 1] / result.error_inf:.3e}"
        print(f"{n:<6d} {result.error_inf:<21.6e} {ratio}")

    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)), errs
    assert errs[-1] < 1e-10, errs[-1]
    assert errs[0] / errs[-1] > 1e4, (errs[0], errs[-1])
    return n_list, errs


def main() -> None:
    print("Chebyshev Spectral Method MVP (MATH-0470)")
    print("=" * 72)
    polynomial_derivative_selfcheck()
    run_convergence_demo()
    print("=" * 72)
    print("All checks passed.")


if __name__ == "__main__":
    main()
