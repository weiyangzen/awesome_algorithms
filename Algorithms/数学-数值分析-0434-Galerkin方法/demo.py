"""Minimal runnable MVP for Galerkin method (MATH-0434).

Model problem:
    -u''(x) = 1,  x in (0, 1)
    u(0) = u(1) = 0

We use a continuous Galerkin spectral basis:
    phi_k(x) = sin(k * pi * x), k = 1..N
which satisfies the homogeneous Dirichlet boundary conditions exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class GalerkinConfig:
    """Configuration for the non-interactive Galerkin MVP."""

    mode_counts: tuple[int, ...] = (2, 4, 8, 16, 32)
    assemble_quadrature_order: int = 240
    error_quadrature_order: int = 600


@dataclass(frozen=True)
class CaseResult:
    """Diagnostics for one truncation level N."""

    n_modes: int
    l2_error: float
    h1_semi_error: float
    max_abs_error: float
    condition_number: float
    residual_inf: float
    boundary_left: float
    boundary_right: float


def forcing_term(x: Array) -> Array:
    """Right-hand side f(x) in -u''=f."""
    return np.ones_like(x)


def exact_solution(x: Array) -> Array:
    """Analytical solution for -u''=1 with zero Dirichlet BC."""
    return 0.5 * x * (1.0 - x)


def exact_derivative(x: Array) -> Array:
    """Derivative of the analytical solution."""
    return 0.5 - x


def gauss_legendre_unit(order: int) -> tuple[Array, Array]:
    """Gauss-Legendre nodes/weights mapped from [-1,1] to [0,1]."""
    if order < 2:
        raise ValueError("quadrature order must be >= 2")

    xi, w = np.polynomial.legendre.leggauss(order)
    x = 0.5 * (xi + 1.0)
    w01 = 0.5 * w
    return x.astype(float), w01.astype(float)


def sine_basis_matrix(x: Array, n_modes: int) -> Array:
    """Return matrix Phi(q,i) = sin((i+1)*pi*x_q)."""
    k = np.arange(1, n_modes + 1, dtype=float)
    theta = np.pi * x[:, None] * k[None, :]
    return np.sin(theta)


def sine_basis_derivative_matrix(x: Array, n_modes: int) -> Array:
    """Return dPhi(q,i) = (i+1)*pi*cos((i+1)*pi*x_q)."""
    k = np.arange(1, n_modes + 1, dtype=float)
    theta = np.pi * x[:, None] * k[None, :]
    return np.cos(theta) * (np.pi * k[None, :])


def assemble_system(n_modes: int, quadrature_order: int) -> tuple[Array, Array]:
    """Assemble Galerkin stiffness matrix A and load vector b.

    Weak form: find u_N in V_N such that
        int_0^1 u_N'(x) v'(x) dx = int_0^1 f(x) v(x) dx,  for all v in V_N.
    """
    if n_modes <= 0:
        raise ValueError("n_modes must be positive")

    xq, wq = gauss_legendre_unit(quadrature_order)
    phi = sine_basis_matrix(xq, n_modes)
    dphi = sine_basis_derivative_matrix(xq, n_modes)

    a = np.einsum("qi,q,qj->ij", dphi, wq, dphi)
    b = np.einsum("qi,q->i", phi, wq * forcing_term(xq))
    return a, b


def evaluate_series(x: Array, coeff: Array) -> Array:
    phi = sine_basis_matrix(x, coeff.size)
    return np.einsum("qi,i->q", phi, coeff)


def evaluate_series_derivative(x: Array, coeff: Array) -> Array:
    dphi = sine_basis_derivative_matrix(x, coeff.size)
    return np.einsum("qi,i->q", dphi, coeff)


def compute_errors(coeff: Array, quadrature_order: int) -> tuple[float, float, float, float, float]:
    """Return L2 error, H1 seminorm error, max error and boundary values."""
    xq, wq = gauss_legendre_unit(quadrature_order)

    u_num = evaluate_series(xq, coeff)
    du_num = evaluate_series_derivative(xq, coeff)

    u_ref = exact_solution(xq)
    du_ref = exact_derivative(xq)

    l2_error = float(np.sqrt(np.sum(wq * (u_num - u_ref) ** 2)))
    h1_semi_error = float(np.sqrt(np.sum(wq * (du_num - du_ref) ** 2)))
    max_abs_error = float(np.max(np.abs(u_num - u_ref)))

    # Basis functions satisfy boundary conditions exactly.
    left = float(evaluate_series(np.array([0.0]), coeff)[0])
    right = float(evaluate_series(np.array([1.0]), coeff)[0])

    return l2_error, h1_semi_error, max_abs_error, left, right


def run_case(n_modes: int, cfg: GalerkinConfig) -> CaseResult:
    a, b = assemble_system(n_modes=n_modes, quadrature_order=cfg.assemble_quadrature_order)
    coeff = np.linalg.solve(a, b)

    l2_error, h1_error, max_abs_error, left, right = compute_errors(
        coeff=coeff,
        quadrature_order=cfg.error_quadrature_order,
    )

    residual_vec = np.einsum("ij,j->i", a, coeff) - b
    residual_inf = float(np.max(np.abs(residual_vec)))
    cond_num = float(np.linalg.cond(a))

    return CaseResult(
        n_modes=n_modes,
        l2_error=l2_error,
        h1_semi_error=h1_error,
        max_abs_error=max_abs_error,
        condition_number=cond_num,
        residual_inf=residual_inf,
        boundary_left=left,
        boundary_right=right,
    )


def run_suite(cfg: GalerkinConfig) -> List[CaseResult]:
    results: List[CaseResult] = []
    for n_modes in cfg.mode_counts:
        results.append(run_case(n_modes=n_modes, cfg=cfg))
    return results


def validate_results(results: List[CaseResult]) -> None:
    if not results:
        raise AssertionError("no Galerkin results produced")

    l2 = np.array([r.l2_error for r in results], dtype=float)
    if l2.size >= 2:
        # With this smooth problem and nested mode sets, errors should decrease.
        assert np.all(np.diff(l2) < 0.0), "L2 error did not decrease monotonically"

    assert l2[-1] < 1e-5, "final truncation is unexpectedly inaccurate"

    for r in results:
        assert abs(r.boundary_left) < 1e-12, "left boundary condition not satisfied"
        assert abs(r.boundary_right) < 1e-12, "right boundary condition not satisfied"
        assert r.residual_inf < 1e-10, "linear system residual is too large"
        assert np.isfinite(r.condition_number), "condition number is not finite"


def print_report(results: List[CaseResult]) -> None:
    print("Galerkin Method MVP (MATH-0434)")
    print("PDE: -u''(x)=1 on (0,1), u(0)=u(1)=0")
    print("Trial/Test basis: sin(k*pi*x), k=1..N")
    print("-" * 102)
    print(
        "{:<8} {:<14} {:<14} {:<14} {:<14} {:<12}".format(
            "N", "L2_error", "H1_semi", "MaxAbs", "cond(A)", "residual_inf"
        )
    )

    for r in results:
        print(
            "{:<8d} {:<14.6e} {:<14.6e} {:<14.6e} {:<14.6e} {:<12.3e}".format(
                r.n_modes,
                r.l2_error,
                r.h1_semi_error,
                r.max_abs_error,
                r.condition_number,
                r.residual_inf,
            )
        )

    if len(results) >= 2:
        print("\nObserved convergence rates (L2, N doubles):")
        for i in range(1, len(results)):
            r0 = results[i - 1]
            r1 = results[i]
            rate = np.log(r0.l2_error / r1.l2_error) / np.log(2.0)
            print(f"N={r0.n_modes:>2d} -> N={r1.n_modes:>2d}: rate={rate:.3f}")


def main() -> None:
    cfg = GalerkinConfig()
    results = run_suite(cfg)
    print_report(results)
    validate_results(results)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
