"""Minimal runnable MVP for superconvergent recovery (MATH-0444).

Scenario:
- Solve 1D Poisson equation with linear finite elements.
- Compute raw element-wise gradients (piecewise constant).
- Recover nodal gradients via local polynomial least-squares fitting
  on superconvergent sampling points (element midpoints).
- Compare L2 gradient errors and observed convergence rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class RecoveryConfig:
    """Configuration for the non-interactive superconvergent recovery demo."""

    element_counts: tuple[int, ...] = (8, 16, 32, 64, 128)
    assembly_quad_order: int = 5
    error_quad_order: int = 6
    patch_radius: int = 2
    recovery_poly_degree: int = 2


@dataclass(frozen=True)
class CaseResult:
    """Diagnostics for one mesh resolution."""

    n_elements: int
    h: float
    l2_raw_grad_error: float
    l2_recovered_grad_error: float
    max_nodal_recovered_error: float
    condition_number: float
    residual_inf: float


def forcing_term(x: Array) -> Array:
    """RHS f(x) for -u'' = f with exact u(x)=sin(pi x)."""
    return (np.pi**2) * np.sin(np.pi * x)


def exact_solution(x: Array) -> Array:
    """Exact solution u(x) = sin(pi x)."""
    return np.sin(np.pi * x)


def exact_gradient(x: Array) -> Array:
    """Exact gradient u'(x) = pi cos(pi x)."""
    return np.pi * np.cos(np.pi * x)


def gauss_legendre_on_unit(order: int) -> Tuple[Array, Array]:
    """Return Gauss-Legendre points/weights on [0, 1]."""
    if order < 2:
        raise ValueError("quadrature order must be >= 2")

    xi, w = np.polynomial.legendre.leggauss(order)
    x = 0.5 * (xi + 1.0)
    w01 = 0.5 * w
    return x.astype(float), w01.astype(float)


def assemble_poisson_system(n_elements: int, quad_order: int) -> Tuple[Array, Array, Array]:
    """Assemble linear FE system for 1D Poisson equation on [0,1].

    Model:
        -u'' = f,  u(0)=u(1)=0
    with linear (P1) finite elements on a uniform mesh.

    Returns:
        x_nodes: mesh nodes, shape (n_elements+1,)
        a: interior stiffness matrix, shape (n_elements-1, n_elements-1)
        b: interior load vector, shape (n_elements-1,)
    """
    if n_elements < 2:
        raise ValueError("n_elements must be >= 2")

    x_nodes = np.linspace(0.0, 1.0, n_elements + 1, dtype=float)
    interior_dim = n_elements - 1

    a = np.zeros((interior_dim, interior_dim), dtype=float)
    b = np.zeros(interior_dim, dtype=float)

    ref_x, ref_w = gauss_legendre_on_unit(quad_order)

    for e in range(n_elements):
        xl = x_nodes[e]
        xr = x_nodes[e + 1]
        h = xr - xl

        k_local = (1.0 / h) * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)

        xq = xl + h * ref_x
        wq = h * ref_w

        phi_left = (xr - xq) / h
        phi_right = (xq - xl) / h
        f_q = forcing_term(xq)

        f_local = np.array(
            [
                np.sum(wq * f_q * phi_left),
                np.sum(wq * f_q * phi_right),
            ],
            dtype=float,
        )

        element_nodes = (e, e + 1)
        for i_local in range(2):
            gi = element_nodes[i_local] - 1
            if 0 <= gi < interior_dim:
                b[gi] += f_local[i_local]

            for j_local in range(2):
                gj = element_nodes[j_local] - 1
                if 0 <= gi < interior_dim and 0 <= gj < interior_dim:
                    a[gi, gj] += k_local[i_local, j_local]

    return x_nodes, a, b


def solve_fem_solution(n_elements: int, quad_order: int) -> Tuple[Array, Array, float, float]:
    """Solve FE linear system and return full nodal solution and diagnostics."""
    x_nodes, a, b = assemble_poisson_system(n_elements=n_elements, quad_order=quad_order)
    interior_u = np.linalg.solve(a, b)

    u = np.zeros(n_elements + 1, dtype=float)
    u[1:-1] = interior_u

    residual_vec = np.einsum("ij,j->i", a, interior_u) - b
    residual_inf = float(np.max(np.abs(residual_vec)))
    cond_number = float(np.linalg.cond(a))
    return x_nodes, u, residual_inf, cond_number


def element_gradients(x_nodes: Array, u: Array) -> Array:
    """Piecewise-constant FE gradient on each element."""
    h = np.diff(x_nodes)
    return (u[1:] - u[:-1]) / h


def _select_patch_indices(node_idx: int, n_elements: int, radius: int, min_count: int) -> Array:
    """Pick nearby element indices around a node for local least-squares fitting."""
    left = max(0, node_idx - radius)
    right = min(n_elements - 1, node_idx + radius - 1)

    while (right - left + 1) < min_count and (left > 0 or right < n_elements - 1):
        if left > 0:
            left -= 1
        if (right - left + 1) >= min_count:
            break
        if right < n_elements - 1:
            right += 1

    return np.arange(left, right + 1, dtype=int)


def recover_nodal_gradients(
    x_nodes: Array,
    raw_element_grads: Array,
    patch_radius: int,
    poly_degree: int,
) -> Array:
    """Recover nodal gradients from element midpoint superconvergent samples.

    We fit a local polynomial (least squares) on each node patch:
        g_h(midpoint_j) ~ p_i(midpoint_j)
    then evaluate p_i at node x_i to obtain recovered gradient G_i.
    """
    n_elements = raw_element_grads.size
    if x_nodes.size != n_elements + 1:
        raise ValueError("x_nodes and raw_element_grads size mismatch")
    if patch_radius < 1:
        raise ValueError("patch_radius must be >= 1")
    if poly_degree < 1:
        raise ValueError("poly_degree must be >= 1")

    x_mid = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    recovered = np.zeros(n_elements + 1, dtype=float)
    min_points = poly_degree + 1

    for i in range(n_elements + 1):
        idx = _select_patch_indices(i, n_elements, patch_radius, min_points)
        xs = x_mid[idx] - x_nodes[i]
        ys = raw_element_grads[idx]

        local_deg = min(poly_degree, xs.size - 1)
        vander = np.vander(xs, N=local_deg + 1, increasing=True)

        coeff, _, _, _ = np.linalg.lstsq(vander, ys, rcond=None)
        recovered[i] = coeff[0]

    return recovered


def l2_error_raw_gradient(x_nodes: Array, raw_element_grads: Array, quad_order: int) -> float:
    """L2 error of piecewise-constant raw gradient against exact gradient."""
    ref_x, ref_w = gauss_legendre_on_unit(quad_order)

    err2 = 0.0
    for e in range(raw_element_grads.size):
        xl = x_nodes[e]
        xr = x_nodes[e + 1]
        h = xr - xl

        xq = xl + h * ref_x
        wq = h * ref_w

        diff = raw_element_grads[e] - exact_gradient(xq)
        err2 += float(np.sum(wq * diff * diff))

    return float(np.sqrt(err2))


def l2_error_recovered_gradient(x_nodes: Array, recovered_nodal_grads: Array, quad_order: int) -> float:
    """L2 error of recovered gradient (piecewise linear interpolation)."""
    ref_x, ref_w = gauss_legendre_on_unit(quad_order)

    err2 = 0.0
    for e in range(x_nodes.size - 1):
        xl = x_nodes[e]
        xr = x_nodes[e + 1]
        h = xr - xl

        xq = xl + h * ref_x
        wq = h * ref_w
        t = (xq - xl) / h

        g_recovered = (1.0 - t) * recovered_nodal_grads[e] + t * recovered_nodal_grads[e + 1]
        diff = g_recovered - exact_gradient(xq)
        err2 += float(np.sum(wq * diff * diff))

    return float(np.sqrt(err2))


def run_case(n_elements: int, cfg: RecoveryConfig) -> CaseResult:
    """Execute one mesh resolution and return diagnostics."""
    x_nodes, u, residual_inf, cond_number = solve_fem_solution(
        n_elements=n_elements,
        quad_order=cfg.assembly_quad_order,
    )

    raw_grads = element_gradients(x_nodes, u)
    recovered_grads = recover_nodal_gradients(
        x_nodes=x_nodes,
        raw_element_grads=raw_grads,
        patch_radius=cfg.patch_radius,
        poly_degree=cfg.recovery_poly_degree,
    )

    l2_raw = l2_error_raw_gradient(
        x_nodes=x_nodes,
        raw_element_grads=raw_grads,
        quad_order=cfg.error_quad_order,
    )
    l2_rec = l2_error_recovered_gradient(
        x_nodes=x_nodes,
        recovered_nodal_grads=recovered_grads,
        quad_order=cfg.error_quad_order,
    )

    max_nodal_rec = float(np.max(np.abs(recovered_grads - exact_gradient(x_nodes))))

    return CaseResult(
        n_elements=n_elements,
        h=1.0 / n_elements,
        l2_raw_grad_error=l2_raw,
        l2_recovered_grad_error=l2_rec,
        max_nodal_recovered_error=max_nodal_rec,
        condition_number=cond_number,
        residual_inf=residual_inf,
    )


def run_suite(cfg: RecoveryConfig) -> List[CaseResult]:
    """Run all configured mesh levels."""
    return [run_case(n, cfg) for n in cfg.element_counts]


def observed_rates(errors: Array) -> Array:
    """Compute convergence rates when mesh is uniformly halved."""
    return np.log(errors[:-1] / errors[1:]) / np.log(2.0)


def print_report(results: List[CaseResult]) -> None:
    """Print non-interactive diagnostics."""
    print("Superconvergent Recovery MVP (MATH-0444)")
    print("Model: 1D Poisson FEM + midpoint-gradient polynomial recovery")
    print("Exact: u(x)=sin(pi*x), u'(x)=pi*cos(pi*x)")
    print("-" * 120)
    print(
        "{:<8} {:<10} {:<16} {:<16} {:<10} {:<14} {:<12}".format(
            "N", "h", "L2_raw_grad", "L2_recovered", "raw/rec", "cond(A)", "residual"
        )
    )

    for r in results:
        ratio = r.l2_raw_grad_error / r.l2_recovered_grad_error
        print(
            "{:<8d} {:<10.3e} {:<16.6e} {:<16.6e} {:<10.3f} {:<14.6e} {:<12.3e}".format(
                r.n_elements,
                r.h,
                r.l2_raw_grad_error,
                r.l2_recovered_grad_error,
                ratio,
                r.condition_number,
                r.residual_inf,
            )
        )

    if len(results) >= 2:
        raw_errors = np.array([r.l2_raw_grad_error for r in results], dtype=float)
        rec_errors = np.array([r.l2_recovered_grad_error for r in results], dtype=float)
        raw_rates = observed_rates(raw_errors)
        rec_rates = observed_rates(rec_errors)

        print("\nObserved rates (mesh halving):")
        for i in range(1, len(results)):
            print(
                f"N={results[i-1].n_elements:>3d} -> N={results[i].n_elements:>3d}: "
                f"raw_rate={raw_rates[i-1]:.3f}, recovered_rate={rec_rates[i-1]:.3f}"
            )


def validate_results(results: List[CaseResult]) -> None:
    """Basic numerical sanity checks for this MVP."""
    if not results:
        raise AssertionError("no results produced")

    raw_errors = np.array([r.l2_raw_grad_error for r in results], dtype=float)
    rec_errors = np.array([r.l2_recovered_grad_error for r in results], dtype=float)

    assert np.all(np.isfinite(raw_errors)), "raw gradient errors contain non-finite values"
    assert np.all(np.isfinite(rec_errors)), "recovered gradient errors contain non-finite values"

    # Both errors should decrease as mesh is refined.
    assert np.all(np.diff(raw_errors) < 0.0), "raw gradient error is not monotonically decreasing"
    assert np.all(np.diff(rec_errors) < 0.0), "recovered gradient error is not monotonically decreasing"

    # Recovery should improve accuracy at each level.
    assert np.all(rec_errors < raw_errors), "recovered gradient is not better than raw gradient"

    # Convergence-rate expectations: raw ~ O(h), recovered ~ O(h^2).
    raw_rates = observed_rates(raw_errors)
    rec_rates = observed_rates(rec_errors)
    assert float(np.min(raw_rates)) > 0.85, "raw gradient convergence is unexpectedly slow"
    assert float(np.min(rec_rates)) > 1.75, "recovered gradient does not show superconvergence"

    # Final-level quality checks.
    assert rec_errors[-1] < 3.0e-4, "final recovered gradient error is too large"

    for r in results:
        assert r.residual_inf < 1e-10, "linear system residual is too large"
        assert np.isfinite(r.condition_number), "condition number is non-finite"
        assert r.max_nodal_recovered_error < 0.03, "recovered nodal gradient is unexpectedly inaccurate"


def main() -> None:
    cfg = RecoveryConfig()
    results = run_suite(cfg)
    print_report(results)
    validate_results(results)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
