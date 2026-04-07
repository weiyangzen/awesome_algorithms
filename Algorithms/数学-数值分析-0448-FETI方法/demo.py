"""FETI method MVP for a 1D Poisson problem on two non-overlapping subdomains.

Model problem:
    -u''(x) = 1 + 0.2*sin(2*pi*x),  x in (0, 1)
    u(0) = u(1) = 0

We split the domain into:
    Omega_1 = [0, 0.5], Omega_2 = [0.5, 1]

Each subdomain has its own copy of the interface displacement. A single
Lagrange multiplier enforces continuity at x = 0.5 (classic dual FETI view).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the minimal FETI experiment."""

    half_nodes: int = 40
    cg_tol: float = 1.0e-14
    cg_max_iter: int = 20

    def validate(self) -> None:
        if self.half_nodes < 4:
            raise ValueError("half_nodes must be >= 4")
        if self.cg_tol <= 0.0:
            raise ValueError("cg_tol must be positive")
        if self.cg_max_iter <= 0:
            raise ValueError("cg_max_iter must be positive")


def build_subdomain_system(half_nodes: int, h: float, is_left: bool) -> tuple[Array, Array]:
    """Build one subdomain matrix/vector with split interface contributions.

    The global tridiagonal Laplacian has interface diagonal contribution "2".
    In a non-overlapping split, each side contributes "1" at the interface.
    For unit forcing, the interface load is also split into halves.
    """

    k_local = np.diag(np.full(half_nodes, 2.0))
    k_local += np.diag(np.full(half_nodes - 1, -1.0), 1)
    k_local += np.diag(np.full(half_nodes - 1, -1.0), -1)
    if is_left:
        local_global_ids = np.arange(1, half_nodes + 1)
    else:
        local_global_ids = np.arange(half_nodes, 2 * half_nodes)

    x_local = local_global_ids * h
    forcing = 1.0 + 0.2 * np.sin(2.0 * np.pi * x_local)
    f_local = h * h * forcing

    if is_left:
        k_local[-1, -1] = 1.0
        f_local[-1] *= 0.5
    else:
        k_local[0, 0] = 1.0
        f_local[0] *= 0.5

    return k_local.astype(np.float64), f_local.astype(np.float64)


def build_interface_operators(half_nodes: int) -> tuple[Array, Array]:
    """Build continuity operators B1*u1 + B2*u2 = 0."""

    b1 = np.zeros((1, half_nodes), dtype=np.float64)
    b2 = np.zeros((1, half_nodes), dtype=np.float64)
    b1[0, -1] = 1.0
    b2[0, 0] = -1.0
    return b1, b2


def conjugate_gradient(
    matvec,
    b: Array,
    tol: float,
    max_iter: int,
) -> tuple[Array, int, list[float]]:
    """Simple CG implementation for SPD systems."""

    x = np.zeros_like(b)
    r = b - matvec(x)
    p = r.copy()
    rr = float(r @ r)
    residual_history = [rr**0.5]

    if residual_history[-1] <= tol:
        return x, 0, residual_history

    for it in range(1, max_iter + 1):
        ap = matvec(p)
        denom = float(p @ ap)
        if abs(denom) < 1.0e-30:
            raise RuntimeError("CG breakdown: near-zero denominator")

        alpha = rr / denom
        x = x + alpha * p
        r = r - alpha * ap
        rr_new = float(r @ r)
        residual_history.append(rr_new**0.5)

        if residual_history[-1] <= tol:
            return x, it, residual_history

        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    raise RuntimeError("CG did not converge within max_iter")


def solve_feti_poisson_1d(config: ExperimentConfig) -> dict[str, float | int]:
    """Solve the model problem with a two-subdomain dual FETI formulation."""

    n = config.half_nodes
    n_total = 2 * n
    h = 1.0 / n_total

    k1, f1 = build_subdomain_system(n, h, is_left=True)
    k2, f2 = build_subdomain_system(n, h, is_left=False)
    b1, b2 = build_interface_operators(n)

    k1_inv_b1_t = np.linalg.solve(k1, b1.T)
    k2_inv_b2_t = np.linalg.solve(k2, b2.T)
    k1_inv_f1 = np.linalg.solve(k1, f1)
    k2_inv_f2 = np.linalg.solve(k2, f2)

    # Dual interface system: F * lambda = d
    f_dual = b1 @ k1_inv_b1_t + b2 @ k2_inv_b2_t
    d_dual = (b1 @ k1_inv_f1 + b2 @ k2_inv_f2).reshape(1)
    cond_f = float(np.linalg.cond(f_dual))

    lam, cg_iters, residuals = conjugate_gradient(
        matvec=lambda v: f_dual @ v,
        b=d_dual,
        tol=config.cg_tol,
        max_iter=config.cg_max_iter,
    )

    u1 = np.linalg.solve(k1, f1 - b1.T @ lam)
    u2 = np.linalg.solve(k2, f2 - b2.T @ lam)
    interface_jump = float((b1 @ u1 + b2 @ u2)[0])

    # Stitch subdomain solutions into one global vector.
    u_feti = np.zeros(n_total + 1, dtype=np.float64)
    u_feti[1 : n + 1] = u1
    u_feti[n + 1 : n_total] = u2[1:]

    # Monolithic solve for verification.
    a_global = np.diag(np.full(n_total - 1, 2.0))
    a_global += np.diag(np.full(n_total - 2, -1.0), 1)
    a_global += np.diag(np.full(n_total - 2, -1.0), -1)
    x_inner = np.arange(1, n_total) * h
    rhs_global = h * h * (1.0 + 0.2 * np.sin(2.0 * np.pi * x_inner))
    u_global = np.zeros(n_total + 1, dtype=np.float64)
    u_global[1:n_total] = np.linalg.solve(a_global, rhs_global)

    x = np.linspace(0.0, 1.0, n_total + 1)
    u_exact = 0.5 * x * (1.0 - x) + (0.05 / (np.pi**2)) * np.sin(2.0 * np.pi * x)

    feti_vs_global_inf = float(np.max(np.abs(u_feti - u_global)))
    inf_err_exact = float(np.max(np.abs(u_feti - u_exact)))
    l2_err_exact = float(np.sqrt(h * np.sum((u_feti - u_exact) ** 2)))

    return {
        "half_nodes": n,
        "global_intervals": n_total,
        "h": h,
        "lambda_value": float(lam[0]),
        "cg_iterations": int(cg_iters),
        "cg_final_residual": float(residuals[-1]),
        "interface_jump": interface_jump,
        "dual_condition_number": cond_f,
        "feti_vs_monolithic_inf": feti_vs_global_inf,
        "inf_err_to_exact": inf_err_exact,
        "l2_err_to_exact": l2_err_exact,
    }


def run_checks(stats: dict[str, float | int]) -> None:
    """Deterministic correctness checks for this MVP."""

    assert abs(float(stats["interface_jump"])) < 1.0e-12, "continuity not enforced"
    assert float(stats["feti_vs_monolithic_inf"]) < 1.0e-12, "FETI mismatch vs monolithic"
    assert float(stats["inf_err_to_exact"]) < 2.0e-4, "discretization error too large"
    assert int(stats["cg_iterations"]) <= 2, "unexpected CG iteration count"
    assert float(stats["cg_final_residual"]) < 1.0e-12, "dual residual too large"


def print_report(stats: dict[str, float | int]) -> None:
    """Pretty-print experiment results."""

    print("FETI MVP: 1D Poisson on [0,1] split into 2 subdomains")
    print("-" * 64)
    print(f"half_nodes per subdomain : {stats['half_nodes']}")
    print(f"global intervals         : {stats['global_intervals']}")
    print(f"grid size h              : {stats['h']:.6e}")
    print(f"lambda                   : {stats['lambda_value']:.12e}")
    print(f"CG iterations            : {stats['cg_iterations']}")
    print(f"CG final residual        : {stats['cg_final_residual']:.3e}")
    print(f"interface jump           : {stats['interface_jump']:.3e}")
    print(f"cond(F_dual)             : {stats['dual_condition_number']:.3e}")
    print(f"||u_feti-u_global||_inf  : {stats['feti_vs_monolithic_inf']:.3e}")
    print(f"||u_feti-u_exact||_inf   : {stats['inf_err_to_exact']:.3e}")
    print(f"||u_feti-u_exact||_L2    : {stats['l2_err_to_exact']:.3e}")


def main() -> None:
    config = ExperimentConfig()
    config.validate()
    stats = solve_feti_poisson_1d(config)
    run_checks(stats)
    print_report(stats)
    print("All checks passed.")


if __name__ == "__main__":
    main()
