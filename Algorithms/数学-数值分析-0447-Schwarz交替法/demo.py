"""Schwarz alternating method MVP on a 1D Poisson problem.

Problem:
    -u''(x) = f(x), x in (0,1)
    u(0) = u(1) = 0

We split the domain into two overlapping subdomains and apply a
multiplicative Schwarz iteration (left subdomain solve, then right
subdomain solve using updated interface data).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SchwarzConfig:
    """Configuration for the demo run."""

    n_internal: int = 127
    overlap_left: float = 0.40
    overlap_right: float = 0.65
    max_iter: int = 30
    tol: float = 1e-8



def solve_poisson_dirichlet_segment(
    f_interior: np.ndarray,
    h: float,
    left_bc: float,
    right_bc: float,
) -> np.ndarray:
    """Solve 1D Poisson on one segment with Dirichlet endpoints.

    Discrete equation on interior points:
        2u_i - u_{i-1} - u_{i+1} = h^2 f_i
    """
    m = f_interior.size
    if m == 0:
        return np.empty(0, dtype=float)

    a = np.zeros((m, m), dtype=float)
    np.fill_diagonal(a, 2.0)
    if m > 1:
        idx = np.arange(m - 1)
        a[idx, idx + 1] = -1.0
        a[idx + 1, idx] = -1.0

    b = (h * h) * f_interior.astype(float, copy=True)
    b[0] += left_bc
    b[-1] += right_bc
    return np.linalg.solve(a, b)



def residual_norm(u: np.ndarray, f: np.ndarray, h: float) -> float:
    """L2 norm of interior residual r = f - A u."""
    au = (2.0 * u[1:-1] - u[:-2] - u[2:]) / (h * h)
    r = f[1:-1] - au
    return float(np.linalg.norm(r, ord=2))



def relative_l2_error(u: np.ndarray, u_true: np.ndarray) -> float:
    """Relative L2 error over all nodes."""
    num = np.linalg.norm(u - u_true, ord=2)
    den = np.linalg.norm(u_true, ord=2)
    return float(num / den)



def run_schwarz(config: SchwarzConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Run multiplicative Schwarz iterations and return history."""
    n_nodes = config.n_internal + 2
    x = np.linspace(0.0, 1.0, n_nodes)
    h = 1.0 / (n_nodes - 1)

    ia = int(round(config.overlap_left / h))
    ib = int(round(config.overlap_right / h))
    ia = max(1, min(ia, n_nodes - 3))
    ib = max(ia + 1, min(ib, n_nodes - 2))

    u_true = np.sin(np.pi * x)
    f = (np.pi ** 2) * np.sin(np.pi * x)

    u = np.zeros_like(x)
    r0 = residual_norm(u, f, h)

    history = []
    converged_iter = config.max_iter

    for k in range(1, config.max_iter + 1):
        # Step 1: solve on left subdomain [0, x_ib]
        right_trace_for_left = u[ib]
        u1_interior = solve_poisson_dirichlet_segment(
            f_interior=f[1:ib],
            h=h,
            left_bc=0.0,
            right_bc=right_trace_for_left,
        )
        u1 = np.zeros(ib + 1, dtype=float)
        u1[1:ib] = u1_interior
        u1[ib] = right_trace_for_left

        # Step 2: solve on right subdomain [x_ia, 1] with updated trace
        left_trace_for_right = u1[ia]
        u2_interior = solve_poisson_dirichlet_segment(
            f_interior=f[ia + 1 : -1],
            h=h,
            left_bc=left_trace_for_right,
            right_bc=0.0,
        )
        u2 = np.zeros(n_nodes - ia, dtype=float)
        u2[0] = left_trace_for_right
        u2[1:-1] = u2_interior

        # Assemble one global iterate (multiplicative update style)
        u_new = np.zeros_like(u)
        u_new[: ia + 1] = u1[: ia + 1]
        u_new[ia:] = u2

        res = residual_norm(u_new, f, h)
        err = relative_l2_error(u_new, u_true)
        interface_gap = abs(u1[ib] - u2[ib - ia])
        history.append((k, res / r0, err, interface_gap))

        u = u_new

        if res / r0 < config.tol:
            converged_iter = k
            break

    return x, u, np.array(history, dtype=float), converged_iter



def main() -> None:
    config = SchwarzConfig()
    x, u, history, converged_iter = run_schwarz(config)

    print("Schwarz交替法 1D Poisson MVP")
    print(
        f"n_internal={config.n_internal}, overlap=[{config.overlap_left:.2f}, {config.overlap_right:.2f}], "
        f"max_iter={config.max_iter}, tol={config.tol:.1e}"
    )
    print("iter | residual_ratio | relative_error | interface_gap")
    print("-----+----------------+----------------+--------------")
    for row in history:
        k, rr, err, gap = row
        print(f"{int(k):4d} | {rr:14.6e} | {err:14.6e} | {gap:12.6e}")

    final_rr = history[-1, 1]
    final_err = history[-1, 2]
    print("\nSummary")
    print(f"iterations_used = {int(history[-1, 0])}")
    print(f"converged_iter  = {converged_iter}")
    print(f"final_residual_ratio = {final_rr:.6e}")
    print(f"final_relative_error = {final_err:.6e}")

    # Small sanity point check near the center
    i_mid = len(x) // 2
    print(
        f"u(0.5) approx = {u[i_mid]:.6f}, "
        f"exact = {np.sin(np.pi * x[i_mid]):.6f}"
    )


if __name__ == "__main__":
    main()
