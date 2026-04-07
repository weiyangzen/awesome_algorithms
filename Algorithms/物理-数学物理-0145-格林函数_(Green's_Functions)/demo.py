"""Green's function MVP for a 1D Poisson problem with Dirichlet boundaries.

Problem solved in this demo:
    -u''(x) = f(x),  x in [0, 1]
    u(0) = 0, u(1) = 0
where f(x) is a smooth source term.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp
from sklearn.metrics import mean_squared_error

try:
    import torch
except Exception:  # pragma: no cover - fallback if torch is unavailable
    torch = None


def forcing(x: np.ndarray) -> np.ndarray:
    """Source term f(x) in -u'' = f."""
    return np.sin(np.pi * x) + 0.25 * np.cos(2.0 * np.pi * x)


def green_kernel(x: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Dirichlet Green's kernel for operator L = -d^2/dx^2 on [0, 1]."""
    x_col = np.atleast_1d(x)[:, None]
    xi_row = np.atleast_1d(xi)[None, :]
    return np.where(x_col <= xi_row, x_col * (1.0 - xi_row), xi_row * (1.0 - x_col))


def solve_with_green(x_eval: np.ndarray, n_quad: int = 2000) -> tuple[np.ndarray, float]:
    """Compute u(x) = ∫ G(x, xi) f(xi) dxi with trapezoidal quadrature.

    Returns:
        u_green: solution values on x_eval.
        torch_linf_diff: Linf diff vs optional torch implementation.
    """
    xi = np.linspace(0.0, 1.0, n_quad)
    dxi = 1.0 / (n_quad - 1)
    weights = np.ones_like(xi)
    weights[[0, -1]] = 0.5
    weights *= dxi

    g_mat = green_kernel(x_eval, xi)
    rhs = forcing(xi) * weights
    u_green = g_mat @ rhs

    torch_linf_diff = 0.0
    if torch is not None:
        g_t = torch.from_numpy(g_mat).to(torch.float64)
        rhs_t = torch.from_numpy(rhs).to(torch.float64)
        u_t = g_t @ rhs_t
        torch_linf_diff = float(np.max(np.abs(u_green - u_t.numpy())))
    return u_green, torch_linf_diff


def solve_with_bvp(x_eval: np.ndarray) -> np.ndarray:
    """Reference solution using SciPy's solve_bvp."""

    def ode(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # y[0] = u, y[1] = u'
        return np.vstack((y[1], -forcing(x)))

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        return np.array([ya[0], yb[0]])

    x_init = np.linspace(0.0, 1.0, 200)
    y_init = np.zeros((2, x_init.size))
    sol = solve_bvp(ode, bc, x_init, y_init, tol=1e-7, max_nodes=20000)
    if sol.status != 0:
        raise RuntimeError(f"solve_bvp failed with status={sol.status}: {sol.message}")
    return sol.sol(x_eval)[0]


def pde_residual_linf(x: np.ndarray, u: np.ndarray) -> float:
    """Estimate ||-u'' - f||_inf by second-order finite differences."""
    dx = x[1] - x[0]
    u_xx = (u[:-2] - 2.0 * u[1:-1] + u[2:]) / (dx * dx)
    residual = -u_xx - forcing(x[1:-1])
    return float(np.max(np.abs(residual)))


def main() -> None:
    x = np.linspace(0.0, 1.0, 401)

    u_green, torch_linf_diff = solve_with_green(x, n_quad=2500)
    u_bvp = solve_with_bvp(x)

    linf_error = float(np.max(np.abs(u_green - u_bvp)))
    rmse = float(np.sqrt(mean_squared_error(u_bvp, u_green)))
    residual_linf = pde_residual_linf(x, u_green)

    sample_idx = np.linspace(0, x.size - 1, 9, dtype=int)
    table = pd.DataFrame(
        {
            "x": x[sample_idx],
            "u_green": u_green[sample_idx],
            "u_bvp": u_bvp[sample_idx],
            "abs_err": np.abs(u_green[sample_idx] - u_bvp[sample_idx]),
        }
    )

    print("=== Green's Function MVP: 1D Poisson on [0,1] ===")
    print(f"grid_points={x.size}, quadrature_points=2500")
    print(f"L_inf(u_green - u_bvp) = {linf_error:.3e}")
    print(f"RMSE(u_green, u_bvp)   = {rmse:.3e}")
    print(f"L_inf(-u'' - f)        = {residual_linf:.3e}")
    print(f"L_inf(numpy - torch)   = {torch_linf_diff:.3e}")
    print("\nSample points:")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
