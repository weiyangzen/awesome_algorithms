"""Minimal mixed finite element MVP for Poisson equation on a unit square.

Model problem (first-order mixed form):
    q + grad(u) = 0
    div(q) = f
with Dirichlet boundary condition u = g on dOmega.

The discrete scheme uses a structured-grid RT0/P0-style layout:
- Flux q is represented by normal components on edges.
- Scalar u is represented by piecewise constants at cell centers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


@dataclass
class MixedPoissonResult:
    nx: int
    ny: int
    hx: float
    hy: float
    q: np.ndarray
    u: np.ndarray
    residual_inf: float


@dataclass
class ErrorMetrics:
    u_l2: float
    u_rel_l2: float
    flux_l2: float
    flux_rel_l2: float


def cell_id(i: int, j: int, nx: int) -> int:
    return j * nx + i


def vertical_edge_id(i: int, j: int, nx: int, ny: int) -> int:
    _ = ny
    return j * (nx + 1) + i


def horizontal_edge_id(i: int, j: int, nx: int, ny: int) -> int:
    n_vertical = (nx + 1) * ny
    return n_vertical + j * nx + i


def exact_u(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def exact_qx(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    return -np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)


def exact_qy(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    return -np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)


def source_f(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    # Because -Delta u = 2*pi^2*sin(pi x)sin(pi y)
    return 2.0 * (np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y)


def boundary_u(x: float, y: float) -> float:
    return float(exact_u(x, y))


def assemble_triplets(
    nx: int, ny: int
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hx = 1.0 / nx
    hy = 1.0 / ny

    n_vertical = (nx + 1) * ny
    n_horizontal = nx * (ny + 1)
    n_edges = n_vertical + n_horizontal
    n_cells = nx * ny
    n_total = n_edges + n_cells

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.zeros(n_total, dtype=float)

    # Constitutive equations on vertical edges: qx + du/dx = 0.
    for j in range(ny):
        y_mid = (j + 0.5) * hy
        for i in range(nx + 1):
            e = vertical_edge_id(i, j, nx, ny)
            rows.append(e)
            cols.append(e)
            vals.append(1.0)

            if i == 0:
                c_right = cell_id(0, j, nx)
                rows.append(e)
                cols.append(n_edges + c_right)
                vals.append(2.0 / hx)
                rhs[e] = (2.0 / hx) * boundary_u(0.0, y_mid)
            elif i == nx:
                c_left = cell_id(nx - 1, j, nx)
                rows.append(e)
                cols.append(n_edges + c_left)
                vals.append(-2.0 / hx)
                rhs[e] = (-2.0 / hx) * boundary_u(1.0, y_mid)
            else:
                c_left = cell_id(i - 1, j, nx)
                c_right = cell_id(i, j, nx)
                rows.append(e)
                cols.append(n_edges + c_left)
                vals.append(-1.0 / hx)
                rows.append(e)
                cols.append(n_edges + c_right)
                vals.append(1.0 / hx)

    # Constitutive equations on horizontal edges: qy + du/dy = 0.
    for j in range(ny + 1):
        for i in range(nx):
            e = horizontal_edge_id(i, j, nx, ny)
            x_mid = (i + 0.5) * hx
            rows.append(e)
            cols.append(e)
            vals.append(1.0)

            if j == 0:
                c_top = cell_id(i, 0, nx)
                rows.append(e)
                cols.append(n_edges + c_top)
                vals.append(2.0 / hy)
                rhs[e] = (2.0 / hy) * boundary_u(x_mid, 0.0)
            elif j == ny:
                c_bottom = cell_id(i, ny - 1, nx)
                rows.append(e)
                cols.append(n_edges + c_bottom)
                vals.append(-2.0 / hy)
                rhs[e] = (-2.0 / hy) * boundary_u(x_mid, 1.0)
            else:
                c_bottom = cell_id(i, j - 1, nx)
                c_top = cell_id(i, j, nx)
                rows.append(e)
                cols.append(n_edges + c_bottom)
                vals.append(-1.0 / hy)
                rows.append(e)
                cols.append(n_edges + c_top)
                vals.append(1.0 / hy)

    # Conservation equations in each cell: div(q) = f.
    for j in range(ny):
        y_mid = (j + 0.5) * hy
        for i in range(nx):
            x_mid = (i + 0.5) * hx
            c = cell_id(i, j, nx)
            row = n_edges + c

            e_left = vertical_edge_id(i, j, nx, ny)
            e_right = vertical_edge_id(i + 1, j, nx, ny)
            e_bottom = horizontal_edge_id(i, j, nx, ny)
            e_top = horizontal_edge_id(i, j + 1, nx, ny)

            rows.append(row)
            cols.append(e_right)
            vals.append(1.0 / hx)

            rows.append(row)
            cols.append(e_left)
            vals.append(-1.0 / hx)

            rows.append(row)
            cols.append(e_top)
            vals.append(1.0 / hy)

            rows.append(row)
            cols.append(e_bottom)
            vals.append(-1.0 / hy)

            rhs[row] = float(source_f(x_mid, y_mid))

    return (
        n_total,
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int),
        np.asarray(vals, dtype=float),
        rhs,
    )


def solve_linear_system(
    n_total: int, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, rhs: np.ndarray
) -> tuple[np.ndarray, float, str]:
    if SCIPY_AVAILABLE:
        mat = coo_matrix((vals, (rows, cols)), shape=(n_total, n_total)).tocsr()
        sol = np.asarray(spsolve(mat, rhs), dtype=float)
        residual = mat @ sol - rhs
        residual_inf = float(np.linalg.norm(residual, ord=np.inf))
        return sol, residual_inf, "scipy.sparse.linalg.spsolve"

    mat = np.zeros((n_total, n_total), dtype=float)
    for r, c, v in zip(rows, cols, vals):
        mat[r, c] += v

    sol = np.linalg.solve(mat, rhs)
    mat_vec = np.einsum("ij,j->i", mat, sol, optimize=True)
    residual = mat_vec - rhs
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))
    return sol, residual_inf, "numpy.linalg.solve (dense fallback)"


def solve_mixed_poisson(nx: int, ny: int) -> tuple[MixedPoissonResult, str]:
    n_total, rows, cols, vals, rhs = assemble_triplets(nx, ny)
    sol, residual_inf, solver_label = solve_linear_system(n_total, rows, cols, vals, rhs)

    n_vertical = (nx + 1) * ny
    n_horizontal = nx * (ny + 1)
    n_edges = n_vertical + n_horizontal

    q = np.asarray(sol[:n_edges], dtype=float)
    u = np.asarray(sol[n_edges:], dtype=float)

    result = MixedPoissonResult(
        nx=nx,
        ny=ny,
        hx=1.0 / nx,
        hy=1.0 / ny,
        q=q,
        u=u,
        residual_inf=residual_inf,
    )
    return result, solver_label


def reshape_unknowns(result: MixedPoissonResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny = result.nx, result.ny
    n_vertical = (nx + 1) * ny

    qx = result.q[:n_vertical].reshape(ny, nx + 1)
    qy = result.q[n_vertical:].reshape(ny + 1, nx)
    u_cell = result.u.reshape(ny, nx)
    return qx, qy, u_cell


def evaluate_errors(result: MixedPoissonResult) -> ErrorMetrics:
    nx, ny = result.nx, result.ny
    hx, hy = result.hx, result.hy

    qx_num, qy_num, u_num = reshape_unknowns(result)

    x_centers = (np.arange(nx) + 0.5) * hx
    y_centers = (np.arange(ny) + 0.5) * hy
    xc, yc = np.meshgrid(x_centers, y_centers)

    u_true = exact_u(xc, yc)

    x_v = np.arange(nx + 1) * hx
    y_v = (np.arange(ny) + 0.5) * hy
    xv, yv = np.meshgrid(x_v, y_v)
    qx_true = exact_qx(xv, yv)

    x_h = (np.arange(nx) + 0.5) * hx
    y_h = np.arange(ny + 1) * hy
    xh, yh = np.meshgrid(x_h, y_h)
    qy_true = exact_qy(xh, yh)

    u_err = u_num - u_true
    u_l2 = float(np.sqrt(np.sum((u_err**2) * hx * hy)))
    u_true_l2 = float(np.sqrt(np.sum((u_true**2) * hx * hy)))
    u_rel_l2 = u_l2 / u_true_l2

    # Control-volume-inspired edge weights for flux norms.
    w_v = np.full((ny, nx + 1), hx)
    w_v[:, 0] *= 0.5
    w_v[:, -1] *= 0.5
    w_v *= hy

    w_h = np.full((ny + 1, nx), hy)
    w_h[0, :] *= 0.5
    w_h[-1, :] *= 0.5
    w_h *= hx

    qx_err = qx_num - qx_true
    qy_err = qy_num - qy_true

    flux_l2 = float(np.sqrt(np.sum((qx_err**2) * w_v) + np.sum((qy_err**2) * w_h)))
    flux_true_l2 = float(np.sqrt(np.sum((qx_true**2) * w_v) + np.sum((qy_true**2) * w_h)))
    flux_rel_l2 = flux_l2 / flux_true_l2

    return ErrorMetrics(
        u_l2=u_l2,
        u_rel_l2=u_rel_l2,
        flux_l2=flux_l2,
        flux_rel_l2=flux_rel_l2,
    )


def convergence_order(err_coarse: float, err_fine: float, h_coarse: float, h_fine: float) -> float:
    return math.log(err_coarse / err_fine) / math.log(h_coarse / h_fine)


def main() -> None:
    grid_levels = [8, 12, 16]

    print("Mixed FEM MVP (RT0/P0-style on structured grid)")
    print("Model: q + grad(u) = 0, div(q) = f, u|boundary = 0")
    print("Exact: u = sin(pi x) sin(pi y)")
    print()

    results: list[tuple[MixedPoissonResult, ErrorMetrics]] = []
    solver_label = ""
    for n in grid_levels:
        result, solver_label = solve_mixed_poisson(n, n)
        metrics = evaluate_errors(result)
        results.append((result, metrics))

    print(f"Linear solver backend: {solver_label}")
    print("Grid convergence summary:")
    print(
        "{:<8} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
            "n", "u_L2", "u_relL2", "flux_L2", "flux_relL2", "res_inf"
        )
    )
    for result, metrics in results:
        print(
            "{:<8d} {:<12.4e} {:<12.4e} {:<12.4e} {:<12.4e} {:<12.4e}".format(
                result.nx,
                metrics.u_l2,
                metrics.u_rel_l2,
                metrics.flux_l2,
                metrics.flux_rel_l2,
                result.residual_inf,
            )
        )

    print()
    print("Estimated orders (based on absolute L2 errors):")
    for idx in range(len(results) - 1):
        r0, m0 = results[idx]
        r1, m1 = results[idx + 1]
        p_u = convergence_order(m0.u_l2, m1.u_l2, r0.hx, r1.hx)
        p_q = convergence_order(m0.flux_l2, m1.flux_l2, r0.hx, r1.hx)
        print(
            f"n={r0.nx:>2d} -> n={r1.nx:>2d}: "
            f"order_u={p_u:.3f}, order_flux={p_q:.3f}"
        )

    print()
    finest_result, finest_metrics = results[-1]
    qx_finest, qy_finest, u_finest = reshape_unknowns(finest_result)
    center_value = u_finest[finest_result.ny // 2, finest_result.nx // 2]
    print("Sanity check on finest grid:")
    print(f"u(center cell)      = {center_value:.6f}")
    print(f"u_relL2(finest)     = {finest_metrics.u_rel_l2:.6e}")
    print(f"flux_relL2(finest)  = {finest_metrics.flux_rel_l2:.6e}")
    print(f"|q_x|max            = {np.max(np.abs(qx_finest)):.6f}")
    print(f"|q_y|max            = {np.max(np.abs(qy_finest)):.6f}")


if __name__ == "__main__":
    main()
