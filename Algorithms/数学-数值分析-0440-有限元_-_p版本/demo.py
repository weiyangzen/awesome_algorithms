"""有限元 p 版本（p-refinement）最小可运行示例。

问题:
    -u''(x) = pi^2 sin(pi x), x in (0, 1)
    u(0)=u(1)=0
精确解:
    u(x)=sin(pi x)

实现策略:
1) 固定单元数 nelems（h 固定）
2) 提升每个单元的多项式阶数 p
3) 采用等参映射 + Gauss-Legendre 积分进行组装
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PFEMResult:
    p: int
    ndof: int
    l2_error: float


def rhs_f(x: np.ndarray) -> np.ndarray:
    return (math.pi**2) * np.sin(math.pi * x)


def exact_u(x: np.ndarray) -> np.ndarray:
    return np.sin(math.pi * x)


def lagrange_basis_and_derivative(
    nodes: np.ndarray, xi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """在参考区间上计算 Lagrange 形函数及其导数。

    参数:
        nodes: 局部插值点, shape=(nloc,)
        xi:    评估点, shape=(nq,)
    返回:
        phi:      shape=(nq, nloc)
        dphi_dxi: shape=(nq, nloc)
    """
    nloc = nodes.size
    nq = xi.size
    phi = np.ones((nq, nloc), dtype=float)
    dphi = np.zeros((nq, nloc), dtype=float)

    for i in range(nloc):
        for j in range(nloc):
            if j == i:
                continue
            phi[:, i] *= (xi - nodes[j]) / (nodes[i] - nodes[j])

    for i in range(nloc):
        for m in range(nloc):
            if m == i:
                continue
            term = np.ones(nq, dtype=float) / (nodes[i] - nodes[m])
            for j in range(nloc):
                if j == i or j == m:
                    continue
                term *= (xi - nodes[j]) / (nodes[i] - nodes[j])
            dphi[:, i] += term

    return phi, dphi


def element_matrices(xl: float, xr: float, p: int) -> tuple[np.ndarray, np.ndarray]:
    """组装一个单元的局部刚度矩阵和载荷向量。"""
    nloc = p + 1
    ref_nodes = np.linspace(-1.0, 1.0, nloc)
    nq = max(2 * p + 3, 6)
    xi_q, w_q = np.polynomial.legendre.leggauss(nq)

    phi, dphi_dxi = lagrange_basis_and_derivative(ref_nodes, xi_q)

    jac = 0.5 * (xr - xl)
    x_q = 0.5 * (xl + xr) + jac * xi_q
    dphi_dx = dphi_dxi / jac

    weighted = w_q * jac
    ke = dphi_dx.T @ (dphi_dx * weighted[:, None])
    fe = phi.T @ (rhs_f(x_q) * weighted)

    return ke, fe


def build_connectivity(nelems: int, p: int) -> list[list[int]]:
    """构建全局自由度编号（相邻单元共享端点自由度）。"""
    elem_dofs: list[list[int]] = []
    next_dof = 0

    for e in range(nelems):
        local: list[int] = []
        for i in range(p + 1):
            if e > 0 and i == 0:
                gid = elem_dofs[e - 1][-1]
            else:
                gid = next_dof
                next_dof += 1
            local.append(gid)
        elem_dofs.append(local)

    return elem_dofs


def assemble_and_solve(nelems: int, p: int) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """组装全局线性系统并施加齐次 Dirichlet 边界求解。"""
    grid = np.linspace(0.0, 1.0, nelems + 1)
    elem_dofs = build_connectivity(nelems=nelems, p=p)
    ndof = elem_dofs[-1][-1] + 1

    k_global = np.zeros((ndof, ndof), dtype=float)
    f_global = np.zeros(ndof, dtype=float)

    for e in range(nelems):
        ke, fe = element_matrices(grid[e], grid[e + 1], p)
        dofs = elem_dofs[e]
        for a, ia in enumerate(dofs):
            f_global[ia] += fe[a]
            for b, ib in enumerate(dofs):
                k_global[ia, ib] += ke[a, b]

    fixed = np.array([0, ndof - 1], dtype=int)
    free = np.setdiff1d(np.arange(ndof), fixed)

    u = np.zeros(ndof, dtype=float)
    k_ff = k_global[np.ix_(free, free)]
    rhs = f_global[free] - k_global[np.ix_(free, fixed)] @ u[fixed]
    u[free] = np.linalg.solve(k_ff, rhs)

    return u, grid, elem_dofs


def evaluate_l2_error(u: np.ndarray, grid: np.ndarray, elem_dofs: list[list[int]], p: int) -> float:
    """用高阶积分计算离散解与精确解的 L2 误差。"""
    nloc = p + 1
    ref_nodes = np.linspace(-1.0, 1.0, nloc)
    nq = max(2 * p + 6, 10)
    xi_q, w_q = np.polynomial.legendre.leggauss(nq)
    phi, _ = lagrange_basis_and_derivative(ref_nodes, xi_q)

    err_sq = 0.0
    for e, dofs in enumerate(elem_dofs):
        xl, xr = grid[e], grid[e + 1]
        jac = 0.5 * (xr - xl)
        x_q = 0.5 * (xl + xr) + jac * xi_q

        uh_q = phi @ u[np.array(dofs)]
        ue_q = exact_u(x_q)
        diff = uh_q - ue_q
        err_sq += np.sum((diff**2) * w_q * jac)

    return float(np.sqrt(err_sq))


def run_p_refinement_demo(nelems: int = 4, p_values: tuple[int, ...] = (1, 2, 3, 4, 5, 6)) -> list[PFEMResult]:
    results: list[PFEMResult] = []
    for p in p_values:
        u, grid, elem_dofs = assemble_and_solve(nelems=nelems, p=p)
        l2 = evaluate_l2_error(u=u, grid=grid, elem_dofs=elem_dofs, p=p)
        results.append(PFEMResult(p=p, ndof=u.size, l2_error=l2))
    return results


def main() -> None:
    results = run_p_refinement_demo()

    print("p-version FEM demo for -u'' = pi^2 sin(pi x), u(0)=u(1)=0")
    print("Fixed mesh: nelems=4, domain=[0,1]")
    print("-" * 58)
    print(f"{'p':>3} | {'ndof':>5} | {'L2 error':>14} | {'improve vs prev':>14}")
    print("-" * 58)

    prev = None
    for r in results:
        improve = "-"
        if prev is not None and r.l2_error > 0.0:
            improve = f"x{(prev / r.l2_error):.2f}"
        print(f"{r.p:>3d} | {r.ndof:>5d} | {r.l2_error:>14.6e} | {improve:>14}")
        prev = r.l2_error


if __name__ == "__main__":
    main()
