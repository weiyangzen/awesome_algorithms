"""Minimal runnable MVP for Adaptive Mesh Refinement (PHYS-0361)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AMRConfig:
    initial_cells: int = 10
    max_cells: int = 280
    max_iters: int = 10
    dorfler_theta: float = 0.60
    indicator_tol: float = 1.0e-4


def exact_solution(x: np.ndarray) -> np.ndarray:
    """Manufactured exact solution with a localized feature and Dirichlet boundaries."""
    x = np.asarray(x, dtype=float)
    a = 120.0
    c = 0.70
    gaussian = np.exp(-a * (x - c) ** 2)
    g = np.sin(3.0 * np.pi * x) + 0.5 * gaussian
    return x * (1.0 - x) * g


def forcing_term(x: np.ndarray) -> np.ndarray:
    """Right-hand side f(x) for -u'' = f, matching exact_solution."""
    x = np.asarray(x, dtype=float)
    a = 120.0
    c = 0.70

    gaussian = np.exp(-a * (x - c) ** 2)
    g = np.sin(3.0 * np.pi * x) + 0.5 * gaussian

    gp = 3.0 * np.pi * np.cos(3.0 * np.pi * x) - a * (x - c) * gaussian
    gpp = -(3.0 * np.pi) ** 2 * np.sin(3.0 * np.pi * x) + 0.5 * gaussian * (
        4.0 * (a**2) * (x - c) ** 2 - 2.0 * a
    )

    u_second = -2.0 * g + 2.0 * (1.0 - 2.0 * x) * gp + x * (1.0 - x) * gpp
    return -u_second


def solve_poisson_fem_1d(nodes: np.ndarray) -> np.ndarray:
    """Solve -u''=f on [0,1] with u(0)=u(1)=0 via linear FEM on nonuniform mesh."""
    x = np.asarray(nodes, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("nodes must be a 1D array with at least 3 nodes")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("nodes must be strictly increasing")

    n_nodes = x.size
    n_int = n_nodes - 2
    a_mat = np.zeros((n_int, n_int), dtype=float)
    rhs = np.zeros(n_int, dtype=float)

    gauss_xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=float)
    gauss_w = np.array([1.0, 1.0], dtype=float)

    for e in range(n_nodes - 1):
        xl = x[e]
        xr = x[e + 1]
        h = xr - xl

        local_k = np.array([[1.0 / h, -1.0 / h], [-1.0 / h, 1.0 / h]], dtype=float)
        local_f = np.zeros(2, dtype=float)

        for xi, w in zip(gauss_xi, gauss_w):
            xq = 0.5 * ((1.0 - xi) * xl + (1.0 + xi) * xr)
            jac = 0.5 * h
            phi_l = (xr - xq) / h
            phi_r = (xq - xl) / h
            fq = forcing_term(np.array([xq]))[0]
            local_f[0] += w * fq * phi_l * jac
            local_f[1] += w * fq * phi_r * jac

        global_ids = [e, e + 1]
        for i_local, gi in enumerate(global_ids):
            if gi == 0 or gi == n_nodes - 1:
                continue
            ii = gi - 1
            rhs[ii] += local_f[i_local]
            for j_local, gj in enumerate(global_ids):
                if gj == 0 or gj == n_nodes - 1:
                    continue
                jj = gj - 1
                a_mat[ii, jj] += local_k[i_local, j_local]

    u_int = np.linalg.solve(a_mat, rhs)
    u = np.zeros(n_nodes, dtype=float)
    u[1:-1] = u_int
    return u


def compute_element_indicators(nodes: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Residual-like indicator using flux jumps across element interfaces."""
    x = np.asarray(nodes, dtype=float)
    uh = np.asarray(u, dtype=float)

    h = np.diff(x)
    slopes = np.diff(uh) / h

    jumps = np.zeros_like(x)
    jumps[1:-1] = np.abs(np.diff(slopes))

    indicators = h * np.sqrt(jumps[:-1] ** 2 + jumps[1:] ** 2)
    return indicators


def dorfler_mark(indicators: np.ndarray, theta: float) -> np.ndarray:
    """Mark a minimal set of cells whose indicator sum reaches theta fraction."""
    eta = np.asarray(indicators, dtype=float)
    if eta.size == 0:
        return np.array([], dtype=int)

    total = float(np.sum(eta))
    if total <= 0.0:
        return np.array([], dtype=int)

    order = np.argsort(-eta)
    target = theta * total
    accum = 0.0
    marked = []
    for idx in order:
        marked.append(int(idx))
        accum += float(eta[idx])
        if accum >= target:
            break
    return np.array(sorted(marked), dtype=int)


def refine_mesh_midpoint(nodes: np.ndarray, marked_cells: np.ndarray) -> np.ndarray:
    """Refine marked cells by midpoint bisection."""
    x = np.asarray(nodes, dtype=float)
    marked = set(int(i) for i in np.asarray(marked_cells, dtype=int))

    new_nodes = [float(x[0])]
    for e in range(x.size - 1):
        xl = float(x[e])
        xr = float(x[e + 1])
        if e in marked:
            new_nodes.append(0.5 * (xl + xr))
        new_nodes.append(xr)

    return np.asarray(new_nodes, dtype=float)


def l2_error(nodes: np.ndarray, u: np.ndarray) -> float:
    """Compute L2 error with 3-point Gauss quadrature per element."""
    x = np.asarray(nodes, dtype=float)
    uh = np.asarray(u, dtype=float)

    # 3-point Gauss-Legendre on [-1, 1]
    xis = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)], dtype=float)
    ws = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)

    acc = 0.0
    for e in range(x.size - 1):
        xl = x[e]
        xr = x[e + 1]
        h = xr - xl
        ul = uh[e]
        ur = uh[e + 1]

        for xi, w in zip(xis, ws):
            xq = 0.5 * ((1.0 - xi) * xl + (1.0 + xi) * xr)
            jac = 0.5 * h
            phi_l = (xr - xq) / h
            phi_r = (xq - xl) / h
            uq = ul * phi_l + ur * phi_r
            diff = uq - exact_solution(np.array([xq]))[0]
            acc += w * (diff**2) * jac

    return float(np.sqrt(acc))


def linf_error(nodes: np.ndarray, u: np.ndarray, n_probe: int = 2000) -> float:
    """Approximate max error on a dense probe grid."""
    x_probe = np.linspace(0.0, 1.0, n_probe)
    uh_probe = np.interp(x_probe, nodes, u)
    u_ex_probe = exact_solution(x_probe)
    return float(np.max(np.abs(uh_probe - u_ex_probe)))


def run_amr(config: AMRConfig) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Execute AMR loop and collect iteration history."""
    nodes = np.linspace(0.0, 1.0, config.initial_cells + 1)
    records: list[dict[str, float | int]] = []

    for k in range(config.max_iters):
        u = solve_poisson_fem_1d(nodes)
        indicators = compute_element_indicators(nodes, u)

        rec = {
            "iter": k,
            "cells": int(nodes.size - 1),
            "l2_error": l2_error(nodes, u),
            "linf_error": linf_error(nodes, u),
            "max_indicator": float(np.max(indicators) if indicators.size > 0 else 0.0),
            "indicator_sum": float(np.sum(indicators)),
        }
        records.append(rec)

        if rec["cells"] >= config.max_cells:
            break
        if rec["max_indicator"] < config.indicator_tol:
            break

        marked = dorfler_mark(indicators, theta=config.dorfler_theta)
        if marked.size == 0:
            break

        nodes_next = refine_mesh_midpoint(nodes, marked)
        if (nodes_next.size - 1) > config.max_cells:
            break
        if nodes_next.size == nodes.size:
            break
        nodes = nodes_next

    u_final = solve_poisson_fem_1d(nodes)
    history = pd.DataFrame.from_records(records)
    return nodes, u_final, history


def run_uniform_baseline(n_cells: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Uniform mesh baseline with the same number of cells as AMR final mesh."""
    nodes = np.linspace(0.0, 1.0, n_cells + 1)
    u = solve_poisson_fem_1d(nodes)
    return nodes, u, l2_error(nodes, u), linf_error(nodes, u)


def main() -> None:
    config = AMRConfig()

    nodes_amr, u_amr, history = run_amr(config)
    cells_final = int(nodes_amr.size - 1)

    _, _, l2_uniform, linf_uniform = run_uniform_baseline(cells_final)
    l2_amr = l2_error(nodes_amr, u_amr)
    linf_amr = linf_error(nodes_amr, u_amr)

    # Nonuniformity and localization metrics.
    h_amr = np.diff(nodes_amr)
    h_ratio = float(np.max(h_amr) / np.min(h_amr))
    local_window = (nodes_amr[:-1] >= 0.58) & (nodes_amr[1:] <= 0.82)
    local_cell_fraction = float(np.sum(local_window) / cells_final)

    summary = pd.DataFrame(
        {
            "quantity": [
                "final AMR cells",
                "uniform baseline cells",
                "AMR L2 error",
                "uniform L2 error",
                "L2 improvement ratio (uniform/AMR)",
                "AMR Linf error",
                "uniform Linf error",
                "Linf improvement ratio (uniform/AMR)",
                "max(h)/min(h) on AMR mesh",
                "cell fraction in [0.58, 0.82]",
                "AMR iterations",
            ],
            "value": [
                cells_final,
                cells_final,
                l2_amr,
                l2_uniform,
                l2_uniform / l2_amr,
                linf_amr,
                linf_uniform,
                linf_uniform / linf_amr,
                h_ratio,
                local_cell_fraction,
                int(history.shape[0]),
            ],
        }
    )

    checks = {
        "AMR loop executed at least 2 iterations": int(history.shape[0]) >= 2,
        "final cells > initial cells": cells_final > config.initial_cells,
        "AMR mesh is truly nonuniform (max(h)/min(h) > 1.2)": h_ratio > 1.2,
        "AMR L2 error <= uniform L2 error": l2_amr <= l2_uniform,
        "AMR Linf error <= uniform Linf error": linf_amr <= linf_uniform,
        "local refinement fraction in [0.58,0.82] >= 0.20": local_cell_fraction >= 0.20,
    }

    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    print("=== Adaptive Mesh Refinement MVP (PHYS-0361) ===")
    print(
        "PDE: -u''(x)=f(x), x in [0,1], u(0)=u(1)=0, "
        "with manufactured exact solution containing a localized feature"
    )
    print(
        f"config: initial_cells={config.initial_cells}, max_cells={config.max_cells}, "
        f"max_iters={config.max_iters}, theta={config.dorfler_theta}, "
        f"indicator_tol={config.indicator_tol}"
    )

    print("\nAMR history:")
    print(history.to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
