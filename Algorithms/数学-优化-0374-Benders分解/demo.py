"""Benders decomposition MVP (continuous master + LP recourse)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class BendersData:
    """Container for the two-stage LP in standard Benders form."""

    c: np.ndarray
    h: np.ndarray
    T: np.ndarray
    W: np.ndarray
    q: np.ndarray
    A_master: np.ndarray
    b_master: np.ndarray
    x_bounds: Sequence[Tuple[float, float]]


@dataclass(frozen=True)
class BendersCut:
    """One optimality cut: theta >= alpha - beta^T x."""

    alpha: float
    beta: np.ndarray
    pi: np.ndarray


def build_demo_instance() -> BendersData:
    """Create a tiny complete-recourse two-stage LP instance."""
    # First-stage cost and constraints
    c = np.array([0.9, 1.1], dtype=float)
    A_master = np.array([[1.0, 1.0]], dtype=float)
    b_master = np.array([8.0], dtype=float)
    x_bounds = [(0.0, 6.0), (0.0, 6.0)]

    # Second-stage primal: min q^T y s.t. W y >= h - T x, y >= 0
    h = np.array([7.0, 12.0, 10.0], dtype=float)
    T = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 1.0],
        ],
        dtype=float,
    )
    W = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=float,
    )
    q = np.array([3.0, 2.0], dtype=float)

    return BendersData(
        c=c,
        h=h,
        T=T,
        W=W,
        q=q,
        A_master=A_master,
        b_master=b_master,
        x_bounds=x_bounds,
    )


def solve_master_problem(
    data: BendersData,
    cuts: List[BendersCut],
    theta_lb: float,
) -> Tuple[np.ndarray, float, float]:
    """Solve the relaxed master problem with current cuts."""
    nx = data.c.size
    objective = np.concatenate([data.c, np.array([1.0])])  # min c^T x + theta

    A_ub_rows: List[np.ndarray] = []
    b_ub_vals: List[float] = []

    # Original master constraints: A_master x <= b_master
    for row, rhs in zip(data.A_master, data.b_master):
        A_ub_rows.append(np.concatenate([row, np.array([0.0])]))
        b_ub_vals.append(float(rhs))

    # Benders optimality cuts: theta >= alpha - beta^T x
    for cut in cuts:
        # Rearranged to linprog's <= form: -beta^T x - theta <= -alpha
        A_ub_rows.append(np.concatenate([-cut.beta, np.array([-1.0])]))
        b_ub_vals.append(-cut.alpha)

    A_ub = np.array(A_ub_rows, dtype=float)
    b_ub = np.array(b_ub_vals, dtype=float)

    bounds = list(data.x_bounds) + [(theta_lb, None)]
    result = linprog(
        c=objective,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if result.status != 0:
        raise RuntimeError(f"Master problem failed: {result.message}")

    x = result.x[:nx]
    theta = float(result.x[-1])
    obj = float(result.fun)
    return x, theta, obj


def solve_dual_subproblem(data: BendersData, x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Solve the dual LP of recourse subproblem for fixed x.

    Primal recourse:
        min q^T y
        s.t. W y >= h - T x, y >= 0

    Dual:
        max pi^T (h - T x)
        s.t. W^T pi <= q, pi >= 0
    """
    rhs = data.h - data.T @ x
    objective = -rhs  # maximize rhs^T pi == minimize -rhs^T pi

    result = linprog(
        c=objective,
        A_ub=data.W.T,
        b_ub=data.q,
        bounds=[(0.0, None)] * data.h.size,
        method="highs",
    )

    if result.status != 0:
        raise RuntimeError(f"Dual subproblem failed: {result.message}")

    pi = result.x
    value = float(pi @ rhs)
    return value, pi, rhs


def solve_extensive_form(data: BendersData) -> Tuple[float, np.ndarray, np.ndarray]:
    """Solve the original LP directly for validation."""
    nx = data.c.size
    ny = data.q.size

    # Variables are z = [x(2), y(2)]
    objective = np.concatenate([data.c, data.q])

    A_ub_rows: List[np.ndarray] = []
    b_ub_vals: List[float] = []

    # Enforce T x + W y >= h  ->  -T x - W y <= -h
    for i in range(data.h.size):
        row = np.concatenate([-data.T[i], -data.W[i]])
        A_ub_rows.append(row)
        b_ub_vals.append(-float(data.h[i]))

    # Original first-stage constraints
    for row, rhs in zip(data.A_master, data.b_master):
        A_ub_rows.append(np.concatenate([row, np.zeros(ny, dtype=float)]))
        b_ub_vals.append(float(rhs))

    bounds = list(data.x_bounds) + [(0.0, None)] * ny

    result = linprog(
        c=objective,
        A_ub=np.array(A_ub_rows, dtype=float),
        b_ub=np.array(b_ub_vals, dtype=float),
        bounds=bounds,
        method="highs",
    )

    if result.status != 0:
        raise RuntimeError(f"Extensive form solve failed: {result.message}")

    x = result.x[:nx]
    y = result.x[nx:]
    return float(result.fun), x, y


def benders_decomposition(
    data: BendersData,
    tol: float = 1e-7,
    max_iters: int = 50,
    theta_lb: float = 0.0,
) -> Tuple[float, np.ndarray, List[dict]]:
    """Run a basic single-cut Benders decomposition loop."""
    cuts: List[BendersCut] = []
    history: List[dict] = []

    best_ub = float("inf")
    best_x = None

    for it in range(1, max_iters + 1):
        x, theta, lb = solve_master_problem(data, cuts, theta_lb)
        recourse_value, pi, _ = solve_dual_subproblem(data, x)

        full_obj = float(data.c @ x + recourse_value)
        if full_obj < best_ub:
            best_ub = full_obj
            best_x = x.copy()

        violation = recourse_value - theta
        gap = best_ub - lb

        history.append(
            {
                "iter": it,
                "x1": float(x[0]),
                "x2": float(x[1]),
                "theta": theta,
                "sub": recourse_value,
                "LB": lb,
                "UB": best_ub,
                "gap": gap,
                "cut_violation": violation,
            }
        )

        if violation <= tol and gap <= tol:
            break

        # Build cut from dual extreme point.
        alpha = float(pi @ data.h)
        beta = data.T.T @ pi
        cuts.append(BendersCut(alpha=alpha, beta=beta, pi=pi.copy()))
    else:
        raise RuntimeError("Benders did not converge within max_iters.")

    assert best_x is not None
    return best_ub, best_x, history


def print_history(history: List[dict]) -> None:
    """Pretty print iteration logs."""
    header = (
        "iter |    x1    x2 |  theta    sub |    LB      UB    gap | cut_violation"
    )
    print(header)
    print("-" * len(header))
    for row in history:
        print(
            f"{row['iter']:>4d} |"
            f" {row['x1']:>5.3f} {row['x2']:>5.3f} |"
            f" {row['theta']:>6.3f} {row['sub']:>6.3f} |"
            f" {row['LB']:>7.3f} {row['UB']:>7.3f} {row['gap']:>6.3e} |"
            f" {row['cut_violation']:>11.3e}"
        )


def main() -> None:
    data = build_demo_instance()

    benders_obj, benders_x, history = benders_decomposition(
        data=data,
        tol=1e-8,
        max_iters=30,
        theta_lb=0.0,
    )

    direct_obj, direct_x, direct_y = solve_extensive_form(data)

    print("=== Benders decomposition iteration log ===")
    print_history(history)

    print("\n=== Final solutions ===")
    print(f"Benders objective: {benders_obj:.6f}")
    print(f"Benders x*: [{benders_x[0]:.6f}, {benders_x[1]:.6f}]")
    print(f"Direct LP objective: {direct_obj:.6f}")
    print(f"Direct LP x*: [{direct_x[0]:.6f}, {direct_x[1]:.6f}]")
    print(f"Direct LP y*: [{direct_y[0]:.6f}, {direct_y[1]:.6f}]")

    diff = abs(benders_obj - direct_obj)
    print(f"Objective difference: {diff:.3e}")

    if diff > 1e-6:
        raise AssertionError("Benders result does not match direct LP within tolerance.")


if __name__ == "__main__":
    main()
