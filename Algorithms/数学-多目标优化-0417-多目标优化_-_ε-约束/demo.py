"""Bi-objective optimization MVP via the epsilon-constraint method.

This script builds a small linear bi-objective problem and traces a Pareto
frontier by repeatedly solving single-objective LP subproblems with an
additional epsilon bound on the second objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class BiObjectiveLP:
    """Container for a bi-objective linear programming problem."""

    c1: np.ndarray  # objective 1 coefficients (minimize)
    c2: np.ndarray  # objective 2 coefficients (minimize)
    A_ub: np.ndarray
    b_ub: np.ndarray
    bounds: tuple[tuple[float, float | None], ...]


def build_problem() -> BiObjectiveLP:
    """Create a 2-variable LP with conflicting objectives.

    Constraints:
        x1 + x2 >= 1
        2x1 + x2 <= 4
        x1 + 2x2 <= 4
        x1, x2 >= 0

    Objectives:
        f1 = x1 + 4x2
        f2 = 4x1 + x2
    """
    c1 = np.array([1.0, 4.0], dtype=float)
    c2 = np.array([4.0, 1.0], dtype=float)

    # Convert to A_ub x <= b_ub form for scipy.linprog.
    A_ub = np.array(
        [
            [-1.0, -1.0],  # x1 + x2 >= 1
            [2.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )
    b_ub = np.array([-1.0, 4.0, 4.0], dtype=float)
    bounds = ((0.0, None), (0.0, None))

    return BiObjectiveLP(c1=c1, c2=c2, A_ub=A_ub, b_ub=b_ub, bounds=bounds)


def solve_lp(
    objective: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    bounds: tuple[tuple[float, float | None], ...],
) -> tuple[bool, np.ndarray | None, str]:
    """Solve a linear program and return success flag, solution, and status."""
    result = linprog(c=objective, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if result.success:
        return True, np.asarray(result.x, dtype=float), str(result.message)
    return False, None, str(result.message)


def objective_values(x: np.ndarray, problem: BiObjectiveLP) -> tuple[float, float]:
    """Return (f1, f2) for a decision vector x."""
    f1 = float(problem.c1 @ x)
    f2 = float(problem.c2 @ x)
    return f1, f2


def epsilon_constraint_sweep(problem: BiObjectiveLP, num_points: int = 11) -> pd.DataFrame:
    """Sweep epsilon values and solve epsilon-constrained subproblems.

    Subproblem for each epsilon:
        minimize f1(x)
        s.t. base constraints
             f2(x) <= epsilon
    """
    ok_f1, x_f1_star, msg_f1 = solve_lp(problem.c1, problem.A_ub, problem.b_ub, problem.bounds)
    ok_f2, x_f2_star, msg_f2 = solve_lp(problem.c2, problem.A_ub, problem.b_ub, problem.bounds)
    if not ok_f1 or x_f1_star is None:
        raise RuntimeError(f"Failed to solve f1 anchor problem: {msg_f1}")
    if not ok_f2 or x_f2_star is None:
        raise RuntimeError(f"Failed to solve f2 anchor problem: {msg_f2}")

    _, f2_at_f1_star = objective_values(x_f1_star, problem)
    _, f2_min = objective_values(x_f2_star, problem)

    eps_values = np.linspace(f2_min, f2_at_f1_star, num_points)

    rows: list[dict[str, float | str]] = []
    for eps in eps_values:
        A_eps = np.vstack([problem.A_ub, problem.c2])
        b_eps = np.hstack([problem.b_ub, eps])
        ok, x_star, msg = solve_lp(problem.c1, A_eps, b_eps, problem.bounds)

        if not ok or x_star is None:
            rows.append(
                {
                    "epsilon": float(eps),
                    "x1": np.nan,
                    "x2": np.nan,
                    "f1": np.nan,
                    "f2": np.nan,
                    "constraint_slack": np.nan,
                    "status": "infeasible_or_error",
                    "message": msg,
                }
            )
            continue

        f1, f2 = objective_values(x_star, problem)
        rows.append(
            {
                "epsilon": float(eps),
                "x1": float(x_star[0]),
                "x2": float(x_star[1]),
                "f1": f1,
                "f2": f2,
                "constraint_slack": float(eps - f2),
                "status": "optimal",
                "message": msg,
            }
        )

    return pd.DataFrame(rows)


def pareto_filter_min(df: pd.DataFrame, cols: tuple[str, str] = ("f1", "f2"), tol: float = 1e-9) -> pd.DataFrame:
    """Keep nondominated points for minimization objectives."""
    points = df.loc[:, list(cols)].to_numpy(dtype=float)
    n = points.shape[0]
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            no_worse = np.all(points[j] <= points[i] + tol)
            strictly_better_somewhere = np.any(points[j] < points[i] - tol)
            if no_worse and strictly_better_somewhere:
                keep[i] = False
                break

    return df.loc[keep].copy()


def validate_results(df: pd.DataFrame) -> None:
    """Sanity checks for the epsilon-constraint run."""
    assert not df.empty, "No results produced."
    assert (df["status"] == "optimal").all(), "Some epsilon subproblems are infeasible or failed."

    tol = 1e-7
    assert (df["f2"] <= df["epsilon"] + tol).all(), "Epsilon-constraint f2 <= epsilon violated."

    ordered = df.sort_values("epsilon", ascending=True).reset_index(drop=True)

    # Relaxing epsilon should not worsen the optimal f1 value.
    f1_diff = np.diff(ordered["f1"].to_numpy(dtype=float))
    assert np.all(f1_diff <= tol), "f1 should be non-increasing as epsilon increases."

    # In this constructed LP, frontier points satisfy approximately f1 + f2 = 5.
    frontier_sum = ordered["f1"].to_numpy(dtype=float) + ordered["f2"].to_numpy(dtype=float)
    assert np.allclose(frontier_sum, 5.0, atol=1e-6), "Unexpected frontier geometry for this toy problem."


def main() -> None:
    problem = build_problem()
    results = epsilon_constraint_sweep(problem, num_points=13)
    validate_results(results)

    pareto = pareto_filter_min(results[["epsilon", "x1", "x2", "f1", "f2"]])
    pareto = pareto.sort_values(["f2", "f1"], ascending=[True, True]).reset_index(drop=True)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print("=== Epsilon-Constraint Sweep Results ===")
    print(results[["epsilon", "x1", "x2", "f1", "f2", "constraint_slack", "status"]].round(6).to_string(index=False))

    print("\n=== Nondominated Points (Pareto Approximation) ===")
    print(pareto.round(6).to_string(index=False))

    anchor_f1 = results.loc[results["f1"].idxmin(), ["epsilon", "f1", "f2", "x1", "x2"]]
    anchor_f2 = results.loc[results["f2"].idxmin(), ["epsilon", "f1", "f2", "x1", "x2"]]
    print("\nAnchor (best f1):")
    print(anchor_f1.to_string())
    print("\nAnchor (best f2):")
    print(anchor_f2.to_string())


if __name__ == "__main__":
    main()
