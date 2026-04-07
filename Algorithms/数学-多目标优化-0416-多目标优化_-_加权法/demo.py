"""Bi-objective optimization MVP via the weighted-sum method.

This script demonstrates how to turn a 2-objective minimization problem into
single-objective subproblems by scanning weight vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class WeightedProblem1D:
    """Container for a 1D bi-objective optimization problem."""

    lower: float
    upper: float
    f1: Callable[[float], float]
    f2: Callable[[float], float]


def build_problem() -> WeightedProblem1D:
    """Create a simple convex bi-objective problem on [0, 2].

    Objectives (both to minimize):
        f1(x) = x^2
        f2(x) = (x - 2)^2

    On x in [0,2], f1 increases while f2 decreases, so the objectives conflict.
    """

    return WeightedProblem1D(
        lower=0.0,
        upper=2.0,
        f1=lambda x: float(x * x),
        f2=lambda x: float((x - 2.0) * (x - 2.0)),
    )


def closed_form_solution(w1: float) -> float:
    """Closed-form optimizer for this toy problem.

    For w2 = 1 - w1:
        minimize w1*x^2 + w2*(x-2)^2  on [0,2]
    gives:
        x* = 2*(1 - w1)
    """

    return 2.0 * (1.0 - w1)


def solve_weighted_subproblem(problem: WeightedProblem1D, w1: float, w2: float) -> dict[str, float | str]:
    """Solve one weighted scalarized subproblem.

    Returns a dict with decision, objectives, weighted objective, and solver stats.
    """

    tol = 1e-12
    if w1 < -tol or w2 < -tol:
        raise ValueError(f"Invalid negative weights: w1={w1}, w2={w2}")
    if not np.isclose(w1 + w2, 1.0, atol=1e-12):
        raise ValueError(f"Weights must sum to 1. Got w1+w2={w1 + w2:.16f}")

    def phi(x: float) -> float:
        return w1 * problem.f1(x) + w2 * problem.f2(x)

    result = minimize_scalar(
        phi,
        bounds=(problem.lower, problem.upper),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )

    x_star = float(result.x)
    f1 = problem.f1(x_star)
    f2 = problem.f2(x_star)
    weighted_obj = float(result.fun)

    x_cf = closed_form_solution(w1)

    return {
        "w1": float(w1),
        "w2": float(w2),
        "x_star": x_star,
        "f1": f1,
        "f2": f2,
        "weighted_obj": weighted_obj,
        "x_closed_form": float(x_cf),
        "abs_err_closed_form": float(abs(x_star - x_cf)),
        "status": "optimal" if bool(result.success) else "failed",
        "nit": float(result.nit),
        "nfev": float(result.nfev),
        "message": str(result.message),
    }


def weight_sweep(problem: WeightedProblem1D, num_weights: int = 17) -> pd.DataFrame:
    """Run weighted-sum optimization over a grid of weights."""

    w1_values = np.linspace(0.0, 1.0, num_weights)
    rows: list[dict[str, float | str]] = []

    for w1 in w1_values:
        w2 = 1.0 - float(w1)
        rows.append(solve_weighted_subproblem(problem, float(w1), float(w2)))

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


def validate_results(results: pd.DataFrame, pareto: pd.DataFrame) -> None:
    """Sanity checks for weighted-sum output."""

    assert not results.empty, "No weighted-sum results produced."
    assert (results["status"] == "optimal").all(), "Some weighted subproblems failed."

    tol = 1e-7
    assert np.allclose(
        results["w1"].to_numpy(dtype=float) + results["w2"].to_numpy(dtype=float),
        1.0,
        atol=1e-12,
    ), "Weights do not sum to 1."

    x_star = results["x_star"].to_numpy(dtype=float)
    assert np.all(x_star >= -tol) and np.all(x_star <= 2.0 + tol), "x_star out of bounds [0,2]."

    x_closed_form = results["x_closed_form"].to_numpy(dtype=float)
    assert np.allclose(x_star, x_closed_form, atol=5e-6), "Numerical solution deviates from closed-form solution."

    ordered = results.sort_values("w1", ascending=True).reset_index(drop=True)
    x_diff = np.diff(ordered["x_star"].to_numpy(dtype=float))
    f1_diff = np.diff(ordered["f1"].to_numpy(dtype=float))
    f2_diff = np.diff(ordered["f2"].to_numpy(dtype=float))

    # As w1 increases, optimizer shifts toward minimizing f1.
    assert np.all(x_diff <= tol), "x_star should be non-increasing with larger w1."
    assert np.all(f1_diff <= tol), "f1 should be non-increasing with larger w1."
    assert np.all(f2_diff >= -tol), "f2 should be non-decreasing with larger w1."

    assert len(pareto) > 0, "Pareto set is unexpectedly empty."


def main() -> None:
    problem = build_problem()
    results = weight_sweep(problem, num_weights=17)
    pareto = pareto_filter_min(results[["w1", "w2", "x_star", "f1", "f2"]])
    pareto = pareto.sort_values(["f1", "f2"], ascending=[True, True]).reset_index(drop=True)

    validate_results(results, pareto)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)

    print("=== Weighted-Sum Sweep Results ===")
    print(
        results[
            [
                "w1",
                "w2",
                "x_star",
                "f1",
                "f2",
                "weighted_obj",
                "x_closed_form",
                "abs_err_closed_form",
                "status",
                "nit",
                "nfev",
            ]
        ]
        .round(8)
        .to_string(index=False)
    )

    print("\n=== Nondominated Points (Pareto Approximation) ===")
    print(pareto.round(8).to_string(index=False))

    anchor_f1 = results.loc[results["f1"].idxmin(), ["w1", "w2", "x_star", "f1", "f2"]]
    anchor_f2 = results.loc[results["f2"].idxmin(), ["w1", "w2", "x_star", "f1", "f2"]]

    print("\nAnchor (best f1):")
    print(anchor_f1.to_string())
    print("\nAnchor (best f2):")
    print(anchor_f2.to_string())


if __name__ == "__main__":
    main()
