"""Minimal runnable MVP for Stochastic Programming (MATH-0421).

This demo implements a two-product, two-stage stochastic inventory planning
problem. The expectation term is approximated by SAA (sample average
approximation), and the convex non-smooth objective is solved by projected
subgradient descent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class SolveResult:
    """Container for one optimization run."""

    x: np.ndarray
    best_objective: float
    objective_history: List[float]


def sample_demands(
    rng: np.random.Generator,
    n: int,
    mean: np.ndarray,
    cov: np.ndarray,
    spike_prob: float,
    spike_size: np.ndarray,
) -> np.ndarray:
    """Sample nonnegative 2D demands with occasional spike events."""
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal(size=(n, mean.shape[0]))
    base = mean[None, :] + np.einsum("ni,ij->nj", z, chol.T)
    spikes = (rng.random(n) < spike_prob).astype(np.float64)[:, None] * spike_size[None, :]
    return np.clip(base + spikes, a_min=0.0, a_max=None)


def build_datasets(seed: int = 421) -> Dict[str, np.ndarray]:
    """Create train / in-distribution test / shifted test demand scenarios."""
    rng = np.random.default_rng(seed)

    train = sample_demands(
        rng=rng,
        n=240,
        mean=np.array([92.0, 58.0]),
        cov=np.array([[225.0, 84.0], [84.0, 156.0]]),
        spike_prob=0.18,
        spike_size=np.array([30.0, 16.0]),
    )
    test_id = sample_demands(
        rng=rng,
        n=4000,
        mean=np.array([92.0, 58.0]),
        cov=np.array([[225.0, 84.0], [84.0, 156.0]]),
        spike_prob=0.18,
        spike_size=np.array([30.0, 16.0]),
    )
    test_shift = sample_demands(
        rng=rng,
        n=4000,
        mean=np.array([103.0, 66.0]),
        cov=np.array([[324.0, 120.0], [120.0, 225.0]]),
        spike_prob=0.26,
        spike_size=np.array([36.0, 22.0]),
    )

    return {
        "train": train,
        "test_id": test_id,
        "test_shift": test_shift,
    }


def stochastic_objective(
    x: np.ndarray,
    demands: np.ndarray,
    unit_purchase: np.ndarray,
    shortage_penalty: np.ndarray,
    holding_penalty: np.ndarray,
) -> float:
    """Compute SAA objective value at x.

    f(x) = c^T x + (1/N) * sum_s [p^T(d_s - x)_+ + h^T(x - d_s)_+]
    """
    shortages = np.maximum(demands - x[None, :], 0.0)
    holdings = np.maximum(x[None, :] - demands, 0.0)
    scenario_terms = (
        np.sum(shortages * shortage_penalty[None, :], axis=1)
        + np.sum(holdings * holding_penalty[None, :], axis=1)
    )
    return float(np.sum(unit_purchase * x) + np.mean(scenario_terms))


def stochastic_subgradient(
    x: np.ndarray,
    demands: np.ndarray,
    unit_purchase: np.ndarray,
    shortage_penalty: np.ndarray,
    holding_penalty: np.ndarray,
) -> np.ndarray:
    """Compute one valid subgradient of the SAA objective."""
    indicator_short = (x[None, :] < demands).astype(np.float64)
    indicator_hold = (x[None, :] > demands).astype(np.float64)

    scenario_grad = (
        -indicator_short * shortage_penalty[None, :]
        + indicator_hold * holding_penalty[None, :]
    )
    return unit_purchase + np.mean(scenario_grad, axis=0)


def project_to_budget_nonnegative(
    z: np.ndarray,
    weight: np.ndarray,
    budget: float,
    tol: float = 1e-10,
    max_iter: int = 80,
) -> np.ndarray:
    """Euclidean projection to {x >= 0, weight^T x <= budget}.

    KKT form for lambda >= 0:
        x_i(lambda) = max(0, z_i - lambda * weight_i)
    Solve lambda by bisection when budget is active.
    """
    if budget < 0.0:
        raise ValueError("budget must be nonnegative")
    if np.any(weight <= 0.0):
        raise ValueError("all weight coefficients must be positive")

    z_pos = np.maximum(z, 0.0)
    if float(np.sum(weight * z_pos)) <= budget + tol:
        return z_pos

    lam_lo = 0.0
    lam_hi = float(np.max(z_pos / weight)) + 1.0

    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        x_mid = np.maximum(0.0, z_pos - lam_mid * weight)
        if float(np.sum(weight * x_mid)) > budget:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    return np.maximum(0.0, z_pos - lam_hi * weight)


def fit_saa_subgradient(
    demands: np.ndarray,
    unit_purchase: np.ndarray,
    shortage_penalty: np.ndarray,
    holding_penalty: np.ndarray,
    weight: np.ndarray,
    budget: float,
    lr0: float,
    epochs: int,
) -> SolveResult:
    """Minimize SAA objective by projected subgradient descent."""
    dim = demands.shape[1]
    x = project_to_budget_nonnegative(
        z=np.full(dim, budget / np.sum(weight), dtype=np.float64),
        weight=weight,
        budget=budget,
    )

    objective_history: List[float] = []
    best_x = x.copy()
    best_obj = stochastic_objective(
        x=best_x,
        demands=demands,
        unit_purchase=unit_purchase,
        shortage_penalty=shortage_penalty,
        holding_penalty=holding_penalty,
    )

    objective_history.append(best_obj)

    for t in range(1, epochs + 1):
        g = stochastic_subgradient(
            x=x,
            demands=demands,
            unit_purchase=unit_purchase,
            shortage_penalty=shortage_penalty,
            holding_penalty=holding_penalty,
        )
        step = lr0 / np.sqrt(float(t))
        x = project_to_budget_nonnegative(
            z=x - step * g,
            weight=weight,
            budget=budget,
        )

        obj = stochastic_objective(
            x=x,
            demands=demands,
            unit_purchase=unit_purchase,
            shortage_penalty=shortage_penalty,
            holding_penalty=holding_penalty,
        )
        objective_history.append(obj)

        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()

    return SolveResult(x=best_x, best_objective=best_obj, objective_history=objective_history)


def solve_expected_value_baseline(
    train_demands: np.ndarray,
    unit_purchase: np.ndarray,
    shortage_penalty: np.ndarray,
    holding_penalty: np.ndarray,
    weight: np.ndarray,
    budget: float,
) -> SolveResult:
    """Deterministic baseline: replace all uncertainty with mean demand."""
    mean_scenario = np.mean(train_demands, axis=0, keepdims=True)
    return fit_saa_subgradient(
        demands=mean_scenario,
        unit_purchase=unit_purchase,
        shortage_penalty=shortage_penalty,
        holding_penalty=holding_penalty,
        weight=weight,
        budget=budget,
        lr0=3.0,
        epochs=240,
    )


def evaluate_policy(
    x: np.ndarray,
    demands: np.ndarray,
    unit_purchase: np.ndarray,
    shortage_penalty: np.ndarray,
    holding_penalty: np.ndarray,
) -> float:
    """Return average total cost on given demand scenarios."""
    return stochastic_objective(
        x=x,
        demands=demands,
        unit_purchase=unit_purchase,
        shortage_penalty=shortage_penalty,
        holding_penalty=holding_penalty,
    )


def main() -> None:
    print("Stochastic Programming MVP (MATH-0421)")
    print("=" * 72)

    data = build_datasets(seed=421)

    unit_purchase = np.array([2.0, 1.8], dtype=np.float64)
    shortage_penalty = np.array([12.0, 9.5], dtype=np.float64)
    holding_penalty = np.array([1.0, 0.8], dtype=np.float64)

    weight = np.array([1.0, 1.25], dtype=np.float64)
    budget = 172.0

    saa = fit_saa_subgradient(
        demands=data["train"],
        unit_purchase=unit_purchase,
        shortage_penalty=shortage_penalty,
        holding_penalty=holding_penalty,
        weight=weight,
        budget=budget,
        lr0=4.2,
        epochs=420,
    )
    ev = solve_expected_value_baseline(
        train_demands=data["train"],
        unit_purchase=unit_purchase,
        shortage_penalty=shortage_penalty,
        holding_penalty=holding_penalty,
        weight=weight,
        budget=budget,
    )

    costs = {
        "train": {
            "SAA": evaluate_policy(
                saa.x,
                data["train"],
                unit_purchase,
                shortage_penalty,
                holding_penalty,
            ),
            "EV": evaluate_policy(
                ev.x,
                data["train"],
                unit_purchase,
                shortage_penalty,
                holding_penalty,
            ),
        },
        "id": {
            "SAA": evaluate_policy(
                saa.x,
                data["test_id"],
                unit_purchase,
                shortage_penalty,
                holding_penalty,
            ),
            "EV": evaluate_policy(
                ev.x,
                data["test_id"],
                unit_purchase,
                shortage_penalty,
                holding_penalty,
            ),
        },
        "shift": {
            "SAA": evaluate_policy(
                saa.x,
                data["test_shift"],
                unit_purchase,
                shortage_penalty,
                holding_penalty,
            ),
            "EV": evaluate_policy(
                ev.x,
                data["test_shift"],
                unit_purchase,
                shortage_penalty,
                holding_penalty,
            ),
        },
    }

    def format_vec(vec: np.ndarray) -> str:
        return "[" + ", ".join(f"{v:.3f}" for v in vec) + "]"

    print(f"Budget weights a: {format_vec(weight)}, budget B={budget:.2f}")
    print(f"SAA decision x*: {format_vec(saa.x)} | train objective(best): {saa.best_objective:.4f}")
    print(f"EV  decision x*: {format_vec(ev.x)} | mean-scenario objective(best): {ev.best_objective:.4f}")
    print("-" * 72)
    print("Average total cost (lower is better):")
    for split in ("train", "id", "shift"):
        c_saa = costs[split]["SAA"]
        c_ev = costs[split]["EV"]
        gain = c_ev - c_saa
        print(f"{split:>5} | SAA={c_saa:.4f} | EV={c_ev:.4f} | EV-SAA={gain:+.4f}")

    print("-" * 72)
    budget_saa = float(np.sum(weight * saa.x))
    budget_ev = float(np.sum(weight * ev.x))
    print(f"Feasibility check: a^T x_saa={budget_saa:.6f}, a^T x_ev={budget_ev:.6f}")

    train_drop = saa.objective_history[0] - min(saa.objective_history)
    print(f"SAA objective decrease from init to best: {train_drop:.6f}")
    print("=" * 72)

    if np.any(saa.x < -1e-9) or np.any(ev.x < -1e-9):
        raise RuntimeError("nonnegativity constraint violated")
    if budget_saa > budget + 1e-6 or budget_ev > budget + 1e-6:
        raise RuntimeError("budget constraint violated")
    if not (min(saa.objective_history) <= saa.objective_history[0] - 1e-6):
        raise RuntimeError("SAA optimization failed to improve objective")
    if not (costs["train"]["SAA"] <= costs["train"]["EV"] + 1e-6):
        raise RuntimeError("SAA should not be worse than EV on training scenarios")

    print("Run completed successfully.")


if __name__ == "__main__":
    main()
