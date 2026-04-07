"""Minimal runnable MVP for Lagrangian relaxation.

Problem used in this demo (binary combinatorial optimization):

    maximize   sum_i v_i x_i
    subject to sum_i w_i x_i <= W      (kept as hard constraint)
               sum_i t_i x_i <= T      (relaxed by Lagrange multiplier)
               x_i in {0, 1}

The relaxed subproblem becomes a 0-1 knapsack with modified profits,
solved here by dynamic programming.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class BinarySelectionInstance:
    values: np.ndarray
    weights: np.ndarray
    times: np.ndarray
    capacity_weight: int
    capacity_time: int


def build_instance() -> BinarySelectionInstance:
    """Create a deterministic small instance."""
    values = np.array([16, 19, 23, 28, 14, 11, 24, 17, 22, 13, 27, 15, 18, 21], dtype=float)
    weights = np.array([4, 5, 7, 8, 3, 2, 6, 5, 7, 3, 9, 4, 5, 6], dtype=int)
    times = np.array([5, 6, 8, 7, 4, 3, 6, 5, 9, 4, 8, 3, 5, 6], dtype=int)
    return BinarySelectionInstance(
        values=values,
        weights=weights,
        times=times,
        capacity_weight=28,
        capacity_time=26,
    )


def solve_knapsack_dp(profits: np.ndarray, weights: np.ndarray, capacity: int) -> Tuple[np.ndarray, float, int]:
    """Solve max sum(profits_i * x_i) with sum(weights_i * x_i) <= capacity, x_i in {0,1}."""
    n = int(len(profits))
    dp = np.full((n + 1, capacity + 1), -np.inf, dtype=float)
    take = np.zeros((n + 1, capacity + 1), dtype=np.int8)
    dp[0, :] = 0.0

    for i in range(1, n + 1):
        p = float(profits[i - 1])
        w = int(weights[i - 1])
        dp[i, :] = dp[i - 1, :]
        for cap in range(w, capacity + 1):
            candidate = dp[i - 1, cap - w] + p
            if candidate > dp[i, cap]:
                dp[i, cap] = candidate
                take[i, cap] = 1

    best_cap = int(np.argmax(dp[n, :]))
    best_value = float(dp[n, best_cap])

    x = np.zeros(n, dtype=int)
    cap = best_cap
    for i in range(n, 0, -1):
        if take[i, cap] == 1:
            x[i - 1] = 1
            cap -= int(weights[i - 1])

    used_weight = int(np.dot(weights, x))
    return x, best_value, used_weight


def lagrangian_subproblem(instance: BinarySelectionInstance, lam: float) -> Tuple[np.ndarray, float, int]:
    """Solve theta(lam) and return (relaxed solution, dual value, violation)."""
    adjusted_profit = instance.values - lam * instance.times
    x_relaxed, core_value, _ = solve_knapsack_dp(
        adjusted_profit,
        instance.weights,
        instance.capacity_weight,
    )
    dual_value = core_value + lam * instance.capacity_time
    violation = int(np.dot(instance.times, x_relaxed) - instance.capacity_time)
    return x_relaxed, float(dual_value), violation


def repair_to_feasible(instance: BinarySelectionInstance, x: np.ndarray) -> np.ndarray:
    """Repair a possibly time-infeasible solution into a feasible one by dropping items."""
    x_rep = x.copy().astype(int)
    used_weight = int(np.dot(instance.weights, x_rep))
    used_time = int(np.dot(instance.times, x_rep))
    if used_time <= instance.capacity_time:
        pass
    else:
        selected = np.where(x_rep == 1)[0].tolist()
        # Remove items with the weakest value density first.
        selected.sort(key=lambda i: (instance.values[i] / max(instance.times[i], 1), -instance.times[i]))

        for idx in selected:
            if used_time <= instance.capacity_time:
                break
            x_rep[idx] = 0
            used_time -= int(instance.times[idx])
            used_weight -= int(instance.weights[idx])

    # Greedy refill to improve value while keeping both constraints feasible.
    unselected = np.where(x_rep == 0)[0].tolist()
    unselected.sort(key=lambda i: instance.values[i] / (instance.weights[i] + instance.times[i]), reverse=True)
    for idx in unselected:
        w = int(instance.weights[idx])
        t = int(instance.times[idx])
        if used_weight + w <= instance.capacity_weight and used_time + t <= instance.capacity_time:
            x_rep[idx] = 1
            used_weight += w
            used_time += t

    # One-item swap local search.
    improved = True
    while improved:
        improved = False
        selected = np.where(x_rep == 1)[0].tolist()
        unselected = np.where(x_rep == 0)[0].tolist()
        for i in unselected:
            for j in selected:
                gain = float(instance.values[i] - instance.values[j])
                if gain <= 0:
                    continue
                new_weight = used_weight + int(instance.weights[i]) - int(instance.weights[j])
                new_time = used_time + int(instance.times[i]) - int(instance.times[j])
                if new_weight <= instance.capacity_weight and new_time <= instance.capacity_time:
                    x_rep[i] = 1
                    x_rep[j] = 0
                    used_weight = new_weight
                    used_time = new_time
                    improved = True
                    break
            if improved:
                break

    return x_rep


def objective_value(values: np.ndarray, x: np.ndarray) -> float:
    return float(np.dot(values, x))


def brute_force_optimal(instance: BinarySelectionInstance) -> Tuple[float, np.ndarray]:
    """Exact solve by enumeration (small-instance validator)."""
    n = int(len(instance.values))
    best_val = -math.inf
    best_x = np.zeros(n, dtype=int)

    for mask in range(1 << n):
        x = np.fromiter(((mask >> i) & 1 for i in range(n)), dtype=int, count=n)
        total_weight = int(np.dot(instance.weights, x))
        if total_weight > instance.capacity_weight:
            continue
        total_time = int(np.dot(instance.times, x))
        if total_time > instance.capacity_time:
            continue

        val = objective_value(instance.values, x)
        if val > best_val:
            best_val = val
            best_x = x.copy()

    return float(best_val), best_x


def run_lagrangian_relaxation(
    instance: BinarySelectionInstance,
    max_iter: int = 120,
    lambda0: float = 0.0,
) -> Dict[str, object]:
    """Subgradient loop for the Lagrangian dual."""
    lam = float(lambda0)
    best_dual_ub = math.inf
    best_primal_lb = -math.inf
    best_feasible = np.zeros_like(instance.weights)
    history: List[Dict[str, float]] = []
    step_scale = 2.5
    last_violation: int | None = None

    for k in range(max_iter):
        x_relaxed, dual_value, violation = lagrangian_subproblem(instance, lam)
        best_dual_ub = min(best_dual_ub, dual_value)

        x_feasible = repair_to_feasible(instance, x_relaxed)
        primal_val = objective_value(instance.values, x_feasible)
        if primal_val > best_primal_lb:
            best_primal_lb = primal_val
            best_feasible = x_feasible.copy()

        # Diminishing subgradient step with sign-change damping.
        if last_violation is not None and violation * last_violation < 0:
            step_scale = max(step_scale * 0.7, 0.05)
        step = step_scale / math.sqrt(k + 1.0)
        lam = max(0.0, lam + step * violation)
        last_violation = violation

        history.append(
            {
                "iter": float(k),
                "lambda": float(lam),
                "dual": float(dual_value),
                "best_dual_ub": float(best_dual_ub),
                "best_primal_lb": float(best_primal_lb),
                "violation": float(violation),
                "step": float(step),
            }
        )

        if best_dual_ub - best_primal_lb <= 1e-8:
            break
        if step < 1e-4 and abs(violation) <= 1:
            break

    return {
        "best_dual_ub": float(best_dual_ub),
        "best_primal_lb": float(best_primal_lb),
        "best_feasible_x": best_feasible,
        "history": history,
    }


def selected_items(x: np.ndarray) -> List[int]:
    """Convert binary vector to 1-based item indices."""
    return [int(i + 1) for i in np.where(x == 1)[0]]


def main() -> None:
    instance = build_instance()

    exact_val, exact_x = brute_force_optimal(instance)
    result = run_lagrangian_relaxation(instance)

    lb = float(result["best_primal_lb"])
    ub = float(result["best_dual_ub"])
    best_x = np.asarray(result["best_feasible_x"], dtype=int)
    gap = ub - lb
    rel_gap = gap / max(abs(lb), 1.0)

    print("=== Lagrangian Relaxation Demo (MATH-0376) ===")
    print(f"items={len(instance.values)}, weight_cap={instance.capacity_weight}, time_cap={instance.capacity_time}")
    print("---")
    print(f"Exact primal optimum (bruteforce): {exact_val:.4f}")
    print(f"Lagrangian best primal lower bound: {lb:.4f}")
    print(f"Lagrangian best dual upper bound:   {ub:.4f}")
    print(f"Duality gap (UB-LB): {gap:.4f}")
    print(f"Relative gap: {rel_gap:.4%}")
    print("---")
    print(f"Best feasible selection (1-based): {selected_items(best_x)}")
    print(
        "Resource usage of best feasible: "
        f"weight={int(np.dot(instance.weights, best_x))}/{instance.capacity_weight}, "
        f"time={int(np.dot(instance.times, best_x))}/{instance.capacity_time}"
    )
    print(f"Exact optimal selection (1-based): {selected_items(exact_x)}")

    # Simple sanity checks for bounds.
    if exact_val - ub > 1e-6:
        raise RuntimeError("Dual upper bound is below exact optimum, which should not happen.")
    if lb - exact_val > 1e-6:
        raise RuntimeError("Lower bound is above exact optimum, which should not happen.")

    history = result["history"]
    print("---")
    print(f"Iterations run: {len(history)}")
    print("Last 5 iterations (iter, lambda, dual, best_ub, best_lb, viol, step):")
    for row in history[-5:]:
        print(
            f"{int(row['iter']):3d} | {row['lambda']:8.4f} | {row['dual']:8.4f} | "
            f"{row['best_dual_ub']:8.4f} | {row['best_primal_lb']:8.4f} | "
            f"{row['violation']:5.0f} | {row['step']:7.4f}"
        )


if __name__ == "__main__":
    main()
