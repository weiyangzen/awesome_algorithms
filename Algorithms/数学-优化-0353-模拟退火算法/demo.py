"""Simulated Annealing (SA) minimal runnable MVP.

The script implements SA from scratch (no optimization black box) and runs
fixed deterministic demo cases without interactive input.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

HistoryItem = Tuple[int, float, float, float, float, float]


def ensure_vector(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape={arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def ensure_bounds(bounds: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.shape != (dim, 2):
        raise ValueError(f"bounds must have shape ({dim}, 2), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("bounds contains non-finite values.")
    lows = arr[:, 0]
    highs = arr[:, 1]
    if not np.all(highs > lows):
        raise ValueError("each bounds row must satisfy high > low.")
    return arr


def rastrigin(x: np.ndarray) -> float:
    n = x.size
    return float(10.0 * n + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def himmelblau(x: np.ndarray) -> float:
    if x.size != 2:
        raise ValueError("himmelblau is defined for 2D vectors.")
    x1, x2 = float(x[0]), float(x[1])
    return float((x1 * x1 + x2 - 11.0) ** 2 + (x1 + x2 * x2 - 7.0) ** 2)


def relative_error(value: float, reference: float, eps: float = 1e-15) -> float:
    return abs(value - reference) / max(abs(reference), 1.0, eps)


def propose_neighbor(
    x: np.ndarray,
    bounds: np.ndarray,
    temp: float,
    temp_init: float,
    rng: np.random.Generator,
) -> np.ndarray:
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    widths = highs - lows

    ratio = max(temp / temp_init, 1e-8)
    sigma = widths * (0.02 + 0.25 * ratio)

    candidate = x + rng.normal(loc=0.0, scale=sigma, size=x.shape)
    return np.clip(candidate, lows, highs)


def simulated_annealing(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: np.ndarray,
    temp_init: float = 10.0,
    temp_min: float = 1e-3,
    cooling: float = 0.965,
    iters_per_temp: int = 120,
    max_iters: int = 50000,
    seed: int = 2026,
) -> Dict[str, object]:
    x0 = ensure_vector("x0", x0)
    bounds = ensure_bounds(bounds, x0.size)

    if temp_init <= 0.0:
        raise ValueError("temp_init must be > 0.")
    if temp_min <= 0.0:
        raise ValueError("temp_min must be > 0.")
    if temp_min >= temp_init:
        raise ValueError("temp_min must be smaller than temp_init.")
    if not (0.0 < cooling < 1.0):
        raise ValueError("cooling must be in (0, 1).")
    if iters_per_temp <= 0:
        raise ValueError("iters_per_temp must be > 0.")
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0.")

    rng = np.random.default_rng(seed)

    lows = bounds[:, 0]
    highs = bounds[:, 1]

    current = np.clip(x0.copy(), lows, highs)
    current_energy = float(objective(current))
    if not np.isfinite(current_energy):
        raise RuntimeError("objective(x0) is non-finite.")

    best = current.copy()
    best_energy = current_energy

    history: List[HistoryItem] = []
    total_iters = 0
    accepted_total = 0
    temperature = float(temp_init)

    while temperature > temp_min and total_iters < max_iters:
        accepted_this_temp = 0
        inner_count = min(iters_per_temp, max_iters - total_iters)

        for _ in range(inner_count):
            total_iters += 1
            candidate = propose_neighbor(
                x=current,
                bounds=bounds,
                temp=temperature,
                temp_init=temp_init,
                rng=rng,
            )
            candidate_energy = float(objective(candidate))
            if not np.isfinite(candidate_energy):
                raise RuntimeError("objective(candidate) is non-finite.")

            delta = candidate_energy - current_energy
            if delta <= 0.0:
                accept = True
            else:
                accept_prob = np.exp(-delta / max(temperature, 1e-12))
                accept = rng.random() < accept_prob

            if accept:
                current = candidate
                current_energy = candidate_energy
                accepted_total += 1
                accepted_this_temp += 1
                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy

        acceptance_rate = accepted_this_temp / inner_count
        distance_best_to_current = float(np.linalg.norm(best - current, ord=2))
        history.append(
            (
                total_iters,
                temperature,
                current_energy,
                best_energy,
                acceptance_rate,
                distance_best_to_current,
            )
        )

        temperature *= cooling

    return {
        "best_x": best,
        "best_energy": float(best_energy),
        "final_x": current,
        "final_energy": float(current_energy),
        "iterations": total_iters,
        "accepted_total": accepted_total,
        "acceptance_rate_total": accepted_total / max(total_iters, 1),
        "history": history,
        "final_temperature": temperature,
    }


def print_history(history: Sequence[HistoryItem], max_lines: int = 10) -> None:
    print(
        "temp_step | temperature      | current_energy   | best_energy      "
        "| accept_rate | ||best-current||"
    )
    print("-" * 106)

    for item in history[:max_lines]:
        step, temp, current_e, best_e, acc_rate, dist = item
        print(
            f"{step:9d} | {temp:16.9e} | {current_e:16.9e} | {best_e:16.9e} "
            f"| {acc_rate:11.6f} | {dist:16.9e}"
        )

    if len(history) > max_lines:
        step, temp, current_e, best_e, acc_rate, dist = history[-1]
        print(f"... ({len(history) - max_lines} more temperature stages omitted)")
        print(
            f"{step:9d} | {temp:16.9e} | {current_e:16.9e} | {best_e:16.9e} "
            f"| {acc_rate:11.6f} | {dist:16.9e}"
        )


def run_case(
    name: str,
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: np.ndarray,
    known_optimum: float,
    params: Dict[str, float],
) -> Dict[str, float]:
    print(f"\n=== Case: {name} ===")
    result = simulated_annealing(
        objective=objective,
        x0=x0,
        bounds=bounds,
        temp_init=float(params["temp_init"]),
        temp_min=float(params["temp_min"]),
        cooling=float(params["cooling"]),
        iters_per_temp=int(params["iters_per_temp"]),
        max_iters=int(params["max_iters"]),
        seed=int(params["seed"]),
    )

    history = result["history"]
    print_history(history)

    best_x = result["best_x"]
    best_energy = float(result["best_energy"])
    final_energy = float(result["final_energy"])
    iterations = int(result["iterations"])
    acc_rate_total = float(result["acceptance_rate_total"])
    opt_gap = best_energy - known_optimum
    rel_gap = relative_error(best_energy, known_optimum)

    print(f"best_x: {best_x}")
    print(f"best_energy: {best_energy:.9e}")
    print(f"known_optimum: {known_optimum:.9e}")
    print(f"optimality_gap: {opt_gap:.9e}")
    print(f"relative_gap: {rel_gap:.9e}")
    print(f"final_energy (at last state): {final_energy:.9e}")
    print(f"iterations: {iterations}")
    print(f"overall_acceptance_rate: {acc_rate_total:.6f}")

    return {
        "best_energy": best_energy,
        "optimality_gap": opt_gap,
        "relative_gap": rel_gap,
        "iterations": float(iterations),
        "overall_acceptance_rate": acc_rate_total,
    }


def main() -> None:
    common_params: Dict[str, float] = {
        "temp_init": 10.0,
        "temp_min": 1e-3,
        "cooling": 0.965,
        "iters_per_temp": 120,
        "max_iters": 50000,
        "seed": 2026,
    }

    cases = [
        {
            "name": "2D Rastrigin",
            "objective": rastrigin,
            "x0": np.array([4.5, -4.0], dtype=np.float64),
            "bounds": np.array([[-5.12, 5.12], [-5.12, 5.12]], dtype=np.float64),
            "known_optimum": 0.0,
        },
        {
            "name": "2D Himmelblau",
            "objective": himmelblau,
            "x0": np.array([-5.0, 5.0], dtype=np.float64),
            "bounds": np.array([[-6.0, 6.0], [-6.0, 6.0]], dtype=np.float64),
            "known_optimum": 0.0,
        },
    ]

    all_results = []
    for case in cases:
        case_result = run_case(
            name=str(case["name"]),
            objective=case["objective"],
            x0=case["x0"],
            bounds=case["bounds"],
            known_optimum=float(case["known_optimum"]),
            params=common_params,
        )
        all_results.append(case_result)

    max_gap = max(r["optimality_gap"] for r in all_results)
    avg_gap = float(np.mean([r["optimality_gap"] for r in all_results]))
    avg_acceptance = float(np.mean([r["overall_acceptance_rate"] for r in all_results]))

    pass_flag = max_gap < 0.1

    print("\n=== Summary ===")
    print(f"max optimality gap: {max_gap:.9e}")
    print(f"avg optimality gap: {avg_gap:.9e}")
    print(f"avg overall acceptance rate: {avg_acceptance:.6f}")
    print(f"pass criterion (max gap < 1e-1): {pass_flag}")


if __name__ == "__main__":
    main()
