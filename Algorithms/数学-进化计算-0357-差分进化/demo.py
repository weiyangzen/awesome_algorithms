"""Differential Evolution (DE) minimal runnable MVP.

This script implements DE/rand/1/bin from scratch (no black-box optimizer)
and runs two deterministic benchmark cases without interactive input.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

HistoryItem = Tuple[int, float, float, float, int]


def ensure_bounds(bounds: np.ndarray) -> np.ndarray:
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"bounds must have shape (dim, 2), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("bounds contains non-finite values")
    lows = arr[:, 0]
    highs = arr[:, 1]
    if not np.all(highs > lows):
        raise ValueError("each bounds row must satisfy high > low")
    return arr


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    return float(10.0 * n + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def rosenbrock_nd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        raise ValueError("rosenbrock_nd requires dim >= 2")
    x_prev = x[:-1]
    x_next = x[1:]
    return float(np.sum(100.0 * (x_next - x_prev * x_prev) ** 2 + (1.0 - x_prev) ** 2))


def init_population(
    rng: np.random.Generator,
    pop_size: int,
    bounds: np.ndarray,
) -> np.ndarray:
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    return rng.uniform(lows, highs, size=(pop_size, bounds.shape[0]))


def population_diversity(population: np.ndarray) -> float:
    center = np.mean(population, axis=0)
    return float(np.mean(np.linalg.norm(population - center, axis=1)))


def differential_evolution(
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    pop_size: int = 80,
    max_gens: int = 500,
    F: float = 0.7,
    CR: float = 0.9,
    tol: float = 1e-10,
    seed: int = 2026,
) -> Dict[str, object]:
    bounds = ensure_bounds(bounds)
    dim = bounds.shape[0]

    if pop_size < 4:
        raise ValueError("pop_size must be >= 4 for DE/rand/1/bin")
    if max_gens <= 0:
        raise ValueError("max_gens must be > 0")
    if not (0.0 < F <= 2.0):
        raise ValueError("F must be in (0, 2]")
    if not (0.0 <= CR <= 1.0):
        raise ValueError("CR must be in [0, 1]")
    if tol <= 0.0:
        raise ValueError("tol must be > 0")

    rng = np.random.default_rng(seed)
    population = init_population(rng=rng, pop_size=pop_size, bounds=bounds)
    lows = bounds[:, 0]
    highs = bounds[:, 1]

    fitness = np.asarray([objective(ind) for ind in population], dtype=np.float64)
    if not np.all(np.isfinite(fitness)):
        raise RuntimeError("non-finite objective value found in initial population")

    history: List[HistoryItem] = []
    converged = False
    message = "max_gens reached"

    for gen in range(1, max_gens + 1):
        improved = 0

        for i in range(pop_size):
            candidate_ids = np.delete(np.arange(pop_size), i)
            r1, r2, r3 = rng.choice(candidate_ids, size=3, replace=False)

            mutant = population[r1] + F * (population[r2] - population[r3])
            mutant = np.clip(mutant, lows, highs)

            j_rand = int(rng.integers(0, dim))
            cross_mask = rng.random(dim) < CR
            cross_mask[j_rand] = True
            trial = np.where(cross_mask, mutant, population[i])
            trial = np.clip(trial, lows, highs)

            trial_f = float(objective(trial))
            if not np.isfinite(trial_f):
                continue

            if trial_f <= float(fitness[i]):
                population[i] = trial
                fitness[i] = trial_f
                improved += 1

        best_f = float(np.min(fitness))
        mean_f = float(np.mean(fitness))
        diversity = population_diversity(population)
        history.append((gen, best_f, mean_f, diversity, improved))

        if float(np.std(fitness)) < tol:
            converged = True
            message = "fitness std below tol"
            break

    best_idx = int(np.argmin(fitness))
    return {
        "x_best": population[best_idx].copy(),
        "f_best": float(fitness[best_idx]),
        "generations": len(history),
        "converged": converged,
        "message": message,
        "history": history,
        "final_fitness_std": float(np.std(fitness)),
    }


def relative_x_error(x: np.ndarray, x_ref: np.ndarray) -> float:
    denom = max(1.0, float(np.linalg.norm(x_ref, ord=2)))
    return float(np.linalg.norm(x - x_ref, ord=2) / denom)


def print_history(history: Sequence[HistoryItem], max_lines: int = 10) -> None:
    print("  gen | best_f             | mean_f             | diversity          | improved")
    print("  ----+--------------------+--------------------+--------------------+---------")
    for gen, best_f, mean_f, diversity, improved in history[:max_lines]:
        print(
            f"  {gen:>3d} | {best_f:>18.10e} | {mean_f:>18.10e} | {diversity:>18.10e} | {improved:>7d}"
        )
    if len(history) > max_lines:
        gen, best_f, mean_f, diversity, improved = history[-1]
        print(f"  ... ({len(history) - max_lines} more generations omitted)")
        print(
            f"  lst | {best_f:>18.10e} | {mean_f:>18.10e} | {diversity:>18.10e} | {improved:>7d}"
        )


def run_case(
    name: str,
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    x_ref: np.ndarray,
    known_optimum: float,
    config: Dict[str, float],
) -> Dict[str, float]:
    print(f"\nCase: {name}")
    bounds = ensure_bounds(bounds)
    dim = bounds.shape[0]
    print(
        f"  dim={dim}, pop_size={int(config['pop_size'])}, max_gens={int(config['max_gens'])}, "
        f"F={float(config['F']):.2f}, CR={float(config['CR']):.2f}"
    )

    result = differential_evolution(
        objective=objective,
        bounds=bounds,
        pop_size=int(config["pop_size"]),
        max_gens=int(config["max_gens"]),
        F=float(config["F"]),
        CR=float(config["CR"]),
        tol=float(config["tol"]),
        seed=int(config["seed"]),
    )

    history = result["history"]
    if not isinstance(history, list):
        raise RuntimeError("history format error")
    print_history(history, max_lines=10)

    x_best = np.asarray(result["x_best"], dtype=np.float64)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    abs_x_error = float(np.linalg.norm(x_best - x_ref, ord=2))
    rel_err = relative_x_error(x_best, x_ref)
    f_best = float(result["f_best"])
    gap = f_best - known_optimum

    print(f"  converged       = {result['converged']}")
    print(f"  message         = {result['message']}")
    print(f"  generations     = {result['generations']}")
    print(f"  f_best          = {f_best:.6e}")
    print(f"  final_fit_std   = {float(result['final_fitness_std']):.6e}")
    print(f"  abs_x_error     = {abs_x_error:.6e}")
    print(f"  rel_x_error     = {rel_err:.6e}")
    print(f"  optimality_gap  = {gap:.6e}")

    return {
        "optimality_gap": float(gap),
        "rel_x_error": float(rel_err),
        "generations": float(result["generations"]),
        "converged": 1.0 if bool(result["converged"]) else 0.0,
    }


def main() -> None:
    print("Differential Evolution MVP (DE/rand/1/bin)")
    print("=" * 84)

    rastrigin_dim = 5
    rosen_dim = 5

    ras_bounds = np.tile(np.array([[-5.12, 5.12]], dtype=np.float64), (rastrigin_dim, 1))
    ros_bounds = np.tile(np.array([[-3.0, 3.0]], dtype=np.float64), (rosen_dim, 1))

    cases = [
        {
            "name": "Rastrigin (5D)",
            "objective": rastrigin,
            "bounds": ras_bounds,
            "x_ref": np.zeros(rastrigin_dim, dtype=np.float64),
            "known_optimum": 0.0,
            "config": {
                "pop_size": 120,
                "max_gens": 1200,
                "F": 0.7,
                "CR": 0.92,
                "tol": 1e-12,
                "seed": 2026,
            },
        },
        {
            "name": "Rosenbrock (5D)",
            "objective": rosenbrock_nd,
            "bounds": ros_bounds,
            "x_ref": np.ones(rosen_dim, dtype=np.float64),
            "known_optimum": 0.0,
            "config": {
                "pop_size": 100,
                "max_gens": 800,
                "F": 0.6,
                "CR": 0.9,
                "tol": 1e-12,
                "seed": 2027,
            },
        },
    ]

    metrics: List[Dict[str, float]] = []
    for case in cases:
        metrics.append(
            run_case(
                name=str(case["name"]),
                objective=case["objective"],
                bounds=case["bounds"],
                x_ref=case["x_ref"],
                known_optimum=float(case["known_optimum"]),
                config=case["config"],
            )
        )

    gaps = np.asarray([m["optimality_gap"] for m in metrics], dtype=np.float64)
    rel_errors = np.asarray([m["rel_x_error"] for m in metrics], dtype=np.float64)
    gens = np.asarray([m["generations"] for m in metrics], dtype=np.float64)
    converged_ratio = float(np.mean([m["converged"] for m in metrics]))

    print("\nSummary")
    print("=" * 84)
    print(f"cases={len(metrics)}")
    print(f"max_optimality_gap={gaps.max():.6e}")
    print(f"mean_optimality_gap={gaps.mean():.6e}")
    print(f"max_rel_x_error={rel_errors.max():.6e}")
    print(f"mean_generations={gens.mean():.2f}")
    print(f"converged_ratio={converged_ratio:.2f}")


if __name__ == "__main__":
    main()
