"""Genetic Algorithm (GA) minimal runnable MVP.

This script implements a real-coded GA from scratch using only NumPy
(tournament selection + arithmetic crossover + Gaussian mutation + elitism)
and runs deterministic benchmark cases without interactive input.
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


def sphere(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(x * x))


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    return float(10.0 * n + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


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


def tournament_select_index(
    rng: np.random.Generator,
    fitness: np.ndarray,
    tournament_size: int,
) -> int:
    # Minimization: lower objective value means better individual.
    candidates = rng.choice(fitness.size, size=tournament_size, replace=False)
    winner_local = int(np.argmin(fitness[candidates]))
    return int(candidates[winner_local])


def arithmetic_crossover(
    rng: np.random.Generator,
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    crossover_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng.random() >= crossover_rate:
        return parent_a.copy(), parent_b.copy()

    alpha = rng.random(parent_a.size)
    child_a = alpha * parent_a + (1.0 - alpha) * parent_b
    child_b = alpha * parent_b + (1.0 - alpha) * parent_a
    return child_a, child_b


def gaussian_mutation(
    rng: np.random.Generator,
    individual: np.ndarray,
    bounds: np.ndarray,
    mutation_rate: float,
    mutation_scale: float,
) -> int:
    dim = individual.size
    mask = rng.random(dim) < mutation_rate
    mutated_genes = int(np.sum(mask))
    if mutated_genes == 0:
        return 0

    span = bounds[:, 1] - bounds[:, 0]
    noise = rng.normal(loc=0.0, scale=mutation_scale * span, size=dim)
    individual[mask] += noise[mask]
    np.clip(individual, bounds[:, 0], bounds[:, 1], out=individual)
    return mutated_genes


def genetic_algorithm(
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    pop_size: int = 80,
    max_gens: int = 300,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    elite_count: int = 2,
    tournament_size: int = 3,
    tol: float = 1e-10,
    seed: int = 2026,
) -> Dict[str, object]:
    bounds = ensure_bounds(bounds)
    dim = bounds.shape[0]

    if pop_size < 4:
        raise ValueError("pop_size must be >= 4")
    if max_gens <= 0:
        raise ValueError("max_gens must be > 0")
    if not (0.0 <= crossover_rate <= 1.0):
        raise ValueError("crossover_rate must be in [0, 1]")
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be in [0, 1]")
    if mutation_scale <= 0.0:
        raise ValueError("mutation_scale must be > 0")
    if not (0 <= elite_count < pop_size):
        raise ValueError("elite_count must satisfy 0 <= elite_count < pop_size")
    if not (2 <= tournament_size <= pop_size):
        raise ValueError("tournament_size must be in [2, pop_size]")
    if tol <= 0.0:
        raise ValueError("tol must be > 0")

    rng = np.random.default_rng(seed)
    population = init_population(rng=rng, pop_size=pop_size, bounds=bounds)
    fitness = np.asarray([objective(ind) for ind in population], dtype=np.float64)

    if not np.all(np.isfinite(fitness)):
        raise RuntimeError("non-finite objective value found in initial population")

    history: List[HistoryItem] = []
    converged = False
    message = "max_gens reached"

    for gen in range(1, max_gens + 1):
        ranked_idx = np.argsort(fitness)
        elites = population[ranked_idx[:elite_count]].copy() if elite_count > 0 else np.empty((0, dim))

        offspring: List[np.ndarray] = []
        mutated_genes = 0

        while len(offspring) < pop_size - elite_count:
            pa_idx = tournament_select_index(rng, fitness, tournament_size=tournament_size)
            pb_idx = tournament_select_index(rng, fitness, tournament_size=tournament_size)

            parent_a = population[pa_idx]
            parent_b = population[pb_idx]
            child_a, child_b = arithmetic_crossover(
                rng=rng,
                parent_a=parent_a,
                parent_b=parent_b,
                crossover_rate=crossover_rate,
            )

            mutated_genes += gaussian_mutation(
                rng=rng,
                individual=child_a,
                bounds=bounds,
                mutation_rate=mutation_rate,
                mutation_scale=mutation_scale,
            )
            offspring.append(child_a)
            if len(offspring) >= pop_size - elite_count:
                break

            mutated_genes += gaussian_mutation(
                rng=rng,
                individual=child_b,
                bounds=bounds,
                mutation_rate=mutation_rate,
                mutation_scale=mutation_scale,
            )
            offspring.append(child_b)

        next_population = np.vstack([elites, np.asarray(offspring, dtype=np.float64)])
        if next_population.shape != (pop_size, dim):
            raise RuntimeError(f"population shape error: {next_population.shape}")

        next_fitness = np.asarray([objective(ind) for ind in next_population], dtype=np.float64)
        if not np.all(np.isfinite(next_fitness)):
            raise RuntimeError("non-finite objective value found in evolution")

        population = next_population
        fitness = next_fitness

        best_f = float(np.min(fitness))
        mean_f = float(np.mean(fitness))
        diversity = population_diversity(population)
        history.append((gen, best_f, mean_f, diversity, mutated_genes))

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
    print("  gen | best_f             | mean_f             | diversity          | mutated_genes")
    print("  ----+--------------------+--------------------+--------------------+---------------")
    for gen, best_f, mean_f, diversity, mutated_genes in history[:max_lines]:
        print(
            f"  {gen:>3d} | {best_f:>18.10e} | {mean_f:>18.10e} | {diversity:>18.10e} | {mutated_genes:>13d}"
        )
    if len(history) > max_lines:
        gen, best_f, mean_f, diversity, mutated_genes = history[-1]
        print(f"  ... ({len(history) - max_lines} more generations omitted)")
        print(
            f"  lst | {best_f:>18.10e} | {mean_f:>18.10e} | {diversity:>18.10e} | {mutated_genes:>13d}"
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
        f"pc={float(config['crossover_rate']):.2f}, pm={float(config['mutation_rate']):.2f}, "
        f"sigma_ratio={float(config['mutation_scale']):.2f}"
    )

    result = genetic_algorithm(
        objective=objective,
        bounds=bounds,
        pop_size=int(config["pop_size"]),
        max_gens=int(config["max_gens"]),
        crossover_rate=float(config["crossover_rate"]),
        mutation_rate=float(config["mutation_rate"]),
        mutation_scale=float(config["mutation_scale"]),
        elite_count=int(config["elite_count"]),
        tournament_size=int(config["tournament_size"]),
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
    print("Genetic Algorithm MVP (real-coded GA)")
    print("=" * 84)

    sphere_dim = 8
    rastrigin_dim = 8

    sph_bounds = np.tile(np.array([[-5.0, 5.0]], dtype=np.float64), (sphere_dim, 1))
    ras_bounds = np.tile(np.array([[-5.12, 5.12]], dtype=np.float64), (rastrigin_dim, 1))

    cases = [
        {
            "name": "Sphere (8D)",
            "objective": sphere,
            "bounds": sph_bounds,
            "x_ref": np.zeros(sphere_dim, dtype=np.float64),
            "known_optimum": 0.0,
            "config": {
                "pop_size": 96,
                "max_gens": 300,
                "crossover_rate": 0.92,
                "mutation_rate": 0.14,
                "mutation_scale": 0.08,
                "elite_count": 2,
                "tournament_size": 3,
                "tol": 1e-12,
                "seed": 2026,
            },
        },
        {
            "name": "Rastrigin (8D)",
            "objective": rastrigin,
            "bounds": ras_bounds,
            "x_ref": np.zeros(rastrigin_dim, dtype=np.float64),
            "known_optimum": 0.0,
            "config": {
                "pop_size": 120,
                "max_gens": 480,
                "crossover_rate": 0.95,
                "mutation_rate": 0.18,
                "mutation_scale": 0.10,
                "elite_count": 4,
                "tournament_size": 4,
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
