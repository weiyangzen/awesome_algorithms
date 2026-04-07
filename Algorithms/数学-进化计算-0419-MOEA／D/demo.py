"""MOEA/D minimal runnable MVP (bi-objective ZDT1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MOEADConfig:
    pop_size: int = 91
    n_var: int = 30
    n_obj: int = 2
    generations: int = 120
    neighborhood_size: int = 15
    crossover_prob: float = 1.0
    mutation_prob: float | None = None
    eta_c: float = 20.0
    eta_m: float = 20.0
    delta: float = 0.9
    seed: int = 42


def zdt1_objectives(x: np.ndarray) -> np.ndarray:
    """Evaluate ZDT1 objectives for one decision vector x in [0, 1]^n."""
    f1 = x[0]
    g = 1.0 + 9.0 * np.mean(x[1:])
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.array([f1, f2], dtype=float)


def evaluate_population(pop: np.ndarray) -> np.ndarray:
    return np.vstack([zdt1_objectives(ind) for ind in pop])


def generate_weight_vectors(pop_size: int, n_obj: int) -> np.ndarray:
    if n_obj != 2:
        raise ValueError("This MVP currently supports only bi-objective problems.")
    if pop_size < 2:
        raise ValueError("pop_size must be >= 2 for bi-objective weights.")

    ws = np.linspace(0.0, 1.0, pop_size)
    return np.column_stack([ws, 1.0 - ws])


def build_neighborhood(weights: np.ndarray, neighborhood_size: int) -> np.ndarray:
    n = weights.shape[0]
    if neighborhood_size <= 0:
        raise ValueError("neighborhood_size must be positive.")
    t = min(neighborhood_size, n)

    diff = weights[:, None, :] - weights[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return np.argsort(dist, axis=1)[:, :t]


def tchebycheff_value(obj: np.ndarray, weight: np.ndarray, ideal: np.ndarray) -> float:
    safe_weight = np.where(weight <= 1e-12, 1e-6, weight)
    return float(np.max(safe_weight * np.abs(obj - ideal)))


def choose_parents(
    i: int,
    neighborhoods: np.ndarray,
    pop_size: int,
    delta: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if rng.random() < delta:
        pool = neighborhoods[i]
    else:
        pool = np.arange(pop_size)

    if pool.size < 2:
        return int(pool[0]), int(pool[0])

    p1, p2 = rng.choice(pool, size=2, replace=False)
    return int(p1), int(p2)


def sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
    eta_c: float,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    child = p1.copy()

    for k in range(p1.size):
        if rng.random() > 0.5:
            continue
        if abs(p1[k] - p2[k]) < 1e-14:
            continue

        x1 = min(p1[k], p2[k])
        x2 = max(p1[k], p2[k])
        rand = rng.random()

        beta = 1.0 + 2.0 * (x1 - low) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if rand <= 1.0 / alpha:
            beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
        c1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))

        beta = 1.0 + 2.0 * (high - x2) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if rand <= 1.0 / alpha:
            beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
        c2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))

        c1 = float(np.clip(c1, low, high))
        c2 = float(np.clip(c2, low, high))
        child[k] = c1 if rng.random() < 0.5 else c2

    return child


def polynomial_mutation(
    x: np.ndarray,
    rng: np.random.Generator,
    eta_m: float,
    mutation_prob: float,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    y = x.copy()

    for k in range(y.size):
        if rng.random() > mutation_prob:
            continue

        v = float(np.clip(y[k], low, high))
        u = rng.random()

        if u < 0.5:
            delta = (2.0 * u + (1.0 - 2.0 * u) * (1.0 - v) ** (eta_m + 1.0)) ** (
                1.0 / (eta_m + 1.0)
            ) - 1.0
        else:
            delta = 1.0 - (
                2.0 * (1.0 - u) + 2.0 * (u - 0.5) * v ** (eta_m + 1.0)
            ) ** (1.0 / (eta_m + 1.0))

        y[k] = float(np.clip(v + delta, low, high))

    return y


def non_dominated_indices(objs: np.ndarray) -> np.ndarray:
    n = objs.shape[0]
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                is_dominated[i] = True
                break

    return np.where(~is_dominated)[0]


def run_moead(cfg: MOEADConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(cfg.seed)

    weights = generate_weight_vectors(cfg.pop_size, cfg.n_obj)
    neighborhoods = build_neighborhood(weights, cfg.neighborhood_size)

    pop = rng.random((cfg.pop_size, cfg.n_var), dtype=float)
    objs = evaluate_population(pop)
    ideal = np.min(objs, axis=0)

    history: list[dict[str, float]] = []
    pm = cfg.mutation_prob if cfg.mutation_prob is not None else 1.0 / cfg.n_var

    for gen in range(cfg.generations):
        replacements = 0
        for i in rng.permutation(cfg.pop_size):
            p1_idx, p2_idx = choose_parents(i, neighborhoods, cfg.pop_size, cfg.delta, rng)
            p1, p2 = pop[p1_idx], pop[p2_idx]

            if rng.random() < cfg.crossover_prob:
                child = sbx_crossover(p1, p2, rng, eta_c=cfg.eta_c)
            else:
                child = 0.5 * (p1 + p2)

            child = polynomial_mutation(child, rng, eta_m=cfg.eta_m, mutation_prob=pm)
            child_obj = zdt1_objectives(child)
            ideal = np.minimum(ideal, child_obj)

            for j in neighborhoods[i]:
                old_score = tchebycheff_value(objs[j], weights[j], ideal)
                new_score = tchebycheff_value(child_obj, weights[j], ideal)
                if new_score <= old_score:
                    pop[j] = child
                    objs[j] = child_obj
                    replacements += 1

        history.append(
            {
                "generation": float(gen + 1),
                "ideal_f1": float(ideal[0]),
                "ideal_f2": float(ideal[1]),
                "replacements": float(replacements),
            }
        )

    return pop, objs, weights, history


def summarize_results(objs: np.ndarray, history: list[dict[str, float]]) -> None:
    nd_idx = non_dominated_indices(objs)
    pareto = objs[nd_idx]
    pareto = pareto[np.argsort(pareto[:, 0])]

    ref_curve = 1.0 - np.sqrt(np.clip(pareto[:, 0], 0.0, 1.0))
    mae = float(np.mean(np.abs(pareto[:, 1] - ref_curve)))

    print("MOEA/D demo finished.")
    print(f"Population size: {objs.shape[0]}")
    print(f"Non-dominated set size: {pareto.shape[0]}")
    print(f"Mean |f2 - (1-sqrt(f1))| on ND set: {mae:.6f}")

    print("First 8 non-dominated points (f1, f2):")
    for row in pareto[:8]:
        print(f"  ({row[0]:.4f}, {row[1]:.4f})")

    print("Last generation diagnostics:")
    last = history[-1]
    print(
        "  generation={gen:.0f}, ideal=({f1:.4f}, {f2:.4f}), replacements={rep:.0f}".format(
            gen=last["generation"],
            f1=last["ideal_f1"],
            f2=last["ideal_f2"],
            rep=last["replacements"],
        )
    )


def main() -> None:
    cfg = MOEADConfig(
        pop_size=91,
        n_var=30,
        n_obj=2,
        generations=120,
        neighborhood_size=15,
        crossover_prob=1.0,
        mutation_prob=None,
        eta_c=20.0,
        eta_m=20.0,
        delta=0.9,
        seed=42,
    )

    _, objs, _, history = run_moead(cfg)
    summarize_results(objs, history)


if __name__ == "__main__":
    main()
