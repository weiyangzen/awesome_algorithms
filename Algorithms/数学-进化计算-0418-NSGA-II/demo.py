"""NSGA-II minimal runnable MVP (bi-objective ZDT1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NSGA2Config:
    pop_size: int = 80
    n_var: int = 30
    n_obj: int = 2
    generations: int = 80
    crossover_prob: float = 0.9
    mutation_prob: float | None = None
    eta_c: float = 15.0
    eta_m: float = 20.0
    seed: int = 42


def zdt1_objectives(x: np.ndarray) -> np.ndarray:
    """Evaluate ZDT1 objective vector for one decision vector x in [0, 1]^n."""
    f1 = x[0]
    g = 1.0 + 9.0 * np.mean(x[1:])
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.array([f1, f2], dtype=float)


def evaluate_population(pop: np.ndarray) -> np.ndarray:
    return np.vstack([zdt1_objectives(ind) for ind in pop])


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(objs: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    n = objs.shape[0]
    dominates_set: list[list[int]] = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=int)
    rank = np.full(n, -1, dtype=int)

    first_front: list[int] = []
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                dominates_set[p].append(q)
            elif dominates(objs[q], objs[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            rank[p] = 0
            first_front.append(p)

    fronts: list[np.ndarray] = [np.array(first_front, dtype=int)]
    i = 0
    while i < len(fronts) and fronts[i].size > 0:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominates_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        if next_front:
            fronts.append(np.array(next_front, dtype=int))
        i += 1

    return fronts, rank


def crowding_distance(objs: np.ndarray, front: np.ndarray) -> np.ndarray:
    dist = np.zeros(front.size, dtype=float)
    if front.size == 0:
        return dist
    if front.size <= 2:
        dist[:] = np.inf
        return dist

    n_obj = objs.shape[1]
    front_objs = objs[front]

    for m in range(n_obj):
        order = np.argsort(front_objs[:, m])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf

        min_v = front_objs[order[0], m]
        max_v = front_objs[order[-1], m]
        if max_v <= min_v:
            continue

        for k in range(1, front.size - 1):
            prev_v = front_objs[order[k - 1], m]
            next_v = front_objs[order[k + 1], m]
            if np.isfinite(dist[order[k]]):
                dist[order[k]] += (next_v - prev_v) / (max_v - min_v)

    return dist


def assign_rank_and_crowding(objs: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    fronts, rank = fast_non_dominated_sort(objs)
    crowd = np.zeros(objs.shape[0], dtype=float)
    for front in fronts:
        c = crowding_distance(objs, front)
        crowd[front] = c
    return rank, crowd, fronts


def binary_tournament(rng: np.random.Generator, rank: np.ndarray, crowd: np.ndarray) -> int:
    i, j = rng.integers(0, rank.size, size=2)
    if rank[i] < rank[j]:
        return int(i)
    if rank[j] < rank[i]:
        return int(j)
    if crowd[i] > crowd[j]:
        return int(i)
    if crowd[j] > crowd[i]:
        return int(j)
    return int(i if rng.random() < 0.5 else j)


def sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
    eta_c: float,
    low: float = 0.0,
    high: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    c1 = p1.copy()
    c2 = p2.copy()

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
        child1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))

        beta = 1.0 + 2.0 * (high - x2) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if rand <= 1.0 / alpha:
            beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
        child2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))

        child1 = float(np.clip(child1, low, high))
        child2 = float(np.clip(child2, low, high))

        if rng.random() <= 0.5:
            c1[k], c2[k] = child2, child1
        else:
            c1[k], c2[k] = child1, child2

    return c1, c2


def polynomial_mutation(
    child: np.ndarray,
    rng: np.random.Generator,
    eta_m: float,
    mutation_prob: float,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    y = child.copy()
    for k in range(y.size):
        if rng.random() > mutation_prob:
            continue

        x = y[k]
        if x <= low:
            x = low
        elif x >= high:
            x = high

        u = rng.random()
        if u < 0.5:
            delta = (2.0 * u + (1.0 - 2.0 * u) * (1.0 - x) ** (eta_m + 1.0)) ** (
                1.0 / (eta_m + 1.0)
            ) - 1.0
        else:
            delta = 1.0 - (
                2.0 * (1.0 - u) + 2.0 * (u - 0.5) * x ** (eta_m + 1.0)
            ) ** (1.0 / (eta_m + 1.0))

        y[k] = float(np.clip(x + delta, low, high))

    return y


def make_offspring(
    pop: np.ndarray,
    rank: np.ndarray,
    crowd: np.ndarray,
    cfg: NSGA2Config,
    rng: np.random.Generator,
) -> np.ndarray:
    children = []
    while len(children) < cfg.pop_size:
        i = binary_tournament(rng, rank, crowd)
        j = binary_tournament(rng, rank, crowd)
        p1 = pop[i]
        p2 = pop[j]

        if rng.random() < cfg.crossover_prob:
            c1, c2 = sbx_crossover(p1, p2, rng, eta_c=cfg.eta_c)
        else:
            c1, c2 = p1.copy(), p2.copy()

        pm = cfg.mutation_prob if cfg.mutation_prob is not None else 1.0 / cfg.n_var
        c1 = polynomial_mutation(c1, rng, eta_m=cfg.eta_m, mutation_prob=pm)
        c2 = polynomial_mutation(c2, rng, eta_m=cfg.eta_m, mutation_prob=pm)

        children.append(c1)
        if len(children) < cfg.pop_size:
            children.append(c2)

    return np.array(children, dtype=float)


def environmental_selection(pop: np.ndarray, objs: np.ndarray, pop_size: int) -> tuple[np.ndarray, np.ndarray]:
    fronts, _ = fast_non_dominated_sort(objs)

    selected_idx: list[int] = []
    for front in fronts:
        if len(selected_idx) + front.size <= pop_size:
            selected_idx.extend(front.tolist())
            continue

        remain = pop_size - len(selected_idx)
        dist = crowding_distance(objs, front)
        order = np.argsort(-dist)
        selected_idx.extend(front[order[:remain]].tolist())
        break

    idx = np.array(selected_idx, dtype=int)
    return pop[idx], objs[idx]


def run_nsga2(cfg: NSGA2Config) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)

    pop = rng.random((cfg.pop_size, cfg.n_var), dtype=float)
    objs = evaluate_population(pop)

    for _ in range(cfg.generations):
        rank, crowd, _ = assign_rank_and_crowding(objs)
        offspring = make_offspring(pop, rank, crowd, cfg, rng)
        off_objs = evaluate_population(offspring)

        merged_pop = np.vstack([pop, offspring])
        merged_objs = np.vstack([objs, off_objs])
        pop, objs = environmental_selection(merged_pop, merged_objs, cfg.pop_size)

    return pop, objs


def summarize_pareto(objs: np.ndarray) -> None:
    fronts, _ = fast_non_dominated_sort(objs)
    pareto_idx = fronts[0]
    pareto = objs[pareto_idx]

    order = np.argsort(pareto[:, 0])
    pareto = pareto[order]

    print("NSGA-II demo finished.")
    print(f"Population size: {objs.shape[0]}")
    print(f"Approximate Pareto front size: {pareto.shape[0]}")
    print("First 8 Pareto points (f1, f2):")
    for row in pareto[:8]:
        print(f"  ({row[0]:.4f}, {row[1]:.4f})")



def main() -> None:
    cfg = NSGA2Config(
        pop_size=80,
        n_var=30,
        n_obj=2,
        generations=80,
        crossover_prob=0.9,
        mutation_prob=None,
        eta_c=15.0,
        eta_m=20.0,
        seed=42,
    )
    _, objs = run_nsga2(cfg)
    summarize_pareto(objs)


if __name__ == "__main__":
    main()
