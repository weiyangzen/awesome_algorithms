"""Particle Swarm Optimization (PSO) minimal runnable MVP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PSOConfig:
    dimensions: int = 5
    swarm_size: int = 40
    max_iters: int = 200
    lower_bound: float = -5.12
    upper_bound: float = 5.12
    inertia_start: float = 0.90
    inertia_end: float = 0.40
    cognitive_coeff: float = 1.80
    social_coeff: float = 1.80
    velocity_clip_ratio: float = 0.20
    target_fitness: float = 1e-8
    seed: int = 42


def validate_config(cfg: PSOConfig) -> None:
    if cfg.dimensions <= 0:
        raise ValueError("dimensions must be positive.")
    if cfg.swarm_size <= 1:
        raise ValueError("swarm_size must be > 1.")
    if cfg.max_iters <= 0:
        raise ValueError("max_iters must be positive.")
    if not np.isfinite(cfg.lower_bound) or not np.isfinite(cfg.upper_bound):
        raise ValueError("bounds must be finite numbers.")
    if cfg.lower_bound >= cfg.upper_bound:
        raise ValueError("lower_bound must be less than upper_bound.")
    if cfg.velocity_clip_ratio <= 0:
        raise ValueError("velocity_clip_ratio must be positive.")
    if cfg.target_fitness < 0:
        raise ValueError("target_fitness must be >= 0.")


def rastrigin_population(x: np.ndarray) -> np.ndarray:
    """Vectorized Rastrigin evaluation for shape (n_particles, dimensions)."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}.")
    a = 10.0
    d = x.shape[1]
    return a * d + np.sum(x * x - a * np.cos(2.0 * np.pi * x), axis=1)


def inertia_weight(iter_idx: int, max_iters: int, w_start: float, w_end: float) -> float:
    if max_iters == 1:
        return w_end
    progress = iter_idx / (max_iters - 1)
    return w_start + (w_end - w_start) * progress


def run_pso(cfg: PSOConfig) -> tuple[np.ndarray, float, np.ndarray]:
    validate_config(cfg)
    rng = np.random.default_rng(cfg.seed)

    span = cfg.upper_bound - cfg.lower_bound
    vmax = cfg.velocity_clip_ratio * span

    positions = rng.uniform(
        cfg.lower_bound,
        cfg.upper_bound,
        size=(cfg.swarm_size, cfg.dimensions),
    )
    velocities = rng.uniform(-vmax, vmax, size=(cfg.swarm_size, cfg.dimensions))

    pbest_positions = positions.copy()
    pbest_fitness = rastrigin_population(positions)

    best_idx = int(np.argmin(pbest_fitness))
    gbest_position = pbest_positions[best_idx].copy()
    gbest_fitness = float(pbest_fitness[best_idx])

    history = [gbest_fitness]

    for it in range(cfg.max_iters):
        w = inertia_weight(it, cfg.max_iters, cfg.inertia_start, cfg.inertia_end)
        r1 = rng.random((cfg.swarm_size, cfg.dimensions))
        r2 = rng.random((cfg.swarm_size, cfg.dimensions))

        cognitive_term = cfg.cognitive_coeff * r1 * (pbest_positions - positions)
        social_term = cfg.social_coeff * r2 * (gbest_position - positions)
        velocities = w * velocities + cognitive_term + social_term
        velocities = np.clip(velocities, -vmax, vmax)

        positions = positions + velocities
        positions = np.clip(positions, cfg.lower_bound, cfg.upper_bound)

        fitness = rastrigin_population(positions)
        improved = fitness < pbest_fitness
        if np.any(improved):
            pbest_positions[improved] = positions[improved]
            pbest_fitness[improved] = fitness[improved]

        best_idx = int(np.argmin(pbest_fitness))
        best_fit = float(pbest_fitness[best_idx])
        if best_fit < gbest_fitness:
            gbest_fitness = best_fit
            gbest_position = pbest_positions[best_idx].copy()

        history.append(gbest_fitness)
        if (it + 1) % 25 == 0 or it == 0:
            print(
                f"iter={it + 1:3d} | w={w:.3f} | "
                f"global_best={gbest_fitness:.8f}"
            )

        if gbest_fitness <= cfg.target_fitness:
            print(f"Early stop at iter={it + 1}: target_fitness reached.")
            break

    return gbest_position, gbest_fitness, np.asarray(history, dtype=float)


def main() -> None:
    cfg = PSOConfig()
    print("=== PSO MVP on Rastrigin (global optimum = 0 at x=0) ===")
    print(
        "Config: "
        f"dim={cfg.dimensions}, swarm={cfg.swarm_size}, iters={cfg.max_iters}, "
        f"bounds=[{cfg.lower_bound}, {cfg.upper_bound}], seed={cfg.seed}"
    )

    best_position, best_fitness, history = run_pso(cfg)
    monotonic = bool(np.all(history[1:] <= history[:-1] + 1e-15))

    print("\n=== Final Summary ===")
    print(f"best fitness: {best_fitness:.10f}")
    print(f"distance to zero vector: {float(np.linalg.norm(best_position)):.10f}")
    print(f"global-best history monotonic non-increasing: {monotonic}")
    print(f"best position (first 5 dims): {best_position[:5]}")
    print(f"history length: {history.size}")
    print(f"last 10 best-fitness values: {history[-10:]}")


if __name__ == "__main__":
    main()
