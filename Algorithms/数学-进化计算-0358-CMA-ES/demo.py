"""Minimal runnable MVP for CMA-ES (Covariance Matrix Adaptation ES)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass
class CMAESConfig:
    dimension: int
    init_mean: Array
    init_sigma: float = 0.8
    population_size: int | None = None
    seed: int = 42


class CMAES:
    """A compact, from-scratch CMA-ES implementation for unconstrained minimization."""

    def __init__(self, config: CMAESConfig) -> None:
        self.n = config.dimension
        self.rng = np.random.default_rng(config.seed)

        self.mean = np.asarray(config.init_mean, dtype=float).copy()
        if self.mean.shape != (self.n,):
            raise ValueError(f"init_mean shape must be ({self.n},), got {self.mean.shape}")
        if config.init_sigma <= 0:
            raise ValueError("init_sigma must be positive")
        self.sigma = float(config.init_sigma)

        self.lam = config.population_size or (4 + int(3 * np.log(self.n)))
        self.mu = self.lam // 2

        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mueff = (np.sum(self.weights) ** 2) / np.sum(self.weights**2)

        self.cs = (self.mueff + 2.0) / (self.n + self.mueff + 5.0)
        self.ds = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (self.n + 1.0)) - 1.0) + self.cs
        self.cc = (4.0 + self.mueff / self.n) / (self.n + 4.0 + 2.0 * self.mueff / self.n)
        self.c1 = 2.0 / ((self.n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((self.n + 2.0) ** 2 + self.mueff),
        )
        self.chi_n = np.sqrt(self.n) * (1.0 - 1.0 / (4.0 * self.n) + 1.0 / (21.0 * self.n * self.n))

        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.C = np.eye(self.n)
        self.B = np.eye(self.n)
        self.D = np.ones(self.n)
        self.invsqrtC = np.eye(self.n)

        self.generation = 0
        self.eigeneval = 0
        self.counteval = 0
        self.eigen_update_interval = max(1, int(self.lam / (self.c1 + self.cmu) / self.n / 10.0))

    def ask(self) -> tuple[Array, Array]:
        """Sample a population.

        Returns:
            x: Candidate solutions in search space, shape (lambda, n).
            y: Standardized displacements, shape (lambda, n), where x = mean + sigma * y.
        """
        z = self.rng.standard_normal((self.lam, self.n))
        y = (z * self.D) @ self.B.T
        x = self.mean + self.sigma * y
        return x, y

    def tell(self, x: Array, y: Array, fitness: Array) -> None:
        """Update distribution parameters from ranked population."""
        order = np.argsort(fitness)
        x_sel = x[order[: self.mu]]
        y_sel = y[order[: self.mu]]

        old_mean = self.mean.copy()
        self.mean = self.weights @ x_sel
        delta_mean = (self.mean - old_mean) / self.sigma

        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * (
            self.invsqrtC @ delta_mean
        )
        norm_ps = np.linalg.norm(self.ps)

        hsig_lhs = norm_ps / np.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * (self.generation + 1.0))) / self.chi_n
        hsig = 1.0 if hsig_lhs < (1.4 + 2.0 / (self.n + 1.0)) else 0.0

        self.pc = (1.0 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * delta_mean

        rank_mu = np.zeros((self.n, self.n))
        for i, w in enumerate(self.weights):
            rank_mu += w * np.outer(y_sel[i], y_sel[i])

        c_factor = 1.0 - self.c1 - self.cmu + (1.0 - hsig) * self.c1 * self.cc * (2.0 - self.cc)
        self.C = c_factor * self.C + self.c1 * np.outer(self.pc, self.pc) + self.cmu * rank_mu
        self.C = (self.C + self.C.T) / 2.0

        self.sigma *= np.exp((self.cs / self.ds) * (norm_ps / self.chi_n - 1.0))
        self.sigma = float(np.clip(self.sigma, 1e-12, 1e5))

        self.generation += 1
        self.counteval += x.shape[0]
        if self.counteval - self.eigeneval >= self.eigen_update_interval:
            self._update_eigensystem()
            self.eigeneval = self.counteval

    def _update_eigensystem(self) -> None:
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.clip(eigvals, 1e-30, None)
        self.D = np.sqrt(eigvals)
        self.B = eigvecs
        self.invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T


def rosenbrock(x: Array) -> float:
    """Classic non-convex benchmark with global minimum f(1,...,1)=0."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def run_optimization(objective: Callable[[Array], float], dim: int = 8, max_iter: int = 350) -> dict:
    rng = np.random.default_rng(7)
    init_mean = rng.uniform(-3.0, 3.0, size=dim)

    optimizer = CMAES(
        CMAESConfig(
            dimension=dim,
            init_mean=init_mean,
            init_sigma=0.9,
            seed=123,
        )
    )

    best_x = init_mean.copy()
    best_f = objective(best_x)
    history_best: list[float] = []

    for gen in range(1, max_iter + 1):
        x, y = optimizer.ask()
        fitness = np.array([objective(ind) for ind in x])
        optimizer.tell(x, y, fitness)

        i = int(np.argmin(fitness))
        if fitness[i] < best_f:
            best_f = float(fitness[i])
            best_x = x[i].copy()

        history_best.append(best_f)
        if gen % 25 == 0 or gen == 1:
            print(
                f"[gen={gen:03d}] best_f={best_f:.6e} "
                f"sigma={optimizer.sigma:.3e} "
                f"||mean-1||={np.linalg.norm(optimizer.mean - 1.0):.3e}"
            )

        if best_f < 1e-12:
            break

    return {
        "generations": gen,
        "best_f": best_f,
        "best_x": best_x,
        "history_best": np.array(history_best),
        "final_sigma": optimizer.sigma,
    }


def main() -> None:
    result = run_optimization(rosenbrock, dim=8, max_iter=350)
    print("\n=== CMA-ES run summary ===")
    print(f"generations: {result['generations']}")
    print(f"best_f:      {result['best_f']:.6e}")
    print(f"final_sigma: {result['final_sigma']:.3e}")
    print(
        "best_x[:4]:  "
        + np.array2string(
            result["best_x"][:4],
            precision=4,
            suppress_small=True,
        )
    )


if __name__ == "__main__":
    main()
