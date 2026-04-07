"""Minimal runnable MVP for phase-field method (MATH-0166).

This demo implements a 2D Allen-Cahn phase-field evolution with periodic
boundary conditions using finite differences and explicit Euler time stepping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List

import numpy as np


@dataclass
class SimulationConfig:
    """Configuration for a small, deterministic phase-field simulation."""

    nx: int = 64
    ny: int = 64
    dx: float = 1.0
    epsilon: float = 1.5
    dt: float = 0.03
    n_steps: int = 400
    report_every: int = 40
    nucleus_radius_ratio: float = 0.22
    noise_amplitude: float = 0.03
    seed: int = 42

    def validate(self) -> None:
        if self.nx <= 4 or self.ny <= 4:
            raise ValueError("nx and ny must be > 4")
        if not math.isfinite(self.dx) or self.dx <= 0.0:
            raise ValueError("dx must be finite and positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.report_every <= 0:
            raise ValueError("report_every must be positive")
        if not (0.01 <= self.nucleus_radius_ratio <= 0.45):
            raise ValueError("nucleus_radius_ratio must be in [0.01, 0.45]")
        if self.noise_amplitude < 0.0:
            raise ValueError("noise_amplitude must be non-negative")


@dataclass
class SimulationResult:
    """Outputs logged from the simulation."""

    phi: np.ndarray
    steps: np.ndarray
    energies: np.ndarray
    positive_fractions: np.ndarray


def build_initial_field(cfg: SimulationConfig) -> np.ndarray:
    """Create a circular nucleus (+1) in a matrix (-1), then add mild noise."""
    rng = np.random.default_rng(cfg.seed)
    y, x = np.mgrid[0 : cfg.ny, 0 : cfg.nx]

    cx = 0.5 * (cfg.nx - 1)
    cy = 0.5 * (cfg.ny - 1)
    radius = cfg.nucleus_radius_ratio * float(min(cfg.nx, cfg.ny))

    dist2 = (x - cx) ** 2 + (y - cy) ** 2
    phi = np.where(dist2 <= radius * radius, 1.0, -1.0).astype(float)

    if cfg.noise_amplitude > 0.0:
        phi += cfg.noise_amplitude * rng.standard_normal(size=phi.shape)

    return phi


def laplacian_periodic(phi: np.ndarray, dx: float) -> np.ndarray:
    """2D five-point Laplacian with periodic boundaries."""
    return (
        np.roll(phi, 1, axis=0)
        + np.roll(phi, -1, axis=0)
        + np.roll(phi, 1, axis=1)
        + np.roll(phi, -1, axis=1)
        - 4.0 * phi
    ) / (dx * dx)


def allen_cahn_rhs(phi: np.ndarray, epsilon: float, dx: float) -> np.ndarray:
    """RHS of Allen-Cahn: phi_t = eps^2*Lap(phi) - (phi^3-phi)/eps^2."""
    lap = laplacian_periodic(phi, dx)
    reaction = (phi * phi * phi - phi) / (epsilon * epsilon)
    return (epsilon * epsilon) * lap - reaction


def free_energy(phi: np.ndarray, epsilon: float, dx: float) -> float:
    """Discrete Ginzburg-Landau energy functional."""
    grad_x = (np.roll(phi, -1, axis=1) - phi) / dx
    grad_y = (np.roll(phi, -1, axis=0) - phi) / dx

    interfacial = 0.5 * (epsilon * epsilon) * (grad_x * grad_x + grad_y * grad_y)
    bulk = ((phi * phi - 1.0) ** 2) / (4.0 * epsilon * epsilon)

    return float(np.sum(interfacial + bulk) * dx * dx)


def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    """Run explicit Euler evolution for the Allen-Cahn phase-field equation."""
    cfg.validate()

    phi = build_initial_field(cfg)

    step_log: List[int] = []
    energy_log: List[float] = []
    positive_fraction_log: List[float] = []

    for step in range(cfg.n_steps + 1):
        should_report = (step % cfg.report_every == 0) or (step == cfg.n_steps)
        if should_report:
            step_log.append(step)
            energy_log.append(free_energy(phi, cfg.epsilon, cfg.dx))
            positive_fraction_log.append(float(np.mean(phi > 0.0)))

        if step == cfg.n_steps:
            break

        phi = phi + cfg.dt * allen_cahn_rhs(phi, cfg.epsilon, cfg.dx)

        if not np.all(np.isfinite(phi)):
            raise RuntimeError("non-finite field value encountered; reduce dt")

    return SimulationResult(
        phi=phi,
        steps=np.array(step_log, dtype=int),
        energies=np.array(energy_log, dtype=float),
        positive_fractions=np.array(positive_fraction_log, dtype=float),
    )


def print_report(result: SimulationResult) -> None:
    """Print compact diagnostics for non-interactive validation."""
    print("step    energy            positive_fraction")
    for s, e, p in zip(result.steps, result.energies, result.positive_fractions):
        print(f"{s:4d}    {e:14.6f}    {p:0.6f}")

    phi = result.phi
    print("\nfinal_field_stats")
    print(f"  min={float(np.min(phi)):.6f}")
    print(f"  max={float(np.max(phi)):.6f}")
    print(f"  mean={float(np.mean(phi)):.6f}")


def run_checks(result: SimulationResult) -> None:
    """Basic self-checks to ensure the MVP behavior is sensible."""
    if result.energies.size < 2:
        raise AssertionError("not enough logged energy samples")

    energy_drop = result.energies[0] - result.energies[-1]
    assert energy_drop > 0.0, "free energy should decrease overall"

    # For this stable configuration, sampled energies should be non-increasing.
    energy_diffs = np.diff(result.energies)
    assert np.all(energy_diffs <= 1e-9), "energy increased between report points"

    assert np.all(np.isfinite(result.phi)), "final field contains non-finite values"
    assert float(np.max(np.abs(result.phi))) < 2.5, "field magnitude exploded"


def main() -> None:
    print("Phase-Field Method MVP (MATH-0166)")
    print("Model: 2D Allen-Cahn equation with periodic finite differences")
    print("=" * 72)

    cfg = SimulationConfig()
    result = run_simulation(cfg)
    print_report(result)
    run_checks(result)

    print("=" * 72)
    print("All checks passed.")


if __name__ == "__main__":
    main()
