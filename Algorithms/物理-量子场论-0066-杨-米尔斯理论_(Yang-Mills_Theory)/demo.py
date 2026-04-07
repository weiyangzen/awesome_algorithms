"""Minimal runnable MVP for Yang-Mills theory on a 2D SU(2) lattice.

The script demonstrates three core pieces of Yang-Mills numerics:
1) Wilson plaquette action,
2) Metropolis evolution of link variables,
3) Gauge-invariance check of an observable (average plaquette).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def su2_from_quaternion(q: np.ndarray) -> np.ndarray:
    """Map a normalized quaternion (a0, a1, a2, a3) to an SU(2) matrix."""
    a0, a1, a2, a3 = q
    return np.array(
        [
            [a0 + 1j * a3, a2 + 1j * a1],
            [-a2 + 1j * a1, a0 - 1j * a3],
        ],
        dtype=np.complex128,
    )


def random_su2(rng: np.random.Generator) -> np.ndarray:
    """Draw a Haar-random SU(2) element by normalizing a 4D Gaussian."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return su2_from_quaternion(q)


def small_random_su2(rng: np.random.Generator, epsilon: float) -> np.ndarray:
    """Small random group element near identity for Metropolis proposals."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.normal(scale=epsilon)
    a0 = np.cos(angle)
    avec = np.sin(angle) * axis
    q = np.array([a0, avec[0], avec[1], avec[2]], dtype=np.float64)
    return su2_from_quaternion(q)


def dagger(u: np.ndarray) -> np.ndarray:
    return np.conjugate(u.T)


def real_trace(u: np.ndarray) -> float:
    return float(np.real(np.trace(u)))


def initialize_links(lattice_size: int, rng: np.random.Generator) -> np.ndarray:
    """Create links[mu, x, y] with mu=0(x-direction),1(y-direction)."""
    links = np.empty((2, lattice_size, lattice_size, 2, 2), dtype=np.complex128)
    for mu in range(2):
        for x in range(lattice_size):
            for y in range(lattice_size):
                links[mu, x, y] = random_su2(rng)
    return links


def plaquette_xy(links: np.ndarray, x: int, y: int) -> np.ndarray:
    """Oriented plaquette in the x-y plane with lower-left corner (x, y)."""
    lattice_size = links.shape[1]
    xp = (x + 1) % lattice_size
    yp = (y + 1) % lattice_size
    return (
        links[0, x, y]
        @ links[1, xp, y]
        @ dagger(links[0, x, yp])
        @ dagger(links[1, x, y])
    )


def average_plaquette(links: np.ndarray) -> float:
    lattice_size = links.shape[1]
    accum = 0.0
    for x in range(lattice_size):
        for y in range(lattice_size):
            accum += 0.5 * real_trace(plaquette_xy(links, x, y))
    return accum / (lattice_size * lattice_size)


def local_action_contribution(
    links: np.ndarray,
    beta: float,
    mu: int,
    x: int,
    y: int,
) -> float:
    """Action contribution of the two plaquettes touching one link in 2D."""
    lattice_size = links.shape[1]
    if mu == 0:
        touched = [(x, y), (x, (y - 1) % lattice_size)]
    else:
        touched = [(x, y), ((x - 1) % lattice_size, y)]

    local_action = 0.0
    for px, py in touched:
        p = plaquette_xy(links, px, py)
        local_action += beta * (1.0 - 0.5 * real_trace(p))
    return local_action


@dataclass
class SweepRecord:
    sweep: int
    phase: str
    accept_rate: float
    avg_plaquette: float
    action_density: float


def metropolis_sweep(
    links: np.ndarray,
    beta: float,
    epsilon: float,
    rng: np.random.Generator,
) -> float:
    """One full-lattice Metropolis sweep. Returns acceptance rate."""
    lattice_size = links.shape[1]
    attempts = 0
    accepted = 0

    for mu in range(2):
        for x in range(lattice_size):
            for y in range(lattice_size):
                attempts += 1
                old_link = links[mu, x, y].copy()
                old_local = local_action_contribution(links, beta, mu, x, y)

                proposal = small_random_su2(rng, epsilon) @ old_link
                links[mu, x, y] = proposal
                new_local = local_action_contribution(links, beta, mu, x, y)
                delta_s = new_local - old_local

                if delta_s <= 0.0 or rng.random() < np.exp(-min(delta_s, 700.0)):
                    accepted += 1
                else:
                    links[mu, x, y] = old_link

    return accepted / attempts


def random_gauge_transformation(
    lattice_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    g = np.empty((lattice_size, lattice_size, 2, 2), dtype=np.complex128)
    for x in range(lattice_size):
        for y in range(lattice_size):
            g[x, y] = random_su2(rng)
    return g


def gauge_transform_links(links: np.ndarray, g: np.ndarray) -> np.ndarray:
    """U'_mu(x) = G(x) U_mu(x) G^dagger(x+mu)."""
    lattice_size = links.shape[1]
    transformed = np.empty_like(links)
    for x in range(lattice_size):
        for y in range(lattice_size):
            xp = (x + 1) % lattice_size
            yp = (y + 1) % lattice_size
            transformed[0, x, y] = g[x, y] @ links[0, x, y] @ dagger(g[xp, y])
            transformed[1, x, y] = g[x, y] @ links[1, x, y] @ dagger(g[x, yp])
    return transformed


def run_simulation(
    lattice_size: int,
    beta: float,
    epsilon: float,
    thermal_sweeps: int,
    production_sweeps: int,
    measure_every: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
    links = initialize_links(lattice_size, rng)
    records: list[SweepRecord] = []

    for sweep in range(1, thermal_sweeps + 1):
        accept_rate = metropolis_sweep(links, beta, epsilon, rng)
        if sweep % measure_every == 0 or sweep == thermal_sweeps:
            p_avg = average_plaquette(links)
            records.append(
                SweepRecord(
                    sweep=sweep,
                    phase="thermal",
                    accept_rate=accept_rate,
                    avg_plaquette=p_avg,
                    action_density=1.0 - p_avg,
                )
            )

    for sweep in range(1, production_sweeps + 1):
        accept_rate = metropolis_sweep(links, beta, epsilon, rng)
        p_avg = average_plaquette(links)
        records.append(
            SweepRecord(
                sweep=sweep,
                phase="production",
                accept_rate=accept_rate,
                avg_plaquette=p_avg,
                action_density=1.0 - p_avg,
            )
        )

    frame = pd.DataFrame([r.__dict__ for r in records])
    return links, frame


def main() -> None:
    seed = 20260407
    lattice_size = 8
    beta = 2.20
    epsilon = 0.30
    thermal_sweeps = 30
    production_sweeps = 20
    measure_every = 5

    rng = np.random.default_rng(seed)
    links, history = run_simulation(
        lattice_size=lattice_size,
        beta=beta,
        epsilon=epsilon,
        thermal_sweeps=thermal_sweeps,
        production_sweeps=production_sweeps,
        measure_every=measure_every,
        rng=rng,
    )

    production = history[history["phase"] == "production"].reset_index(drop=True)
    mean_plaquette = float(production["avg_plaquette"].mean())
    std_plaquette = float(production["avg_plaquette"].std(ddof=1))
    mean_accept = float(production["accept_rate"].mean())

    plaquette_before = average_plaquette(links)
    gauge_field = random_gauge_transformation(lattice_size, rng)
    transformed_links = gauge_transform_links(links, gauge_field)
    plaquette_after = average_plaquette(transformed_links)
    gauge_invariance_error = abs(plaquette_before - plaquette_after)

    print("=== Yang-Mills Lattice MVP (SU(2), 2D) ===")
    print(
        f"seed={seed}, L={lattice_size}, beta={beta:.2f}, "
        f"epsilon={epsilon:.2f}, thermal={thermal_sweeps}, production={production_sweeps}"
    )
    print("\nRecent measurements:")
    print(history.tail(12).to_string(index=False, float_format=lambda x: f"{x:0.6f}"))

    summary = pd.DataFrame(
        {
            "metric": [
                "mean_plaquette(prod)",
                "std_plaquette(prod)",
                "mean_acceptance(prod)",
                "gauge_invariance_error",
            ],
            "value": [
                mean_plaquette,
                std_plaquette,
                mean_accept,
                gauge_invariance_error,
            ],
        }
    )
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:0.8f}"))

    assert 0.0 < mean_plaquette < 1.0, "Average plaquette must stay in (0, 1)."
    assert 0.05 < mean_accept < 0.95, "Acceptance rate is out of a healthy range."
    assert gauge_invariance_error < 1e-10, "Gauge invariance check failed."
    assert np.isfinite(std_plaquette) and std_plaquette > 0.0, "Invalid fluctuation estimate."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
