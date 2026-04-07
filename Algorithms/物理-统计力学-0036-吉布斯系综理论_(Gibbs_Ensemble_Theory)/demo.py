"""Minimal runnable MVP for Gibbs Ensemble Theory.

This script builds a tiny lattice-gas model and demonstrates three Gibbs ensembles:
- microcanonical ensemble (fixed energy shell),
- canonical ensemble (fixed particle number and temperature),
- grand-canonical ensemble (variable particle number with chemical potential).

For the grand-canonical case, we compare exact probabilities (state enumeration)
against a Metropolis sampler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LatticeGasParams:
    """Parameters for a 1D periodic lattice-gas model."""

    length: int = 6
    epsilon: float = 1.0
    interaction: float = 1.0
    beta: float = 0.7
    mu: float = 0.4


def enumerate_states(length: int) -> np.ndarray:
    """Enumerate all binary occupation states with shape (2^L, L)."""
    if length <= 0:
        raise ValueError("length must be positive")

    state_ids = np.arange(1 << length, dtype=np.uint32)
    bit_positions = np.arange(length, dtype=np.uint32)
    states = ((state_ids[:, None] >> bit_positions[None, :]) & 1).astype(np.int8)
    return states


def particle_numbers(states: np.ndarray) -> np.ndarray:
    return states.sum(axis=1).astype(np.int64)


def energies(states: np.ndarray, params: LatticeGasParams) -> np.ndarray:
    n_particles = particle_numbers(states)
    occupied_neighbor_pairs = np.sum(states * np.roll(states, shift=-1, axis=1), axis=1).astype(np.int64)
    return params.epsilon * n_particles + params.interaction * occupied_neighbor_pairs


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def microcanonical_distribution(
    energy_levels: np.ndarray,
    target_energy: float,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.abs(energy_levels - target_energy) <= tol
    if not np.any(mask):
        raise ValueError("No states found in the requested microcanonical shell")

    probs = np.zeros_like(energy_levels, dtype=np.float64)
    probs[mask] = 1.0 / float(np.sum(mask))
    return probs, mask


def canonical_distribution(
    energy_levels: np.ndarray,
    particle_counts: np.ndarray,
    beta: float,
    fixed_particles: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = particle_counts == fixed_particles
    if not np.any(mask):
        raise ValueError("No states satisfy the requested fixed particle number")

    probs = np.zeros_like(energy_levels, dtype=np.float64)
    probs_support = normalize_log_weights(-beta * energy_levels[mask])
    probs[mask] = probs_support
    return probs, mask


def grand_canonical_distribution(
    energy_levels: np.ndarray,
    particle_counts: np.ndarray,
    beta: float,
    mu: float,
) -> np.ndarray:
    log_weights = -beta * (energy_levels - mu * particle_counts)
    return normalize_log_weights(log_weights)


def expectation(values: np.ndarray, probs: np.ndarray) -> float:
    return float(np.dot(values.astype(np.float64), probs.astype(np.float64)))


def shannon_entropy(probs: np.ndarray) -> float:
    mask = probs > 0.0
    return float(-np.sum(probs[mask] * np.log(probs[mask])))


def state_energy(state: np.ndarray, params: LatticeGasParams) -> float:
    n_particles = int(np.sum(state))
    occupied_neighbor_pairs = int(np.sum(state * np.roll(state, shift=-1)))
    return params.epsilon * n_particles + params.interaction * occupied_neighbor_pairs


def metropolis_grand_canonical(
    params: LatticeGasParams,
    steps: int = 160_000,
    burn_in: int = 20_000,
    thin: int = 20,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample grand-canonical distribution using single-site flip proposals."""
    if steps <= burn_in:
        raise ValueError("steps must be greater than burn_in")
    if thin <= 0:
        raise ValueError("thin must be positive")

    rng = np.random.default_rng(seed)
    state = rng.integers(0, 2, size=params.length).astype(np.int8)

    current_n = int(np.sum(state))
    current_e = float(state_energy(state, params))

    n_states = 1 << params.length
    counts = np.zeros(n_states, dtype=np.int64)
    sampled_energies: list[float] = []
    sampled_particles: list[int] = []
    powers = (1 << np.arange(params.length, dtype=np.int64)).astype(np.int64)

    for step in range(steps):
        idx = int(rng.integers(0, params.length))

        old_val = int(state[idx])
        new_val = 1 - old_val

        left_val = int(state[(idx - 1) % params.length])
        right_val = int(state[(idx + 1) % params.length])

        delta_n = new_val - old_val
        old_pair_count = old_val * left_val + old_val * right_val
        new_pair_count = new_val * left_val + new_val * right_val
        delta_e = params.epsilon * delta_n + params.interaction * (new_pair_count - old_pair_count)

        delta_action = params.beta * (delta_e - params.mu * delta_n)

        if delta_action <= 0.0 or rng.random() < np.exp(-delta_action):
            state[idx] = new_val
            current_n += delta_n
            current_e += float(delta_e)

        if step >= burn_in and ((step - burn_in) % thin == 0):
            code = int(np.dot(state.astype(np.int64), powers))
            counts[code] += 1
            sampled_energies.append(current_e)
            sampled_particles.append(current_n)

    if np.sum(counts) == 0:
        raise RuntimeError("No samples collected; adjust steps/burn_in/thin")

    probs = counts / float(np.sum(counts))
    return probs, np.asarray(sampled_energies, dtype=np.float64), np.asarray(sampled_particles, dtype=np.float64)


def bitstring(state_row: np.ndarray) -> str:
    return "".join(str(int(x)) for x in state_row[::-1])


def choose_microcanonical_energy(energy_levels: np.ndarray) -> tuple[float, int]:
    rounded = np.rint(energy_levels).astype(np.int64)
    values, counts = np.unique(rounded, return_counts=True)
    valid = counts >= 2
    if not np.any(valid):
        raise RuntimeError("Could not find a degenerate energy shell for microcanonical demo")

    valid_values = values[valid]
    valid_counts = counts[valid]
    best_idx = int(np.argmax(valid_counts))
    return float(valid_values[best_idx]), int(valid_counts[best_idx])


def main() -> None:
    params = LatticeGasParams(length=6, epsilon=1.0, interaction=1.0, beta=0.7, mu=0.4)

    states = enumerate_states(params.length)
    particle_counts = particle_numbers(states)
    energy_levels = energies(states, params)

    target_energy, shell_size = choose_microcanonical_energy(energy_levels)
    p_micro, micro_mask = microcanonical_distribution(energy_levels, target_energy)
    p_canonical, canonical_mask = canonical_distribution(
        energy_levels,
        particle_counts,
        beta=params.beta,
        fixed_particles=3,
    )
    p_grand = grand_canonical_distribution(energy_levels, particle_counts, params.beta, params.mu)

    p_grand_mc, sampled_energies, sampled_particles = metropolis_grand_canonical(params)

    e_micro = expectation(energy_levels, p_micro)
    n_micro = expectation(particle_counts, p_micro)

    e_canonical = expectation(energy_levels, p_canonical)
    n_canonical = expectation(particle_counts, p_canonical)

    e_grand_exact = expectation(energy_levels, p_grand)
    n_grand_exact = expectation(particle_counts, p_grand)

    e_grand_mc = expectation(energy_levels, p_grand_mc)
    n_grand_mc = expectation(particle_counts, p_grand_mc)

    total_variation = 0.5 * float(np.sum(np.abs(p_grand_mc - p_grand)))

    summary = pd.DataFrame(
        [
            {
                "ensemble": "microcanonical",
                "constraint": f"E={target_energy:.0f}",
                "mean_E": e_micro,
                "mean_N": n_micro,
                "entropy": shannon_entropy(p_micro),
            },
            {
                "ensemble": "canonical",
                "constraint": "N=3",
                "mean_E": e_canonical,
                "mean_N": n_canonical,
                "entropy": shannon_entropy(p_canonical),
            },
            {
                "ensemble": "grand_canonical_exact",
                "constraint": f"mu={params.mu:.2f}",
                "mean_E": e_grand_exact,
                "mean_N": n_grand_exact,
                "entropy": shannon_entropy(p_grand),
            },
            {
                "ensemble": "grand_canonical_mc",
                "constraint": f"samples={sampled_energies.size}",
                "mean_E": e_grand_mc,
                "mean_N": n_grand_mc,
                "entropy": shannon_entropy(p_grand_mc),
            },
        ]
    )

    top_idx = np.argsort(p_grand)[-6:][::-1]
    top_states = pd.DataFrame(
        {
            "state": [bitstring(states[i]) for i in top_idx],
            "E": energy_levels[top_idx],
            "N": particle_counts[top_idx],
            "p_exact": p_grand[top_idx],
            "p_mc": p_grand_mc[top_idx],
        }
    )

    print("Gibbs Ensemble Theory MVP (1D lattice gas)")
    print(
        f"params: L={params.length}, epsilon={params.epsilon}, interaction={params.interaction}, "
        f"beta={params.beta}, mu={params.mu}"
    )
    print(f"microcanonical_shell: E={target_energy:.0f}, shell_size={shell_size}")
    print(f"canonical_support_size: {int(np.sum(canonical_mask))}")
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print("Top states in grand-canonical ensemble (exact vs MC):")
    print(top_states.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(f"TV_distance(exact, mc)={total_variation:.6f}")
    print(f"MC_sample_mean_E={float(np.mean(sampled_energies)):.6f}")
    print(f"MC_sample_mean_N={float(np.mean(sampled_particles)):.6f}")

    assert np.isclose(np.sum(p_micro), 1.0), "Microcanonical probabilities do not sum to 1"
    assert np.isclose(np.sum(p_canonical), 1.0), "Canonical probabilities do not sum to 1"
    assert np.isclose(np.sum(p_grand), 1.0), "Grand-canonical exact probabilities do not sum to 1"
    assert np.isclose(np.sum(p_grand_mc), 1.0), "Grand-canonical MC probabilities do not sum to 1"

    assert int(np.sum(micro_mask)) >= 2, "Microcanonical shell is not degenerate"
    assert np.all(p_canonical[canonical_mask] > 0.0), "Canonical support must have positive mass"
    assert np.all(p_canonical[~canonical_mask] == 0.0), "Canonical distribution leaked outside fixed-N support"

    assert total_variation < 0.08, f"MC did not match exact distribution well enough: TV={total_variation}"
    assert abs(e_grand_mc - e_grand_exact) < 0.20, "Grand-canonical <E> mismatch too large"
    assert abs(n_grand_mc - n_grand_exact) < 0.12, "Grand-canonical <N> mismatch too large"

    print("All checks passed.")


if __name__ == "__main__":
    main()
