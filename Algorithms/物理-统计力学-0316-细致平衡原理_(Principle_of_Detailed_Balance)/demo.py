"""Minimal MVP for the Principle of Detailed Balance.

This script builds a finite-state Metropolis Markov chain and verifies:
1) theoretical detailed balance: pi_i P_ij = pi_j P_ji;
2) empirical detailed balance from Monte Carlo transitions;
3) stationarity consistency with Boltzmann distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetailedBalanceConfig:
    energies: tuple[float, ...] = (0.0, 0.8, 1.6, 0.3, 1.1, 0.5)
    beta: float = 1.30
    n_steps: int = 300000
    burn_in: int = 5000
    seed: int = 20260407


@dataclass(frozen=True)
class SimulationResult:
    transition_counts: np.ndarray
    empirical_distribution: np.ndarray
    acceptance_rate: float


def boltzmann_distribution(energies: np.ndarray, beta: float) -> np.ndarray:
    """Compute normalized Boltzmann weights pi_i ∝ exp(-beta * E_i)."""
    shifted = energies - np.min(energies)
    weights = np.exp(-beta * shifted)
    return weights / np.sum(weights)


def ring_neighbors(n_states: int) -> list[tuple[int, int]]:
    """Two symmetric proposal neighbors for each state on a ring graph."""
    return [((i - 1) % n_states, (i + 1) % n_states) for i in range(n_states)]


def build_metropolis_kernel(energies: np.ndarray, beta: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build transition matrix P for symmetric proposal + Metropolis acceptance.

    Proposal q(i->j) is 1/2 for each ring neighbor and 0 otherwise.
    Acceptance a(i->j) = min(1, exp(-beta * (E_j - E_i))).
    """
    n_states = energies.size
    neighbors = ring_neighbors(n_states)
    p = np.zeros((n_states, n_states), dtype=float)

    for i in range(n_states):
        for j in neighbors[i]:
            proposal_prob = 0.5
            delta_e = energies[j] - energies[i]
            accept_prob = min(1.0, float(np.exp(-beta * delta_e)))
            p[i, j] += proposal_prob * accept_prob

        p[i, i] = 1.0 - np.sum(p[i])

    return p, neighbors


def detailed_balance_residual(pi: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    """Return max absolute residual and full residual matrix of detailed balance."""
    flux_diff = np.abs(pi[:, None] * p - pi[None, :] * p.T)
    return float(np.max(flux_diff)), flux_diff


def simulate_chain(
    *,
    energies: np.ndarray,
    beta: float,
    n_steps: int,
    burn_in: int,
    seed: int,
) -> SimulationResult:
    """Simulate one Metropolis chain and collect transition statistics."""
    n_states = energies.size
    neighbors = ring_neighbors(n_states)
    rng = np.random.default_rng(seed)

    state = 0
    accepted = 0
    transition_counts = np.zeros((n_states, n_states), dtype=np.int64)
    visit_counts = np.zeros(n_states, dtype=np.int64)

    for t in range(n_steps):
        old_state = state
        left, right = neighbors[old_state]
        proposal = left if rng.random() < 0.5 else right

        delta_e = energies[proposal] - energies[old_state]
        accept_prob = min(1.0, float(np.exp(-beta * delta_e)))

        if rng.random() < accept_prob:
            state = proposal
            accepted += 1

        transition_counts[old_state, state] += 1

        if t >= burn_in:
            visit_counts[state] += 1

    empirical_distribution = visit_counts / np.sum(visit_counts)
    acceptance_rate = accepted / n_steps
    return SimulationResult(
        transition_counts=transition_counts,
        empirical_distribution=empirical_distribution,
        acceptance_rate=acceptance_rate,
    )


def row_normalize_counts(counts: np.ndarray) -> np.ndarray:
    """Convert transition counts to row-stochastic empirical transition matrix."""
    row_sum = counts.sum(axis=1, keepdims=True)
    p_hat = np.zeros_like(counts, dtype=float)
    np.divide(counts, row_sum, out=p_hat, where=row_sum > 0)
    return p_hat


def run_detailed_balance_demo(config: DetailedBalanceConfig) -> dict[str, object]:
    """Run the full experiment and return printable artifacts."""
    energies = np.asarray(config.energies, dtype=float)
    if energies.ndim != 1 or energies.size < 3:
        raise ValueError("energies must be a 1D array with at least 3 states.")
    if config.beta <= 0.0:
        raise ValueError("beta must be positive.")
    if config.n_steps <= config.burn_in + 100:
        raise ValueError("n_steps must be sufficiently larger than burn_in.")

    pi = boltzmann_distribution(energies, config.beta)
    p, _ = build_metropolis_kernel(energies, config.beta)

    max_db_theory, db_matrix_theory = detailed_balance_residual(pi, p)
    stationarity_max_abs = float(np.max(np.abs(pi @ p - pi)))

    sim = simulate_chain(
        energies=energies,
        beta=config.beta,
        n_steps=config.n_steps,
        burn_in=config.burn_in,
        seed=config.seed,
    )
    p_hat = row_normalize_counts(sim.transition_counts)

    max_db_emp, db_matrix_emp = detailed_balance_residual(sim.empirical_distribution, p_hat)
    empirical_vs_theory_l1 = float(np.sum(np.abs(sim.empirical_distribution - pi)))

    summary = pd.DataFrame(
        {
            "metric": [
                "n_states",
                "beta",
                "acceptance_rate",
                "max_db_residual_theory",
                "max_db_residual_empirical",
                "stationarity_max_abs_theory",
                "L1(empirical_pi, theory_pi)",
            ],
            "value": [
                float(energies.size),
                config.beta,
                sim.acceptance_rate,
                max_db_theory,
                max_db_emp,
                stationarity_max_abs,
                empirical_vs_theory_l1,
            ],
        }
    )

    state_df = pd.DataFrame(
        {
            "state": np.arange(energies.size),
            "energy": energies,
            "pi_theory": pi,
            "pi_empirical": sim.empirical_distribution,
            "abs_diff": np.abs(sim.empirical_distribution - pi),
        }
    )

    db_pairs = []
    for i in range(energies.size):
        for j in range(i + 1, energies.size):
            if p[i, j] > 0.0 or p[j, i] > 0.0:
                db_pairs.append(
                    {
                        "i": i,
                        "j": j,
                        "theory_flux_ij": pi[i] * p[i, j],
                        "theory_flux_ji": pi[j] * p[j, i],
                        "theory_abs_gap": db_matrix_theory[i, j],
                        "emp_flux_ij": sim.empirical_distribution[i] * p_hat[i, j],
                        "emp_flux_ji": sim.empirical_distribution[j] * p_hat[j, i],
                        "emp_abs_gap": db_matrix_emp[i, j],
                    }
                )
    db_df = pd.DataFrame(db_pairs)

    checks = {
        "max_db_residual_theory": max_db_theory,
        "max_db_residual_empirical": max_db_emp,
        "stationarity_max_abs_theory": stationarity_max_abs,
        "l1_empirical_theory": empirical_vs_theory_l1,
        "acceptance_rate": sim.acceptance_rate,
    }

    return {
        "config": config,
        "summary": summary,
        "state_df": state_df,
        "db_df": db_df,
        "checks": checks,
    }


def main() -> None:
    config = DetailedBalanceConfig()
    result = run_detailed_balance_demo(config)

    summary = result["summary"]
    state_df = result["state_df"]
    db_df = result["db_df"]
    checks = result["checks"]

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 140)

    print("Principle of Detailed Balance: Metropolis chain MVP")
    print("=" * 78)
    print(config)
    print("\n--- Summary ---")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.8f}"))

    print("\n--- State probabilities (theory vs empirical) ---")
    print(state_df.to_string(index=False, float_format=lambda v: f"{v:.8f}"))

    print("\n--- Pairwise flux check (non-zero edges) ---")
    print(db_df.to_string(index=False, float_format=lambda v: f"{v:.8f}"))

    assert checks["max_db_residual_theory"] < 1e-12, (
        f"Theoretical detailed-balance residual too large: {checks['max_db_residual_theory']:.3e}"
    )
    assert checks["stationarity_max_abs_theory"] < 1e-12, (
        f"Theoretical stationarity residual too large: {checks['stationarity_max_abs_theory']:.3e}"
    )
    assert checks["max_db_residual_empirical"] < 1.2e-2, (
        f"Empirical detailed-balance residual too large: {checks['max_db_residual_empirical']:.3e}"
    )
    assert checks["l1_empirical_theory"] < 3.0e-2, (
        f"Empirical distribution mismatch too large: {checks['l1_empirical_theory']:.3e}"
    )
    assert 0.1 < checks["acceptance_rate"] < 0.95, (
        f"Acceptance rate out of expected range: {checks['acceptance_rate']:.3f}"
    )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
