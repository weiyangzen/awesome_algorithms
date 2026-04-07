"""Detailed Balance MVP in statistical mechanics.

This script builds a finite-state Ising Metropolis Markov chain and verifies:
1) theoretical detailed balance with Boltzmann distribution;
2) theoretical stationarity consistency;
3) empirical detailed-balance residual from Monte Carlo transitions.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chisquare


@dataclass(frozen=True)
class DetailedBalanceConfig:
    n_spins: int = 4
    beta: float = 0.90
    coupling_j: float = 1.0
    field_h: float = 0.25
    n_steps: int = 250000
    burn_in: int = 5000
    seed: int = 20260407


@dataclass(frozen=True)
class SimulationResult:
    transition_counts: np.ndarray
    empirical_distribution: np.ndarray
    acceptance_rate: float
    visit_counts: np.ndarray


def enumerate_ising_states(n_spins: int) -> np.ndarray:
    """Enumerate all Ising states in {-1, +1}^n via bit representation."""
    if n_spins <= 0:
        raise ValueError("n_spins must be positive.")

    n_states = 1 << n_spins
    ids = np.arange(n_states, dtype=np.int64)
    bits = (ids[:, None] >> np.arange(n_spins, dtype=np.int64)) & 1
    spins = 2 * bits - 1
    return spins.astype(np.int8)


def ising_energies(states: np.ndarray, coupling_j: float, field_h: float) -> np.ndarray:
    """Compute 1D periodic Ising energies.

    E(s) = -J * sum_i s_i s_{i+1} - h * sum_i s_i
    """
    if states.ndim != 2:
        raise ValueError("states must be a 2D array of shape (n_states, n_spins).")

    pair_term = np.sum(states * np.roll(states, -1, axis=1), axis=1)
    field_term = np.sum(states, axis=1)
    energies = -coupling_j * pair_term - field_h * field_term
    return energies.astype(float)


def boltzmann_distribution(energies: np.ndarray, beta: float) -> np.ndarray:
    """Compute normalized Boltzmann probabilities pi_i ∝ exp(-beta E_i)."""
    if beta <= 0.0:
        raise ValueError("beta must be positive.")

    shifted = energies - np.min(energies)
    weights = np.exp(-beta * shifted)
    return weights / np.sum(weights)


def build_metropolis_kernel(
    energies: np.ndarray,
    beta: float,
    n_spins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build transition matrix for single-spin-flip Metropolis dynamics.

    Proposal: choose one spin uniformly, then flip it.
    q(i->j) = 1/n_spins if j differs by one bit from i.
    Acceptance: a(i->j) = min(1, exp(-beta (E_j - E_i))).
    """
    n_states = energies.size
    p = np.zeros((n_states, n_states), dtype=float)
    next_state = np.zeros((n_states, n_spins), dtype=np.int64)
    accept_prob = np.zeros((n_states, n_spins), dtype=float)

    for i in range(n_states):
        for k in range(n_spins):
            j = i ^ (1 << k)
            next_state[i, k] = j
            delta_e = energies[j] - energies[i]
            a = min(1.0, float(np.exp(-beta * delta_e)))
            accept_prob[i, k] = a
            p[i, j] += (1.0 / n_spins) * a

        p[i, i] = 1.0 - np.sum(p[i])

    return p, next_state, accept_prob


def detailed_balance_residual(pi: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    """Return max |pi_i P_ij - pi_j P_ji| and full residual matrix."""
    flux = pi[:, None] * p
    residual = np.abs(flux - flux.T)
    return float(np.max(residual)), residual


def row_normalize_counts(counts: np.ndarray) -> np.ndarray:
    """Convert transition count matrix into row-stochastic matrix."""
    row_sum = counts.sum(axis=1, keepdims=True)
    p_hat = np.zeros_like(counts, dtype=float)
    np.divide(counts, row_sum, out=p_hat, where=row_sum > 0)
    return p_hat


def simulate_chain(
    *,
    n_states: int,
    n_spins: int,
    next_state: np.ndarray,
    accept_prob: np.ndarray,
    n_steps: int,
    burn_in: int,
    seed: int,
) -> SimulationResult:
    """Run a Metropolis chain and collect transition/visit statistics."""
    if n_steps <= burn_in + 100:
        raise ValueError("n_steps must be significantly larger than burn_in.")

    rng = np.random.default_rng(seed)
    current = 0

    accepted = 0
    transition_counts = np.zeros((n_states, n_states), dtype=np.int64)
    visit_counts = np.zeros(n_states, dtype=np.int64)

    for t in range(n_steps):
        old = current
        k = int(rng.integers(0, n_spins))
        candidate = int(next_state[old, k])

        if rng.random() < accept_prob[old, k]:
            current = candidate
            accepted += 1

        transition_counts[old, current] += 1

        if t >= burn_in:
            visit_counts[current] += 1

    empirical_distribution = visit_counts / np.sum(visit_counts)
    acceptance_rate = accepted / n_steps
    return SimulationResult(
        transition_counts=transition_counts,
        empirical_distribution=empirical_distribution,
        acceptance_rate=acceptance_rate,
        visit_counts=visit_counts,
    )


def spin_string(state: np.ndarray) -> str:
    """Compact spin representation, e.g. [+1 -1 +1 -1]."""
    return "[" + " ".join(f"{int(v):+d}" for v in state) + "]"


def build_edge_flux_table(
    *,
    energies: np.ndarray,
    beta: float,
    p: np.ndarray,
    p_hat: np.ndarray,
    pi: np.ndarray,
    pi_hat: np.ndarray,
    n_spins: int,
) -> pd.DataFrame:
    """Build pairwise flux table on single-spin-flip edges."""
    n_states = energies.size
    rows: list[dict[str, float | int]] = []

    for i in range(n_states):
        for k in range(n_spins):
            j = i ^ (1 << k)
            if i < j:
                delta_e = float(energies[j] - energies[i])
                ratio_expected = float(np.exp(-beta * delta_e))
                ratio_transition = float(p[i, j] / p[j, i])

                rows.append(
                    {
                        "i": i,
                        "j": j,
                        "delta_E": delta_e,
                        "P_ij": float(p[i, j]),
                        "P_ji": float(p[j, i]),
                        "P_ratio": ratio_transition,
                        "exp(-beta*deltaE)": ratio_expected,
                        "theory_flux_ij": float(pi[i] * p[i, j]),
                        "theory_flux_ji": float(pi[j] * p[j, i]),
                        "theory_abs_gap": float(abs(pi[i] * p[i, j] - pi[j] * p[j, i])),
                        "emp_flux_ij": float(pi_hat[i] * p_hat[i, j]),
                        "emp_flux_ji": float(pi_hat[j] * p_hat[j, i]),
                        "emp_abs_gap": float(abs(pi_hat[i] * p_hat[i, j] - pi_hat[j] * p_hat[j, i])),
                    }
                )

    df = pd.DataFrame(rows)
    df = df.sort_values(["i", "j"]).reset_index(drop=True)
    return df


def run_detailed_balance_demo(config: DetailedBalanceConfig) -> dict[str, object]:
    """Run full detailed-balance checks and return printable artifacts."""
    if config.n_spins < 2:
        raise ValueError("n_spins must be at least 2.")
    if config.beta <= 0.0:
        raise ValueError("beta must be positive.")

    states = enumerate_ising_states(config.n_spins)
    energies = ising_energies(states, config.coupling_j, config.field_h)
    pi = boltzmann_distribution(energies, config.beta)

    p, next_state, accept_prob = build_metropolis_kernel(
        energies=energies,
        beta=config.beta,
        n_spins=config.n_spins,
    )

    row_sum_max_err = float(np.max(np.abs(np.sum(p, axis=1) - 1.0)))
    max_db_theory, db_mat_theory = detailed_balance_residual(pi, p)
    stationarity_residual = float(np.max(np.abs(pi @ p - pi)))

    sim = simulate_chain(
        n_states=states.shape[0],
        n_spins=config.n_spins,
        next_state=next_state,
        accept_prob=accept_prob,
        n_steps=config.n_steps,
        burn_in=config.burn_in,
        seed=config.seed,
    )

    p_hat = row_normalize_counts(sim.transition_counts)
    pi_hat = sim.empirical_distribution

    max_db_emp, db_mat_emp = detailed_balance_residual(pi_hat, p_hat)
    l1_emp_theory = float(np.sum(np.abs(pi_hat - pi)))

    observed = sim.visit_counts.astype(float)
    expected = np.sum(sim.visit_counts) * pi
    chi2_stat, chi2_pvalue = chisquare(f_obs=observed, f_exp=expected)

    edge_df = build_edge_flux_table(
        energies=energies,
        beta=config.beta,
        p=p,
        p_hat=p_hat,
        pi=pi,
        pi_hat=pi_hat,
        n_spins=config.n_spins,
    )

    ratio_rel_err = np.abs(edge_df["P_ratio"] - edge_df["exp(-beta*deltaE)"]) / np.maximum(
        1.0, np.abs(edge_df["exp(-beta*deltaE)"])
    )
    ratio_rel_err_max = float(np.max(ratio_rel_err))

    summary_df = pd.DataFrame(
        {
            "metric": [
                "n_spins",
                "n_states",
                "beta",
                "J",
                "h",
                "acceptance_rate",
                "kernel_row_sum_max_error",
                "max_db_residual_theory",
                "max_db_residual_empirical",
                "stationarity_max_abs_theory",
                "L1(empirical_pi, theory_pi)",
                "chi2_statistic",
                "chi2_pvalue",
                "max_rel_err(P_ratio, exp(-beta*deltaE))",
            ],
            "value": [
                float(config.n_spins),
                float(states.shape[0]),
                config.beta,
                config.coupling_j,
                config.field_h,
                sim.acceptance_rate,
                row_sum_max_err,
                max_db_theory,
                max_db_emp,
                stationarity_residual,
                l1_emp_theory,
                float(chi2_stat),
                float(chi2_pvalue),
                ratio_rel_err_max,
            ],
        }
    )

    state_df = pd.DataFrame(
        {
            "state_id": np.arange(states.shape[0]),
            "spins": [spin_string(s) for s in states],
            "energy": energies,
            "pi_theory": pi,
            "pi_empirical": pi_hat,
            "abs_diff": np.abs(pi_hat - pi),
            "theory_db_residual_row_max": np.max(db_mat_theory, axis=1),
            "emp_db_residual_row_max": np.max(db_mat_emp, axis=1),
        }
    ).sort_values("state_id")

    checks = {
        "kernel_row_sum_max_error": row_sum_max_err,
        "max_db_residual_theory": max_db_theory,
        "stationarity_max_abs_theory": stationarity_residual,
        "max_db_residual_empirical": max_db_emp,
        "l1_empirical_theory": l1_emp_theory,
        "acceptance_rate": sim.acceptance_rate,
        "ratio_rel_err_max": ratio_rel_err_max,
    }

    return {
        "config": config,
        "summary_df": summary_df,
        "state_df": state_df,
        "edge_df": edge_df,
        "checks": checks,
    }


def main() -> None:
    config = DetailedBalanceConfig()
    result = run_detailed_balance_demo(config)

    summary_df = result["summary_df"]
    state_df = result["state_df"]
    edge_df = result["edge_df"]
    checks = result["checks"]

    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 180)

    print("Detailed Balance MVP: finite-state Ising Metropolis chain")
    print("=" * 90)
    print(config)

    print("\n--- Summary ---")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.10f}"))

    print("\n--- State probabilities (theory vs empirical) ---")
    print(state_df.to_string(index=False, float_format=lambda v: f"{v:.10f}"))

    print("\n--- Pairwise single-flip edge flux check ---")
    print(edge_df.to_string(index=False, float_format=lambda v: f"{v:.10f}"))

    assert checks["kernel_row_sum_max_error"] < 1e-12, (
        "Kernel row sums must be 1. "
        f"Got max error {checks['kernel_row_sum_max_error']:.3e}."
    )
    assert checks["max_db_residual_theory"] < 1e-12, (
        "Theoretical detailed balance residual too large. "
        f"Got {checks['max_db_residual_theory']:.3e}."
    )
    assert checks["stationarity_max_abs_theory"] < 1e-12, (
        "Theoretical stationarity residual too large. "
        f"Got {checks['stationarity_max_abs_theory']:.3e}."
    )
    assert checks["ratio_rel_err_max"] < 1e-12, (
        "Transition-ratio local detailed-balance check failed. "
        f"Got {checks['ratio_rel_err_max']:.3e}."
    )
    assert checks["max_db_residual_empirical"] < 1.2e-2, (
        "Empirical detailed-balance residual too large. "
        f"Got {checks['max_db_residual_empirical']:.3e}."
    )
    assert checks["l1_empirical_theory"] < 6.0e-2, (
        "Empirical distribution too far from Boltzmann distribution. "
        f"Got {checks['l1_empirical_theory']:.3e}."
    )
    assert 0.05 < checks["acceptance_rate"] < 0.95, (
        "Acceptance rate is out of expected range. "
        f"Got {checks['acceptance_rate']:.3f}."
    )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
