"""Minimal MVP for the Master Equation in statistical mechanics.

We model a finite-state continuous-time Markov process:
    dp/dt = K p
where K is the generator matrix built from physically motivated transition rates.

This script implements:
1) A reversible 4-state ring model with local detailed balance.
2) Numerical integration via explicit RK4.
3) A source-level exact reference via eigen-decomposition.
4) Self-checks: probability conservation, detailed balance, stationarity, and error bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MasterEquationConfig:
    n_states: int = 4
    beta: float = 1.8
    gamma: float = 1.0
    energies: tuple[float, ...] = (0.0, 0.35, 0.9, 1.4)
    initial_prob: tuple[float, ...] = (0.82, 0.12, 0.04, 0.02)
    dt: float = 0.02
    n_steps: int = 800


def normalize_prob(p: np.ndarray) -> np.ndarray:
    """Normalize a probability vector and guard against invalid values."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("Probability vector must be 1D.")
    if np.any(~np.isfinite(p)):
        raise ValueError("Probability vector contains non-finite values.")
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0.0:
        raise ValueError("Probability vector sum must be positive.")
    return p / total


def build_ring_adjacency(n_states: int) -> np.ndarray:
    """Undirected nearest-neighbor ring adjacency matrix."""
    if n_states < 3:
        raise ValueError("Need at least 3 states for a ring.")

    adj = np.zeros((n_states, n_states), dtype=bool)
    for i in range(n_states):
        j = (i + 1) % n_states
        adj[i, j] = True
        adj[j, i] = True
    return adj


def build_rate_matrix(
    energies: np.ndarray,
    beta: float,
    gamma: float,
    adjacency: np.ndarray,
) -> np.ndarray:
    """Build W with convention W[dst, src] = rate(src -> dst).

    Local detailed-balance construction:
        w(src->dst) = gamma * exp[-0.5 * beta * (E_dst - E_src)]
    gives
        w(src->dst)/w(dst->src) = exp[-beta*(E_dst-E_src)].
    """
    n = energies.shape[0]
    if adjacency.shape != (n, n):
        raise ValueError("Adjacency shape mismatch.")

    w = np.zeros((n, n), dtype=float)
    for src in range(n):
        for dst in range(n):
            if src == dst or not adjacency[src, dst]:
                continue
            delta_e = energies[dst] - energies[src]
            w[dst, src] = gamma * math.exp(-0.5 * beta * delta_e)
    return w


def build_generator(w: np.ndarray) -> np.ndarray:
    """Build generator K for dp/dt = Kp with column-sum conservation."""
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("Rate matrix must be square.")

    k = w.copy()
    outgoing = np.sum(w, axis=0)
    np.fill_diagonal(k, -outgoing)
    return k


def boltzmann_distribution(energies: np.ndarray, beta: float) -> np.ndarray:
    """Compute equilibrium Boltzmann distribution for discrete energy levels."""
    shifted = energies - float(np.min(energies))
    weights = np.exp(-beta * shifted)
    return weights / float(np.sum(weights))


def stationary_distribution_from_generator(k: np.ndarray) -> np.ndarray:
    """Extract stationary distribution from the eigenvector near eigenvalue 0."""
    eigvals, eigvecs = np.linalg.eig(k)
    idx = int(np.argmin(np.abs(eigvals)))
    vec = np.real_if_close(eigvecs[:, idx], tol=1000).astype(float)

    if np.all(vec <= 0.0):
        vec = -vec
    vec = np.clip(vec, 0.0, None)

    if float(vec.sum()) <= 0.0:
        raise RuntimeError("Failed to extract a valid stationary vector from K.")
    return vec / float(vec.sum())


def solve_master_rk4(
    k: np.ndarray,
    p0: np.ndarray,
    dt: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve dp/dt = Kp by RK4 with light projection for numerical robustness."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    n = p0.shape[0]
    times = np.linspace(0.0, dt * n_steps, n_steps + 1)
    states = np.empty((n_steps + 1, n), dtype=float)

    p = normalize_prob(p0)
    states[0] = p

    for i in range(n_steps):
        k1 = k @ p
        k2 = k @ (p + 0.5 * dt * k1)
        k3 = k @ (p + 0.5 * dt * k2)
        k4 = k @ (p + dt * k3)
        p_next = p + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Projection keeps tiny numerical negatives from accumulating.
        p = normalize_prob(p_next)
        states[i + 1] = p

    return times, states


def solve_master_exact_eig(k: np.ndarray, p0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Reference solution via eigen-decomposition: p(t)=V exp(Lambda t) V^{-1} p(0)."""
    eigvals, eigvecs = np.linalg.eig(k)
    eigvecs_inv = np.linalg.inv(eigvecs)
    coeff = eigvecs_inv @ p0

    out = np.empty((times.shape[0], p0.shape[0]), dtype=float)
    for i, t in enumerate(times):
        mode = np.exp(eigvals * t) * coeff
        p_t = eigvecs @ mode
        p_real = np.real_if_close(p_t, tol=1000).astype(float)
        out[i] = normalize_prob(p_real)
    return out


def detailed_balance_residual(w: np.ndarray, pi: np.ndarray) -> float:
    """Max |w(i<-j)pi_j - w(j<-i)pi_i| over connected pairs."""
    n = w.shape[0]
    residual = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            lhs = w[i, j] * pi[j]
            rhs = w[j, i] * pi[i]
            residual = max(residual, abs(lhs - rhs))
    return residual


def entropy_production_rate(w: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    """Compute sigma = sum_{i<j} J_ij * log(F_ij/F_ji), nonnegative in theory."""
    n = w.shape[0]
    sigma = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            f_ij = w[i, j] * p[j]
            f_ji = w[j, i] * p[i]
            if f_ij <= 0.0 and f_ji <= 0.0:
                continue
            sigma += (f_ij - f_ji) * math.log((f_ij + eps) / (f_ji + eps))
    return sigma


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-15) -> float:
    """KL(p||q) for discrete distributions."""
    return float(np.sum(p * np.log((p + eps) / (q + eps))))


def build_observation_table(
    times: np.ndarray,
    p_rk4: np.ndarray,
    p_exact: np.ndarray,
    pi_eq: np.ndarray,
    sigma: np.ndarray,
) -> pd.DataFrame:
    """Construct a compact diagnostics table."""
    l1_to_eq = np.sum(np.abs(p_rk4 - pi_eq), axis=1)
    kl_to_eq = np.array([kl_divergence(p, pi_eq) for p in p_rk4], dtype=float)
    rk4_exact_linf = np.max(np.abs(p_rk4 - p_exact), axis=1)

    data: dict[str, np.ndarray] = {
        "t": times,
        "l1_to_eq": l1_to_eq,
        "kl_to_eq": kl_to_eq,
        "entropy_prod": sigma,
        "rk4_exact_linf": rk4_exact_linf,
    }

    for i in range(p_rk4.shape[1]):
        data[f"p{i}"] = p_rk4[:, i]

    return pd.DataFrame(data)


def main() -> None:
    cfg = MasterEquationConfig()

    energies = np.asarray(cfg.energies, dtype=float)
    p0 = normalize_prob(np.asarray(cfg.initial_prob, dtype=float))

    if energies.shape[0] != cfg.n_states:
        raise ValueError("n_states and energies length mismatch.")
    if p0.shape[0] != cfg.n_states:
        raise ValueError("n_states and initial_prob length mismatch.")

    adjacency = build_ring_adjacency(cfg.n_states)
    w = build_rate_matrix(energies, cfg.beta, cfg.gamma, adjacency)
    k = build_generator(w)

    pi_eq = boltzmann_distribution(energies, cfg.beta)
    pi_from_k = stationary_distribution_from_generator(k)

    times, p_rk4 = solve_master_rk4(k, p0, cfg.dt, cfg.n_steps)
    p_exact = solve_master_exact_eig(k, p0, times)

    sigma = np.array([entropy_production_rate(w, p) for p in p_rk4], dtype=float)
    table = build_observation_table(times, p_rk4, p_exact, pi_eq, sigma)

    sample_idx = sorted(set(list(range(0, cfg.n_steps + 1, 100)) + [cfg.n_steps]))
    snapshot = table.iloc[sample_idx].reset_index(drop=True)

    print("Master Equation MVP: reversible finite-state continuous-time Markov process")
    print(f"n_states={cfg.n_states}, beta={cfg.beta:.3f}, dt={cfg.dt:.3f}, steps={cfg.n_steps}")
    print(f"energies={energies.tolist()}")
    print("\nBoltzmann stationary distribution:", np.round(pi_eq, 6))
    print("Stationary distribution from K-eigenvector:", np.round(pi_from_k, 6))

    with pd.option_context("display.width", 180, "display.max_columns", 20, "display.precision", 6):
        print("\nTime snapshots:")
        print(snapshot.to_string(index=False))

    colsum_residual = float(np.max(np.abs(np.sum(k, axis=0))))
    db_residual = detailed_balance_residual(w, pi_eq)
    stationarity_gap = float(np.sum(np.abs(pi_eq - pi_from_k)))

    prob_mass_error = float(np.max(np.abs(np.sum(p_rk4, axis=1) - 1.0)))
    min_prob = float(np.min(p_rk4))
    final_l1 = float(table["l1_to_eq"].iloc[-1])
    max_rk4_exact_err = float(table["rk4_exact_linf"].max())
    max_kl_increase = float(np.max(np.diff(table["kl_to_eq"].to_numpy())))
    min_entropy_prod = float(np.min(table["entropy_prod"]))

    print(
        "\nChecks summary:"
        f" colsum_residual={colsum_residual:.3e},"
        f" detailed_balance_residual={db_residual:.3e},"
        f" stationarity_gap={stationarity_gap:.3e},"
        f" final_l1_to_eq={final_l1:.3e},"
        f" max_rk4_exact_linf={max_rk4_exact_err:.3e}"
    )

    # Deterministic self-validation gates for this MVP.
    assert colsum_residual < 1e-12
    assert db_residual < 1e-12
    assert stationarity_gap < 1e-10

    assert prob_mass_error < 1e-12
    assert min_prob >= 0.0

    assert final_l1 < 2.5e-3
    assert max_rk4_exact_err < 1.2e-4

    # For reversible chains, KL to equilibrium should not increase (up to tiny numerics).
    assert max_kl_increase < 3e-8

    # Entropy production should be nonnegative in theory; allow tiny numeric noise.
    assert min_entropy_prod > -1e-10

    print("All checks passed.")


if __name__ == "__main__":
    main()
