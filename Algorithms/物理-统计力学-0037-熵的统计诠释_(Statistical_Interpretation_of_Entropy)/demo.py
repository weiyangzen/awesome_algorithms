"""MVP for the statistical interpretation of entropy.

This script demonstrates three closely related entropy notions on a finite system
of N independent two-level particles (ground/excited):
1) Boltzmann entropy of a macrostate: S_B(m) = k_B ln Omega(m)
2) Gibbs entropy over microstates: S_G = -k_B sum_i p_i ln p_i
3) Coarse-grained macro entropy: S_macro = -k_B sum_m P(m) ln P(m)

The implementation is deterministic, non-interactive, and self-validating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp


@dataclass(frozen=True)
class EntropyConfig:
    n_particles: int = 80
    epsilon: float = 1.0
    k_b: float = 1.0
    temperature: float = 1.4
    n_mc_samples: int = 220_000
    seed: int = 20260407
    temperature_grid: tuple[float, ...] = (0.35, 0.5, 0.75, 1.0, 1.4, 1.8, 2.5, 4.0)

    def validate(self) -> None:
        if self.n_particles <= 0:
            raise ValueError("n_particles must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if self.k_b <= 0.0:
            raise ValueError("k_b must be > 0")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if self.n_mc_samples < 20_000:
            raise ValueError("n_mc_samples must be >= 20000 for stable Monte Carlo statistics")

        temps = np.asarray(self.temperature_grid, dtype=np.float64)
        if temps.ndim != 1 or temps.size < 2:
            raise ValueError("temperature_grid must contain at least two temperatures")
        if np.any(~np.isfinite(temps)) or np.any(temps <= 0.0):
            raise ValueError("temperature_grid must contain finite positive values")
        if np.any(np.diff(temps) <= 0.0):
            raise ValueError("temperature_grid must be strictly increasing")


def beta_from_temperature(temperature: float, k_b: float) -> float:
    return 1.0 / (k_b * temperature)


def log_binomial_coefficients(n: int) -> np.ndarray:
    m = np.arange(n + 1, dtype=np.float64)
    return gammaln(n + 1.0) - gammaln(m + 1.0) - gammaln((n - m) + 1.0)


def entropy_from_probabilities(probs: np.ndarray, k_b: float) -> float:
    mask = probs > 0.0
    return float(-k_b * np.sum(probs[mask] * np.log(probs[mask])))


def exact_entropy_terms(
    n_particles: int,
    epsilon: float,
    k_b: float,
    temperature: float,
) -> dict[str, np.ndarray | float | int]:
    beta = beta_from_temperature(temperature, k_b)

    m_values = np.arange(n_particles + 1, dtype=np.int64)
    log_omega = log_binomial_coefficients(n_particles)

    log_z_single = np.log1p(np.exp(-beta * epsilon))
    log_p_microstate = -beta * epsilon * m_values.astype(np.float64) - n_particles * log_z_single

    log_p_macro_raw = log_omega + log_p_microstate
    log_norm = float(logsumexp(log_p_macro_raw))
    log_p_macro = log_p_macro_raw - log_norm
    p_macro = np.exp(log_p_macro)

    s_macro = float(-k_b * np.sum(p_macro * log_p_macro))
    s_conditional = float(k_b * np.sum(p_macro * log_omega))
    s_gibbs = float(-k_b * np.sum(p_macro * log_p_microstate))

    mode_idx = int(np.argmax(p_macro))
    mode_m = int(m_values[mode_idx])
    s_boltzmann_mode = float(k_b * log_omega[mode_idx])

    p_excited = 1.0 / (np.exp(beta * epsilon) + 1.0)
    mean_m_theory = float(np.dot(m_values.astype(np.float64), p_macro))

    return {
        "beta": float(beta),
        "p_excited": float(p_excited),
        "m_values": m_values,
        "log_omega": log_omega,
        "log_p_microstate": log_p_microstate,
        "log_p_macro": log_p_macro,
        "p_macro": p_macro,
        "s_macro": s_macro,
        "s_conditional": s_conditional,
        "s_gibbs": s_gibbs,
        "mode_m": mode_m,
        "mode_index": mode_idx,
        "s_boltzmann_mode": s_boltzmann_mode,
        "mean_m_theory": mean_m_theory,
        "log_norm": log_norm,
    }


def monte_carlo_estimates(
    cfg: EntropyConfig,
    p_excited: float,
    log_p_microstate: np.ndarray,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(cfg.seed)
    sampled_m = rng.binomial(cfg.n_particles, p_excited, size=cfg.n_mc_samples)

    counts = np.bincount(sampled_m, minlength=cfg.n_particles + 1).astype(np.int64)
    p_macro_emp = counts / float(cfg.n_mc_samples)

    s_macro_emp = entropy_from_probabilities(p_macro_emp, cfg.k_b)
    s_gibbs_emp = float(-cfg.k_b * np.mean(log_p_microstate[sampled_m]))
    mean_m_emp = float(np.mean(sampled_m))

    return {
        "counts": counts,
        "p_macro_emp": p_macro_emp,
        "s_macro_emp": s_macro_emp,
        "s_gibbs_emp": s_gibbs_emp,
        "mean_m_emp": mean_m_emp,
    }


def build_local_window_table(
    m_values: np.ndarray,
    log_omega: np.ndarray,
    p_macro: np.ndarray,
    log_p_microstate: np.ndarray,
    k_b: float,
    mode_index: int,
    radius: int = 4,
) -> pd.DataFrame:
    lo = max(0, mode_index - radius)
    hi = min(int(m_values[-1]), mode_index + radius) + 1

    rows = []
    for idx in range(lo, hi):
        rows.append(
            {
                "m": int(m_values[idx]),
                "S_B(m)=k_BlnOmega": float(k_b * log_omega[idx]),
                "P_macro(m)": float(p_macro[idx]),
                "log_p_microstate": float(log_p_microstate[idx]),
            }
        )
    return pd.DataFrame(rows)


def build_temperature_scan(cfg: EntropyConfig) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for t in cfg.temperature_grid:
        terms = exact_entropy_terms(cfg.n_particles, cfg.epsilon, cfg.k_b, t)
        rows.append(
            {
                "T": float(t),
                "beta": float(terms["beta"]),
                "S_gibbs": float(terms["s_gibbs"]),
                "S_macro": float(terms["s_macro"]),
                "S_conditional": float(terms["s_conditional"]),
                "S_B_mode": float(terms["s_boltzmann_mode"]),
                "S_gibbs/(N*k_B)": float(terms["s_gibbs"] / (cfg.n_particles * cfg.k_b)),
                "mode_m": float(terms["mode_m"]),
            }
        )
    return pd.DataFrame(rows)


def run_checks(
    cfg: EntropyConfig,
    exact_terms: dict[str, np.ndarray | float | int],
    mc_terms: dict[str, np.ndarray | float],
    scan_df: pd.DataFrame,
) -> None:
    p_macro = np.asarray(exact_terms["p_macro"], dtype=np.float64)
    log_omega = np.asarray(exact_terms["log_omega"], dtype=np.float64)
    s_gibbs = float(exact_terms["s_gibbs"])
    s_macro = float(exact_terms["s_macro"])
    s_conditional = float(exact_terms["s_conditional"])
    mean_m_theory = float(exact_terms["mean_m_theory"])
    p_excited = float(exact_terms["p_excited"])
    log_norm = float(exact_terms["log_norm"])

    p_macro_emp = np.asarray(mc_terms["p_macro_emp"], dtype=np.float64)
    s_gibbs_emp = float(mc_terms["s_gibbs_emp"])
    mean_m_emp = float(mc_terms["mean_m_emp"])

    decomposition_residual = abs(s_gibbs - (s_macro + s_conditional))
    total_variation = 0.5 * float(np.sum(np.abs(p_macro_emp - p_macro)))
    rel_err_s_gibbs_mc = abs(s_gibbs_emp - s_gibbs) / max(abs(s_gibbs), 1e-12)

    if not np.isclose(np.sum(p_macro), 1.0, atol=1e-12):
        raise AssertionError("exact macro distribution does not sum to 1")
    if not np.isclose(np.sum(p_macro_emp), 1.0, atol=1e-12):
        raise AssertionError("empirical macro distribution does not sum to 1")
    if abs(log_norm) > 1e-10:
        raise AssertionError(f"macro normalization log_norm too large: {log_norm:.3e}")

    if decomposition_residual > 1e-10:
        raise AssertionError(f"entropy decomposition residual too large: {decomposition_residual:.3e}")

    expected_mean = cfg.n_particles * p_excited
    if abs(mean_m_theory - expected_mean) > 1e-10:
        raise AssertionError("mean excitation mismatch between combinatorial and Bernoulli forms")

    if total_variation > 0.018:
        raise AssertionError(f"Monte Carlo macro distribution mismatch too large: TV={total_variation:.6f}")
    if rel_err_s_gibbs_mc > 0.008:
        raise AssertionError(f"Monte Carlo Gibbs entropy relative error too large: {rel_err_s_gibbs_mc:.6f}")
    if abs(mean_m_emp - mean_m_theory) > 0.18:
        raise AssertionError("Monte Carlo mean excitation mismatch too large")

    s_upper_bound = cfg.n_particles * cfg.k_b * np.log(2.0)
    if not (0.0 <= s_gibbs <= s_upper_bound + 1e-10):
        raise AssertionError("Gibbs entropy out of physical bounds for two-level finite system")

    s_scan = scan_df["S_gibbs"].to_numpy(dtype=np.float64)
    if np.any(np.diff(s_scan) < -1e-8):
        raise AssertionError("S_gibbs should be nondecreasing with temperature in this model")


def main() -> None:
    cfg = EntropyConfig()
    cfg.validate()

    exact_terms = exact_entropy_terms(cfg.n_particles, cfg.epsilon, cfg.k_b, cfg.temperature)
    mc_terms = monte_carlo_estimates(
        cfg,
        p_excited=float(exact_terms["p_excited"]),
        log_p_microstate=np.asarray(exact_terms["log_p_microstate"], dtype=np.float64),
    )
    scan_df = build_temperature_scan(cfg)

    p_macro = np.asarray(exact_terms["p_macro"], dtype=np.float64)
    p_macro_emp = np.asarray(mc_terms["p_macro_emp"], dtype=np.float64)

    local_df = build_local_window_table(
        m_values=np.asarray(exact_terms["m_values"], dtype=np.int64),
        log_omega=np.asarray(exact_terms["log_omega"], dtype=np.float64),
        p_macro=p_macro,
        log_p_microstate=np.asarray(exact_terms["log_p_microstate"], dtype=np.float64),
        k_b=cfg.k_b,
        mode_index=int(exact_terms["mode_index"]),
        radius=4,
    )

    s_gibbs = float(exact_terms["s_gibbs"])
    s_macro = float(exact_terms["s_macro"])
    s_conditional = float(exact_terms["s_conditional"])
    s_b_mode = float(exact_terms["s_boltzmann_mode"])

    s_gibbs_emp = float(mc_terms["s_gibbs_emp"])
    s_macro_emp = float(mc_terms["s_macro_emp"])
    mean_m_theory = float(exact_terms["mean_m_theory"])
    mean_m_emp = float(mc_terms["mean_m_emp"])

    total_variation = 0.5 * float(np.sum(np.abs(p_macro_emp - p_macro)))
    decomposition_residual = abs(s_gibbs - (s_macro + s_conditional))

    print("=== Statistical Interpretation of Entropy MVP ===")
    print(
        f"config: N={cfg.n_particles}, epsilon={cfg.epsilon}, k_B={cfg.k_b}, "
        f"T={cfg.temperature}, n_mc_samples={cfg.n_mc_samples}, seed={cfg.seed}"
    )
    print(
        f"beta={float(exact_terms['beta']):.6f}, p_excited={float(exact_terms['p_excited']):.6f}, "
        f"mode_m={int(exact_terms['mode_m'])}"
    )
    print()
    print("Entropy decomposition at target temperature:")
    print(
        f"S_gibbs={s_gibbs:.6f}, S_macro={s_macro:.6f}, "
        f"S_conditional={s_conditional:.6f}, S_B_mode={s_b_mode:.6f}"
    )
    print(
        f"S_gibbs_MC={s_gibbs_emp:.6f}, S_macro_MC={s_macro_emp:.6f}, "
        f"mean_m_theory={mean_m_theory:.6f}, mean_m_MC={mean_m_emp:.6f}"
    )
    print(
        f"TV(P_macro_exact, P_macro_MC)={total_variation:.6f}, "
        f"decomposition_residual={decomposition_residual:.3e}"
    )
    print()
    print("Macrostate window around the most probable excitation count:")
    print(local_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print("Temperature scan (exact):")
    print(scan_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    run_checks(cfg, exact_terms, mc_terms, scan_df)
    print("All checks passed.")


if __name__ == "__main__":
    main()
