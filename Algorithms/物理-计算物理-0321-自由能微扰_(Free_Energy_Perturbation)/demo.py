"""Minimal runnable MVP for Free Energy Perturbation (PHYS-0318)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HarmonicState:
    """1D harmonic thermodynamic state."""

    k: float
    mu: float
    c: float

    def energy(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return 0.5 * self.k * (x - self.mu) ** 2 + self.c


@dataclass(frozen=True)
class MCMCSamples:
    samples: np.ndarray
    acceptance_rate: float


def metropolis_sample_1d(
    state: HarmonicState,
    beta: float,
    n_steps: int,
    burn_in: int,
    proposal_std: float,
    rng: np.random.Generator,
) -> MCMCSamples:
    """Sample from Boltzmann distribution of a 1D potential via random-walk Metropolis."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if proposal_std <= 0.0:
        raise ValueError("proposal_std must be positive")

    x = float(state.mu)
    u_x = float(state.energy(np.array([x]))[0])

    chain = np.empty(n_steps + burn_in, dtype=float)
    accepted = 0

    for t in range(n_steps + burn_in):
        x_prop = x + proposal_std * rng.normal()
        u_prop = float(state.energy(np.array([x_prop]))[0])

        log_alpha = -beta * (u_prop - u_x)
        if np.log(rng.random()) < min(0.0, log_alpha):
            x = x_prop
            u_x = u_prop
            accepted += 1

        chain[t] = x

    kept = chain[burn_in:]
    acceptance_rate = accepted / (n_steps + burn_in)
    return MCMCSamples(samples=kept, acceptance_rate=float(acceptance_rate))


def logmeanexp(a: np.ndarray) -> float:
    """Stable log(mean(exp(a)))."""
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        raise ValueError("input must be non-empty")
    m = float(np.max(arr))
    return m + float(np.log(np.mean(np.exp(arr - m))))


def fep_delta_f_from_samples(delta_u: np.ndarray, beta: float) -> float:
    """Zwanzig FEP estimator: DeltaF = -1/beta * log <exp(-beta*DeltaU)>."""
    delta_u = np.asarray(delta_u, dtype=float)
    return -(1.0 / beta) * logmeanexp(-beta * delta_u)


def effective_sample_size_from_logweights(log_w: np.ndarray) -> float:
    """Compute ESS = (sum w)^2 / sum(w^2) in a numerically stable way."""
    log_w = np.asarray(log_w, dtype=float)
    lse1 = logsumexp(log_w)
    lse2 = logsumexp(2.0 * log_w)
    return float(np.exp(2.0 * lse1 - lse2))


def logsumexp(a: np.ndarray) -> float:
    """Stable log(sum(exp(a)))."""
    arr = np.asarray(a, dtype=float)
    m = float(np.max(arr))
    return m + float(np.log(np.sum(np.exp(arr - m))))


def bootstrap_fep(
    delta_u: np.ndarray,
    beta: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return (mean, 2.5%, 97.5%) of bootstrapped FEP estimates."""
    delta_u = np.asarray(delta_u, dtype=float)
    n = delta_u.size
    if n == 0:
        raise ValueError("delta_u must be non-empty")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    est = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        est[i] = fep_delta_f_from_samples(delta_u[idx], beta)

    return float(np.mean(est)), float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))


def harmonic_exact_delta_f(state_a: HarmonicState, state_b: HarmonicState, beta: float) -> float:
    """Analytical DeltaF for 1D harmonic states, F_B - F_A."""
    return (state_b.c - state_a.c) + 0.5 * (1.0 / beta) * np.log(state_b.k / state_a.k)


def histogram_overlap_coefficient(x_a: np.ndarray, x_b: np.ndarray, bins: int = 80) -> float:
    """Histogram-based overlap coefficient in [0, 1], larger means better phase-space overlap."""
    x_a = np.asarray(x_a, dtype=float)
    x_b = np.asarray(x_b, dtype=float)
    low = float(min(np.min(x_a), np.min(x_b)))
    high = float(max(np.max(x_a), np.max(x_b)))
    edges = np.linspace(low, high, bins + 1)
    h_a, _ = np.histogram(x_a, bins=edges, density=True)
    h_b, _ = np.histogram(x_b, bins=edges, density=True)
    width = edges[1] - edges[0]
    overlap = np.sum(np.minimum(h_a, h_b)) * width
    return float(np.clip(overlap, 0.0, 1.0))


def main() -> None:
    seed = 20260407
    rng = np.random.default_rng(seed)

    beta = 1.0
    state_a = HarmonicState(k=1.2, mu=-0.4, c=0.15)
    state_b = HarmonicState(k=1.8, mu=0.1, c=0.50)

    n_steps = 60_000
    burn_in = 5_000
    proposal_std = 0.75
    n_bootstrap = 400

    samples_a = metropolis_sample_1d(state_a, beta, n_steps, burn_in, proposal_std, rng)
    samples_b = metropolis_sample_1d(state_b, beta, n_steps, burn_in, proposal_std, rng)

    x_a = samples_a.samples
    x_b = samples_b.samples

    delta_u_ab = state_b.energy(x_a) - state_a.energy(x_a)
    delta_u_ba = state_a.energy(x_b) - state_b.energy(x_b)

    delta_f_exact = harmonic_exact_delta_f(state_a, state_b, beta)
    delta_f_fwd = fep_delta_f_from_samples(delta_u_ab, beta)
    delta_f_ba = fep_delta_f_from_samples(delta_u_ba, beta)
    delta_f_from_reverse = -delta_f_ba

    boot_mean, boot_q025, boot_q975 = bootstrap_fep(delta_u_ab, beta, n_bootstrap, rng)

    log_w_fwd = -beta * delta_u_ab
    log_w_rev = -beta * delta_u_ba
    ess_fwd = effective_sample_size_from_logweights(log_w_fwd)
    ess_rev = effective_sample_size_from_logweights(log_w_rev)
    overlap = histogram_overlap_coefficient(x_a, x_b)

    abs_err_fwd = abs(delta_f_fwd - delta_f_exact)
    abs_err_rev = abs(delta_f_from_reverse - delta_f_exact)
    fwd_rev_gap = abs(delta_f_fwd - delta_f_from_reverse)
    ci_tolerance = 1.0e-2
    ci_contains_exact = (boot_q025 - ci_tolerance) <= delta_f_exact <= (boot_q975 + ci_tolerance)

    summary_table = pd.DataFrame(
        {
            "quantity": [
                "DeltaF exact (B-A)",
                "DeltaF FEP forward A->B",
                "DeltaF inferred from reverse B->A",
                "abs error forward",
                "abs error reverse-inferred",
                "forward/reverse consistency gap",
                "bootstrap mean (forward)",
                "bootstrap 95% CI lower",
                "bootstrap 95% CI upper",
                "ESS forward",
                "ESS reverse",
                "histogram overlap coefficient",
                "acceptance rate A chain",
                "acceptance rate B chain",
            ],
            "value": [
                delta_f_exact,
                delta_f_fwd,
                delta_f_from_reverse,
                abs_err_fwd,
                abs_err_rev,
                fwd_rev_gap,
                boot_mean,
                boot_q025,
                boot_q975,
                ess_fwd,
                ess_rev,
                overlap,
                samples_a.acceptance_rate,
                samples_b.acceptance_rate,
            ],
        }
    )

    checks = {
        "forward abs error < 0.10": abs_err_fwd < 0.10,
        "reverse-inferred abs error < 0.10": abs_err_rev < 0.10,
        "forward/reverse gap < 0.12": fwd_rev_gap < 0.12,
        "bootstrap CI(+/-0.01) contains exact DeltaF": ci_contains_exact,
        "ESS forward > 2000": ess_fwd > 2000.0,
        "ESS reverse > 2000": ess_rev > 2000.0,
        "overlap coefficient > 0.35": overlap > 0.35,
        "acceptance A in [0.2, 0.8]": 0.2 <= samples_a.acceptance_rate <= 0.8,
        "acceptance B in [0.2, 0.8]": 0.2 <= samples_b.acceptance_rate <= 0.8,
    }

    pd.set_option("display.float_format", lambda x: f"{x:.6f}")

    print("=== Free Energy Perturbation MVP (PHYS-0318) ===")
    print(f"seed={seed}, beta={beta}")
    print(f"state A: k={state_a.k}, mu={state_a.mu}, c={state_a.c}")
    print(f"state B: k={state_b.k}, mu={state_b.mu}, c={state_b.c}")
    print(
        f"sampling: n_steps={n_steps}, burn_in={burn_in}, proposal_std={proposal_std}, "
        f"n_bootstrap={n_bootstrap}"
    )

    print("\nSummary:")
    print(summary_table.to_string(index=False))

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
