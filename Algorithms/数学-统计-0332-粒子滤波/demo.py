"""Minimal runnable MVP for Particle Filter (MATH-0332).

This demo implements a bootstrap particle filter from scratch (no black-box
state-estimation library) on a classic nonlinear/non-Gaussian state-space
benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PFResult:
    """Container for online filtering outputs."""

    estimates: np.ndarray  # shape (T,)
    variances: np.ndarray  # shape (T,)
    ess_history: np.ndarray  # shape (T,)
    resampled: np.ndarray  # shape (T,), bool
    log_evidence: float
    final_particles: np.ndarray  # shape (N,)
    final_weights: np.ndarray  # shape (N,)


def state_transition(x_prev: np.ndarray, t: int) -> np.ndarray:
    """Nonlinear transition used in common PF benchmarks."""
    return 0.5 * x_prev + 25.0 * x_prev / (1.0 + x_prev * x_prev) + 8.0 * np.cos(1.2 * t)


def observation_fn(x: np.ndarray) -> np.ndarray:
    """Nonlinear observation mapping."""
    return x * x / 20.0


def simulate_nonlinear_system(
    t_len: int,
    q: float,
    r: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate latent states and observations."""
    if q <= 0.0 or r <= 0.0:
        raise ValueError("q and r must be positive.")

    rng = np.random.default_rng(seed)
    states = np.zeros(t_len, dtype=np.float64)
    obs = np.zeros(t_len, dtype=np.float64)

    states[0] = rng.normal(loc=0.0, scale=1.0)
    obs[0] = observation_fn(np.array([states[0]]))[0] + rng.normal(0.0, np.sqrt(r))

    for t in range(1, t_len):
        states[t] = state_transition(np.array([states[t - 1]]), t=t)[0] + rng.normal(0.0, np.sqrt(q))
        obs[t] = observation_fn(np.array([states[t]]))[0] + rng.normal(0.0, np.sqrt(r))

    return states, obs


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling in O(N)."""
    n = weights.shape[0]
    positions = (rng.random() + np.arange(n)) / n
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0

    indices = np.zeros(n, dtype=np.int64)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices


def particle_filter(
    observations: np.ndarray,
    n_particles: int,
    q: float,
    r: float,
    init_mean: float = 0.0,
    init_std: float = 5.0,
    ess_ratio: float = 0.5,
    seed: int = 0,
) -> PFResult:
    """Bootstrap particle filter for 1D nonlinear state-space model."""
    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")
    if q <= 0.0 or r <= 0.0:
        raise ValueError("q and r must be positive.")
    if not (0.0 < ess_ratio <= 1.0):
        raise ValueError("ess_ratio must be in (0, 1].")

    rng = np.random.default_rng(seed)
    t_len = observations.shape[0]
    n = n_particles
    eps = 1e-300

    particles = rng.normal(loc=init_mean, scale=init_std, size=n)
    weights = np.full(n, 1.0 / n, dtype=np.float64)

    estimates = np.zeros(t_len, dtype=np.float64)
    variances = np.zeros(t_len, dtype=np.float64)
    ess_history = np.zeros(t_len, dtype=np.float64)
    resampled = np.zeros(t_len, dtype=bool)
    log_evidence = 0.0

    for t in range(t_len):
        # 1) Propagate particles through nonlinear transition.
        particles = state_transition(particles, t=t + 1) + rng.normal(0.0, np.sqrt(q), size=n)

        # 2) Update weights in log-domain for numerical stability.
        y_pred = observation_fn(particles)
        log_likelihood = -0.5 * np.log(2.0 * np.pi * r) - 0.5 * ((observations[t] - y_pred) ** 2) / r
        logw_raw = np.log(weights + eps) + log_likelihood
        m = float(np.max(logw_raw))
        w_unnorm = np.exp(logw_raw - m)
        z = float(np.sum(w_unnorm))
        if not np.isfinite(z) or z <= 0.0:
            raise RuntimeError(f"Weight normalization failed at t={t}.")
        weights = w_unnorm / z

        # Incremental evidence estimate: log sum_i exp(logw_raw_i).
        log_evidence += m + np.log(z)

        # 3) Estimate posterior moments.
        mean_t = float(np.sum(weights * particles))
        centered = particles - mean_t
        var_t = float(np.sum(weights * centered * centered))
        estimates[t] = mean_t
        variances[t] = var_t

        # 4) Degeneracy check via effective sample size.
        ess_t = 1.0 / float(np.sum(weights * weights))
        ess_history[t] = ess_t

        # 5) Resample if ESS below threshold.
        if ess_t < ess_ratio * n:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            weights.fill(1.0 / n)
            resampled[t] = True

    return PFResult(
        estimates=estimates,
        variances=variances,
        ess_history=ess_history,
        resampled=resampled,
        log_evidence=float(log_evidence),
        final_particles=particles,
        final_weights=weights,
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    print("Particle Filter MVP (MATH-0332)")
    print("=" * 72)

    t_len = 60
    q = 10.0
    r = 1.0
    n_particles = 1500

    true_states, observations = simulate_nonlinear_system(
        t_len=t_len,
        q=q,
        r=r,
        seed=332,
    )

    result = particle_filter(
        observations=observations,
        n_particles=n_particles,
        q=q,
        r=r,
        init_mean=0.0,
        init_std=5.0,
        ess_ratio=0.5,
        seed=332,
    )

    score_rmse = rmse(true_states, result.estimates)
    avg_ess = float(np.mean(result.ess_history))
    n_resampled = int(np.sum(result.resampled))

    print(f"T={t_len}, particles={n_particles}, q={q:.2f}, r={r:.2f}")
    print(f"log-evidence estimate: {result.log_evidence:.6f}")
    print(f"state RMSE: {score_rmse:.6f}")
    print(f"average ESS: {avg_ess:.2f} / {n_particles}")
    print(f"resampling count: {n_resampled}")
    print("-" * 72)
    print("t   obs        true_x      est_x       abs_err     ESS      resampled")
    for t in range(min(12, t_len)):
        err = abs(true_states[t] - result.estimates[t])
        flag = "Y" if result.resampled[t] else "N"
        print(
            f"{t:>2d}  {observations[t]:>8.3f}  {true_states[t]:>9.3f}  "
            f"{result.estimates[t]:>9.3f}  {err:>9.3f}  {result.ess_history[t]:>7.1f}     {flag}"
        )
    print("-" * 72)

    # Consistency checks.
    assert np.isfinite(result.log_evidence)
    assert np.all(np.isfinite(result.estimates))
    assert np.all(np.isfinite(result.variances))
    assert np.all(result.variances >= 0.0)
    assert np.all(result.ess_history >= 1.0 - 1e-9)
    assert np.all(result.ess_history <= n_particles + 1e-9)
    assert np.isclose(np.sum(result.final_weights), 1.0, atol=1e-12)

    print("All checks passed.")


if __name__ == "__main__":
    main()
