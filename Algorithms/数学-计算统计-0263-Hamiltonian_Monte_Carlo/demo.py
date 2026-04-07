"""Minimal runnable MVP for Hamiltonian Monte Carlo (HMC).

This implementation is intentionally small and explicit:
- define target log density and gradient
- simulate Hamiltonian dynamics with leapfrog integration
- apply Metropolis correction
- perform simple warmup step-size adaptation

Target distribution for this demo is a 2D Gaussian with known mean/covariance,
so we can verify sampling quality numerically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

Array = np.ndarray
LogpGradFn = Callable[[Array], Tuple[float, Array]]


@dataclass
class HMCConfig:
    """Configuration for one HMC chain."""

    num_warmup: int = 700
    num_samples: int = 2200
    step_size: float = 0.22
    num_leapfrog: int = 12
    target_accept: float = 0.75
    adapt_rate: float = 0.03
    seed: int = 20260407


def make_gaussian_target(mean: Array, cov: Array) -> LogpGradFn:
    """Return callable that computes log-density and gradient for Gaussian target."""

    inv_cov = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")

    dim = mean.shape[0]
    log_norm = -0.5 * (dim * math.log(2.0 * math.pi) + logdet)

    def logp_and_grad(q: Array) -> Tuple[float, Array]:
        diff = q - mean
        logp = log_norm - 0.5 * float(diff.T @ inv_cov @ diff)
        grad = -(inv_cov @ diff)
        return logp, grad

    return logp_and_grad


def joint_log_density(logp: float, momentum: Array) -> float:
    """Joint log density of augmented HMC state (q, p)."""

    return float(logp - 0.5 * np.dot(momentum, momentum))


def leapfrog(
    q: Array,
    p: Array,
    logp: float,
    grad: Array,
    step_size: float,
    num_steps: int,
    logp_grad_fn: LogpGradFn,
) -> Tuple[Array, Array, float, Array]:
    """Perform velocity-Verlet integration for Hamiltonian dynamics."""

    q_new = q.copy()
    p_new = p.copy()
    logp_new = logp
    grad_new = grad.copy()

    for _ in range(num_steps):
        p_new += 0.5 * step_size * grad_new
        q_new += step_size * p_new
        logp_new, grad_new = logp_grad_fn(q_new)
        p_new += 0.5 * step_size * grad_new

    return q_new, p_new, logp_new, grad_new


def hmc_transition(
    q_current: Array,
    logp_current: float,
    grad_current: Array,
    step_size: float,
    num_leapfrog: int,
    rng: np.random.Generator,
    logp_grad_fn: LogpGradFn,
) -> Tuple[Array, float, Array, bool, float, float]:
    """Run one HMC proposal and accept/reject it."""

    p0 = rng.normal(size=q_current.shape)
    joint_start = joint_log_density(logp_current, p0)

    q_prop, p_prop, logp_prop, grad_prop = leapfrog(
        q=q_current,
        p=p0,
        logp=logp_current,
        grad=grad_current,
        step_size=step_size,
        num_steps=num_leapfrog,
        logp_grad_fn=logp_grad_fn,
    )

    joint_end = joint_log_density(logp_prop, p_prop)
    log_accept_ratio = joint_end - joint_start
    accept_prob = 1.0 if log_accept_ratio >= 0.0 else math.exp(log_accept_ratio)

    accepted = bool(rng.uniform() < accept_prob)
    if accepted:
        return q_prop, logp_prop, grad_prop, True, accept_prob, joint_end - joint_start

    return q_current, logp_current, grad_current, False, accept_prob, joint_end - joint_start


def adapt_step_size(
    step_size: float,
    accept_prob: float,
    target_accept: float,
    adapt_rate: float,
) -> float:
    """Small warmup adaptation on log(step_size) scale."""

    log_step = math.log(step_size) + adapt_rate * (accept_prob - target_accept)
    return float(np.clip(math.exp(log_step), 1e-4, 2.0))


def run_hmc(initial_q: Array, config: HMCConfig, logp_grad_fn: LogpGradFn) -> Dict[str, Array | float]:
    """Run one HMC chain with warmup and sampling phases."""

    rng = np.random.default_rng(config.seed)
    q = initial_q.astype(float).copy()
    logp, grad = logp_grad_fn(q)

    step_size = config.step_size
    total_iters = config.num_warmup + config.num_samples
    dim = q.shape[0]
    samples = np.zeros((config.num_samples, dim), dtype=float)

    warmup_accept_prob_sum = 0.0
    sample_accept_prob_sum = 0.0
    accepted_count = 0
    energy_deltas = np.zeros(total_iters, dtype=float)

    for itr in range(total_iters):
        q, logp, grad, accepted, accept_prob, energy_delta = hmc_transition(
            q_current=q,
            logp_current=logp,
            grad_current=grad,
            step_size=step_size,
            num_leapfrog=config.num_leapfrog,
            rng=rng,
            logp_grad_fn=logp_grad_fn,
        )

        energy_deltas[itr] = energy_delta

        if itr < config.num_warmup:
            warmup_accept_prob_sum += accept_prob
            step_size = adapt_step_size(
                step_size=step_size,
                accept_prob=accept_prob,
                target_accept=config.target_accept,
                adapt_rate=config.adapt_rate,
            )
        else:
            sample_idx = itr - config.num_warmup
            samples[sample_idx] = q
            sample_accept_prob_sum += accept_prob
            accepted_count += int(accepted)

    return {
        "samples": samples,
        "final_step_size": float(step_size),
        "warmup_accept_prob": warmup_accept_prob_sum / max(1, config.num_warmup),
        "sample_accept_prob": sample_accept_prob_sum / max(1, config.num_samples),
        "sample_accept_rate": accepted_count / max(1, config.num_samples),
        "mean_abs_energy_delta": float(np.mean(np.abs(energy_deltas))),
        "max_abs_energy_delta": float(np.max(np.abs(energy_deltas))),
    }


def empirical_covariance(samples: Array) -> Array:
    """Sample covariance with Bessel correction."""

    centered = samples - np.mean(samples, axis=0, keepdims=True)
    return (centered.T @ centered) / (samples.shape[0] - 1)


def estimate_ess_per_dim(samples: Array, max_lag: int = 500) -> Array:
    """Estimate ESS per dimension with initial positive sequence truncation."""

    n, dim = samples.shape
    ess = np.zeros(dim, dtype=float)

    for d in range(dim):
        x = samples[:, d] - np.mean(samples[:, d])
        var = np.dot(x, x) / n
        if var <= 1e-15:
            ess[d] = 1.0
            continue

        rho_sum = 0.0
        lag_upper = min(max_lag, n - 1)
        for lag in range(1, lag_upper + 1):
            autocov = np.dot(x[:-lag], x[lag:]) / (n - lag)
            rho = autocov / var
            if rho <= 0.0:
                break
            rho_sum += rho

        ess[d] = n / (1.0 + 2.0 * rho_sum)
        ess[d] = float(np.clip(ess[d], 1.0, float(n)))

    return ess


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    true_mean = np.array([1.0, -1.0], dtype=float)
    true_cov = np.array([[1.0, 0.75], [0.75, 1.4]], dtype=float)

    logp_grad_fn = make_gaussian_target(mean=true_mean, cov=true_cov)
    config = HMCConfig()
    initial_q = np.array([2.5, -2.0], dtype=float)

    result = run_hmc(initial_q=initial_q, config=config, logp_grad_fn=logp_grad_fn)
    samples = result["samples"]

    est_mean = np.mean(samples, axis=0)
    est_cov = empirical_covariance(samples)
    ess = estimate_ess_per_dim(samples=samples)

    mean_l2_error = float(np.linalg.norm(est_mean - true_mean))
    cov_fro_error = float(np.linalg.norm(est_cov - true_cov, ord="fro"))

    print("=== Hamiltonian Monte Carlo MVP ===")
    print(f"seed: {config.seed}")
    print(f"warmup_iters: {config.num_warmup}, sample_iters: {config.num_samples}")
    print(
        "step_size(init->final): "
        f"{config.step_size:.4f} -> {result['final_step_size']:.4f}, "
        f"leapfrog_steps: {config.num_leapfrog}"
    )
    print(
        "accept_prob(warmup/sample): "
        f"{result['warmup_accept_prob']:.4f} / {result['sample_accept_prob']:.4f}"
    )
    print(f"accept_rate(sample): {result['sample_accept_rate']:.4f}")
    print(
        "energy_delta(|.| mean/max): "
        f"{result['mean_abs_energy_delta']:.6f} / {result['max_abs_energy_delta']:.6f}"
    )
    print(f"true_mean: {true_mean}")
    print(f"estimated_mean: {est_mean}")
    print(f"mean_l2_error: {mean_l2_error:.6f}")
    print(f"true_cov:\n{true_cov}")
    print(f"estimated_cov:\n{est_cov}")
    print(f"cov_fro_error: {cov_fro_error:.6f}")
    print(f"ESS per dim: {ess}")

    assert mean_l2_error < 0.18, "Mean estimation error is too large."
    assert cov_fro_error < 0.25, "Covariance estimation error is too large."
    assert result["sample_accept_rate"] > 0.55, "Acceptance rate is too low."
    assert float(np.min(ess)) > 250.0, "ESS is too low for this demo setup."

    print("All checks passed.")


if __name__ == "__main__":
    main()
