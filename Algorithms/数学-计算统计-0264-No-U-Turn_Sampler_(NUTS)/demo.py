"""Minimal runnable MVP for No-U-Turn Sampler (NUTS).

This script implements NUTS from source-level components:
- leapfrog integrator
- recursive tree doubling
- no-u-turn stopping criterion
- dual-averaging step size adaptation (warmup)

Target distribution in this MVP is a 2D Gaussian with known mean/covariance,
so we can verify the sampler quantitatively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class NutsConfig:
    """Configuration for one NUTS chain."""

    num_warmup: int = 800
    num_samples: int = 1200
    max_depth: int = 8
    target_accept: float = 0.80
    delta_max: float = 1000.0
    seed: int = 20260407


@dataclass
class TreeState:
    """Container for one subtree return in recursive NUTS building."""

    q_minus: Array
    p_minus: Array
    logp_minus: float
    grad_minus: Array

    q_plus: Array
    p_plus: Array
    logp_plus: float
    grad_plus: Array

    q_proposed: Array
    logp_proposed: float
    grad_proposed: Array

    n_valid: int
    s_continue: int
    alpha: float
    n_alpha: int
    n_divergent: int


LogpGradFn = Callable[[Array], Tuple[float, Array]]


def make_gaussian_target(mean: Array, cov: Array) -> LogpGradFn:
    """Return log-density and gradient callable for multivariate Gaussian."""

    inv_cov = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance must be positive definite.")
    dim = mean.shape[0]
    log_norm = -0.5 * (dim * math.log(2.0 * math.pi) + logdet)

    def logp_and_grad(q: Array) -> Tuple[float, Array]:
        diff = q - mean
        logp = log_norm - 0.5 * float(diff.T @ inv_cov @ diff)
        grad = -(inv_cov @ diff)
        return logp, grad

    return logp_and_grad


def joint_log_density(logp: float, p: Array) -> float:
    """Joint log density of position/momentum under HMC augmentation."""

    return float(logp - 0.5 * np.dot(p, p))


def leapfrog(
    q: Array,
    p: Array,
    logp: float,
    grad: Array,
    step_size: float,
    logp_grad_fn: LogpGradFn,
) -> Tuple[Array, Array, float, Array]:
    """One velocity-Verlet (leapfrog) step."""

    p_half = p + 0.5 * step_size * grad
    q_new = q + step_size * p_half
    logp_new, grad_new = logp_grad_fn(q_new)
    p_new = p_half + 0.5 * step_size * grad_new
    return q_new, p_new, logp_new, grad_new


def stop_criterion(q_minus: Array, q_plus: Array, p_minus: Array, p_plus: Array) -> bool:
    """No-U-Turn criterion from NUTS paper."""

    delta_q = q_plus - q_minus
    return (np.dot(delta_q, p_minus) >= 0.0) and (np.dot(delta_q, p_plus) >= 0.0)


def build_tree(
    q: Array,
    p: Array,
    logp: float,
    grad: Array,
    log_u: float,
    direction: int,
    depth: int,
    step_size: float,
    initial_joint: float,
    delta_max: float,
    rng: np.random.Generator,
    logp_grad_fn: LogpGradFn,
) -> TreeState:
    """Recursive binary tree expansion used by NUTS."""

    if depth == 0:
        q1, p1, logp1, grad1 = leapfrog(
            q=q,
            p=p,
            logp=logp,
            grad=grad,
            step_size=direction * step_size,
            logp_grad_fn=logp_grad_fn,
        )
        joint1 = joint_log_density(logp1, p1)
        n_valid = int(log_u <= joint1)
        diverged = int((log_u - delta_max) >= joint1)
        s_continue = int(diverged == 0)

        alpha = min(1.0, math.exp(min(0.0, joint1 - initial_joint)))

        return TreeState(
            q_minus=q1,
            p_minus=p1,
            logp_minus=logp1,
            grad_minus=grad1,
            q_plus=q1,
            p_plus=p1,
            logp_plus=logp1,
            grad_plus=grad1,
            q_proposed=q1,
            logp_proposed=logp1,
            grad_proposed=grad1,
            n_valid=n_valid,
            s_continue=s_continue,
            alpha=alpha,
            n_alpha=1,
            n_divergent=diverged,
        )

    left = build_tree(
        q=q,
        p=p,
        logp=logp,
        grad=grad,
        log_u=log_u,
        direction=direction,
        depth=depth - 1,
        step_size=step_size,
        initial_joint=initial_joint,
        delta_max=delta_max,
        rng=rng,
        logp_grad_fn=logp_grad_fn,
    )

    if left.s_continue == 0:
        return left

    if direction == -1:
        right = build_tree(
            q=left.q_minus,
            p=left.p_minus,
            logp=left.logp_minus,
            grad=left.grad_minus,
            log_u=log_u,
            direction=direction,
            depth=depth - 1,
            step_size=step_size,
            initial_joint=initial_joint,
            delta_max=delta_max,
            rng=rng,
            logp_grad_fn=logp_grad_fn,
        )
        q_minus, p_minus, logp_minus, grad_minus = (
            right.q_minus,
            right.p_minus,
            right.logp_minus,
            right.grad_minus,
        )
        q_plus, p_plus, logp_plus, grad_plus = (
            left.q_plus,
            left.p_plus,
            left.logp_plus,
            left.grad_plus,
        )
    else:
        right = build_tree(
            q=left.q_plus,
            p=left.p_plus,
            logp=left.logp_plus,
            grad=left.grad_plus,
            log_u=log_u,
            direction=direction,
            depth=depth - 1,
            step_size=step_size,
            initial_joint=initial_joint,
            delta_max=delta_max,
            rng=rng,
            logp_grad_fn=logp_grad_fn,
        )
        q_minus, p_minus, logp_minus, grad_minus = (
            left.q_minus,
            left.p_minus,
            left.logp_minus,
            left.grad_minus,
        )
        q_plus, p_plus, logp_plus, grad_plus = (
            right.q_plus,
            right.p_plus,
            right.logp_plus,
            right.grad_plus,
        )

    n_total = left.n_valid + right.n_valid

    q_proposed = left.q_proposed
    logp_proposed = left.logp_proposed
    grad_proposed = left.grad_proposed

    if n_total > 0 and rng.uniform() < (right.n_valid / n_total):
        q_proposed = right.q_proposed
        logp_proposed = right.logp_proposed
        grad_proposed = right.grad_proposed

    s_continue = int(
        left.s_continue
        and right.s_continue
        and stop_criterion(q_minus=q_minus, q_plus=q_plus, p_minus=p_minus, p_plus=p_plus)
    )

    return TreeState(
        q_minus=q_minus,
        p_minus=p_minus,
        logp_minus=logp_minus,
        grad_minus=grad_minus,
        q_plus=q_plus,
        p_plus=p_plus,
        logp_plus=logp_plus,
        grad_plus=grad_plus,
        q_proposed=q_proposed,
        logp_proposed=logp_proposed,
        grad_proposed=grad_proposed,
        n_valid=n_total,
        s_continue=s_continue,
        alpha=left.alpha + right.alpha,
        n_alpha=left.n_alpha + right.n_alpha,
        n_divergent=left.n_divergent + right.n_divergent,
    )


def find_reasonable_step_size(
    q: Array,
    logp: float,
    grad: Array,
    logp_grad_fn: LogpGradFn,
    rng: np.random.Generator,
) -> float:
    """Heuristic from NUTS paper for initial leapfrog step size."""

    step_size = 1.0
    p = rng.normal(size=q.shape[0])
    joint0 = joint_log_density(logp, p)

    q1, p1, logp1, grad1 = leapfrog(
        q=q,
        p=p,
        logp=logp,
        grad=grad,
        step_size=step_size,
        logp_grad_fn=logp_grad_fn,
    )
    _ = grad1  # not used, but kept explicit for readability.
    log_accept = joint_log_density(logp1, p1) - joint0
    direction = 1.0 if log_accept > math.log(0.5) else -1.0

    while True:
        threshold = -direction * math.log(2.0)
        if not (direction * log_accept > threshold):
            break
        step_size *= 2.0**direction

        if step_size < 1e-4 or step_size > 2.0:
            break

        q1, p1, logp1, _ = leapfrog(
            q=q,
            p=p,
            logp=logp,
            grad=grad,
            step_size=step_size,
            logp_grad_fn=logp_grad_fn,
        )
        log_accept = joint_log_density(logp1, p1) - joint0

    return float(np.clip(step_size, 1e-4, 2.0))


def effective_sample_size(x: Array, max_lag: int = 300) -> float:
    """Simple ESS estimator using positive autocorrelation truncation."""

    n = x.shape[0]
    if n < 4:
        return float(n)

    centered = x - np.mean(x)
    var = float(np.var(centered))
    if var <= 1e-12:
        return 0.0

    max_lag = min(max_lag, n - 1)
    rho_sum = 0.0

    for lag in range(1, max_lag + 1):
        c = float(np.dot(centered[:-lag], centered[lag:]) / (n - lag))
        rho = c / var
        if rho < 0.0:
            break
        rho_sum += rho

    tau = 1.0 + 2.0 * rho_sum
    return float(n / tau)


def run_nuts(logp_grad_fn: LogpGradFn, q_init: Array, cfg: NutsConfig) -> Tuple[Array, Dict[str, float]]:
    """Run one NUTS chain with warmup and return post-warmup samples."""

    rng = np.random.default_rng(cfg.seed)

    q = q_init.astype(float).copy()
    logp, grad = logp_grad_fn(q)
    dim = q.shape[0]

    step_size = find_reasonable_step_size(
        q=q,
        logp=logp,
        grad=grad,
        logp_grad_fn=logp_grad_fn,
        rng=rng,
    )

    mu = math.log(10.0 * step_size)
    step_size_bar = 1.0
    h_bar = 0.0

    gamma = 0.05
    t0 = 10.0
    kappa = 0.75

    total_iters = cfg.num_warmup + cfg.num_samples
    samples = np.zeros((cfg.num_samples, dim), dtype=float)

    accept_hist = np.zeros(total_iters, dtype=float)
    depth_hist = np.zeros(total_iters, dtype=int)
    divergent_total = 0

    sample_pos = 0

    for m in range(1, total_iters + 1):
        p0 = rng.normal(size=dim)
        initial_joint = joint_log_density(logp, p0)
        log_u = initial_joint - rng.exponential(scale=1.0)

        q_minus = q.copy()
        p_minus = p0.copy()
        logp_minus = logp
        grad_minus = grad.copy()

        q_plus = q.copy()
        p_plus = p0.copy()
        logp_plus = logp
        grad_plus = grad.copy()

        q_proposed = q.copy()
        logp_proposed = logp
        grad_proposed = grad.copy()

        n_valid = 1
        s_continue = 1
        depth = 0

        alpha = 0.0
        n_alpha = 0
        diverged_iter = 0

        while s_continue == 1 and depth < cfg.max_depth:
            direction = -1 if rng.uniform() < 0.5 else 1

            if direction == -1:
                subtree = build_tree(
                    q=q_minus,
                    p=p_minus,
                    logp=logp_minus,
                    grad=grad_minus,
                    log_u=log_u,
                    direction=direction,
                    depth=depth,
                    step_size=step_size,
                    initial_joint=initial_joint,
                    delta_max=cfg.delta_max,
                    rng=rng,
                    logp_grad_fn=logp_grad_fn,
                )
                q_minus, p_minus, logp_minus, grad_minus = (
                    subtree.q_minus,
                    subtree.p_minus,
                    subtree.logp_minus,
                    subtree.grad_minus,
                )
            else:
                subtree = build_tree(
                    q=q_plus,
                    p=p_plus,
                    logp=logp_plus,
                    grad=grad_plus,
                    log_u=log_u,
                    direction=direction,
                    depth=depth,
                    step_size=step_size,
                    initial_joint=initial_joint,
                    delta_max=cfg.delta_max,
                    rng=rng,
                    logp_grad_fn=logp_grad_fn,
                )
                q_plus, p_plus, logp_plus, grad_plus = (
                    subtree.q_plus,
                    subtree.p_plus,
                    subtree.logp_plus,
                    subtree.grad_plus,
                )

            if subtree.s_continue == 1:
                denom = n_valid + subtree.n_valid
                if denom > 0 and rng.uniform() < (subtree.n_valid / denom):
                    q_proposed = subtree.q_proposed.copy()
                    logp_proposed = subtree.logp_proposed
                    grad_proposed = subtree.grad_proposed.copy()

            n_valid += subtree.n_valid
            s_continue = int(
                subtree.s_continue
                and stop_criterion(
                    q_minus=q_minus,
                    q_plus=q_plus,
                    p_minus=p_minus,
                    p_plus=p_plus,
                )
            )

            alpha += subtree.alpha
            n_alpha += subtree.n_alpha
            diverged_iter += subtree.n_divergent

            depth += 1

        q = q_proposed
        logp = logp_proposed
        grad = grad_proposed

        mean_accept = alpha / max(1, n_alpha)
        accept_hist[m - 1] = mean_accept
        depth_hist[m - 1] = depth
        divergent_total += diverged_iter

        if m <= cfg.num_warmup:
            eta = 1.0 / (m + t0)
            h_bar = (1.0 - eta) * h_bar + eta * (cfg.target_accept - mean_accept)

            log_step = mu - (math.sqrt(m) / gamma) * h_bar
            step_size = math.exp(log_step)

            m_pow = m ** (-kappa)
            step_size_bar = math.exp(m_pow * log_step + (1.0 - m_pow) * math.log(step_size_bar))

            step_size = float(np.clip(step_size, 1e-4, 2.0))
            step_size_bar = float(np.clip(step_size_bar, 1e-4, 2.0))
        else:
            step_size = step_size_bar
            samples[sample_pos] = q
            sample_pos += 1

    stats = {
        "initial_step_size": float(find_reasonable_step_size(q_init, *logp_grad_fn(q_init), logp_grad_fn, np.random.default_rng(cfg.seed + 1))),
        "final_step_size": float(step_size_bar),
        "mean_accept_warmup": float(np.mean(accept_hist[: cfg.num_warmup])),
        "mean_accept_sampling": float(np.mean(accept_hist[cfg.num_warmup :])),
        "mean_tree_depth": float(np.mean(depth_hist[cfg.num_warmup :])),
        "divergent_total": float(divergent_total),
        "divergent_rate": float(divergent_total / total_iters),
    }
    return samples, stats


def main() -> None:
    mean_true = np.array([1.0, -1.0], dtype=float)
    cov_true = np.array([[1.0, 0.8], [0.8, 1.5]], dtype=float)

    logp_grad_fn = make_gaussian_target(mean=mean_true, cov=cov_true)

    cfg = NutsConfig(
        num_warmup=800,
        num_samples=1200,
        max_depth=8,
        target_accept=0.80,
        delta_max=1000.0,
        seed=20260407,
    )

    q_init = np.array([3.5, 3.0], dtype=float)

    samples, stats = run_nuts(logp_grad_fn=logp_grad_fn, q_init=q_init, cfg=cfg)

    mean_est = np.mean(samples, axis=0)
    cov_est = np.cov(samples, rowvar=False)

    mean_err = float(np.linalg.norm(mean_est - mean_true))
    cov_err = float(np.linalg.norm(cov_est - cov_true, ord="fro"))

    ess_dims = np.array([effective_sample_size(samples[:, d]) for d in range(samples.shape[1])])

    print("=== No-U-Turn Sampler (NUTS) MVP ===")
    print(f"Warmup steps: {cfg.num_warmup}, sampling steps: {cfg.num_samples}")
    print(f"Initial step size (heuristic): {stats['initial_step_size']:.5f}")
    print(f"Final step size (after adaptation): {stats['final_step_size']:.5f}")
    print(f"Mean accept (warmup): {stats['mean_accept_warmup']:.3f}")
    print(f"Mean accept (sampling): {stats['mean_accept_sampling']:.3f}")
    print(f"Mean tree depth (sampling): {stats['mean_tree_depth']:.2f}")
    print(f"Divergent transitions: {int(stats['divergent_total'])} ({stats['divergent_rate']:.4f})")

    print("\nTarget mean:", mean_true)
    print("Estimated mean:", np.round(mean_est, 4))
    print(f"Mean L2 error: {mean_err:.4f}")

    print("\nTarget covariance:\n", cov_true)
    print("Estimated covariance:\n", np.round(cov_est, 4))
    print(f"Covariance Frobenius error: {cov_err:.4f}")

    print("\nESS per dimension:", np.round(ess_dims, 1))
    print(f"Min ESS: {np.min(ess_dims):.1f}")

    assert mean_err < 0.35, f"Mean error too large: {mean_err:.4f}"
    assert cov_err < 0.55, f"Covariance error too large: {cov_err:.4f}"
    assert stats["mean_accept_sampling"] > 0.55, "Acceptance too low."
    assert np.min(ess_dims) > 120.0, "ESS too low; sampler quality is poor."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
