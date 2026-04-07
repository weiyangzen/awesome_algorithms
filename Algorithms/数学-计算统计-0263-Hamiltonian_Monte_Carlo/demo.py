"""Minimal runnable MVP for Hamiltonian Monte Carlo (HMC)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

Array = np.ndarray
PotentialFn = Callable[[Array], float]
GradientFn = Callable[[Array], Array]


@dataclass
class HMCConfig:
    """Configuration for a basic HMC sampler."""

    step_size: float = 0.25
    leapfrog_steps: int = 20
    num_samples: int = 3000
    burn_in: int = 1200
    thin: int = 1
    seed: int = 20260407


@dataclass
class HMCResult:
    """Sampling output and basic diagnostics."""

    samples: Array
    acceptance_rate: float
    mean_abs_energy_error: float
    max_abs_energy_error: float


def make_gaussian_target(mean: Array, covariance: Array) -> Tuple[PotentialFn, GradientFn]:
    """Create potential U(q) and grad U(q) for a Gaussian target N(mean, covariance)."""
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    if mean.ndim != 1:
        raise ValueError("mean must be a 1D vector")
    if covariance.shape != (mean.size, mean.size):
        raise ValueError("covariance shape does not match mean dimension")
    if not np.allclose(covariance, covariance.T, atol=1e-12):
        raise ValueError("covariance must be symmetric")

    # Cholesky is both a PSD check and numerically stable factorization.
    np.linalg.cholesky(covariance)
    precision = np.linalg.inv(covariance)

    def potential(q: Array) -> float:
        delta = q - mean
        value = 0.5 * float(delta @ precision @ delta)
        if not np.isfinite(value):
            raise ValueError("potential is not finite")
        return value

    def grad_potential(q: Array) -> Array:
        grad = precision @ (q - mean)
        if not np.all(np.isfinite(grad)):
            raise ValueError("gradient contains non-finite values")
        return grad

    return potential, grad_potential


def validate_hmc_inputs(config: HMCConfig, initial_position: Array, mass_matrix: Array) -> None:
    """Validate basic HMC configuration and tensor shapes."""
    if config.step_size <= 0:
        raise ValueError("step_size must be positive")
    if config.leapfrog_steps < 1:
        raise ValueError("leapfrog_steps must be >= 1")
    if config.num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if config.burn_in < 0:
        raise ValueError("burn_in must be >= 0")
    if config.thin < 1:
        raise ValueError("thin must be >= 1")

    if initial_position.ndim != 1:
        raise ValueError("initial_position must be 1D")

    dim = initial_position.size
    if mass_matrix.shape != (dim, dim):
        raise ValueError("mass_matrix has incompatible shape")
    if not np.allclose(mass_matrix, mass_matrix.T, atol=1e-12):
        raise ValueError("mass_matrix must be symmetric")

    np.linalg.cholesky(mass_matrix)


def leapfrog(
    position: Array,
    momentum: Array,
    step_size: float,
    leapfrog_steps: int,
    grad_potential: GradientFn,
    mass_inv: Array,
) -> Tuple[Array, Array]:
    """Perform leapfrog integration for Hamiltonian dynamics."""
    q = position.copy()
    p = momentum.copy()

    grad = grad_potential(q)
    p -= 0.5 * step_size * grad

    for step in range(leapfrog_steps):
        q += step_size * (mass_inv @ p)
        grad = grad_potential(q)
        if step != leapfrog_steps - 1:
            p -= step_size * grad

    p -= 0.5 * step_size * grad
    return q, p


def kinetic_energy(momentum: Array, mass_inv: Array) -> float:
    """K(p) = 1/2 p^T M^{-1} p."""
    value = 0.5 * float(momentum @ (mass_inv @ momentum))
    if not np.isfinite(value):
        raise ValueError("kinetic energy is not finite")
    return value


def hmc_sample(
    potential: PotentialFn,
    grad_potential: GradientFn,
    initial_position: Array,
    config: HMCConfig,
    mass_matrix: Array | None = None,
) -> HMCResult:
    """Run Hamiltonian Monte Carlo with fixed (epsilon, L)."""
    q = np.asarray(initial_position, dtype=float).copy()
    dim = q.size

    if mass_matrix is None:
        mass_matrix = np.eye(dim, dtype=float)
    else:
        mass_matrix = np.asarray(mass_matrix, dtype=float)

    validate_hmc_inputs(config, q, mass_matrix)

    rng = np.random.default_rng(config.seed)
    mass_cholesky = np.linalg.cholesky(mass_matrix)
    mass_inv = np.linalg.inv(mass_matrix)

    total_steps = config.burn_in + config.num_samples * config.thin
    samples = np.empty((config.num_samples, dim), dtype=float)
    sample_idx = 0

    accept_count = 0
    abs_energy_errors = []

    for t in range(total_steps):
        p0 = mass_cholesky @ rng.normal(size=dim)
        current_h = potential(q) + kinetic_energy(p0, mass_inv)

        q_prop, p_prop = leapfrog(
            position=q,
            momentum=p0,
            step_size=config.step_size,
            leapfrog_steps=config.leapfrog_steps,
            grad_potential=grad_potential,
            mass_inv=mass_inv,
        )
        p_prop = -p_prop

        proposed_h = potential(q_prop) + kinetic_energy(p_prop, mass_inv)
        delta_h = proposed_h - current_h
        abs_energy_errors.append(abs(delta_h))

        if np.log(rng.uniform()) < -delta_h:
            q = q_prop
            accept_count += 1

        if t >= config.burn_in and (t - config.burn_in) % config.thin == 0:
            samples[sample_idx] = q
            sample_idx += 1

    return HMCResult(
        samples=samples,
        acceptance_rate=accept_count / total_steps,
        mean_abs_energy_error=float(np.mean(abs_energy_errors)),
        max_abs_energy_error=float(np.max(abs_energy_errors)),
    )


def estimate_ess_per_dimension(samples: Array, max_lag: int = 200) -> Array:
    """Estimate effective sample size (ESS) per dimension via positive autocorrelation sum."""
    n, d = samples.shape
    ess = np.empty(d, dtype=float)
    max_lag = min(max_lag, n - 1)

    for j in range(d):
        x = samples[:, j] - np.mean(samples[:, j])
        variance = float(np.dot(x, x) / n)
        if variance <= 0:
            ess[j] = float(n)
            continue

        rho_sum = 0.0
        for lag in range(1, max_lag + 1):
            autocov = float(np.dot(x[:-lag], x[lag:]) / (n - lag))
            rho = autocov / variance
            if rho <= 0:
                break
            rho_sum += rho

        tau = 1.0 + 2.0 * rho_sum
        ess[j] = n / tau

    return ess


def run_demo() -> None:
    """Run HMC on a correlated 2D Gaussian and print diagnostics."""
    true_mean = np.array([1.5, -0.8], dtype=float)
    true_cov = np.array([[1.0, 0.75], [0.75, 1.6]], dtype=float)

    potential, grad_potential = make_gaussian_target(true_mean, true_cov)

    config = HMCConfig(
        step_size=0.24,
        leapfrog_steps=18,
        num_samples=3000,
        burn_in=1200,
        thin=1,
        seed=20260407,
    )

    initial_position = np.array([4.0, -3.0], dtype=float)

    result = hmc_sample(
        potential=potential,
        grad_potential=grad_potential,
        initial_position=initial_position,
        config=config,
        mass_matrix=true_cov,
    )

    samples = result.samples
    empirical_mean = np.mean(samples, axis=0)
    empirical_cov = np.cov(samples, rowvar=False, ddof=1)

    mean_error_l2 = float(np.linalg.norm(empirical_mean - true_mean))
    cov_error_fro = float(np.linalg.norm(empirical_cov - true_cov, ord="fro"))
    ess = estimate_ess_per_dimension(samples)

    print("=== Hamiltonian Monte Carlo MVP ===")
    print(f"seed={config.seed}")
    print(f"step_size={config.step_size}, leapfrog_steps={config.leapfrog_steps}")
    print(f"num_samples={config.num_samples}, burn_in={config.burn_in}, thin={config.thin}")
    print()
    print("Target mean:", true_mean)
    print("Empirical mean:", np.round(empirical_mean, 4))
    print(f"L2(mean error): {mean_error_l2:.6f}")
    print()
    print("Target covariance:\n", true_cov)
    print("Empirical covariance:\n", np.round(empirical_cov, 4))
    print(f"Frobenius(cov error): {cov_error_fro:.6f}")
    print()
    print(f"Acceptance rate: {result.acceptance_rate:.4f}")
    print(f"Mean |delta H|: {result.mean_abs_energy_error:.6f}")
    print(f"Max  |delta H|: {result.max_abs_energy_error:.6f}")
    print("ESS per dimension:", np.round(ess, 1))

    # Lightweight quality gates for deterministic regression checks.
    if not (0.45 <= result.acceptance_rate <= 0.999):
        raise RuntimeError("Unexpected acceptance rate; tune step_size/leapfrog_steps")
    if mean_error_l2 > 0.25:
        raise RuntimeError("Mean estimate too far from target; check sampler configuration")
    if cov_error_fro > 0.35:
        raise RuntimeError("Covariance estimate too far from target; check sampler configuration")

    print("Status: PASS")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
