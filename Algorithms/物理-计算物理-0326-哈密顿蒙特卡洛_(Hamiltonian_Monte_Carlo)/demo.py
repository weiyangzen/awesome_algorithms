"""Minimal runnable MVP for Hamiltonian Monte Carlo (HMC).

This demo samples a 2D Boltzmann/Gaussian target induced by a coupled
harmonic potential. The HMC transition is implemented explicitly with
momentum refresh, leapfrog integration, and Metropolis correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HMCConfig:
    """Configuration for a single HMC run."""

    n_steps: int = 24_000
    burn_in: int = 4_000
    step_size: float = 0.40
    n_leapfrog: int = 12
    q0: tuple[float, float] = (1.2, -0.8)
    mass_diag: tuple[float, float] = (1.0, 1.0)
    seed: int = 2026


def stiffness_matrix() -> np.ndarray:
    """Return the positive-definite stiffness matrix K of U(q)=0.5*q^T*K*q."""

    return np.array([[3.0, 0.8], [0.8, 1.7]], dtype=float)


def potential_energy(q: np.ndarray, k_mat: np.ndarray) -> float:
    """Potential U(q) = 0.5 * q^T K q."""

    return 0.5 * float(q.T @ k_mat @ q)


def grad_potential(q: np.ndarray, k_mat: np.ndarray) -> np.ndarray:
    """Gradient of the quadratic potential: grad U = K q."""

    return k_mat @ q


def kinetic_energy(p: np.ndarray, mass_diag: np.ndarray) -> float:
    """Kinetic energy K(p)=0.5*p^T*M^{-1}*p for diagonal mass M."""

    inv_mass = 1.0 / mass_diag
    return 0.5 * float(np.sum((p**2) * inv_mass))


def hamiltonian(q: np.ndarray, p: np.ndarray, k_mat: np.ndarray, mass_diag: np.ndarray) -> float:
    """Total Hamiltonian H(q,p)=U(q)+K(p)."""

    return potential_energy(q, k_mat) + kinetic_energy(p, mass_diag)


def sample_momentum(rng: np.random.Generator, mass_diag: np.ndarray) -> np.ndarray:
    """Draw momentum from N(0, M) where M is diagonal."""

    return rng.normal(loc=0.0, scale=np.sqrt(mass_diag), size=mass_diag.shape[0])


def leapfrog(
    q: np.ndarray,
    p: np.ndarray,
    step_size: float,
    n_leapfrog: int,
    k_mat: np.ndarray,
    mass_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform one reversible/symplectic leapfrog trajectory."""

    q_new = q.copy()
    p_new = p.copy()
    inv_mass = 1.0 / mass_diag

    p_new -= 0.5 * step_size * grad_potential(q_new, k_mat)
    for i in range(n_leapfrog):
        q_new += step_size * inv_mass * p_new
        if i != n_leapfrog - 1:
            p_new -= step_size * grad_potential(q_new, k_mat)
    p_new -= 0.5 * step_size * grad_potential(q_new, k_mat)

    # Momentum flip keeps the proposal map exactly reversible.
    p_new = -p_new
    return q_new, p_new


def hmc_sample(config: HMCConfig) -> tuple[np.ndarray, float, float]:
    """Run an HMC chain and return post-burn samples and diagnostics."""

    if config.burn_in >= config.n_steps:
        raise ValueError("burn_in must be smaller than n_steps")
    if config.step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if config.n_leapfrog <= 0:
        raise ValueError("n_leapfrog must be positive")

    k_mat = stiffness_matrix()
    mass_diag = np.asarray(config.mass_diag, dtype=float)
    rng = np.random.default_rng(config.seed)

    q = np.asarray(config.q0, dtype=float)
    chain = np.empty((config.n_steps, q.size), dtype=float)
    abs_delta_h = np.empty(config.n_steps, dtype=float)
    accepted = 0

    for t in range(config.n_steps):
        p0 = sample_momentum(rng, mass_diag)
        q_prop, p_prop = leapfrog(
            q=q,
            p=p0,
            step_size=config.step_size,
            n_leapfrog=config.n_leapfrog,
            k_mat=k_mat,
            mass_diag=mass_diag,
        )

        h_now = hamiltonian(q, p0, k_mat, mass_diag)
        h_prop = hamiltonian(q_prop, p_prop, k_mat, mass_diag)
        delta_h = h_prop - h_now

        log_alpha = min(0.0, -delta_h)
        if np.log(rng.random()) < log_alpha:
            q = q_prop
            accepted += 1

        chain[t] = q
        abs_delta_h[t] = abs(delta_h)

    samples = chain[config.burn_in :]
    acceptance_rate = accepted / config.n_steps
    mean_abs_delta_h = float(np.mean(abs_delta_h[config.burn_in :]))
    return samples, acceptance_rate, mean_abs_delta_h


def integrated_autocorrelation_time_1d(series: np.ndarray, max_lag: int = 2_000) -> float:
    """Estimate IACT using positive-sequence truncation."""

    centered = series - np.mean(series)
    var = float(np.var(centered))
    if var <= 0.0:
        return 1.0

    n = centered.size
    max_lag = min(max_lag, n - 1)
    tau = 1.0

    for lag in range(1, max_lag + 1):
        acov = float(np.dot(centered[:-lag], centered[lag:]) / (n - lag))
        rho = acov / var
        if rho <= 0.0:
            break
        tau += 2.0 * rho

    return max(tau, 1.0)


def effective_sample_size_per_dim(samples: np.ndarray, max_lag: int = 2_000) -> np.ndarray:
    """Compute ESS for each dimension from IACT."""

    n = samples.shape[0]
    ess = np.empty(samples.shape[1], dtype=float)
    for d in range(samples.shape[1]):
        tau = integrated_autocorrelation_time_1d(samples[:, d], max_lag=max_lag)
        ess[d] = n / tau
    return ess


def reference_gaussian_stats(k_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For pi(q)∝exp(-0.5*q^T*K*q), mean=0 and cov=K^{-1}."""

    mean = np.zeros(k_mat.shape[0], dtype=float)
    cov = np.linalg.inv(k_mat)
    return mean, cov


def build_report(config: HMCConfig) -> tuple[pd.DataFrame, dict[str, float | bool], bool]:
    """Run HMC, compare with analytic reference, and produce a validation report."""

    samples, acceptance_rate, mean_abs_delta_h = hmc_sample(config)

    mean_est = np.mean(samples, axis=0)
    cov_est = np.cov(samples, rowvar=False, bias=False)

    k_mat = stiffness_matrix()
    mean_ref, cov_ref = reference_gaussian_stats(k_mat)

    rows = []
    metric_pairs = [
        ("mean_q1", float(mean_est[0]), float(mean_ref[0])),
        ("mean_q2", float(mean_est[1]), float(mean_ref[1])),
        ("cov_11", float(cov_est[0, 0]), float(cov_ref[0, 0])),
        ("cov_12", float(cov_est[0, 1]), float(cov_ref[0, 1])),
        ("cov_22", float(cov_est[1, 1]), float(cov_ref[1, 1])),
    ]
    for name, est, ref in metric_pairs:
        abs_err = abs(est - ref)
        rel_err = abs_err / abs(ref) if abs(ref) > 1e-12 else np.nan
        rows.append(
            {
                "metric": name,
                "estimate": est,
                "reference": ref,
                "abs_error": abs_err,
                "rel_error": rel_err,
            }
        )

    table = pd.DataFrame(rows)

    ess = effective_sample_size_per_dim(samples)
    mean_norm = float(np.linalg.norm(mean_est))
    cov_rel_error = float(np.linalg.norm(cov_est - cov_ref) / np.linalg.norm(cov_ref))

    checks: dict[str, float | bool] = {
        "acceptance_rate": acceptance_rate,
        "mean_abs_delta_h": mean_abs_delta_h,
        "ess_q1": float(ess[0]),
        "ess_q2": float(ess[1]),
        "min_ess": float(np.min(ess)),
        "mean_norm": mean_norm,
        "cov_rel_error": cov_rel_error,
        "pass_acceptance": 0.60 <= acceptance_rate <= 0.98,
        "pass_energy_error": mean_abs_delta_h < 0.10,
        "pass_mean": mean_norm < 0.07,
        "pass_cov": cov_rel_error < 0.08,
        "pass_ess": float(np.min(ess)) > 1_500.0,
    }

    passed = bool(
        checks["pass_acceptance"]
        and checks["pass_energy_error"]
        and checks["pass_mean"]
        and checks["pass_cov"]
        and checks["pass_ess"]
    )
    return table, checks, passed


def main() -> None:
    config = HMCConfig()
    table, checks, passed = build_report(config)

    pd.set_option("display.float_format", lambda v: f"{v: .6f}")

    print("=== Hamiltonian Monte Carlo MVP ===")
    print(
        f"n_steps={config.n_steps}, burn_in={config.burn_in}, "
        f"step_size={config.step_size}, n_leapfrog={config.n_leapfrog}, seed={config.seed}"
    )
    print()

    print("Moment/Covariance summary:")
    print(table.to_string(index=False))
    print()

    print("Diagnostics:")
    print(f"acceptance_rate={checks['acceptance_rate']:.4f}")
    print(f"mean_abs_delta_h={checks['mean_abs_delta_h']:.6f}")
    print(f"mean_vector_l2={checks['mean_norm']:.6f}")
    print(f"cov_relative_error={checks['cov_rel_error']:.6f}")
    print(f"ESS(q1)={checks['ess_q1']:.1f}, ESS(q2)={checks['ess_q2']:.1f}")
    print()

    print("Validation checks:")
    print(f"- acceptance in [0.60, 0.98]: {checks['pass_acceptance']}")
    print(f"- mean |delta_H| < 0.10: {checks['pass_energy_error']}")
    print(f"- ||E[q]||_2 < 0.07: {checks['pass_mean']}")
    print(f"- relative covariance error < 8%: {checks['pass_cov']}")
    print(f"- min ESS > 1500: {checks['pass_ess']}")
    print()

    print(f"Validation: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
