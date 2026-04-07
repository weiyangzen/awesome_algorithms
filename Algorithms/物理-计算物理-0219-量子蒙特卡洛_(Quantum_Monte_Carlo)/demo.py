"""Quantum Monte Carlo MVP: Variational Monte Carlo for a 1D harmonic oscillator.

Model (dimensionless units with m=omega=hbar=1):
    H = -1/2 d^2/dx^2 + 1/2 x^2
Trial wavefunction:
    psi_alpha(x) = exp(-alpha * x^2), alpha > 0
Sampling target:
    |psi_alpha(x)|^2 = exp(-2 * alpha * x^2)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VMCConfig:
    """Configuration for a single VMC chain."""

    n_steps: int
    burn_in: int
    thin: int
    proposal_std: float
    x0: float = 0.0


def local_energy(x: np.ndarray | float, alpha: float) -> np.ndarray | float:
    """Local energy E_L(x) = (H psi_alpha)/psi_alpha for the oscillator."""

    return alpha + (0.5 - 2.0 * alpha * alpha) * np.square(x)


def analytic_variational_energy(alpha: float) -> float:
    """Exact variational expectation for this trial family.

    E(alpha) = alpha / 2 + 1 / (8 * alpha)
    """

    return 0.5 * alpha + 1.0 / (8.0 * alpha)


def metropolis_vmc(
    alpha: float,
    config: VMCConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run Metropolis-Hastings to sample |psi_alpha|^2 and estimate local energies."""

    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    if config.n_steps <= config.burn_in:
        raise ValueError("n_steps must be larger than burn_in")
    if config.thin <= 0:
        raise ValueError("thin must be positive")
    if config.proposal_std <= 0.0:
        raise ValueError("proposal_std must be positive")

    x = float(config.x0)
    accepted = 0
    samples: list[float] = []
    energies: list[float] = []

    for step in range(config.n_steps):
        x_new = x + rng.normal(0.0, config.proposal_std)

        # Acceptance ratio for target pi(x) proportional to exp(-2 * alpha * x^2).
        log_ratio = -2.0 * alpha * (x_new * x_new - x * x)

        if log_ratio >= 0.0 or rng.random() < np.exp(log_ratio):
            x = x_new
            accepted += 1

        if step >= config.burn_in and (step - config.burn_in) % config.thin == 0:
            samples.append(x)
            energies.append(float(local_energy(x, alpha)))

    return np.asarray(samples), np.asarray(energies), accepted / config.n_steps


def estimate_mean_sem_tau(values: np.ndarray) -> tuple[float, float, float]:
    """Estimate mean, SEM, and integrated autocorrelation time.

    Uses a simple positive-sequence estimate for autocorrelation time.
    """

    n = int(values.size)
    if n == 0:
        raise ValueError("empty sample array")
    if n == 1:
        return float(values[0]), float("nan"), 1.0

    mean = float(values.mean())
    centered = values - mean
    var = float(np.dot(centered, centered) / (n - 1))

    if var == 0.0:
        return mean, 0.0, 1.0

    max_lag = min(300, n // 5)
    if max_lag < 1:
        tau_int = 1.0
    else:
        rho_sum = 0.0
        for lag in range(1, max_lag + 1):
            cov = float(np.dot(centered[:-lag], centered[lag:]) / (n - lag))
            rho = cov / var
            if rho <= 0.0:
                break
            rho_sum += rho
        tau_int = max(1.0, 1.0 + 2.0 * rho_sum)

    effective_n = max(1.0, n / tau_int)
    sem = float(np.sqrt(var / effective_n))
    return mean, sem, tau_int


def run_alpha(alpha: float, config: VMCConfig, rng: np.random.Generator) -> dict[str, float]:
    """Run one alpha and return summary metrics."""

    _, energies, accept_rate = metropolis_vmc(alpha=alpha, config=config, rng=rng)
    energy_mean, energy_sem, tau_int = estimate_mean_sem_tau(energies)
    return {
        "alpha": alpha,
        "energy": energy_mean,
        "sem": energy_sem,
        "accept_rate": accept_rate,
        "tau_int": tau_int,
        "analytic_energy": analytic_variational_energy(alpha),
    }


def main() -> None:
    rng = np.random.default_rng(20260407)

    # Coarse search over variational parameter alpha.
    scan_config = VMCConfig(
        n_steps=12_000,
        burn_in=2_000,
        thin=4,
        proposal_std=1.0,
        x0=0.0,
    )
    alphas = np.linspace(0.3, 0.8, 11)

    scan_rows = [run_alpha(alpha=float(alpha), config=scan_config, rng=rng) for alpha in alphas]
    best_row = min(scan_rows, key=lambda row: row["energy"])

    # Refine with a longer run at the best alpha from the coarse scan.
    refine_config = VMCConfig(
        n_steps=60_000,
        burn_in=10_000,
        thin=5,
        proposal_std=1.0,
        x0=0.0,
    )
    final_row = run_alpha(alpha=best_row["alpha"], config=refine_config, rng=rng)

    exact_ground = 0.5

    print("VMC MVP: 1D Harmonic Oscillator")
    print("Hamiltonian: H = -1/2 d^2/dx^2 + 1/2 x^2")
    print("trial psi_alpha(x) = exp(-alpha * x^2)")
    print()
    print("Coarse alpha scan:")
    print("  alpha    E_MC +/- SEM      E_analytic    acc_rate    tau_int")
    for row in scan_rows:
        print(
            "  "
            f"{row['alpha']:.3f}    "
            f"{row['energy']:.6f} +/- {row['sem']:.6f}    "
            f"{row['analytic_energy']:.6f}    "
            f"{row['accept_rate']:.3f}      "
            f"{row['tau_int']:.2f}"
        )

    print()
    print("Refined run at best alpha from scan:")
    print(f"  alpha* = {final_row['alpha']:.3f}")
    print(f"  E_MC   = {final_row['energy']:.6f} +/- {final_row['sem']:.6f}")
    print(f"  E_exact(ground) = {exact_ground:.6f}")
    print(f"  |E_MC - E_exact| = {abs(final_row['energy'] - exact_ground):.6f}")
    print(f"  acceptance_rate = {final_row['accept_rate']:.3f}")
    print(f"  tau_int ~ {final_row['tau_int']:.2f}")


if __name__ == "__main__":
    main()
