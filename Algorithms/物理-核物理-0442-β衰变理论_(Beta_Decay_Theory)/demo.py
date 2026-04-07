"""Minimal runnable MVP for beta decay theory.

This script demonstrates two core pieces of beta-decay modeling:
1) Exponential decay law and half-life estimation.
2) Allowed beta-spectrum shape and Monte-Carlo sampling.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.integrate import cumulative_trapezoid, trapezoid
from sklearn.linear_model import LinearRegression

M_E_MEV = 0.511  # Electron rest mass energy in MeV
ALPHA = 1.0 / 137.035999084  # Fine-structure constant


def beta_spectrum_shape(kinetic_energy_mev: np.ndarray, q_value_mev: float, z_daughter: int) -> np.ndarray:
    """Return unnormalized allowed beta spectrum shape S(T).

    S(T) ∝ F(Z, T) * p * W * (Q - T)^2
    where W = T + m_e, p = sqrt(W^2 - m_e^2).
    """
    t = np.asarray(kinetic_energy_mev, dtype=float)
    w = t + M_E_MEV
    p2 = np.clip(w**2 - M_E_MEV**2, 0.0, None)
    p = np.sqrt(p2)

    # Stable Coulomb correction approximation (Fermi function for beta- decay).
    p_safe = np.where(p > 1e-12, p, 1e-12)
    eta = ALPHA * z_daughter * w / p_safe
    two_pi_eta = 2.0 * math.pi * eta
    denom = -np.expm1(-two_pi_eta)  # 1 - exp(-x), stable near x=0
    fermi_factor = np.where(denom > 1e-14, two_pi_eta / denom, 1.0)

    phase_space = np.clip(q_value_mev - t, 0.0, None) ** 2
    shape = fermi_factor * p * w * phase_space
    return np.where((t >= 0.0) & (t <= q_value_mev), shape, 0.0)


def build_normalized_spectrum(q_value_mev: float, z_daughter: int, n_grid: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build grid-based normalized PDF and CDF for beta spectrum."""
    grid = np.linspace(0.0, q_value_mev, n_grid)
    shape = beta_spectrum_shape(grid, q_value_mev, z_daughter)

    area = trapezoid(shape, grid)
    if area <= 0.0:
        raise ValueError("Spectrum normalization failed: non-positive area.")

    pdf = shape / area
    cdf = cumulative_trapezoid(pdf, grid, initial=0.0)
    cdf /= cdf[-1]
    return grid, pdf, cdf, float(np.max(shape))


def sample_beta_energies(
    n_samples: int,
    q_value_mev: float,
    z_daughter: int,
    shape_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample beta kinetic energies by rejection sampling."""
    accepted_chunks: list[np.ndarray] = []
    total = 0

    while total < n_samples:
        batch = max(4096, int((n_samples - total) * 2.0))
        proposal_t = rng.uniform(0.0, q_value_mev, size=batch)
        proposal_y = rng.uniform(0.0, shape_max, size=batch)

        keep_mask = proposal_y < beta_spectrum_shape(proposal_t, q_value_mev, z_daughter)
        accepted = proposal_t[keep_mask]
        if accepted.size == 0:
            continue

        needed = n_samples - total
        if accepted.size > needed:
            accepted = accepted[:needed]

        accepted_chunks.append(accepted)
        total += accepted.size

    return np.concatenate(accepted_chunks)


def estimate_lambda_mle(decay_times_s: np.ndarray) -> float:
    """MLE for exponential distribution parameter lambda."""
    return 1.0 / float(np.mean(decay_times_s))


def survivor_counts(decay_times_s: np.ndarray, time_grid_s: np.ndarray) -> np.ndarray:
    """Compute N(t) = number of undecayed nuclei at each time point."""
    sorted_times = np.sort(decay_times_s)
    left_idx = np.searchsorted(sorted_times, time_grid_s, side="left")
    return sorted_times.size - left_idx


def estimate_lambda_linear_regression(time_grid_s: np.ndarray, survivors: np.ndarray) -> tuple[float, float, float]:
    """Fit log N(t) = log N0 - lambda t using sklearn LinearRegression."""
    mask = survivors > 0
    x = time_grid_s[mask].reshape(-1, 1)
    y = np.log(survivors[mask])

    model = LinearRegression()
    model.fit(x, y)

    lambda_hat = max(0.0, -float(model.coef_[0]))
    n0_hat = float(np.exp(model.intercept_))
    r2 = float(model.score(x, y))
    return lambda_hat, n0_hat, r2


def estimate_lambda_torch(time_grid_s: np.ndarray, survivors: np.ndarray, steps: int = 500, lr: float = 0.05) -> tuple[float, float]:
    """Estimate lambda with PyTorch optimization on log-survivor curve.

    We fit y = a + b * x_scaled with x standardized for numerical stability,
    then map slope back to physical lambda via slope = -lambda.
    """
    mask = survivors > 0
    x = torch.tensor(time_grid_s[mask], dtype=torch.float64)
    y = torch.tensor(np.log(survivors[mask]), dtype=torch.float64)

    x_mean = torch.mean(x)
    x_std = torch.std(x)
    x_scaled = (x - x_mean) / x_std

    a = torch.tensor(float(torch.mean(y)), dtype=torch.float64, requires_grad=True)
    b = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.Adam([a, b], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        pred = a + b * x_scaled
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()

    slope = float((b / x_std).detach().cpu().item())
    intercept = float((a - b * x_mean / x_std).detach().cpu().item())

    lambda_hat = max(0.0, -slope)
    n0_hat = float(math.exp(intercept))
    return lambda_hat, n0_hat


def main() -> None:
    seed = 20260407
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Example nuclide setup: neutron beta decay n -> p + e- + anti-nu_e
    q_value_mev = 0.782
    z_daughter = 1
    half_life_true_s = 610.0
    lambda_true = math.log(2.0) / half_life_true_s

    n_nuclei = 50_000
    n_electrons = 30_000

    decay_times_s = rng.exponential(scale=1.0 / lambda_true, size=n_nuclei)
    lambda_mle = estimate_lambda_mle(decay_times_s)

    time_grid_s = np.linspace(0.0, 4.0 * half_life_true_s, 80)
    survivors = survivor_counts(decay_times_s, time_grid_s)

    lambda_lr, n0_lr, r2_lr = estimate_lambda_linear_regression(time_grid_s, survivors)
    lambda_torch, n0_torch = estimate_lambda_torch(time_grid_s, survivors)

    grid, _pdf, cdf, shape_max = build_normalized_spectrum(q_value_mev, z_daughter)
    energies_mev = sample_beta_energies(n_electrons, q_value_mev, z_daughter, shape_max, rng)

    cdf_fn = lambda x: np.interp(np.asarray(x, dtype=float), grid, cdf, left=0.0, right=1.0)
    ks_result = stats.kstest(energies_mev, cdf_fn)

    methods = ["true", "mle", "linear_regression", "torch_optim"]
    lambdas = [lambda_true, lambda_mle, lambda_lr, lambda_torch]
    half_lives = [math.log(2.0) / x for x in lambdas]

    summary = pd.DataFrame(
        {
            "method": methods,
            "lambda_1_per_s": lambdas,
            "half_life_s": half_lives,
        }
    )
    summary["half_life_error_pct"] = (
        (summary["half_life_s"] - half_life_true_s).abs() / half_life_true_s * 100.0
    )

    print("=== Beta Decay Theory MVP ===")
    print(f"seed={seed}, n_nuclei={n_nuclei}, n_electrons={n_electrons}")
    print("\n[Half-life estimation summary]")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    print("\n[Survivor-curve fit diagnostics]")
    print(f"LinearRegression R^2: {r2_lr:.6f}")
    print(f"LinearRegression N0 estimate: {n0_lr:.2f}")
    print(f"PyTorch N0 estimate: {n0_torch:.2f}")

    print("\n[Beta spectrum diagnostics]")
    print(f"Sample mean kinetic energy (MeV): {float(np.mean(energies_mev)):.6f}")
    print(f"Sample std kinetic energy (MeV): {float(np.std(energies_mev)):.6f}")
    print(f"KS statistic vs theoretical CDF: {ks_result.statistic:.6f}")
    print(f"KS p-value: {ks_result.pvalue:.6f}")


if __name__ == "__main__":
    main()
