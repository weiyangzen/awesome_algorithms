"""Minimal runnable MVP for nuclear fission chain-reaction simulation.

This script models neutron-induced fission as a Galton-Watson branching process,
then estimates the effective multiplication factor k with multiple methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression

EV_TO_J = 1.602176634e-19


@dataclass(frozen=True)
class ReactorParams:
    """Parameters for a one-group, generation-wise fission model."""

    nu_prompt_mean: float
    p_fission: float
    p_leak: float
    energy_mev_per_fission: float

    @property
    def k_theory(self) -> float:
        """Expected multiplication factor per generation."""
        return self.p_fission * self.nu_prompt_mean * (1.0 - self.p_leak)


def simulate_one_generation(
    n_in: int,
    params: ReactorParams,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    """Simulate one neutron generation transition.

    Returns
    -------
    n_fissions, n_prompt_produced, n_out
    """
    if n_in < 0:
        raise ValueError("n_in must be non-negative")
    if n_in == 0:
        return 0, 0, 0

    n_fissions = int(rng.binomial(n_in, params.p_fission))

    # Sum of n_fissions Poisson(nu) random variables is Poisson(n_fissions*nu).
    lambda_prompt = n_fissions * params.nu_prompt_mean
    n_prompt_produced = int(rng.poisson(lambda_prompt))

    n_out = int(rng.binomial(n_prompt_produced, 1.0 - params.p_leak))
    return n_fissions, n_prompt_produced, n_out


def simulate_chain(
    initial_neutrons: int,
    n_generations: int,
    params: ReactorParams,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    """Run generation-by-generation chain-reaction simulation."""
    if initial_neutrons <= 0:
        raise ValueError("initial_neutrons must be positive")
    if n_generations <= 0:
        raise ValueError("n_generations must be positive")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    n_current = int(initial_neutrons)
    cumulative_fissions = 0

    for g in range(n_generations):
        n_fissions, n_prompt_produced, n_next = simulate_one_generation(n_current, params, rng)
        cumulative_fissions += n_fissions

        ratio = float(n_next / n_current) if n_current > 0 else float("nan")
        rows.append(
            {
                "generation": g,
                "n_in": n_current,
                "n_fissions": n_fissions,
                "n_prompt_produced": n_prompt_produced,
                "n_out": n_next,
                "ratio_nout_nin": ratio,
            }
        )

        n_current = n_next

    return pd.DataFrame(rows), cumulative_fissions


def estimate_k_moment(df: pd.DataFrame) -> float:
    """Estimate k by ratio of total offspring to total parents."""
    x = df["n_in"].to_numpy(dtype=float)
    y = df["n_out"].to_numpy(dtype=float)
    denom = float(np.sum(x))
    if denom <= 0.0:
        raise ValueError("No positive parent counts for moment estimation")
    return float(np.sum(y) / denom)


def estimate_k_log_linear(df: pd.DataFrame) -> tuple[float, float, float]:
    """Estimate k from log(N_g) = a + g*log(k)."""
    mask = df["n_in"] > 0
    x = df.loc[mask, ["generation"]].to_numpy(dtype=float)
    y = np.log(df.loc[mask, "n_in"].to_numpy(dtype=float))

    if x.shape[0] < 3:
        raise ValueError("Need at least 3 positive generations for regression")

    model = LinearRegression()
    model.fit(x, y)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    k_hat = float(math.exp(slope))
    r2 = float(model.score(x, y))
    return k_hat, intercept, r2


def fit_k_torch_poisson(
    df: pd.DataFrame,
    steps: int = 1200,
    lr: float = 0.08,
) -> tuple[float, float]:
    """Fit k via Poisson likelihood: N_{g+1} ~ Poisson(k * N_g)."""
    x_np = df["n_in"].to_numpy(dtype=np.float64)
    y_np = df["n_out"].to_numpy(dtype=np.float64)

    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64)

    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        k = torch.nn.functional.softplus(theta) + 1e-12
        lam = torch.clamp(k * x, min=1e-12)

        # Poisson negative log-likelihood without constant log(y!).
        loss = torch.mean(lam - y * torch.log(lam))
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        k_final = float((torch.nn.functional.softplus(theta) + 1e-12).item())
        lam_final = np.clip(k_final * x_np, 1e-12, None)
        nll_mean = float(np.mean(lam_final - y_np * np.log(lam_final)))

    return k_final, nll_mean


def poisson_pearson_test(df: pd.DataFrame, k_hat: float) -> tuple[float, float]:
    """Pearson chi-square goodness check for Poisson branching model."""
    x = df["n_in"].to_numpy(dtype=float)
    y = df["n_out"].to_numpy(dtype=float)

    lam = np.clip(k_hat * x, 1e-12, None)
    chi2_stat = float(np.sum((y - lam) ** 2 / lam))
    dof = max(1, len(df) - 1)
    chi2_ndf = chi2_stat / dof
    p_value = float(chi2.sf(chi2_stat, dof))
    return chi2_ndf, p_value


def classify_regime(k_value: float) -> str:
    """Classify neutron population regime from k."""
    if k_value > 1.005:
        return "supercritical"
    if k_value < 0.995:
        return "subcritical"
    return "near-critical"


def main() -> None:
    seed = 20260407
    torch.manual_seed(seed)

    params = ReactorParams(
        nu_prompt_mean=2.43,
        p_fission=0.64,
        p_leak=0.33,
        energy_mev_per_fission=200.0,
    )
    initial_neutrons = 4000
    n_generations = 22

    df, cumulative_fissions = simulate_chain(
        initial_neutrons=initial_neutrons,
        n_generations=n_generations,
        params=params,
        seed=seed,
    )

    k_theory = params.k_theory
    k_moment = estimate_k_moment(df)
    k_log, log_intercept, r2_log = estimate_k_log_linear(df)
    k_torch, nll_mean = fit_k_torch_poisson(df)
    chi2_ndf, chi2_p = poisson_pearson_test(df, k_torch)

    energy_j = cumulative_fissions * params.energy_mev_per_fission * 1e6 * EV_TO_J

    df = df.copy()
    df["expected_n_out_torch"] = np.clip(k_torch * df["n_in"].to_numpy(dtype=float), 1e-12, None)
    df["residual"] = df["n_out"] - df["expected_n_out_torch"]

    print("=== Nuclear Fission MVP (Branching Process) ===")
    print(f"seed={seed}")
    print(f"initial_neutrons={initial_neutrons}, generations={n_generations}")
    print(
        "params: "
        f"nu_prompt_mean={params.nu_prompt_mean}, "
        f"p_fission={params.p_fission}, "
        f"p_leak={params.p_leak}"
    )

    print("\n[Generation Table]")
    cols = [
        "generation",
        "n_in",
        "n_fissions",
        "n_prompt_produced",
        "n_out",
        "ratio_nout_nin",
        "expected_n_out_torch",
        "residual",
    ]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    print("\n[Global Diagnostics]")
    print(f"k_theory:          {k_theory:.6f}")
    print(f"k_moment:          {k_moment:.6f}")
    print(f"k_log_regression:  {k_log:.6f} (R^2={r2_log:.6f}, intercept={log_intercept:.6f})")
    print(f"k_torch_poisson:   {k_torch:.6f} (mean NLL={nll_mean:.6f})")
    print(f"Pearson chi2/ndf:  {chi2_ndf:.6f}")
    print(f"Pearson p-value:   {chi2_p:.6f}")
    print(f"Regime(theory):    {classify_regime(k_theory)}")
    print(f"Regime(torch):     {classify_regime(k_torch)}")
    print(f"cumulative_fissions: {cumulative_fissions}")
    print(f"released_energy:     {energy_j:.6e} J")

    # Minimal sanity checks for automated validation.
    assert (df["n_in"] >= 0).all()
    assert (df["n_out"] >= 0).all()
    assert cumulative_fissions > 0
    assert abs(k_torch - k_theory) < 0.20
    assert energy_j > 0.0

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
