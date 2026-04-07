"""Monte Carlo Method MVP: high-dimensional integral estimation.

This demo estimates the d-dimensional integral:
    I_d = ∫_[0,1]^d exp(-||x||^2) dx

We provide:
1) Standard Monte Carlo (i.i.d. uniform samples).
2) Antithetic variates (variance reduction).
3) A reproducible benchmark table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MCConfig:
    """Configuration for one Monte Carlo estimation run."""

    dim: int
    n_samples: int
    seed: int


def integrand(points: np.ndarray) -> np.ndarray:
    """Evaluate f(x) = exp(-||x||^2) for points of shape (n, dim)."""
    return np.exp(-np.sum(points * points, axis=1))


def exact_integral(dim: int) -> float:
    """Closed-form exact value of the test integral."""
    one_d = 0.5 * math.sqrt(math.pi) * math.erf(1.0)
    return one_d**dim


def standard_mc(config: MCConfig) -> tuple[float, float]:
    """Plain Monte Carlo using i.i.d. uniform samples on [0,1]^d.

    Returns:
        (estimate, sem)
    """
    rng = np.random.default_rng(config.seed)
    points = rng.random((config.n_samples, config.dim))
    fx = integrand(points)
    estimate = float(fx.mean())
    sem = float(fx.std(ddof=1) / np.sqrt(config.n_samples))
    return estimate, sem


def antithetic_mc(config: MCConfig) -> tuple[float, float]:
    """Antithetic variates Monte Carlo.

    For each u ~ U([0,1]^d), we pair with (1-u). For monotone integrands this
    typically reduces variance. We estimate SEM from pair-means.
    """
    m = config.n_samples // 2
    if m < 1:
        raise ValueError("n_samples must be >= 2 for antithetic sampling")

    rng = np.random.default_rng(config.seed)
    u = rng.random((m, config.dim))
    fu = integrand(u)
    fv = integrand(1.0 - u)
    pair_means = 0.5 * (fu + fv)

    estimate = float(pair_means.mean())
    sem = float(pair_means.std(ddof=1) / np.sqrt(m))
    return estimate, sem


def run_convergence_study(dim: int, n_grid: list[int], seed: int) -> pd.DataFrame:
    """Run both MC variants on a sample-size grid."""
    exact = exact_integral(dim)
    rows: list[dict[str, float | int | str]] = []

    for i, n in enumerate(n_grid):
        base_seed = seed + 1000 * i
        std_est, std_sem = standard_mc(MCConfig(dim=dim, n_samples=n, seed=base_seed))
        anti_est, anti_sem = antithetic_mc(
            MCConfig(dim=dim, n_samples=n, seed=base_seed + 1)
        )

        rows.append(
            {
                "method": "standard",
                "n_samples": n,
                "estimate": std_est,
                "abs_error": abs(std_est - exact),
                "sem": std_sem,
                "ci95_low": std_est - 1.96 * std_sem,
                "ci95_high": std_est + 1.96 * std_sem,
            }
        )
        rows.append(
            {
                "method": "antithetic",
                "n_samples": n,
                "estimate": anti_est,
                "abs_error": abs(anti_est - exact),
                "sem": anti_sem,
                "ci95_low": anti_est - 1.96 * anti_sem,
                "ci95_high": anti_est + 1.96 * anti_sem,
            }
        )

    return pd.DataFrame(rows)


def benchmark_variance_reduction(
    dim: int, n_samples: int, n_trials: int, seed: int
) -> pd.DataFrame:
    """Empirically compare estimator dispersion across repeated runs."""
    exact = exact_integral(dim)
    method_rows: list[dict[str, float | int | str]] = []

    for method in ("standard", "antithetic"):
        estimates = []
        for t in range(n_trials):
            cfg = MCConfig(dim=dim, n_samples=n_samples, seed=seed + t)
            est, _ = standard_mc(cfg) if method == "standard" else antithetic_mc(cfg)
            estimates.append(est)

        arr = np.asarray(estimates, dtype=float)
        err = arr - exact
        method_rows.append(
            {
                "method": method,
                "n_trials": n_trials,
                "mean_estimate": float(arr.mean()),
                "bias": float(err.mean()),
                "std_of_estimate": float(arr.std(ddof=1)),
                "rmse": float(np.sqrt(np.mean(err * err))),
            }
        )

    return pd.DataFrame(method_rows)


def main() -> None:
    dim = 6
    base_seed = 20260407
    n_grid = [2_000, 5_000, 20_000, 100_000]

    exact = exact_integral(dim)
    conv_df = run_convergence_study(dim=dim, n_grid=n_grid, seed=base_seed)
    bench_df = benchmark_variance_reduction(
        dim=dim, n_samples=5_000, n_trials=80, seed=base_seed + 77
    )

    print("Monte Carlo Method MVP")
    print("-" * 72)
    print(f"Target integral: I_d = ∫_[0,1]^{dim} exp(-||x||^2) dx")
    print(f"Exact value:     {exact:.10f}")
    print()
    print("[Convergence Study]")
    print(conv_df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print()
    print("[Variance Reduction Benchmark]")
    print(bench_df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    # Lightweight correctness checks for reproducible validation.
    max_n_rows = conv_df[conv_df["n_samples"] == max(n_grid)]
    best_abs_error = float(max_n_rows["abs_error"].min())
    assert best_abs_error < 2e-3, "Largest-sample estimate is unexpectedly inaccurate."

    std_rmse = float(bench_df.loc[bench_df["method"] == "standard", "rmse"].iloc[0])
    anti_rmse = float(bench_df.loc[bench_df["method"] == "antithetic", "rmse"].iloc[0])
    assert anti_rmse < std_rmse, "Antithetic variates should reduce RMSE in this setup."

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
