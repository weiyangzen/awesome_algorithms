"""Variance reduction techniques: minimal runnable MVP.

Task context:
- UID: MATH-0140
- Name: 方差缩减技术

This demo estimates
    mu = E[exp(X)],  X ~ N(0, 1)
where the exact value is exp(1/2).

Implemented estimators:
1) Crude Monte Carlo
2) Antithetic variates
3) Control variates (control C=X, E[C]=0)
4) Importance sampling (proposal N(theta, 1))
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

import numpy as np


Estimator = Callable[[int, np.random.Generator], float]


def target_true_value() -> float:
    """Exact value of E[exp(X)] for X ~ N(0, 1)."""
    return math.exp(0.5)


def crude_mc(n_samples: int, rng: np.random.Generator) -> float:
    x = rng.standard_normal(n_samples)
    return float(np.exp(x).mean())


def antithetic_mc(n_samples: int, rng: np.random.Generator) -> float:
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2 for antithetic variates")

    n_pairs = n_samples // 2
    z = rng.standard_normal(n_pairs)
    y_pair_avg = 0.5 * (np.exp(z) + np.exp(-z))
    estimate = float(y_pair_avg.mean())

    # If sample count is odd, spend one extra function call to keep API simple.
    if n_samples % 2 == 1:
        x_extra = rng.standard_normal(1)
        y_extra = float(np.exp(x_extra)[0])
        estimate = (estimate * n_pairs + y_extra) / (n_pairs + 1)

    return estimate


def control_variate_mc(n_samples: int, rng: np.random.Generator) -> Tuple[float, float]:
    x = rng.standard_normal(n_samples)
    y = np.exp(x)

    # Control variable C = X with known E[C] = 0.
    c = x
    var_c = float(np.var(c, ddof=1))
    cov_yc = float(np.cov(y, c, ddof=1)[0, 1])
    beta_hat = cov_yc / var_c if var_c > 0 else 0.0

    y_cv = y - beta_hat * c
    return float(y_cv.mean()), beta_hat


def importance_sampling_mc(
    n_samples: int,
    rng: np.random.Generator,
    theta: float = 0.7,
) -> float:
    """Importance sampling with proposal q = N(theta, 1)."""
    x = rng.normal(loc=theta, scale=1.0, size=n_samples)

    # log(p/q) for p=N(0,1), q=N(theta,1)
    log_weight = -theta * x + 0.5 * theta * theta
    weighted_values = np.exp(x + log_weight)
    return float(weighted_values.mean())


def run_trials(
    estimator: Estimator,
    n_trials: int,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    master_rng = np.random.default_rng(seed)
    estimates = np.empty(n_trials, dtype=float)

    for i in range(n_trials):
        trial_seed = int(master_rng.integers(1, 2**63 - 1))
        trial_rng = np.random.default_rng(trial_seed)
        estimates[i] = estimator(n_samples, trial_rng)

    return estimates


def summarize(
    method_name: str,
    estimates: np.ndarray,
    true_value: float,
    baseline_variance: float,
) -> Dict[str, float]:
    mean_est = float(np.mean(estimates))
    variance_est = float(np.var(estimates, ddof=1))
    bias = mean_est - true_value
    mse = float(np.mean((estimates - true_value) ** 2))
    vrf = baseline_variance / variance_est if variance_est > 0 else float("inf")

    return {
        "method": method_name,
        "mean": mean_est,
        "bias": bias,
        "var": variance_est,
        "mse": mse,
        "vrf_vs_crude": vrf,
    }


def format_table(rows: List[Dict[str, float]]) -> str:
    headers = ["method", "mean", "bias", "var", "mse", "vrf_vs_crude"]
    widths = {h: len(h) for h in headers}

    formatted_rows: List[Dict[str, str]] = []
    for r in rows:
        fr = {
            "method": str(r["method"]),
            "mean": f"{r['mean']:.8f}",
            "bias": f"{r['bias']:+.8f}",
            "var": f"{r['var']:.10f}",
            "mse": f"{r['mse']:.10f}",
            "vrf_vs_crude": f"{r['vrf_vs_crude']:.4f}",
        }
        formatted_rows.append(fr)
        for h in headers:
            widths[h] = max(widths[h], len(fr[h]))

    def mk_line(cols: Dict[str, str]) -> str:
        return " | ".join(cols[h].ljust(widths[h]) for h in headers)

    header_line = mk_line({h: h for h in headers})
    sep_line = "-+-".join("-" * widths[h] for h in headers)
    row_lines = [mk_line(r) for r in formatted_rows]
    return "\n".join([header_line, sep_line, *row_lines])


def main() -> None:
    n_samples = 2000
    n_trials = 500

    truth = target_true_value()

    crude_estimates = run_trials(crude_mc, n_trials=n_trials, n_samples=n_samples, seed=11)
    antithetic_estimates = run_trials(
        antithetic_mc,
        n_trials=n_trials,
        n_samples=n_samples,
        seed=22,
    )

    beta_values = np.empty(n_trials, dtype=float)

    def cv_estimator(n: int, rng: np.random.Generator) -> float:
        est, beta = control_variate_mc(n, rng)
        cv_estimator.counter += 1
        beta_values[cv_estimator.counter - 1] = beta
        return est

    cv_estimator.counter = 0  # type: ignore[attr-defined]
    control_estimates = run_trials(
        cv_estimator,
        n_trials=n_trials,
        n_samples=n_samples,
        seed=33,
    )

    importance_estimates = run_trials(
        lambda n, rng: importance_sampling_mc(n, rng, theta=0.7),
        n_trials=n_trials,
        n_samples=n_samples,
        seed=44,
    )

    baseline_var = float(np.var(crude_estimates, ddof=1))

    rows = [
        summarize("Crude MC", crude_estimates, truth, baseline_var),
        summarize("Antithetic", antithetic_estimates, truth, baseline_var),
        summarize("Control Variate", control_estimates, truth, baseline_var),
        summarize("Importance Sampling", importance_estimates, truth, baseline_var),
    ]

    print("Variance Reduction Demo: estimate E[exp(X)], X~N(0,1)")
    print(f"True value = exp(1/2) = {truth:.8f}")
    print(f"n_samples per trial = {n_samples}, n_trials = {n_trials}")
    print()
    print(format_table(rows))
    print()
    print(
        "Average beta_hat (Control Variate, C=X): "
        f"{float(beta_values.mean()):.6f}"
    )


if __name__ == "__main__":
    main()
