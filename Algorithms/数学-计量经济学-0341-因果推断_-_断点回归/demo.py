"""Sharp Regression Discontinuity (RDD) minimal reproducible MVP.

This script is self-contained and non-interactive:
1) Simulate data with a true cutoff effect.
2) Estimate local-linear sharp RD via weighted least squares.
3) Select bandwidth by weighted MSE on a candidate grid.
4) Report robust standard errors and placebo-cutoff checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_squared_error


@dataclass
class RDEstimate:
    cutoff: float
    bandwidth: float
    n_obs: int
    n_left: int
    n_right: int
    tau_hat: float
    robust_se: float
    t_stat: float
    p_value: float
    ci95_low: float
    ci95_high: float
    weighted_mse: float


def triangular_kernel(u: np.ndarray) -> np.ndarray:
    """Triangular kernel K(u)=max(1-|u|,0)."""
    return np.clip(1.0 - np.abs(u), 0.0, None)


def _safe_inv(a: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Numerically robust inverse fallback for tiny local windows."""
    p = a.shape[0]
    regularized = a + ridge * np.eye(p)
    try:
        return np.linalg.inv(regularized)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(regularized)


def fit_local_linear_rd(
    x: np.ndarray,
    y: np.ndarray,
    cutoff: float,
    bandwidth: float,
) -> tuple[RDEstimate, np.ndarray, np.ndarray, np.ndarray]:
    """Fit sharp RD with local linear specification and triangular kernel.

    Model inside window |x-c|<=h:
        y = b0 + tau*D + b1*(x-c) + b2*D*(x-c) + e
    where D = 1[x>=c].
    """
    dist = x - cutoff
    mask = np.abs(dist) <= bandwidth
    if mask.sum() < 30:
        raise ValueError("Too few observations in local window.")

    xw = x[mask]
    yw = y[mask]
    dw = (xw >= cutoff).astype(float)
    centered = xw - cutoff
    u = centered / bandwidth
    w = triangular_kernel(u)

    n_left = int((dw == 0).sum())
    n_right = int((dw == 1).sum())
    if min(n_left, n_right) < 10:
        raise ValueError("Insufficient support on one side of cutoff.")

    # Explicit design matrix for local linear RD.
    X = np.column_stack(
        [
            np.ones_like(centered),
            dw,
            centered,
            dw * centered,
        ]
    )

    xtw = X.T * w
    xtwx = xtw @ X
    xtwy = xtw @ yw

    bread = _safe_inv(xtwx)
    beta = bread @ xtwy
    y_hat = X @ beta
    resid = yw - y_hat

    n, p = X.shape
    finite_sample = n / max(n - p, 1)
    meat = (X.T * (w * w * resid * resid)) @ X
    cov = finite_sample * (bread @ meat @ bread)

    tau_hat = float(beta[1])
    robust_var = float(cov[1, 1])
    robust_se = float(np.sqrt(robust_var)) if robust_var > 0 else float("nan")

    if np.isfinite(robust_se) and robust_se > 0:
        t_stat = tau_hat / robust_se
        dof = max(n - p, 1)
        p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=dof))
        z975 = stats.t.ppf(0.975, df=dof)
        ci95_low = tau_hat - z975 * robust_se
        ci95_high = tau_hat + z975 * robust_se
    else:
        t_stat = float("nan")
        p_value = float("nan")
        ci95_low = float("nan")
        ci95_high = float("nan")

    weighted_mse = mean_squared_error(yw, y_hat, sample_weight=w)

    estimate = RDEstimate(
        cutoff=float(cutoff),
        bandwidth=float(bandwidth),
        n_obs=int(n),
        n_left=n_left,
        n_right=n_right,
        tau_hat=tau_hat,
        robust_se=robust_se,
        t_stat=float(t_stat),
        p_value=float(p_value),
        ci95_low=float(ci95_low),
        ci95_high=float(ci95_high),
        weighted_mse=float(weighted_mse),
    )
    return estimate, mask, y_hat, w


def select_bandwidth(
    x: np.ndarray,
    y: np.ndarray,
    cutoff: float,
    candidates: Iterable[float],
) -> tuple[float, RDEstimate, pd.DataFrame]:
    """Grid search bandwidth with weighted MSE objective."""
    rows: list[dict[str, float]] = []
    best: RDEstimate | None = None

    for h in candidates:
        h = float(h)
        if h <= 0:
            continue
        try:
            est, _, _, _ = fit_local_linear_rd(x=x, y=y, cutoff=cutoff, bandwidth=h)
        except ValueError:
            rows.append(
                {
                    "bandwidth": h,
                    "n_obs": np.nan,
                    "tau_hat": np.nan,
                    "weighted_mse": np.nan,
                    "valid": 0.0,
                }
            )
            continue

        rows.append(
            {
                "bandwidth": h,
                "n_obs": float(est.n_obs),
                "tau_hat": est.tau_hat,
                "weighted_mse": est.weighted_mse,
                "valid": 1.0,
            }
        )

        if best is None or est.weighted_mse < best.weighted_mse:
            best = est

    trace = pd.DataFrame(rows).sort_values(by="bandwidth").reset_index(drop=True)
    if best is None:
        raise RuntimeError("No valid bandwidth candidate found.")

    return best.bandwidth, best, trace


def simulate_sharp_rd_data(
    n: int = 2500,
    cutoff: float = 0.0,
    tau_true: float = 1.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate heteroskedastic synthetic data for sharp RD."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    x = np.random.uniform(-1.0, 1.0, size=n)
    mu = 1.2 + 0.9 * x - 1.1 * x * x + 0.4 * np.sin(3.0 * x)
    sigma = 0.25 + 0.15 * np.abs(x)
    eps = torch.randn(n, dtype=torch.float64).numpy() * sigma

    d = (x >= cutoff).astype(float)
    y = mu + tau_true * d + eps

    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "d": d,
            "mu": mu,
        }
    )


def run_placebo_tests(
    x: np.ndarray,
    y: np.ndarray,
    placebo_cutoffs: Iterable[float],
    bandwidth_candidates: np.ndarray,
) -> pd.DataFrame:
    """Estimate RD at fake cutoffs to check for spurious discontinuities."""
    rows: list[dict[str, float]] = []
    for c in placebo_cutoffs:
        try:
            _, est, _ = select_bandwidth(
                x=x,
                y=y,
                cutoff=float(c),
                candidates=bandwidth_candidates,
            )
            rows.append(
                {
                    "cutoff": float(c),
                    "bandwidth": est.bandwidth,
                    "tau_hat": est.tau_hat,
                    "robust_se": est.robust_se,
                    "p_value": est.p_value,
                }
            )
        except RuntimeError:
            rows.append(
                {
                    "cutoff": float(c),
                    "bandwidth": np.nan,
                    "tau_hat": np.nan,
                    "robust_se": np.nan,
                    "p_value": np.nan,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    cutoff = 0.0
    tau_true = 1.5
    df = simulate_sharp_rd_data(n=2500, cutoff=cutoff, tau_true=tau_true, seed=2026)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    bandwidth_candidates = np.round(np.linspace(0.08, 0.60, 14), 3)
    selected_h, rd_est, bw_trace = select_bandwidth(
        x=x,
        y=y,
        cutoff=cutoff,
        candidates=bandwidth_candidates,
    )

    # Naive within-window mean gap for a simple baseline comparison.
    mask = np.abs(x - cutoff) <= selected_h
    local = df.loc[mask]
    naive_gap = float(local.loc[local["d"] == 1.0, "y"].mean() - local.loc[local["d"] == 0.0, "y"].mean())

    placebo_df = run_placebo_tests(
        x=x,
        y=y,
        placebo_cutoffs=[-0.45, -0.25, 0.25, 0.45],
        bandwidth_candidates=bandwidth_candidates,
    )

    effect_detected = np.isfinite(rd_est.p_value) and rd_est.p_value < 0.05 and rd_est.tau_hat > 0
    magnitude_ok = abs(rd_est.tau_hat - tau_true) < 0.40
    placebo_ok = bool((placebo_df["p_value"].fillna(1.0) > 0.05).all())
    validation = effect_detected and magnitude_ok and placebo_ok

    print("=== Sharp RD MVP (Local Linear + Triangular Kernel) ===")
    print(f"Sample size: {len(df)}")
    print(f"True cutoff: {cutoff:.2f}, true tau: {tau_true:.3f}")
    print(f"Selected bandwidth: {selected_h:.3f}")
    print()

    summary = pd.DataFrame(
        [
            {
                "tau_hat": rd_est.tau_hat,
                "robust_se": rd_est.robust_se,
                "t_stat": rd_est.t_stat,
                "p_value": rd_est.p_value,
                "ci95_low": rd_est.ci95_low,
                "ci95_high": rd_est.ci95_high,
                "n_obs": rd_est.n_obs,
                "n_left": rd_est.n_left,
                "n_right": rd_est.n_right,
                "weighted_mse": rd_est.weighted_mse,
                "naive_mean_gap": naive_gap,
            }
        ]
    )
    print("Main RD estimate:")
    print(summary.round(6).to_string(index=False))
    print()

    print("Bandwidth search trace:")
    print(bw_trace.round(6).to_string(index=False))
    print()

    print("Placebo cutoff tests:")
    print(placebo_df.round(6).to_string(index=False))
    print()

    print(
        "Validation:",
        "PASS" if validation else "FAIL",
        f"(effect_detected={effect_detected}, magnitude_ok={magnitude_ok}, placebo_ok={placebo_ok})",
    )


if __name__ == "__main__":
    main()
