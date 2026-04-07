"""Difference-in-Differences (DiD) minimal runnable MVP.

This script builds a synthetic panel dataset, estimates ATT with two approaches:
1) Group-mean DiD formula
2) OLS regression with a DiD interaction term and cluster-robust SE by unit

Run:
    uv run python demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def simulate_panel(
    n_units: int = 400,
    n_periods: int = 4,
    treatment_start: int = 2,
    true_tau: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a balanced panel for a canonical DiD setup.

    Half of units are treated. Treatment activates from `treatment_start` onward.
    The DGP satisfies parallel trends by construction.
    """
    if n_periods <= treatment_start:
        raise ValueError("n_periods must be larger than treatment_start")

    rng = np.random.default_rng(seed)
    unit_ids = np.arange(n_units)

    treated_flag = np.zeros(n_units, dtype=int)
    treated_units = rng.choice(unit_ids, size=n_units // 2, replace=False)
    treated_flag[treated_units] = 1

    alpha_i = rng.normal(loc=8.0, scale=1.5, size=n_units)

    rows: list[dict[str, float | int]] = []
    for i in unit_ids:
        treated = treated_flag[i]
        for t in range(n_periods):
            post = int(t >= treatment_start)

            # Shared time trend (parallel trend assumption holds in expectation).
            gamma_t = 0.7 * t

            # Mild heteroskedasticity at unit level.
            eps_sd = 1.0 + 0.3 * treated
            eps = rng.normal(loc=0.0, scale=eps_sd)

            y = alpha_i[i] + gamma_t + true_tau * treated * post + eps
            rows.append(
                {
                    "unit": int(i),
                    "period": int(t),
                    "treated": int(treated),
                    "post": int(post),
                    "y": float(y),
                }
            )

    df = pd.DataFrame(rows)
    return df


def did_from_group_means(df: pd.DataFrame) -> tuple[float, pd.Series]:
    """Compute DiD ATT from 2x2 grouped means."""
    means = df.groupby(["treated", "post"], as_index=True)["y"].mean()
    att = (means[(1, 1)] - means[(1, 0)]) - (means[(0, 1)] - means[(0, 0)])
    return float(att), means


def build_design_matrix(df: pd.DataFrame, post_col: str = "post") -> np.ndarray:
    """Create X=[1, treated, post, treated*post] for DiD regression."""
    treated = df["treated"].to_numpy(dtype=float)
    post = df[post_col].to_numpy(dtype=float)
    interaction = treated * post
    intercept = np.ones(len(df), dtype=float)
    return np.column_stack([intercept, treated, post, interaction])


def ols_cluster_robust(
    y: np.ndarray,
    x: np.ndarray,
    clusters: np.ndarray,
    term_names: list[str],
) -> pd.DataFrame:
    """OLS with cluster-robust (CR1) covariance estimator.

    Covariance formula:
    V = c * (X'X)^(-1) [sum_g (X_g' u_g)(X_g' u_g)'] (X'X)^(-1)
    where c applies finite-sample correction.
    """
    n, k = x.shape
    xtx = x.T @ x
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta

    unique_clusters = np.unique(clusters)
    g = len(unique_clusters)
    if g < 2:
        raise ValueError("Need at least 2 clusters for cluster-robust inference")

    meat = np.zeros((k, k), dtype=float)
    for cid in unique_clusters:
        idx = clusters == cid
        xg = x[idx]
        ug = resid[idx]
        score_g = xg.T @ ug
        meat += np.outer(score_g, score_g)

    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    cov = correction * (xtx_inv @ meat @ xtx_inv)

    se = np.sqrt(np.diag(cov))
    t_stat = beta / se
    dof = g - 1
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=dof))

    return pd.DataFrame(
        {
            "term": term_names,
            "coef": beta,
            "std_err": se,
            "t": t_stat,
            "p_value": p_value,
        }
    )


def run_placebo_pretrend_test(df: pd.DataFrame) -> pd.DataFrame:
    """Simple placebo test on pre-periods only.

    Uses pre-period observations and pretends period==1 is a fake policy date.
    The treated*fake_post coefficient should be close to zero under parallel trends.
    """
    pre_df = df[df["post"] == 0].copy()
    if pre_df["period"].nunique() < 2:
        raise ValueError("Need at least two pre-periods for placebo test")

    pre_df["fake_post"] = (pre_df["period"] == pre_df["period"].max()).astype(int)

    x = build_design_matrix(pre_df, post_col="fake_post")
    y = pre_df["y"].to_numpy(dtype=float)
    clusters = pre_df["unit"].to_numpy(dtype=int)

    return ols_cluster_robust(
        y=y,
        x=x,
        clusters=clusters,
        term_names=["intercept", "treated", "fake_post", "treated:fake_post"],
    )


def main() -> None:
    true_tau = 2.0
    df = simulate_panel(true_tau=true_tau, seed=42)

    att_means, means = did_from_group_means(df)

    x = build_design_matrix(df)
    y = df["y"].to_numpy(dtype=float)
    clusters = df["unit"].to_numpy(dtype=int)

    result = ols_cluster_robust(
        y=y,
        x=x,
        clusters=clusters,
        term_names=["intercept", "treated", "post", "treated:post"],
    )

    did_row = result[result["term"] == "treated:post"].iloc[0]
    att_reg = float(did_row["coef"])

    placebo = run_placebo_pretrend_test(df)
    placebo_row = placebo[placebo["term"] == "treated:fake_post"].iloc[0]

    print("=== Difference-in-Differences MVP ===")
    print(f"Rows: {len(df)}, Units: {df['unit'].nunique()}, Periods: {df['period'].nunique()}")
    print(f"True treatment effect (tau): {true_tau:.4f}")
    print()

    print("Group means E[Y|treated, post]:")
    print(means.unstack("post").rename(columns={0: "pre", 1: "post"}).round(4))
    print()

    print(f"ATT from group means: {att_means:.4f}")
    print(f"ATT from regression (treated:post): {att_reg:.4f}")
    print(f"Absolute estimation error: {abs(att_reg - true_tau):.4f}")
    print()

    print("Regression table (cluster-robust by unit):")
    print(result.round(4).to_string(index=False))
    print()

    print("Pre-trend placebo (should be near zero):")
    print(placebo.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
