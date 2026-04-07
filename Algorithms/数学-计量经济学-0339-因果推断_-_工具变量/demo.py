"""Instrumental Variables (IV) / 2SLS minimal runnable MVP.

This script is self-contained and non-interactive:
1) Simulate an endogeneity setting with one endogenous regressor x.
2) Estimate naive OLS and compare with two-stage least squares (2SLS).
3) Report first-stage diagnostics (relevance F-test) and DWH endogeneity test.
4) Use explicit matrix formulas for estimation and HC1 robust covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class RegressionResult:
    term_names: list[str]
    coef: np.ndarray
    std_err: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    ci95_low: np.ndarray
    ci95_high: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "term": self.term_names,
                "coef": self.coef,
                "std_err": self.std_err,
                "t": self.t_stat,
                "p_value": self.p_value,
                "ci95_low": self.ci95_low,
                "ci95_high": self.ci95_high,
            }
        )


def _safe_inv(a: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Numerically robust inverse with pseudo-inverse fallback."""
    p = a.shape[0]
    regularized = a + ridge * np.eye(p)
    try:
        return np.linalg.inv(regularized)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(regularized)


def _hc1_covariance(x: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """HC1 robust covariance for OLS-style estimators."""
    n, k = x.shape
    xtx_inv = _safe_inv(x.T @ x)
    meat = (x.T * (residuals * residuals)) @ x
    correction = n / max(n - k, 1)
    return correction * (xtx_inv @ meat @ xtx_inv)


def _result_from_beta(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    cov: np.ndarray,
    term_names: list[str],
) -> RegressionResult:
    """Package coefficients and inference stats."""
    n, k = x.shape
    fitted = x @ beta
    residuals = y - fitted

    var_diag = np.diag(cov)
    std_err = np.where(var_diag > 0, np.sqrt(var_diag), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = beta / std_err

    dof = max(n - k, 1)
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=dof))
    crit = stats.t.ppf(0.975, df=dof)
    ci95_low = beta - crit * std_err
    ci95_high = beta + crit * std_err

    return RegressionResult(
        term_names=term_names,
        coef=beta,
        std_err=std_err,
        t_stat=t_stat,
        p_value=p_value,
        ci95_low=ci95_low,
        ci95_high=ci95_high,
        fitted=fitted,
        residuals=residuals,
    )


def ols_hc1(y: np.ndarray, x: np.ndarray, term_names: list[str]) -> RegressionResult:
    """Ordinary least squares with HC1 robust standard errors."""
    beta = _safe_inv(x.T @ x) @ (x.T @ y)
    residuals = y - x @ beta
    cov = _hc1_covariance(x, residuals)
    return _result_from_beta(x=x, y=y, beta=beta, cov=cov, term_names=term_names)


def two_stage_least_squares_hc1(
    y: np.ndarray,
    x_endog: np.ndarray,
    w_exog: np.ndarray,
    z_excluded: np.ndarray,
    term_names: list[str],
) -> RegressionResult:
    """2SLS with HC1 robust covariance.

    Structural equation:
        y = X*beta + u,  with X=[W, x_endog]
    Instruments:
        Z=[W, z_excluded]

    2SLS estimator:
        beta = (X'PzX)^(-1) X'Pz y
        Pz = Z (Z'Z)^(-1) Z'
    """
    x = np.column_stack([w_exog, x_endog])
    z = np.column_stack([w_exog, z_excluded])

    ztz_inv = _safe_inv(z.T @ z)
    x_t_z = x.T @ z

    a = x_t_z @ ztz_inv @ x_t_z.T
    a_inv = _safe_inv(a)
    beta = a_inv @ (x_t_z @ ztz_inv @ (z.T @ y))

    residuals = y - x @ beta

    # Sandwich robust covariance for linear IV/2SLS.
    n, k = x.shape
    s = (z.T * (residuals * residuals)) @ z
    b = x_t_z @ ztz_inv @ s @ ztz_inv @ x_t_z.T
    correction = n / max(n - k, 1)
    cov = correction * (a_inv @ b @ a_inv)

    return _result_from_beta(x=x, y=y, beta=beta, cov=cov, term_names=term_names)


def first_stage_f_test(
    x_endog: np.ndarray,
    w_exog: np.ndarray,
    z_excluded: np.ndarray,
) -> tuple[float, float]:
    """Classical relevance F-test for excluded instrument(s).

    H0: coefficients on excluded instruments are all zero in first stage.
    """
    x = x_endog.reshape(-1)
    xr = w_exog
    xu = np.column_stack([w_exog, z_excluded])

    beta_r = _safe_inv(xr.T @ xr) @ (xr.T @ x)
    beta_u = _safe_inv(xu.T @ xu) @ (xu.T @ x)

    resid_r = x - xr @ beta_r
    resid_u = x - xu @ beta_u

    ssr_r = float(resid_r @ resid_r)
    ssr_u = float(resid_u @ resid_u)

    n = xu.shape[0]
    q = xu.shape[1] - xr.shape[1]
    k_u = xu.shape[1]
    df2 = max(n - k_u, 1)

    numerator = max(ssr_r - ssr_u, 0.0) / max(q, 1)
    denominator = ssr_u / df2
    f_stat = numerator / denominator if denominator > 0 else np.nan
    p_value = 1.0 - stats.f.cdf(f_stat, dfn=max(q, 1), dfd=df2) if np.isfinite(f_stat) else np.nan
    return float(f_stat), float(p_value)


def durbin_wu_hausman_test(
    y: np.ndarray,
    x_endog: np.ndarray,
    w_exog: np.ndarray,
    z_excluded: np.ndarray,
) -> tuple[float, float]:
    """Residual-inclusion DWH endogeneity test.

    Steps:
    1) First stage: x_endog on [W, Z_excluded], collect residual v_hat.
    2) Augmented regression: y on [W, x_endog, v_hat].
    3) Test H0: coef(v_hat)=0 (exogeneity of x_endog).
    """
    z_full = np.column_stack([w_exog, z_excluded])
    fs_beta = _safe_inv(z_full.T @ z_full) @ (z_full.T @ x_endog.reshape(-1))
    x_hat = z_full @ fs_beta
    v_hat = x_endog.reshape(-1) - x_hat

    x_aug = np.column_stack([w_exog, x_endog.reshape(-1), v_hat])
    term_names = ["intercept", "w", "x", "v_hat"]
    aug = ols_hc1(y=y, x=x_aug, term_names=term_names)

    row = aug.to_frame().set_index("term").loc["v_hat"]
    return float(row["t"]), float(row["p_value"])


def simulate_iv_data(
    n: int = 2500,
    beta_x_true: float = 2.0,
    seed: int = 2026,
) -> pd.DataFrame:
    """Generate synthetic data with endogeneity and a valid instrument.

    Data generating process:
      z, w, u independent in expectation (u is unobserved confounder in practice).
      x = 0.9 z + 0.6 w + 0.95 u + v
      y = 1.0 + beta_x_true x + 0.7 w + 1.1 u + eps

    Because u enters both x and y, OLS on y~x+w is biased.
    Instrument z affects y only through x (exclusion in DGP) and strongly shifts x.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    z = rng.normal(loc=0.0, scale=1.0, size=n)
    w = rng.normal(loc=0.0, scale=1.0, size=n)
    u = rng.normal(loc=0.0, scale=1.0, size=n)
    v = rng.normal(loc=0.0, scale=1.0, size=n)

    eps = torch.randn(n, dtype=torch.float64).numpy() * (0.8 + 0.2 * np.abs(w))

    x = 0.9 * z + 0.6 * w + 0.95 * u + v
    y = 1.0 + beta_x_true * x + 0.7 * w + 1.1 * u + eps

    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "z": z,
            "w": w,
            "u_hidden": u,
        }
    )


def main() -> None:
    beta_x_true = 2.0
    df = simulate_iv_data(n=2500, beta_x_true=beta_x_true, seed=2026)

    y = df["y"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float).reshape(-1, 1)
    z = df["z"].to_numpy(dtype=float).reshape(-1, 1)
    w = df["w"].to_numpy(dtype=float)

    w_exog = np.column_stack([np.ones(len(df), dtype=float), w])

    # Naive OLS ignores endogeneity.
    x_ols = np.column_stack([w_exog, x.reshape(-1)])
    ols_res = ols_hc1(y=y, x=x_ols, term_names=["intercept", "w", "x"])

    # 2SLS uses instrument z.
    iv_res = two_stage_least_squares_hc1(
        y=y,
        x_endog=x,
        w_exog=w_exog,
        z_excluded=z,
        term_names=["intercept", "w", "x"],
    )

    # First-stage quality.
    fs_x = np.column_stack([w_exog, z])
    fs_res = ols_hc1(y=x.reshape(-1), x=fs_x, term_names=["intercept", "w", "z"])
    fs_f, fs_f_p = first_stage_f_test(x_endog=x, w_exog=w_exog, z_excluded=z)
    fs_mse = mean_squared_error(x.reshape(-1), fs_res.fitted)
    fs_r2 = r2_score(x.reshape(-1), fs_res.fitted)

    # Endogeneity test.
    dwh_t, dwh_p = durbin_wu_hausman_test(y=y, x_endog=x, w_exog=w_exog, z_excluded=z)

    ols_x = ols_res.to_frame().set_index("term").loc["x"]
    iv_x = iv_res.to_frame().set_index("term").loc["x"]

    ols_bias = abs(float(ols_x["coef"]) - beta_x_true)
    iv_bias = abs(float(iv_x["coef"]) - beta_x_true)

    pass_relevance = fs_f > 10.0
    pass_endogeneity = dwh_p < 0.05
    pass_iv_better = iv_bias < ols_bias
    overall_pass = pass_relevance and pass_endogeneity and pass_iv_better

    print("=== Instrumental Variables (2SLS) MVP ===")
    print(f"Rows: {len(df)}")
    print(f"True structural coefficient beta_x: {beta_x_true:.4f}")
    print()

    print("Naive OLS (y ~ x + w):")
    print(ols_res.to_frame().round(4).to_string(index=False))
    print()

    print("2SLS-IV (endogenous x instrumented by z, controls include w):")
    print(iv_res.to_frame().round(4).to_string(index=False))
    print()

    print("First-stage diagnostics (x ~ z + w):")
    print(fs_res.to_frame().round(4).to_string(index=False))
    print(f"Relevance F-stat (excluded z): {fs_f:.4f}, p-value: {fs_f_p:.6f}")
    print(f"First-stage R^2: {fs_r2:.4f}, MSE: {fs_mse:.4f}")
    print()

    print("Durbin-Wu-Hausman residual-inclusion test:")
    print(f"t(v_hat) = {dwh_t:.4f}, p-value = {dwh_p:.6f}")
    print()

    print("Bias comparison on coefficient x:")
    print(f"|OLS beta_x - true| = {ols_bias:.4f}")
    print(f"|2SLS beta_x - true| = {iv_bias:.4f}")
    print()

    print("Checks:")
    print(f"- Relevance (F > 10): {'PASS' if pass_relevance else 'FAIL'}")
    print(f"- Endogeneity detected (DWH p < 0.05): {'PASS' if pass_endogeneity else 'FAIL'}")
    print(f"- IV closer to truth than OLS: {'PASS' if pass_iv_better else 'FAIL'}")
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
