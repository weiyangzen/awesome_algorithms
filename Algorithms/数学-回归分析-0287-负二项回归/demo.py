"""Negative Binomial Regression (NB2) minimal runnable MVP.

This script implements NB2 regression from source-level equations:
- mean: mu_i = exp(x_i^T beta)
- variance: Var(Y_i|x_i) = mu_i + alpha * mu_i^2

Training uses maximum likelihood with L-BFGS-B and analytic gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class NegativeBinomialRegression:
    """Fitted NB2 regression model."""

    coef_: np.ndarray  # includes intercept at index 0
    alpha_: float
    n_iter_: int
    converged_: bool
    train_nll_: float

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        x_design = add_intercept(x)
        eta = x_design @ self.coef_
        return np.exp(eta)


def add_intercept(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    return np.c_[np.ones(x.shape[0]), x]


def make_synthetic_nb_data(
    n_samples: int = 900,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate synthetic NB2 data via Gamma-Poisson mixture."""
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 2))

    true_beta = np.array([0.55, 0.85, -0.60])
    true_alpha = 0.70

    eta = add_intercept(x) @ true_beta
    mu = np.exp(eta)

    # NB2 can be sampled as: lambda ~ Gamma(shape=r, scale=mu/r), y ~ Poisson(lambda)
    r = 1.0 / true_alpha
    lam = rng.gamma(shape=r, scale=mu / r)
    y = rng.poisson(lam)

    return x, y.astype(np.int64), true_beta, true_alpha


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.25,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")

    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def nb2_nll_and_grad(
    params: np.ndarray,
    x_design: np.ndarray,
    y: np.ndarray,
    l2_reg: float,
) -> tuple[float, np.ndarray]:
    """Negative log-likelihood and gradient for NB2 regression."""
    beta = params[:-1]
    log_alpha = params[-1]
    alpha = np.exp(log_alpha)
    r = 1.0 / alpha

    eta = x_design @ beta
    mu = np.exp(eta)

    # log p(y|mu, alpha) under NB2 parameterization
    # p(y) = Gamma(y+r)/(Gamma(r) y!) * (r/(r+mu))^r * (mu/(r+mu))^y
    loglik = (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1.0)
        + r * (np.log(r) - np.log(r + mu))
        + y * (np.log(mu) - np.log(r + mu))
    )

    penalty = 0.5 * l2_reg * float(beta[1:] @ beta[1:])
    nll = -float(np.sum(loglik)) + penalty

    # d nll / d beta
    # d loglik / d eta = y - mu * (r+y)/(r+mu)
    common = mu * (r + y) / (r + mu) - y
    grad_beta = x_design.T @ common
    grad_beta[1:] += l2_reg * beta[1:]

    # d nll / d log_alpha using chain rule through r = exp(-log_alpha)
    dloglik_dr = (
        digamma(y + r)
        - digamma(r)
        + np.log(r)
        + 1.0
        - np.log(r + mu)
        - (r + y) / (r + mu)
    )
    grad_log_alpha = float(np.sum(r * dloglik_dr))

    grad = np.concatenate([grad_beta, np.array([grad_log_alpha])])
    return nll, grad


def fit_nb2_regression(
    x: np.ndarray,
    y: np.ndarray,
    l2_reg: float = 1e-4,
    max_iter: int = 500,
) -> NegativeBinomialRegression:
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y size mismatch")
    if np.any(y < 0):
        raise ValueError("negative count found in y")

    x_design = add_intercept(x)

    # Warm start: regress log(y+0.5) on x for beta; moments for alpha.
    z = np.log(y + 0.5)
    beta0, *_ = np.linalg.lstsq(x_design, z, rcond=None)

    mean_y = float(np.mean(y))
    var_y = float(np.var(y))
    alpha0 = max((var_y - mean_y) / (mean_y**2 + 1e-8), 0.05)

    params0 = np.concatenate([beta0, np.array([np.log(alpha0)])])

    bounds = [(-4.0, 4.0)] * x_design.shape[1] + [(-6.0, 3.0)]

    def fun(p: np.ndarray) -> float:
        value, _ = nb2_nll_and_grad(p, x_design, y, l2_reg=l2_reg)
        return value

    def jac(p: np.ndarray) -> np.ndarray:
        _, g = nb2_nll_and_grad(p, x_design, y, l2_reg=l2_reg)
        return g

    result = minimize(
        fun=fun,
        x0=params0,
        jac=jac,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-6},
    )

    if not result.success:
        raise RuntimeError(f"NB2 optimization failed: {result.message}")

    beta_hat = result.x[:-1]
    alpha_hat = float(np.exp(result.x[-1]))
    train_nll, _ = nb2_nll_and_grad(result.x, x_design, y, l2_reg=l2_reg)

    return NegativeBinomialRegression(
        coef_=beta_hat,
        alpha_=alpha_hat,
        n_iter_=int(result.nit),
        converged_=bool(result.success),
        train_nll_=float(train_nll),
    )


def nb2_mean_nll(y_true: np.ndarray, y_pred_mu: np.ndarray, alpha: float) -> float:
    r = 1.0 / alpha
    loglik = (
        gammaln(y_true + r)
        - gammaln(r)
        - gammaln(y_true + 1.0)
        + r * (np.log(r) - np.log(r + y_pred_mu))
        + y_true * (np.log(y_pred_mu) - np.log(r + y_pred_mu))
    )
    return float(-np.mean(loglik))


def main() -> None:
    x, y, true_beta, true_alpha = make_synthetic_nb_data(n_samples=900, seed=2026)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.25, seed=17)

    model = fit_nb2_regression(x_train, y_train, l2_reg=1e-4, max_iter=600)

    train_pred = model.predict_mean(x_train)
    test_pred = model.predict_mean(x_test)

    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    train_mae = float(mean_absolute_error(y_train, train_pred))
    test_mae = float(mean_absolute_error(y_test, test_pred))
    test_mean_nll = nb2_mean_nll(y_test, test_pred, model.alpha_)

    print("=== Negative Binomial Regression (NB2) MVP ===")
    print(f"train size={len(y_train)}, test size={len(y_test)}")
    print()

    print("True parameters")
    print(f"  beta_true: {np.array2string(true_beta, precision=4)}")
    print(f"  alpha_true: {true_alpha:.4f}")
    print()

    print("Estimated parameters")
    print(f"  beta_hat : {np.array2string(model.coef_, precision=4)}")
    print(f"  alpha_hat: {model.alpha_:.4f}")
    print(f"  optimizer_converged: {model.converged_}, n_iter={model.n_iter_}")
    print(f"  train_nll (sum + l2): {model.train_nll_:.4f}")
    print()

    print("Metrics")
    print(f"  train_rmse: {train_rmse:.4f}")
    print(f"  test_rmse : {test_rmse:.4f}")
    print(f"  train_mae : {train_mae:.4f}")
    print(f"  test_mae  : {test_mae:.4f}")
    print(f"  test_mean_nll (NB2): {test_mean_nll:.4f}")
    print()

    sample = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred_mean": test_pred,
            "abs_error": np.abs(y_test - test_pred),
        }
    ).head(12)
    print("Sample predictions (first 12 rows)")
    print(sample.to_string(index=False, justify="center", col_space=12, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
