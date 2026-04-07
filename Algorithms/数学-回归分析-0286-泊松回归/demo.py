"""Poisson regression MVP using hand-written IRLS (log link).

This script is deterministic and requires no interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.special import gammaln

Array = np.ndarray
HistoryItem = Tuple[int, float, float]


@dataclass
class PoissonIRLSResult:
    coef: Array
    fitted_mean: Array
    linear_predictor: Array
    converged: bool
    n_iter: int
    log_likelihood: float
    mean_deviance: float
    history: List[HistoryItem]


@dataclass
class EvalReport:
    log_likelihood: float
    mean_deviance: float
    mse: float
    mae: float
    mcfadden_r2: float


def validate_dataset(x: Array, y: Array) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape={x.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"sample mismatch: x has {x.shape[0]}, y has {y.shape[0]}.")
    if x.shape[0] < 8:
        raise ValueError("need at least 8 samples.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values.")
    if np.any(y < 0) or np.any(np.floor(y) != y):
        raise ValueError("y must contain non-negative integer counts.")


def add_intercept(x: Array) -> Array:
    ones = np.ones((x.shape[0], 1), dtype=float)
    return np.hstack([ones, x.astype(float, copy=False)])


def poisson_log_likelihood(y: Array, mu: Array, eps: float = 1e-12) -> float:
    mu_safe = np.maximum(mu, eps)
    return float(np.sum(y * np.log(mu_safe) - mu_safe - gammaln(y + 1.0)))


def poisson_mean_deviance(y: Array, mu: Array, eps: float = 1e-12) -> float:
    mu_safe = np.maximum(mu, eps)
    term = np.zeros_like(mu_safe)
    positive = y > 0
    term[positive] = y[positive] * np.log(y[positive] / mu_safe[positive])
    deviance = 2.0 * np.sum(term - (y - mu_safe))
    return float(deviance / y.size)


def mcfadden_pseudo_r2(y: Array, mu_model: Array) -> float:
    ll_model = poisson_log_likelihood(y, mu_model)
    mu_null = np.full_like(y, fill_value=np.mean(y), dtype=float)
    ll_null = poisson_log_likelihood(y, mu_null)
    if abs(ll_null) < 1e-12:
        return 0.0
    return float(1.0 - ll_model / ll_null)


def fit_poisson_regression_irls(
    x: Array,
    y: Array,
    max_iter: int = 120,
    tol: float = 1e-8,
    l2_reg: float = 1e-9,
    max_eta: float = 20.0,
) -> PoissonIRLSResult:
    """Fit Poisson GLM with log link by IRLS (iteratively reweighted least squares)."""
    validate_dataset(x, y)
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")
    if l2_reg < 0.0:
        raise ValueError("l2_reg must be >= 0.")

    x_aug = add_intercept(x)
    n_samples, n_params = x_aug.shape

    coef = np.zeros(n_params, dtype=float)
    coef[0] = np.log(np.mean(y) + 1e-8)  # stable intercept initialization

    reg = np.eye(n_params, dtype=float) * l2_reg
    reg[0, 0] = 0.0  # keep intercept unpenalized

    history: List[HistoryItem] = []
    converged = False

    for it in range(1, max_iter + 1):
        eta = np.clip(x_aug @ coef, -max_eta, max_eta)
        mu = np.exp(eta)

        # For Poisson + log link: W = diag(mu), z = eta + (y - mu)/mu
        w = np.maximum(mu, 1e-12)
        z = eta + (y - mu) / w

        hessian_approx = x_aug.T @ (x_aug * w[:, None]) + reg
        rhs = x_aug.T @ (w * z)

        try:
            coef_new = np.linalg.solve(hessian_approx, rhs)
        except np.linalg.LinAlgError:
            coef_new = np.linalg.pinv(hessian_approx) @ rhs

        delta_inf = float(np.max(np.abs(coef_new - coef)))
        coef = coef_new

        eta_new = np.clip(x_aug @ coef, -max_eta, max_eta)
        mu_new = np.exp(eta_new)
        ll = poisson_log_likelihood(y, mu_new)
        history.append((it, ll, delta_inf))

        if delta_inf < tol:
            converged = True
            break

    eta = np.clip(x_aug @ coef, -max_eta, max_eta)
    mu = np.exp(eta)

    return PoissonIRLSResult(
        coef=coef,
        fitted_mean=mu,
        linear_predictor=eta,
        converged=converged,
        n_iter=len(history),
        log_likelihood=poisson_log_likelihood(y, mu),
        mean_deviance=poisson_mean_deviance(y, mu),
        history=history,
    )


def predict_mean_count(x: Array, coef: Array, max_eta: float = 20.0) -> Array:
    x_aug = add_intercept(x)
    eta = np.clip(x_aug @ coef, -max_eta, max_eta)
    return np.exp(eta)


def evaluate_poisson_regression(y_true: Array, mu_pred: Array) -> EvalReport:
    return EvalReport(
        log_likelihood=poisson_log_likelihood(y_true, mu_pred),
        mean_deviance=poisson_mean_deviance(y_true, mu_pred),
        mse=float(np.mean((y_true - mu_pred) ** 2)),
        mae=float(np.mean(np.abs(y_true - mu_pred))),
        mcfadden_r2=mcfadden_pseudo_r2(y_true, mu_pred),
    )


def train_test_split(
    x: Array,
    y: Array,
    test_ratio: float = 0.25,
    seed: int = 2026,
) -> Tuple[Array, Array, Array, Array]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1).")

    n = x.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_test = int(round(n * test_ratio))
    n_test = min(max(1, n_test), n - 1)

    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def make_synthetic_poisson_data(
    n_samples: int = 320,
    seed: int = 314,
) -> Tuple[Array, Array, Array]:
    """Generate deterministic data for a log-linear count model."""
    if n_samples < 16:
        raise ValueError("n_samples must be >= 16.")

    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 4.0, size=n_samples)
    x2 = rng.normal(0.0, 1.0, size=n_samples)
    x3 = rng.binomial(1, 0.4, size=n_samples).astype(float)
    x = np.column_stack([x1, x2, x3])

    # eta = beta0 + beta1*x1 + beta2*x2 + beta3*x3
    beta_true = np.array([0.45, 0.25, -0.40, 0.70], dtype=float)
    eta = beta_true[0] + x @ beta_true[1:]
    mu = np.exp(np.clip(eta, -20.0, 20.0))
    y = rng.poisson(mu)

    return x, y.astype(float), beta_true


def print_dataset_preview(x: Array, y: Array, rows: int = 6) -> None:
    rows = max(1, min(rows, x.shape[0]))
    print("dataset preview (first rows):")
    print(f"{'idx':>4} {'x1':>8} {'x2':>10} {'x3':>6} {'count_y':>10}")
    for i in range(rows):
        print(f"{i:4d} {x[i, 0]:8.4f} {x[i, 1]:10.4f} {x[i, 2]:6.1f} {int(y[i]):10d}")


def print_eval(label: str, report: EvalReport) -> None:
    print(f"\n[{label}]")
    print(f"log_likelihood : {report.log_likelihood:.6f}")
    print(f"mean_deviance  : {report.mean_deviance:.6f}")
    print(f"MSE            : {report.mse:.6f}")
    print(f"MAE            : {report.mae:.6f}")
    print(f"McFadden R^2   : {report.mcfadden_r2:.6f}")


def print_prediction_preview(y_true: Array, mu_pred: Array, rows: int = 8) -> None:
    rows = max(1, min(rows, y_true.shape[0]))
    print("\nprediction preview (first rows):")
    print(f"{'idx':>4} {'y_true':>10} {'mu_pred':>12} {'abs_err':>10}")
    for i in range(rows):
        err = abs(y_true[i] - mu_pred[i])
        print(f"{i:4d} {int(y_true[i]):10d} {mu_pred[i]:12.6f} {err:10.6f}")


def main() -> None:
    print("Poisson Regression MVP (hand-written IRLS)")
    print("=" * 52)

    x, y, beta_true = make_synthetic_poisson_data(n_samples=320, seed=314)
    print(f"samples={x.shape[0]}, features={x.shape[1]}")
    print_dataset_preview(x, y, rows=6)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.25, seed=2026)

    result = fit_poisson_regression_irls(
        x_train,
        y_train,
        max_iter=120,
        tol=1e-8,
        l2_reg=1e-9,
        max_eta=20.0,
    )

    mu_train = predict_mean_count(x_train, result.coef)
    mu_test = predict_mean_count(x_test, result.coef)

    train_report = evaluate_poisson_regression(y_train, mu_train)
    test_report = evaluate_poisson_regression(y_test, mu_test)

    print("\nfit summary:")
    print(f"converged      : {result.converged}")
    print(f"iterations     : {result.n_iter}")
    print(f"train mean(y)  : {float(np.mean(y_train)):.6f}")
    print(
        "true beta      :",
        np.array2string(beta_true, precision=6, suppress_small=True),
    )
    print(
        "estimated beta :",
        np.array2string(result.coef, precision=6, suppress_small=True),
    )
    print(
        "beta error     :",
        np.array2string(result.coef - beta_true, precision=6, suppress_small=True),
    )

    print_eval("train", train_report)
    print_eval("test", test_report)
    print_prediction_preview(y_test, mu_test, rows=8)

    pass_flag = result.converged and test_report.mean_deviance < 1.50 and test_report.mcfadden_r2 > 0.15
    print("\nsummary:")
    print(f"pass={pass_flag}")


if __name__ == "__main__":
    main()
