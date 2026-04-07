"""Minimal runnable MVP for Probit model (binary choice econometrics).

This script:
1) synthesizes data from a latent-variable Probit DGP,
2) estimates coefficients by MLE (custom likelihood + custom gradient),
3) reports coefficient table and hold-out metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


Array = np.ndarray


@dataclass
class ProbitResult:
    beta_hat: Array
    std_err: Array
    converged: bool
    iterations: int
    log_likelihood: float
    message: str


def add_intercept(x: Array) -> Array:
    """Append an intercept column as the first column."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a 2D matrix")
    return np.column_stack([np.ones(x.shape[0], dtype=float), x])


def probit_prob(x: Array, beta: Array, eps: float = 1e-9) -> Array:
    """Return clipped Probit probabilities Phi(X beta)."""
    z = np.asarray(x, dtype=float) @ np.asarray(beta, dtype=float)
    p = norm.cdf(z)
    return np.clip(p, eps, 1.0 - eps)


def neg_loglik_and_grad(beta: Array, x: Array, y: Array) -> Tuple[float, Array]:
    """Negative log-likelihood and gradient for binary Probit."""
    p = probit_prob(x, beta)
    z = x @ beta
    pdf = norm.pdf(z)

    loglik = np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p))

    # d/dz log L_i = phi(z_i) * (y_i - p_i) / [p_i * (1 - p_i)]
    score_z = pdf * (y - p) / (p * (1.0 - p))
    grad = x.T @ score_z

    return -float(loglik), -grad


def fit_probit_mle(x: Array, y: Array) -> ProbitResult:
    """Estimate Probit coefficients via MLE using BFGS."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("y must be 1D and aligned with x")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("y must contain only 0/1 values")

    beta0 = np.zeros(x.shape[1], dtype=float)

    objective = lambda b: neg_loglik_and_grad(b, x, y)[0]
    gradient = lambda b: neg_loglik_and_grad(b, x, y)[1]

    result = minimize(
        objective,
        beta0,
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 400, "disp": False},
    )

    hess_inv_raw = result.hess_inv
    if hasattr(hess_inv_raw, "todense"):
        hess_inv = np.asarray(hess_inv_raw.todense(), dtype=float)
    else:
        hess_inv = np.asarray(hess_inv_raw, dtype=float)

    if hess_inv.shape != (x.shape[1], x.shape[1]):
        std_err = np.full(x.shape[1], np.nan)
    else:
        std_err = np.sqrt(np.clip(np.diag(hess_inv), 0.0, None))

    return ProbitResult(
        beta_hat=np.asarray(result.x, dtype=float),
        std_err=std_err,
        converged=bool(result.success),
        iterations=int(result.nit),
        log_likelihood=-float(result.fun),
        message=str(result.message),
    )


def evaluate_binary_metrics(y_true: Array, prob: Array, threshold: float = 0.5) -> Dict[str, float]:
    """Compute core classification/calibration metrics."""
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "positive_rate": float(np.mean(y_true)),
    }


def simulate_probit_data(
    n_samples: int = 3000,
    n_features: int = 4,
    seed: int = 42,
) -> Tuple[Array, Array, Array]:
    """Generate synthetic data from latent-variable Probit process."""
    rng = np.random.default_rng(seed)
    x_raw = rng.normal(size=(n_samples, n_features))
    x = add_intercept(x_raw)

    beta_true = np.array([0.4, 1.1, -0.9, 0.7, -1.2], dtype=float)
    if beta_true.shape[0] != x.shape[1]:
        raise ValueError("beta_true dimension mismatch")

    latent = x @ beta_true + rng.normal(size=n_samples)
    y = (latent > 0.0).astype(float)
    return x, y, beta_true


def main() -> None:
    x, y, beta_true = simulate_probit_data(n_samples=3000, n_features=4, seed=42)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=7,
        stratify=y,
    )

    result = fit_probit_mle(x_train, y_train)

    train_prob = probit_prob(x_train, result.beta_hat)
    test_prob = probit_prob(x_test, result.beta_hat)
    train_metrics = evaluate_binary_metrics(y_train, train_prob)
    test_metrics = evaluate_binary_metrics(y_test, test_prob)

    coef_names = ["intercept"] + [f"x{i}" for i in range(1, x.shape[1])]
    coef_table = pd.DataFrame(
        {
            "feature": coef_names,
            "beta_true": beta_true,
            "beta_hat": result.beta_hat,
            "std_err_bfgs_approx": result.std_err,
        }
    )

    print("Probit MLE demo (synthetic latent-variable data)")
    print(f"Train size: {x_train.shape[0]}, Test size: {x_test.shape[0]}")
    print(f"Converged: {result.converged} | Iterations: {result.iterations}")
    print(f"Optimization message: {result.message}")
    print(f"Train log-likelihood at optimum: {result.log_likelihood:.6f}")

    print("\nCoefficient table:")
    print(coef_table.to_string(index=False, float_format=lambda v: f"{v: .5f}"))

    print("\nTrain metrics:")
    for k, v in train_metrics.items():
        print(f"  {k:>13s}: {v:.6f}")

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k:>13s}: {v:.6f}")

    # Deterministic sanity checks for this generated dataset.
    if not result.converged:
        raise RuntimeError("Probit MLE failed to converge.")
    if test_metrics["auc"] < 0.90:
        raise RuntimeError(f"AUC too low for this setup: {test_metrics['auc']:.4f}")
    if test_metrics["accuracy"] < 0.80:
        raise RuntimeError(f"Accuracy too low for this setup: {test_metrics['accuracy']:.4f}")

    print("\nValidation checks passed.")


if __name__ == "__main__":
    main()
