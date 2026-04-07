"""Generalized Linear Model (GLM) MVP via transparent IRLS.

This script demonstrates two canonical GLM cases:
1) Bernoulli family with logit link (binary classification)
2) Poisson family with log link (count regression)

Implementation is intentionally explicit: no black-box fit() from external GLM
libraries. Core optimization uses Iteratively Reweighted Least Squares (IRLS).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

Array = np.ndarray
EPS = 1e-12


@dataclass
class GLMModel:
    family: str
    link: str
    coefficients: Array
    n_iter: int
    converged: bool


class BernoulliLogitFamily:
    name = "bernoulli"
    link = "logit"

    @staticmethod
    def validate_target(y: Array) -> None:
        if np.any((y != 0.0) & (y != 1.0)):
            raise ValueError("Bernoulli target y must contain only 0/1 values.")

    @staticmethod
    def inv_link(eta: Array) -> Array:
        return expit(eta)

    @staticmethod
    def variance(mu: Array) -> Array:
        return mu * (1.0 - mu)

    @staticmethod
    def dmu_deta(mu: Array) -> Array:
        return mu * (1.0 - mu)

    @staticmethod
    def clip_mu(mu: Array) -> Array:
        return np.clip(mu, 1e-8, 1.0 - 1e-8)


class PoissonLogFamily:
    name = "poisson"
    link = "log"

    @staticmethod
    def validate_target(y: Array) -> None:
        if np.any(y < 0.0):
            raise ValueError("Poisson target y must be non-negative.")
        rounded = np.round(y)
        if np.any(np.abs(y - rounded) > 1e-8):
            raise ValueError("Poisson target y must be integer-like counts.")

    @staticmethod
    def inv_link(eta: Array) -> Array:
        # Clip eta for numerical safety, exp(6)=403.43 remains manageable.
        return np.exp(np.clip(eta, -20.0, 6.0))

    @staticmethod
    def variance(mu: Array) -> Array:
        return mu

    @staticmethod
    def dmu_deta(mu: Array) -> Array:
        return mu

    @staticmethod
    def clip_mu(mu: Array) -> Array:
        return np.clip(mu, 1e-8, np.inf)


def validate_inputs(X: Array, y: Array) -> tuple[Array, Array]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if X.shape[0] < 8:
        raise ValueError("Need at least 8 samples.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values.")
    return X, y


def add_intercept(X: Array) -> Array:
    return np.column_stack([np.ones(X.shape[0], dtype=float), X])


def train_test_split(
    X: Array,
    y: Array,
    train_ratio: float = 0.75,
    seed: int = 42,
) -> tuple[Array, Array, Array, Array]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1).")

    X, y = validate_inputs(X, y)
    n_samples = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)

    n_train = int(np.floor(train_ratio * n_samples))
    if n_train < 4 or n_train >= n_samples:
        raise ValueError("Invalid split size from train_ratio.")

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def irls_fit(
    X: Array,
    y: Array,
    family: BernoulliLogitFamily | PoissonLogFamily,
    max_iter: int = 120,
    tol: float = 1e-8,
    l2_reg: float = 1e-6,
) -> GLMModel:
    """Fit GLM by Iteratively Reweighted Least Squares (IRLS)."""
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if l2_reg < 0.0:
        raise ValueError("l2_reg must be >= 0.")

    X, y = validate_inputs(X, y)
    family.validate_target(y)

    X_aug = add_intercept(X)
    n_features = X_aug.shape[1]
    beta = np.zeros(n_features, dtype=float)

    # Intercept warm-start based on empirical mean.
    y_mean = float(np.mean(y))
    if family.name == "bernoulli":
        y_mean = float(np.clip(y_mean, 1e-6, 1.0 - 1e-6))
        beta[0] = np.log(y_mean / (1.0 - y_mean))
    elif family.name == "poisson":
        beta[0] = np.log(max(y_mean, 1e-6))

    converged = False

    for iteration in range(1, max_iter + 1):
        eta = X_aug @ beta
        mu = family.clip_mu(family.inv_link(eta))

        var_mu = np.maximum(family.variance(mu), EPS)
        dmu = np.maximum(family.dmu_deta(mu), EPS)

        w = np.maximum((dmu * dmu) / var_mu, EPS)
        z = eta + (y - mu) / dmu

        sqrt_w = np.sqrt(w)
        X_w = X_aug * sqrt_w[:, None]
        z_w = z * sqrt_w

        gram = X_w.T @ X_w
        rhs = X_w.T @ z_w

        reg = l2_reg * np.eye(n_features, dtype=float)
        reg[0, 0] = 0.0  # Intercept is not regularized.

        try:
            beta_new = np.linalg.solve(gram + reg, rhs)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.pinv(gram + reg) @ rhs

        delta = np.linalg.norm(beta_new - beta)
        scale = 1.0 + np.linalg.norm(beta)
        beta = beta_new

        if delta <= tol * scale:
            converged = True
            break

    return GLMModel(
        family=family.name,
        link=family.link,
        coefficients=beta,
        n_iter=iteration,
        converged=converged,
    )


def predict_mean(
    model: GLMModel,
    X: Array,
    family: BernoulliLogitFamily | PoissonLogFamily,
) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D for prediction.")
    X_aug = add_intercept(X)
    return family.clip_mu(family.inv_link(X_aug @ model.coefficients))


def binary_accuracy(y_true: Array, prob: Array, threshold: float = 0.5) -> float:
    pred = (prob >= threshold).astype(float)
    return float(np.mean(pred == y_true))


def binary_logloss(y_true: Array, prob: Array) -> float:
    p = np.clip(prob, 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def poisson_rmse(y_true: Array, mu_pred: Array) -> float:
    return float(np.sqrt(np.mean((y_true - mu_pred) ** 2)))


def poisson_mean_deviance(y_true: Array, mu_pred: Array) -> float:
    mu = np.clip(mu_pred, 1e-8, np.inf)
    term = np.zeros_like(mu)
    positive = y_true > 0.0
    term[positive] = y_true[positive] * np.log(y_true[positive] / mu[positive])
    dev = 2.0 * np.sum(term - (y_true - mu))
    return float(dev / y_true.size)


def generate_logistic_data(
    n_samples: int = 900,
    seed: int = 2026,
) -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_samples, 3))
    beta_true = np.array([-0.35, 1.15, -0.95, 0.65], dtype=float)
    eta = add_intercept(X) @ beta_true
    prob = expit(eta)
    y = rng.binomial(1, prob, size=n_samples).astype(float)
    return X, y, beta_true


def generate_poisson_data(
    n_samples: int = 1000,
    seed: int = 314,
) -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_samples, 2))
    beta_true = np.array([0.25, 0.55, -0.42], dtype=float)
    eta = add_intercept(X) @ beta_true
    rate = np.exp(np.clip(eta, -2.5, 2.5))
    y = rng.poisson(rate, size=n_samples).astype(float)
    return X, y, beta_true


def print_coef_table(name: str, beta_est: Array, beta_true: Array) -> None:
    print(f"{name} coefficient comparison (intercept first):")
    print("idx | beta_true     | beta_est")
    print("-" * 38)
    for i, (bt, be) in enumerate(zip(beta_true, beta_est)):
        print(f"{i:3d} | {bt:12.6f} | {be:12.6f}")


def run_logistic_demo() -> None:
    family = BernoulliLogitFamily()
    X, y, beta_true = generate_logistic_data()
    X_tr, y_tr, X_te, y_te = train_test_split(X, y, train_ratio=0.75, seed=42)

    model = irls_fit(
        X_tr,
        y_tr,
        family=family,
        max_iter=120,
        tol=1e-8,
        l2_reg=1e-6,
    )
    prob_te = predict_mean(model, X_te, family)

    acc = binary_accuracy(y_te, prob_te)
    ll = binary_logloss(y_te, prob_te)

    print("\n=== Bernoulli GLM (Logit) ===")
    print(f"train_size={X_tr.shape[0]}, test_size={X_te.shape[0]}")
    print(f"converged={model.converged}, n_iter={model.n_iter}")
    print(f"test_accuracy={acc:.6f}")
    print(f"test_logloss={ll:.6f}")
    print_coef_table("Bernoulli", model.coefficients, beta_true)

    print("Sample probabilities:")
    print("idx | y_true | prob_pred")
    for i in range(8):
        print(f"{i:3d} | {int(y_te[i]):6d} | {prob_te[i]:9.6f}")


def run_poisson_demo() -> None:
    family = PoissonLogFamily()
    X, y, beta_true = generate_poisson_data()
    X_tr, y_tr, X_te, y_te = train_test_split(X, y, train_ratio=0.75, seed=11)

    model = irls_fit(
        X_tr,
        y_tr,
        family=family,
        max_iter=120,
        tol=1e-8,
        l2_reg=1e-6,
    )
    mu_te = predict_mean(model, X_te, family)

    rmse = poisson_rmse(y_te, mu_te)
    dev = poisson_mean_deviance(y_te, mu_te)

    print("\n=== Poisson GLM (Log) ===")
    print(f"train_size={X_tr.shape[0]}, test_size={X_te.shape[0]}")
    print(f"converged={model.converged}, n_iter={model.n_iter}")
    print(f"test_rmse={rmse:.6f}")
    print(f"test_mean_deviance={dev:.6f}")
    print_coef_table("Poisson", model.coefficients, beta_true)

    print("Sample rates:")
    print("idx | y_true | mu_pred")
    for i in range(8):
        print(f"{i:3d} | {int(y_te[i]):6d} | {mu_te[i]:8.4f}")


def main() -> None:
    print("Generalized Linear Model (GLM) MVP via IRLS")
    run_logistic_demo()
    run_poisson_demo()


if __name__ == "__main__":
    main()
