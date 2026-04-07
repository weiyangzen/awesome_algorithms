"""Multinomial Logistic Regression minimal runnable MVP.

This script implements multinomial logistic regression from source equations
with full-batch gradient descent, instead of calling a one-line black-box API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp


@dataclass
class MultinomialLogisticModel:
    """Fitted multinomial logistic regression model."""

    weights: np.ndarray  # shape: (n_classes, n_features + 1), includes intercept
    classes_: np.ndarray
    loss_history: list[float]
    n_iter_: int
    converged_: bool


def validate_xy(x: np.ndarray, y: np.ndarray) -> None:
    """Validate input feature/label arrays."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")
    if x.shape[0] < 10:
        raise ValueError("need at least 10 samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values")



def add_intercept(x: np.ndarray) -> np.ndarray:
    """Add intercept column to feature matrix."""
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    return np.c_[np.ones(x.shape[0]), x]



def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute row-wise softmax probabilities in a numerically stable way."""
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))



def one_hot(indices: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert class indices to one-hot matrix."""
    y_mat = np.zeros((indices.shape[0], n_classes), dtype=float)
    y_mat[np.arange(indices.shape[0]), indices] = 1.0
    return y_mat



def multiclass_logloss(y_true_idx: np.ndarray, proba: np.ndarray) -> float:
    """Compute average multiclass negative log-likelihood."""
    eps = 1e-12
    p = np.clip(proba[np.arange(len(y_true_idx)), y_true_idx], eps, 1.0)
    return float(-np.mean(np.log(p)))



def split_train_test(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.25,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple reproducible train/test split."""
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")

    n = x.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(n * test_ratio)

    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]



def make_synthetic_multiclass_data(
    n_samples: int = 900,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic 3-class data from a known softmax model."""
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 4))

    # True parameters: 3 classes, 4 features + 1 intercept
    w_true = np.array(
        [
            [0.80, 2.00, -1.00, 0.50, 0.80],
            [-0.10, -1.20, 1.80, -0.60, -1.00],
            [-0.70, -0.80, -0.60, 0.10, 0.20],
        ],
        dtype=float,
    )

    logits = add_intercept(x) @ w_true.T
    probs = softmax(logits)

    y = np.array([rng.choice(3, p=probs[i]) for i in range(n_samples)], dtype=int)
    return x, y, w_true



def fit_multinomial_logistic_gd(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.15,
    max_iter: int = 3000,
    l2_reg: float = 1e-3,
    tol: float = 1e-7,
) -> MultinomialLogisticModel:
    """Fit multinomial logistic regression with full-batch gradient descent."""
    validate_xy(x, y)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if l2_reg < 0:
        raise ValueError("l2_reg must be >= 0")
    if tol <= 0:
        raise ValueError("tol must be > 0")

    classes = np.unique(y)
    if classes.size < 3:
        raise ValueError("multinomial logistic regression requires at least 3 classes")

    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_idx = np.array([class_to_idx[val] for val in y], dtype=int)

    x_design = add_intercept(x)
    n_samples, n_features = x_design.shape
    n_classes = classes.size

    w = np.zeros((n_classes, n_features), dtype=float)
    y_mat = one_hot(y_idx, n_classes)

    loss_history: list[float] = []
    converged = False

    for it in range(1, max_iter + 1):
        logits = x_design @ w.T
        probs = softmax(logits)

        ce = multiclass_logloss(y_idx, probs)
        reg = 0.5 * l2_reg * float(np.sum(w[:, 1:] ** 2))
        loss = ce + reg
        loss_history.append(loss)

        grad = (probs - y_mat).T @ x_design / n_samples
        grad[:, 1:] += l2_reg * w[:, 1:]

        w_next = w - learning_rate * grad
        delta = np.linalg.norm(w_next - w)
        w = w_next

        if delta < tol:
            converged = True
            n_iter = it
            break
    else:
        n_iter = max_iter

    return MultinomialLogisticModel(
        weights=w,
        classes_=classes,
        loss_history=loss_history,
        n_iter_=n_iter,
        converged_=converged,
    )



def predict_proba(model: MultinomialLogisticModel, x: np.ndarray) -> np.ndarray:
    """Predict class probabilities for each sample."""
    logits = add_intercept(x) @ model.weights.T
    return softmax(logits)



def predict(model: MultinomialLogisticModel, x: np.ndarray) -> np.ndarray:
    """Predict class labels."""
    proba = predict_proba(model, x)
    idx = np.argmax(proba, axis=1)
    return model.classes_[idx]



def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute simple classification accuracy."""
    return float(np.mean(y_true == y_pred))



def main() -> None:
    x, y, w_true = make_synthetic_multiclass_data(n_samples=900, seed=2026)
    x_train, x_test, y_train, y_test = split_train_test(x, y, test_ratio=0.25, seed=99)

    model = fit_multinomial_logistic_gd(
        x=x_train,
        y=y_train,
        learning_rate=0.15,
        max_iter=4000,
        l2_reg=1e-3,
        tol=1e-7,
    )

    train_proba = predict_proba(model, x_train)
    test_proba = predict_proba(model, x_test)

    class_to_idx = {cls: i for i, cls in enumerate(model.classes_)}
    y_train_idx = np.array([class_to_idx[v] for v in y_train], dtype=int)
    y_test_idx = np.array([class_to_idx[v] for v in y_test], dtype=int)

    y_train_pred = predict(model, x_train)
    y_test_pred = predict(model, x_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_ll = multiclass_logloss(y_train_idx, train_proba)
    test_ll = multiclass_logloss(y_test_idx, test_proba)

    print("=== Multinomial Logistic Regression MVP ===")
    print(f"train size={len(y_train)}, test size={len(y_test)}")
    print(f"classes={model.classes_.tolist()}")
    print(f"converged={model.converged_}, n_iter={model.n_iter_}")
    print(f"final_train_objective={model.loss_history[-1]:.6f}")
    print()

    print("Metrics")
    print(f"  train_accuracy: {train_acc:.4f}")
    print(f"  test_accuracy : {test_acc:.4f}")
    print(f"  train_logloss : {train_ll:.4f}")
    print(f"  test_logloss  : {test_ll:.4f}")
    print()

    print("Parameter overview")
    print(f"  true_weights shape: {w_true.shape}")
    print(f"  learned_weights shape: {model.weights.shape}")
    print("  first row of learned weights:", np.array2string(model.weights[0], precision=4))
    print()

    sample = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_test_pred,
            "p(class=0)": test_proba[:, 0],
            "p(class=1)": test_proba[:, 1],
            "p(class=2)": test_proba[:, 2],
        }
    ).head(12)
    print("Sample predictions (first 12 rows)")
    print(sample.to_string(index=False, justify="center", float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
