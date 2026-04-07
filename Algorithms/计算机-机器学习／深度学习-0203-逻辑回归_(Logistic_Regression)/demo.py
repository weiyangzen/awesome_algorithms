"""Minimal runnable MVP for Logistic Regression (from-scratch + optional sklearn baseline)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


EPS = 1e-12


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    clipped = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def make_synthetic_binary_data(
    n_samples: int = 900,
    n_features: int = 6,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a reproducible binary dataset with moderate linear separability."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n_samples, n_features))

    # Hidden linear rule used to generate labels.
    true_w = np.array([1.5, -2.2, 0.7, 1.0, -1.3, 0.5], dtype=float)
    true_b = -0.2
    noise = rng.normal(0.0, 0.8, size=n_samples)

    logits = x @ true_w + true_b + noise
    probs = sigmoid(logits)
    y = (rng.random(n_samples) < probs).astype(np.int64)

    # Ensure both classes exist for metrics stability.
    if y.min() == y.max():
        y[: n_samples // 2] = 0
        y[n_samples // 2 :] = 1
    return x, y


def train_test_split_np(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    perm = rng.permutation(n)
    test_size = int(n * test_ratio)
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (x_train - mu) / sigma, (x_test - mu) / sigma, mu, sigma


@dataclass
class NumpyLogisticRegression:
    lr: float = 0.1
    max_iter: int = 2500
    l2: float = 0.05
    tol: float = 1e-9
    fit_intercept: bool = True

    def __post_init__(self) -> None:
        self.weights_: np.ndarray | None = None
        self.n_iter_: int = 0
        self.loss_history_: List[float] = []

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return x
        ones = np.ones((x.shape[0], 1), dtype=x.dtype)
        return np.hstack([ones, x])

    def _regularization_loss(self, w: np.ndarray, n: int) -> float:
        if self.l2 <= 0:
            return 0.0
        if self.fit_intercept:
            return (self.l2 / (2.0 * n)) * float(np.dot(w[1:], w[1:]))
        return (self.l2 / (2.0 * n)) * float(np.dot(w, w))

    def _regularization_grad(self, w: np.ndarray, n: int) -> np.ndarray:
        grad = np.zeros_like(w)
        if self.l2 <= 0:
            return grad
        if self.fit_intercept:
            grad[1:] = (self.l2 / n) * w[1:]
        else:
            grad = (self.l2 / n) * w
        return grad

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        x_aug = self._augment(x)
        n, d = x_aug.shape
        w = np.zeros(d, dtype=float)

        prev_loss = float("inf")
        for i in range(1, self.max_iter + 1):
            logits = x_aug @ w
            probs = sigmoid(logits)

            data_loss = -np.mean(y * np.log(probs + EPS) + (1.0 - y) * np.log(1.0 - probs + EPS))
            loss = data_loss + self._regularization_loss(w, n)

            err = probs - y
            grad = (x_aug.T @ err) / n
            grad += self._regularization_grad(w, n)

            w -= self.lr * grad

            if i % 50 == 0 or i == 1:
                self.loss_history_.append(float(loss))

            if abs(prev_loss - loss) < self.tol:
                self.n_iter_ = i
                break
            prev_loss = loss
        else:
            self.n_iter_ = self.max_iter

        self.weights_ = w
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted yet.")
        x_aug = self._augment(x)
        return sigmoid(x_aug @ self.weights_)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(np.int64)


def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, EPS)
    logloss = -np.mean(y_true * np.log(y_prob + EPS) + (1.0 - y_true) * np.log(1.0 - y_prob + EPS))

    auc = roc_auc_score_np(y_true, y_prob)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "log_loss": float(logloss),
        "roc_auc": float(auc),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC via ranking (Mann-Whitney U)."""
    y_true = y_true.astype(np.int64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    pos_rank_sum = float(np.sum(ranks[y_true == 1]))
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def try_train_sklearn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    l2: float,
) -> Dict[str, float] | None:
    """Optional sklearn baseline. Returns None when sklearn is unavailable."""
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return None

    c_value = 1.0 / max(l2, 1e-8)
    model = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=5000,
        random_state=42,
    )
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    metrics = binary_metrics(y_test, prob)
    metrics["n_iter"] = float(np.max(model.n_iter_))
    return metrics


def print_result_table(rows: List[Dict[str, float]]) -> None:
    columns = ["model", "accuracy", "precision", "recall", "f1", "log_loss", "roc_auc", "n_iter"]

    if pd is not None:
        frame = pd.DataFrame(rows)
        for col in columns:
            if col not in frame.columns:
                frame[col] = np.nan
        frame = frame[columns].sort_values(by="f1", ascending=False)
        print("\n=== Model Comparison ===")
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        return

    print("\n=== Model Comparison ===")
    header = " | ".join(columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        values = []
        for col in columns:
            val = row.get(col)
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        print(" | ".join(values))


def main() -> None:
    x, y = make_synthetic_binary_data(n_samples=900, n_features=6, seed=42)
    x_train, x_test, y_train, y_test = train_test_split_np(x, y, test_ratio=0.2, seed=7)
    x_train_s, x_test_s, _, _ = standardize_train_test(x_train, x_test)

    model = NumpyLogisticRegression(lr=0.12, max_iter=3000, l2=0.05, tol=1e-10)
    model.fit(x_train_s, y_train)

    prob_np = model.predict_proba(x_test_s)
    metrics_np = binary_metrics(y_test, prob_np)

    results: List[Dict[str, float]] = []
    np_row: Dict[str, float] = {
        "model": "numpy_logreg_gd",
        "n_iter": float(model.n_iter_),
        **metrics_np,
    }
    results.append(np_row)

    sk_metrics = try_train_sklearn(x_train_s, y_train, x_test_s, y_test, l2=0.05)
    if sk_metrics is not None:
        sk_row: Dict[str, float] = {
            "model": "sklearn_logreg_lbfgs",
            **sk_metrics,
        }
        results.append(sk_row)

    print("Logistic Regression MVP (binary classification)")
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    print(f"Positive rate in test set: {float(np.mean(y_test)):.4f}")
    print(f"Numpy training iterations: {model.n_iter_}")
    if model.loss_history_:
        print(f"First tracked loss: {model.loss_history_[0]:.6f}")
        print(f"Last tracked loss:  {model.loss_history_[-1]:.6f}")

    print_result_table(results)

    # Show the most influential coefficients from the hand-written model.
    feature_names = [f"x{i}" for i in range(x_train_s.shape[1])]
    weights = model.weights_[1:] if model.fit_intercept else model.weights_
    coef_pairs = list(zip(feature_names, weights.tolist()))
    coef_pairs.sort(key=lambda t: abs(t[1]), reverse=True)

    print("\nTop coefficients by |weight| (numpy model):")
    for name, value in coef_pairs:
        print(f"  {name}: {value:+.4f}")


if __name__ == "__main__":
    main()
