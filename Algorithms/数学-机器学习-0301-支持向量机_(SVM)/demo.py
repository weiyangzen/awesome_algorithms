"""Linear soft-margin SVM MVP implemented with subgradient descent.

This script is intentionally minimal and self-contained:
- binary classification only
- hinge loss + L2 regularization
- full-batch subgradient optimization
- optional scikit-learn sanity check
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np


@dataclass
class StandardScalerMVP:
    """Simple standardization utility (zero mean, unit variance)."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScalerMVP":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={x.shape}")
        self.mean_ = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        self.scale_ = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted.")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={x.shape}")
        if x.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"Feature mismatch: scaler has {self.mean_.shape[0]}, got {x.shape[1]}"
            )
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


class LinearSVMMVP:
    """Binary linear SVM trained by minimizing primal hinge objective."""

    def __init__(
        self,
        c: float = 1.0,
        lr0: float = 0.12,
        lr_decay: float = 2e-3,
        epochs: int = 3500,
        fit_intercept: bool = True,
        support_tol: float = 1e-3,
    ) -> None:
        if c <= 0.0:
            raise ValueError("c must be > 0")
        if lr0 <= 0.0:
            raise ValueError("lr0 must be > 0")
        if lr_decay < 0.0:
            raise ValueError("lr_decay must be >= 0")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if support_tol < 0.0:
            raise ValueError("support_tol must be >= 0")

        self.c = float(c)
        self.lr0 = float(lr0)
        self.lr_decay = float(lr_decay)
        self.epochs = int(epochs)
        self.fit_intercept = bool(fit_intercept)
        self.support_tol = float(support_tol)

        self.scaler_ = StandardScalerMVP()
        self.w_: np.ndarray | None = None
        self.b_: float = 0.0
        self.n_features_in_: int = 0
        self.classes_: np.ndarray | None = None
        self.support_indices_: np.ndarray | None = None
        self.support_vectors_: np.ndarray | None = None
        self.train_objective_: float | None = None

    @staticmethod
    def _validate_xy(x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={x.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape={y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Sample mismatch: X has {x.shape[0]}, y has {y.shape[0]}")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature")
        if not np.all(np.isfinite(x)):
            raise ValueError("X contains non-finite values")
        if not np.all(np.isfinite(y.astype(float, copy=False))):
            raise ValueError("y contains non-finite values")

    def _to_signed_labels(self, y: np.ndarray) -> np.ndarray:
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(f"Binary classification only, got classes={classes}")
        self.classes_ = classes
        neg_cls, pos_cls = classes[0], classes[1]
        signed = np.where(y == neg_cls, -1.0, 1.0)
        return signed

    @staticmethod
    def _objective(x_std: np.ndarray, y_signed: np.ndarray, w: np.ndarray, b: float, c: float) -> float:
        margins = y_signed * (x_std @ w + b)
        hinge = np.maximum(0.0, 1.0 - margins)
        return float(0.5 * np.dot(w, w) + c * np.mean(hinge))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearSVMMVP":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        self._validate_xy(x, y)

        y_signed = self._to_signed_labels(y)
        x_std = self.scaler_.fit_transform(x)

        n_samples, n_features = x_std.shape
        self.n_features_in_ = n_features

        w = np.zeros(n_features, dtype=float)
        b = 0.0

        for t in range(1, self.epochs + 1):
            margins = y_signed * (x_std @ w + b)
            active = margins < 1.0

            grad_w = w.copy()
            if np.any(active):
                grad_w -= (self.c / n_samples) * np.sum(
                    y_signed[active, None] * x_std[active], axis=0
                )

            grad_b = 0.0
            if self.fit_intercept and np.any(active):
                grad_b = float(-(self.c / n_samples) * np.sum(y_signed[active]))

            lr_t = self.lr0 / (1.0 + self.lr_decay * t)
            w -= lr_t * grad_w
            if self.fit_intercept:
                b -= lr_t * grad_b

        self.w_ = w
        self.b_ = float(b)
        self.train_objective_ = self._objective(x_std, y_signed, w, b, self.c)

        final_margins = y_signed * (x_std @ w + b)
        support_idx = np.where(final_margins <= (1.0 + self.support_tol))[0]
        self.support_indices_ = support_idx
        self.support_vectors_ = x[support_idx]
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be 2D in predict, got shape={x.shape}")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature mismatch: fitted with {self.n_features_in_}, got {x.shape[1]}"
            )
        x_std = self.scaler_.transform(x)
        return x_std @ self.w_ + self.b_

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted yet.")
        score = self.decision_function(x)
        neg_cls, pos_cls = self.classes_[0], self.classes_[1]
        return np.where(score >= 0.0, pos_cls, neg_cls)


def make_binary_dataset(seed: int = 2026, n_samples: int = 420) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a moderately overlapping 3D binary dataset."""
    if n_samples < 20:
        raise ValueError("n_samples must be >= 20")

    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0

    mean0 = np.array([-1.6, -1.2])
    cov0 = np.array([[1.00, 0.55], [0.55, 1.05]])
    x0 = rng.multivariate_normal(mean=mean0, cov=cov0, size=n0)

    mean1 = np.array([1.45, 1.55])
    cov1 = np.array([[1.10, -0.35], [-0.35, 0.90]])
    x1 = rng.multivariate_normal(mean=mean1, cov=cov1, size=n1)

    x0_noise = 0.35 * x0[:, 0] - 0.18 * x0[:, 1] + rng.normal(0.0, 0.45, size=n0)
    x1_noise = 0.35 * x1[:, 0] - 0.18 * x1[:, 1] + rng.normal(0.0, 0.45, size=n1)

    x = np.vstack([
        np.column_stack([x0, x0_noise]),
        np.column_stack([x1, x1_noise]),
    ])
    y = np.concatenate([
        np.zeros(n0, dtype=int),
        np.ones(n1, dtype=int),
    ])

    perm = rng.permutation(n_samples)
    return x[perm], y[perm]


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.30,
    seed: int = 13,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")

    rng = np.random.default_rng(seed)
    n_samples = x.shape[0]
    idx = rng.permutation(n_samples)
    test_size = int(round(n_samples * test_ratio))

    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, positive_label: Any) -> Tuple[int, int, int, int]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    fn = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))
    fp = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    tn = int(np.sum((y_true != positive_label) & (y_pred != positive_label)))
    return tp, tn, fp, fn


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: Any,
) -> Tuple[float, float, float]:
    tp, _tn, fp, fn = confusion_counts(y_true, y_pred, positive_label)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def majority_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    values, counts = np.unique(y_train, return_counts=True)
    majority = values[np.argmax(counts)]
    return np.full(shape=n_test, fill_value=majority)


def maybe_compare_with_sklearn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c: float,
) -> None:
    try:
        from sklearn.svm import SVC
    except Exception:
        print("sklearn check: skipped (scikit-learn not available)")
        return

    model = SVC(C=c, kernel="linear", random_state=0)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    acc = accuracy(y_test, pred)
    p, r, f1 = precision_recall_f1(y_test, pred, positive_label=1)
    sv_count = int(model.support_.shape[0])
    print(
        "sklearn check (SVC linear): "
        f"acc={acc:.4f}, precision={p:.4f}, recall={r:.4f}, f1={f1:.4f}, "
        f"support_vectors={sv_count}"
    )


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    x, y = make_binary_dataset(seed=2026, n_samples=420)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.30, seed=23)

    model = LinearSVMMVP(
        c=1.2,
        lr0=0.12,
        lr_decay=2e-3,
        epochs=3500,
        fit_intercept=True,
        support_tol=1e-3,
    )
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    score_test = model.decision_function(x_test)

    train_acc = accuracy(y_train, pred_train)
    test_acc = accuracy(y_test, pred_test)

    p_train, r_train, f1_train = precision_recall_f1(y_train, pred_train, positive_label=1)
    p_test, r_test, f1_test = precision_recall_f1(y_test, pred_test, positive_label=1)

    base_pred = majority_baseline(y_train, n_test=y_test.shape[0])
    base_acc = accuracy(y_test, base_pred)

    tp, tn, fp, fn = confusion_counts(y_test, pred_test, positive_label=1)
    assert model.w_ is not None
    assert model.train_objective_ is not None
    assert model.support_indices_ is not None

    sv_count = int(model.support_indices_.size)
    sv_ratio = float(sv_count / x_train.shape[0])

    print("=== Linear SVM MVP (hinge + L2, subgradient) ===")
    print(f"train size={x_train.shape[0]}, test size={x_test.shape[0]}, features={x_train.shape[1]}")
    print(
        "hyperparameters: "
        f"C={model.c:.4f}, lr0={model.lr0:.4f}, lr_decay={model.lr_decay:.4f}, "
        f"epochs={model.epochs}, fit_intercept={model.fit_intercept}"
    )
    print(f"weight norm ||w||={np.linalg.norm(model.w_):.6f}, bias b={model.b_:.6f}")
    print(f"objective(train)={model.train_objective_:.6f}")
    print(f"approx support vectors={sv_count} ({sv_ratio:.2%} of train)")

    print(f"train metrics: acc={train_acc:.4f}, precision={p_train:.4f}, recall={r_train:.4f}, f1={f1_train:.4f}")
    print(f"test  metrics: acc={test_acc:.4f}, precision={p_test:.4f}, recall={r_test:.4f}, f1={f1_test:.4f}")
    print(f"baseline (majority class) test acc={base_acc:.4f}")
    print(f"confusion matrix (test): TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    print("\nSample predictions (first 10 test samples):")
    header = "idx | y_true | y_pred | decision_score"
    print(header)
    print("-" * len(header))
    show_n = min(10, y_test.shape[0])
    for i in range(show_n):
        print(f"{i:3d} | {y_test[i]:6d} | {pred_test[i]:6d} | {score_test[i]:14.6f}")

    maybe_compare_with_sklearn(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        c=model.c,
    )


if __name__ == "__main__":
    main()
