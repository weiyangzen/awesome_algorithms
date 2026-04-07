"""Naive Bayes MVP (from-scratch GaussianNB + sklearn baseline).

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


@dataclass
class GaussianNBFromScratch:
    """Minimal Gaussian Naive Bayes implementation.

    Notes:
    - Assumes each feature is conditionally independent given the class.
    - Uses class-wise Gaussian likelihood with diagonal covariance.
    - Applies variance smoothing to avoid division by zero.
    """

    var_smoothing: float = 1e-9
    classes_: np.ndarray | None = None
    class_prior_: np.ndarray | None = None
    theta_: np.ndarray | None = None
    var_: np.ndarray | None = None
    class_count_: np.ndarray | None = None
    epsilon_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianNBFromScratch":
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        if x.shape[0] == 0:
            raise ValueError("empty training data is not allowed")

        x = np.asarray(x, dtype=float)
        y = np.asarray(y)

        classes, y_inv = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]
        n_samples, n_features = x.shape

        self.classes_ = classes
        self.theta_ = np.zeros((n_classes, n_features), dtype=float)
        self.var_ = np.zeros((n_classes, n_features), dtype=float)
        self.class_count_ = np.zeros(n_classes, dtype=float)

        x_var_global = np.var(x, axis=0)
        self.epsilon_ = self.var_smoothing * float(np.max(x_var_global))

        for class_index in range(n_classes):
            mask = y_inv == class_index
            x_i = x[mask]
            if x_i.shape[0] == 0:
                continue
            self.class_count_[class_index] = x_i.shape[0]
            self.theta_[class_index, :] = np.mean(x_i, axis=0)
            self.var_[class_index, :] = np.var(x_i, axis=0) + self.epsilon_

        self.class_prior_ = self.class_count_ / n_samples
        return self

    def _joint_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(x, dtype=float)

        assert self.classes_ is not None
        assert self.class_prior_ is not None
        assert self.theta_ is not None
        assert self.var_ is not None

        jll_rows: list[np.ndarray] = []
        for i in range(self.classes_.shape[0]):
            log_prior = np.log(self.class_prior_[i])
            log_det = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[i, :]))
            quad = -0.5 * np.sum(((x - self.theta_[i, :]) ** 2) / self.var_[i, :], axis=1)
            jll_rows.append(log_prior + log_det + quad)

        return np.vstack(jll_rows).T

    def predict(self, x: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(x)
        assert self.classes_ is not None
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(x)
        row_max = np.max(jll, axis=1, keepdims=True)
        exp_shifted = np.exp(jll - row_max)
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def _check_fitted(self) -> None:
        if (
            self.classes_ is None
            or self.class_prior_ is None
            or self.theta_ is None
            or self.var_ is None
        ):
            raise RuntimeError("model is not fitted")


def make_gaussian_classification_data(seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    """Generate a reproducible multi-class Gaussian-like dataset."""
    x, y = make_classification(
        n_samples=900,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.7,
        weights=[0.5, 0.3, 0.2],
        flip_y=0.02,
        random_state=seed,
    )
    return x.astype(np.float64), y.astype(np.int64)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def main() -> None:
    var_smoothing = 1e-9
    x, y = make_gaussian_classification_data(seed=2026)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    custom_nb = GaussianNBFromScratch(var_smoothing=var_smoothing).fit(x_train, y_train)
    sklearn_nb = GaussianNB(var_smoothing=var_smoothing)
    sklearn_nb.fit(x_train, y_train)

    pred_custom = custom_nb.predict(x_test)
    pred_sklearn = sklearn_nb.predict(x_test)
    proba_custom = custom_nb.predict_proba(x_test)
    proba_sklearn = sklearn_nb.predict_proba(x_test)

    metrics_df = pd.DataFrame(
        [
            {"model": "GaussianNB(from-scratch)", **evaluate_metrics(y_test, pred_custom)},
            {"model": "GaussianNB(sklearn)", **evaluate_metrics(y_test, pred_sklearn)},
        ]
    )

    prediction_agreement = float(np.mean(pred_custom == pred_sklearn))
    theta_gap = float(np.linalg.norm(custom_nb.theta_ - sklearn_nb.theta_))
    var_gap = float(np.linalg.norm(custom_nb.var_ - sklearn_nb.var_))
    prior_gap = float(np.linalg.norm(custom_nb.class_prior_ - sklearn_nb.class_prior_))
    proba_gap = float(np.linalg.norm(proba_custom - proba_sklearn) / np.sqrt(proba_custom.size))

    confusion = confusion_matrix(y_test, pred_custom, labels=custom_nb.classes_)
    confusion_df = pd.DataFrame(
        confusion,
        index=[f"true_{c}" for c in custom_nb.classes_],
        columns=[f"pred_{c}" for c in custom_nb.classes_],
    )

    preview_count = 8
    preview_df = pd.DataFrame(
        {
            "y_true": y_test[:preview_count],
            "pred_custom": pred_custom[:preview_count],
            "pred_sklearn": pred_sklearn[:preview_count],
            "max_prob_custom": np.max(proba_custom[:preview_count], axis=1),
            "max_prob_sklearn": np.max(proba_sklearn[:preview_count], axis=1),
        }
    )

    print("=== Naive Bayes MVP: GaussianNB ===")
    print(f"train_size={x_train.shape[0]}, test_size={x_test.shape[0]}, n_features={x_train.shape[1]}")
    print(f"classes={custom_nb.classes_.tolist()}")
    print(f"var_smoothing={var_smoothing}")
    print()

    print("[Metrics]")
    print(metrics_df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print()

    print("[From-Scratch vs sklearn Consistency]")
    print(f"prediction_agreement                = {prediction_agreement:.6f}")
    print(f"L2(theta_custom - theta_sklearn)   = {theta_gap:.6e}")
    print(f"L2(var_custom - var_sklearn)       = {var_gap:.6e}")
    print(f"L2(prior_custom - prior_sklearn)   = {prior_gap:.6e}")
    print(f"RMSE(proba_custom - proba_sklearn) = {proba_gap:.6e}")
    print()

    print("[Confusion Matrix - From Scratch]")
    print(confusion_df.to_string())
    print()

    print("[Prediction Preview]")
    print(preview_df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    # Lightweight quality gates for validation.
    custom_accuracy = float(metrics_df.loc[metrics_df["model"] == "GaussianNB(from-scratch)", "accuracy"].iloc[0])
    if custom_accuracy < 0.82:
        raise RuntimeError(f"unexpectedly low accuracy: {custom_accuracy:.4f}")
    if prediction_agreement < 0.98:
        raise RuntimeError(f"from-scratch and sklearn predictions diverged: {prediction_agreement:.4f}")
    if not np.allclose(np.sum(proba_custom, axis=1), 1.0, atol=1e-9):
        raise RuntimeError("custom probabilities do not sum to 1")


if __name__ == "__main__":
    main()
