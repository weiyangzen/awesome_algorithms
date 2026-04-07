"""Gaussian Naive Bayes MVP (from-scratch + sklearn parity check)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


@dataclass
class ExperimentConfig:
    test_size: float = 0.30
    random_state: int = 42
    var_smoothing: float = 1e-9
    preview_rows: int = 8


class GaussianNBScratch:
    """Minimal Gaussian Naive Bayes implementation."""

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = float(var_smoothing)
        self.is_fitted = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianNBScratch":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.classes_, class_inverse = np.unique(y, return_inverse=True)

        n_classes = self.classes_.shape[0]
        n_features = x.shape[1]

        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.class_prior_ = np.zeros(n_classes, dtype=np.float64)
        self.theta_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var_ = np.zeros((n_classes, n_features), dtype=np.float64)

        epsilon = self.var_smoothing * np.var(x, axis=0).max()

        for class_idx in range(n_classes):
            x_c = x[class_inverse == class_idx]
            self.class_count_[class_idx] = x_c.shape[0]
            self.class_prior_[class_idx] = x_c.shape[0] / x.shape[0]
            self.theta_[class_idx, :] = np.mean(x_c, axis=0)
            self.var_[class_idx, :] = np.var(x_c, axis=0) + epsilon

        self.is_fitted = True
        return self

    def _joint_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        x = np.asarray(x, dtype=np.float64)
        all_classes_ll = []

        for class_idx in range(self.classes_.shape[0]):
            mean = self.theta_[class_idx]
            var = self.var_[class_idx]

            log_prior = np.log(self.class_prior_[class_idx])
            log_gaussian = -0.5 * np.sum(np.log(2.0 * np.pi * var), axis=0)
            sq_term = -0.5 * np.sum(((x - mean) ** 2) / var, axis=1)
            all_classes_ll.append(log_prior + log_gaussian + sq_term)

        return np.vstack(all_classes_ll).T

    def predict(self, x: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(x)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(x)
        shifted = jll - np.max(jll, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def main() -> None:
    cfg = ExperimentConfig()

    iris = load_iris()
    x = iris.data.astype(np.float64)
    y = iris.target
    target_names = iris.target_names

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    scratch = GaussianNBScratch(var_smoothing=cfg.var_smoothing).fit(x_train, y_train)
    y_pred_scratch = scratch.predict(x_test)
    y_prob_scratch = scratch.predict_proba(x_test)
    acc_scratch = accuracy_score(y_test, y_pred_scratch)

    sklearn_model = GaussianNB(var_smoothing=cfg.var_smoothing)
    sklearn_model.fit(x_train, y_train)
    y_pred_sklearn = sklearn_model.predict(x_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

    parity = np.mean(y_pred_scratch == y_pred_sklearn)

    preview_df = pd.DataFrame(
        {
            "true": [target_names[i] for i in y_test[: cfg.preview_rows]],
            "pred_scratch": [target_names[i] for i in y_pred_scratch[: cfg.preview_rows]],
            "pred_sklearn": [target_names[i] for i in y_pred_sklearn[: cfg.preview_rows]],
            "p_setosa": y_prob_scratch[: cfg.preview_rows, 0],
            "p_versicolor": y_prob_scratch[: cfg.preview_rows, 1],
            "p_virginica": y_prob_scratch[: cfg.preview_rows, 2],
        }
    )

    cm = pd.crosstab(
        pd.Series([target_names[i] for i in y_test], name="true"),
        pd.Series([target_names[i] for i in y_pred_scratch], name="pred"),
    )

    print("=== Gaussian Naive Bayes MVP ===")
    print(f"dataset: Iris (n_samples={x.shape[0]}, n_features={x.shape[1]})")
    print(f"train/test: {x_train.shape[0]}/{x_test.shape[0]}")
    print(f"scratch accuracy: {acc_scratch:.4f}")
    print(f"sklearn accuracy: {acc_sklearn:.4f}")
    print(f"prediction parity (scratch vs sklearn): {parity:.4f}")

    print("\n[Preview predictions]")
    print(preview_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n[Confusion matrix - scratch]")
    print(cm)

    assert acc_scratch >= 0.85, "Scratch GaussianNB accuracy is unexpectedly low."
    assert parity >= 0.95, "Scratch implementation diverges too much from sklearn GaussianNB."


if __name__ == "__main__":
    main()
