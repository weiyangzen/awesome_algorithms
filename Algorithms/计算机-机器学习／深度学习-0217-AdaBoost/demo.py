"""AdaBoost minimal runnable MVP.

This script implements binary AdaBoost from scratch using weighted decision stumps.
It runs offline on sklearn's breast cancer dataset and prints training diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


EPS = 1e-12


@dataclass
class DecisionStump:
    """A one-level decision tree for labels in {-1, +1}."""

    feature_index: int = 0
    threshold: float = 0.0
    polarity: int = 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = np.ones(x.shape[0], dtype=np.int8)
        if self.polarity == 1:
            preds[x[:, self.feature_index] < self.threshold] = -1
        else:
            preds[x[:, self.feature_index] >= self.threshold] = -1
        return preds

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> float:
        n_samples, n_features = x.shape
        best_error = float("inf")
        best_feature = 0
        best_threshold = 0.0
        best_polarity = 1

        for feature_idx in range(n_features):
            values = np.unique(x[:, feature_idx])
            if values.size == 1:
                thresholds = values
            else:
                mids = (values[:-1] + values[1:]) / 2.0
                thresholds = np.concatenate(([values[0] - EPS], mids, [values[-1] + EPS]))

            for threshold in thresholds:
                for polarity in (1, -1):
                    preds = np.ones(n_samples, dtype=np.int8)
                    if polarity == 1:
                        preds[x[:, feature_idx] < threshold] = -1
                    else:
                        preds[x[:, feature_idx] >= threshold] = -1

                    err = float(np.sum(sample_weight[preds != y]))
                    if err < best_error:
                        best_error = err
                        best_feature = feature_idx
                        best_threshold = float(threshold)
                        best_polarity = polarity

        self.feature_index = best_feature
        self.threshold = best_threshold
        self.polarity = best_polarity
        return best_error


class AdaBoostBinary:
    """Discrete AdaBoost for binary classification with y in {-1, +1}."""

    def __init__(self, n_estimators: int = 30, learning_rate: float = 1.0) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.stumps: List[DecisionStump] = []
        self.alphas: List[float] = []
        self.round_logs: List[dict] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "AdaBoostBinary":
        if set(np.unique(y)) != {-1, 1}:
            raise ValueError("AdaBoostBinary.fit expects labels in {-1, +1}")

        n_samples = x.shape[0]
        weights = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)

        self.stumps.clear()
        self.alphas.clear()
        self.round_logs.clear()

        for round_idx in range(1, self.n_estimators + 1):
            stump = DecisionStump()
            weighted_error = stump.fit(x, y, weights)
            weighted_error = float(np.clip(weighted_error, EPS, 1.0 - EPS))

            if weighted_error >= 0.5:
                break

            alpha = 0.5 * self.learning_rate * np.log((1.0 - weighted_error) / weighted_error)
            preds = stump.predict(x)

            weights *= np.exp(-alpha * y * preds)
            weights_sum = float(np.sum(weights))
            if weights_sum <= 0.0:
                raise RuntimeError("Invalid weight normalization factor")
            weights /= weights_sum

            self.stumps.append(stump)
            self.alphas.append(float(alpha))

            train_pred = self.predict(x)
            train_acc = float(np.mean(train_pred == y))
            self.round_logs.append(
                {
                    "round": round_idx,
                    "feature_index": stump.feature_index,
                    "threshold": stump.threshold,
                    "polarity": stump.polarity,
                    "weighted_error": weighted_error,
                    "alpha": float(alpha),
                    "weight_min": float(weights.min()),
                    "weight_max": float(weights.max()),
                    "train_acc": train_acc,
                }
            )

            if weighted_error <= EPS:
                break

        if not self.stumps:
            raise RuntimeError("No weak learner accepted. Check data or implementation.")

        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        scores = np.zeros(x.shape[0], dtype=np.float64)
        for alpha, stump in zip(self.alphas, self.stumps):
            scores += alpha * stump.predict(x)
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        return np.where(scores >= 0.0, 1, -1).astype(np.int8)


def to_pm_one(y01: np.ndarray) -> np.ndarray:
    return np.where(y01 == 1, 1, -1).astype(np.int8)


def from_pm_one(ypm: np.ndarray) -> np.ndarray:
    return np.where(ypm == 1, 1, 0).astype(np.int8)


def build_rounds_dataframe(round_logs: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(round_logs)
    cols = [
        "round",
        "feature_index",
        "threshold",
        "polarity",
        "weighted_error",
        "alpha",
        "weight_min",
        "weight_max",
        "train_acc",
    ]
    return df.loc[:, cols]


def main() -> None:
    data = load_breast_cancer()
    x = data.data.astype(np.float64)
    y01 = data.target.astype(np.int8)
    y = to_pm_one(y01)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = AdaBoostBinary(n_estimators=30, learning_rate=1.0)
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    train_acc = accuracy_score(from_pm_one(y_train), from_pm_one(y_pred_train))
    test_acc = accuracy_score(from_pm_one(y_test), from_pm_one(y_pred_test))

    stump_baseline = DecisionStump()
    _ = stump_baseline.fit(x_train, y_train, np.full(x_train.shape[0], 1.0 / x_train.shape[0]))
    stump_test_pred = stump_baseline.predict(x_test)
    stump_test_acc = accuracy_score(from_pm_one(y_test), from_pm_one(stump_test_pred))

    rounds_df = build_rounds_dataframe(model.round_logs)

    print(f"Dataset shape: {x.shape}, classes={np.bincount(y01).tolist()}")
    print(f"Train/Test size: {x_train.shape[0]}/{x_test.shape[0]}")
    print(f"Weak learners used: {len(model.stumps)}")

    print("\nTop boosting rounds (first 10):")
    print(rounds_df.head(10).to_string(index=False))

    print(f"\nSingle stump baseline accuracy: {stump_test_acc:.4f}")
    print(f"AdaBoost train accuracy: {train_acc:.4f}")
    print(f"AdaBoost test accuracy:  {test_acc:.4f}")

    print("\nClassification report (test):")
    print(classification_report(from_pm_one(y_test), from_pm_one(y_pred_test), digits=4))

    assert len(model.stumps) >= 3, "Too few weak learners; boosting did not proceed as expected."
    assert test_acc >= 0.90, f"Test accuracy too low: {test_acc:.4f}"
    assert test_acc >= stump_test_acc, (
        f"Boosting should not underperform single stump in this setup: "
        f"boost={test_acc:.4f}, stump={stump_test_acc:.4f}"
    )

    print("All checks passed.")


if __name__ == "__main__":
    main()
