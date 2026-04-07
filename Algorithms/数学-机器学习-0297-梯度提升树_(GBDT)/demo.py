"""Minimal runnable MVP for Gradient Boosting Decision Trees (GBDT) binary classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class GBDTConfig:
    n_estimators: int = 120
    learning_rate: float = 0.1
    n_thresholds: int = 31
    min_samples_leaf: int = 12
    random_state: int = 42

    def validate(self) -> None:
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")
        if self.n_thresholds < 5:
            raise ValueError("n_thresholds must be >= 5")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def binary_logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)

    idx0 = np.where(y == 0.0)[0]
    idx1 = np.where(y == 1.0)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n_test0 = int(len(idx0) * test_size)
    n_test1 = int(len(idx1) * test_size)

    test_idx = np.concatenate([idx0[:n_test0], idx1[:n_test1]])
    train_idx = np.concatenate([idx0[n_test0:], idx1[n_test1:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def make_dataset(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_samples = 1800
    X = rng.normal(0.0, 1.0, size=(n_samples, 10))

    raw_score = (
        1.3 * X[:, 0]
        - 1.1 * X[:, 1]
        + 0.9 * (X[:, 2] > 0.0).astype(np.float64)
        + 0.7 * np.sin(1.8 * X[:, 3])
        - 0.6 * (X[:, 4] ** 2)
        + 0.8 * (X[:, 5] * X[:, 6] > 0.0).astype(np.float64)
        + rng.normal(0.0, 0.7, size=n_samples)
    )
    y_prob = sigmoid(raw_score)
    y = rng.binomial(1, y_prob).astype(np.float64)

    return stratified_split(X, y, test_size=0.25, random_state=random_state)


@dataclass
class RegressionStump:
    feature_index: int = 0
    threshold: float = 0.0
    left_value: float = 0.0
    right_value: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        target: np.ndarray,
        n_thresholds: int,
        min_samples_leaf: int,
    ) -> "RegressionStump":
        n_samples, n_features = X.shape
        best_loss = np.inf
        target_mean = float(np.mean(target))

        for feature in range(n_features):
            col = X[:, feature]
            quantiles = np.linspace(0.05, 0.95, n_thresholds)
            thresholds = np.unique(np.quantile(col, quantiles))

            for threshold in thresholds:
                left_mask = col <= threshold
                left_count = int(np.sum(left_mask))
                right_count = n_samples - left_count
                if left_count < min_samples_leaf or right_count < min_samples_leaf:
                    continue

                left_target = target[left_mask]
                right_target = target[~left_mask]
                left_value = float(np.mean(left_target))
                right_value = float(np.mean(right_target))

                left_loss = float(np.sum((left_target - left_value) ** 2))
                right_loss = float(np.sum((right_target - right_value) ** 2))
                loss = left_loss + right_loss

                if loss < best_loss:
                    best_loss = loss
                    self.feature_index = feature
                    self.threshold = float(threshold)
                    self.left_value = left_value
                    self.right_value = right_value

        if not np.isfinite(best_loss):
            self.feature_index = 0
            self.threshold = float(np.median(X[:, 0]))
            self.left_value = target_mean
            self.right_value = target_mean

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        col = X[:, self.feature_index]
        return np.where(col <= self.threshold, self.left_value, self.right_value)


class SimpleGBDTBinaryClassifier:
    """A minimal GBDT (log-loss) implementation with regression stumps as weak learners."""

    def __init__(self, config: GBDTConfig):
        self.config = config
        self.init_raw_score_: float = 0.0
        self.trees_: List[RegressionStump] = []
        self.train_loss_history_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, verbose_every: int = 10) -> "SimpleGBDTBinaryClassifier":
        self.config.validate()
        n_samples = X.shape[0]

        pos_rate = np.clip(np.mean(y), 1e-6, 1.0 - 1e-6)
        self.init_raw_score_ = float(np.log(pos_rate / (1.0 - pos_rate)))

        raw_scores = np.full(n_samples, self.init_raw_score_, dtype=np.float64)
        self.trees_.clear()
        self.train_loss_history_.clear()

        for m in range(self.config.n_estimators):
            prob = sigmoid(raw_scores)
            residual = y - prob  # negative gradient of log-loss wrt raw score

            tree = RegressionStump().fit(
                X,
                residual,
                n_thresholds=self.config.n_thresholds,
                min_samples_leaf=self.config.min_samples_leaf,
            )
            update = tree.predict(X)
            raw_scores += self.config.learning_rate * update

            self.trees_.append(tree)
            train_loss = binary_logloss(y, sigmoid(raw_scores))
            self.train_loss_history_.append(train_loss)

            if (m + 1) == 1 or (m + 1) % verbose_every == 0 or (m + 1) == self.config.n_estimators:
                print(f"iter={m + 1:03d} train_logloss={train_loss:.6f}")

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        raw_scores = np.full(X.shape[0], self.init_raw_score_, dtype=np.float64)
        for tree in self.trees_:
            raw_scores += self.config.learning_rate * tree.predict(X)
        return raw_scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_scores = self.decision_function(X)
        prob_pos = sigmoid(raw_scores)
        prob_neg = 1.0 - prob_pos
        return np.column_stack([prob_neg, prob_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.float64)


def main() -> None:
    config = GBDTConfig(
        n_estimators=120,
        learning_rate=0.1,
        n_thresholds=31,
        min_samples_leaf=12,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = make_dataset(random_state=config.random_state)

    train_pos_rate = float(np.mean(y_train))
    baseline_train_prob = np.full_like(y_train, train_pos_rate, dtype=np.float64)
    baseline_test_prob = np.full_like(y_test, train_pos_rate, dtype=np.float64)
    majority_label = float(train_pos_rate >= 0.5)

    baseline_train_loss = binary_logloss(y_train, baseline_train_prob)
    baseline_test_loss = binary_logloss(y_test, baseline_test_prob)
    baseline_test_acc = accuracy(y_test, np.full_like(y_test, majority_label))

    print("=== Baseline (constant model) ===")
    print(f"baseline_train_logloss={baseline_train_loss:.6f}")
    print(f"baseline_test_logloss={baseline_test_loss:.6f}")
    print(f"baseline_test_accuracy={baseline_test_acc:.4f}")

    print("\n=== Training GBDT ===")
    model = SimpleGBDTBinaryClassifier(config).fit(X_train, y_train, verbose_every=12)

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_loss = binary_logloss(y_train, train_prob)
    test_loss = binary_logloss(y_test, test_prob)
    train_acc = accuracy(y_train, train_pred)
    test_acc = accuracy(y_test, test_pred)

    print("\n=== Final Metrics ===")
    print(f"train_logloss={train_loss:.6f}")
    print(f"test_logloss={test_loss:.6f}")
    print(f"train_accuracy={train_acc:.4f}")
    print(f"test_accuracy={test_acc:.4f}")

    assert len(model.trees_) == config.n_estimators, "Number of trained trees mismatch"
    assert np.all(np.isfinite(train_prob)) and np.all(np.isfinite(test_prob)), "Non-finite probabilities"
    assert train_loss < baseline_train_loss * 0.92, "Train log-loss did not improve enough"
    assert test_loss < baseline_test_loss * 0.95, "Test log-loss did not improve enough"
    assert test_acc > baseline_test_acc + 0.12, "Test accuracy below expectation"

    print("All checks passed.")


if __name__ == "__main__":
    main()
