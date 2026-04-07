"""Minimal runnable MVP for XGBoost-style binary classification.

This is a from-scratch educational implementation (NumPy only), covering:
- logistic objective with first/second-order derivatives
- regularized leaf weight
- split gain with gamma / lambda / min_child_weight
- row and column subsampling per boosting round
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class XGBConfig:
    n_estimators: int = 80
    learning_rate: float = 0.12
    max_depth: int = 3
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_child_weight: float = 1.0
    min_samples_leaf: int = 16
    n_thresholds: int = 18
    subsample: float = 0.85
    colsample_bytree: float = 0.8
    random_state: int = 42

    def validate(self) -> None:
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        if self.reg_lambda < 0.0:
            raise ValueError("reg_lambda must be >= 0")
        if self.gamma < 0.0:
            raise ValueError("gamma must be >= 0")
        if self.min_child_weight <= 0.0:
            raise ValueError("min_child_weight must be > 0")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be > 0")
        if self.n_thresholds < 4:
            raise ValueError("n_thresholds must be >= 4")
        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0, 1]")
        if not (0.0 < self.colsample_bytree <= 1.0):
            raise ValueError("colsample_bytree must be in (0, 1]")


@dataclass
class TreeNode:
    is_leaf: bool
    weight: float
    feature: int = -1
    threshold: float = 0.0
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


@dataclass
class SplitCandidate:
    feature: int
    threshold: float
    gain: float
    left_idx: np.ndarray
    right_idx: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def make_dataset(random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_samples = 2200
    x = rng.normal(0.0, 1.0, size=(n_samples, 12))

    latent = (
        1.15 * x[:, 0]
        - 1.0 * x[:, 1]
        + 0.85 * (x[:, 2] > 0.3).astype(np.float64)
        + 0.8 * np.sin(1.6 * x[:, 3])
        - 0.75 * (x[:, 4] ** 2)
        + 0.95 * (x[:, 5] * x[:, 6] > 0.0).astype(np.float64)
        - 0.55 * np.abs(x[:, 7])
        + 0.5 * x[:, 8] * x[:, 9]
        + rng.normal(0.0, 0.65, size=n_samples)
    )

    y_prob = sigmoid(latent)
    y = rng.binomial(1, y_prob).astype(np.float64)
    return stratified_split(x, y, test_size=0.25, random_state=random_state)


class XGBTree:
    def __init__(self, config: XGBConfig, feature_indices: np.ndarray):
        self.config = config
        self.feature_indices = feature_indices
        self.root_: TreeNode | None = None

    def _node_weight(self, g_sum: float, h_sum: float) -> float:
        return -g_sum / (h_sum + self.config.reg_lambda)

    def _score(self, g_sum: float, h_sum: float) -> float:
        return (g_sum * g_sum) / (h_sum + self.config.reg_lambda)

    def _best_split(
        self,
        x: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        idx: np.ndarray,
    ) -> SplitCandidate | None:
        if idx.size < 2 * self.config.min_samples_leaf:
            return None

        g_parent = float(np.sum(g[idx]))
        h_parent = float(np.sum(h[idx]))
        parent_score = self._score(g_parent, h_parent)

        best_gain = -np.inf
        best: SplitCandidate | None = None

        for feature in self.feature_indices:
            values = x[idx, feature]
            v_min = float(np.min(values))
            v_max = float(np.max(values))
            if not np.isfinite(v_min) or not np.isfinite(v_max) or v_min == v_max:
                continue

            q = np.linspace(0.05, 0.95, self.config.n_thresholds)
            thresholds = np.unique(np.quantile(values, q))
            if thresholds.size == 0:
                continue

            for threshold in thresholds:
                left_mask = values <= threshold
                left_idx = idx[left_mask]
                right_idx = idx[~left_mask]

                if (
                    left_idx.size < self.config.min_samples_leaf
                    or right_idx.size < self.config.min_samples_leaf
                ):
                    continue

                g_left = float(np.sum(g[left_idx]))
                h_left = float(np.sum(h[left_idx]))
                g_right = float(np.sum(g[right_idx]))
                h_right = float(np.sum(h[right_idx]))

                if (
                    h_left < self.config.min_child_weight
                    or h_right < self.config.min_child_weight
                ):
                    continue

                gain = 0.5 * (
                    self._score(g_left, h_left)
                    + self._score(g_right, h_right)
                    - parent_score
                ) - self.config.gamma

                if gain > best_gain:
                    best_gain = gain
                    best = SplitCandidate(
                        feature=int(feature),
                        threshold=float(threshold),
                        gain=float(gain),
                        left_idx=left_idx,
                        right_idx=right_idx,
                    )

        if best is None or best.gain <= 0.0:
            return None
        return best

    def _build(
        self,
        x: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        idx: np.ndarray,
        depth: int,
    ) -> TreeNode:
        g_sum = float(np.sum(g[idx]))
        h_sum = float(np.sum(h[idx]))
        leaf_weight = self._node_weight(g_sum, h_sum)

        if depth >= self.config.max_depth:
            return TreeNode(is_leaf=True, weight=leaf_weight)
        if idx.size < 2 * self.config.min_samples_leaf:
            return TreeNode(is_leaf=True, weight=leaf_weight)
        if h_sum < 2.0 * self.config.min_child_weight:
            return TreeNode(is_leaf=True, weight=leaf_weight)

        split = self._best_split(x, g, h, idx)
        if split is None:
            return TreeNode(is_leaf=True, weight=leaf_weight)

        left = self._build(x, g, h, split.left_idx, depth + 1)
        right = self._build(x, g, h, split.right_idx, depth + 1)
        return TreeNode(
            is_leaf=False,
            weight=leaf_weight,
            feature=split.feature,
            threshold=split.threshold,
            left=left,
            right=right,
        )

    def fit(
        self,
        x: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        row_indices: np.ndarray,
    ) -> "XGBTree":
        self.root_ = self._build(x, g, h, row_indices, depth=0)
        return self

    def _predict_one(self, row: np.ndarray) -> float:
        if self.root_ is None:
            raise RuntimeError("Tree is not fitted")
        node = self.root_
        while not node.is_leaf:
            if row[node.feature] <= node.threshold:
                node = node.left  # type: ignore[assignment]
            else:
                node = node.right  # type: ignore[assignment]
        return float(node.weight)

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            out[i] = self._predict_one(x[i])
        return out


class SimpleXGBoostBinary:
    """Minimal XGBoost-like binary classifier (logistic objective, second-order tree boosting)."""

    def __init__(self, config: XGBConfig):
        self.config = config
        self.base_score_: float = 0.0
        self.trees_: list[XGBTree] = []
        self.train_loss_history_: list[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray, verbose_every: int = 10) -> "SimpleXGBoostBinary":
        self.config.validate()

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be 2D")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x/y sample size mismatch")
        if not np.isfinite(x).all():
            raise ValueError("x contains NaN/Inf")
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN/Inf")
        if not np.all((y == 0.0) | (y == 1.0)):
            raise ValueError("y must be binary in {0,1}")

        rng = np.random.default_rng(self.config.random_state)
        n_samples, n_features = x.shape

        pos_rate = np.clip(np.mean(y), 1e-6, 1.0 - 1e-6)
        self.base_score_ = float(np.log(pos_rate / (1.0 - pos_rate)))

        raw_scores = np.full(n_samples, self.base_score_, dtype=np.float64)
        self.trees_.clear()
        self.train_loss_history_.clear()

        row_sample_size = max(2 * self.config.min_samples_leaf, int(n_samples * self.config.subsample))
        feat_sample_size = max(1, int(n_features * self.config.colsample_bytree))

        for m in range(self.config.n_estimators):
            p = sigmoid(raw_scores)
            g = p - y
            h = np.clip(p * (1.0 - p), 1e-8, None)

            if row_sample_size >= n_samples:
                row_idx = np.arange(n_samples)
            else:
                row_idx = np.sort(rng.choice(n_samples, size=row_sample_size, replace=False))

            feature_idx = np.sort(rng.choice(n_features, size=feat_sample_size, replace=False))

            tree = XGBTree(config=self.config, feature_indices=feature_idx).fit(x, g, h, row_idx)
            update = tree.predict(x)
            raw_scores += self.config.learning_rate * update

            self.trees_.append(tree)
            train_loss = binary_logloss(y, sigmoid(raw_scores))
            self.train_loss_history_.append(train_loss)

            if (m + 1) == 1 or (m + 1) % verbose_every == 0 or (m + 1) == self.config.n_estimators:
                print(f"iter={m + 1:03d} train_logloss={train_loss:.6f}")

        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        raw_scores = np.full(x.shape[0], self.base_score_, dtype=np.float64)
        for tree in self.trees_:
            raw_scores += self.config.learning_rate * tree.predict(x)
        return raw_scores

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p1 = sigmoid(self.decision_function(x))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(np.float64)


def main() -> None:
    config = XGBConfig(
        n_estimators=80,
        learning_rate=0.12,
        max_depth=3,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=1.0,
        min_samples_leaf=16,
        n_thresholds=18,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=42,
    )

    x_train, x_test, y_train, y_test = make_dataset(random_state=config.random_state)

    train_pos_rate = float(np.mean(y_train))
    baseline_train_prob = np.full_like(y_train, train_pos_rate, dtype=np.float64)
    baseline_test_prob = np.full_like(y_test, train_pos_rate, dtype=np.float64)
    majority = float(train_pos_rate >= 0.5)

    baseline_train_loss = binary_logloss(y_train, baseline_train_prob)
    baseline_test_loss = binary_logloss(y_test, baseline_test_prob)
    baseline_test_acc = accuracy(y_test, np.full_like(y_test, majority))

    print("=== Baseline (constant prior model) ===")
    print(f"baseline_train_logloss={baseline_train_loss:.6f}")
    print(f"baseline_test_logloss={baseline_test_loss:.6f}")
    print(f"baseline_test_accuracy={baseline_test_acc:.4f}")

    print("\n=== Training Simple XGBoost ===")
    model = SimpleXGBoostBinary(config).fit(x_train, y_train, verbose_every=10)

    train_prob = model.predict_proba(x_train)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_loss = binary_logloss(y_train, train_prob)
    test_loss = binary_logloss(y_test, test_prob)
    train_acc = accuracy(y_train, train_pred)
    test_acc = accuracy(y_test, test_pred)

    print("\n=== Final Metrics ===")
    print(f"n_trees={len(model.trees_)}")
    print(f"train_logloss={train_loss:.6f}")
    print(f"test_logloss={test_loss:.6f}")
    print(f"train_accuracy={train_acc:.4f}")
    print(f"test_accuracy={test_acc:.4f}")

    assert len(model.trees_) == config.n_estimators, "Tree count mismatch"
    assert np.all(np.isfinite(train_prob)) and np.all(np.isfinite(test_prob)), "Non-finite probabilities"
    assert model.train_loss_history_[0] > model.train_loss_history_[-1], "Training did not reduce logloss"
    assert test_loss < baseline_test_loss * 0.90, "Test logloss improvement is insufficient"
    assert test_acc > baseline_test_acc + 0.12, "Test accuracy improvement is insufficient"

    print("All checks passed.")


if __name__ == "__main__":
    main()
