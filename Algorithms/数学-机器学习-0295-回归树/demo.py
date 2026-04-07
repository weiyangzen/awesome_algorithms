"""Regression tree (CART-style) MVP implemented from scratch.

This script trains a regression tree using squared-error reduction,
prints training/testing metrics, and runs without interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TreeNode:
    is_leaf: bool
    prediction: float
    n_samples: int
    sse: float
    feature_index: int = -1
    threshold: float = 0.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


class RegressionTreeMVP:
    """Minimal CART regression tree using variance (SSE) split criterion."""

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_split: int = 8,
        min_samples_leaf: int = 4,
        min_impurity_decrease: float = 1e-8,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if min_impurity_decrease < 0.0:
            raise ValueError("min_impurity_decrease must be >= 0")

        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_impurity_decrease = float(min_impurity_decrease)

        self.root: Optional[TreeNode] = None
        self.n_features_in_: int = 0
        self._feature_gain: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RegressionTreeMVP":
        self._validate_xy(x, y)
        self.n_features_in_ = x.shape[1]
        self._feature_gain = np.zeros(self.n_features_in_, dtype=float)
        self.root = self._build_tree(x=x, y=y, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Model is not fitted yet.")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be 2D in predict, got shape={x.shape}")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature mismatch: fitted with {self.n_features_in_}, got {x.shape[1]}"
            )

        pred = np.empty(x.shape[0], dtype=float)
        for i in range(x.shape[0]):
            pred[i] = self._predict_one(x[i], self.root)
        return pred

    def tree_stats(self) -> Tuple[int, int, int]:
        if self.root is None:
            raise RuntimeError("Model is not fitted yet.")
        return self._count_nodes(self.root), self._count_leaves(self.root), self._depth(self.root)

    def feature_importances(self) -> np.ndarray:
        if self._feature_gain is None:
            raise RuntimeError("Model is not fitted yet.")
        total = float(np.sum(self._feature_gain))
        if total <= 0.0:
            return np.zeros_like(self._feature_gain)
        return self._feature_gain / total

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
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values")

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        n_samples = y.size
        prediction = float(np.mean(y))
        centered = y - prediction
        parent_sse = float(np.dot(centered, centered))

        stop = (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or parent_sse <= 1e-14
        )
        if stop:
            return TreeNode(
                is_leaf=True,
                prediction=prediction,
                n_samples=n_samples,
                sse=parent_sse,
            )

        best = self._best_split(x=x, y=y, parent_sse=parent_sse)
        if best is None:
            return TreeNode(
                is_leaf=True,
                prediction=prediction,
                n_samples=n_samples,
                sse=parent_sse,
            )

        feature_index, threshold, gain = best
        if gain < self.min_impurity_decrease:
            return TreeNode(
                is_leaf=True,
                prediction=prediction,
                n_samples=n_samples,
                sse=parent_sse,
            )

        assert self._feature_gain is not None
        self._feature_gain[feature_index] += gain

        left_mask = x[:, feature_index] <= threshold
        right_mask = ~left_mask

        left = self._build_tree(x[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(x[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            is_leaf=False,
            prediction=prediction,
            n_samples=n_samples,
            sse=parent_sse,
            feature_index=feature_index,
            threshold=threshold,
            left=left,
            right=right,
        )

    def _best_split(
        self,
        x: np.ndarray,
        y: np.ndarray,
        parent_sse: float,
    ) -> Optional[Tuple[int, float, float]]:
        n_samples, n_features = x.shape

        best_gain = -np.inf
        best_feature = -1
        best_threshold = 0.0

        for feature in range(n_features):
            values = x[:, feature]
            order = np.argsort(values, kind="mergesort")
            x_sorted = values[order]
            y_sorted = y[order]

            prefix_sum = np.cumsum(y_sorted)
            prefix_sq_sum = np.cumsum(y_sorted * y_sorted)

            total_sum = float(prefix_sum[-1])
            total_sq_sum = float(prefix_sq_sum[-1])

            start = self.min_samples_leaf
            end = n_samples - self.min_samples_leaf
            if start >= end:
                continue

            for i in range(start, end + 1):
                if i >= n_samples:
                    break
                if x_sorted[i - 1] == x_sorted[i]:
                    continue

                left_n = i
                right_n = n_samples - i

                left_sum = float(prefix_sum[i - 1])
                left_sq = float(prefix_sq_sum[i - 1])
                right_sum = total_sum - left_sum
                right_sq = total_sq_sum - left_sq

                left_sse = left_sq - (left_sum * left_sum) / left_n
                right_sse = right_sq - (right_sum * right_sum) / right_n
                child_sse = max(left_sse, 0.0) + max(right_sse, 0.0)
                gain = parent_sse - child_sse

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = 0.5 * (x_sorted[i - 1] + x_sorted[i])

        if best_feature < 0:
            return None
        return best_feature, float(best_threshold), float(best_gain)

    def _predict_one(self, row: np.ndarray, node: TreeNode) -> float:
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node.prediction

    def _count_nodes(self, node: TreeNode) -> int:
        if node.is_leaf:
            return 1
        assert node.left is not None and node.right is not None
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def _count_leaves(self, node: TreeNode) -> int:
        if node.is_leaf:
            return 1
        assert node.left is not None and node.right is not None
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _depth(self, node: TreeNode) -> int:
        if node.is_leaf:
            return 0
        assert node.left is not None and node.right is not None
        return 1 + max(self._depth(node.left), self._depth(node.right))


def make_piecewise_regression_data(
    seed: int = 2026,
    n_samples: int = 360,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-3.0, 3.0, size=n_samples)
    x1 = rng.uniform(-2.0, 2.0, size=n_samples)
    x2 = rng.normal(0.0, 1.0, size=n_samples)

    y = np.where(
        x0 < 0.0,
        1.5 + 0.7 * (x0 + 1.0) ** 2 - 0.9 * x1,
        -0.4 + np.sin(2.2 * x0) + 0.8 * (x1**2),
    )
    y += 0.15 * x2 + rng.normal(0.0, 0.12, size=n_samples)

    x = np.column_stack([x0, x1, x2])
    return x, y


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.30,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")

    rng = np.random.default_rng(seed)
    n_samples = x.shape[0]
    indices = rng.permutation(n_samples)
    test_size = int(round(n_samples * test_ratio))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 1e-15:
        return 1.0
    return 1.0 - ss_res / ss_tot


def baseline_mean_predictor(y_train: np.ndarray, n_test: int) -> np.ndarray:
    return np.full(n_test, fill_value=float(np.mean(y_train)), dtype=float)


def maybe_compare_with_sklearn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
) -> None:
    try:
        from sklearn.tree import DecisionTreeRegressor
    except Exception:
        print("sklearn check: skipped (scikit-learn not available)")
        return

    model = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=0,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print(
        "sklearn check: "
        f"MSE={mse(y_test, pred):.6f}, "
        f"R2={r2_score(y_test, pred):.6f}"
    )


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    x, y = make_piecewise_regression_data(seed=2026, n_samples=360)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.30, seed=11)

    model = RegressionTreeMVP(
        max_depth=5,
        min_samples_split=12,
        min_samples_leaf=6,
        min_impurity_decrease=1e-6,
    )
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    pred_base = baseline_mean_predictor(y_train, n_test=y_test.size)

    train_mse = mse(y_train, pred_train)
    test_mse = mse(y_test, pred_test)
    test_r2 = r2_score(y_test, pred_test)
    base_mse = mse(y_test, pred_base)
    nodes, leaves, depth = model.tree_stats()
    importances = model.feature_importances()

    print("=== Regression Tree MVP (CART, squared error) ===")
    print(f"train size={x_train.shape[0]}, test size={x_test.shape[0]}, features={x_train.shape[1]}")
    print(
        "hyperparameters: "
        f"max_depth={model.max_depth}, "
        f"min_samples_split={model.min_samples_split}, "
        f"min_samples_leaf={model.min_samples_leaf}, "
        f"min_impurity_decrease={model.min_impurity_decrease:.1e}"
    )
    print(f"tree stats: nodes={nodes}, leaves={leaves}, depth={depth}")
    print(f"train MSE={train_mse:.6f}")
    print(f"test  MSE={test_mse:.6f}")
    print(f"test   R2={test_r2:.6f}")
    print(f"baseline mean-predictor test MSE={base_mse:.6f}")
    print("feature importances (gain normalized):", np.array2string(importances, precision=4))

    print("\nSample predictions (first 8 test samples):")
    header = "idx | y_true     | y_pred"
    print(header)
    print("-" * len(header))
    for i in range(min(8, y_test.size)):
        print(f"{i:3d} | {y_test[i]:10.5f} | {pred_test[i]:10.5f}")

    maybe_compare_with_sklearn(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        max_depth=model.max_depth,
        min_samples_split=model.min_samples_split,
        min_samples_leaf=model.min_samples_leaf,
    )


if __name__ == "__main__":
    main()
