"""Decision Tree (CART-style classifier) minimal runnable MVP.

The core tree training and inference logic is implemented from scratch with NumPy.
An optional scikit-learn comparison is included for sanity checking only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TreeNode:
    is_leaf: bool
    prediction: int
    proba: np.ndarray
    n_samples: int
    gini: float
    feature_index: int = -1
    threshold: float = 0.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


class DecisionTreeClassifierMVP:
    """Minimal CART decision tree classifier using Gini impurity."""

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        min_impurity_decrease: float = 1e-10,
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
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self._feature_gain: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifierMVP":
        x, y_encoded = self._validate_and_encode(x, y)
        self.n_features_in_ = x.shape[1]
        self._feature_gain = np.zeros(self.n_features_in_, dtype=float)
        self.root = self._build_tree(x=x, y=y_encoded, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None or self.classes_ is None:
            raise RuntimeError("Model is not fitted yet.")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={x.shape}")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature mismatch: fitted with {self.n_features_in_}, got {x.shape[1]}"
            )

        pred_encoded = np.empty(x.shape[0], dtype=int)
        for i in range(x.shape[0]):
            pred_encoded[i] = self._predict_one(x[i], self.root).prediction
        return self.classes_[pred_encoded]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Model is not fitted yet.")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={x.shape}")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature mismatch: fitted with {self.n_features_in_}, got {x.shape[1]}"
            )

        out = np.empty((x.shape[0], self.n_classes_), dtype=float)
        for i in range(x.shape[0]):
            out[i] = self._predict_one(x[i], self.root).proba
        return out

    def tree_stats(self) -> Tuple[int, int, int]:
        if self.root is None:
            raise RuntimeError("Model is not fitted yet.")
        return self._count_nodes(self.root), self._count_leaves(self.root), self._depth(self.root)

    def feature_importances(self) -> np.ndarray:
        if self._feature_gain is None:
            raise RuntimeError("Model is not fitted yet.")
        total_gain = float(np.sum(self._feature_gain))
        if total_gain <= 0.0:
            return np.zeros_like(self._feature_gain)
        return self._feature_gain / total_gain

    def _validate_and_encode(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
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

        classes, y_encoded = np.unique(y, return_inverse=True)
        if classes.size < 2:
            raise ValueError("At least two classes are required")

        self.classes_ = classes
        self.n_classes_ = int(classes.size)
        return x, y_encoded.astype(int)

    @staticmethod
    def _gini_from_counts(counts: np.ndarray) -> float:
        n = float(np.sum(counts))
        if n <= 0.0:
            return 0.0
        p = counts / n
        return float(1.0 - np.dot(p, p))

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        n_samples = y.size
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        gini = self._gini_from_counts(counts)
        prediction = int(np.argmax(counts))
        proba = counts / np.sum(counts)

        stop = (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or gini <= 1e-14
        )
        if stop:
            return TreeNode(
                is_leaf=True,
                prediction=prediction,
                proba=proba,
                n_samples=n_samples,
                gini=gini,
            )

        best = self._best_split(x=x, y=y, parent_gini=gini)
        if best is None:
            return TreeNode(
                is_leaf=True,
                prediction=prediction,
                proba=proba,
                n_samples=n_samples,
                gini=gini,
            )

        feature_index, threshold, gain = best
        if gain < self.min_impurity_decrease:
            return TreeNode(
                is_leaf=True,
                prediction=prediction,
                proba=proba,
                n_samples=n_samples,
                gini=gini,
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
            proba=proba,
            n_samples=n_samples,
            gini=gini,
            feature_index=feature_index,
            threshold=threshold,
            left=left,
            right=right,
        )

    def _best_split(
        self,
        x: np.ndarray,
        y: np.ndarray,
        parent_gini: float,
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

            left_counts = np.zeros(self.n_classes_, dtype=float)
            right_counts = np.bincount(y_sorted, minlength=self.n_classes_).astype(float)

            for i in range(1, n_samples):
                cls = y_sorted[i - 1]
                left_counts[cls] += 1.0
                right_counts[cls] -= 1.0

                left_n = i
                right_n = n_samples - i

                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue
                if x_sorted[i - 1] == x_sorted[i]:
                    continue

                gini_left = self._gini_from_counts(left_counts)
                gini_right = self._gini_from_counts(right_counts)
                weighted_child_gini = (left_n * gini_left + right_n * gini_right) / n_samples
                gain = parent_gini - weighted_child_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = 0.5 * (x_sorted[i - 1] + x_sorted[i])

        if best_feature < 0:
            return None
        return best_feature, float(best_threshold), float(best_gain)

    def _predict_one(self, row: np.ndarray, node: TreeNode) -> TreeNode:
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node

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


def make_multiclass_data(seed: int = 2026, n_samples: int = 480) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    x0 = rng.uniform(-3.0, 3.0, size=n_samples)
    x1 = rng.uniform(-2.5, 2.5, size=n_samples)
    x2 = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    x3 = rng.normal(loc=0.0, scale=1.2, size=n_samples)

    signal = (
        0.85 * x0
        - 0.55 * x1
        + 0.90 * (x0 * x1 > 1.1).astype(float)
        + 0.45 * np.sin(1.7 * x0)
        - 0.35 * (x2 > 0.2).astype(float)
        + 0.30 * x3
    )
    noise = rng.normal(loc=0.0, scale=0.35, size=n_samples)
    score = signal + noise

    q1, q2 = np.quantile(score, [1.0 / 3.0, 2.0 / 3.0])
    y = np.zeros(n_samples, dtype=int)
    y[score >= q1] = 1
    y[score >= q2] = 2

    x = np.column_stack([x0, x1, x2, x3])
    return x, y


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")

    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(round(n * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1_values = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1_values.append(f1)
    return float(np.mean(f1_values))


def confusion_matrix_int(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


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
        from sklearn.tree import DecisionTreeClassifier
    except Exception as exc:  # pragma: no cover
        print(f"[optional] scikit-learn unavailable, skip compare: {exc}")
        return

    clf = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=7,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    acc = accuracy_score(y_test, pred)
    f1 = macro_f1_score(y_test, pred, n_classes=int(np.max(y_train) + 1))
    print(f"[optional sklearn] test accuracy={acc:.4f}, macro_f1={f1:.4f}")


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    x, y = make_multiclass_data(seed=2026, n_samples=480)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.30, seed=99)

    model = DecisionTreeClassifierMVP(
        max_depth=5,
        min_samples_split=12,
        min_samples_leaf=5,
        min_impurity_decrease=1e-8,
    )
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = macro_f1_score(y_test, test_pred, n_classes=3)

    majority_class = int(np.bincount(y_train).argmax())
    baseline_pred = np.full_like(y_test, fill_value=majority_class)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    baseline_f1 = macro_f1_score(y_test, baseline_pred, n_classes=3)

    n_nodes, n_leaves, depth = model.tree_stats()
    importances = model.feature_importances()
    cm = confusion_matrix_int(y_test, test_pred, n_classes=3)

    print("=== Decision Tree (CART Classification) MVP ===")
    print(f"train shape={x_train.shape}, test shape={x_test.shape}")
    print(
        "hyperparameters: "
        f"max_depth={model.max_depth}, "
        f"min_samples_split={model.min_samples_split}, "
        f"min_samples_leaf={model.min_samples_leaf}, "
        f"min_impurity_decrease={model.min_impurity_decrease}"
    )
    print(f"tree stats: nodes={n_nodes}, leaves={n_leaves}, depth={depth}")
    print(f"train accuracy={train_acc:.4f}")
    print(f"test  accuracy={test_acc:.4f}")
    print(f"test  macro_f1={test_f1:.4f}")
    print(f"baseline majority test accuracy={baseline_acc:.4f}")
    print(f"baseline majority test macro_f1={baseline_f1:.4f}")
    print(f"feature importances={importances}")
    print("confusion matrix (rows=true class, cols=pred class):")
    print(cm)

    print("sample predictions (first 8 test samples):")
    sample_proba = model.predict_proba(x_test[:8])
    for i in range(8):
        print(
            f"  idx={i:02d}, true={y_test[i]}, pred={test_pred[i]}, "
            f"proba={np.round(sample_proba[i], 4)}"
        )

    maybe_compare_with_sklearn(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        max_depth=model.max_depth,
        min_samples_split=model.min_samples_split,
        min_samples_leaf=model.min_samples_leaf,
    )

    # Minimal quality gates for deterministic synthetic data.
    assert np.isfinite(test_acc), "test accuracy must be finite"
    assert np.isfinite(test_f1), "test macro_f1 must be finite"
    assert test_acc >= 0.66, f"test accuracy too low: {test_acc:.4f}"
    assert test_f1 >= 0.65, f"test macro_f1 too low: {test_f1:.4f}"
    assert test_acc > baseline_acc + 0.10, "model should beat majority baseline"
    print("All checks passed.")


if __name__ == "__main__":
    main()
