"""Minimal runnable MVP for Random Forest (MATH-0296).

Pure NumPy implementation:
- CART-style binary decision tree (classification)
- Random forest with bootstrap + random feature subspace
- OOB (out-of-bag) accuracy estimate
- Deterministic synthetic dataset and built-in quality checks
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForestFitSummary:
    n_estimators: int
    n_samples: int
    n_features: int
    max_features_used: int
    oob_coverage: float
    oob_accuracy: float


@dataclass
class TreeNode:
    is_leaf: bool
    proba: np.ndarray
    feature: int = -1
    threshold: float = 0.0
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


class SimpleDecisionTreeClassifier:
    """A compact CART-style binary split tree for classification."""

    def __init__(
        self,
        n_classes: int,
        max_depth: int = 8,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: int | None = None,
        n_thresholds: int = 9,
        random_state: int = 0,
    ) -> None:
        if n_classes <= 1:
            raise ValueError("n_classes must be >= 2")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if n_thresholds < 2:
            raise ValueError("n_thresholds must be >= 2")

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state

        self.root_: TreeNode | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have same number of rows")
        if not np.isfinite(x).all():
            raise ValueError("x contains NaN/Inf")
        if np.any(y < 0) or np.any(y >= self.n_classes):
            raise ValueError("y contains class ids out of [0, n_classes)")

        self._rng = np.random.default_rng(self.random_state)
        self.root_ = self._grow(x, y, depth=0)

    def _class_proba(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        return counts / max(1, y.size)

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        p = np.bincount(y).astype(float) / y.size
        return float(1.0 - np.sum(p * p))

    def _best_split(
        self, x: np.ndarray, y: np.ndarray, feature_indices: np.ndarray
    ) -> tuple[int, float, np.ndarray, np.ndarray] | None:
        best_feature = -1
        best_threshold = 0.0
        best_impurity = float("inf")
        best_left_mask = None

        n = y.size
        for feature in feature_indices:
            values = x[:, feature]
            v_min = float(np.min(values))
            v_max = float(np.max(values))
            if not np.isfinite(v_min) or not np.isfinite(v_max) or v_min == v_max:
                continue

            # Quantile thresholds keep this MVP compact and deterministic.
            q = np.linspace(0.1, 0.9, self.n_thresholds)
            thresholds = np.unique(np.quantile(values, q))
            for thr in thresholds:
                left_mask = values <= thr
                left_n = int(np.sum(left_mask))
                right_n = n - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[~left_mask]
                impurity = (left_n / n) * self._gini(y_left) + (right_n / n) * self._gini(y_right)

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = int(feature)
                    best_threshold = float(thr)
                    best_left_mask = left_mask

        if best_feature < 0 or best_left_mask is None:
            return None

        return best_feature, best_threshold, best_left_mask, ~best_left_mask

    def _grow(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        proba = self._class_proba(y)

        if (
            depth >= self.max_depth
            or y.size < self.min_samples_split
            or np.unique(y).size == 1
        ):
            return TreeNode(is_leaf=True, proba=proba)

        n_features = x.shape[1]
        max_features = n_features if self.max_features is None else min(self.max_features, n_features)
        feature_indices = self._rng.choice(n_features, size=max_features, replace=False)

        split = self._best_split(x, y, feature_indices)
        if split is None:
            return TreeNode(is_leaf=True, proba=proba)

        feature, threshold, left_mask, right_mask = split
        left_node = self._grow(x[left_mask], y[left_mask], depth + 1)
        right_node = self._grow(x[right_mask], y[right_mask], depth + 1)
        return TreeNode(
            is_leaf=False,
            proba=proba,
            feature=feature,
            threshold=threshold,
            left=left_node,
            right=right_node,
        )

    def _predict_one_proba(self, row: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("tree is not fitted")
        node = self.root_
        while not node.is_leaf:
            if row[node.feature] <= node.threshold:
                node = node.left  # type: ignore[assignment]
            else:
                node = node.right  # type: ignore[assignment]
        return node.proba

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if not np.isfinite(x).all():
            raise ValueError("x contains NaN/Inf")
        out = np.zeros((x.shape[0], self.n_classes), dtype=float)
        for i in range(x.shape[0]):
            out[i] = self._predict_one_proba(x[i])
        return out


class SimpleRandomForestClassifier:
    """A compact random forest classifier with explicit forest-level logic."""

    def __init__(
        self,
        n_estimators: int = 31,
        max_depth: int = 8,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: str | int = "sqrt",
        bootstrap: bool = True,
        random_state: int = 42,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self.trees_: list[SimpleDecisionTreeClassifier] = []
        self.bootstrap_indices_: list[np.ndarray] = []
        self._summary: ForestFitSummary | None = None

    def _resolve_max_features(self, n_features: int) -> int:
        if isinstance(self.max_features, int):
            if self.max_features <= 0:
                raise ValueError("max_features int must be positive")
            return min(self.max_features, n_features)
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        raise ValueError("max_features must be int or 'sqrt'")

    def fit(self, x: np.ndarray, y: np.ndarray) -> ForestFitSummary:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y size mismatch")
        if not np.isfinite(x).all():
            raise ValueError("x contains NaN/Inf")

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("need at least 2 classes")
        self.classes_ = classes

        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        y_idx = np.array([class_to_idx[v] for v in y], dtype=int)

        n_samples, n_features = x.shape
        k_features = self._resolve_max_features(n_features)

        self.trees_.clear()
        self.bootstrap_indices_.clear()

        oob_prob_sum = np.zeros((n_samples, classes.size), dtype=float)
        oob_count = np.zeros(n_samples, dtype=int)

        for t in range(self.n_estimators):
            rng = np.random.default_rng(self.random_state + t)

            if self.bootstrap:
                sample_idx = rng.integers(0, n_samples, size=n_samples)
            else:
                sample_idx = np.arange(n_samples)
                rng.shuffle(sample_idx)

            tree = SimpleDecisionTreeClassifier(
                n_classes=classes.size,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=k_features,
                n_thresholds=9,
                random_state=self.random_state + t,
            )
            tree.fit(x[sample_idx], y_idx[sample_idx])

            self.trees_.append(tree)
            self.bootstrap_indices_.append(sample_idx)

            if self.bootstrap:
                in_bag = np.zeros(n_samples, dtype=bool)
                in_bag[sample_idx] = True
                oob_idx = np.where(~in_bag)[0]
                if oob_idx.size > 0:
                    oob_prob_sum[oob_idx] += tree.predict_proba(x[oob_idx])
                    oob_count[oob_idx] += 1

        valid_oob = oob_count > 0
        if np.any(valid_oob):
            pred_idx = np.argmax(oob_prob_sum[valid_oob], axis=1)
            y_oob_true = y_idx[valid_oob]
            oob_accuracy = float(np.mean(pred_idx == y_oob_true))
            oob_coverage = float(np.mean(valid_oob))
        else:
            oob_accuracy = float("nan")
            oob_coverage = 0.0

        self._summary = ForestFitSummary(
            n_estimators=self.n_estimators,
            n_samples=n_samples,
            n_features=n_features,
            max_features_used=k_features,
            oob_coverage=oob_coverage,
            oob_accuracy=oob_accuracy,
        )
        return self._summary

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None or not self.trees_:
            raise RuntimeError("model is not fitted")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if not np.isfinite(x).all():
            raise ValueError("x contains NaN/Inf")

        prob_sum = np.zeros((x.shape[0], self.classes_.size), dtype=float)
        for tree in self.trees_:
            prob_sum += tree.predict_proba(x)
        return prob_sum / len(self.trees_)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("model is not fitted")
        pred_idx = np.argmax(self.predict_proba(x), axis=1)
        return self.classes_[pred_idx]

    @property
    def summary(self) -> ForestFitSummary:
        if self._summary is None:
            raise RuntimeError("model is not fitted")
        return self._summary


def stratified_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.25,
    random_state: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")
    rng = np.random.default_rng(random_state)

    test_indices: list[np.ndarray] = []
    train_indices: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(cls_idx.size * test_ratio)))
        test_indices.append(cls_idx[:n_test])
        train_indices.append(cls_idx[n_test:])

    test_idx = np.concatenate(test_indices)
    train_idx = np.concatenate(train_indices)
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def build_dataset(random_state: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_samples = 1200
    n_features = 20

    y = rng.integers(0, 2, size=n_samples)
    x = rng.normal(0.0, 1.0, size=(n_samples, n_features))

    # Add class-dependent signal on informative features.
    signal = (2 * y - 1).reshape(-1, 1)
    x[:, :6] += 1.15 * signal

    # Mild nonlinear interactions + noise perturbation.
    x[:, 6] += 0.7 * (x[:, 0] * x[:, 1] > 0).astype(float)
    x[:, 7] += 0.5 * np.sin(x[:, 2])

    # Label noise for realism.
    flip_mask = rng.random(n_samples) < 0.04
    y = np.where(flip_mask, 1 - y, y)

    return stratified_train_test_split(x, y, test_ratio=0.25, random_state=random_state)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    acc = (tp + tn) / max(1, y_true.size)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def run_baseline_tree(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    baseline = SimpleDecisionTreeClassifier(
        n_classes=2,
        max_depth=8,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features=x_train.shape[1],
        random_state=0,
    )
    baseline.fit(x_train, y_train)
    pred = np.argmax(baseline.predict_proba(x_test), axis=1)
    return float(np.mean(pred == y_test))


def main() -> None:
    x_train, x_test, y_train, y_test = build_dataset(random_state=7)

    forest = SimpleRandomForestClassifier(
        n_estimators=31,
        max_depth=8,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features="sqrt",
        bootstrap=True,
        random_state=2026,
    )
    summary = forest.fit(x_train, y_train)

    pred = forest.predict(x_test).astype(int)
    metrics = binary_metrics(y_test, pred)
    baseline_acc = run_baseline_tree(x_train, y_train, x_test, y_test)

    print("=== Random Forest MVP (MATH-0296) ===")
    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
    print(
        "Forest config: "
        f"n_estimators={summary.n_estimators}, max_depth={forest.max_depth}, "
        f"max_features={summary.max_features_used}"
    )
    print(
        "OOB: "
        f"coverage={summary.oob_coverage:.3f}, "
        f"accuracy={summary.oob_accuracy:.4f}"
    )
    print(f"Baseline single-tree test accuracy: {baseline_acc:.4f}")
    print(f"Random forest test accuracy:        {metrics['accuracy']:.4f}")
    print(f"Random forest precision:            {metrics['precision']:.4f}")
    print(f"Random forest recall:               {metrics['recall']:.4f}")
    print(f"Random forest F1-score:             {metrics['f1']:.4f}")
    print(
        "Confusion matrix counts: "
        f"TP={int(metrics['tp'])}, TN={int(metrics['tn'])}, "
        f"FP={int(metrics['fp'])}, FN={int(metrics['fn'])}"
    )

    # Quality gates for automated validation.
    if not np.isfinite(metrics["accuracy"]):
        raise RuntimeError("Non-finite test accuracy")
    if not np.isfinite(summary.oob_accuracy):
        raise RuntimeError("Non-finite OOB accuracy")
    if summary.oob_coverage < 0.60:
        raise RuntimeError(f"OOB coverage too low: {summary.oob_coverage:.3f}")
    if metrics["accuracy"] < 0.84:
        raise RuntimeError(f"Test accuracy too low: {metrics['accuracy']:.4f}")
    if metrics["accuracy"] + 1e-12 < baseline_acc - 0.02:
        raise RuntimeError(
            f"Forest underperforms baseline too much: "
            f"forest={metrics['accuracy']:.4f}, baseline={baseline_acc:.4f}"
        )

    print("All checks passed.")


if __name__ == "__main__":
    main()
