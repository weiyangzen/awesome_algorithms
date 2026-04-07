"""Minimal runnable MVP for LightGBM-style binary classification.

This demo implements a compact, source-visible approximation of core LightGBM ideas:
- histogram-based feature binning
- leaf-wise (best-first) tree growth
- second-order (gradient/hessian) split gain and leaf value
- gradient boosting with logistic loss

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LightGBMConfig:
    n_estimators: int = 80
    learning_rate: float = 0.1
    num_leaves: int = 15
    max_depth: int = -1
    max_bin: int = 63
    min_data_in_leaf: int = 20
    min_sum_hessian_in_leaf: float = 1e-3
    lambda_l2: float = 1.0
    min_gain_to_split: float = 0.0
    feature_fraction: float = 0.9
    bagging_fraction: float = 1.0
    random_state: int = 42

    def validate(self) -> None:
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be >= 2")
        if self.max_depth == 0 or self.max_depth < -1:
            raise ValueError("max_depth must be -1 or positive")
        if self.max_bin < 8:
            raise ValueError("max_bin must be >= 8")
        if self.min_data_in_leaf < 1:
            raise ValueError("min_data_in_leaf must be positive")
        if self.min_sum_hessian_in_leaf <= 0.0:
            raise ValueError("min_sum_hessian_in_leaf must be positive")
        if self.lambda_l2 < 0.0:
            raise ValueError("lambda_l2 must be >= 0")
        if not (0.0 < self.feature_fraction <= 1.0):
            raise ValueError("feature_fraction must be in (0, 1]")
        if not (0.0 < self.bagging_fraction <= 1.0):
            raise ValueError("bagging_fraction must be in (0, 1]")


def sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_true == 1.0) & (y_pred == 1.0)))
    tn = int(np.sum((y_true == 0.0) & (y_pred == 0.0)))
    fp = int(np.sum((y_true == 0.0) & (y_pred == 1.0)))
    fn = int(np.sum((y_true == 1.0) & (y_pred == 0.0)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2.0 * precision * recall) / max(1e-12, precision + recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


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

    n_test0 = int(idx0.size * test_size)
    n_test1 = int(idx1.size * test_size)

    test_idx = np.concatenate([idx0[:n_test0], idx1[:n_test1]])
    train_idx = np.concatenate([idx0[n_test0:], idx1[n_test1:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def make_dataset(random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_samples = 2200
    x = rng.normal(0.0, 1.0, size=(n_samples, 12))

    signal = (
        1.4 * x[:, 0]
        - 1.2 * x[:, 1]
        + 0.9 * (x[:, 2] > 0.0).astype(np.float64)
        + 0.8 * np.sin(1.7 * x[:, 3])
        + 0.7 * x[:, 4] * x[:, 5]
        - 0.6 * (x[:, 6] ** 2)
        + 0.5 * (x[:, 7] > x[:, 8]).astype(np.float64)
        + rng.normal(0.0, 0.85, size=n_samples)
    )
    y_prob = sigmoid(signal)
    y = rng.binomial(1, y_prob).astype(np.float64)

    return stratified_split(x, y, test_size=0.25, random_state=random_state)


class QuantileBinMapper:
    """Quantile-based binning used by histogram split search."""

    def __init__(self, max_bin: int) -> None:
        self.max_bin = max_bin
        self.bin_edges_: list[np.ndarray] = []

    def fit(self, x: np.ndarray) -> "QuantileBinMapper":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be 2D")

        n_features = x.shape[1]
        self.bin_edges_ = []
        quantiles = np.linspace(0.0, 1.0, self.max_bin + 1)[1:-1]

        for j in range(n_features):
            col = x[:, j]
            edges = np.unique(np.quantile(col, quantiles))
            self.bin_edges_.append(edges.astype(np.float64))

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.bin_edges_:
            raise RuntimeError("bin mapper is not fitted")

        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be 2D")

        n_samples, n_features = x.shape
        if n_features != len(self.bin_edges_):
            raise ValueError("feature count mismatch with fitted bin mapper")

        x_bin = np.zeros((n_samples, n_features), dtype=np.int32)
        for j, edges in enumerate(self.bin_edges_):
            x_bin[:, j] = np.searchsorted(edges, x[:, j], side="right")

        return x_bin

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def n_bins_per_feature(self) -> list[int]:
        return [edges.size + 1 for edges in self.bin_edges_]


@dataclass
class TreeNode:
    is_leaf: bool = True
    value: float = 0.0
    feature_index: int = -1
    threshold_bin: int = -1
    left_child: int = -1
    right_child: int = -1


@dataclass
class SplitInfo:
    gain: float
    feature_index: int
    threshold_bin: int
    left_indices: np.ndarray
    right_indices: np.ndarray


class LeafWiseHistogramTree:
    """Single tree trained with LightGBM-style histogram split gain."""

    def __init__(
        self,
        num_leaves: int,
        max_depth: int,
        min_data_in_leaf: int,
        min_sum_hessian_in_leaf: float,
        lambda_l2: float,
        min_gain_to_split: float,
        n_bins_per_feature: list[int],
    ) -> None:
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.lambda_l2 = lambda_l2
        self.min_gain_to_split = min_gain_to_split
        self.n_bins_per_feature = n_bins_per_feature

        self.nodes_: dict[int, TreeNode] = {0: TreeNode(is_leaf=True)}

    def _leaf_value(self, grad_sum: float, hess_sum: float) -> float:
        return float(-grad_sum / (hess_sum + self.lambda_l2))

    def _split_gain(
        self,
        g_left: float,
        h_left: float,
        g_right: float,
        h_right: float,
        g_total: float,
        h_total: float,
    ) -> float:
        parent_term = (g_total * g_total) / (h_total + self.lambda_l2)
        left_term = (g_left * g_left) / (h_left + self.lambda_l2)
        right_term = (g_right * g_right) / (h_right + self.lambda_l2)
        return float(0.5 * (left_term + right_term - parent_term) - self.min_gain_to_split)

    def _find_best_split(
        self,
        x_bin: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        sample_indices: np.ndarray,
        candidate_features: np.ndarray,
    ) -> SplitInfo | None:
        n_samples = sample_indices.size
        if n_samples < 2 * self.min_data_in_leaf:
            return None

        g_total = float(np.sum(grad[sample_indices]))
        h_total = float(np.sum(hess[sample_indices]))
        if h_total < 2.0 * self.min_sum_hessian_in_leaf:
            return None

        best_gain = -np.inf
        best_feature = -1
        best_threshold = -1

        for feature_idx in candidate_features:
            n_bins = self.n_bins_per_feature[int(feature_idx)]
            if n_bins <= 1:
                continue

            bins = x_bin[sample_indices, int(feature_idx)]
            grad_hist = np.bincount(bins, weights=grad[sample_indices], minlength=n_bins)
            hess_hist = np.bincount(bins, weights=hess[sample_indices], minlength=n_bins)
            cnt_hist = np.bincount(bins, minlength=n_bins)

            g_left = 0.0
            h_left = 0.0
            cnt_left = 0

            for threshold_bin in range(n_bins - 1):
                g_left += float(grad_hist[threshold_bin])
                h_left += float(hess_hist[threshold_bin])
                cnt_left += int(cnt_hist[threshold_bin])

                cnt_right = n_samples - cnt_left
                if cnt_left < self.min_data_in_leaf or cnt_right < self.min_data_in_leaf:
                    continue

                h_right = h_total - h_left
                if h_left < self.min_sum_hessian_in_leaf or h_right < self.min_sum_hessian_in_leaf:
                    continue

                g_right = g_total - g_left
                gain = self._split_gain(g_left, h_left, g_right, h_right, g_total, h_total)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature_idx)
                    best_threshold = int(threshold_bin)

        if best_feature < 0 or not np.isfinite(best_gain) or best_gain <= 0.0:
            return None

        split_col = x_bin[sample_indices, best_feature]
        left_mask = split_col <= best_threshold
        left_indices = sample_indices[left_mask]
        right_indices = sample_indices[~left_mask]

        if left_indices.size < self.min_data_in_leaf or right_indices.size < self.min_data_in_leaf:
            return None

        return SplitInfo(
            gain=float(best_gain),
            feature_index=best_feature,
            threshold_bin=best_threshold,
            left_indices=left_indices,
            right_indices=right_indices,
        )

    def fit(
        self,
        x_bin: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        root_sample_indices: np.ndarray,
        candidate_features: np.ndarray,
    ) -> "LeafWiseHistogramTree":
        leaf_samples: dict[int, np.ndarray] = {0: root_sample_indices}
        leaf_depth: dict[int, int] = {0: 0}
        next_node_id = 1

        while len(leaf_samples) < self.num_leaves:
            best_leaf_id = -1
            best_split: SplitInfo | None = None

            for leaf_id, samples in leaf_samples.items():
                depth = leaf_depth[leaf_id]
                if self.max_depth > 0 and depth >= self.max_depth:
                    continue

                split = self._find_best_split(
                    x_bin=x_bin,
                    grad=grad,
                    hess=hess,
                    sample_indices=samples,
                    candidate_features=candidate_features,
                )

                if split is None:
                    continue
                if best_split is None or split.gain > best_split.gain:
                    best_split = split
                    best_leaf_id = leaf_id

            if best_split is None or best_leaf_id < 0:
                break

            left_id = next_node_id
            right_id = next_node_id + 1
            next_node_id += 2

            self.nodes_[best_leaf_id] = TreeNode(
                is_leaf=False,
                feature_index=best_split.feature_index,
                threshold_bin=best_split.threshold_bin,
                left_child=left_id,
                right_child=right_id,
            )
            self.nodes_[left_id] = TreeNode(is_leaf=True)
            self.nodes_[right_id] = TreeNode(is_leaf=True)

            parent_depth = leaf_depth[best_leaf_id]
            del leaf_samples[best_leaf_id]
            del leaf_depth[best_leaf_id]

            leaf_samples[left_id] = best_split.left_indices
            leaf_samples[right_id] = best_split.right_indices
            leaf_depth[left_id] = parent_depth + 1
            leaf_depth[right_id] = parent_depth + 1

        for leaf_id, samples in leaf_samples.items():
            g_sum = float(np.sum(grad[samples]))
            h_sum = float(np.sum(hess[samples]))
            self.nodes_[leaf_id].value = self._leaf_value(g_sum, h_sum)

        return self

    def predict_raw(self, x_bin: np.ndarray) -> np.ndarray:
        out = np.zeros(x_bin.shape[0], dtype=np.float64)
        for i in range(x_bin.shape[0]):
            node_id = 0
            while True:
                node = self.nodes_[node_id]
                if node.is_leaf:
                    out[i] = node.value
                    break
                if x_bin[i, node.feature_index] <= node.threshold_bin:
                    node_id = node.left_child
                else:
                    node_id = node.right_child
        return out


class SimpleLightGBMBinaryClassifier:
    """Compact, inspectable LightGBM-style binary classifier."""

    def __init__(self, config: LightGBMConfig) -> None:
        self.config = config
        self.bin_mapper_: QuantileBinMapper | None = None
        self.trees_: list[LeafWiseHistogramTree] = []
        self.init_raw_score_: float = 0.0
        self.train_loss_history_: list[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray, verbose_every: int = 10) -> "SimpleLightGBMBinaryClassifier":
        self.config.validate()

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be 2D")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y row mismatch")
        if not np.isfinite(x).all():
            raise ValueError("x contains NaN/Inf")
        if not np.isin(y, [0.0, 1.0]).all():
            raise ValueError("y must be binary values in {0, 1}")

        self.bin_mapper_ = QuantileBinMapper(max_bin=self.config.max_bin)
        x_bin = self.bin_mapper_.fit_transform(x)
        n_bins_per_feature = self.bin_mapper_.n_bins_per_feature()

        n_samples, n_features = x_bin.shape
        rng = np.random.default_rng(self.config.random_state)

        pos_rate = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
        self.init_raw_score_ = float(np.log(pos_rate / (1.0 - pos_rate)))

        raw_scores = np.full(n_samples, self.init_raw_score_, dtype=np.float64)
        self.trees_.clear()
        self.train_loss_history_.clear()

        for it in range(self.config.n_estimators):
            prob = sigmoid(raw_scores)
            grad = prob - y
            hess = np.clip(prob * (1.0 - prob), 1e-6, None)

            if self.config.feature_fraction < 1.0:
                n_feat_used = max(1, int(np.ceil(self.config.feature_fraction * n_features)))
                feature_idx = np.sort(rng.choice(n_features, size=n_feat_used, replace=False))
            else:
                feature_idx = np.arange(n_features, dtype=np.int32)

            if self.config.bagging_fraction < 1.0:
                bag_n = max(2 * self.config.min_data_in_leaf, int(np.ceil(self.config.bagging_fraction * n_samples)))
                sample_idx = np.sort(rng.choice(n_samples, size=bag_n, replace=False))
            else:
                sample_idx = np.arange(n_samples, dtype=np.int32)

            tree = LeafWiseHistogramTree(
                num_leaves=self.config.num_leaves,
                max_depth=self.config.max_depth,
                min_data_in_leaf=self.config.min_data_in_leaf,
                min_sum_hessian_in_leaf=self.config.min_sum_hessian_in_leaf,
                lambda_l2=self.config.lambda_l2,
                min_gain_to_split=self.config.min_gain_to_split,
                n_bins_per_feature=n_bins_per_feature,
            ).fit(
                x_bin=x_bin,
                grad=grad,
                hess=hess,
                root_sample_indices=sample_idx,
                candidate_features=feature_idx,
            )

            raw_scores += self.config.learning_rate * tree.predict_raw(x_bin)
            self.trees_.append(tree)

            train_loss = binary_logloss(y, sigmoid(raw_scores))
            self.train_loss_history_.append(train_loss)

            if (it + 1) == 1 or (it + 1) % verbose_every == 0 or (it + 1) == self.config.n_estimators:
                print(f"iter={it + 1:03d} train_logloss={train_loss:.6f}")

        return self

    def _ensure_fitted(self) -> None:
        if self.bin_mapper_ is None or not self.trees_:
            raise RuntimeError("model is not fitted")

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be 2D")
        if not np.isfinite(x).all():
            raise ValueError("x contains NaN/Inf")

        x_bin = self.bin_mapper_.transform(x)  # type: ignore[union-attr]
        raw_scores = np.full(x_bin.shape[0], self.init_raw_score_, dtype=np.float64)
        for tree in self.trees_:
            raw_scores += self.config.learning_rate * tree.predict_raw(x_bin)
        return raw_scores

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raw = self.decision_function(x)
        p1 = sigmoid(raw)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(np.float64)


def main() -> None:
    config = LightGBMConfig(
        n_estimators=80,
        learning_rate=0.1,
        num_leaves=15,
        max_depth=-1,
        max_bin=63,
        min_data_in_leaf=20,
        min_sum_hessian_in_leaf=1e-3,
        lambda_l2=1.0,
        min_gain_to_split=0.0,
        feature_fraction=0.9,
        bagging_fraction=1.0,
        random_state=42,
    )

    x_train, x_test, y_train, y_test = make_dataset(random_state=config.random_state)

    train_pos_rate = float(np.mean(y_train))
    baseline_train_prob = np.full(y_train.shape[0], train_pos_rate, dtype=np.float64)
    baseline_test_prob = np.full(y_test.shape[0], train_pos_rate, dtype=np.float64)
    majority_label = float(train_pos_rate >= 0.5)

    baseline_train_loss = binary_logloss(y_train, baseline_train_prob)
    baseline_test_loss = binary_logloss(y_test, baseline_test_prob)
    baseline_test_acc = accuracy(y_test, np.full(y_test.shape[0], majority_label, dtype=np.float64))

    print("=== Baseline (constant model) ===")
    print(f"baseline_train_logloss={baseline_train_loss:.6f}")
    print(f"baseline_test_logloss={baseline_test_loss:.6f}")
    print(f"baseline_test_accuracy={baseline_test_acc:.4f}")

    print("\n=== Training LightGBM-style model ===")
    model = SimpleLightGBMBinaryClassifier(config).fit(x_train, y_train, verbose_every=10)

    train_prob = model.predict_proba(x_train)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_loss = binary_logloss(y_train, train_prob)
    test_loss = binary_logloss(y_test, test_prob)
    train_acc = accuracy(y_train, train_pred)
    test_acc = accuracy(y_test, test_pred)
    m = binary_metrics(y_test, test_pred)

    print("\n=== Final Metrics ===")
    print(f"train_logloss={train_loss:.6f}")
    print(f"test_logloss={test_loss:.6f}")
    print(f"train_accuracy={train_acc:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    print(
        "test_precision={:.4f} test_recall={:.4f} test_f1={:.4f}".format(
            m["precision"], m["recall"], m["f1"]
        )
    )
    print(
        "confusion_counts: TP={tp:.0f} TN={tn:.0f} FP={fp:.0f} FN={fn:.0f}".format(
            tp=m["tp"],
            tn=m["tn"],
            fp=m["fp"],
            fn=m["fn"],
        )
    )

    assert len(model.trees_) == config.n_estimators, "tree count mismatch"
    assert np.all(np.isfinite(train_prob)) and np.all(np.isfinite(test_prob)), "non-finite probabilities"
    assert train_loss < baseline_train_loss * 0.82, "train loss did not improve enough"
    assert test_loss < baseline_test_loss * 0.90, "test loss did not improve enough"
    assert test_acc > baseline_test_acc + 0.14, "test accuracy below expectation"
    assert m["f1"] > 0.75, "test F1 below expectation"

    print("All checks passed.")


if __name__ == "__main__":
    main()
