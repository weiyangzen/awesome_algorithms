"""C4.5-style decision tree classifier MVP.

This implementation focuses on source-level transparency:
- split criterion: information gain ratio
- feature types: continuous + categorical
- tree growth: recursive top-down induction
- no post-pruning (kept minimal for an auditable MVP)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class SplitDecision:
    feature_index: int
    feature_name: str
    feature_type: str
    threshold: Optional[float]
    gain_ratio: float
    info_gain: float
    split_info: float
    category_keys: Tuple[str, ...] = ()


@dataclass
class C45Node:
    is_leaf: bool
    prediction: int
    n_samples: int
    entropy: float
    class_counts: np.ndarray
    feature_index: int = -1
    feature_name: str = ""
    feature_type: str = ""
    threshold: Optional[float] = None
    gain_ratio: float = 0.0
    info_gain: float = 0.0
    split_info: float = 0.0
    children: Dict[str, "C45Node"] = field(default_factory=dict)


class C45ClassifierMVP:
    """A compact C4.5-style classifier with gain-ratio splits."""

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        min_gain_ratio: float = 1e-3,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1.")
        if min_gain_ratio < 0.0:
            raise ValueError("min_gain_ratio must be >= 0.")

        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_gain_ratio = float(min_gain_ratio)

        self.root: Optional[C45Node] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.feature_names_: List[str] = []
        self.feature_types_: List[str] = []

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        feature_types: Optional[Sequence[str]] = None,
    ) -> "C45ClassifierMVP":
        x_arr, y_encoded = self._validate_and_prepare(
            x=x,
            y=y,
            feature_names=feature_names,
            feature_types=feature_types,
        )

        categorical_available = {
            i for i, ftype in enumerate(self.feature_types_) if ftype == "categorical"
        }
        self.root = self._build_tree(
            x=x_arr,
            y=y_encoded,
            depth=0,
            categorical_available=categorical_available,
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None or self.classes_ is None:
            raise RuntimeError("Model is not fitted yet.")

        x_arr = np.asarray(x, dtype=object)
        if x_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={x_arr.shape}.")
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature mismatch: fitted with {self.n_features_in_}, got {x_arr.shape[1]}."
            )

        pred_encoded = np.empty(x_arr.shape[0], dtype=int)
        for i in range(x_arr.shape[0]):
            pred_encoded[i] = self._predict_encoded_one(x_arr[i], self.root)
        return self.classes_[pred_encoded]

    def tree_stats(self) -> Tuple[int, int, int]:
        if self.root is None:
            raise RuntimeError("Model is not fitted yet.")
        return self._count_nodes(self.root), self._count_leaves(self.root), self._depth(self.root)

    def export_rules(self, max_rules: int = 20) -> List[str]:
        if self.root is None or self.classes_ is None:
            raise RuntimeError("Model is not fitted yet.")
        rules: List[str] = []
        self._collect_rules(self.root, conditions=[], out=rules)
        return rules[:max_rules]

    def _validate_and_prepare(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[Sequence[str]],
        feature_types: Optional[Sequence[str]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=object)
        y_arr = np.asarray(y)

        if x_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={x_arr.shape}.")
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1D, got shape={y_arr.shape}.")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"Sample mismatch: X has {x_arr.shape[0]} rows, y has {y_arr.shape[0]}.")
        if x_arr.shape[0] == 0 or x_arr.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature.")

        self.n_features_in_ = int(x_arr.shape[1])

        if feature_names is None:
            self.feature_names_ = [f"x{i}" for i in range(self.n_features_in_)]
        else:
            if len(feature_names) != self.n_features_in_:
                raise ValueError("feature_names length must match number of columns in X.")
            self.feature_names_ = [str(name) for name in feature_names]

        if feature_types is None:
            self.feature_types_ = ["continuous"] * self.n_features_in_
        else:
            if len(feature_types) != self.n_features_in_:
                raise ValueError("feature_types length must match number of columns in X.")
            normalized: List[str] = []
            for item in feature_types:
                text = str(item).strip().lower()
                if text not in {"continuous", "categorical"}:
                    raise ValueError(f"Unknown feature type: {item}. Use continuous/categorical.")
                normalized.append(text)
            self.feature_types_ = normalized

        for j, ftype in enumerate(self.feature_types_):
            if ftype == "continuous":
                col = np.asarray(x_arr[:, j], dtype=float)
                if not np.all(np.isfinite(col)):
                    raise ValueError(
                        f"Continuous feature '{self.feature_names_[j]}' contains non-finite values."
                    )

        classes, y_encoded = np.unique(y_arr, return_inverse=True)
        if classes.size < 2:
            raise ValueError("At least two classes are required.")
        self.classes_ = classes
        self.n_classes_ = int(classes.size)

        return x_arr, y_encoded.astype(int)

    @staticmethod
    def _entropy_from_counts(counts: np.ndarray) -> float:
        total = float(np.sum(counts))
        if total <= 0.0:
            return 0.0
        probs = counts[counts > 0.0] / total
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def _split_info_from_sizes(part_sizes: Sequence[int], total: int) -> float:
        probs = np.asarray(part_sizes, dtype=float) / float(total)
        probs = probs[probs > 0.0]
        if probs.size == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def _cat_key(value: object) -> str:
        return str(value)

    def _build_tree(
        self,
        x: np.ndarray,
        y: np.ndarray,
        depth: int,
        categorical_available: Set[int],
    ) -> C45Node:
        n_samples = int(y.size)
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        entropy = self._entropy_from_counts(counts)
        prediction = int(np.argmax(counts))

        stop = (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or entropy <= 1e-12
        )
        if stop:
            return C45Node(
                is_leaf=True,
                prediction=prediction,
                n_samples=n_samples,
                entropy=entropy,
                class_counts=counts,
            )

        best = self._best_split(
            x=x,
            y=y,
            parent_entropy=entropy,
            categorical_available=categorical_available,
        )
        if best is None or best.gain_ratio < self.min_gain_ratio:
            return C45Node(
                is_leaf=True,
                prediction=prediction,
                n_samples=n_samples,
                entropy=entropy,
                class_counts=counts,
            )

        node = C45Node(
            is_leaf=False,
            prediction=prediction,
            n_samples=n_samples,
            entropy=entropy,
            class_counts=counts,
            feature_index=best.feature_index,
            feature_name=best.feature_name,
            feature_type=best.feature_type,
            threshold=best.threshold,
            gain_ratio=best.gain_ratio,
            info_gain=best.info_gain,
            split_info=best.split_info,
        )

        j = best.feature_index
        if best.feature_type == "continuous":
            assert best.threshold is not None
            col = np.asarray(x[:, j], dtype=float)
            left_mask = col <= best.threshold
            right_mask = ~left_mask
            if not np.any(left_mask) or not np.any(right_mask):
                node.is_leaf = True
                node.children = {}
                return node

            node.children["<="] = self._build_tree(
                x=x[left_mask],
                y=y[left_mask],
                depth=depth + 1,
                categorical_available=set(categorical_available),
            )
            node.children[">"] = self._build_tree(
                x=x[right_mask],
                y=y[right_mask],
                depth=depth + 1,
                categorical_available=set(categorical_available),
            )
            return node

        col_keys = np.asarray([self._cat_key(v) for v in x[:, j]], dtype=object)
        next_available = set(categorical_available)
        next_available.discard(j)

        for key in best.category_keys:
            mask = col_keys == key
            if not np.any(mask):
                continue
            node.children[key] = self._build_tree(
                x=x[mask],
                y=y[mask],
                depth=depth + 1,
                categorical_available=set(next_available),
            )

        if len(node.children) < 2:
            node.is_leaf = True
            node.children = {}
        return node

    def _best_split(
        self,
        x: np.ndarray,
        y: np.ndarray,
        parent_entropy: float,
        categorical_available: Set[int],
    ) -> Optional[SplitDecision]:
        n_features = x.shape[1]
        best: Optional[SplitDecision] = None

        for j in range(n_features):
            ftype = self.feature_types_[j]
            if ftype == "categorical" and j not in categorical_available:
                continue

            if ftype == "continuous":
                candidate = self._best_continuous_split(
                    col=x[:, j],
                    y=y,
                    parent_entropy=parent_entropy,
                )
                if candidate is None:
                    continue
                threshold, gain_ratio, info_gain, split_info = candidate
                decision = SplitDecision(
                    feature_index=j,
                    feature_name=self.feature_names_[j],
                    feature_type=ftype,
                    threshold=threshold,
                    gain_ratio=gain_ratio,
                    info_gain=info_gain,
                    split_info=split_info,
                )
            else:
                candidate = self._best_categorical_split(
                    col=x[:, j],
                    y=y,
                    parent_entropy=parent_entropy,
                )
                if candidate is None:
                    continue
                gain_ratio, info_gain, split_info, keys = candidate
                decision = SplitDecision(
                    feature_index=j,
                    feature_name=self.feature_names_[j],
                    feature_type=ftype,
                    threshold=None,
                    gain_ratio=gain_ratio,
                    info_gain=info_gain,
                    split_info=split_info,
                    category_keys=keys,
                )

            if best is None:
                best = decision
                continue

            better = (
                decision.gain_ratio > best.gain_ratio + 1e-12
                or (
                    abs(decision.gain_ratio - best.gain_ratio) <= 1e-12
                    and decision.info_gain > best.info_gain + 1e-12
                )
                or (
                    abs(decision.gain_ratio - best.gain_ratio) <= 1e-12
                    and abs(decision.info_gain - best.info_gain) <= 1e-12
                    and decision.feature_index < best.feature_index
                )
            )
            if better:
                best = decision

        return best

    def _best_continuous_split(
        self,
        col: np.ndarray,
        y: np.ndarray,
        parent_entropy: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        x_num = np.asarray(col, dtype=float)
        n = x_num.size
        if n < 2:
            return None

        order = np.argsort(x_num, kind="mergesort")
        x_sorted = x_num[order]
        y_sorted = y[order]

        left_counts = np.zeros(self.n_classes_, dtype=float)
        right_counts = np.bincount(y_sorted, minlength=self.n_classes_).astype(float)

        best_threshold: Optional[float] = None
        best_gain_ratio = -np.inf
        best_info_gain = -np.inf
        best_split_info = 0.0

        for i in range(n - 1):
            cls = int(y_sorted[i])
            left_counts[cls] += 1.0
            right_counts[cls] -= 1.0

            left_n = i + 1
            right_n = n - left_n
            if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                continue
            if x_sorted[i] == x_sorted[i + 1]:
                continue
            if y_sorted[i] == y_sorted[i + 1]:
                continue

            ent_left = self._entropy_from_counts(left_counts)
            ent_right = self._entropy_from_counts(right_counts)
            weighted_entropy = (left_n * ent_left + right_n * ent_right) / n
            info_gain = parent_entropy - weighted_entropy
            if info_gain <= 1e-15:
                continue

            split_info = self._split_info_from_sizes([left_n, right_n], total=n)
            if split_info <= 1e-15:
                continue
            gain_ratio = info_gain / split_info

            if (
                gain_ratio > best_gain_ratio + 1e-12
                or (
                    abs(gain_ratio - best_gain_ratio) <= 1e-12
                    and info_gain > best_info_gain + 1e-12
                )
            ):
                best_gain_ratio = gain_ratio
                best_info_gain = info_gain
                best_split_info = split_info
                best_threshold = 0.5 * float(x_sorted[i] + x_sorted[i + 1])

        if best_threshold is None:
            return None
        return best_threshold, float(best_gain_ratio), float(best_info_gain), float(best_split_info)

    def _best_categorical_split(
        self,
        col: np.ndarray,
        y: np.ndarray,
        parent_entropy: float,
    ) -> Optional[Tuple[float, float, float, Tuple[str, ...]]]:
        keys = np.asarray([self._cat_key(v) for v in col], dtype=object)
        unique_keys = np.unique(keys)
        if unique_keys.size <= 1:
            return None

        n = keys.size
        part_sizes: List[int] = []
        weighted_entropy = 0.0

        for key in unique_keys:
            mask = keys == key
            n_k = int(np.sum(mask))
            if n_k < self.min_samples_leaf:
                return None
            counts_k = np.bincount(y[mask], minlength=self.n_classes_).astype(float)
            ent_k = self._entropy_from_counts(counts_k)
            weighted_entropy += (n_k / n) * ent_k
            part_sizes.append(n_k)

        info_gain = parent_entropy - weighted_entropy
        if info_gain <= 1e-15:
            return None

        split_info = self._split_info_from_sizes(part_sizes, total=n)
        if split_info <= 1e-15:
            return None

        gain_ratio = info_gain / split_info
        return (
            float(gain_ratio),
            float(info_gain),
            float(split_info),
            tuple(str(k) for k in unique_keys.tolist()),
        )

    def _predict_encoded_one(self, row: np.ndarray, node: C45Node) -> int:
        while not node.is_leaf:
            if node.feature_type == "continuous":
                assert node.threshold is not None
                value = float(row[node.feature_index])
                branch_key = "<=" if value <= node.threshold else ">"
                next_node = node.children.get(branch_key)
            else:
                key = self._cat_key(row[node.feature_index])
                next_node = node.children.get(key)

            if next_node is None:
                return node.prediction
            node = next_node
        return node.prediction

    def _collect_rules(self, node: C45Node, conditions: List[str], out: List[str]) -> None:
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted yet.")
        if node.is_leaf:
            if conditions:
                prefix = " AND ".join(conditions)
            else:
                prefix = "TRUE"
            label = self.classes_[node.prediction]
            out.append(f"IF {prefix} THEN class={label} (n={node.n_samples})")
            return

        if node.feature_type == "continuous":
            assert node.threshold is not None
            left_cond = f"{node.feature_name} <= {node.threshold:.4f}"
            right_cond = f"{node.feature_name} > {node.threshold:.4f}"
            left = node.children.get("<=")
            right = node.children.get(">")
            if left is not None:
                self._collect_rules(left, conditions + [left_cond], out)
            if right is not None:
                self._collect_rules(right, conditions + [right_cond], out)
            return

        for key in sorted(node.children.keys()):
            cond = f"{node.feature_name} == '{key}'"
            self._collect_rules(node.children[key], conditions + [cond], out)

    def _count_nodes(self, node: C45Node) -> int:
        total = 1
        for child in node.children.values():
            total += self._count_nodes(child)
        return total

    def _count_leaves(self, node: C45Node) -> int:
        if node.is_leaf:
            return 1
        total = 0
        for child in node.children.values():
            total += self._count_leaves(child)
        return total

    def _depth(self, node: C45Node) -> int:
        if node.is_leaf:
            return 0
        child_depths = [self._depth(child) for child in node.children.values()]
        return 1 + max(child_depths, default=0)


def make_mixed_classification_data(
    seed: int = 2026,
    n_samples: int = 420,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    rng = np.random.default_rng(seed)

    x1 = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    x2 = rng.normal(loc=0.0, scale=1.2, size=n_samples)
    x3 = rng.uniform(-2.0, 2.0, size=n_samples)

    soil = np.where(x3 < -0.5, "clay", np.where(x3 < 0.8, "silt", "sand"))
    weather = rng.choice(["sunny", "cloudy", "rainy"], size=n_samples, p=[0.45, 0.35, 0.20])

    noise = rng.normal(loc=0.0, scale=0.20, size=n_samples)
    score = (
        1.20 * x1
        - 0.95 * x2
        + 0.85 * (soil == "sand").astype(float)
        - 0.55 * (weather == "rainy").astype(float)
        + 0.45 * (weather == "sunny").astype(float)
        + noise
    )

    y = np.zeros(n_samples, dtype=int)
    y[score > -0.15] = 1
    y[score > 1.00] = 2

    x = np.column_stack([x1, x2, soil, weather])
    feature_names = ["x1", "x2", "soil_type", "weather"]
    feature_types = ["continuous", "continuous", "categorical", "categorical"]
    return x, y, feature_names, feature_types


def stratified_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1).")

    rng = np.random.default_rng(seed)
    unique_classes = np.unique(y)
    test_indices: List[int] = []

    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_cls = cls_idx.size
        n_test_cls = max(1, int(round(n_cls * test_ratio)))
        n_test_cls = min(n_test_cls, n_cls - 1)
        test_indices.extend(cls_idx[:n_test_cls].tolist())

    test_idx = np.array(sorted(test_indices), dtype=int)
    all_idx = np.arange(y.size)
    train_mask = np.ones(y.size, dtype=bool)
    train_mask[test_idx] = False
    train_idx = all_idx[train_mask]

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores: List[float] = []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def confusion_matrix_int(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def main() -> None:
    x, y, feature_names, feature_types = make_mixed_classification_data(seed=2026, n_samples=420)
    x_train, x_test, y_train, y_test = stratified_train_test_split(
        x=x,
        y=y,
        test_ratio=0.3,
        seed=42,
    )

    model = C45ClassifierMVP(
        max_depth=6,
        min_samples_split=6,
        min_samples_leaf=3,
        min_gain_ratio=1e-3,
    )
    model.fit(
        x=x_train,
        y=y_train,
        feature_names=feature_names,
        feature_types=feature_types,
    )

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = macro_f1_score(y_test, test_pred, n_classes=3)
    cm = confusion_matrix_int(y_test, test_pred, n_classes=3)
    n_nodes, n_leaves, tree_depth = model.tree_stats()

    print("=== C4.5 MVP Demo ===")
    print(f"train shape: {x_train.shape}, test shape: {x_test.shape}")
    print(
        "hyperparameters: "
        f"max_depth={model.max_depth}, "
        f"min_samples_split={model.min_samples_split}, "
        f"min_samples_leaf={model.min_samples_leaf}, "
        f"min_gain_ratio={model.min_gain_ratio}"
    )
    print(f"tree stats: nodes={n_nodes}, leaves={n_leaves}, depth={tree_depth}")
    print(f"train accuracy: {train_acc:.4f}")
    print(f"test accuracy: {test_acc:.4f}")
    print(f"test macro_f1: {test_f1:.4f}")
    print("confusion matrix (rows=true, cols=pred):")
    print(cm)

    print("\nSample rules (first 12):")
    rules = model.export_rules(max_rules=12)
    for idx, rule in enumerate(rules, start=1):
        print(f"{idx:02d}. {rule}")

    print("\nSample predictions (first 8 test rows):")
    for i in range(min(8, x_test.shape[0])):
        print(
            f"idx={i:02d} true={int(y_test[i])} pred={int(test_pred[i])} "
            f"x1={float(x_test[i, 0]):+.3f} x2={float(x_test[i, 1]):+.3f} "
            f"soil={x_test[i, 2]} weather={x_test[i, 3]}"
        )

    if not np.isfinite(train_acc) or not np.isfinite(test_acc) or not np.isfinite(test_f1):
        raise RuntimeError("Non-finite metric encountered.")
    if test_acc < 0.72:
        raise RuntimeError(f"Test accuracy too low for this dataset: {test_acc:.4f}")
    if test_f1 < 0.68:
        raise RuntimeError(f"Macro-F1 too low for this dataset: {test_f1:.4f}")


if __name__ == "__main__":
    main()
