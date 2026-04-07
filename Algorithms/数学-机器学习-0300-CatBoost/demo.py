"""Minimal runnable MVP for CatBoost (MATH-0300).

This implementation is intentionally compact and explicit:
1) ordered target statistics for categorical features (to reduce leakage),
2) symmetric/oblivious trees as weak learners,
3) gradient boosting for binary log-loss.

It runs without interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def binary_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = np.clip(y_prob, EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    accuracy = float(np.mean(y_pred == y_true))

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    out = {
        "accuracy": accuracy,
        "f1": float(f1),
        "logloss": binary_logloss(y_true, y_prob),
    }

    try:
        from sklearn.metrics import roc_auc_score

        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auc"] = float("nan")

    return out


def stratified_train_test_split(
    y: np.ndarray,
    test_ratio: float = 0.30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")

    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = int(round(cls_idx.size * test_ratio))
        test_parts.append(cls_idx[:n_test])
        train_parts.append(cls_idx[n_test:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def make_synthetic_catboost_data(
    n_samples: int = 2200,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(seed)

    # Numeric block
    x0 = rng.normal(loc=0.0, scale=1.1, size=n_samples)
    x1 = rng.uniform(-2.0, 2.0, size=n_samples)
    x2 = rng.normal(loc=0.0, scale=0.8, size=n_samples)
    x_num = np.column_stack([x0, x1, x2]).astype(float)

    # Categorical block
    c0_values = np.array(["red", "blue", "green", "gold", "black"], dtype=object)
    c1_values = np.array(["S", "M", "L"], dtype=object)
    c0 = rng.choice(c0_values, size=n_samples, p=[0.22, 0.20, 0.24, 0.18, 0.16])
    c1 = rng.choice(c1_values, size=n_samples, p=[0.40, 0.35, 0.25])
    x_cat = np.column_stack([c0, c1]).astype(object)

    cat0_effect = {"red": 1.15, "blue": 0.45, "green": -0.15, "gold": -0.95, "black": 0.05}
    cat1_effect = {"S": -0.45, "M": 0.05, "L": 0.65}

    e0 = np.array([cat0_effect[v] for v in c0], dtype=float)
    e1 = np.array([cat1_effect[v] for v in c1], dtype=float)

    interaction = np.where((c0 == "red") & (c1 == "L"), 0.90, 0.0)
    interaction += np.where((c0 == "gold") & (c1 == "S"), -0.85, 0.0)

    raw = 1.30 * x0 - 0.85 * x1 + 0.50 * x2 + e0 + e1 + interaction
    raw += rng.normal(0.0, 0.55, size=n_samples)

    prob = sigmoid(raw)
    y = rng.binomial(1, prob).astype(int)

    num_names = ["num_x0", "num_x1", "num_x2"]
    cat_names = ["color", "size"]
    return x_num, x_cat, y, num_names, cat_names


class OrderedTargetEncoder:
    """Ordered target statistics encoding for categorical columns."""

    def __init__(self, prior_weight: float = 2.0, random_state: int = 42) -> None:
        if prior_weight <= 0.0:
            raise ValueError("prior_weight must be positive")
        self.prior_weight = float(prior_weight)
        self.random_state = int(random_state)

        self.global_mean_: float = 0.5
        self.category_stats_: list[dict[str, tuple[float, int]]] = []
        self.n_features_in_: int = 0

    @staticmethod
    def _as_2d_object(x_cat: np.ndarray) -> np.ndarray:
        arr = np.asarray(x_cat, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("categorical array must be 1D or 2D")
        return arr

    def fit_transform(self, x_cat: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_cat = self._as_2d_object(x_cat)
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if x_cat.shape[0] != y.shape[0]:
            raise ValueError("x_cat and y sample size mismatch")

        n_samples, n_features = x_cat.shape
        self.n_features_in_ = n_features
        self.global_mean_ = float(np.mean(y))
        self.category_stats_ = []

        encoded = np.zeros((n_samples, n_features), dtype=float)
        rng = np.random.default_rng(self.random_state)

        for j in range(n_features):
            col = x_cat[:, j].astype(str)
            order = rng.permutation(n_samples)

            running_sum: dict[str, float] = {}
            running_count: dict[str, int] = {}
            col_encoded = np.empty(n_samples, dtype=float)

            for idx in order:
                key = col[idx]
                s = running_sum.get(key, 0.0)
                c = running_count.get(key, 0)
                col_encoded[idx] = (s + self.prior_weight * self.global_mean_) / (c + self.prior_weight)
                running_sum[key] = s + float(y[idx])
                running_count[key] = c + 1

            encoded[:, j] = col_encoded

            stats_sum: dict[str, float] = {}
            stats_count: dict[str, int] = {}
            for key, target in zip(col, y):
                stats_sum[key] = stats_sum.get(key, 0.0) + float(target)
                stats_count[key] = stats_count.get(key, 0) + 1
            stats = {k: (stats_sum[k], stats_count[k]) for k in stats_sum}
            self.category_stats_.append(stats)

        return encoded

    def transform(self, x_cat: np.ndarray) -> np.ndarray:
        x_cat = self._as_2d_object(x_cat)
        if x_cat.shape[1] != self.n_features_in_:
            raise ValueError("categorical feature count mismatch")

        n_samples, n_features = x_cat.shape
        encoded = np.zeros((n_samples, n_features), dtype=float)

        for j in range(n_features):
            col = x_cat[:, j].astype(str)
            stats = self.category_stats_[j]
            for i, key in enumerate(col):
                if key in stats:
                    s, c = stats[key]
                    encoded[i, j] = (s + self.prior_weight * self.global_mean_) / (c + self.prior_weight)
                else:
                    encoded[i, j] = self.global_mean_

        return encoded


@dataclass
class ObliviousTree:
    splits: list[tuple[int, float]]
    leaf_values: np.ndarray

    def leaf_indices(self, x: np.ndarray) -> np.ndarray:
        idx = np.zeros(x.shape[0], dtype=np.int64)
        for feature, threshold in self.splits:
            go_right = (x[:, feature] > threshold).astype(np.int64)
            idx = (idx << 1) | go_right
        return idx

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        idx = self.leaf_indices(x)
        return self.leaf_values[idx]


def build_split_candidates(x: np.ndarray, max_bins: int = 8) -> list[tuple[int, float]]:
    candidates: list[tuple[int, float]] = []
    n_features = x.shape[1]

    for j in range(n_features):
        col = x[:, j]
        cmin = float(np.min(col))
        cmax = float(np.max(col))
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin == cmax:
            continue

        q = np.linspace(0.10, 0.90, max_bins - 1)
        thresholds = np.unique(np.quantile(col, q))
        thresholds = thresholds[(thresholds > cmin) & (thresholds < cmax)]

        for thr in thresholds:
            candidates.append((j, float(thr)))

    if not candidates:
        candidates = [(0, float(np.median(x[:, 0])))]

    return candidates


class CatBoostLikeBinaryClassifier:
    """Compact CatBoost-like binary classifier (ordered CTR + oblivious trees)."""

    def __init__(
        self,
        n_estimators: int = 70,
        depth: int = 3,
        learning_rate: float = 0.12,
        l2_leaf_reg: float = 3.0,
        min_data_in_leaf: int = 8,
        max_bins: int = 8,
        random_state: int = 42,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if l2_leaf_reg <= 0.0:
            raise ValueError("l2_leaf_reg must be positive")
        if min_data_in_leaf <= 0:
            raise ValueError("min_data_in_leaf must be positive")

        self.n_estimators = int(n_estimators)
        self.depth = int(depth)
        self.learning_rate = float(learning_rate)
        self.l2_leaf_reg = float(l2_leaf_reg)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.max_bins = int(max_bins)
        self.random_state = int(random_state)

        self.encoder_ = OrderedTargetEncoder(prior_weight=2.0, random_state=random_state)
        self.base_score_: float = 0.0
        self.trees_: list[ObliviousTree] = []
        self.history_: list[float] = []
        self.feature_names_: list[str] = []

    def _assemble_features(self, x_num: np.ndarray, x_cat_encoded: np.ndarray) -> np.ndarray:
        x_num = np.asarray(x_num, dtype=float)
        if x_num.ndim != 2:
            raise ValueError("x_num must be 2D")
        if not np.isfinite(x_num).all():
            raise ValueError("x_num contains NaN/Inf")
        return np.hstack([x_num, x_cat_encoded])

    def _fit_one_tree(self, x: np.ndarray, grad: np.ndarray, hess: np.ndarray) -> ObliviousTree:
        candidates = build_split_candidates(x, max_bins=self.max_bins)

        leaf_idx = np.zeros(x.shape[0], dtype=np.int64)
        chosen_splits: list[tuple[int, float]] = []

        for level in range(self.depth):
            best_score = -np.inf
            best_leaf_idx: np.ndarray | None = None
            best_split: tuple[int, float] | None = None

            for feature, threshold in candidates:
                right = (x[:, feature] > threshold).astype(np.int64)
                candidate_leaf_idx = (leaf_idx << 1) | right
                leaf_count = np.bincount(candidate_leaf_idx, minlength=1 << (level + 1))

                if np.any(leaf_count < self.min_data_in_leaf):
                    continue

                g_sum = np.bincount(candidate_leaf_idx, weights=grad, minlength=leaf_count.size)
                h_sum = np.bincount(candidate_leaf_idx, weights=hess, minlength=leaf_count.size)
                score = float(np.sum((g_sum * g_sum) / (h_sum + self.l2_leaf_reg)))

                if score > best_score:
                    best_score = score
                    best_leaf_idx = candidate_leaf_idx
                    best_split = (feature, threshold)

            if best_leaf_idx is None or best_split is None:
                break

            leaf_idx = best_leaf_idx
            chosen_splits.append(best_split)

        leaf_count = 1 << len(chosen_splits)
        g_sum = np.bincount(leaf_idx, weights=grad, minlength=leaf_count)
        h_sum = np.bincount(leaf_idx, weights=hess, minlength=leaf_count)
        leaf_values = g_sum / (h_sum + self.l2_leaf_reg)
        leaf_values = np.clip(leaf_values, -8.0, 8.0)

        return ObliviousTree(splits=chosen_splits, leaf_values=leaf_values.astype(float))

    def fit(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        y: np.ndarray,
        num_feature_names: list[str],
        cat_feature_names: list[str],
    ) -> "CatBoostLikeBinaryClassifier":
        y = np.asarray(y, dtype=int)
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if np.any((y != 0) & (y != 1)):
            raise ValueError("y must contain only 0/1")

        x_cat_encoded = self.encoder_.fit_transform(x_cat, y)
        x = self._assemble_features(x_num, x_cat_encoded)

        self.feature_names_ = list(num_feature_names) + [f"ctr({name})" for name in cat_feature_names]
        self.trees_.clear()
        self.history_.clear()

        pos_rate = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
        self.base_score_ = float(np.log(pos_rate / (1.0 - pos_rate)))

        raw_score = np.full(y.size, fill_value=self.base_score_, dtype=float)

        for _ in range(self.n_estimators):
            prob = sigmoid(raw_score)
            grad = y - prob
            hess = np.clip(prob * (1.0 - prob), 1e-6, None)

            tree = self._fit_one_tree(x=x, grad=grad, hess=hess)
            raw_score += self.learning_rate * tree.predict_raw(x)

            self.trees_.append(tree)
            self.history_.append(binary_logloss(y, sigmoid(raw_score)))

        return self

    def decision_function(self, x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
        x_cat_encoded = self.encoder_.transform(x_cat)
        x = self._assemble_features(x_num, x_cat_encoded)

        raw_score = np.full(x.shape[0], fill_value=self.base_score_, dtype=float)
        for tree in self.trees_:
            raw_score += self.learning_rate * tree.predict_raw(x)
        return raw_score

    def predict_proba(self, x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
        raw_score = self.decision_function(x_num, x_cat)
        prob1 = sigmoid(raw_score)
        return np.column_stack([1.0 - prob1, prob1])

    def predict(self, x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x_num, x_cat)[:, 1]
        return (proba >= 0.5).astype(int)


def maybe_compare_with_official_catboost(
    x_train_num: np.ndarray,
    x_train_cat: np.ndarray,
    y_train: np.ndarray,
    x_test_num: np.ndarray,
    x_test_cat: np.ndarray,
    y_test: np.ndarray,
) -> None:
    try:
        from catboost import CatBoostClassifier
    except Exception:
        print("official catboost package: skipped (not installed)")
        return

    x_train = np.column_stack([x_train_num.astype(object), x_train_cat.astype(object)])
    x_test = np.column_stack([x_test_num.astype(object), x_test_cat.astype(object)])
    cat_features = list(range(x_train_num.shape[1], x_train.shape[1]))

    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=120,
        learning_rate=0.10,
        depth=3,
        l2_leaf_reg=3.0,
        random_seed=2026,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_train, y_train, cat_features=cat_features)

    prob = model.predict_proba(x_test)[:, 1]
    m = classification_metrics(y_test, prob)
    print(
        "official catboost (optional): "
        f"acc={m['accuracy']:.4f}, f1={m['f1']:.4f}, "
        f"logloss={m['logloss']:.4f}, auc={m['auc']:.4f}"
    )


def print_history(history: list[float]) -> None:
    if not history:
        return

    try:
        import pandas as pd

        df = pd.DataFrame({"iter": np.arange(1, len(history) + 1), "train_logloss": history})
        if len(df) > 10:
            head = df.head(5)
            tail = df.tail(5)
            print("\nloss history (head):")
            print(head.to_string(index=False))
            print("\nloss history (tail):")
            print(tail.to_string(index=False))
        else:
            print("\nloss history:")
            print(df.to_string(index=False))
    except Exception:
        print("\nloss history (first/last):")
        print("first:", [round(v, 5) for v in history[:5]])
        print("last:", [round(v, 5) for v in history[-5:]])


def main() -> None:
    x_num, x_cat, y, num_names, cat_names = make_synthetic_catboost_data(n_samples=2200, seed=2026)

    train_idx, test_idx = stratified_train_test_split(y, test_ratio=0.30, seed=9)

    x_train_num = x_num[train_idx]
    x_test_num = x_num[test_idx]
    x_train_cat = x_cat[train_idx]
    x_test_cat = x_cat[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    baseline_prob = np.full(y_test.shape[0], fill_value=float(np.mean(y_train)), dtype=float)
    baseline_metrics = classification_metrics(y_test, baseline_prob)

    model = CatBoostLikeBinaryClassifier(
        n_estimators=70,
        depth=3,
        learning_rate=0.12,
        l2_leaf_reg=3.0,
        min_data_in_leaf=8,
        max_bins=8,
        random_state=42,
    )
    model.fit(
        x_num=x_train_num,
        x_cat=x_train_cat,
        y=y_train,
        num_feature_names=num_names,
        cat_feature_names=cat_names,
    )

    train_prob = model.predict_proba(x_train_num, x_train_cat)[:, 1]
    test_prob = model.predict_proba(x_test_num, x_test_cat)[:, 1]

    train_metrics = classification_metrics(y_train, train_prob)
    test_metrics = classification_metrics(y_test, test_prob)

    print("CatBoost-like MVP (ordered target statistics + oblivious trees)")
    print(f"train size={y_train.size}, test size={y_test.size}, positive rate(train)={np.mean(y_train):.4f}")

    print(
        "baseline test: "
        f"acc={baseline_metrics['accuracy']:.4f}, f1={baseline_metrics['f1']:.4f}, "
        f"logloss={baseline_metrics['logloss']:.4f}, auc={baseline_metrics['auc']:.4f}"
    )
    print(
        "model train:   "
        f"acc={train_metrics['accuracy']:.4f}, f1={train_metrics['f1']:.4f}, "
        f"logloss={train_metrics['logloss']:.4f}, auc={train_metrics['auc']:.4f}"
    )
    print(
        "model test:    "
        f"acc={test_metrics['accuracy']:.4f}, f1={test_metrics['f1']:.4f}, "
        f"logloss={test_metrics['logloss']:.4f}, auc={test_metrics['auc']:.4f}"
    )

    print_history(model.history_)

    maybe_compare_with_official_catboost(
        x_train_num=x_train_num,
        x_train_cat=x_train_cat,
        y_train=y_train,
        x_test_num=x_test_num,
        x_test_cat=x_test_cat,
        y_test=y_test,
    )

    # Deterministic quality checks for quick validation.
    if not np.isfinite(test_prob).all():
        raise RuntimeError("Non-finite probabilities detected")
    if len(model.history_) != model.n_estimators:
        raise RuntimeError("Unexpected number of boosting rounds")
    if model.history_[-1] >= model.history_[0]:
        raise RuntimeError("Training logloss did not improve")
    if test_metrics["logloss"] >= baseline_metrics["logloss"]:
        raise RuntimeError("Model failed to beat baseline logloss")
    if test_metrics["accuracy"] <= baseline_metrics["accuracy"] + 0.05:
        raise RuntimeError("Model accuracy gain is too small")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
