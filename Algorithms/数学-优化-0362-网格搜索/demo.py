"""Grid Search MVP: manual hyper-parameter enumeration + cross-validation.

This demo is fully self-contained (NumPy + Pandas only):
- synthetic multiclass dataset generation
- manual stratified train/test split
- manual stratified K-fold CV
- manual KNN classifier
- manual grid search loop
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd

Params = Dict[str, Any]
ScoreList = List[float]


class KNNClassifier:
    """Minimal KNN classifier with configurable k / Minkowski p / weighted voting."""

    def __init__(self, k: int = 3, p: int = 2, weighted: bool = False) -> None:
        self.k = int(k)
        self.p = int(p)
        self.weighted = bool(weighted)
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def get_params(self) -> Params:
        return {"k": self.k, "p": self.p, "weighted": self.weighted}

    def set_params(self, **params: Any) -> "KNNClassifier":
        valid = {"k", "p", "weighted"}
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Unknown parameter '{key}' for KNNClassifier.")
            setattr(self, key, value)

        self.k = int(self.k)
        self.p = int(self.p)
        self.weighted = bool(self.weighted)
        if self.k <= 0:
            raise ValueError("k must be > 0.")
        if self.p not in (1, 2):
            raise ValueError("p must be 1 (L1) or 2 (L2).")
        return self

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={x.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape={y.shape}.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x/y size mismatch.")
        if self.k > x.shape[0]:
            raise ValueError(
                f"k={self.k} cannot exceed training sample count={x.shape[0]}."
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains non-finite values.")

        self._x_train = x.astype(float).copy()
        self._y_train = y.astype(int).copy()
        return self

    def _predict_one(self, row: np.ndarray) -> int:
        if self._x_train is None or self._y_train is None:
            raise RuntimeError("Model is not fitted yet.")

        diff = self._x_train - row
        if self.p == 1:
            dist = np.sum(np.abs(diff), axis=1)
        else:  # self.p == 2
            dist = np.sqrt(np.sum(diff * diff, axis=1))

        nn_idx = np.argpartition(dist, self.k - 1)[: self.k]
        nn_labels = self._y_train[nn_idx]

        if self.weighted:
            nn_dist = dist[nn_idx]
            weights = 1.0 / (nn_dist + 1e-12)
            score_by_label: Dict[int, float] = {}
            for label, w in zip(nn_labels, weights):
                key = int(label)
                score_by_label[key] = score_by_label.get(key, 0.0) + float(w)

            # Tie-breaker: higher weight first, then smaller label.
            ranked = sorted(score_by_label.items(), key=lambda kv: (-kv[1], kv[0]))
            return int(ranked[0][0])

        counts = np.bincount(nn_labels)
        return int(np.argmax(counts))

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={x.shape}.")
        if self._x_train is None:
            raise RuntimeError("Model is not fitted yet.")
        if x.shape[1] != self._x_train.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: x has {x.shape[1]}, "
                f"train has {self._x_train.shape[1]}."
            )

        pred = [self._predict_one(row) for row in x]
        return np.asarray(pred, dtype=int)


def validate_param_grid(param_grid: Dict[str, Sequence[Any]]) -> None:
    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError("param_grid must be a non-empty dict[str, sequence].")

    for key, values in param_grid.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"Invalid grid key: {key!r}.")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(
                f"Grid values for key '{key}' must be a non-string sequence."
            )
        if len(values) == 0:
            raise ValueError(f"Grid values for key '{key}' cannot be empty.")


def expand_param_grid(param_grid: Dict[str, Sequence[Any]]) -> List[Params]:
    """Cartesian expansion: {'a':[1,2], 'b':[3]} -> [{'a':1,'b':3}, {'a':2,'b':3}]"""
    validate_param_grid(param_grid)
    keys = list(param_grid.keys())
    values_product = product(*(param_grid[k] for k in keys))
    return [dict(zip(keys, values)) for values in values_product]


def clone_estimator(estimator: KNNClassifier) -> KNNClassifier:
    return estimator.__class__(**estimator.get_params())


def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}.")
    return float(np.mean(y_true == y_pred))


def stratified_kfold_indices(
    y: np.ndarray, n_splits: int, random_state: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}.")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")

    rng = np.random.default_rng(random_state)
    n_samples = y.shape[0]
    all_indices = np.arange(n_samples)
    fold_parts: List[List[np.ndarray]] = [[] for _ in range(n_splits)]

    for label in np.unique(y):
        label_idx = np.where(y == label)[0]
        if label_idx.shape[0] < n_splits:
            raise ValueError(
                f"Class {label} has {label_idx.shape[0]} samples, "
                f"smaller than n_splits={n_splits}."
            )
        shuffled = label_idx.copy()
        rng.shuffle(shuffled)
        chunks = np.array_split(shuffled, n_splits)
        for fold_id, chunk in enumerate(chunks):
            fold_parts[fold_id].append(chunk)

    for fold_id in range(n_splits):
        valid_idx = np.concatenate(fold_parts[fold_id])
        valid_idx = np.sort(valid_idx)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[valid_idx] = False
        train_idx = all_indices[train_mask]
        yield train_idx, valid_idx


def stratified_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("x must be 2D and y must be 1D.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x/y sample count mismatch.")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1).")

    rng = np.random.default_rng(random_state)
    train_idx_parts: List[np.ndarray] = []
    test_idx_parts: List[np.ndarray] = []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        shuffled = idx.copy()
        rng.shuffle(shuffled)

        n_test = int(round(shuffled.shape[0] * test_size))
        n_test = min(max(n_test, 1), shuffled.shape[0] - 1)

        test_idx_parts.append(shuffled[:n_test])
        train_idx_parts.append(shuffled[n_test:])

    train_idx = np.sort(np.concatenate(train_idx_parts))
    test_idx = np.sort(np.concatenate(test_idx_parts))

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def cross_val_score_manual(
    estimator: KNNClassifier,
    x: np.ndarray,
    y: np.ndarray,
    params: Params,
    cv: int,
    random_state: int,
) -> ScoreList:
    fold_scores: ScoreList = []

    for train_idx, valid_idx in stratified_kfold_indices(
        y=y, n_splits=cv, random_state=random_state
    ):
        x_train, x_valid = x[train_idx], x[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = clone_estimator(estimator)
        model.set_params(**params)
        model.fit(x_train, y_train)

        pred = model.predict(x_valid)
        score = accuracy_score_np(y_valid, pred)
        fold_scores.append(score)

    return fold_scores


def manual_grid_search(
    estimator: KNNClassifier,
    param_grid: Dict[str, Sequence[Any]],
    x: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[Params, float, pd.DataFrame]:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D array, got shape={x.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array, got shape={y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x/y sample count mismatch: {x.shape[0]} vs {y.shape[0]}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values.")

    candidates = expand_param_grid(param_grid)
    records: List[Dict[str, Any]] = []

    best_params: Params = {}
    best_score = -np.inf

    for idx, params in enumerate(candidates, start=1):
        scores = cross_val_score_manual(
            estimator=estimator,
            x=x,
            y=y,
            params=params,
            cv=cv,
            random_state=random_state,
        )
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        records.append(
            {
                "candidate_id": idx,
                "params": params,
                "mean_score": mean_score,
                "std_score": std_score,
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
            }
        )

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    result_df = pd.DataFrame(records).sort_values(
        by=["mean_score", "std_score"], ascending=[False, True]
    )
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    return best_params, float(best_score), result_df


def print_top_results(result_df: pd.DataFrame, top_k: int = 8) -> None:
    show = result_df.head(top_k).copy()
    show["params"] = show["params"].astype(str)
    pd.set_option("display.max_colwidth", 120)
    print(show.to_string(index=False))


def generate_synthetic_multiclass_data(
    n_per_class: int = 80,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a stable 3-class dataset with moderate overlap."""
    if n_per_class <= 3:
        raise ValueError("n_per_class must be > 3.")

    rng = np.random.default_rng(random_state)
    centers = np.array(
        [
            [0.0, 0.5, -0.5, 1.0],
            [2.4, -1.1, 1.7, 0.2],
            [-1.8, 2.2, 0.9, -1.0],
        ],
        dtype=float,
    )
    scales = np.array([0.9, 1.2, 0.8, 1.0], dtype=float)

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for cls, center in enumerate(centers):
        noise = rng.normal(loc=0.0, scale=scales, size=(n_per_class, center.shape[0]))
        x_cls = center + noise
        y_cls = np.full(n_per_class, cls, dtype=int)
        x_parts.append(x_cls)
        y_parts.append(y_cls)

    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    return x, y


def main() -> None:
    seed = 42

    x, y = generate_synthetic_multiclass_data(n_per_class=80, random_state=seed)
    x_train, x_test, y_train, y_test = stratified_train_test_split(
        x=x,
        y=y,
        test_size=0.25,
        random_state=seed,
    )

    estimator = KNNClassifier(k=3, p=2, weighted=False)
    param_grid = {
        "k": [1, 3, 5, 7, 9],
        "p": [1, 2],
        "weighted": [False, True],
    }

    best_params, best_cv_score, result_df = manual_grid_search(
        estimator=estimator,
        param_grid=param_grid,
        x=x_train,
        y=y_train,
        cv=5,
        random_state=seed,
    )

    best_model = clone_estimator(estimator)
    best_model.set_params(**best_params)
    best_model.fit(x_train, y_train)
    test_acc = accuracy_score_np(y_test, best_model.predict(x_test))

    baseline_model = clone_estimator(estimator)
    baseline_model.fit(x_train, y_train)
    baseline_acc = accuracy_score_np(y_test, baseline_model.predict(x_test))

    print("=== Grid Search Demo (fully manual implementation) ===")
    print(f"total samples: {x.shape[0]}, feature dim: {x.shape[1]}")
    print(f"train size: {x_train.shape[0]}, test size: {x_test.shape[0]}")
    print(f"candidate count: {len(result_df)}")

    print("\nTop candidates by CV mean accuracy:")
    print_top_results(result_df=result_df, top_k=8)

    print("\n=== Best Result ===")
    print(f"best params: {best_params}")
    print(f"best CV mean accuracy: {best_cv_score:.6f}")
    print(f"test accuracy (best params): {test_acc:.6f}")
    print(f"test accuracy (baseline/default KNN): {baseline_acc:.6f}")


if __name__ == "__main__":
    main()
