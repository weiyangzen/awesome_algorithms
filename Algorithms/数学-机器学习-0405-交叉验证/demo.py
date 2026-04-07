"""Cross-Validation MVP: manual stratified K-fold with a simple classifier.

This script is fully self-contained (NumPy + Pandas) and demonstrates
cross-validation algorithm flow without black-box CV helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class FoldResult:
    fold: int
    train_size: int
    valid_size: int
    accuracy: float


class NearestCentroidClassifier:
    """A tiny classifier used to demonstrate cross-validation.

    For each class c, compute centroid mu_c in feature space.
    Predict a sample x by argmin_c distance(x, mu_c) under Minkowski p.
    """

    def __init__(self, metric_p: int = 2) -> None:
        self.metric_p = int(metric_p)
        self._classes: np.ndarray | None = None
        self._centroids: np.ndarray | None = None

    def get_params(self) -> Dict[str, Any]:
        return {"metric_p": self.metric_p}

    def set_params(self, **params: Any) -> "NearestCentroidClassifier":
        if "metric_p" in params:
            self.metric_p = int(params["metric_p"])
        if self.metric_p <= 0:
            raise ValueError("metric_p must be positive.")
        return self

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestCentroidClassifier":
        validate_xy(x, y)
        self.set_params(metric_p=self.metric_p)

        classes = np.unique(y)
        centroids: List[np.ndarray] = []
        for c in classes:
            mask = y == c
            centroids.append(np.mean(x[mask], axis=0))

        self._classes = classes.astype(int)
        self._centroids = np.vstack(centroids).astype(float)
        return self

    def _distance_matrix(self, x: np.ndarray) -> np.ndarray:
        if self._centroids is None:
            raise RuntimeError("Model is not fitted yet.")

        # shape: [n_samples, n_classes, n_features]
        diff = x[:, None, :] - self._centroids[None, :, :]

        if self.metric_p == 1:
            dist = np.sum(np.abs(diff), axis=2)
        elif self.metric_p == 2:
            dist = np.sqrt(np.sum(diff * diff, axis=2))
        else:
            dist = np.sum(np.abs(diff) ** self.metric_p, axis=2) ** (1.0 / self.metric_p)

        return dist

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={x.shape}.")
        if self._centroids is None or self._classes is None:
            raise RuntimeError("Model is not fitted yet.")
        if x.shape[1] != self._centroids.shape[1]:
            raise ValueError(
                f"feature dimension mismatch: x={x.shape[1]}, train={self._centroids.shape[1]}."
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains non-finite values.")

        dist = self._distance_matrix(x)
        nearest = np.argmin(dist, axis=1)
        return self._classes[nearest]


def clone_estimator(estimator: NearestCentroidClassifier) -> NearestCentroidClassifier:
    return estimator.__class__(**estimator.get_params())


def validate_xy(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape={x.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"sample count mismatch: x={x.shape[0]}, y={y.shape[0]}.")
    if x.shape[0] < 2:
        raise ValueError("Need at least 2 samples.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values.")


def stratified_kfold_indices(
    y: np.ndarray,
    n_splits: int,
    random_state: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generate stratified train/validation indices manually."""
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}.")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")

    rng = np.random.default_rng(random_state)
    n_samples = y.shape[0]
    all_indices = np.arange(n_samples)
    fold_buckets: List[List[np.ndarray]] = [[] for _ in range(n_splits)]

    labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(labels, counts):
        if int(count) < n_splits:
            raise ValueError(
                f"class {int(label)} has {int(count)} samples, "
                f"smaller than n_splits={n_splits}."
            )

        class_idx = np.where(y == label)[0]
        shuffled = class_idx.copy()
        rng.shuffle(shuffled)

        chunks = np.array_split(shuffled, n_splits)
        for fold_id, chunk in enumerate(chunks):
            fold_buckets[fold_id].append(chunk)

    for fold_id in range(n_splits):
        valid_idx = np.sort(np.concatenate(fold_buckets[fold_id]))
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[valid_idx] = False
        train_idx = all_indices[train_mask]
        yield train_idx, valid_idx


def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}."
        )
    return float(np.mean(y_true == y_pred))


def cross_validate_manual(
    estimator: NearestCentroidClassifier,
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    random_state: int,
) -> Tuple[List[FoldResult], float, float, float]:
    """Run manual stratified K-fold cross-validation.

    Returns:
    - per-fold records
    - mean fold accuracy
    - std fold accuracy
    - out-of-fold (OOF) global accuracy
    """
    validate_xy(x, y)

    fold_results: List[FoldResult] = []
    oof_pred = np.empty_like(y)

    for fold_id, (train_idx, valid_idx) in enumerate(
        stratified_kfold_indices(y=y, n_splits=n_splits, random_state=random_state),
        start=1,
    ):
        x_train, y_train = x[train_idx], y[train_idx]
        x_valid, y_valid = x[valid_idx], y[valid_idx]

        model = clone_estimator(estimator)
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)

        acc = accuracy_score_np(y_valid, pred)
        oof_pred[valid_idx] = pred

        fold_results.append(
            FoldResult(
                fold=fold_id,
                train_size=int(x_train.shape[0]),
                valid_size=int(x_valid.shape[0]),
                accuracy=acc,
            )
        )

    fold_scores = np.asarray([fr.accuracy for fr in fold_results], dtype=float)
    mean_acc = float(np.mean(fold_scores))
    std_acc = float(np.std(fold_scores, ddof=0))
    oof_acc = accuracy_score_np(y, oof_pred)
    return fold_results, mean_acc, std_acc, oof_acc


def generate_synthetic_multiclass_data(
    n_classes: int = 3,
    n_per_class: int = 60,
    n_features: int = 4,
    random_state: int = 2026,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a deterministic, moderately separable multiclass dataset."""
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2.")
    if n_per_class < 2:
        raise ValueError("n_per_class must be >= 2.")
    if n_features < 1:
        raise ValueError("n_features must be >= 1.")

    rng = np.random.default_rng(random_state)
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for c in range(n_classes):
        center = rng.normal(loc=0.0, scale=1.2, size=n_features) + c * 1.2
        noise = rng.normal(loc=0.0, scale=1.6, size=(n_per_class, n_features))
        x_c = center + noise
        y_c = np.full(shape=(n_per_class,), fill_value=c, dtype=int)
        x_parts.append(x_c)
        y_parts.append(y_c)

    x = np.vstack(x_parts).astype(float)
    y = np.concatenate(y_parts).astype(int)

    shuffle_idx = np.arange(x.shape[0])
    rng.shuffle(shuffle_idx)
    return x[shuffle_idx], y[shuffle_idx]


def evaluate_candidates(
    x: np.ndarray,
    y: np.ndarray,
    candidate_params: Sequence[Dict[str, Any]],
    n_splits: int,
    random_state: int,
) -> Tuple[Dict[str, Any], List[FoldResult], pd.DataFrame]:
    """Evaluate multiple estimator configs by CV and return best candidate."""
    if len(candidate_params) == 0:
        raise ValueError("candidate_params cannot be empty.")

    records: List[Dict[str, Any]] = []
    best_params: Dict[str, Any] | None = None
    best_mean = -np.inf
    best_folds: List[FoldResult] = []

    for candidate_id, params in enumerate(candidate_params, start=1):
        estimator = NearestCentroidClassifier(**params)
        fold_results, mean_acc, std_acc, oof_acc = cross_validate_manual(
            estimator=estimator,
            x=x,
            y=y,
            n_splits=n_splits,
            random_state=random_state,
        )

        records.append(
            {
                "candidate_id": candidate_id,
                "params": str(params),
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "oof_accuracy": oof_acc,
            }
        )

        if mean_acc > best_mean:
            best_mean = mean_acc
            best_params = dict(params)
            best_folds = fold_results

    assert best_params is not None

    summary_df = pd.DataFrame(records)
    summary_df = summary_df.sort_values(
        by=["mean_accuracy", "std_accuracy"], ascending=[False, True]
    ).reset_index(drop=True)
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))
    return best_params, best_folds, summary_df


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    x, y = generate_synthetic_multiclass_data(
        n_classes=3,
        n_per_class=60,
        n_features=4,
        random_state=2026,
    )

    n_splits = 5
    random_state = 2026
    candidate_params = [{"metric_p": 1}, {"metric_p": 2}]

    best_params, best_folds, summary_df = evaluate_candidates(
        x=x,
        y=y,
        candidate_params=candidate_params,
        n_splits=n_splits,
        random_state=random_state,
    )

    fold_df = pd.DataFrame([fr.__dict__ for fr in best_folds])

    print("=== Cross-Validation MVP (MATH-0405) ===")
    print(
        f"dataset: synthetic multiclass, samples={x.shape[0]}, "
        f"features={x.shape[1]}, classes={len(np.unique(y))}"
    )
    print(f"n_splits={n_splits}, random_state={random_state}")
    print()

    print("Candidate summary:")
    print(summary_df.to_string(index=False))
    print()

    print(f"Best params by mean CV accuracy: {best_params}")
    print("Fold details (best candidate):")
    print(fold_df.to_string(index=False))


if __name__ == "__main__":
    main()
