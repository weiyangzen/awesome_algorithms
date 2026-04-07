"""Minimal runnable MVP for k-nearest neighbors (KNN) classification.

This script implements KNN inference from scratch with NumPy:
- deterministic synthetic dataset generation,
- train/test split,
- vectorized distance computation,
- majority voting with deterministic tie-break,
- basic evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class EvalResult:
    """Container for evaluation outputs."""

    accuracy: float
    confusion_matrix: np.ndarray


class KNNClassifier:
    """A small KNN classifier using brute-force nearest-neighbor search."""

    def __init__(
        self,
        k: int = 5,
        weighted: bool = False,
        eps: float = 1e-12,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = int(k)
        self.weighted = bool(weighted)
        self.eps = float(eps)

        self.x_train: np.ndarray | None = None
        self.y_encoded: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Store the training set for lazy KNN inference."""
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")

        classes, inverse = np.unique(y, return_inverse=True)
        if classes.size < 2:
            raise ValueError("need at least 2 classes for classification")
        if self.k > x.shape[0]:
            raise ValueError("k cannot exceed number of training samples")

        self.x_train = np.asarray(x, dtype=np.float64)
        self.y_encoded = np.asarray(inverse, dtype=np.int64)
        self.classes_ = np.asarray(classes)
        return self

    def _check_is_fitted(self) -> None:
        if self.x_train is None or self.y_encoded is None or self.classes_ is None:
            raise RuntimeError("model is not fitted")

    def _pairwise_sq_euclidean(self, x_query: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances with shape (n_query, n_train)."""
        self._check_is_fitted()
        if x_query.ndim != 2:
            raise ValueError("x_query must be a 2D array")

        x_train = self.x_train
        # (q,1,d) - (1,n,d) -> (q,n,d); then reduce along feature axis.
        diff = x_query[:, None, :] - x_train[None, :, :]
        return np.einsum("qnd,qnd->qn", diff, diff)

    def _predict_encoded(self, x_query: np.ndarray) -> np.ndarray:
        """Predict encoded class indices in [0, n_classes)."""
        self._check_is_fitted()
        dist_sq = self._pairwise_sq_euclidean(x_query)

        # Get indices of k nearest training points for each query sample.
        # argpartition gives O(n) selection per row, then we sort these k for stability.
        knn_idx = np.argpartition(dist_sq, kth=self.k - 1, axis=1)[:, : self.k]
        knn_dist_sq = np.take_along_axis(dist_sq, knn_idx, axis=1)
        order = np.argsort(knn_dist_sq, axis=1)
        knn_idx = np.take_along_axis(knn_idx, order, axis=1)
        knn_dist_sq = np.take_along_axis(knn_dist_sq, order, axis=1)

        y_encoded = self.y_encoded
        n_classes = int(self.classes_.size)
        pred = np.empty(x_query.shape[0], dtype=np.int64)

        for i in range(x_query.shape[0]):
            neighbor_labels = y_encoded[knn_idx[i]]
            if self.weighted:
                # Distance-weighted vote: closer neighbor contributes larger weight.
                weights = 1.0 / (np.sqrt(knn_dist_sq[i]) + self.eps)
                class_scores = np.bincount(
                    neighbor_labels, weights=weights, minlength=n_classes
                )
            else:
                class_scores = np.bincount(neighbor_labels, minlength=n_classes)

            # Deterministic tie-break: np.argmax returns first max index.
            pred[i] = int(np.argmax(class_scores))

        return pred

    def predict(self, x_query: np.ndarray) -> np.ndarray:
        """Predict original class labels."""
        encoded = self._predict_encoded(np.asarray(x_query, dtype=np.float64))
        return self.classes_[encoded]


def make_synthetic_dataset(
    n_per_class: int = 180,
    seed: int = 99,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a 2D, 3-class dataset with moderate overlap."""
    rng = np.random.default_rng(seed)

    centers = np.array(
        [
            [-2.8, -1.5],
            [0.2, 2.8],
            [3.0, -0.5],
        ],
        dtype=np.float64,
    )
    stds = np.array([0.85, 0.9, 0.8], dtype=np.float64)

    xs = []
    ys = []
    for cls, (center, std) in enumerate(zip(centers, stds)):
        block = rng.normal(loc=center, scale=std, size=(n_per_class, 2))
        xs.append(block)
        ys.append(np.full(n_per_class, cls, dtype=np.int64))

    x = np.vstack(xs)
    y = np.concatenate(ys)

    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    return x[idx], y[idx]


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.30,
    seed: int = 99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic random split."""
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0,1)")

    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(round(n * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Build confusion matrix where rows=true and cols=pred."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    return cm


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> EvalResult:
    """Compute simple evaluation metrics."""
    acc = float(np.mean(y_true == y_pred))
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    return EvalResult(accuracy=acc, confusion_matrix=cm)


def main() -> None:
    print("k-Nearest Neighbors (KNN) MVP (CS-0099)")
    print("=" * 64)

    x, y = make_synthetic_dataset(n_per_class=180, seed=99)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.30, seed=99)

    model = KNNClassifier(k=7, weighted=True)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    result = evaluate(y_test, y_pred, n_classes=3)

    print(f"dataset: total={x.shape[0]}, train={x_train.shape[0]}, test={x_test.shape[0]}")
    print(f"feature_dim: {x.shape[1]}")
    print(f"hyperparameters: k={model.k}, weighted={model.weighted}")
    print(f"accuracy: {result.accuracy:.4f}")
    print("confusion matrix (rows=true, cols=pred):")
    print(result.confusion_matrix)

    # Basic quality guards for deterministic synthetic data.
    if result.accuracy < 0.90:
        raise RuntimeError(f"accuracy too low: {result.accuracy:.4f}")

    if result.confusion_matrix.shape != (3, 3):
        raise RuntimeError("unexpected confusion matrix shape")

    print("=" * 64)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
