"""Random Search MVP for hyper-parameter tuning.

This script is intentionally self-contained and reproducible.
It demonstrates random hyper-parameter sampling + stratified CV +
final hold-out evaluation.

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False


RANDOM_STATE = 20260363


@dataclass
class SearchOutcome:
    backend: str
    n_candidates: int
    best_params: Dict[str, Any]
    best_cv_f1: float
    cv_ci_low: float
    cv_ci_high: float
    tuned_test_f1: float
    tuned_test_acc: float
    baseline_test_f1: float
    top_candidates: pd.DataFrame


def generate_dataset(random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    """Generate deterministic binary classification data."""
    rng = np.random.default_rng(random_state)

    n_samples = 900
    n_features = 12
    x = rng.normal(0.0, 1.0, size=(n_samples, n_features))

    linear = 1.0 * x[:, 0] - 1.1 * x[:, 1] + 0.9 * x[:, 2] - 0.4 * x[:, 3]
    nonlinear = 0.7 * x[:, 0] * x[:, 1] - 0.6 * (x[:, 4] ** 2) + 0.5 * np.sin(x[:, 5])
    noise = rng.normal(0.0, 0.85, size=n_samples)
    logits = linear + nonlinear + noise

    threshold = float(np.median(logits))
    y = (logits > threshold).astype(int)
    return x, y


def stratified_train_test_split_np(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple stratified split without sklearn dependency."""
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("x must be 2D and y must be 1D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x/y sample count mismatch")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")

    rng = np.random.default_rng(random_state)
    train_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        shuffled = idx.copy()
        rng.shuffle(shuffled)

        n_test = int(round(shuffled.shape[0] * test_size))
        n_test = min(max(n_test, 1), shuffled.shape[0] - 1)

        test_parts.append(shuffled[:n_test])
        train_parts.append(shuffled[n_test:])

    train_idx = np.sort(np.concatenate(train_parts))
    test_idx = np.sort(np.concatenate(test_parts))
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def stratified_kfold_indices(
    y: np.ndarray,
    n_splits: int,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate stratified K-fold indices for fallback backend."""
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    rng = np.random.default_rng(random_state)
    n_samples = y.shape[0]
    all_idx = np.arange(n_samples)
    fold_buckets: List[List[np.ndarray]] = [[] for _ in range(n_splits)]

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        if cls_idx.shape[0] < n_splits:
            raise ValueError(
                f"class {int(cls)} has {cls_idx.shape[0]} samples, "
                f"smaller than n_splits={n_splits}"
            )
        shuffled = cls_idx.copy()
        rng.shuffle(shuffled)
        chunks = np.array_split(shuffled, n_splits)
        for fold_id, chunk in enumerate(chunks):
            fold_buckets[fold_id].append(chunk)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_id in range(n_splits):
        valid_idx = np.sort(np.concatenate(fold_buckets[fold_id]))
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[valid_idx] = False
        train_idx = all_idx[train_mask]
        folds.append((train_idx, valid_idx))
    return folds


def standardize_by_train(
    x_train: np.ndarray,
    x_eval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize with train statistics only."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_eval - mean) / std


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def approx_ci95(scores: np.ndarray) -> Tuple[float, float]:
    """Normal approximation CI for mean score."""
    mean = float(np.mean(scores))
    if scores.size <= 1:
        return mean, mean
    std = float(np.std(scores, ddof=1))
    sem = std / np.sqrt(scores.size)
    margin = 1.96 * sem
    return mean - margin, mean + margin


def canonical_signature(params: Dict[str, Any]) -> str:
    parts = [f"{k}={params[k]}" for k in sorted(params.keys())]
    return "|".join(parts)


def sample_svc_params(rng: np.random.Generator) -> Dict[str, Any]:
    """Randomly sample one SVC candidate from mixed search space."""
    kernel = "linear" if rng.random() < 0.35 else "rbf"
    c = float(10.0 ** rng.uniform(-2.0, 2.0))
    class_weight = None if rng.random() < 0.75 else "balanced"

    params: Dict[str, Any] = {
        "svc__kernel": kernel,
        "svc__C": c,
        "svc__class_weight": class_weight,
    }
    if kernel == "rbf":
        params["svc__gamma"] = float(10.0 ** rng.uniform(-4.0, 0.5))
    return params


def evaluate_candidate_sklearn(
    x: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: List[float] = []

    for train_idx, valid_idx in cv.split(x, y):
        x_train, x_valid = x[train_idx], x[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
        model.set_params(**params)
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        fold_scores.append(float(f1_score(y_valid, pred)))

    return np.asarray(fold_scores, dtype=float)


def run_random_search_sklearn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_iter: int,
    n_splits: int,
    random_state: int,
) -> SearchOutcome:
    rng = np.random.default_rng(random_state)

    seen: set[str] = set()
    records: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = n_iter * 30

    while len(records) < n_iter and attempts < max_attempts:
        attempts += 1
        params = sample_svc_params(rng)
        sig = canonical_signature(params)
        if sig in seen:
            continue
        seen.add(sig)

        fold_scores = evaluate_candidate_sklearn(
            x=x_train,
            y=y_train,
            params=params,
            n_splits=n_splits,
            random_state=random_state,
        )

        records.append(
            {
                "candidate_id": len(records) + 1,
                "params": params,
                "mean_f1": float(np.mean(fold_scores)),
                "std_f1": float(np.std(fold_scores, ddof=1)),
                "min_f1": float(np.min(fold_scores)),
                "max_f1": float(np.max(fold_scores)),
                "fold_scores": fold_scores.tolist(),
            }
        )

    if not records:
        raise RuntimeError("random search failed: no valid candidates evaluated")

    df = pd.DataFrame(records)
    df = df.sort_values(["mean_f1", "std_f1"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, df.shape[0] + 1))

    best_row = df.iloc[0]
    best_params = dict(best_row["params"])
    best_fold_scores = np.asarray(best_row["fold_scores"], dtype=float)
    ci_low, ci_high = approx_ci95(best_fold_scores)

    tuned = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    tuned.set_params(**best_params)
    tuned.fit(x_train, y_train)
    tuned_pred = tuned.predict(x_test)

    baseline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    baseline.fit(x_train, y_train)
    baseline_pred = baseline.predict(x_test)

    top = df[["rank", "candidate_id", "mean_f1", "std_f1", "params"]].head(5).copy()

    return SearchOutcome(
        backend="sklearn-SVC",
        n_candidates=int(df.shape[0]),
        best_params=best_params,
        best_cv_f1=float(best_row["mean_f1"]),
        cv_ci_low=ci_low,
        cv_ci_high=ci_high,
        tuned_test_f1=float(f1_score(y_test, tuned_pred)),
        tuned_test_acc=float(accuracy_score(y_test, tuned_pred)),
        baseline_test_f1=float(f1_score(y_test, baseline_pred)),
        top_candidates=top,
    )


def sample_knn_params(rng: np.random.Generator, max_k: int) -> Dict[str, Any]:
    """Sample one KNN candidate for fallback random search."""
    upper = min(max_k if max_k % 2 == 1 else max_k - 1, 31)
    upper = max(3, upper)
    odd_values = np.arange(1, upper + 1, 2)

    k = int(rng.choice(odd_values))
    p = int(rng.choice(np.array([1, 2], dtype=int)))
    weighting = "distance" if rng.random() < 0.5 else "uniform"
    return {"k": k, "p": p, "weighting": weighting}


def knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    k: int,
    p: int,
    weighting: str,
) -> np.ndarray:
    """Small KNN inference used by fallback backend."""
    preds = np.zeros(x_query.shape[0], dtype=int)

    for i, q in enumerate(x_query):
        diff = x_train - q
        if p == 1:
            dist = np.sum(np.abs(diff), axis=1)
        else:
            dist = np.sqrt(np.sum(diff * diff, axis=1))

        nn_idx = np.argpartition(dist, k - 1)[:k]
        nn_labels = y_train[nn_idx]

        if weighting == "distance":
            weights = 1.0 / (dist[nn_idx] + 1e-12)
            score_1 = float(np.sum(weights[nn_labels == 1]))
            score_0 = float(np.sum(weights[nn_labels == 0]))
            preds[i] = 1 if score_1 >= score_0 else 0
        else:
            ones = int(np.sum(nn_labels == 1))
            zeros = int(np.sum(nn_labels == 0))
            preds[i] = 1 if ones >= zeros else 0

    return preds


def evaluate_candidate_knn_cv(
    x: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    folds = stratified_kfold_indices(y=y, n_splits=n_splits, random_state=random_state)
    fold_scores: List[float] = []

    for train_idx, valid_idx in folds:
        x_train_raw, x_valid_raw = x[train_idx], x[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        x_train, x_valid = standardize_by_train(x_train_raw, x_valid_raw)
        pred = knn_predict(
            x_train=x_train,
            y_train=y_train,
            x_query=x_valid,
            k=int(params["k"]),
            p=int(params["p"]),
            weighting=str(params["weighting"]),
        )
        fold_scores.append(binary_f1(y_valid, pred))

    return np.asarray(fold_scores, dtype=float)


def run_random_search_numpy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_iter: int,
    n_splits: int,
    random_state: int,
) -> SearchOutcome:
    rng = np.random.default_rng(random_state)

    seen: set[str] = set()
    records: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = n_iter * 50
    max_k = max(3, x_train.shape[0] - x_train.shape[0] // n_splits - 1)

    while len(records) < n_iter and attempts < max_attempts:
        attempts += 1
        params = sample_knn_params(rng=rng, max_k=max_k)
        sig = canonical_signature(params)
        if sig in seen:
            continue
        seen.add(sig)

        fold_scores = evaluate_candidate_knn_cv(
            x=x_train,
            y=y_train,
            params=params,
            n_splits=n_splits,
            random_state=random_state,
        )

        records.append(
            {
                "candidate_id": len(records) + 1,
                "params": params,
                "mean_f1": float(np.mean(fold_scores)),
                "std_f1": float(np.std(fold_scores, ddof=1)),
                "min_f1": float(np.min(fold_scores)),
                "max_f1": float(np.max(fold_scores)),
                "fold_scores": fold_scores.tolist(),
            }
        )

    if not records:
        raise RuntimeError("fallback random search failed: no valid candidates evaluated")

    df = pd.DataFrame(records)
    df = df.sort_values(["mean_f1", "std_f1"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, df.shape[0] + 1))

    best_row = df.iloc[0]
    best_params = dict(best_row["params"])
    best_fold_scores = np.asarray(best_row["fold_scores"], dtype=float)
    ci_low, ci_high = approx_ci95(best_fold_scores)

    x_train_std, x_test_std = standardize_by_train(x_train, x_test)
    tuned_pred = knn_predict(
        x_train=x_train_std,
        y_train=y_train,
        x_query=x_test_std,
        k=int(best_params["k"]),
        p=int(best_params["p"]),
        weighting=str(best_params["weighting"]),
    )

    baseline_params = {"k": 5, "p": 2, "weighting": "uniform"}
    baseline_pred = knn_predict(
        x_train=x_train_std,
        y_train=y_train,
        x_query=x_test_std,
        k=int(baseline_params["k"]),
        p=int(baseline_params["p"]),
        weighting=str(baseline_params["weighting"]),
    )

    top = df[["rank", "candidate_id", "mean_f1", "std_f1", "params"]].head(5).copy()

    return SearchOutcome(
        backend="numpy-KNN-fallback",
        n_candidates=int(df.shape[0]),
        best_params=best_params,
        best_cv_f1=float(best_row["mean_f1"]),
        cv_ci_low=ci_low,
        cv_ci_high=ci_high,
        tuned_test_f1=binary_f1(y_test, tuned_pred),
        tuned_test_acc=binary_accuracy(y_test, tuned_pred),
        baseline_test_f1=binary_f1(y_test, baseline_pred),
        top_candidates=top,
    )


def main() -> None:
    x, y = generate_dataset()
    x_train, x_test, y_train, y_test = stratified_train_test_split_np(
        x=x,
        y=y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    n_iter = 18
    n_splits = 5

    if SKLEARN_AVAILABLE:
        outcome = run_random_search_sklearn(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            n_iter=n_iter,
            n_splits=n_splits,
            random_state=RANDOM_STATE,
        )
    else:
        outcome = run_random_search_numpy(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            n_iter=n_iter,
            n_splits=n_splits,
            random_state=RANDOM_STATE,
        )

    print("=== Random Search (Hyper-parameter Tuning) ===")
    print(f"backend: {outcome.backend}")
    print(f"samples: total={x.shape[0]}, train={x_train.shape[0]}, test={x_test.shape[0]}")
    print(f"feature_dim: {x.shape[1]}")
    print(f"n_candidates: {outcome.n_candidates}")
    print(f"best_params: {outcome.best_params}")
    print(f"best_cv_f1: {outcome.best_cv_f1:.6f}")
    print(f"best_cv_f1_95%_CI: [{outcome.cv_ci_low:.6f}, {outcome.cv_ci_high:.6f}]")
    print(f"tuned_test_f1: {outcome.tuned_test_f1:.6f}")
    print(f"tuned_test_accuracy: {outcome.tuned_test_acc:.6f}")
    print(f"baseline_test_f1: {outcome.baseline_test_f1:.6f}")
    print(f"f1_gain_vs_baseline: {outcome.tuned_test_f1 - outcome.baseline_test_f1:+.6f}")

    print("\nTop-5 candidates:")
    print(outcome.top_candidates.to_string(index=False))


if __name__ == "__main__":
    main()
