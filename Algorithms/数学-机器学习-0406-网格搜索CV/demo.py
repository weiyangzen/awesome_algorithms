"""Minimal runnable MVP for Grid Search + Cross Validation.

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False


RANDOM_STATE = 42


@dataclass
class SearchOutcome:
    backend: str
    best_params: dict
    best_cv_score: float
    ci_low: float
    ci_high: float
    top5: pd.DataFrame
    tuned_accuracy: float
    tuned_f1: float
    baseline_f1: float
    y_test: np.ndarray
    y_pred: np.ndarray


def generate_dataset(random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic synthetic binary dataset."""
    rng = np.random.default_rng(random_state)
    n_samples, n_features = 1000, 12
    x = rng.normal(0.0, 1.0, size=(n_samples, n_features))

    linear_term = 1.2 * x[:, 0] - 1.0 * x[:, 1] + 0.8 * x[:, 2] + 0.6 * x[:, 3]
    nonlinear_term = 0.8 * x[:, 0] * x[:, 1] - 0.5 * (x[:, 2] ** 2)
    noise = rng.normal(0.0, 0.9, size=n_samples)
    logits = linear_term + nonlinear_term + noise

    threshold = float(np.median(logits))
    y = (logits > threshold).astype(int)
    return x, y


def stratified_train_test_split(
    x: np.ndarray, y: np.ndarray, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple stratified split without sklearn dependency."""
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)

    train_parts = []
    test_parts = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(len(cls_idx) * test_size)))
        test_parts.append(cls_idx[:n_test])
        train_parts.append(cls_idx[n_test:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def stratified_kfold_indices(
    y: np.ndarray, n_splits: int, random_state: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build stratified K-fold train/validation indices."""
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    class_folds: dict[int, list[np.ndarray]] = {}

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        class_folds[int(cls)] = list(np.array_split(cls_idx, n_splits))

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_id in range(n_splits):
        val_idx = np.concatenate([class_folds[int(cls)][fold_id] for cls in classes])
        train_idx = np.concatenate(
            [
                part
                for cls in classes
                for i, part in enumerate(class_folds[int(cls)])
                if i != fold_id
            ]
        )
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        folds.append((train_idx, val_idx))
    return folds


def standardize_by_train(x_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize by train statistics only to avoid leakage."""
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
    return 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)


def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    k: int,
    weighting: str,
) -> np.ndarray:
    """Small KNN predictor used by the numpy fallback path."""
    preds = np.zeros(len(x_query), dtype=int)

    for i, q in enumerate(x_query):
        d2 = np.sum((x_train - q) ** 2, axis=1)
        nn_idx = np.argpartition(d2, k - 1)[:k]
        nn_labels = y_train[nn_idx]

        if weighting == "distance":
            weights = 1.0 / (np.sqrt(d2[nn_idx]) + 1e-12)
            score1 = float(np.sum(weights[nn_labels == 1]))
            score0 = float(np.sum(weights[nn_labels == 0]))
            preds[i] = 1 if score1 >= score0 else 0
        else:
            ones = int(np.sum(nn_labels))
            zeros = int(k - ones)
            preds[i] = 1 if ones >= zeros else 0

    return preds


def approx_ci95(scores: np.ndarray) -> tuple[float, float]:
    """Normal-approximation confidence interval for mean score."""
    mean = float(np.mean(scores))
    if len(scores) <= 1:
        return mean, mean

    std = float(np.std(scores, ddof=1))
    sem = std / np.sqrt(len(scores))
    margin = 1.96 * sem
    return mean - margin, mean + margin


def run_with_sklearn(
    x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> SearchOutcome:
    """Primary path: sklearn Pipeline + GridSearchCV."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    param_grid = [
        {"svc__kernel": ["linear"], "svc__C": [0.1, 1.0, 10.0, 30.0]},
        {
            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1.0, 10.0, 30.0],
            "svc__gamma": [0.01, 0.1, 1.0],
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        n_jobs=-1,
        cv=cv,
        refit=True,
        return_train_score=True,
    )
    search.fit(x_train, y_train)

    results = pd.DataFrame(search.cv_results_)
    split_cols = [col for col in results.columns if col.startswith("split") and col.endswith("_test_score")]
    split_scores = results.loc[search.best_index_, split_cols].astype(float).to_numpy()
    ci_low, ci_high = approx_ci95(split_scores)

    y_pred = search.predict(x_test)
    tuned_f1 = float(f1_score(y_test, y_pred))
    tuned_acc = float(accuracy_score(y_test, y_pred))

    baseline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    baseline.fit(x_train, y_train)
    baseline_pred = baseline.predict(x_test)
    baseline_f1 = float(f1_score(y_test, baseline_pred))

    top5 = (
        results[["rank_test_score", "mean_test_score", "std_test_score", "params"]]
        .sort_values(["rank_test_score", "std_test_score"], ascending=[True, True])
        .head(5)
    )

    return SearchOutcome(
        backend="sklearn",
        best_params=dict(search.best_params_),
        best_cv_score=float(search.best_score_),
        ci_low=ci_low,
        ci_high=ci_high,
        top5=top5,
        tuned_accuracy=tuned_acc,
        tuned_f1=tuned_f1,
        baseline_f1=baseline_f1,
        y_test=y_test,
        y_pred=np.asarray(y_pred),
    )


def run_with_numpy_fallback(
    x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> SearchOutcome:
    """Fallback path: pure-numpy grid search + stratified K-fold CV."""
    n_splits = 5
    folds = stratified_kfold_indices(y_train, n_splits=n_splits, random_state=RANDOM_STATE)

    param_grid = [
        {"k": 1, "weighting": "uniform"},
        {"k": 3, "weighting": "uniform"},
        {"k": 5, "weighting": "uniform"},
        {"k": 7, "weighting": "uniform"},
        {"k": 11, "weighting": "uniform"},
        {"k": 3, "weighting": "distance"},
        {"k": 5, "weighting": "distance"},
        {"k": 7, "weighting": "distance"},
        {"k": 11, "weighting": "distance"},
    ]

    rows: list[dict] = []
    for params in param_grid:
        fold_scores: list[float] = []
        row: dict = {"params": params}

        for fold_id, (tr_idx, val_idx) in enumerate(folds):
            x_tr, x_val = x_train[tr_idx], x_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            x_tr_std, x_val_std = standardize_by_train(x_tr, x_val)
            pred = knn_predict(
                x_train=x_tr_std,
                y_train=y_tr,
                x_query=x_val_std,
                k=int(params["k"]),
                weighting=str(params["weighting"]),
            )
            score = binary_f1(y_val, pred)
            fold_scores.append(score)
            row[f"split{fold_id}_test_score"] = score

        row["mean_test_score"] = float(np.mean(fold_scores))
        row["std_test_score"] = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
        rows.append(row)

    results = pd.DataFrame(rows)
    results["rank_test_score"] = (
        results["mean_test_score"].rank(ascending=False, method="min").astype(int)
    )

    best_row = results.sort_values(["rank_test_score", "std_test_score"]).iloc[0]
    best_params = dict(best_row["params"])

    x_train_std, x_test_std = standardize_by_train(x_train, x_test)
    y_pred = knn_predict(
        x_train=x_train_std,
        y_train=y_train,
        x_query=x_test_std,
        k=int(best_params["k"]),
        weighting=str(best_params["weighting"]),
    )

    baseline_pred = knn_predict(
        x_train=x_train_std,
        y_train=y_train,
        x_query=x_test_std,
        k=5,
        weighting="uniform",
    )

    split_cols = [f"split{i}_test_score" for i in range(n_splits)]
    split_scores = best_row[split_cols].astype(float).to_numpy()
    ci_low, ci_high = approx_ci95(split_scores)

    top5 = (
        results[["rank_test_score", "mean_test_score", "std_test_score", "params"]]
        .sort_values(["rank_test_score", "std_test_score"], ascending=[True, True])
        .head(5)
    )

    return SearchOutcome(
        backend="numpy-fallback",
        best_params=best_params,
        best_cv_score=float(best_row["mean_test_score"]),
        ci_low=ci_low,
        ci_high=ci_high,
        top5=top5,
        tuned_accuracy=binary_accuracy(y_test, y_pred),
        tuned_f1=binary_f1(y_test, y_pred),
        baseline_f1=binary_f1(y_test, baseline_pred),
        y_test=y_test,
        y_pred=y_pred,
    )


def print_report(outcome: SearchOutcome) -> None:
    print("=== Grid Search CV MVP (MATH-0406) ===")
    print(f"Backend: {outcome.backend}")
    print(f"Best params: {outcome.best_params}")
    print(f"Best mean CV F1: {outcome.best_cv_score:.4f}")
    print(f"95% CI (best CV F1): [{outcome.ci_low:.4f}, {outcome.ci_high:.4f}]")

    print("\nTop-5 candidates:")
    print(outcome.top5.to_string(index=False))

    print("\nTest metrics:")
    print(f"Tuned  accuracy: {outcome.tuned_accuracy:.4f}")
    print(f"Tuned  F1      : {outcome.tuned_f1:.4f}")
    print(f"Default F1     : {outcome.baseline_f1:.4f}")
    print(f"F1 gain        : {outcome.tuned_f1 - outcome.baseline_f1:+.4f}")

    if SKLEARN_AVAILABLE:
        print("\nClassification report (tuned model):")
        print(classification_report(outcome.y_test, outcome.y_pred, digits=4))


def main() -> None:
    x, y = generate_dataset()

    if SKLEARN_AVAILABLE:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        outcome = run_with_sklearn(x_train, x_test, y_train, y_test)
    else:
        x_train, x_test, y_train, y_test = stratified_train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=RANDOM_STATE,
        )
        outcome = run_with_numpy_fallback(x_train, x_test, y_train, y_test)

    print_report(outcome)


if __name__ == "__main__":
    main()
