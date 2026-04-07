"""Elastic Net MVP: cyclic coordinate descent from scratch + sklearn sanity check."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet as SkElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class EpochRecord:
    epoch: int
    objective: float
    max_delta: float
    nnz: int


@dataclass
class ElasticNetResult:
    coef: np.ndarray
    intercept: float
    converged: bool
    epochs_run: int
    history: list[EpochRecord]
    final_objective: float


@dataclass
class SyntheticData:
    X: np.ndarray
    y: np.ndarray
    true_coef: np.ndarray
    feature_names: list[str]


def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    tol: float,
    max_epochs: int,
) -> None:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be non-empty")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if not (0.0 <= l1_ratio <= 1.0):
        raise ValueError("l1_ratio must be in [0, 1]")
    if tol <= 0:
        raise ValueError("tol must be > 0")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be > 0")


def soft_threshold(value: float, lam: float) -> float:
    if value > lam:
        return value - lam
    if value < -lam:
        return value + lam
    return 0.0


def elastic_net_objective(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    alpha: float,
    l1_ratio: float,
) -> float:
    residual = y - (X @ coef + intercept)
    data_term = 0.5 * np.mean(residual * residual)
    l1_term = alpha * l1_ratio * np.sum(np.abs(coef))
    l2_term = 0.5 * alpha * (1.0 - l1_ratio) * np.sum(coef * coef)
    return float(data_term + l1_term + l2_term)


def elastic_net_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
    tol: float = 1e-8,
    max_epochs: int = 4000,
    fit_intercept: bool = True,
) -> ElasticNetResult:
    validate_inputs(X, y, alpha, l1_ratio, tol, max_epochs)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_samples, n_features = X.shape

    if fit_intercept:
        x_mean = X.mean(axis=0)
        y_mean = float(y.mean())
    else:
        x_mean = np.zeros(n_features, dtype=np.float64)
        y_mean = 0.0

    X_centered = X - x_mean
    y_centered = y - y_mean

    coef = np.zeros(n_features, dtype=np.float64)
    residual = y_centered.copy()  # residual = y_centered - X_centered @ coef

    z = np.mean(X_centered * X_centered, axis=0)
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)

    if np.any(z + l2 <= 1e-14):
        raise ValueError(
            "Degenerate feature detected: denominator z_j + alpha*(1-l1_ratio) is too small"
        )

    history: list[EpochRecord] = []
    converged = False

    for epoch in range(1, max_epochs + 1):
        max_delta = 0.0

        for j in range(n_features):
            old_coef = coef[j]

            # Remove old contribution of feature j from residual.
            residual += X_centered[:, j] * old_coef

            rho = float(np.dot(X_centered[:, j], residual) / n_samples)
            new_coef = soft_threshold(rho, l1) / (z[j] + l2)

            # Add new contribution back so residual remains y - Xw.
            residual -= X_centered[:, j] * new_coef
            coef[j] = new_coef

            delta = abs(new_coef - old_coef)
            if delta > max_delta:
                max_delta = delta

        current_intercept = y_mean - float(np.dot(x_mean, coef)) if fit_intercept else 0.0
        objective = elastic_net_objective(
            X=X,
            y=y,
            coef=coef,
            intercept=current_intercept,
            alpha=alpha,
            l1_ratio=l1_ratio,
        )
        nnz = int(np.sum(np.abs(coef) > 1e-8))
        history.append(
            EpochRecord(
                epoch=epoch,
                objective=objective,
                max_delta=max_delta,
                nnz=nnz,
            )
        )

        if max_delta < tol:
            converged = True
            break

    intercept = y_mean - float(np.dot(x_mean, coef)) if fit_intercept else 0.0
    final_objective = elastic_net_objective(
        X=X,
        y=y,
        coef=coef,
        intercept=intercept,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )

    return ElasticNetResult(
        coef=coef,
        intercept=intercept,
        converged=converged,
        epochs_run=len(history),
        history=history,
        final_objective=final_objective,
    )


def make_synthetic_regression(
    random_state: int = 2026,
    n_samples: int = 260,
    n_groups: int = 5,
    group_size: int = 8,
    noise_std: float = 0.6,
) -> SyntheticData:
    rng = np.random.default_rng(random_state)
    n_features = n_groups * group_size

    X = np.zeros((n_samples, n_features), dtype=np.float64)
    for g in range(n_groups):
        latent = rng.normal(size=n_samples)
        for k in range(group_size):
            idx = g * group_size + k
            X[:, idx] = 0.85 * latent + 0.15 * rng.normal(size=n_samples)

    # Normalize columns to comparable scale for coordinate descent behavior.
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    true_coef = np.zeros(n_features, dtype=np.float64)
    active_indices = [0, 1, 8, 16, 24, 32]
    active_values = [3.0, 1.4, -2.6, 2.1, -1.7, 1.3]
    for idx, val in zip(active_indices, active_values):
        true_coef[idx] = val

    y = X @ true_coef + rng.normal(scale=noise_std, size=n_samples)
    feature_names = [f"x{j:02d}" for j in range(n_features)]

    return SyntheticData(X=X, y=y, true_coef=true_coef, feature_names=feature_names)


def support_metrics(
    estimated_coef: np.ndarray,
    true_coef: np.ndarray,
    threshold: float = 0.05,
) -> dict[str, float]:
    estimated_support = set(np.flatnonzero(np.abs(estimated_coef) > threshold))
    true_support = set(np.flatnonzero(np.abs(true_coef) > 1e-12))

    intersection = len(estimated_support & true_support)
    precision = intersection / max(len(estimated_support), 1)
    recall = intersection / max(len(true_support), 1)

    return {
        "estimated_nnz": float(len(estimated_support)),
        "true_nnz": float(len(true_support)),
        "support_precision": precision,
        "support_recall": recall,
    }


def history_snapshot(history: list[EpochRecord]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "epoch": [h.epoch for h in history],
            "objective": [h.objective for h in history],
            "max_delta": [h.max_delta for h in history],
            "nnz": [h.nnz for h in history],
        }
    )

    if len(frame) <= 8:
        return frame

    selected = sorted(
        {
            0,
            1,
            2,
            len(frame) // 2,
            len(frame) // 2 + 1,
            len(frame) - 3,
            len(frame) - 2,
            len(frame) - 1,
        }
    )
    return frame.iloc[selected].reset_index(drop=True)


def run_quality_checks(
    custom_result: ElasticNetResult,
    sklearn_model: SkElasticNet,
    custom_test_mse: float,
    sklearn_test_mse: float,
    true_coef: np.ndarray,
) -> None:
    objectives = np.array([h.objective for h in custom_result.history], dtype=np.float64)
    if not np.isfinite(custom_result.final_objective):
        raise AssertionError("Final objective is not finite")
    if len(objectives) < 2:
        raise AssertionError("History is unexpectedly short")
    if not np.all(np.diff(objectives) <= 1e-8):
        raise AssertionError("Objective is not monotonically non-increasing")

    mse_gap = abs(custom_test_mse - sklearn_test_mse)
    if mse_gap > 1e-4:
        raise AssertionError(f"Custom and sklearn test MSE differ too much: {mse_gap:.6f}")

    coef_gap = np.linalg.norm(custom_result.coef - sklearn_model.coef_)
    if coef_gap > 1e-3:
        raise AssertionError(f"Coefficient gap to sklearn is too large: {coef_gap:.6f}")

    top8 = set(np.argsort(np.abs(custom_result.coef))[::-1][:8])
    true_support = set(np.flatnonzero(np.abs(true_coef) > 1e-12))
    if len(top8 & true_support) < 5:
        raise AssertionError("Top-weight features do not recover enough true signal")


def main() -> None:
    data = make_synthetic_regression()

    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.30, random_state=42
    )

    alpha = 0.08
    l1_ratio = 0.60

    custom = elastic_net_coordinate_descent(
        X_train,
        y_train,
        alpha=alpha,
        l1_ratio=l1_ratio,
        tol=1e-8,
        max_epochs=4000,
        fit_intercept=True,
    )

    sklearn_model = SkElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=50000,
        tol=1e-8,
        selection="cyclic",
        random_state=0,
    )
    sklearn_model.fit(X_train, y_train)

    y_pred_train_custom = X_train @ custom.coef + custom.intercept
    y_pred_test_custom = X_test @ custom.coef + custom.intercept

    y_pred_train_sklearn = sklearn_model.predict(X_train)
    y_pred_test_sklearn = sklearn_model.predict(X_test)

    custom_metrics = {
        "train_mse": mean_squared_error(y_train, y_pred_train_custom),
        "test_mse": mean_squared_error(y_test, y_pred_test_custom),
        "test_r2": r2_score(y_test, y_pred_test_custom),
    }
    sklearn_metrics = {
        "train_mse": mean_squared_error(y_train, y_pred_train_sklearn),
        "test_mse": mean_squared_error(y_test, y_pred_test_sklearn),
        "test_r2": r2_score(y_test, y_pred_test_sklearn),
    }

    support = support_metrics(custom.coef, data.true_coef)

    print("=== Elastic Net via Cyclic Coordinate Descent (MVP) ===")
    print(
        f"train_samples={X_train.shape[0]}, test_samples={X_test.shape[0]}, features={X_train.shape[1]}"
    )
    print(f"alpha={alpha:.4f}, l1_ratio={l1_ratio:.2f}")
    print(
        f"custom_converged={custom.converged}, epochs={custom.epochs_run}, "
        f"final_objective={custom.final_objective:.6f}"
    )

    metrics_table = pd.DataFrame(
        [
            {
                "model": "custom_cd",
                "train_mse": custom_metrics["train_mse"],
                "test_mse": custom_metrics["test_mse"],
                "test_r2": custom_metrics["test_r2"],
            },
            {
                "model": "sklearn_elasticnet",
                "train_mse": sklearn_metrics["train_mse"],
                "test_mse": sklearn_metrics["test_mse"],
                "test_r2": sklearn_metrics["test_r2"],
            },
        ]
    )

    print("\n[Metrics Comparison]")
    print(metrics_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n[Support Recovery @ |coef|>0.05]")
    print(
        f"estimated_nnz={int(support['estimated_nnz'])}, true_nnz={int(support['true_nnz'])}, "
        f"precision={support['support_precision']:.3f}, recall={support['support_recall']:.3f}"
    )

    print("\n[Training History Snapshot]")
    print(
        history_snapshot(custom.history).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    coef_table = pd.DataFrame(
        {
            "feature": data.feature_names,
            "true_coef": data.true_coef,
            "custom_coef": custom.coef,
            "abs_custom": np.abs(custom.coef),
        }
    ).sort_values("abs_custom", ascending=False)

    print("\n[Top 10 |custom_coef|]")
    print(
        coef_table.head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    run_quality_checks(
        custom_result=custom,
        sklearn_model=sklearn_model,
        custom_test_mse=custom_metrics["test_mse"],
        sklearn_test_mse=sklearn_metrics["test_mse"],
        true_coef=data.true_coef,
    )
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
