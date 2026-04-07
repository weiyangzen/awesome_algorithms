"""Coordinate descent MVP for Lasso regression.

Objective:
    min_w (1/(2n)) * ||y - Xw||_2^2 + alpha * ||w||_1

This script is fully deterministic and requires no interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


HistoryItem = Tuple[int, float, float, int]


@dataclass
class CDResult:
    coef: np.ndarray
    history: List[HistoryItem]
    converged: bool
    epochs_used: int


def validate_inputs(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    tol: float,
    max_epochs: int,
) -> None:
    if x.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape={x.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array, got shape={y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample mismatch: X has {x.shape[0]} rows, y has {y.shape[0]}.")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("X must have non-zero rows and columns.")
    if not np.all(np.isfinite(x)):
        raise ValueError("X contains non-finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values.")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be > 0.")


def soft_threshold(value: float, lam: float) -> float:
    if value > lam:
        return value - lam
    if value < -lam:
        return value + lam
    return 0.0


def matvec(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Stable matrix-vector product helper."""
    return np.einsum("ij,j->i", x, w, optimize=True)


def lasso_objective(x: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> float:
    n_samples = x.shape[0]
    residual = y - matvec(x, w)
    data_term = 0.5 * float(np.dot(residual, residual)) / n_samples
    reg_term = alpha * float(np.sum(np.abs(w)))
    return data_term + reg_term


def coordinate_descent_lasso(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    tol: float = 1e-8,
    max_epochs: int = 2000,
) -> CDResult:
    validate_inputs(x=x, y=y, alpha=alpha, tol=tol, max_epochs=max_epochs)

    n_samples, n_features = x.shape
    z = np.sum(x * x, axis=0) / n_samples
    if np.any(z <= 1e-15):
        min_col = int(np.argmin(z))
        raise ValueError(
            f"Feature column {min_col} is near-constant (z={z[min_col]:.3e}); "
            "remove constant columns or standardize features."
        )

    w = np.zeros(n_features, dtype=float)
    residual = y.copy()

    history: List[HistoryItem] = []
    converged = False

    for epoch in range(1, max_epochs + 1):
        max_delta = 0.0

        for j in range(n_features):
            xj = x[:, j]
            w_old = w[j]

            rho_j = float(np.dot(xj, residual + xj * w_old)) / n_samples
            w_new = soft_threshold(rho_j, alpha) / z[j]
            delta = w_new - w_old

            if delta != 0.0:
                w[j] = w_new
                residual -= xj * delta
                abs_delta = abs(delta)
                if abs_delta > max_delta:
                    max_delta = abs_delta

        obj = 0.5 * float(np.dot(residual, residual)) / n_samples + alpha * float(np.sum(np.abs(w)))
        if not np.isfinite(obj):
            raise RuntimeError("Non-finite objective encountered during optimization.")

        nnz = int(np.count_nonzero(np.abs(w) > 1e-12))
        history.append((epoch, obj, max_delta, nnz))

        if max_delta < tol:
            converged = True
            break

    return CDResult(
        coef=w,
        history=history,
        converged=converged,
        epochs_used=len(history),
    )


def make_synthetic_regression(
    seed: int = 2026,
    n_samples: int = 240,
    n_features: int = 12,
    noise_std: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    x = rng.normal(size=(n_samples, n_features))
    x -= np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    if np.any(x_std < 1e-12):
        raise RuntimeError("Unexpected near-constant column in synthetic data generation.")
    x /= x_std

    true_w = np.array(
        [2.4, -1.9, 0.0, 0.0, 1.4, 0.0, 0.0, -2.2, 0.0, 0.9, 0.0, 0.0],
        dtype=float,
    )
    y = matvec(x, true_w) + noise_std * rng.normal(size=n_samples)
    y -= np.mean(y)

    return x, y, true_w


def support_metrics(
    true_w: np.ndarray,
    est_w: np.ndarray,
    coef_tol: float = 1e-3,
) -> Dict[str, float]:
    true_mask = np.abs(true_w) > coef_tol
    est_mask = np.abs(est_w) > coef_tol

    tp = int(np.sum(true_mask & est_mask))
    fp = int(np.sum(~true_mask & est_mask))
    fn = int(np.sum(true_mask & ~est_mask))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "true_nnz": float(np.sum(true_mask)),
        "est_nnz": float(np.sum(est_mask)),
        "precision": float(precision),
        "recall": float(recall),
    }


def objective_monotone_check(history: Sequence[HistoryItem], tol: float = 1e-10) -> Tuple[bool, int]:
    violations = 0
    for i in range(1, len(history)):
        if history[i][1] > history[i - 1][1] + tol:
            violations += 1
    return violations == 0, violations


def print_history(history: Sequence[HistoryItem], max_lines: int = 8) -> None:
    print("epoch | objective          | max|delta_w|      | nnz")
    print("---------------------------------------------------------")

    show = min(len(history), max_lines)
    for i in range(show):
        epoch, obj, max_delta, nnz = history[i]
        print(f"{epoch:5d} | {obj:18.10e} | {max_delta:16.8e} | {nnz:3d}")

    if len(history) > max_lines:
        epoch, obj, max_delta, nnz = history[-1]
        omitted = len(history) - max_lines
        print(f"... ({omitted} more epochs omitted)")
        print(f"{epoch:5d} | {obj:18.10e} | {max_delta:16.8e} | {nnz:3d}  (last)")


def run_case(
    x: np.ndarray,
    y: np.ndarray,
    true_w: np.ndarray,
    alpha: float,
    tol: float,
    max_epochs: int,
) -> Dict[str, float]:
    print(f"\n=== Lasso Coordinate Descent | alpha={alpha:.4f} ===")

    result = coordinate_descent_lasso(
        x=x,
        y=y,
        alpha=alpha,
        tol=tol,
        max_epochs=max_epochs,
    )

    print_history(result.history, max_lines=8)

    est_w = result.coef
    pred = matvec(x, est_w)
    mse = float(np.mean((y - pred) ** 2))
    coef_l2_error = float(np.linalg.norm(est_w - true_w))
    metrics = support_metrics(true_w=true_w, est_w=est_w, coef_tol=1e-3)
    monotone_ok, violations = objective_monotone_check(result.history)

    print(f"converged: {result.converged}")
    print(f"epochs used: {result.epochs_used}")
    print(f"train MSE: {mse:.8f}")
    print(f"coefficient L2 error: {coef_l2_error:.8f}")
    print(
        "support metrics: "
        f"true_nnz={metrics['true_nnz']:.0f}, "
        f"est_nnz={metrics['est_nnz']:.0f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}"
    )
    print(f"objective monotone check: {monotone_ok} (violations={violations})")

    print("true coefficients:", np.array2string(true_w, precision=4, suppress_small=True))
    print("estimated coefficients:", np.array2string(est_w, precision=4, suppress_small=True))

    return {
        "alpha": float(alpha),
        "mse": mse,
        "coef_l2_error": coef_l2_error,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "epochs": float(result.epochs_used),
        "converged": float(result.converged),
        "monotone_ok": float(monotone_ok),
    }


def main() -> None:
    x, y, true_w = make_synthetic_regression(seed=2026)

    tol = 1e-8
    max_epochs = 2000
    alphas = [0.05, 0.15]

    print("Coordinate Descent demo on synthetic sparse regression")
    print(f"dataset shape: X={x.shape}, y={y.shape}")
    print(f"hyper-parameters: tol={tol}, max_epochs={max_epochs}, alphas={alphas}")

    summaries: List[Dict[str, float]] = []
    for alpha in alphas:
        summary = run_case(
            x=x,
            y=y,
            true_w=true_w,
            alpha=alpha,
            tol=tol,
            max_epochs=max_epochs,
        )
        summaries.append(summary)

    max_mse = max(item["mse"] for item in summaries)
    min_precision = min(item["precision"] for item in summaries)
    min_recall = min(item["recall"] for item in summaries)
    all_converged = all(item["converged"] > 0.5 for item in summaries)
    all_monotone = all(item["monotone_ok"] > 0.5 for item in summaries)

    print("\n=== Summary ===")
    for item in summaries:
        print(
            f"alpha={item['alpha']:.4f}, "
            f"mse={item['mse']:.8f}, "
            f"coef_l2_error={item['coef_l2_error']:.8f}, "
            f"precision={item['precision']:.4f}, "
            f"recall={item['recall']:.4f}, "
            f"epochs={int(item['epochs'])}, "
            f"converged={bool(item['converged'])}"
        )

    pass_flag = all_converged and all_monotone and max_mse < 0.20 and min_recall >= 0.80
    print(f"global checks pass: {pass_flag}")
    print(
        "aggregate stats: "
        f"max_mse={max_mse:.8f}, min_precision={min_precision:.4f}, min_recall={min_recall:.4f}, "
        f"all_converged={all_converged}, all_monotone={all_monotone}"
    )


if __name__ == "__main__":
    main()
