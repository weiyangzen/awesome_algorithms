"""Robust linear regression MVP (Huber M-estimator via IRLS).

Task UID: MATH-0281

This script compares:
1) ordinary least squares (OLS),
2) robust linear regression with Huber loss solved by IRLS,
on a dataset where the training targets contain injected outliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class LinearRegressionModel:
    """Linear model in augmented form: y_hat = [1, x] @ theta."""

    theta: Array
    method: str
    delta: Optional[float] = None


def validate_inputs(x: Array, y: Array) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape={x.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows.")
    if x.shape[0] < 5:
        raise ValueError("need at least 5 samples.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain finite numbers only.")


def add_bias_column(x: Array) -> Array:
    validate_inputs(x, np.zeros(x.shape[0], dtype=float))
    return np.column_stack([np.ones(x.shape[0], dtype=float), x])


def solve_weighted_least_squares(
    design: Array,
    y: Array,
    weights: Array,
    l2_reg: float,
) -> Array:
    if design.ndim != 2:
        raise ValueError("design must be 2D.")
    if y.ndim != 1 or weights.ndim != 1:
        raise ValueError("y and weights must be 1D.")
    if not (design.shape[0] == y.shape[0] == weights.shape[0]):
        raise ValueError("design, y, weights length mismatch.")
    if l2_reg < 0.0:
        raise ValueError("l2_reg must be >= 0.")

    safe_w = np.maximum(weights, 1e-12)
    sqrt_w = np.sqrt(safe_w)
    xw = design * sqrt_w[:, None]
    yw = y * sqrt_w

    gram = xw.T @ xw
    reg = np.eye(gram.shape[0], dtype=float) * l2_reg
    reg[0, 0] = 0.0  # Do not regularize the intercept.
    rhs = xw.T @ yw

    try:
        theta = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(gram + reg) @ rhs
    return theta


def fit_ols(x: Array, y: Array, l2_reg: float = 1e-8) -> LinearRegressionModel:
    validate_inputs(x, y)
    design = add_bias_column(x)
    theta = solve_weighted_least_squares(
        design=design,
        y=y,
        weights=np.ones_like(y, dtype=float),
        l2_reg=l2_reg,
    )
    return LinearRegressionModel(theta=theta, method="OLS", delta=None)


def mad_scale(residual: Array, eps: float = 1e-12) -> float:
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    return max(1.4826 * mad, eps)


def choose_huber_delta(
    residual: Array,
    min_delta: float = 0.5,
    max_delta: float = 25.0,
) -> float:
    sigma = mad_scale(residual)
    raw_delta = 1.345 * sigma
    return float(np.clip(raw_delta, min_delta, max_delta))


def huber_weights(residual: Array, delta: float, eps: float = 1e-12) -> Array:
    if delta <= 0.0:
        raise ValueError("delta must be > 0.")
    abs_r = np.abs(residual)
    weights = np.ones_like(abs_r, dtype=float)
    mask = abs_r > delta
    weights[mask] = delta / np.maximum(abs_r[mask], eps)
    return weights


def huber_objective(residual: Array, theta: Array, delta: float, l2_reg: float) -> float:
    abs_r = np.abs(residual)
    quad_mask = abs_r <= delta
    loss = np.empty_like(abs_r, dtype=float)
    loss[quad_mask] = 0.5 * residual[quad_mask] ** 2
    loss[~quad_mask] = delta * (abs_r[~quad_mask] - 0.5 * delta)
    reg = 0.5 * l2_reg * float(np.dot(theta[1:], theta[1:]))
    return float(np.mean(loss) + reg)


def fit_huber_irls(
    x: Array,
    y: Array,
    delta: Optional[float] = None,
    max_iter: int = 120,
    tol: float = 1e-7,
    l2_reg: float = 1e-8,
) -> Tuple[LinearRegressionModel, List[Tuple[int, float, float, float]], bool]:
    """Fit Huber robust regression with IRLS.

    Returns:
    - model
    - history rows: (iteration, objective, rel_change, min_weight)
    - converged flag
    """
    validate_inputs(x, y)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")

    design = add_bias_column(x)

    init_model = fit_ols(x, y, l2_reg=l2_reg)
    theta = init_model.theta.copy()

    initial_residual = y - design @ theta
    if delta is None:
        delta = choose_huber_delta(initial_residual)

    history: List[Tuple[int, float, float, float]] = []

    for iteration in range(1, max_iter + 1):
        residual = y - design @ theta
        obj = huber_objective(residual=residual, theta=theta, delta=delta, l2_reg=l2_reg)
        weights = huber_weights(residual=residual, delta=delta)
        theta_new = solve_weighted_least_squares(
            design=design,
            y=y,
            weights=weights,
            l2_reg=l2_reg,
        )

        step_norm = float(np.linalg.norm(theta_new - theta))
        base_norm = max(float(np.linalg.norm(theta)), 1e-12)
        rel_change = step_norm / base_norm
        min_weight = float(np.min(weights))
        history.append((iteration, obj, rel_change, min_weight))

        theta = theta_new
        if rel_change < tol:
            model = LinearRegressionModel(theta=theta, method="Huber-IRLS", delta=delta)
            return model, history, True

    model = LinearRegressionModel(theta=theta, method="Huber-IRLS", delta=delta)
    return model, history, False


def predict(model: LinearRegressionModel, x: Array) -> Array:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape={x.shape}.")
    design = add_bias_column(x)
    return design @ model.theta


def rmse(y_true: Array, y_pred: Array) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: Array, y_pred: Array) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.mean(np.abs(y_true - y_pred)))


def split_train_test(
    x: Array,
    y: Array,
    train_ratio: float = 0.72,
    seed: int = 281,
) -> Tuple[Array, Array, Array, Array]:
    validate_inputs(x, y)
    if not (0.1 < train_ratio < 0.95):
        raise ValueError("train_ratio must be in (0.1, 0.95).")

    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(np.floor(train_ratio * n))
    if split < 5 or n - split < 5:
        raise ValueError("split too extreme for this sample size.")

    train_idx = idx[:split]
    test_idx = idx[split:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def build_synthetic_dataset(
    n_samples: int = 320,
    n_features: int = 2,
    noise_std: float = 0.45,
    outlier_fraction: float = 0.18,
    seed: int = 281,
) -> Dict[str, Array]:
    if n_samples < 50:
        raise ValueError("n_samples must be >= 50.")
    if n_features < 1:
        raise ValueError("n_features must be >= 1.")
    if noise_std < 0.0:
        raise ValueError("noise_std must be >= 0.")
    if not (0.0 <= outlier_fraction < 0.5):
        raise ValueError("outlier_fraction must be in [0, 0.5).")

    rng = np.random.default_rng(seed)

    x = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    if n_features >= 2:
        x[:, 1] = 0.65 * x[:, 0] + 0.35 * x[:, 1]

    true_theta = np.array([0.9, 2.6, -1.8], dtype=float)
    if n_features != 2:
        true_theta = np.concatenate(
            [np.array([0.9], dtype=float), rng.uniform(-2.5, 2.5, size=n_features)]
        )

    y_clean = true_theta[0] + x @ true_theta[1:] + rng.normal(0.0, noise_std, size=n_samples)

    x_train, y_train_clean, x_test, y_test_clean = split_train_test(
        x,
        y_clean,
        train_ratio=0.72,
        seed=seed + 1,
    )

    y_train_contaminated = y_train_clean.copy()
    n_outliers = int(round(outlier_fraction * y_train_contaminated.size))
    outlier_idx = rng.choice(y_train_contaminated.size, size=n_outliers, replace=False)

    shock = rng.normal(0.0, 11.0, size=n_outliers)
    sign = rng.choice(np.array([-1.0, 1.0], dtype=float), size=n_outliers)
    y_train_contaminated[outlier_idx] += shock + 16.0 * sign

    outlier_mask = np.zeros_like(y_train_contaminated, dtype=bool)
    outlier_mask[outlier_idx] = True

    return {
        "x_train": x_train,
        "y_train_clean": y_train_clean,
        "y_train_contaminated": y_train_contaminated,
        "x_test": x_test,
        "y_test_clean": y_test_clean,
        "true_theta": true_theta,
        "outlier_mask": outlier_mask,
    }


def summarize_history(history: List[Tuple[int, float, float, float]], top_k: int = 4) -> None:
    print("IRLS iteration log (iteration, objective, rel_change, min_weight):")
    print("-" * 72)
    if len(history) <= 2 * top_k:
        rows = history
    else:
        rows = history[:top_k] + history[-top_k:]

    for iteration, obj, rel_change, min_weight in rows:
        print(
            f"{iteration:4d} | obj={obj:11.6f} | rel_change={rel_change:10.3e} | "
            f"min_weight={min_weight:8.5f}"
        )

    if len(history) > 2 * top_k:
        print("...")
    print("-" * 72)


def print_metric_table(
    y_train: Array,
    y_test: Array,
    pred_train_ols: Array,
    pred_test_ols: Array,
    pred_train_huber: Array,
    pred_test_huber: Array,
) -> None:
    ols_train_rmse = rmse(y_train, pred_train_ols)
    ols_test_rmse = rmse(y_test, pred_test_ols)
    huber_train_rmse = rmse(y_train, pred_train_huber)
    huber_test_rmse = rmse(y_test, pred_test_huber)

    ols_train_mae = mae(y_train, pred_train_ols)
    ols_test_mae = mae(y_test, pred_test_ols)
    huber_train_mae = mae(y_train, pred_train_huber)
    huber_test_mae = mae(y_test, pred_test_huber)

    print("Metrics:")
    print("-" * 72)
    print(f"{'model':<12} {'train_RMSE':>12} {'test_RMSE':>12} {'train_MAE':>12} {'test_MAE':>12}")
    print(
        f"{'OLS':<12} {ols_train_rmse:12.6f} {ols_test_rmse:12.6f} "
        f"{ols_train_mae:12.6f} {ols_test_mae:12.6f}"
    )
    print(
        f"{'Huber-IRLS':<12} {huber_train_rmse:12.6f} {huber_test_rmse:12.6f} "
        f"{huber_train_mae:12.6f} {huber_test_mae:12.6f}"
    )
    print("-" * 72)


def print_sample_predictions(
    x_test: Array,
    y_test: Array,
    pred_ols: Array,
    pred_huber: Array,
    top_k: int = 8,
) -> None:
    top_k = int(max(1, min(top_k, x_test.shape[0])))
    order = np.argsort(x_test[:, 0])

    print("Sample predictions on clean test set:")
    print("x0        | x1        | y_true     | y_ols      | y_huber    | abs_err_huber")
    print("-" * 86)
    for idx in order[:top_k]:
        err_h = abs(y_test[idx] - pred_huber[idx])
        print(
            f"{x_test[idx, 0]:9.4f} | {x_test[idx, 1]:9.4f} | {y_test[idx]:10.4f} | "
            f"{pred_ols[idx]:10.4f} | {pred_huber[idx]:10.4f} | {err_h:13.4f}"
        )


def main() -> None:
    print("Robust Regression MVP (MATH-0281)")
    print("=" * 72)

    data = build_synthetic_dataset(
        n_samples=320,
        n_features=2,
        noise_std=0.45,
        outlier_fraction=0.18,
        seed=281,
    )

    x_train = data["x_train"]
    y_train_clean = data["y_train_clean"]
    y_train = data["y_train_contaminated"]
    x_test = data["x_test"]
    y_test = data["y_test_clean"]
    true_theta = data["true_theta"]
    outlier_mask = data["outlier_mask"]

    outlier_count = int(np.sum(outlier_mask))
    print(
        f"dataset: train={x_train.shape[0]}, test={x_test.shape[0]}, "
        f"features={x_train.shape[1]}, outliers_in_train={outlier_count}"
    )
    print(
        f"train contamination mean abs shift: "
        f"{np.mean(np.abs(y_train - y_train_clean)):.4f}"
    )

    ols = fit_ols(x_train, y_train, l2_reg=1e-8)
    huber, history, converged = fit_huber_irls(
        x_train,
        y_train,
        delta=None,
        max_iter=120,
        tol=1e-8,
        l2_reg=1e-8,
    )

    pred_train_ols = predict(ols, x_train)
    pred_test_ols = predict(ols, x_test)
    pred_train_huber = predict(huber, x_train)
    pred_test_huber = predict(huber, x_test)

    print(f"huber delta={huber.delta:.6f}, converged={converged}, iterations={len(history)}")
    summarize_history(history)

    print("True vs estimated parameters [bias, w1, w2]:")
    print(f"true      : {np.array2string(true_theta, precision=5, floatmode='fixed')}")
    print(f"ols       : {np.array2string(ols.theta, precision=5, floatmode='fixed')}")
    print(f"huber-irls: {np.array2string(huber.theta, precision=5, floatmode='fixed')}")

    ols_param_err = float(np.linalg.norm(ols.theta - true_theta))
    huber_param_err = float(np.linalg.norm(huber.theta - true_theta))
    print(f"parameter L2 error: OLS={ols_param_err:.6f}, Huber={huber_param_err:.6f}")

    print_metric_table(
        y_train=y_train,
        y_test=y_test,
        pred_train_ols=pred_train_ols,
        pred_test_ols=pred_test_ols,
        pred_train_huber=pred_train_huber,
        pred_test_huber=pred_test_huber,
    )

    print_sample_predictions(
        x_test=x_test,
        y_test=y_test,
        pred_ols=pred_test_ols,
        pred_huber=pred_test_huber,
        top_k=8,
    )

    ols_test_rmse = rmse(y_test, pred_test_ols)
    huber_test_rmse = rmse(y_test, pred_test_huber)
    pass_flag = huber_test_rmse < ols_test_rmse and huber_param_err < ols_param_err

    print("\nSummary:")
    print(
        f"ols_test_rmse={ols_test_rmse:.6f}, huber_test_rmse={huber_test_rmse:.6f}, "
        f"pass={pass_flag}"
    )


if __name__ == "__main__":
    main()
