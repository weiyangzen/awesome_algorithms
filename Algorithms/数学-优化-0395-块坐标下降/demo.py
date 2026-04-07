"""块坐标下降（BCD）求解 Ridge 回归的最小可运行示例。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


Array = np.ndarray


def safe_matmul(a: Array, b: Array) -> Array:
    """在当前运行环境下抑制 matmul 的伪告警。"""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return a @ b


def validate_inputs(
    X: Array,
    y: Array,
    lambda_: float,
    block_size: int,
    tol: float,
    max_epochs: int,
) -> None:
    """校验输入形状、数值和超参数。"""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values.")

    if lambda_ <= 0:
        raise ValueError("lambda_ must be > 0.")
    if block_size <= 0:
        raise ValueError("block_size must be > 0.")
    if tol <= 0:
        raise ValueError("tol must be > 0.")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be > 0.")


def build_blocks(n_features: int, block_size: int) -> List[np.ndarray]:
    """按 block_size 进行顺序分块。"""
    return [
        np.arange(start, min(start + block_size, n_features), dtype=int)
        for start in range(0, n_features, block_size)
    ]


def ridge_objective(X: Array, y: Array, w: Array, lambda_: float) -> float:
    """Ridge 目标函数值。"""
    n_samples = X.shape[0]
    residual = y - safe_matmul(X, w)
    loss = 0.5 / n_samples * float(residual @ residual)
    reg = 0.5 * lambda_ * float(w @ w)
    return loss + reg


def closed_form_ridge(X: Array, y: Array, lambda_: float) -> Array:
    """闭式 Ridge 解，用于数值对照。"""
    n_samples, n_features = X.shape
    lhs = safe_matmul(X.T, X) / n_samples + lambda_ * np.eye(n_features)
    rhs = safe_matmul(X.T, y) / n_samples
    return np.linalg.solve(lhs, rhs)


def block_coordinate_descent_ridge(
    X: Array,
    y: Array,
    lambda_: float,
    block_size: int,
    tol: float = 1e-8,
    max_epochs: int = 200,
) -> Dict[str, object]:
    """使用循环块坐标下降求解 Ridge 回归。"""
    validate_inputs(X, y, lambda_, block_size, tol, max_epochs)

    n_samples, n_features = X.shape
    blocks = build_blocks(n_features, block_size)

    w = np.zeros(n_features, dtype=float)
    residual = y.copy()  # 因为 w=0，所以 residual = y - X@w = y

    block_gram: List[Array] = []
    for idx in blocks:
        x_block = X[:, idx]
        gram = safe_matmul(x_block.T, x_block) / n_samples + lambda_ * np.eye(len(idx))
        block_gram.append(gram)

    history: List[Dict[str, float]] = []
    converged = False

    for epoch in range(1, max_epochs + 1):
        max_block_delta = 0.0

        for block_id, idx in enumerate(blocks):
            x_block = X[:, idx]
            w_old = w[idx].copy()

            rhs = safe_matmul(x_block.T, residual + safe_matmul(x_block, w_old)) / n_samples
            try:
                w_new = np.linalg.solve(block_gram[block_id], rhs)
            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    f"Failed to solve block system at block_id={block_id}."
                ) from exc

            delta = w_new - w_old
            delta_norm = float(np.linalg.norm(delta, ord=2))
            if delta_norm > 0.0:
                w[idx] = w_new
                residual -= safe_matmul(x_block, delta)

            if delta_norm > max_block_delta:
                max_block_delta = delta_norm

        obj = ridge_objective(X, y, w, lambda_)
        history.append(
            {
                "epoch": float(epoch),
                "objective": obj,
                "max_block_delta": max_block_delta,
            }
        )

        if max_block_delta < tol:
            converged = True
            break

    return {
        "w": w,
        "history": history,
        "converged": converged,
        "epochs": len(history),
        "blocks": blocks,
    }


def make_synthetic_regression(
    n_samples: int = 240,
    n_features: int = 36,
    noise_std: float = 0.15,
    seed: int = 7,
) -> Tuple[Array, Array, Array]:
    """生成可复现实验数据。"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))

    true_w = rng.normal(size=n_features)
    true_w[np.abs(true_w) < 0.55] = 0.0  # 稀疏一点，便于可视化理解

    y = safe_matmul(X, true_w) + noise_std * rng.normal(size=n_samples)

    # 仅做特征标准化，不中心化 y，保持目标形式简单直接
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    return X, y, true_w


def is_almost_monotone(values: Sequence[float], atol: float = 1e-10) -> bool:
    """检查序列是否近似单调不增。"""
    if len(values) < 2:
        return True
    diffs = np.diff(np.asarray(values, dtype=float))
    return bool(np.all(diffs <= atol))


def run_case(
    X: Array,
    y: Array,
    lambda_: float,
    block_size: int,
    tol: float,
    max_epochs: int,
) -> None:
    """运行单个实验配置并打印结果。"""
    result = block_coordinate_descent_ridge(
        X=X,
        y=y,
        lambda_=lambda_,
        block_size=block_size,
        tol=tol,
        max_epochs=max_epochs,
    )

    w_hat = result["w"]
    history = result["history"]
    objectives = [item["objective"] for item in history]

    w_star = closed_form_ridge(X, y, lambda_)
    l2_to_closed = float(np.linalg.norm(w_hat - w_star, ord=2))

    print(f"\n=== Case: block_size={block_size}, lambda={lambda_:.4f} ===")
    print(f"Converged: {result['converged']} in {result['epochs']} epochs")
    print(f"Final objective: {objectives[-1]:.10f}")
    print(f"L2 distance to closed-form: {l2_to_closed:.6e}")
    print(f"Objective monotone: {is_almost_monotone(objectives)}")

    preview = history[:3] + history[-2:] if len(history) > 5 else history
    print("History preview (epoch, objective, max_block_delta):")
    for row in preview:
        epoch = int(row["epoch"])
        print(
            f"  epoch={epoch:3d}, objective={row['objective']:.10f}, "
            f"max_block_delta={row['max_block_delta']:.6e}"
        )


def main() -> None:
    """执行两组块大小实验。"""
    X, y, true_w = make_synthetic_regression()
    print("Block Coordinate Descent MVP for Ridge Regression")
    print(f"Dataset shape: X={X.shape}, y={y.shape}, true non-zeros={(np.abs(true_w) > 0).sum()}")

    lambda_ = 0.2
    tol = 1e-9
    max_epochs = 300

    run_case(X, y, lambda_=lambda_, block_size=1, tol=tol, max_epochs=max_epochs)
    run_case(X, y, lambda_=lambda_, block_size=6, tol=tol, max_epochs=max_epochs)


if __name__ == "__main__":
    main()
