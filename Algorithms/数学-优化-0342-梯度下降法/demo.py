"""Gradient descent MVP for L2-regularized linear regression (ridge)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HistoryItem = Tuple[int, float, float, float]


def check_vector(name: str, x: np.ndarray) -> None:
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def check_matrix(name: str, x: np.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def objective_ridge(x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float) -> float:
    m = x.shape[0]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        residual = x @ w - y
    data_term = 0.5 * float(residual.T @ residual) / float(m)
    reg_term = 0.5 * l2 * float(w.T @ w)
    return data_term + reg_term


def gradient_ridge(x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float) -> np.ndarray:
    m = x.shape[0]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        residual = x @ w - y
        grad = (x.T @ residual) / float(m) + l2 * w
    return grad


def estimate_lipschitz_constant(x: np.ndarray, l2: float) -> float:
    m = x.shape[0]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        h = (x.T @ x) / float(m)
    eigvals = np.linalg.eigvalsh(h)
    l_max = float(np.max(eigvals) + l2)
    return l_max


def ridge_closed_form(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    m, n = x.shape
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        a = (x.T @ x) + float(m) * l2 * np.eye(n, dtype=float)
        b = x.T @ y
    return np.linalg.solve(a, b)


def relative_error(error_norm: float, reference_norm: float, eps: float = 1e-15) -> float:
    return abs(error_norm) / (abs(reference_norm) + eps)


def gradient_descent_ridge(
    x: np.ndarray,
    y: np.ndarray,
    w0: np.ndarray,
    l2: float = 1e-2,
    lr: Optional[float] = None,
    tol: float = 1e-10,
    max_iter: int = 20000,
) -> Tuple[np.ndarray, List[HistoryItem], float]:
    check_matrix("X", x)
    check_vector("y", y)
    check_vector("w0", w0)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch: X has {x.shape[0]} rows, y has {y.shape[0]}.")
    if x.shape[1] != w0.shape[0]:
        raise ValueError(
            f"Dimension mismatch: X has {x.shape[1]} columns, w0 has {w0.shape[0]} entries."
        )
    if l2 < 0.0:
        raise ValueError("l2 must be >= 0.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")

    l_const = estimate_lipschitz_constant(x, l2)
    if not np.isfinite(l_const) or l_const <= 0.0:
        raise RuntimeError("Estimated Lipschitz constant is invalid; cannot choose stable lr.")

    if lr is None:
        lr_use = 0.9 / l_const
    else:
        if lr <= 0.0:
            raise ValueError("lr must be > 0 when provided.")
        lr_use = float(lr)

    w = w0.astype(float).copy()
    history: List[HistoryItem] = []

    for k in range(1, max_iter + 1):
        loss = objective_ridge(x, y, w, l2)
        grad = gradient_ridge(x, y, w, l2)
        grad_norm = float(np.linalg.norm(grad))

        if not np.isfinite(loss) or not np.isfinite(grad_norm):
            raise RuntimeError("Non-finite loss or gradient encountered.")

        if grad_norm <= tol:
            history.append((k, loss, grad_norm, 0.0))
            return w, history, lr_use

        w_next = w - lr_use * grad
        if not np.all(np.isfinite(w_next)):
            raise RuntimeError("Non-finite iterate encountered.")

        step_norm = float(np.linalg.norm(w_next - w))
        history.append((k, loss, grad_norm, step_norm))

        if step_norm <= tol * (1.0 + float(np.linalg.norm(w_next))):
            return w_next, history, lr_use

        w = w_next

    return w, history, lr_use


def print_history(history: Sequence[HistoryItem], max_lines: int = 12) -> None:
    print("iter | loss             | ||grad||         | ||step||")
    print("-" * 62)
    for item in history[:max_lines]:
        k, loss, grad_norm, step_norm = item
        print(f"{k:4d} | {loss:16.9e} | {grad_norm:16.9e} | {step_norm:16.9e}")
    if len(history) > max_lines:
        print(f"... ({len(history) - max_lines} more iterations omitted)")


def make_synthetic_case(
    seed: int,
    m: int,
    n: int,
    noise_std: float,
    collinearity: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(m, n))

    if collinearity > 0.0:
        # Blend all columns with a shared latent direction to create correlation.
        latent = rng.normal(size=(m, 1))
        x = (1.0 - collinearity) * x + collinearity * latent @ np.ones((1, n))

    x = x - np.mean(x, axis=0, keepdims=True)
    x_scale = np.std(x, axis=0, keepdims=True)
    x_scale = np.where(x_scale < 1e-12, 1.0, x_scale)
    x = x / x_scale

    true_w = rng.normal(size=n)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        y = x @ true_w + noise_std * rng.normal(size=m)
    return x.astype(float), y.astype(float), true_w.astype(float)


def run_case(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    l2: float,
    tol: float,
    max_iter: int,
    lr: Optional[float],
) -> Dict[str, float]:
    print(f"\n=== Case: {name} ===")

    w0 = np.zeros(x.shape[1], dtype=float)
    w_est, history, lr_used = gradient_descent_ridge(
        x=x,
        y=y,
        w0=w0,
        l2=l2,
        lr=lr,
        tol=tol,
        max_iter=max_iter,
    )

    print(f"learning rate used: {lr_used:.9e}")
    print_history(history)

    w_ref = ridge_closed_form(x, y, l2)
    abs_err = float(np.linalg.norm(w_est - w_ref))
    rel_err = relative_error(abs_err, float(np.linalg.norm(w_ref)))
    final_loss = objective_ridge(x, y, w_est, l2)
    final_grad = gradient_ridge(x, y, w_est, l2)
    final_grad_norm = float(np.linalg.norm(final_grad))

    print(f"w* estimate: {w_est}")
    print(f"w* closed-form: {w_ref}")
    print(f"final loss: {final_loss:.9e}")
    print(f"absolute error (vector 2-norm): {abs_err:.9e}")
    print(f"relative error: {rel_err:.9e}")
    print(f"final gradient norm: {final_grad_norm:.9e}")
    print(f"iterations used: {len(history)}")

    return {
        "iterations": float(len(history)),
        "abs_error": abs_err,
        "rel_error": rel_err,
        "final_grad_norm": final_grad_norm,
        "final_loss": final_loss,
    }


def main() -> None:
    tol = 1e-10
    max_iter = 20000

    x1, y1, _ = make_synthetic_case(
        seed=7,
        m=120,
        n=5,
        noise_std=0.05,
        collinearity=0.0,
    )
    x2, y2, _ = make_synthetic_case(
        seed=13,
        m=140,
        n=6,
        noise_std=0.08,
        collinearity=0.85,
    )
    x3, y3, _ = make_synthetic_case(
        seed=21,
        m=200,
        n=12,
        noise_std=0.10,
        collinearity=0.45,
    )

    cases = [
        {
            "name": "Well-conditioned synthetic",
            "X": x1,
            "y": y1,
            "l2": 1e-2,
            "lr": None,
        },
        {
            "name": "Correlated features",
            "X": x2,
            "y": y2,
            "l2": 5e-2,
            "lr": None,
        },
        {
            "name": "Higher-dimensional",
            "X": x3,
            "y": y3,
            "l2": 2e-2,
            "lr": None,
        },
    ]

    results = []
    for case in cases:
        result = run_case(
            name=case["name"],
            x=case["X"],
            y=case["y"],
            l2=case["l2"],
            tol=tol,
            max_iter=max_iter,
            lr=case["lr"],
        )
        results.append(result)

    max_rel_error = max(item["rel_error"] for item in results)
    avg_rel_error = float(np.mean([item["rel_error"] for item in results]))
    max_grad_norm = max(item["final_grad_norm"] for item in results)
    max_iters = max(item["iterations"] for item in results)
    pass_flag = max_rel_error < 1e-7 and max_grad_norm < 1e-8

    print("\n=== Summary ===")
    print(f"max relative error: {max_rel_error:.9e}")
    print(f"avg relative error: {avg_rel_error:.9e}")
    print(f"max final gradient norm: {max_grad_norm:.9e}")
    print(f"max iterations used: {int(max_iters)}")
    print(f"all cases pass strict check: {pass_flag}")


if __name__ == "__main__":
    main()
