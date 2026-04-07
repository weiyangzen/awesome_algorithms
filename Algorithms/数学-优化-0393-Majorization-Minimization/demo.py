"""Majorization-Minimization (MM) MVP: L1-regularized logistic regression.

This demo implements a classic MM scheme by majorizing the smooth logistic
loss with a global quadratic upper bound and minimizing the surrogate exactly
via proximal updates (soft-thresholding for L1 part).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

HistoryItem = Tuple[int, float, float, float, int]


def check_finite_matrix(name: str, x: np.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def check_binary_vector(name: str, y: np.ndarray) -> None:
    if y.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={y.shape}.")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{name} contains non-finite values.")
    unique = np.unique(y)
    if not np.all(np.isin(unique, [0.0, 1.0])):
        raise ValueError(f"{name} must contain only 0/1 labels, got {unique}.")


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0.0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def linear_response(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.einsum("ij,j->i", x, w) + b


def logistic_objective(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lam: float,
) -> float:
    z = linear_response(x, w, b)
    # mean(log(1 + exp(z)) - y*z) + lam * ||w||_1
    loss = np.mean(np.logaddexp(0.0, z) - y * z)
    reg = lam * np.sum(np.abs(w))
    return float(loss + reg)


def logistic_grad(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
) -> Tuple[np.ndarray, float]:
    n = x.shape[0]
    z = linear_response(x, w, b)
    p = sigmoid(z)
    residual = p - y
    grad_w = np.einsum("ij,i->j", x, residual) / n
    grad_b = float(np.mean(residual))
    return grad_w, grad_b


def estimate_lipschitz_constant(x: np.ndarray) -> float:
    n = x.shape[0]
    x_aug = np.hstack([x, np.ones((n, 1), dtype=float)])
    # For logistic loss average, Hessian <= 0.25 / n * X_aug^T X_aug.
    spectral_norm_sq = float(np.linalg.norm(x_aug, ord=2) ** 2)
    return 0.25 * spectral_norm_sq / n


def mm_l1_logistic(
    x: np.ndarray,
    y: np.ndarray,
    lam: float = 0.08,
    tol: float = 1e-7,
    max_iter: int = 2000,
) -> Tuple[np.ndarray, float, List[HistoryItem]]:
    check_finite_matrix("X", x)
    check_binary_vector("y", y)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Shape mismatch: X={x.shape}, y={y.shape}.")
    if lam < 0.0:
        raise ValueError("lam must be >= 0.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")

    n, d = x.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    lipschitz = estimate_lipschitz_constant(x)
    if not np.isfinite(lipschitz) or lipschitz <= 0.0:
        raise RuntimeError(f"Invalid Lipschitz constant: {lipschitz}.")

    history: List[HistoryItem] = []
    prev_obj = logistic_objective(x, y, w, b, lam)

    for k in range(1, max_iter + 1):
        grad_w, grad_b = logistic_grad(x, y, w, b)
        grad_norm = float(np.sqrt(np.dot(grad_w, grad_w) + grad_b * grad_b))

        w_tilde = w - grad_w / lipschitz
        b_tilde = b - grad_b / lipschitz

        w_next = soft_threshold(w_tilde, lam / lipschitz)
        b_next = b_tilde

        step_norm = float(
            np.sqrt(np.dot(w_next - w, w_next - w) + (b_next - b) ** 2)
        )
        current_obj = logistic_objective(x, y, w_next, b_next, lam)

        # MM should be monotone; allow tiny numerical tolerance.
        if current_obj > prev_obj + 1e-12:
            raise RuntimeError(
                "Objective increased unexpectedly; MM monotonicity violated. "
                f"prev={prev_obj:.12e}, curr={current_obj:.12e}"
            )

        nnz = int(np.count_nonzero(np.abs(w_next) > 1e-10))
        history.append((k, current_obj, grad_norm, step_norm, nnz))

        w, b = w_next, b_next
        obj_drop = prev_obj - current_obj
        prev_obj = current_obj

        if grad_norm <= tol and step_norm <= np.sqrt(tol):
            return w, b, history
        if obj_drop <= tol * (1.0 + abs(current_obj)) and step_norm <= np.sqrt(tol):
            return w, b, history

    raise RuntimeError(f"MM did not converge within max_iter={max_iter}.")


def predict_proba(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(linear_response(x, w, b))


def predict_label(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return (predict_proba(x, w, b) >= 0.5).astype(float)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def make_synthetic_data(
    seed: int = 42,
    n_samples: int = 600,
    n_features: int = 24,
    train_ratio: float = 0.75,
) -> Dict[str, np.ndarray]:
    if n_samples <= 10 or n_features <= 2:
        raise ValueError("Need larger n_samples/n_features for this demo.")
    if not (0.1 < train_ratio < 0.9):
        raise ValueError("train_ratio must be in (0.1, 0.9).")

    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n_samples, n_features))

    w_true = np.zeros(n_features, dtype=float)
    active = np.array([1, 4, 7, 11, 16, 20])
    w_true[active] = np.array([2.2, -1.6, 1.4, -2.1, 1.1, -1.3])
    b_true = -0.35

    logits = linear_response(x, w_true, b_true)
    probs = sigmoid(logits)
    y = rng.binomial(1, probs).astype(float)

    split = int(n_samples * train_ratio)
    idx = rng.permutation(n_samples)
    train_idx = idx[:split]
    test_idx = idx[split:]

    return {
        "x_train": x[train_idx],
        "y_train": y[train_idx],
        "x_test": x[test_idx],
        "y_test": y[test_idx],
        "w_true": w_true,
        "b_true": np.array([b_true], dtype=float),
    }


def print_history(history: Sequence[HistoryItem], max_lines: int = 14) -> None:
    print("iter | objective        | ||grad||         | ||step||         | nnz(w)")
    print("-" * 75)
    for row in history[:max_lines]:
        k, obj, gnorm, snorm, nnz = row
        print(f"{k:4d} | {obj:16.9e} | {gnorm:16.9e} | {snorm:16.9e} | {nnz:6d}")
    if len(history) > max_lines:
        print(f"... ({len(history) - max_lines} more iterations omitted)")


def support_metrics(w_true: np.ndarray, w_est: np.ndarray, thresh: float = 1e-2) -> Dict[str, float]:
    true_support = np.abs(w_true) > thresh
    est_support = np.abs(w_est) > thresh

    tp = int(np.sum(true_support & est_support))
    fp = int(np.sum(~true_support & est_support))
    fn = int(np.sum(true_support & ~est_support))

    precision = tp / (tp + fp + 1e-15)
    recall = tp / (tp + fn + 1e-15)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
    }


def main() -> None:
    data = make_synthetic_data(seed=7, n_samples=640, n_features=24, train_ratio=0.75)

    lam = 0.07
    tol = 1e-7
    max_iter = 2500

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    w_true = data["w_true"]

    print("=== Majorization-Minimization for L1 Logistic Regression ===")
    print(f"train shape: {x_train.shape}, test shape: {x_test.shape}")
    print(f"lambda: {lam}, tol: {tol}, max_iter: {max_iter}")

    w_hat, b_hat, history = mm_l1_logistic(
        x=x_train,
        y=y_train,
        lam=lam,
        tol=tol,
        max_iter=max_iter,
    )

    print_history(history)

    train_obj = logistic_objective(x_train, y_train, w_hat, b_hat, lam)
    test_obj = logistic_objective(x_test, y_test, w_hat, b_hat, lam)

    y_train_pred = predict_label(x_train, w_hat, b_hat)
    y_test_pred = predict_label(x_test, w_hat, b_hat)
    train_acc = accuracy(y_train, y_train_pred)
    test_acc = accuracy(y_test, y_test_pred)

    support_stat = support_metrics(w_true, w_hat)

    monotone_ok = all(
        history[i][1] <= history[i - 1][1] + 1e-12 for i in range(1, len(history))
    )

    print("\n=== Result Summary ===")
    print(f"iterations: {len(history)}")
    print(f"train objective: {train_obj:.9e}")
    print(f"test objective:  {test_obj:.9e}")
    print(f"train accuracy:  {train_acc:.6f}")
    print(f"test accuracy:   {test_acc:.6f}")
    print(f"estimated intercept b: {b_hat:.9e}")
    print(f"nnz(w_hat): {np.count_nonzero(np.abs(w_hat) > 1e-2)} / {w_hat.size}")
    print(f"support precision: {support_stat['precision']:.6f}")
    print(f"support recall:    {support_stat['recall']:.6f}")
    print(f"objective monotone non-increasing: {monotone_ok}")

    if not monotone_ok:
        raise RuntimeError("Monotonicity check failed.")


if __name__ == "__main__":
    main()
