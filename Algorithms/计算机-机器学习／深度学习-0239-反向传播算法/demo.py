"""Backpropagation MVP (CS-0110).

This script implements a tiny MLP binary classifier from scratch using NumPy,
with explicit forward and backward passes. It also includes:
- scipy sigmoid for stable activation,
- sklearn dataset + metrics,
- pandas training log snapshots,
- PyTorch autograd gradient alignment check.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
import torch


@dataclass
class BackpropConfig:
    seed: int = 2026
    n_samples: int = 800
    noise: float = 0.24
    test_size: float = 0.30
    hidden_dim: int = 16
    epochs: int = 1600
    lr: float = 0.12
    l2: float = 1e-3
    log_every: int = 100
    gradient_check_points: int = 12
    gradient_check_eps: float = 1e-5


@dataclass
class BackpropResult:
    converged: bool
    final_train_loss: float
    final_test_loss: float
    final_test_acc: float
    best_test_acc: float
    gradcheck_max_rel_error: float
    torch_grad_max_abs_diff: float


class ManualMLP:
    """1-hidden-layer MLP trained with explicit backpropagation."""

    def __init__(self, input_dim: int, hidden_dim: int, rng: np.random.Generator) -> None:
        # Xavier-like scaling for stable starts.
        self.W1 = rng.normal(loc=0.0, scale=np.sqrt(2.0 / (input_dim + hidden_dim)), size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float64)
        self.W2 = rng.normal(loc=0.0, scale=np.sqrt(2.0 / (hidden_dim + 1)), size=(hidden_dim, 1))
        self.b2 = np.zeros((1, 1), dtype=np.float64)

    def forward(self, X: np.ndarray) -> dict[str, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        y_prob = expit(z2)
        return {"z1": z1, "a1": a1, "z2": z2, "y_prob": y_prob}

    def loss(self, y_true: np.ndarray, y_prob: np.ndarray, l2: float) -> float:
        eps = 1e-12
        y_clipped = np.clip(y_prob, eps, 1.0 - eps)
        data_loss = -np.mean(y_true * np.log(y_clipped) + (1.0 - y_true) * np.log(1.0 - y_clipped))
        reg = 0.5 * l2 * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        return float(data_loss + reg)

    def backward(self, X: np.ndarray, y_true: np.ndarray, cache: dict[str, np.ndarray], l2: float) -> dict[str, np.ndarray]:
        n = X.shape[0]

        y_prob = cache["y_prob"]
        a1 = cache["a1"]

        dz2 = (y_prob - y_true) / n
        dW2 = a1.T @ dz2 + l2 * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1.0 - a1 * a1)
        dW1 = X.T @ dz1 + l2 * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
        }

    def step(self, grads: dict[str, np.ndarray], lr: float) -> None:
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)["y_prob"]

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int64)


def make_dataset(config: BackpropConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_moons(
        n_samples=config.n_samples,
        noise=config.noise,
        random_state=config.seed,
    )
    X = X.astype(np.float64)
    y = y.astype(np.float64).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y.ravel().astype(int),
    )

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test, y_train, y_test


def evaluate_model(model: ManualMLP, X: np.ndarray, y: np.ndarray, l2: float) -> dict[str, float]:
    prob = model.predict_proba(X)
    pred = (prob >= 0.5).astype(np.int64)
    loss = model.loss(y, prob, l2=l2)
    acc = accuracy_score(y.ravel().astype(int), pred.ravel())
    f1 = f1_score(y.ravel().astype(int), pred.ravel())
    return {
        "loss": float(loss),
        "acc": float(acc),
        "f1": float(f1),
    }


def _loss_for_gradcheck(model: ManualMLP, X: np.ndarray, y: np.ndarray, l2: float) -> float:
    cache = model.forward(X)
    return model.loss(y, cache["y_prob"], l2=l2)


def gradient_check(
    model: ManualMLP,
    X: np.ndarray,
    y: np.ndarray,
    l2: float,
    eps: float,
    n_points: int,
    rng: np.random.Generator,
) -> float:
    cache = model.forward(X)
    grads = model.backward(X, y, cache, l2=l2)

    tensors = {
        "W1": model.W1,
        "b1": model.b1,
        "W2": model.W2,
        "b2": model.b2,
    }
    analytic = {
        "W1": grads["dW1"],
        "b1": grads["db1"],
        "W2": grads["dW2"],
        "b2": grads["db2"],
    }

    names = list(tensors.keys())
    max_rel_error = 0.0

    for _ in range(n_points):
        name = names[int(rng.integers(0, len(names)))]
        arr = tensors[name]
        idx = tuple(int(rng.integers(0, dim)) for dim in arr.shape)

        old_val = arr[idx]

        arr[idx] = old_val + eps
        loss_pos = _loss_for_gradcheck(model, X, y, l2)

        arr[idx] = old_val - eps
        loss_neg = _loss_for_gradcheck(model, X, y, l2)

        arr[idx] = old_val

        num_grad = (loss_pos - loss_neg) / (2.0 * eps)
        ana_grad = analytic[name][idx]
        rel = abs(num_grad - ana_grad) / max(1e-12, abs(num_grad) + abs(ana_grad))
        if rel > max_rel_error:
            max_rel_error = rel

    return float(max_rel_error)


def torch_gradient_alignment(model: ManualMLP, X: np.ndarray, y: np.ndarray, l2: float) -> float:
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)

    W1 = torch.tensor(model.W1, dtype=torch.float64, requires_grad=True)
    b1 = torch.tensor(model.b1, dtype=torch.float64, requires_grad=True)
    W2 = torch.tensor(model.W2, dtype=torch.float64, requires_grad=True)
    b2 = torch.tensor(model.b2, dtype=torch.float64, requires_grad=True)

    z1 = X_t @ W1 + b1
    a1 = torch.tanh(z1)
    z2 = a1 @ W2 + b2
    y_prob = torch.sigmoid(z2)

    eps = 1e-12
    y_prob = torch.clamp(y_prob, eps, 1.0 - eps)
    data_loss = -torch.mean(y_t * torch.log(y_prob) + (1.0 - y_t) * torch.log(1.0 - y_prob))
    reg = 0.5 * l2 * (torch.sum(W1 * W1) + torch.sum(W2 * W2))
    loss = data_loss + reg
    loss.backward()

    cache = model.forward(X)
    grads = model.backward(X, y, cache, l2=l2)

    diffs = [
        np.max(np.abs(grads["dW1"] - W1.grad.detach().numpy())),
        np.max(np.abs(grads["db1"] - b1.grad.detach().numpy())),
        np.max(np.abs(grads["dW2"] - W2.grad.detach().numpy())),
        np.max(np.abs(grads["db2"] - b2.grad.detach().numpy())),
    ]
    return float(np.max(diffs))


def train_with_backprop(
    model: ManualMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: BackpropConfig,
) -> tuple[pd.DataFrame, BackpropResult]:
    history: list[dict[str, float]] = []

    init_train = evaluate_model(model, X_train, y_train, l2=config.l2)
    init_test = evaluate_model(model, X_test, y_test, l2=config.l2)

    history.append(
        {
            "epoch": 0,
            "train_loss": init_train["loss"],
            "train_acc": init_train["acc"],
            "test_loss": init_test["loss"],
            "test_acc": init_test["acc"],
        }
    )

    for epoch in range(1, config.epochs + 1):
        cache = model.forward(X_train)
        grads = model.backward(X_train, y_train, cache, l2=config.l2)
        model.step(grads, lr=config.lr)

        if epoch % config.log_every == 0 or epoch == 1 or epoch == config.epochs:
            train_m = evaluate_model(model, X_train, y_train, l2=config.l2)
            test_m = evaluate_model(model, X_test, y_test, l2=config.l2)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_m["loss"],
                    "train_acc": train_m["acc"],
                    "test_loss": test_m["loss"],
                    "test_acc": test_m["acc"],
                }
            )

    history_df = pd.DataFrame(history)
    final = history_df.iloc[-1]
    best_test_acc = float(history_df["test_acc"].max())

    rng = np.random.default_rng(config.seed + 7)
    small_X = X_train[:32]
    small_y = y_train[:32]
    gradcheck = gradient_check(
        model=model,
        X=small_X,
        y=small_y,
        l2=config.l2,
        eps=config.gradient_check_eps,
        n_points=config.gradient_check_points,
        rng=rng,
    )

    torch_diff = torch_gradient_alignment(
        model=model,
        X=small_X,
        y=small_y,
        l2=config.l2,
    )

    result = BackpropResult(
        converged=bool(final["train_loss"] < history_df.iloc[0]["train_loss"]),
        final_train_loss=float(final["train_loss"]),
        final_test_loss=float(final["test_loss"]),
        final_test_acc=float(final["test_acc"]),
        best_test_acc=best_test_acc,
        gradcheck_max_rel_error=gradcheck,
        torch_grad_max_abs_diff=torch_diff,
    )

    return history_df, result


def main() -> None:
    config = BackpropConfig()
    rng = np.random.default_rng(config.seed)

    X_train, X_test, y_train, y_test = make_dataset(config)

    model = ManualMLP(input_dim=X_train.shape[1], hidden_dim=config.hidden_dim, rng=rng)

    history_df, result = train_with_backprop(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        config=config,
    )

    y_test_pred = model.predict_label(X_test).ravel().astype(int)
    y_test_true = y_test.ravel().astype(int)

    print("=== Backpropagation MVP (Manual NumPy MLP) ===")
    print(
        f"train_samples={X_train.shape[0]}, test_samples={X_test.shape[0]}, "
        f"features={X_train.shape[1]}, hidden_dim={config.hidden_dim}"
    )
    print(
        f"epochs={config.epochs}, lr={config.lr:.4f}, l2={config.l2:.6f}, "
        f"log_every={config.log_every}"
    )

    print("\n[History Snapshot]")
    print(
        history_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n[Test Classification Report]")
    print(classification_report(y_test_true, y_test_pred, digits=4))

    print("[Backprop Checks]")
    print(
        f"gradcheck_max_rel_error={result.gradcheck_max_rel_error:.3e}, "
        f"torch_grad_max_abs_diff={result.torch_grad_max_abs_diff:.3e}"
    )

    # CI-style acceptance checks.
    init_train_loss = float(history_df.iloc[0]["train_loss"])
    assert result.final_train_loss < init_train_loss, "Train loss did not decrease."
    assert result.best_test_acc >= 0.88, f"Best test accuracy too low: {result.best_test_acc:.4f}"
    assert result.gradcheck_max_rel_error < 1e-4, (
        "Gradient check relative error too high: "
        f"{result.gradcheck_max_rel_error:.3e}"
    )
    assert result.torch_grad_max_abs_diff < 1e-8, (
        "Backprop gradients do not align with torch autograd: "
        f"{result.torch_grad_max_abs_diff:.3e}"
    )

    print(
        "All checks passed. "
        f"final_test_acc={result.final_test_acc:.4f}, best_test_acc={result.best_test_acc:.4f}"
    )


if __name__ == "__main__":
    main()
