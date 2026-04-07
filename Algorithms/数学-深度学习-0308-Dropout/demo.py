"""Dropout MVP: minimal, runnable, and fully transparent (no black-box trainer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def bce_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-9) -> float:
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_hat = (y_prob >= 0.5).astype(np.float64)
    return float(np.mean(y_hat == y_true))


def make_dataset(
    seed: int = 42,
    n_train: int = 220,
    n_test: int = 780,
    d: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a moderately overfit-prone binary dataset with fixed seed."""
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    x = rng.normal(0.0, 1.0, size=(n_total, d))
    w_true = rng.normal(0.0, 1.0, size=(d,))
    nonlinear = 0.8 * x[:, 0] * x[:, 1] - 0.6 * (x[:, 2] ** 2) + 0.35 * np.sin(x[:, 3] * x[:, 4])
    logits = x @ w_true + nonlinear
    prob = sigmoid(logits / 3.2)
    y = (prob > 0.5).astype(np.float64).reshape(-1, 1)

    # Add some train-time label noise to make overfitting behavior visible.
    noise_idx = rng.choice(n_train, size=max(1, n_train // 8), replace=False)
    y[noise_idx] = 1.0 - y[noise_idx]

    # Standardize by train stats only.
    x_train = x[:n_train].copy()
    y_train = y[:n_train].copy()
    x_test = x[n_train:].copy()
    y_test = y[n_train:].copy()

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-8
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, y_train, x_test, y_test


@dataclass
class TrainResult:
    final_loss: float
    train_acc: float
    test_acc: float
    gap: float


class BinaryMLPWithDropout:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        seed: int,
    ) -> None:
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError("dropout_rate must satisfy 0 <= p < 1.")
        self.dropout_rate = dropout_rate
        self.rng = np.random.default_rng(seed)

        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.w1 = self.rng.normal(0.0, scale1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float64)
        self.w2 = self.rng.normal(0.0, scale2, size=(hidden_dim, 1))
        self.b2 = np.zeros((1, 1), dtype=np.float64)

        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        z1 = x @ self.w1 + self.b1
        h = np.maximum(0.0, z1)

        if training and self.dropout_rate > 0.0:
            keep_prob = 1.0 - self.dropout_rate
            mask = (self.rng.random(h.shape) < keep_prob).astype(np.float64) / keep_prob
            h_used = h * mask
        else:
            mask = np.ones_like(h, dtype=np.float64)
            h_used = h

        logits = h_used @ self.w2 + self.b2
        y_prob = sigmoid(logits)

        self._cache = {
            "x": x,
            "z1": z1,
            "h": h,
            "mask": mask,
            "h_used": h_used,
            "y_prob": y_prob,
        }
        return y_prob

    def backward(self, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        x = self._cache["x"]
        z1 = self._cache["z1"]
        h = self._cache["h"]
        mask = self._cache["mask"]
        h_used = self._cache["h_used"]
        y_prob = self._cache["y_prob"]
        n = y_true.shape[0]

        dlogits = (y_prob - y_true) / n
        dw2 = h_used.T @ dlogits
        db2 = np.sum(dlogits, axis=0, keepdims=True)

        dh_used = dlogits @ self.w2.T
        dh = dh_used * mask
        dz1 = dh * (z1 > 0.0)
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}

    def step(self, grads: Dict[str, np.ndarray], lr: float) -> None:
        self.w1 -= lr * grads["dw1"]
        self.b1 -= lr * grads["db1"]
        self.w2 -= lr * grads["dw2"]
        self.b2 -= lr * grads["db2"]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, training=False)


def train_model(
    model: BinaryMLPWithDropout,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    lr: float = 0.06,
    epochs: int = 800,
) -> TrainResult:
    report_every = max(1, epochs // 4)
    loss_val = float("nan")
    for ep in range(1, epochs + 1):
        y_prob_train = model.forward(x_train, training=True)
        loss_val = bce_loss(y_train, y_prob_train)
        if not np.isfinite(loss_val):
            raise RuntimeError(f"loss became non-finite at epoch={ep}")
        grads = model.backward(y_train)
        model.step(grads, lr=lr)

        if ep % report_every == 0 or ep == 1:
            y_prob_train_eval = model.predict_proba(x_train)
            y_prob_test_eval = model.predict_proba(x_test)
            tr_acc = accuracy(y_train, y_prob_train_eval)
            te_acc = accuracy(y_test, y_prob_test_eval)
            print(
                f"epoch={ep:4d} "
                f"loss={loss_val:.4f} "
                f"train_acc={tr_acc:.4f} "
                f"test_acc={te_acc:.4f}"
            )

    y_prob_train_eval = model.predict_proba(x_train)
    y_prob_test_eval = model.predict_proba(x_test)
    train_acc = accuracy(y_train, y_prob_train_eval)
    test_acc = accuracy(y_test, y_prob_test_eval)
    return TrainResult(
        final_loss=loss_val,
        train_acc=train_acc,
        test_acc=test_acc,
        gap=train_acc - test_acc,
    )


def run_experiment(
    dropout_rate: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> TrainResult:
    model = BinaryMLPWithDropout(
        input_dim=x_train.shape[1],
        hidden_dim=64,
        dropout_rate=dropout_rate,
        seed=seed,
    )
    return train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        lr=0.06,
        epochs=800,
    )


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    x_train, y_train, x_test, y_test = make_dataset(seed=42)
    print(
        f"Dataset: train={x_train.shape[0]}x{x_train.shape[1]}, "
        f"test={x_test.shape[0]}x{x_test.shape[1]}"
    )

    print("\n[1/2] Baseline(no dropout)")
    baseline = run_experiment(
        dropout_rate=0.0,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        seed=7,
    )

    print("\n[2/2] Dropout(p=0.5)")
    dropout = run_experiment(
        dropout_rate=0.5,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        seed=7,
    )

    print("\n=== Final Summary ===")
    print(
        "Baseline(no dropout): "
        f"loss={baseline.final_loss:.4f}, "
        f"train_acc={baseline.train_acc:.4f}, "
        f"test_acc={baseline.test_acc:.4f}, "
        f"gap={baseline.gap:.4f}"
    )
    print(
        "Dropout(p=0.5):      "
        f"loss={dropout.final_loss:.4f}, "
        f"train_acc={dropout.train_acc:.4f}, "
        f"test_acc={dropout.test_acc:.4f}, "
        f"gap={dropout.gap:.4f}"
    )
    print(f"Gap improvement (baseline - dropout): {baseline.gap - dropout.gap:.4f}")

    # Minimal quality gates for deterministic MVP behavior.
    assert np.isfinite(baseline.final_loss) and np.isfinite(dropout.final_loss)
    assert baseline.test_acc > 0.60
    assert dropout.test_acc > 0.60
    assert dropout.gap <= baseline.gap + 0.03
    print("All checks passed.")


if __name__ == "__main__":
    main()
