"""Minimal runnable MVP for 多层感知机 (MLP, CS-0109).

This script implements a small NumPy-based MLP classifier end-to-end:
- deterministic synthetic dataset
- forward pass + backpropagation (no deep-learning framework black box)
- mini-batch gradient descent
- evaluation with reproducible quality gates
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


@dataclass
class TrainSummary:
    epochs: int
    batch_size: int
    learning_rate: float
    l2: float
    final_loss: float


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


class NumpyMLPClassifier:
    """A compact MLP classifier with explicit source-level algorithm steps."""

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (32, 16),
        learning_rate: float = 0.05,
        l2: float = 1e-4,
        epochs: int = 320,
        batch_size: int = 32,
        random_state: int = 0,
    ) -> None:
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes cannot be empty")
        if any(h <= 0 for h in hidden_sizes):
            raise ValueError("all hidden layer sizes must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0.0:
            raise ValueError("l2 must be non-negative")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.classes_: np.ndarray | None = None
        self.loss_curve_: list[float] = []
        self._rng = np.random.default_rng(self.random_state)

    def _init_parameters(self, input_dim: int, n_classes: int) -> None:
        layer_dims = [input_dim, *self.hidden_sizes, n_classes]
        self.weights_.clear()
        self.biases_.clear()

        for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:]):
            scale = np.sqrt(2.0 / fan_in)
            w = self._rng.normal(loc=0.0, scale=scale, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out), dtype=float)
            self.weights_.append(w)
            self.biases_.append(b)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        activations: list[np.ndarray] = [x]
        pre_activations: list[np.ndarray] = []

        a = x
        for layer_idx in range(len(self.weights_) - 1):
            z = a @ self.weights_[layer_idx] + self.biases_[layer_idx]
            a = relu(z)
            pre_activations.append(z)
            activations.append(a)

        logits = a @ self.weights_[-1] + self.biases_[-1]
        pre_activations.append(logits)
        probs = softmax(logits)
        return probs, activations, pre_activations

    def _compute_loss(self, probs: np.ndarray, y_onehot: np.ndarray) -> float:
        eps = 1e-12
        ce_loss = -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))
        l2_penalty = 0.5 * self.l2 * sum(np.sum(w * w) for w in self.weights_)
        return float(ce_loss + l2_penalty)

    def _backward(
        self,
        probs: np.ndarray,
        y_onehot: np.ndarray,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        n_layers = len(self.weights_)
        batch_size = y_onehot.shape[0]

        grad_w = [np.zeros_like(w) for w in self.weights_]
        grad_b = [np.zeros_like(b) for b in self.biases_]

        delta = (probs - y_onehot) / batch_size

        for layer_idx in range(n_layers - 1, -1, -1):
            a_prev = activations[layer_idx]
            grad_w[layer_idx] = a_prev.T @ delta + self.l2 * self.weights_[layer_idx]
            grad_b[layer_idx] = np.sum(delta, axis=0, keepdims=True)

            if layer_idx > 0:
                z_prev = pre_activations[layer_idx - 1]
                delta = (delta @ self.weights_[layer_idx].T) * relu_grad(z_prev)

        return grad_w, grad_b

    def _update(self, grad_w: list[np.ndarray], grad_b: list[np.ndarray]) -> None:
        for idx in range(len(self.weights_)):
            self.weights_[idx] -= self.learning_rate * grad_w[idx]
            self.biases_[idx] -= self.learning_rate * grad_b[idx]

    def fit(self, x: np.ndarray, y: np.ndarray) -> TrainSummary:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)

        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if not np.isfinite(x).all():
            raise ValueError("x contains NaN or Inf")

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("need at least 2 classes")

        self.classes_ = classes
        class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        y_idx = np.array([class_to_index[label] for label in y], dtype=int)

        n_samples, input_dim = x.shape
        n_classes = classes.size
        self._init_parameters(input_dim=input_dim, n_classes=n_classes)

        self.loss_curve_.clear()

        for _ in range(self.epochs):
            permutation = self._rng.permutation(n_samples)
            x_shuffled = x[permutation]
            y_shuffled = y_idx[permutation]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                xb = x_shuffled[start:end]
                yb = y_shuffled[start:end]

                y_onehot = np.eye(n_classes, dtype=float)[yb]
                probs, activations, pre_activations = self._forward(xb)
                loss = self._compute_loss(probs=probs, y_onehot=y_onehot)

                grad_w, grad_b = self._backward(
                    probs=probs,
                    y_onehot=y_onehot,
                    activations=activations,
                    pre_activations=pre_activations,
                )
                self._update(grad_w=grad_w, grad_b=grad_b)

                epoch_loss += loss
                n_batches += 1

            self.loss_curve_.append(epoch_loss / max(1, n_batches))

        return TrainSummary(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            l2=self.l2,
            final_loss=self.loss_curve_[-1],
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None or len(self.weights_) == 0:
            raise RuntimeError("model is not fitted")

        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if not np.isfinite(x).all():
            raise ValueError("x contains NaN or Inf")

        probs, _, _ = self._forward(x)
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        indices = np.argmax(probs, axis=1)
        if self.classes_ is None:
            raise RuntimeError("model is not fitted")
        return self.classes_[indices]


def build_dataset(random_state: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = make_moons(n_samples=1200, noise=0.24, random_state=random_state)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return x_train.astype(float), x_test.astype(float), y_train.astype(int), y_test.astype(int)


def main() -> None:
    x_train, x_test, y_train, y_test = build_dataset(random_state=7)

    model = NumpyMLPClassifier(
        hidden_sizes=(32, 16),
        learning_rate=0.05,
        l2=1e-4,
        epochs=320,
        batch_size=32,
        random_state=0,
    )
    summary = model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")

    print("=== CS-0109 多层感知机 (MLP) MVP ===")
    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
    print(
        "Config: "
        f"hidden={model.hidden_sizes}, lr={summary.learning_rate}, "
        f"l2={summary.l2}, epochs={summary.epochs}, batch_size={summary.batch_size}"
    )
    print(f"Final training loss: {summary.final_loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test F1 (binary): {f1:.4f}")
    print(f"Sample probabilities (first 3 rows):\n{np.round(y_proba[:3], 4)}")

    if not np.isfinite(summary.final_loss):
        raise AssertionError("final_loss must be finite")
    if np.any(~np.isfinite(y_proba)):
        raise AssertionError("predicted probabilities contain NaN/Inf")
    if accuracy < 0.88:
        raise AssertionError(f"accuracy too low: {accuracy:.4f}")
    if f1 < 0.88:
        raise AssertionError(f"f1 too low: {f1:.4f}")

    print("All checks passed.")


if __name__ == "__main__":
    main()
