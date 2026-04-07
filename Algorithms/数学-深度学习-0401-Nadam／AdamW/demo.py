"""Minimal runnable MVP for Nadam and AdamW (MATH-0401).

This script trains the same tiny MLP twice on a synthetic non-linear dataset:
- once with Nadam (Nesterov-accelerated Adam style update),
- once with AdamW (decoupled weight decay).

Everything is implemented with NumPy only, without calling black-box optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

ArrayDict = Dict[str, np.ndarray]


@dataclass
class OptimizerState:
    """Shared optimizer state structure for Nadam/AdamW."""

    m: ArrayDict
    v: ArrayDict
    t: int


def make_moons_like_dataset(
    n_samples: int = 700,
    noise: float = 0.18,
    seed: int = 401,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a two-moons-like binary dataset with NumPy only."""
    rng = np.random.default_rng(seed)

    n0 = n_samples // 2
    n1 = n_samples - n0

    theta0 = rng.uniform(0.0, np.pi, size=n0)
    theta1 = rng.uniform(0.0, np.pi, size=n1)

    moon0 = np.stack([np.cos(theta0), np.sin(theta0)], axis=1)
    moon1 = np.stack([1.0 - np.cos(theta1), 1.0 - np.sin(theta1) - 0.5], axis=1)

    x = np.vstack([moon0, moon1])
    y = np.concatenate(
        [np.zeros(n0, dtype=np.float64), np.ones(n1, dtype=np.float64)]
    )

    x += rng.normal(0.0, noise, size=x.shape)

    # Shuffle once so split by index has mixed labels.
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    return x[idx], y[idx]


def standardize_features(x: np.ndarray) -> np.ndarray:
    """Standardize each feature to zero mean / unit variance."""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-12
    return (x - mean) / std


def split_train_test(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.75,
    seed: int = 401,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic random split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)

    n_train = int(train_ratio * len(idx))
    tr = idx[:n_train]
    te = idx[n_train:]
    return x[tr], y[tr], x[te], y[te]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z_clip = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def init_mlp_params(
    input_dim: int = 2,
    hidden_dim: int = 16,
    seed: int = 123,
) -> ArrayDict:
    """Initialize a tiny 2-layer MLP."""
    rng = np.random.default_rng(seed)

    w1 = rng.normal(0.0, np.sqrt(1.0 / input_dim), size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    w2 = rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, 1))
    b2 = np.zeros(1, dtype=np.float64)

    return {"W1": w1, "b1": b1, "W2": w2, "b2": b2}


def clone_params(params: ArrayDict) -> ArrayDict:
    """Deep copy parameter dict."""
    return {k: v.copy() for k, v in params.items()}


def mlp_forward(params: ArrayDict, x: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """Forward pass for a 2-layer tanh MLP."""
    z1 = np.einsum("nd,dh->nh", x, params["W1"]) + params["b1"]
    a1 = np.tanh(z1)
    logits = np.einsum("nh,hk->n", a1, params["W2"]) + params["b2"][0]
    prob = sigmoid(logits)
    return prob, (x, a1, prob)


def mlp_loss_and_grads(params: ArrayDict, x: np.ndarray, y: np.ndarray) -> Tuple[float, ArrayDict]:
    """Binary cross-entropy loss and analytic gradients."""
    n = x.shape[0]
    prob, cache = mlp_forward(params, x)
    x_cached, a1, _ = cache

    eps = 1e-12
    bce = -np.mean(y * np.log(prob + eps) + (1.0 - y) * np.log(1.0 - prob + eps))

    dlogits = (prob - y) / n
    grad_w2 = np.einsum("nh,n->h", a1, dlogits)[:, None]
    grad_b2 = np.array([np.sum(dlogits)], dtype=np.float64)

    da1 = np.einsum("n,hk->nh", dlogits, params["W2"].reshape(-1, 1))
    dz1 = da1 * (1.0 - a1 * a1)

    grad_w1 = np.einsum("nd,nh->dh", x_cached, dz1)
    grad_b1 = np.sum(dz1, axis=0)

    grads: ArrayDict = {
        "W1": grad_w1,
        "b1": grad_b1,
        "W2": grad_w2,
        "b2": grad_b2,
    }
    return float(bce), grads


def clip_gradients(grads: ArrayDict, max_norm: float = 5.0) -> Tuple[ArrayDict, float]:
    """Clip global gradient norm for stability."""
    total_sq = 0.0
    for g in grads.values():
        total_sq += float(np.sum(g * g))
    norm = float(np.sqrt(total_sq))

    if norm <= max_norm:
        return grads, norm

    scale = max_norm / (norm + 1e-12)
    clipped = {k: g * scale for k, g in grads.items()}
    return clipped, norm


def init_state(params: ArrayDict) -> OptimizerState:
    """Create zero-initialized optimizer moments."""
    zeros = {k: np.zeros_like(v) for k, v in params.items()}
    return OptimizerState(m=zeros, v=clone_params(zeros), t=0)


def nadam_step(
    params: ArrayDict,
    grads: ArrayDict,
    state: OptimizerState,
    lr: float = 0.006,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[ArrayDict, OptimizerState]:
    """One Nadam update (Dozat-style Nesterov term)."""
    t = state.t + 1
    new_m: ArrayDict = {}
    new_v: ArrayDict = {}

    for key, param in params.items():
        g = grads[key]
        m = beta1 * state.m[key] + (1.0 - beta1) * g
        v = beta2 * state.v[key] + (1.0 - beta2) * (g * g)

        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        g_hat = g / (1.0 - beta1**t)

        # Nesterov look-ahead first moment.
        nesterov_m = beta1 * m_hat + (1.0 - beta1) * g_hat
        params[key] = param - lr * nesterov_m / (np.sqrt(v_hat) + eps)

        new_m[key] = m
        new_v[key] = v

    return params, OptimizerState(m=new_m, v=new_v, t=t)


def adamw_step(
    params: ArrayDict,
    grads: ArrayDict,
    state: OptimizerState,
    lr: float = 0.008,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.02,
    decay_keys: Sequence[str] = ("W1", "W2"),
) -> Tuple[ArrayDict, OptimizerState]:
    """One AdamW update with decoupled weight decay."""
    t = state.t + 1
    new_m: ArrayDict = {}
    new_v: ArrayDict = {}

    for key, param in params.items():
        p = param
        if key in decay_keys and weight_decay > 0.0:
            # Decoupled weight decay, independent from gradient moments.
            p = p * (1.0 - lr * weight_decay)

        g = grads[key]
        m = beta1 * state.m[key] + (1.0 - beta1) * g
        v = beta2 * state.v[key] + (1.0 - beta2) * (g * g)

        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)

        params[key] = p - lr * m_hat / (np.sqrt(v_hat) + eps)

        new_m[key] = m
        new_v[key] = v

    return params, OptimizerState(m=new_m, v=new_v, t=t)


def predict_proba(params: ArrayDict, x: np.ndarray) -> np.ndarray:
    prob, _ = mlp_forward(params, x)
    return prob


def accuracy(params: ArrayDict, x: np.ndarray, y: np.ndarray) -> float:
    y_pred = (predict_proba(params, x) >= 0.5).astype(np.float64)
    return float(np.mean(y_pred == y))


def finite_diff_gradient_check(
    params: ArrayDict,
    x: np.ndarray,
    y: np.ndarray,
    delta: float = 1e-5,
) -> float:
    """Check a few coordinates with central differences."""
    _, analytic = mlp_loss_and_grads(params, x, y)
    checks = [
        ("W1", (0, 0)),
        ("W2", (3, 0)),
        ("b1", (2,)),
        ("b2", (0,)),
    ]

    max_err = 0.0
    for key, idx in checks:
        p_pos = clone_params(params)
        p_neg = clone_params(params)
        p_pos[key][idx] += delta
        p_neg[key][idx] -= delta

        l_pos, _ = mlp_loss_and_grads(p_pos, x, y)
        l_neg, _ = mlp_loss_and_grads(p_neg, x, y)

        numeric = (l_pos - l_neg) / (2.0 * delta)
        err = abs(float(analytic[key][idx]) - numeric)
        max_err = max(max_err, err)

    return max_err


def train_model(
    optimizer_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    seed: int,
) -> Tuple[ArrayDict, List[float]]:
    """Train the same model with either Nadam or AdamW."""
    params = init_mlp_params(seed=seed)
    state = init_state(params)

    loss_history: List[float] = []
    for _ in range(epochs):
        loss, grads = mlp_loss_and_grads(params, x_train, y_train)
        grads, _ = clip_gradients(grads, max_norm=5.0)

        if optimizer_name == "nadam":
            params, state = nadam_step(params, grads, state, lr=0.006)
        elif optimizer_name == "adamw":
            params, state = adamw_step(
                params,
                grads,
                state,
                lr=0.008,
                weight_decay=0.02,
            )
        else:
            raise ValueError(f"unknown optimizer_name: {optimizer_name}")

        loss_history.append(loss)

    return params, loss_history


def summarize_run(
    name: str,
    params: ArrayDict,
    loss_history: List[float],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Compute and print summary metrics for one optimizer run."""
    train_loss, _ = mlp_loss_and_grads(params, x_train, y_train)
    test_loss, _ = mlp_loss_and_grads(params, x_test, y_test)
    train_acc = accuracy(params, x_train, y_train)
    test_acc = accuracy(params, x_test, y_test)

    print(f"[{name}] initial loss: {loss_history[0]:.6f}")
    print(f"[{name}] middle loss:  {loss_history[len(loss_history) // 2]:.6f}")
    print(f"[{name}] final loss:   {loss_history[-1]:.6f}")
    print(f"[{name}] train loss:   {train_loss:.6f}")
    print(f"[{name}] test loss:    {test_loss:.6f}")
    print(f"[{name}] train acc:    {train_acc:.4f}")
    print(f"[{name}] test acc:     {test_acc:.4f}")

    return {
        "initial_loss": float(loss_history[0]),
        "final_loss": float(loss_history[-1]),
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
    }


def main() -> None:
    print("Nadam / AdamW MVP (MATH-0401)")
    print("=" * 72)

    x, y = make_moons_like_dataset(n_samples=700, noise=0.18, seed=401)
    x = standardize_features(x)
    x_train, y_train, x_test, y_test = split_train_test(x, y, train_ratio=0.75, seed=401)

    print(f"dataset: train={x_train.shape[0]}, test={x_test.shape[0]}, dim={x_train.shape[1]}")

    check_params = init_mlp_params(seed=7)
    grad_err = finite_diff_gradient_check(check_params, x_train[:64], y_train[:64], delta=1e-5)
    print(f"gradient-check max abs error: {grad_err:.3e}")
    if grad_err > 3e-4:
        raise RuntimeError("gradient check failed")

    nadam_params, nadam_history = train_model(
        optimizer_name="nadam",
        x_train=x_train,
        y_train=y_train,
        epochs=900,
        seed=123,
    )
    adamw_params, adamw_history = train_model(
        optimizer_name="adamw",
        x_train=x_train,
        y_train=y_train,
        epochs=900,
        seed=123,
    )

    print("-" * 72)
    nadam_metrics = summarize_run(
        "Nadam",
        nadam_params,
        nadam_history,
        x_train,
        y_train,
        x_test,
        y_test,
    )
    print("-" * 72)
    adamw_metrics = summarize_run(
        "AdamW",
        adamw_params,
        adamw_history,
        x_train,
        y_train,
        x_test,
        y_test,
    )

    if not (nadam_metrics["final_loss"] < 0.25 * nadam_metrics["initial_loss"]):
        raise RuntimeError("Nadam did not reduce loss enough")
    if not (adamw_metrics["final_loss"] < 0.25 * adamw_metrics["initial_loss"]):
        raise RuntimeError("AdamW did not reduce loss enough")
    if not (nadam_metrics["test_acc"] > 0.90):
        raise RuntimeError("Nadam test accuracy is unexpectedly low")
    if not (adamw_metrics["test_acc"] > 0.90):
        raise RuntimeError("AdamW test accuracy is unexpectedly low")

    print("=" * 72)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
