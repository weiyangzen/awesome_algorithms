"""Minimal runnable MVP for backpropagation (MATH-0304).

Core training is implemented with NumPy (explicit forward/backward).
If SciPy / scikit-learn / pandas / PyTorch are available, the script uses them;
otherwise it falls back to pure-NumPy equivalents so `python3 demo.py` remains runnable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
except ModuleNotFoundError:
    pd = None
    HAS_PANDAS = False

try:
    from scipy.optimize import check_grad as scipy_check_grad
    from scipy.special import expit as scipy_expit

    HAS_SCIPY = True
except ModuleNotFoundError:
    scipy_check_grad = None
    scipy_expit = None
    HAS_SCIPY = False

try:
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ModuleNotFoundError:
    make_moons = None
    train_test_split = None
    StandardScaler = None
    HAS_SKLEARN = False

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ModuleNotFoundError:
    torch = None
    F = None
    HAS_TORCH = False


ArrayDict = Dict[str, np.ndarray]


@dataclass
class ForwardCache:
    """Intermediate tensors needed by backpropagation."""

    x: np.ndarray
    z1: np.ndarray
    a1: np.ndarray
    z2: np.ndarray
    p: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid with SciPy fast path."""
    if HAS_SCIPY:
        return scipy_expit(x)
    x_clip = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _make_moons_fallback(n_samples: int, noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy fallback for make_moons."""
    rng = np.random.default_rng(seed)

    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = rng.uniform(0.0, np.pi, size=n_outer)
    theta_inner = rng.uniform(0.0, np.pi, size=n_inner)

    outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    inner = np.column_stack([1.0 - np.cos(theta_inner), 1.0 - np.sin(theta_inner) - 0.5])

    x = np.vstack([outer, inner]).astype(np.float64)
    y = np.concatenate([np.zeros(n_outer), np.ones(n_inner)]).astype(np.float64)

    x += rng.normal(loc=0.0, scale=noise, size=x.shape)
    return x, y


def _stratified_split_fallback(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NumPy fallback for stratified train/test split."""
    rng = np.random.default_rng(seed)

    idx0 = np.where(y == 0.0)[0]
    idx1 = np.where(y == 1.0)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_test = int(round(idx0.size * test_size))
    n1_test = int(round(idx1.size * test_size))

    test_idx = np.concatenate([idx0[:n0_test], idx1[:n1_test]])
    train_idx = np.concatenate([idx0[n0_test:], idx1[n1_test:]])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def _standardize_fallback(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy fallback for standardization."""
    mu = np.mean(x_train, axis=0, keepdims=True)
    sigma = np.std(x_train, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (x_train - mu) / sigma, (x_test - mu) / sigma


def make_dataset(seed: int = 304) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate and standardize binary data."""
    if HAS_SKLEARN:
        x, y = make_moons(n_samples=1200, noise=0.22, random_state=seed)
        x = x.astype(np.float64)
        y = y.astype(np.float64)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=seed,
            stratify=y,
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        return x_train, x_test, y_train, y_test

    x, y = _make_moons_fallback(n_samples=1200, noise=0.22, seed=seed)
    x_train, x_test, y_train, y_test = _stratified_split_fallback(x, y, test_size=0.25, seed=seed)
    x_train, x_test = _standardize_fallback(x_train, x_test)
    return x_train, x_test, y_train, y_test


def init_params(input_dim: int, hidden_dim: int, seed: int = 304) -> ArrayDict:
    """Initialize MLP parameters with fan-in scaling."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, 1.0 / np.sqrt(input_dim), size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), size=(hidden_dim, 1))
    b2 = np.zeros(1, dtype=np.float64)
    return {"W1": w1, "b1": b1, "W2": w2, "b2": b2}


def forward_pass(params: ArrayDict, x: np.ndarray) -> ForwardCache:
    """Compute forward propagation for a one-hidden-layer MLP."""
    z1 = np.einsum("nd,dh->nh", x, params["W1"], optimize=True) + params["b1"]
    a1 = np.tanh(z1)
    z2 = np.einsum("nh,hk->nk", a1, params["W2"], optimize=True) + params["b2"]
    p = sigmoid(z2)
    return ForwardCache(x=x, z1=z1, a1=a1, z2=z2, p=p)


def loss_and_grads(
    params: ArrayDict,
    x: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-3,
) -> Tuple[float, ArrayDict, np.ndarray]:
    """Compute BCE loss and analytical gradients via backpropagation."""
    cache = forward_pass(params, x)
    y_col = y.reshape(-1, 1)
    n = x.shape[0]

    eps = 1e-12
    bce = -np.mean(y_col * np.log(cache.p + eps) + (1.0 - y_col) * np.log(1.0 - cache.p + eps))
    reg = 0.5 * l2 * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))
    loss = float(bce + reg)

    dz2 = (cache.p - y_col) / n
    dW2 = np.einsum("nh,nk->hk", cache.a1, dz2, optimize=True) + l2 * params["W2"]
    db2 = np.sum(dz2, axis=0)

    da1 = np.einsum("nk,hk->nh", dz2, params["W2"], optimize=True)
    dz1 = da1 * (1.0 - cache.a1**2)
    dW1 = np.einsum("nd,nh->dh", cache.x, dz1, optimize=True) + l2 * params["W1"]
    db1 = np.sum(dz1, axis=0)

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    return loss, grads, cache.p


def predict(params: ArrayDict, x: np.ndarray) -> np.ndarray:
    """Predict binary labels."""
    probs = forward_pass(params, x).p.ravel()
    return (probs >= 0.5).astype(np.float64)


def accuracy(params: ArrayDict, x: np.ndarray, y: np.ndarray) -> float:
    """Compute classification accuracy."""
    return float(np.mean(predict(params, x) == y))


def flatten_params(params: ArrayDict) -> np.ndarray:
    """Flatten parameter dictionary into one vector."""
    return np.concatenate(
        [
            params["W1"].ravel(),
            params["b1"].ravel(),
            params["W2"].ravel(),
            params["b2"].ravel(),
        ]
    )


def unflatten_params(vec: np.ndarray, input_dim: int, hidden_dim: int) -> ArrayDict:
    """Recover parameter dictionary from one vector."""
    idx = 0

    w1_size = input_dim * hidden_dim
    w1 = vec[idx : idx + w1_size].reshape(input_dim, hidden_dim)
    idx += w1_size

    b1 = vec[idx : idx + hidden_dim]
    idx += hidden_dim

    w2_size = hidden_dim
    w2 = vec[idx : idx + w2_size].reshape(hidden_dim, 1)
    idx += w2_size

    b2 = vec[idx : idx + 1]

    return {
        "W1": w1.copy(),
        "b1": b1.copy(),
        "W2": w2.copy(),
        "b2": b2.copy(),
    }


def flatten_grads(grads: ArrayDict) -> np.ndarray:
    """Flatten gradient dictionary."""
    return np.concatenate(
        [
            grads["W1"].ravel(),
            grads["b1"].ravel(),
            grads["W2"].ravel(),
            grads["b2"].ravel(),
        ]
    )


def _finite_diff_grad_error(
    f,
    g,
    x0: np.ndarray,
    delta: float = 1e-6,
) -> float:
    """NumPy fallback for gradient-check error (L2 norm)."""
    analytic = g(x0)
    numeric = np.zeros_like(analytic)

    for i in range(x0.size):
        xp = x0.copy()
        xn = x0.copy()
        xp[i] += delta
        xn[i] -= delta
        numeric[i] = (f(xp) - f(xn)) / (2.0 * delta)

    return float(np.linalg.norm(analytic - numeric))


def scipy_gradient_check(
    params: ArrayDict,
    x: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    l2: float = 1e-3,
) -> Tuple[float, str]:
    """Gradient check with SciPy when available, else NumPy fallback."""

    def f(vec: np.ndarray) -> float:
        p = unflatten_params(vec, input_dim=input_dim, hidden_dim=hidden_dim)
        loss, _, _ = loss_and_grads(p, x, y, l2=l2)
        return loss

    def g(vec: np.ndarray) -> np.ndarray:
        p = unflatten_params(vec, input_dim=input_dim, hidden_dim=hidden_dim)
        _, grads, _ = loss_and_grads(p, x, y, l2=l2)
        return flatten_grads(grads)

    x0 = flatten_params(params)
    if HAS_SCIPY:
        return float(scipy_check_grad(f, g, x0)), "scipy.check_grad"
    return _finite_diff_grad_error(f, g, x0), "numpy finite-diff"


def torch_gradient_compare(
    params: ArrayDict,
    x: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-3,
) -> Dict[str, float] | None:
    """Compare manual gradients against PyTorch autograd on same batch."""
    if not HAS_TORCH:
        return None

    _, manual_grads, _ = loss_and_grads(params, x, y, l2=l2)

    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float64)

    w1 = torch.tensor(params["W1"], dtype=torch.float64, requires_grad=True)
    b1 = torch.tensor(params["b1"], dtype=torch.float64, requires_grad=True)
    w2 = torch.tensor(params["W2"], dtype=torch.float64, requires_grad=True)
    b2 = torch.tensor(params["b2"], dtype=torch.float64, requires_grad=True)

    z1 = x_t @ w1 + b1
    a1 = torch.tanh(z1)
    logits = a1 @ w2 + b2

    loss_torch = F.binary_cross_entropy_with_logits(logits, y_t)
    loss_torch = loss_torch + 0.5 * l2 * (torch.sum(w1 * w1) + torch.sum(w2 * w2))
    loss_torch.backward()

    auto_grads = {
        "W1": w1.grad.detach().cpu().numpy(),
        "b1": b1.grad.detach().cpu().numpy(),
        "W2": w2.grad.detach().cpu().numpy(),
        "b2": b2.grad.detach().cpu().numpy(),
    }

    return {
        name: float(np.max(np.abs(manual_grads[name] - auto_grads[name])))
        for name in ("W1", "b1", "W2", "b2")
    }


def train(
    params: ArrayDict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 1200,
    lr: float = 0.12,
    l2: float = 1e-3,
    log_every: int = 120,
) -> Tuple[ArrayDict, List[Dict[str, float]], object]:
    """Train MLP with full-batch SGD and return logs."""
    records: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_loss, grads, _ = loss_and_grads(params, x_train, y_train, l2=l2)

        params["W1"] -= lr * grads["W1"]
        params["b1"] -= lr * grads["b1"]
        params["W2"] -= lr * grads["W2"]
        params["b2"] -= lr * grads["b2"]

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            train_acc = accuracy(params, x_train, y_train)
            test_acc = accuracy(params, x_test, y_test)
            records.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "test_acc": float(test_acc),
                }
            )

    log_obj = pd.DataFrame.from_records(records) if HAS_PANDAS else records
    return params, records, log_obj


def print_log_summary(log_obj: object, records: List[Dict[str, float]]) -> Tuple[float, float, Dict[str, float]]:
    """Print logs and return (initial_loss, final_loss, best_record)."""
    print("-" * 72)
    if HAS_PANDAS:
        df = log_obj
        print("training log head:")
        print(df.head(4).to_string(index=False))
        print("-" * 72)
        print("training log tail:")
        print(df.tail(4).to_string(index=False))
        best_row = df.loc[df["test_acc"].idxmax()]
        best_record = {
            "epoch": float(best_row["epoch"]),
            "train_loss": float(best_row["train_loss"]),
            "train_acc": float(best_row["train_acc"]),
            "test_acc": float(best_row["test_acc"]),
        }
    else:
        print("training log (first 4 rows):")
        for row in records[:4]:
            print(
                f"epoch={int(row['epoch'])}, "
                f"train_loss={row['train_loss']:.6f}, "
                f"train_acc={row['train_acc']:.4f}, "
                f"test_acc={row['test_acc']:.4f}"
            )
        print("-" * 72)
        print("training log (last 4 rows):")
        for row in records[-4:]:
            print(
                f"epoch={int(row['epoch'])}, "
                f"train_loss={row['train_loss']:.6f}, "
                f"train_acc={row['train_acc']:.4f}, "
                f"test_acc={row['test_acc']:.4f}"
            )
        best_record = max(records, key=lambda row: row["test_acc"])

    initial_loss = records[0]["train_loss"]
    final_loss = records[-1]["train_loss"]
    return initial_loss, final_loss, best_record


def main() -> None:
    np.random.seed(304)
    if HAS_TORCH:
        torch.manual_seed(304)

    print("Backpropagation MVP (MATH-0304)")
    print("=" * 72)
    print(
        "deps: "
        f"scipy={'yes' if HAS_SCIPY else 'no'}, "
        f"sklearn={'yes' if HAS_SKLEARN else 'no'}, "
        f"pandas={'yes' if HAS_PANDAS else 'no'}, "
        f"torch={'yes' if HAS_TORCH else 'no'}"
    )

    x_train, x_test, y_train, y_test = make_dataset(seed=304)

    input_dim = x_train.shape[1]
    hidden_dim = 12
    params = init_params(input_dim=input_dim, hidden_dim=hidden_dim, seed=304)

    # Gradient checks on a smaller subset for speed.
    x_small = x_train[:64]
    y_small = y_train[:64]

    grad_err, grad_method = scipy_gradient_check(
        params=params,
        x=x_small,
        y=y_small,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        l2=1e-3,
    )
    print(f"gradient check ({grad_method}) error: {grad_err:.3e}")

    torch_errs = torch_gradient_compare(params=params, x=x_small, y=y_small, l2=1e-3)
    if torch_errs is None:
        print("PyTorch not installed; skip autograd gradient comparison.")
    else:
        print("max abs gradient diff vs torch autograd:")
        for name, err in torch_errs.items():
            print(f"  {name}: {err:.3e}")

    trained_params, records, log_obj = train(
        params=params,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=1200,
        lr=0.12,
        l2=1e-3,
        log_every=120,
    )

    final_train_acc = accuracy(trained_params, x_train, y_train)
    final_test_acc = accuracy(trained_params, x_test, y_test)

    initial_loss, final_loss, best_record = print_log_summary(log_obj=log_obj, records=records)

    print("-" * 72)
    print(
        "best checkpoint: "
        f"epoch={int(best_record['epoch'])}, "
        f"train_loss={best_record['train_loss']:.6f}, "
        f"train_acc={best_record['train_acc']:.4f}, "
        f"test_acc={best_record['test_acc']:.4f}"
    )
    print(f"final train acc: {final_train_acc:.4f}")
    print(f"final test acc:  {final_test_acc:.4f}")

    if not (grad_err < 5e-5):
        raise RuntimeError("Gradient check failed: error too large.")
    if torch_errs is not None and not all(err < 5e-8 for err in torch_errs.values()):
        raise RuntimeError("PyTorch autograd gradient comparison failed.")
    if not (final_loss < initial_loss):
        raise RuntimeError("Training loss did not decrease.")
    if not (final_test_acc > 0.85):
        raise RuntimeError("Final test accuracy is unexpectedly low.")

    print("=" * 72)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
