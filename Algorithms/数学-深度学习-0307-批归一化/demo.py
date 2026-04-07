"""Batch Normalization MVP with NumPy implementation and PyTorch alignment checks."""

from __future__ import annotations

import numpy as np
import torch


class NumpyBatchNorm1D:
    """Minimal 1D BatchNorm for input shape (batch_size, num_features)."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not (0.0 < momentum <= 1.0):
            raise ValueError("momentum must be in (0, 1]")

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = np.ones(num_features, dtype=np.float64)
        self.beta = np.zeros(num_features, dtype=np.float64)
        self.running_mean = np.zeros(num_features, dtype=np.float64)
        self.running_var = np.ones(num_features, dtype=np.float64)

        self._cache: dict[str, np.ndarray] | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected 2D input (N, C), got shape={x.shape}")
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"feature mismatch: input has C={x.shape[1]}, expected C={self.num_features}"
            )

        if training:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            inv_std = 1.0 / np.sqrt(batch_var + self.eps)
            x_hat = (x - batch_mean) * inv_std
            y = self.gamma * x_hat + self.beta

            # Match PyTorch BatchNorm: running_var tracks unbiased variance.
            if x.shape[0] > 1:
                unbiased_var = batch_var * (x.shape[0] / (x.shape[0] - 1.0))
            else:
                unbiased_var = batch_var

            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * unbiased_var

            self._cache = {
                "x": x,
                "x_hat": x_hat,
                "mean": batch_mean,
                "var": batch_var,
                "inv_std": inv_std,
            }
            return y

        inv_std = 1.0 / np.sqrt(self.running_var + self.eps)
        x_hat = (x - self.running_mean) * inv_std
        return self.gamma * x_hat + self.beta

    def backward(self, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dy = np.asarray(dy, dtype=np.float64)
        if self._cache is None:
            raise RuntimeError("backward called before a training forward pass")
        if dy.ndim != 2:
            raise ValueError(f"expected 2D grad (N, C), got shape={dy.shape}")

        x = self._cache["x"]
        x_hat = self._cache["x_hat"]
        mean = self._cache["mean"]
        var = self._cache["var"]
        inv_std = self._cache["inv_std"]

        if dy.shape != x.shape:
            raise ValueError(f"grad shape mismatch: dy={dy.shape}, x={x.shape}")

        n = x.shape[0]

        dgamma = np.sum(dy * x_hat, axis=0)
        dbeta = np.sum(dy, axis=0)

        dx_hat = dy * self.gamma
        x_mu = x - mean

        dvar = np.sum(dx_hat * x_mu * (-0.5) * (var + self.eps) ** (-1.5), axis=0)
        dmean = np.sum(dx_hat * (-inv_std), axis=0) + dvar * np.mean(-2.0 * x_mu, axis=0)

        dx = dx_hat * inv_std + dvar * (2.0 * x_mu / n) + dmean / n
        return dx, dgamma, dbeta


def run_forward_alignment() -> dict[str, float]:
    rng = np.random.default_rng(7)
    n, c = 16, 6
    x_train = rng.normal(loc=3.0, scale=4.0, size=(n, c)).astype(np.float64)
    x_eval = rng.normal(loc=-2.0, scale=5.0, size=(n, c)).astype(np.float64)

    eps = 1e-5
    momentum = 0.1

    np_bn = NumpyBatchNorm1D(num_features=c, eps=eps, momentum=momentum)
    np_bn.gamma = rng.normal(loc=1.0, scale=0.2, size=c)
    np_bn.beta = rng.normal(loc=0.0, scale=0.3, size=c)

    torch_bn = torch.nn.BatchNorm1d(c, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
    torch_bn = torch_bn.to(dtype=torch.float64)

    with torch.no_grad():
        torch_bn.weight.copy_(torch.from_numpy(np_bn.gamma))
        torch_bn.bias.copy_(torch.from_numpy(np_bn.beta))
        torch_bn.running_mean.copy_(torch.from_numpy(np_bn.running_mean))
        torch_bn.running_var.copy_(torch.from_numpy(np_bn.running_var))

    torch_bn.train()
    y_np_train = np_bn.forward(x_train, training=True)
    y_t_train = torch_bn(torch.from_numpy(x_train)).detach().cpu().numpy()

    torch_bn.eval()
    y_np_eval = np_bn.forward(x_eval, training=False)
    y_t_eval = torch_bn(torch.from_numpy(x_eval)).detach().cpu().numpy()

    if np_bn._cache is None:
        raise RuntimeError("missing cache after training forward")
    x_hat = np_bn._cache["x_hat"]
    mean_abs = float(np.max(np.abs(x_hat.mean(axis=0))))
    var_abs = float(np.max(np.abs(x_hat.var(axis=0) - 1.0)))

    return {
        "forward_train_max_abs": float(np.max(np.abs(y_np_train - y_t_train))),
        "forward_eval_max_abs": float(np.max(np.abs(y_np_eval - y_t_eval))),
        "running_mean_max_abs": float(
            np.max(np.abs(np_bn.running_mean - torch_bn.running_mean.detach().cpu().numpy()))
        ),
        "running_var_max_abs": float(
            np.max(np.abs(np_bn.running_var - torch_bn.running_var.detach().cpu().numpy()))
        ),
        "train_output_mean_abs": mean_abs,
        "train_output_var_abs": var_abs,
    }


def run_backward_alignment() -> dict[str, float]:
    rng = np.random.default_rng(2026)
    n, c = 10, 5

    x = rng.normal(loc=0.5, scale=2.0, size=(n, c)).astype(np.float64)
    upstream = rng.normal(loc=0.0, scale=1.0, size=(n, c)).astype(np.float64)

    eps = 1e-5
    momentum = 0.1

    np_bn = NumpyBatchNorm1D(num_features=c, eps=eps, momentum=momentum)
    np_bn.gamma = rng.normal(loc=1.0, scale=0.15, size=c)
    np_bn.beta = rng.normal(loc=0.0, scale=0.25, size=c)

    y_np = np_bn.forward(x, training=True)
    dx_np, dgamma_np, dbeta_np = np_bn.backward(upstream)

    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    upstream_t = torch.tensor(upstream, dtype=torch.float64)

    torch_bn = torch.nn.BatchNorm1d(c, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
    torch_bn = torch_bn.to(dtype=torch.float64)

    with torch.no_grad():
        torch_bn.weight.copy_(torch.from_numpy(np_bn.gamma))
        torch_bn.bias.copy_(torch.from_numpy(np_bn.beta))

    torch_bn.train()
    y_t = torch_bn(x_t)
    loss = torch.sum(y_t * upstream_t)
    loss.backward()

    dx_t = x_t.grad.detach().cpu().numpy()
    dgamma_t = torch_bn.weight.grad.detach().cpu().numpy()
    dbeta_t = torch_bn.bias.grad.detach().cpu().numpy()

    return {
        "backward_dx_max_abs": float(np.max(np.abs(dx_np - dx_t))),
        "backward_dgamma_max_abs": float(np.max(np.abs(dgamma_np - dgamma_t))),
        "backward_dbeta_max_abs": float(np.max(np.abs(dbeta_np - dbeta_t))),
        "forward_train_recheck": float(np.max(np.abs(y_np - y_t.detach().cpu().numpy()))),
    }


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    forward_stats = run_forward_alignment()
    backward_stats = run_backward_alignment()

    print("=== BatchNorm MVP: NumPy vs PyTorch ===")
    print(f"Forward(train) max abs diff: {forward_stats['forward_train_max_abs']:.3e}")
    print(f"Forward(eval)  max abs diff: {forward_stats['forward_eval_max_abs']:.3e}")
    print(f"Running mean   max abs diff: {forward_stats['running_mean_max_abs']:.3e}")
    print(f"Running var    max abs diff: {forward_stats['running_var_max_abs']:.3e}")
    print(f"Train output mean(abs):      {forward_stats['train_output_mean_abs']:.3e}")
    print(f"Train output var(abs):       {forward_stats['train_output_var_abs']:.3e}")
    print(f"Backward dx     max abs diff: {backward_stats['backward_dx_max_abs']:.3e}")
    print(f"Backward dgamma max abs diff: {backward_stats['backward_dgamma_max_abs']:.3e}")
    print(f"Backward dbeta  max abs diff: {backward_stats['backward_dbeta_max_abs']:.3e}")

    assert forward_stats["forward_train_max_abs"] < 1e-10
    assert forward_stats["forward_eval_max_abs"] < 1e-10
    assert forward_stats["running_mean_max_abs"] < 1e-12
    assert forward_stats["running_var_max_abs"] < 1e-12
    assert forward_stats["train_output_mean_abs"] < 1e-10
    assert forward_stats["train_output_var_abs"] < 5e-5

    assert backward_stats["backward_dx_max_abs"] < 1e-10
    assert backward_stats["backward_dgamma_max_abs"] < 1e-10
    assert backward_stats["backward_dbeta_max_abs"] < 1e-12
    assert backward_stats["forward_train_recheck"] < 1e-10

    print("All BatchNorm checks passed.")


if __name__ == "__main__":
    main()
