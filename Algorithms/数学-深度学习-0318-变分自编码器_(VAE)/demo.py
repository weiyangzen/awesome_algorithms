"""Minimal runnable MVP for Variational AutoEncoder (VAE) on digits data."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainConfig:
    input_dim: int = 64
    hidden_dim: int = 64
    latent_dim: int = 8
    batch_size: int = 128
    epochs: int = 12
    lr: float = 1e-3
    beta: float = 1.0


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def build_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    digits = load_digits()
    x = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    x_train, x_test, _, _ = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_ds = TensorDataset(torch.from_numpy(x_train))
    test_ds = TensorDataset(torch.from_numpy(x_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def run_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    beta: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_n = 0

    for (x_batch,) in loader:
        if is_train:
            optimizer.zero_grad()

        recon, mu, logvar = model(x_batch)
        loss, recon_loss, kl_loss = vae_loss(x_batch, recon, mu, logvar, beta)

        if is_train:
            loss.backward()
            optimizer.step()

        bsz = x_batch.size(0)
        total_n += bsz
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    return {
        "loss_per_sample": total_loss / total_n,
        "recon_per_sample": total_recon / total_n,
        "kl_per_sample": total_kl / total_n,
    }


def evaluate_latent_regularization(model: VAE, loader: DataLoader) -> float:
    model.eval()
    mu_all: list[np.ndarray] = []
    with torch.no_grad():
        for (x_batch,) in loader:
            mu, _ = model.encode(x_batch)
            mu_all.append(mu.numpy())
    mu_mat = np.concatenate(mu_all, axis=0)

    # Compare each latent dimension to N(0,1) using 1D Wasserstein distance.
    distances = []
    for i in range(mu_mat.shape[1]):
        ref = np.random.normal(loc=0.0, scale=1.0, size=mu_mat.shape[0])
        distances.append(wasserstein_distance(mu_mat[:, i], ref))
    return float(np.mean(distances))


def main() -> None:
    seed_everything(42)
    cfg = TrainConfig()

    train_loader, test_loader = build_dataloaders(cfg.batch_size)
    model = VAE(cfg.input_dim, cfg.hidden_dim, cfg.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    logs: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, cfg.beta)
        with torch.no_grad():
            test_metrics = run_epoch(model, test_loader, None, cfg.beta)
        logs.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss_per_sample"],
                "train_recon": train_metrics["recon_per_sample"],
                "train_kl": train_metrics["kl_per_sample"],
                "test_loss": test_metrics["loss_per_sample"],
                "test_recon": test_metrics["recon_per_sample"],
                "test_kl": test_metrics["kl_per_sample"],
            }
        )

    history = pd.DataFrame(logs)
    final_row = history.iloc[-1]

    # Additional post-training diagnostics.
    model.eval()
    with torch.no_grad():
        (x_test_batch,) = next(iter(test_loader))
        recon_batch, _, _ = model(x_test_batch)
        recon_mse = float(torch.mean((x_test_batch - recon_batch) ** 2).item())

        z = torch.randn(16, cfg.latent_dim)
        gen = model.decode(z)
        gen_mean = float(gen.mean().item())
        gen_std = float(gen.std().item())

    latent_wd = evaluate_latent_regularization(model, test_loader)

    print("VAE MVP on sklearn digits")
    print(
        f"epochs={cfg.epochs}, batch_size={cfg.batch_size}, latent_dim={cfg.latent_dim}, beta={cfg.beta}"
    )
    print("final_metrics:")
    print(
        f"  train_loss={final_row['train_loss']:.4f}, "
        f"test_loss={final_row['test_loss']:.4f}, "
        f"test_recon={final_row['test_recon']:.4f}, "
        f"test_kl={final_row['test_kl']:.4f}"
    )
    print(
        f"  recon_mse={recon_mse:.6f}, latent_wasserstein_to_N01={latent_wd:.4f}, "
        f"generated_mean={gen_mean:.4f}, generated_std={gen_std:.4f}"
    )
    print("training_curve_tail:")
    print(history.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
