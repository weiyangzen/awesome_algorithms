"""Minimal runnable MVP for Generative Adversarial Network (GAN) on 1D toy data.

Run:
    uv run python demo.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from scipy.stats import ks_2samp, wasserstein_distance
except Exception:  # pragma: no cover - optional dependency fallback
    ks_2samp = None
    wasserstein_distance = None


@dataclass(frozen=True)
class GANConfig:
    seed: int = 42
    n_samples: int = 2048
    latent_dim: int = 4
    hidden_dim: int = 64
    batch_size: int = 256
    epochs: int = 300
    lr_g: float = 1e-3
    lr_d: float = 1e-3
    real_label_smooth: float = 0.9
    device: str = "cpu"


class Generator(nn.Module):
    """Map latent noise z to one-dimensional generated sample x_hat."""

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Binary classifier that outputs real/fake logits for 1D sample x."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_toy_data(n_samples: int, seed: int) -> np.ndarray:
    """Generate a 1D two-mode Gaussian mixture as real data distribution."""
    rng = np.random.default_rng(seed)
    comp_id = rng.integers(0, 2, size=n_samples)
    means = np.where(comp_id == 0, -2.0, 2.0)
    samples = rng.normal(loc=means, scale=0.45, size=n_samples)
    return samples.astype(np.float32)


def sample_latent(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, latent_dim, device=device)


def train_gan(
    cfg: GANConfig,
    real_data: torch.Tensor,
    generator: Generator,
    discriminator: Discriminator,
) -> pd.DataFrame:
    """Alternating updates for discriminator and generator."""
    device = real_data.device
    criterion = nn.BCEWithLogitsLoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.lr_g)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_d)

    n_samples = real_data.shape[0]
    history = []

    for epoch in range(1, cfg.epochs + 1):
        perm = torch.randperm(n_samples, device=device)
        d_loss_sum = 0.0
        g_loss_sum = 0.0
        batch_count = 0

        for start in range(0, n_samples, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            x_real = real_data[idx]
            bsz = x_real.shape[0]

            # 1) Update D: maximize log D(x) + log(1 - D(G(z))).
            z = sample_latent(bsz, cfg.latent_dim, device)
            x_fake = generator(z).detach()

            real_targets = torch.full((bsz, 1), cfg.real_label_smooth, device=device)
            fake_targets = torch.zeros((bsz, 1), device=device)

            real_logits = discriminator(x_real)
            fake_logits = discriminator(x_fake)

            d_loss_real = criterion(real_logits, real_targets)
            d_loss_fake = criterion(fake_logits, fake_targets)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 2) Update G: maximize log D(G(z)) via non-saturating loss.
            z = sample_latent(bsz, cfg.latent_dim, device)
            x_fake = generator(z)
            fooled_targets = torch.ones((bsz, 1), device=device)
            fake_logits_for_g = discriminator(x_fake)
            g_loss = criterion(fake_logits_for_g, fooled_targets)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_loss_sum += float(d_loss.item())
            g_loss_sum += float(g_loss.item())
            batch_count += 1

        d_loss_avg = d_loss_sum / max(batch_count, 1)
        g_loss_avg = g_loss_sum / max(batch_count, 1)
        history.append({"epoch": epoch, "d_loss": d_loss_avg, "g_loss": g_loss_avg})

        if epoch == 1 or epoch % 50 == 0 or epoch == cfg.epochs:
            print(f"[train] epoch={epoch:03d} d_loss={d_loss_avg:.6f} g_loss={g_loss_avg:.6f}")

    return pd.DataFrame(history)


@torch.no_grad()
def generate_samples(
    generator: Generator,
    n_samples: int,
    latent_dim: int,
    device: torch.device,
) -> np.ndarray:
    z = sample_latent(n_samples, latent_dim, device)
    generated = generator(z)
    return generated.squeeze(1).detach().cpu().numpy().astype(np.float32)


def summarize_distribution(data: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "q10": float(np.quantile(data, 0.10)),
        "q50": float(np.quantile(data, 0.50)),
        "q90": float(np.quantile(data, 0.90)),
    }


def fallback_wasserstein(x: np.ndarray, y: np.ndarray) -> float:
    """Approximate Wasserstein-1 by sorted sample matching."""
    n = min(len(x), len(y))
    x_sorted = np.sort(x)[:n]
    y_sorted = np.sort(y)[:n]
    return float(np.mean(np.abs(x_sorted - y_sorted)))


def evaluate_distribution(real: np.ndarray, generated: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    stats_table = pd.DataFrame(
        [summarize_distribution(real), summarize_distribution(generated)],
        index=["real", "generated"],
    )

    metrics: Dict[str, float] = {}

    if wasserstein_distance is not None:
        metrics["wasserstein_1"] = float(wasserstein_distance(real, generated))
    else:
        metrics["wasserstein_1"] = fallback_wasserstein(real, generated)

    data_min = float(min(real.min(), generated.min()))
    data_max = float(max(real.max(), generated.max()))
    bins = np.linspace(data_min, data_max, 41)
    real_hist, _ = np.histogram(real, bins=bins, density=True)
    gen_hist, _ = np.histogram(generated, bins=bins, density=True)
    metrics["hist_l1"] = float(np.mean(np.abs(real_hist - gen_hist)))

    real_pos = float(np.mean(real > 0.0))
    gen_pos = float(np.mean(generated > 0.0))
    metrics["positive_mode_fraction_real"] = real_pos
    metrics["positive_mode_fraction_generated"] = gen_pos
    metrics["positive_mode_fraction_gap"] = float(abs(real_pos - gen_pos))

    if ks_2samp is not None:
        ks_result = ks_2samp(real, generated)
        metrics["ks_stat"] = float(ks_result.statistic)
        metrics["ks_pvalue"] = float(ks_result.pvalue)

    return stats_table, metrics


def run_sanity_checks(loss_df: pd.DataFrame, generated: np.ndarray, metrics: Dict[str, float]) -> None:
    if loss_df.empty:
        raise AssertionError("Loss history is empty.")

    if not np.isfinite(loss_df[["d_loss", "g_loss"]].to_numpy()).all():
        raise AssertionError("Loss contains non-finite values.")

    if not np.isfinite(generated).all():
        raise AssertionError("Generated samples contain non-finite values.")

    # Collapse detector: pure mode collapse often creates extremely small variance.
    if float(np.std(generated)) < 0.10:
        raise AssertionError("Generated distribution variance is too small, likely collapse.")

    if metrics.get("wasserstein_1", 99.0) > 2.5:
        raise AssertionError("Distribution mismatch is too large; training likely failed.")


def main() -> None:
    cfg = GANConfig()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    real_np = make_toy_data(n_samples=cfg.n_samples, seed=cfg.seed)
    real_tensor = torch.from_numpy(real_np).unsqueeze(1).to(device)

    generator = Generator(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim).to(device)
    discriminator = Discriminator(hidden_dim=cfg.hidden_dim).to(device)

    loss_df = train_gan(
        cfg=cfg,
        real_data=real_tensor,
        generator=generator,
        discriminator=discriminator,
    )

    generated_np = generate_samples(
        generator=generator,
        n_samples=cfg.n_samples,
        latent_dim=cfg.latent_dim,
        device=device,
    )

    stats_table, metrics = evaluate_distribution(real=real_np, generated=generated_np)
    run_sanity_checks(loss_df=loss_df, generated=generated_np, metrics=metrics)

    print("\n=== Training Loss (tail) ===")
    print(loss_df.tail(5).to_string(index=False))

    print("\n=== Distribution Summary ===")
    print(stats_table.to_string(float_format=lambda v: f"{v: .4f}"))

    print("\n=== Metrics ===")
    for key in sorted(metrics.keys()):
        print(f"{key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    main()
