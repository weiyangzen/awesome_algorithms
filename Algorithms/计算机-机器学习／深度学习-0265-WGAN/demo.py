"""Minimal runnable MVP for Wasserstein GAN (WGAN).

This script implements the original weight-clipping WGAN training loop with
PyTorch on a tiny offline dataset (scikit-learn digits).

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
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class WGANConfig:
    seed: int = 42
    latent_dim: int = 32
    data_dim: int = 64  # 8x8 digits flattened
    g_hidden: int = 128
    c_hidden: int = 128
    batch_size: int = 128
    epochs: int = 45
    n_critic: int = 5
    lr: float = 5e-5
    weight_clip: float = 0.01
    max_train_samples: int = 1400
    eval_samples: int = 256
    device: str = "cpu"


class Generator(nn.Module):
    """Small MLP generator: z -> flattened 8x8 image in [-1, 1]."""

    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Critic(nn.Module):
    """WGAN critic: flattened image -> unrestricted real score."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_digit_vectors(cfg: WGANConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Load sklearn digits and normalize to [-1, 1] flat vectors."""
    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0  # [0, 1]
    vectors = images.reshape(images.shape[0], -1) * 2.0 - 1.0

    train_vecs, eval_vecs = train_test_split(
        vectors,
        test_size=cfg.eval_samples,
        random_state=cfg.seed,
        shuffle=True,
    )

    if cfg.max_train_samples > 0:
        train_vecs = train_vecs[: cfg.max_train_samples]

    return train_vecs.astype(np.float32), eval_vecs.astype(np.float32)


def build_train_loader(train_vectors: np.ndarray, batch_size: int) -> torch.utils.data.DataLoader:
    tensor = torch.from_numpy(train_vectors)
    dataset = torch.utils.data.TensorDataset(tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def sample_latent(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, latent_dim, device=device)


def clip_critic_weights(critic: nn.Module, clip_value: float) -> None:
    for param in critic.parameters():
        param.data.clamp_(-clip_value, clip_value)


def train_wgan(
    cfg: WGANConfig,
    generator: Generator,
    critic: Critic,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """Train original WGAN with weight clipping and RMSprop."""
    opt_g = torch.optim.RMSprop(generator.parameters(), lr=cfg.lr)
    opt_c = torch.optim.RMSprop(critic.parameters(), lr=cfg.lr)

    history = []

    for epoch in range(1, cfg.epochs + 1):
        critic_loss_sum = 0.0
        gen_loss_sum = 0.0
        wasserstein_est_sum = 0.0
        batch_count = 0

        for (real_batch,) in train_loader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.shape[0]

            # 1) Update critic multiple times to better approximate EM distance.
            for _ in range(cfg.n_critic):
                z = sample_latent(batch_size, cfg.latent_dim, device)
                fake_batch = generator(z).detach()

                real_score = critic(real_batch)
                fake_score = critic(fake_batch)

                critic_loss = fake_score.mean() - real_score.mean()

                opt_c.zero_grad(set_to_none=True)
                critic_loss.backward()
                opt_c.step()
                clip_critic_weights(critic, cfg.weight_clip)

            # 2) Update generator once.
            z = sample_latent(batch_size, cfg.latent_dim, device)
            generated = generator(z)
            generator_loss = -critic(generated).mean()

            opt_g.zero_grad(set_to_none=True)
            generator_loss.backward()
            opt_g.step()

            with torch.no_grad():
                real_score_now = critic(real_batch).mean()
                fake_score_now = critic(generated.detach()).mean()
                wasserstein_est = real_score_now - fake_score_now

            critic_loss_sum += float(critic_loss.item())
            gen_loss_sum += float(generator_loss.item())
            wasserstein_est_sum += float(wasserstein_est.item())
            batch_count += 1

        epoch_record = {
            "epoch": epoch,
            "critic_loss": critic_loss_sum / max(batch_count, 1),
            "generator_loss": gen_loss_sum / max(batch_count, 1),
            "wasserstein_est": wasserstein_est_sum / max(batch_count, 1),
        }
        history.append(epoch_record)

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            print(
                "[train] "
                f"epoch={epoch:03d} "
                f"critic_loss={epoch_record['critic_loss']:.6f} "
                f"generator_loss={epoch_record['generator_loss']:.6f} "
                f"wasserstein_est={epoch_record['wasserstein_est']:.6f}"
            )

    return pd.DataFrame(history)


@torch.no_grad()
def generate_vectors(
    generator: Generator,
    n_samples: int,
    latent_dim: int,
    device: torch.device,
) -> np.ndarray:
    z = sample_latent(n_samples, latent_dim, device)
    return generator(z).detach().cpu().numpy().astype(np.float32)


def denormalize_to_unit(vectors: np.ndarray) -> np.ndarray:
    return np.clip((vectors + 1.0) * 0.5, 0.0, 1.0)


def summarize_vectors(vectors_unit: np.ndarray) -> Dict[str, float]:
    per_sample_mean = vectors_unit.mean(axis=1)
    return {
        "pixel_mean": float(vectors_unit.mean()),
        "pixel_std": float(vectors_unit.std()),
        "sample_mean_q10": float(np.quantile(per_sample_mean, 0.10)),
        "sample_mean_q50": float(np.quantile(per_sample_mean, 0.50)),
        "sample_mean_q90": float(np.quantile(per_sample_mean, 0.90)),
    }


def evaluate_distribution(
    real_unit: np.ndarray,
    generated_unit: np.ndarray,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    table = pd.DataFrame(
        [summarize_vectors(real_unit), summarize_vectors(generated_unit)],
        index=["real", "generated"],
    )

    metrics: Dict[str, float] = {}
    real_flat = real_unit.ravel()
    gen_flat = generated_unit.ravel()

    metrics["pixel_wasserstein_1"] = float(stats.wasserstein_distance(real_flat, gen_flat))

    bins = np.linspace(0.0, 1.0, 33)
    real_hist, _ = np.histogram(real_flat, bins=bins, density=True)
    gen_hist, _ = np.histogram(gen_flat, bins=bins, density=True)
    metrics["pixel_hist_l1"] = float(np.mean(np.abs(real_hist - gen_hist)))

    eps = 1e-8
    p = real_hist + eps
    q = gen_hist + eps
    p /= p.sum()
    q /= q.sum()
    metrics["pixel_kl_real_to_gen"] = float(stats.entropy(p, q))

    pca = PCA(n_components=10, random_state=seed)
    pca.fit(real_unit)
    proj_real = pca.transform(real_unit)
    proj_gen = pca.transform(generated_unit)
    metrics["pca_mean_l2"] = float(np.linalg.norm(proj_real.mean(axis=0) - proj_gen.mean(axis=0)))

    return table, metrics


def run_sanity_checks(loss_df: pd.DataFrame, generated_unit: np.ndarray, metrics: Dict[str, float]) -> None:
    if loss_df.empty:
        raise AssertionError("Loss history is empty.")

    numeric = loss_df[["critic_loss", "generator_loss", "wasserstein_est"]].to_numpy()
    if not np.isfinite(numeric).all():
        raise AssertionError("Loss history contains non-finite values.")

    if not np.isfinite(generated_unit).all():
        raise AssertionError("Generated vectors contain non-finite values.")

    if float(generated_unit.std()) < 0.02:
        raise AssertionError("Generated vectors have too little variance, likely mode collapse.")

    if metrics.get("pixel_wasserstein_1", 1.0) > 0.40:
        raise AssertionError("Pixel Wasserstein distance is too large; WGAN training likely failed.")


def main() -> None:
    cfg = WGANConfig()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    train_np, eval_np = load_digit_vectors(cfg)

    train_loader = build_train_loader(train_np, cfg.batch_size)
    generator = Generator(cfg.latent_dim, cfg.g_hidden, cfg.data_dim).to(device)
    critic = Critic(cfg.data_dim, cfg.c_hidden).to(device)

    loss_df = train_wgan(
        cfg=cfg,
        generator=generator,
        critic=critic,
        train_loader=train_loader,
        device=device,
    )

    generated_np = generate_vectors(
        generator=generator,
        n_samples=eval_np.shape[0],
        latent_dim=cfg.latent_dim,
        device=device,
    )

    real_unit = denormalize_to_unit(eval_np)
    generated_unit = denormalize_to_unit(generated_np)

    stats_table, metrics = evaluate_distribution(real_unit=real_unit, generated_unit=generated_unit, seed=cfg.seed)
    run_sanity_checks(loss_df=loss_df, generated_unit=generated_unit, metrics=metrics)

    print("\n=== Training Log (tail) ===")
    print(loss_df.tail(5).to_string(index=False))

    print("\n=== Distribution Summary ===")
    print(stats_table.to_string(float_format=lambda v: f"{v: .5f}"))

    print("\n=== Metrics ===")
    for key in sorted(metrics.keys()):
        print(f"{key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    main()
