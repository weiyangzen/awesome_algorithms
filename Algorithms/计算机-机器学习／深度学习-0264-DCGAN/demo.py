"""Minimal runnable MVP for Deep Convolutional GAN (DCGAN).

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
from scipy import ndimage, stats
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DCGANConfig:
    seed: int = 42
    image_size: int = 32
    latent_dim: int = 64
    g_channels: int = 32
    d_channels: int = 32
    batch_size: int = 128
    epochs: int = 20
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    real_label_smooth: float = 0.9
    max_train_samples: int = 1500
    eval_samples: int = 256
    device: str = "cpu"


class Generator(nn.Module):
    """DCGAN generator: latent z -> 1x32x32 grayscale image."""

    def __init__(self, latent_dim: int, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """DCGAN discriminator: 1x32x32 image -> real/fake logits."""

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_dcgan_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def _resize_digits_to_32(images_8x8: np.ndarray) -> np.ndarray:
    """Upscale sklearn digits (8x8) to 32x32 using bilinear interpolation."""
    upscaled = [ndimage.zoom(img, zoom=4.0, order=1) for img in images_8x8]
    return np.stack(upscaled, axis=0).astype(np.float32)


def load_real_images(cfg: DCGANConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return train/eval image arrays in NCHW format normalized to [-1, 1]."""
    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0  # from [0,16] -> [0,1]
    images = _resize_digits_to_32(images)

    train_images, eval_images = train_test_split(
        images,
        test_size=cfg.eval_samples,
        random_state=cfg.seed,
        shuffle=True,
    )

    if cfg.max_train_samples > 0:
        train_images = train_images[: cfg.max_train_samples]

    train_images = train_images[:, None, :, :] * 2.0 - 1.0
    eval_images = eval_images[:, None, :, :] * 2.0 - 1.0

    return train_images.astype(np.float32), eval_images.astype(np.float32)


def sample_latent(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, latent_dim, 1, 1, device=device)


def build_train_loader(train_images: np.ndarray, batch_size: int) -> torch.utils.data.DataLoader:
    tensor = torch.from_numpy(train_images)
    dataset = torch.utils.data.TensorDataset(tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def train_dcgan(
    cfg: DCGANConfig,
    generator: Generator,
    discriminator: Discriminator,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    criterion = nn.BCEWithLogitsLoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, 0.999))

    history = []

    for epoch in range(1, cfg.epochs + 1):
        d_loss_sum = 0.0
        g_loss_sum = 0.0
        d_real_prob_sum = 0.0
        d_fake_prob_sum = 0.0
        batch_count = 0

        for (real_batch,) in train_loader:
            real_batch = real_batch.to(device)
            bsz = real_batch.size(0)

            # 1) Update discriminator on real and generated images.
            z = sample_latent(bsz, cfg.latent_dim, device)
            fake_batch = generator(z).detach()

            real_targets = torch.full((bsz, 1), cfg.real_label_smooth, device=device)
            fake_targets = torch.zeros((bsz, 1), device=device)

            real_logits = discriminator(real_batch)
            fake_logits = discriminator(fake_batch)

            d_loss_real = criterion(real_logits, real_targets)
            d_loss_fake = criterion(fake_logits, fake_targets)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 2) Update generator to fool discriminator.
            z = sample_latent(bsz, cfg.latent_dim, device)
            generated = generator(z)
            fooled_targets = torch.ones((bsz, 1), device=device)
            fooled_logits = discriminator(generated)
            g_loss = criterion(fooled_logits, fooled_targets)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_loss_sum += float(d_loss.item())
            g_loss_sum += float(g_loss.item())
            d_real_prob_sum += float(torch.sigmoid(real_logits).mean().item())
            d_fake_prob_sum += float(torch.sigmoid(fake_logits).mean().item())
            batch_count += 1

        d_loss_avg = d_loss_sum / max(batch_count, 1)
        g_loss_avg = g_loss_sum / max(batch_count, 1)
        d_real_prob_avg = d_real_prob_sum / max(batch_count, 1)
        d_fake_prob_avg = d_fake_prob_sum / max(batch_count, 1)

        history.append(
            {
                "epoch": epoch,
                "d_loss": d_loss_avg,
                "g_loss": g_loss_avg,
                "d_real_prob": d_real_prob_avg,
                "d_fake_prob": d_fake_prob_avg,
            }
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            print(
                "[train] "
                f"epoch={epoch:03d} "
                f"d_loss={d_loss_avg:.6f} "
                f"g_loss={g_loss_avg:.6f} "
                f"d_real_prob={d_real_prob_avg:.4f} "
                f"d_fake_prob={d_fake_prob_avg:.4f}"
            )

    return pd.DataFrame(history)


@torch.no_grad()
def generate_images(
    generator: Generator,
    n_samples: int,
    latent_dim: int,
    device: torch.device,
) -> np.ndarray:
    z = sample_latent(n_samples, latent_dim, device)
    generated = generator(z).detach().cpu().numpy().astype(np.float32)
    return generated


def denormalize_to_unit(images: np.ndarray) -> np.ndarray:
    return np.clip((images + 1.0) * 0.5, 0.0, 1.0)


def summarize_images(images_unit: np.ndarray) -> Dict[str, float]:
    per_img_mean = images_unit.mean(axis=(1, 2, 3))
    edge_x = ndimage.sobel(images_unit[:, 0, :, :], axis=1, mode="reflect")
    edge_y = ndimage.sobel(images_unit[:, 0, :, :], axis=2, mode="reflect")
    edge_energy = np.sqrt(edge_x**2 + edge_y**2).mean()

    return {
        "pixel_mean": float(images_unit.mean()),
        "pixel_std": float(images_unit.std()),
        "image_mean_q10": float(np.quantile(per_img_mean, 0.10)),
        "image_mean_q50": float(np.quantile(per_img_mean, 0.50)),
        "image_mean_q90": float(np.quantile(per_img_mean, 0.90)),
        "edge_energy": float(edge_energy),
    }


def evaluate_distribution(
    real_unit: np.ndarray,
    generated_unit: np.ndarray,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    table = pd.DataFrame(
        [summarize_images(real_unit), summarize_images(generated_unit)],
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

    flat_real = real_unit.reshape(real_unit.shape[0], -1)
    flat_gen = generated_unit.reshape(generated_unit.shape[0], -1)
    pca = PCA(n_components=12, random_state=seed)
    pca.fit(flat_real)
    proj_real = pca.transform(flat_real)
    proj_gen = pca.transform(flat_gen)
    metrics["pca_mean_l2"] = float(np.linalg.norm(proj_real.mean(axis=0) - proj_gen.mean(axis=0)))
    metrics["pca_var_l1"] = float(np.mean(np.abs(proj_real.var(axis=0) - proj_gen.var(axis=0))))

    return table, metrics


def run_sanity_checks(loss_df: pd.DataFrame, generated_unit: np.ndarray, metrics: Dict[str, float]) -> None:
    if loss_df.empty:
        raise AssertionError("Loss history is empty.")

    numeric = loss_df[["d_loss", "g_loss", "d_real_prob", "d_fake_prob"]].to_numpy()
    if not np.isfinite(numeric).all():
        raise AssertionError("Loss history contains non-finite values.")

    if not np.isfinite(generated_unit).all():
        raise AssertionError("Generated images contain non-finite values.")

    if float(generated_unit.std()) < 0.02:
        raise AssertionError("Generated images have too little variance, likely collapse.")

    if metrics.get("pixel_wasserstein_1", 1.0) > 0.35:
        raise AssertionError("Pixel Wasserstein distance is too large; training likely failed.")


def main() -> None:
    cfg = DCGANConfig()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    train_np, eval_np = load_real_images(cfg)

    train_loader = build_train_loader(train_np, cfg.batch_size)
    generator = Generator(cfg.latent_dim, cfg.g_channels).to(device)
    discriminator = Discriminator(cfg.d_channels).to(device)

    generator.apply(init_dcgan_weights)
    discriminator.apply(init_dcgan_weights)

    loss_df = train_dcgan(
        cfg=cfg,
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        device=device,
    )

    generated_np = generate_images(
        generator=generator,
        n_samples=eval_np.shape[0],
        latent_dim=cfg.latent_dim,
        device=device,
    )

    real_unit = denormalize_to_unit(eval_np)
    generated_unit = denormalize_to_unit(generated_np)

    stats_table, metrics = evaluate_distribution(real_unit=real_unit, generated_unit=generated_unit, seed=cfg.seed)
    run_sanity_checks(loss_df=loss_df, generated_unit=generated_unit, metrics=metrics)

    print("\n=== Training Loss (tail) ===")
    print(loss_df.tail(5).to_string(index=False))

    print("\n=== Distribution Summary ===")
    print(stats_table.to_string(float_format=lambda v: f"{v: .5f}"))

    print("\n=== Metrics ===")
    for key in sorted(metrics.keys()):
        print(f"{key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    main()
