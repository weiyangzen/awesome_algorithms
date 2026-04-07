"""StyleGAN minimal runnable MVP on synthetic 16x16 grayscale images.

This script keeps StyleGAN's core ideas in a compact form:
- mapping network (z -> w),
- per-layer style modulation via AdaIN,
- per-layer stochastic noise injection,
- style mixing regularization,
- truncation trick at sampling.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    seed: int = 42
    image_size: int = 16
    dataset_size: int = 1536
    batch_size: int = 64
    latent_dim: int = 64
    w_dim: int = 64
    epochs: int = 10
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.0, 0.99)
    mixing_prob: float = 0.8
    truncation_psi: float = 0.7


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_synthetic_real_images(
    n_samples: int,
    image_size: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Create a simple multi-modal real distribution in [-1, 1]."""
    xs = np.linspace(-1.0, 1.0, image_size, dtype=np.float32)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")

    images = np.zeros((n_samples, image_size, image_size), dtype=np.float32)
    for i in range(n_samples):
        # Compose 1-3 Gaussian blobs with random centers and scales.
        blob_count = int(rng.integers(1, 4))
        img = np.zeros((image_size, image_size), dtype=np.float32)
        for _ in range(blob_count):
            cx = float(rng.uniform(-0.65, 0.65))
            cy = float(rng.uniform(-0.65, 0.65))
            sigma = float(rng.uniform(0.12, 0.35))
            amp = float(rng.uniform(0.5, 1.2))
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            img += amp * np.exp(-dist2 / (2.0 * sigma * sigma))

        # Add a weak oriented sine texture to increase diversity.
        theta = float(rng.uniform(0.0, np.pi))
        freq = float(rng.uniform(1.0, 3.5))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        texture = np.sin(freq * (np.cos(theta) * xx + np.sin(theta) * yy) * np.pi + phase)
        img += 0.15 * texture.astype(np.float32)

        # Normalize to [0, 1], then map to [-1, 1].
        img -= img.min()
        denom = max(float(img.max()), 1e-6)
        img = img / denom
        img = np.clip(img + rng.normal(0.0, 0.03, size=img.shape).astype(np.float32), 0.0, 1.0)
        images[i] = img * 2.0 - 1.0

    images = images[:, None, :, :]  # NCHW
    return torch.from_numpy(images)


class PixelNorm(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + self.eps)


class MappingNetwork(nn.Module):
    """z -> w mapping network."""

    def __init__(self, latent_dim: int, w_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            PixelNorm(),
            nn.Linear(latent_dim, w_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(w_dim, w_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(w_dim, w_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class StyledConv(nn.Module):
    """Conv + noise injection + AdaIN style modulation."""

    def __init__(self, in_channels: int, out_channels: int, w_dim: int, upsample: bool = False) -> None:
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.noise_weight = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.style_affine = nn.Linear(w_dim, out_channels * 2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)

        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_weight * noise

        x = self.act(x)
        x = self.instance_norm(x)

        style = self.style_affine(w).view(w.shape[0], 2, x.shape[1], 1, 1)
        gamma = style[:, 0]
        beta = style[:, 1]
        x = (1.0 + gamma) * x + beta
        return x


class Generator(nn.Module):
    """Tiny style-based generator for 16x16 grayscale outputs."""

    def __init__(self, w_dim: int) -> None:
        super().__init__()
        self.num_layers = 6
        self.const = nn.Parameter(torch.randn(1, 64, 4, 4))

        self.layers = nn.ModuleList(
            [
                StyledConv(64, 64, w_dim, upsample=False),
                StyledConv(64, 64, w_dim, upsample=False),
                StyledConv(64, 48, w_dim, upsample=True),
                StyledConv(48, 48, w_dim, upsample=False),
                StyledConv(48, 32, w_dim, upsample=True),
                StyledConv(32, 32, w_dim, upsample=False),
            ]
        )
        self.to_rgb = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, ws: Iterable[torch.Tensor]) -> torch.Tensor:
        ws_list = list(ws)
        if len(ws_list) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} style tensors, got {len(ws_list)}")

        batch = ws_list[0].shape[0]
        x = self.const.expand(batch, -1, -1, -1)
        for layer, w in zip(self.layers, ws_list):
            x = layer(x, w)
        image = torch.tanh(self.to_rgb(x))
        return image


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),  # 16 -> 8
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),  # 8 -> 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_mixed_ws(
    mapping: MappingNetwork,
    z1: torch.Tensor,
    z2: Optional[torch.Tensor],
    num_layers: int,
    mixing_prob: float,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Return per-layer w list; optionally perform style mixing."""
    w1 = mapping(z1)

    do_mixing = z2 is not None and random.random() < mixing_prob
    if do_mixing:
        w2 = mapping(z2)
        cutoff = random.randint(1, num_layers - 1)
        ws = [w1 if i < cutoff else w2 for i in range(num_layers)]
    else:
        ws = [w1 for _ in range(num_layers)]

    return ws, w1


def sample_with_truncation(
    mapping: MappingNetwork,
    generator: Generator,
    z: torch.Tensor,
    w_avg: torch.Tensor,
    psi: float,
) -> torch.Tensor:
    with torch.no_grad():
        w = mapping(z)
        w_trunc = w_avg.unsqueeze(0) + psi * (w - w_avg.unsqueeze(0))
        ws = [w_trunc for _ in range(generator.num_layers)]
        out = generator(ws)
    return out


def train_stylegan_mvp(cfg: TrainConfig) -> None:
    set_global_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg.seed)

    real_images = make_synthetic_real_images(
        n_samples=cfg.dataset_size,
        image_size=cfg.image_size,
        rng=rng,
    )
    train_loader = DataLoader(
        TensorDataset(real_images),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    mapping = MappingNetwork(cfg.latent_dim, cfg.w_dim).to(device)
    generator = Generator(cfg.w_dim).to(device)
    discriminator = Discriminator().to(device)

    g_params = list(mapping.parameters()) + list(generator.parameters())
    g_opt = torch.optim.Adam(g_params, lr=cfg.lr, betas=cfg.betas)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=cfg.betas)

    w_avg = torch.zeros(cfg.w_dim, device=device)

    print(f"device: {device}")
    print(f"real dataset shape: {tuple(real_images.shape)}")
    print(
        "model params: "
        f"mapping={sum(p.numel() for p in mapping.parameters())}, "
        f"generator={sum(p.numel() for p in generator.parameters())}, "
        f"discriminator={sum(p.numel() for p in discriminator.parameters())}"
    )

    fixed_z = torch.randn(16, cfg.latent_dim, device=device)

    global_step = 0
    last_d_loss = float("nan")
    last_g_loss = float("nan")

    for epoch in range(1, cfg.epochs + 1):
        for (real_batch,) in train_loader:
            real_batch = real_batch.to(device)
            batch = real_batch.shape[0]

            # 1) Discriminator step.
            z1 = torch.randn(batch, cfg.latent_dim, device=device)
            z2 = torch.randn(batch, cfg.latent_dim, device=device)
            ws, w_primary = build_mixed_ws(
                mapping=mapping,
                z1=z1,
                z2=z2,
                num_layers=generator.num_layers,
                mixing_prob=cfg.mixing_prob,
            )

            with torch.no_grad():
                fake_batch = generator(ws)

            d_opt.zero_grad(set_to_none=True)
            real_logits = discriminator(real_batch)
            fake_logits = discriminator(fake_batch)
            d_loss = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
            d_loss.backward()
            d_opt.step()

            # 2) Generator (and mapping) step.
            z1 = torch.randn(batch, cfg.latent_dim, device=device)
            z2 = torch.randn(batch, cfg.latent_dim, device=device)
            ws, w_primary = build_mixed_ws(
                mapping=mapping,
                z1=z1,
                z2=z2,
                num_layers=generator.num_layers,
                mixing_prob=cfg.mixing_prob,
            )

            g_opt.zero_grad(set_to_none=True)
            fake_batch = generator(ws)
            fake_logits = discriminator(fake_batch)
            g_loss = F.softplus(-fake_logits).mean()
            g_loss.backward()
            g_opt.step()

            with torch.no_grad():
                w_avg.mul_(0.995).add_(0.005 * w_primary.mean(dim=0))

            global_step += 1
            last_d_loss = float(d_loss.item())
            last_g_loss = float(g_loss.item())

        # Epoch logging with truncation sampling stats.
        sampled = sample_with_truncation(mapping, generator, fixed_z, w_avg, cfg.truncation_psi)
        s_mean = float(sampled.mean().item())
        s_std = float(sampled.std().item())
        s_min = float(sampled.min().item())
        s_max = float(sampled.max().item())

        if epoch == 1 or epoch % 2 == 0 or epoch == cfg.epochs:
            print(
                f"epoch {epoch:02d} | step={global_step:04d} | "
                f"d_loss={last_d_loss:.4f} g_loss={last_g_loss:.4f} | "
                f"sample(mean={s_mean:.4f}, std={s_std:.4f}, min={s_min:.3f}, max={s_max:.3f})"
            )

    # Final report and lightweight checks.
    real_mean = float(real_images.mean().item())
    real_std = float(real_images.std().item())

    final_samples = sample_with_truncation(mapping, generator, fixed_z, w_avg, cfg.truncation_psi)
    gen_mean = float(final_samples.mean().item())
    gen_std = float(final_samples.std().item())

    print("final stats:")
    print(f"  real(mean={real_mean:.4f}, std={real_std:.4f})")
    print(f"  gen (mean={gen_mean:.4f}, std={gen_std:.4f})")
    print(f"  final losses: d={last_d_loss:.4f}, g={last_g_loss:.4f}")

    # Quick sanity checks for this MVP.
    if not np.isfinite(last_d_loss) or not np.isfinite(last_g_loss):
        raise RuntimeError("non-finite GAN loss detected")
    if gen_std < 0.08:
        raise RuntimeError(f"generator collapsed (std too small): {gen_std:.4f}")
    if abs(gen_mean - real_mean) > 0.35:
        raise RuntimeError(
            "generated mean diverges too much from real distribution: "
            f"|{gen_mean:.4f} - {real_mean:.4f}| > 0.35"
        )

    # Show a compact per-sample summary instead of saving files.
    print("sample summary (first 6 generated images):")
    flat = final_samples[:6].view(6, -1)
    for i in range(6):
        img_mean = float(flat[i].mean().item())
        img_std = float(flat[i].std().item())
        img_max = float(flat[i].max().item())
        print(f"  idx={i:02d} mean={img_mean:+.4f} std={img_std:.4f} max={img_max:.4f}")

    print("All checks passed.")


def main() -> None:
    cfg = TrainConfig()
    train_stylegan_mvp(cfg)


if __name__ == "__main__":
    main()
