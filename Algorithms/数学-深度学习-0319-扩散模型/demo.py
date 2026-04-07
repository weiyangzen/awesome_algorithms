"""Minimal runnable MVP for diffusion model (DDPM on 1D toy data)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.stats import ks_2samp, wasserstein_distance
except Exception:  # pragma: no cover - optional dependency fallback
    ks_2samp = None
    wasserstein_distance = None


@dataclass
class NoiseSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


class NoisePredictor(nn.Module):
    """Predict epsilon_theta(x_t, t) for 1D data."""

    def __init__(self, time_embed_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(1 + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(t, self.time_embed_dim)
        model_input = torch.cat([x_t, t_emb], dim=1)
        return self.net(model_input)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_toy_data(n_samples: int, seed: int) -> np.ndarray:
    """Generate 1D Gaussian-mixture data with two modes."""
    rng = np.random.default_rng(seed)
    mixture_id = rng.integers(0, 2, size=n_samples)
    means = np.where(mixture_id == 0, -2.0, 2.0)
    data = rng.normal(loc=means, scale=0.45, size=n_samples)
    return data.astype(np.float32)


def build_noise_schedule(
    t_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device | None = None,
) -> NoiseSchedule:
    betas = torch.linspace(beta_start, beta_end, t_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return NoiseSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """Sin/cos positional embedding for diffusion timestep."""
    half = dim // 2
    if half == 0:
        return timesteps.float().unsqueeze(1)

    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def train_ddpm(
    model: NoisePredictor,
    x0_data: torch.Tensor,
    schedule: NoiseSchedule,
    epochs: int = 250,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> pd.DataFrame:
    device = x0_data.device
    t_steps = schedule.betas.shape[0]
    n_samples = x0_data.shape[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n_samples, device=device)
        running_loss = 0.0
        batch_count = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            x0 = x0_data[idx]

            t = torch.randint(1, t_steps + 1, (x0.shape[0],), device=device)
            alpha_bar_t = schedule.alpha_bars[t - 1].unsqueeze(1)
            noise = torch.randn_like(x0)
            x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

            noise_pred = model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batch_count += 1

        avg_loss = running_loss / max(batch_count, 1)
        loss_history.append({"epoch": epoch, "mse_loss": avg_loss})

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"[train] epoch={epoch:03d} mse_loss={avg_loss:.6f}")

    return pd.DataFrame(loss_history)


@torch.no_grad()
def sample_ddpm(
    model: NoisePredictor,
    schedule: NoiseSchedule,
    n_samples: int,
    device: torch.device,
) -> np.ndarray:
    t_steps = schedule.betas.shape[0]
    x = torch.randn(n_samples, 1, device=device)

    for t in range(t_steps, 0, -1):
        t_index = t - 1
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

        beta_t = schedule.betas[t_index]
        alpha_t = schedule.alphas[t_index]
        alpha_bar_t = schedule.alpha_bars[t_index]

        noise_pred = model(x, t_batch)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * noise_pred
        )

        if t > 1:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * z
        else:
            x = mean

    return x.squeeze(1).detach().cpu().numpy().astype(np.float32)


def summarize_distribution(data: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "q10": float(np.quantile(data, 0.10)),
        "q50": float(np.quantile(data, 0.50)),
        "q90": float(np.quantile(data, 0.90)),
    }


def fallback_wasserstein(x: np.ndarray, y: np.ndarray) -> float:
    """Approximate Wasserstein-1 distance via sorted quantile matching."""
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

    if ks_2samp is not None:
        ks_result = ks_2samp(real, generated)
        metrics["ks_stat"] = float(ks_result.statistic)
        metrics["ks_pvalue"] = float(ks_result.pvalue)

    return stats_table, metrics


def main() -> None:
    set_seed(42)

    device = torch.device("cpu")
    n_samples = 2048
    t_steps = 100

    real_data_np = make_toy_data(n_samples=n_samples, seed=42)
    x0_data = torch.from_numpy(real_data_np).unsqueeze(1).to(device)

    schedule = build_noise_schedule(t_steps=t_steps, device=device)
    model = NoisePredictor(time_embed_dim=32, hidden_dim=64).to(device)

    loss_df = train_ddpm(
        model=model,
        x0_data=x0_data,
        schedule=schedule,
        epochs=250,
        batch_size=256,
        lr=1e-3,
    )

    generated_np = sample_ddpm(
        model=model,
        schedule=schedule,
        n_samples=n_samples,
        device=device,
    )

    stats_table, metrics = evaluate_distribution(real=real_data_np, generated=generated_np)

    print("\n=== Training Loss (tail) ===")
    print(loss_df.tail(5).to_string(index=False))

    print("\n=== Distribution Summary ===")
    print(stats_table.to_string(float_format=lambda x: f"{x: .4f}"))

    print("\n=== Distance Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
