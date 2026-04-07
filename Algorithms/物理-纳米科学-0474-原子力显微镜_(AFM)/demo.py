"""Atomic Force Microscopy (AFM) minimal runnable MVP.

Pipeline implemented in this script:
1) Generate synthetic nanoscale surface topography.
2) Simulate AFM imaging with finite tip broadening, line drift, and noise.
3) Estimate/remove drift with scikit-learn ridge regression.
4) Reconstruct surface by morphological erosion.
5) Refine reconstruction with a PyTorch inverse-model optimization.

The script is deterministic and requires no interactive input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, grey_erosion, maximum_filter
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class AFMConfig:
    """Configuration for synthetic AFM experiment."""

    seed: int = 20260407
    grid_size: int = 96
    n_particles: int = 20
    tip_radius_px: int = 3
    noise_sigma_nm: float = 0.30

    # Drift amplitudes
    drift_offset_nm: float = 1.2
    drift_slope_nm_per_px: float = 0.015

    # Surface generation
    min_height_nm: float = 1.2
    max_height_nm: float = 8.0
    min_sigma_px: float = 1.5
    max_sigma_px: float = 4.8

    # Torch refinement
    torch_steps: int = 180
    torch_lr: float = 0.06
    tv_weight: float = 0.030
    nonneg_weight: float = 0.20


@dataclass(frozen=True)
class TorchDiagnostics:
    """Diagnostics for the torch optimization stage."""

    final_loss: float
    final_data_loss: float
    final_tv: float


def disk_footprint(radius: int) -> np.ndarray:
    """Build a disk-shaped footprint for grayscale morphology."""
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= (radius * radius)


def generate_surface(cfg: AFMConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic nanoparticle topography (ground truth)."""
    n = cfg.grid_size
    yy, xx = np.mgrid[0:n, 0:n]
    surface = np.zeros((n, n), dtype=np.float64)

    margin = 8
    for _ in range(cfg.n_particles):
        cx = rng.uniform(margin, n - margin)
        cy = rng.uniform(margin, n - margin)
        amp = rng.uniform(cfg.min_height_nm, cfg.max_height_nm)
        sigma = rng.uniform(cfg.min_sigma_px, cfg.max_sigma_px)
        rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
        surface += amp * np.exp(-0.5 * rr2 / (sigma * sigma))

    # Mild long-wavelength substrate undulation.
    x_norm = (xx / (n - 1)) * 2.0 * np.pi
    y_norm = (yy / (n - 1)) * 2.0 * np.pi
    substrate = 0.35 * np.sin(1.3 * x_norm) * np.cos(0.8 * y_norm)
    surface = surface + substrate

    # Physical convention: non-negative heights.
    surface -= np.min(surface)
    return surface


def simulate_line_drift(cfg: AFMConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate scan-line drift map (offset + row-dependent slope)."""
    n = cfg.grid_size
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)

    offset = cfg.drift_offset_nm * (0.60 * np.sin(np.pi * y) + 0.40 * np.cos(2.0 * np.pi * y))
    slope = cfg.drift_slope_nm_per_px * (
        0.70 * np.sin(1.5 * np.pi * y) + 0.30 * rng.normal(0.0, 1.0, size=n)
    )
    slope = gaussian_filter(slope, sigma=2.0)

    drift = offset[:, None] + slope[:, None] * (x[None, :] * n)
    return drift


def forward_afm_scan(
    true_surface: np.ndarray,
    cfg: AFMConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate AFM measurement: dilation + drift + noise."""
    fp = disk_footprint(cfg.tip_radius_px)
    tip_broadened = maximum_filter(true_surface, footprint=fp, mode="nearest")

    drift = simulate_line_drift(cfg, rng)
    white_noise = rng.normal(0.0, cfg.noise_sigma_nm, size=true_surface.shape)
    colored_noise = gaussian_filter(white_noise, sigma=0.8)

    measured = tip_broadened + drift + colored_noise
    return measured, tip_broadened, drift


def estimate_and_remove_drift(measured: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate low-frequency drift plane with ridge regression.

    Fit uses low-height pixels as approximate substrate anchors.
    """
    n = measured.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx.ravel() / (n - 1)) * 2.0 - 1.0
    y = (yy.ravel() / (n - 1)) * 2.0 - 1.0

    design = np.column_stack(
        [
            np.ones_like(x),
            x,
            y,
            x * y,
            x**2,
            y**2,
        ]
    )
    target = measured.ravel()

    # Use lower-height points to reduce nanoparticle influence when fitting drift.
    threshold = np.percentile(target, 45.0)
    mask = target <= threshold

    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(design[mask], target[mask])

    drift_hat = (design @ model.coef_).reshape(measured.shape)
    corrected = measured - drift_hat
    corrected -= np.min(corrected)

    r2 = float(model.score(design[mask], target[mask]))
    return corrected, drift_hat, r2


def morphological_reconstruction(corrected: np.ndarray, tip_radius_px: int) -> np.ndarray:
    """Approximate AFM tip deconvolution using grayscale erosion."""
    fp = disk_footprint(tip_radius_px)
    recon = grey_erosion(corrected, footprint=fp, mode="nearest")
    recon -= np.min(recon)
    return recon


def torch_refine_inverse(
    initial_surface: np.ndarray,
    observed_corrected: np.ndarray,
    tip_radius_px: int,
    steps: int,
    lr: float,
    tv_weight: float,
    nonneg_weight: float,
) -> tuple[np.ndarray, TorchDiagnostics]:
    """Refine reconstruction by differentiable inverse imaging.

    Forward model in torch uses max-pooling as a flat-tip approximation.
    """
    kernel_size = 2 * tip_radius_px + 1

    z = torch.tensor(initial_surface, dtype=torch.float64, requires_grad=True)
    target = torch.tensor(observed_corrected, dtype=torch.float64)

    optimizer = torch.optim.Adam([z], lr=lr)

    final_loss = 0.0
    final_data_loss = 0.0
    final_tv = 0.0

    for _ in range(steps):
        optimizer.zero_grad()

        pred = F.max_pool2d(
            z.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=tip_radius_px,
        ).squeeze(0).squeeze(0)

        data_loss = torch.mean((pred - target) ** 2)
        tv = torch.mean(torch.abs(z[:, 1:] - z[:, :-1])) + torch.mean(torch.abs(z[1:, :] - z[:-1, :]))
        nonneg_penalty = torch.mean(torch.relu(-z))

        loss = data_loss + tv_weight * tv + nonneg_weight * nonneg_penalty
        loss.backward()
        optimizer.step()

        # Height maps should remain physically non-negative.
        with torch.no_grad():
            z.clamp_(min=0.0)

        final_loss = float(loss.detach().cpu().item())
        final_data_loss = float(data_loss.detach().cpu().item())
        final_tv = float(tv.detach().cpu().item())

    refined = z.detach().cpu().numpy()
    return refined, TorchDiagnostics(final_loss=final_loss, final_data_loss=final_data_loss, final_tv=final_tv)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square error."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation on flattened maps."""
    a_flat = a.ravel()
    b_flat = b.ravel()
    c = np.corrcoef(a_flat, b_flat)[0, 1]
    return float(c)


def forward_from_surface(surface: np.ndarray, tip_radius_px: int) -> np.ndarray:
    """AFM forward operator used for consistency diagnostics."""
    fp = disk_footprint(tip_radius_px)
    return maximum_filter(surface, footprint=fp, mode="nearest")


def main() -> None:
    cfg = AFMConfig()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    true_surface = generate_surface(cfg, rng)
    measured, _tip_broadened, true_drift = forward_afm_scan(true_surface, cfg, rng)

    corrected, drift_hat, drift_fit_r2 = estimate_and_remove_drift(measured, alpha=1.0)
    recon_erosion = morphological_reconstruction(corrected, cfg.tip_radius_px)

    recon_torch, torch_diag = torch_refine_inverse(
        initial_surface=recon_erosion,
        observed_corrected=corrected,
        tip_radius_px=cfg.tip_radius_px,
        steps=cfg.torch_steps,
        lr=cfg.torch_lr,
        tv_weight=cfg.tv_weight,
        nonneg_weight=cfg.nonneg_weight,
    )

    rows = []
    for stage_name, stage_map in [
        ("measured_raw", measured),
        ("drift_corrected", corrected),
        ("erosion_recon", recon_erosion),
        ("torch_refined", recon_torch),
    ]:
        rows.append(
            {
                "stage": stage_name,
                "rmse_to_truth_nm": rmse(stage_map, true_surface),
                "corr_to_truth": corrcoef(stage_map, true_surface),
                "forward_rmse_to_corrected_nm": rmse(
                    forward_from_surface(stage_map, cfg.tip_radius_px),
                    corrected,
                ),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values("rmse_to_truth_nm", ignore_index=True)

    output_dir = Path(__file__).resolve().parent
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    print("=== AFM Minimal MVP ===")
    print("[Config]")
    print(pd.Series(asdict(cfg)).to_string())

    print("\n[Drift diagnostics]")
    print(f"true_drift_std_nm: {np.std(true_drift):.4f}")
    print(f"estimated_drift_std_nm: {np.std(drift_hat):.4f}")
    print(f"ridge_r2_on_anchor_pixels: {drift_fit_r2:.4f}")

    print("\n[Torch diagnostics]")
    print(f"final_loss: {torch_diag.final_loss:.6f}")
    print(f"final_data_loss: {torch_diag.final_data_loss:.6f}")
    print(f"final_tv: {torch_diag.final_tv:.6f}")

    print("\n[Reconstruction metrics] (sorted by rmse_to_truth_nm)")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    best_stage = metrics.iloc[0]["stage"]
    assert best_stage in {"erosion_recon", "torch_refined"}, "Reconstruction did not improve over raw scans."

    torch_row = metrics[metrics["stage"] == "torch_refined"].iloc[0]
    measured_row = metrics[metrics["stage"] == "measured_raw"].iloc[0]
    assert (
        torch_row["forward_rmse_to_corrected_nm"] < measured_row["forward_rmse_to_corrected_nm"]
    ), "Torch inverse model failed to improve forward consistency."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
