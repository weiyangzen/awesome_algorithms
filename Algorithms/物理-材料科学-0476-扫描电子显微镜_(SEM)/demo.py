"""SEM (Scanning Electron Microscopy) analysis MVP.

The script builds a tiny but complete pipeline:
1) synthesize SEM-like microstructure images with defects and sensor noise,
2) denoise + edge extraction,
3) unsupervised defect segmentation,
4) connected-component feature extraction,
5) defect-level classification with sklearn and PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import (
    binary_closing,
    binary_erosion,
    binary_opening,
    gaussian_filter,
    label as cc_label,
)
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SEMConfig:
    seed: int = 20260407
    n_images: int = 24
    image_size: int = 96
    grain_count: int = 26
    photon_count: float = 75.0
    elec_noise_sigma: float = 0.03
    min_region_area: int = 8


def make_grain_microstructure(size: int, grain_count: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Voronoi-like grain intensity field."""
    centers = rng.uniform(0, size, size=(grain_count, 2))
    yy, xx = np.indices((size, size))
    dist2 = (yy[..., None] - centers[:, 0]) ** 2 + (xx[..., None] - centers[:, 1]) ** 2
    grain_ids = np.argmin(dist2, axis=2)
    grain_brightness = np.clip(rng.normal(loc=0.56, scale=0.11, size=grain_count), 0.20, 0.92)
    base = grain_brightness[grain_ids]
    return gaussian_filter(base, sigma=0.9)


def draw_thick_line_mask(
    shape: tuple[int, int],
    p0: tuple[int, int],
    p1: tuple[int, int],
    half_width: int,
) -> np.ndarray:
    """Rasterize a thick line into a boolean mask."""
    h, w = shape
    y0, x0 = p0
    y1, x1 = p1
    steps = int(max(abs(y1 - y0), abs(x1 - x0)) * 1.8) + 1
    ys = np.linspace(y0, y1, steps).astype(np.int32)
    xs = np.linspace(x0, x1, steps).astype(np.int32)
    mask = np.zeros(shape, dtype=bool)
    for y, x in zip(ys, xs):
        y_min = max(0, y - half_width)
        y_max = min(h, y + half_width + 1)
        x_min = max(0, x - half_width)
        x_max = min(w, x + half_width + 1)
        mask[y_min:y_max, x_min:x_max] = True
    return mask


def inject_defects(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    """Inject pore / inclusion / scratch defects and return ground-truth metadata."""
    out = image.copy()
    h, w = out.shape
    yy, xx = np.indices((h, w))
    defect_mask = np.zeros_like(out, dtype=bool)

    high_defect_mode = bool(rng.random() < 0.5)
    if high_defect_mode:
        pore_count = int(rng.integers(4, 9))
        pore_radius_range = (3, 8)
        inclusion_count = int(rng.integers(1, 4))
        scratch_count = int(rng.integers(1, 3))
    else:
        pore_count = int(rng.integers(1, 4))
        pore_radius_range = (2, 5)
        inclusion_count = int(rng.integers(0, 3))
        scratch_count = int(rng.integers(0, 2))

    for _ in range(pore_count):
        cy = int(rng.integers(6, h - 6))
        cx = int(rng.integers(6, w - 6))
        radius = int(rng.integers(pore_radius_range[0], pore_radius_range[1]))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        out[mask] -= float(rng.uniform(0.18, 0.36))
        defect_mask |= mask

    for _ in range(inclusion_count):
        cy = int(rng.integers(4, h - 4))
        cx = int(rng.integers(4, w - 4))
        radius = int(rng.integers(1, 4))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        out[mask] += float(rng.uniform(0.10, 0.22))
        defect_mask |= mask

    for _ in range(scratch_count):
        p0 = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        p1 = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        half_width = int(rng.integers(1, 3))
        mask = draw_thick_line_mask((h, w), p0, p1, half_width=half_width)
        out[mask] -= float(rng.uniform(0.12, 0.26))
        defect_mask |= mask

    out = np.clip(out, 0.0, 1.0)
    gt_area_ratio = float(defect_mask.mean())
    gt_label = int(high_defect_mode)
    gt = {
        "true_pore_count": float(pore_count),
        "true_inclusion_count": float(inclusion_count),
        "true_scratch_count": float(scratch_count),
        "true_defect_area_ratio": gt_area_ratio,
        "true_label": float(gt_label),
    }
    return out, gt


def add_sem_noise(image: np.ndarray, photon_count: float, elec_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Apply SEM-like shot noise + electronic noise."""
    lam = np.clip(image, 0.0, 1.0) * photon_count
    shot = rng.poisson(lam=lam) / photon_count
    elec = rng.normal(0.0, elec_sigma, size=image.shape)
    noisy = np.clip(shot + elec, 0.0, 1.0)
    return noisy


def sobel_edge_density_torch(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute edge magnitude by torch conv2d and return density above q90."""
    x = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kx = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]])
    ky = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]])
    gx = torch.nn.functional.conv2d(x, kx, padding=1)
    gy = torch.nn.functional.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-12).squeeze().cpu().numpy()
    thr = float(mag.mean() + mag.std())
    density = float((mag > thr).mean())
    return mag, density


def segment_dark_defects_kmeans(image: np.ndarray, random_state: int) -> np.ndarray:
    """KMeans(2) segmentation on residual dark map + morphology cleanup."""
    # Strong blur approximates slow-varying background; defects are darker residuals.
    background = gaussian_filter(image, sigma=2.2)
    residual = image - background
    flat = residual.reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=8, random_state=random_state)
    labels = km.fit_predict(flat).reshape(image.shape)
    centers = km.cluster_centers_.reshape(-1)
    defect_cluster = int(np.argmin(centers))
    mask = labels == defect_cluster
    # Guardrail: require relatively dark intensity in original image too.
    mask &= image < np.quantile(image, 0.45)
    mask = binary_closing(mask, structure=np.ones((3, 3)))
    mask = binary_opening(mask, structure=np.ones((3, 3)))
    return mask


def component_stats(mask: np.ndarray, intensity: np.ndarray, min_region_area: int) -> dict[str, float]:
    """Extract connected-component statistics from defect mask."""
    labeled, n = cc_label(mask)
    areas: list[int] = []
    perims: list[int] = []
    mean_ints: list[float] = []
    for idx in range(1, n + 1):
        region = labeled == idx
        area = int(region.sum())
        if area < min_region_area:
            continue
        boundary = region & (~binary_erosion(region))
        areas.append(area)
        perims.append(int(boundary.sum()))
        mean_ints.append(float(intensity[region].mean()))

    if not areas:
        return {
            "region_count": 0.0,
            "defect_area_ratio": 0.0,
            "max_region_area": 0.0,
            "mean_region_area": 0.0,
            "mean_region_perimeter": 0.0,
            "mean_region_intensity": float(intensity.mean()),
        }

    total_pixels = float(mask.size)
    return {
        "region_count": float(len(areas)),
        "defect_area_ratio": float(sum(areas) / total_pixels),
        "max_region_area": float(max(areas)),
        "mean_region_area": float(np.mean(areas)),
        "mean_region_perimeter": float(np.mean(perims)),
        "mean_region_intensity": float(np.mean(mean_ints)),
    }


def build_dataset(cfg: SEMConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, float]] = []
    for i in range(cfg.n_images):
        base = make_grain_microstructure(cfg.image_size, cfg.grain_count, rng)
        with_defect, gt = inject_defects(base, rng)
        noisy = add_sem_noise(with_defect, cfg.photon_count, cfg.elec_noise_sigma, rng)
        denoised = gaussian_filter(noisy, sigma=1.0)
        edge_mag, edge_density = sobel_edge_density_torch(denoised)
        defect_mask = segment_dark_defects_kmeans(denoised, random_state=cfg.seed + i)
        stats = component_stats(defect_mask, denoised, min_region_area=cfg.min_region_area)

        row = {
            "image_id": float(i),
            "mean_intensity": float(denoised.mean()),
            "std_intensity": float(denoised.std()),
            "edge_density": edge_density,
            "edge_mean": float(edge_mag.mean()),
        }
        row.update(stats)
        row.update(gt)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["label"] = df["true_label"].astype(int)
    return df


def train_sklearn_baseline(df: pd.DataFrame) -> tuple[float, str]:
    features = [
        "defect_area_ratio",
        "region_count",
        "max_region_area",
        "mean_region_perimeter",
        "edge_density",
        "std_intensity",
    ]
    X = df[features].to_numpy()
    y = df["label"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=7, stratify=y
    )
    clf = LogisticRegression(max_iter=500, random_state=7)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, digits=3)
    return acc, report


def train_torch_baseline(df: pd.DataFrame) -> float:
    features = [
        "defect_area_ratio",
        "region_count",
        "max_region_area",
        "mean_region_perimeter",
        "edge_density",
        "std_intensity",
    ]
    X = df[features].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=13, stratify=y
    )

    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    model = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.04)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(120):
        logits = model(xtr)
        loss = loss_fn(logits, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        prob = torch.sigmoid(model(xte))
        pred = (prob >= 0.5).float()
    acc = float((pred.eq(yte)).float().mean().item())
    return acc


def main() -> None:
    cfg = SEMConfig()
    df = build_dataset(cfg)

    sklearn_acc, sklearn_report = train_sklearn_baseline(df)
    torch_acc = train_torch_baseline(df)

    print("=== SEM MVP: synthetic dataset overview ===")
    print(df[["image_id", "label", "true_defect_area_ratio", "defect_area_ratio", "region_count"]].head(8))
    print()
    print("Label distribution:")
    print(df["label"].value_counts().sort_index())
    print()
    print("Feature means by label:")
    print(
        df.groupby("label")[["defect_area_ratio", "region_count", "edge_density", "std_intensity"]]
        .mean()
        .round(4)
    )
    print()
    print(f"Sklearn LogisticRegression accuracy: {sklearn_acc:.3f}")
    print("Sklearn classification report:")
    print(sklearn_report)
    print(f"PyTorch MLP accuracy: {torch_acc:.3f}")


if __name__ == "__main__":
    main()
