"""Cloud Chamber MVP: synthetic droplet data -> track clustering -> model fitting.

This demo intentionally stays small and explicit while using a practical scientific stack:
- numpy: numerical simulation and geometry
- scipy: nonlinear least-squares circle fitting
- pandas: tabular outputs
- scikit-learn: clustering (DBSCAN) and scaling
- PyTorch: robust refinement of circle parameters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


@dataclass
class TrackEstimate:
    cluster_id: int
    n_points: int
    chosen_model: str
    line_bic: float
    circle_bic: float
    radius_mm: Optional[float]
    momentum_gev_c: Optional[float]


def simulate_cloud_chamber(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic cloud chamber droplet points in millimeters."""
    rng = np.random.default_rng(seed)

    # Track A: near-straight high-momentum particle (muon-like)
    x1 = np.linspace(-55, 55, 170)
    y1 = 0.45 * x1 + 8.0 + rng.normal(0.0, 0.9, size=x1.shape)

    # Track B: positively curved trajectory
    theta2 = np.linspace(-2.0, -0.5, 190)
    c2 = np.array([18.0, -12.0])
    r2 = 36.0
    x2 = c2[0] + r2 * np.cos(theta2) + rng.normal(0.0, 0.8, size=theta2.shape)
    y2 = c2[1] + r2 * np.sin(theta2) + rng.normal(0.0, 0.8, size=theta2.shape)

    # Track C: tighter curvature (lower momentum)
    theta3 = np.linspace(2.1, 3.5, 150)
    c3 = np.array([-20.0, 10.0])
    r3 = 22.0
    x3 = c3[0] + r3 * np.cos(theta3) + rng.normal(0.0, 0.7, size=theta3.shape)
    y3 = c3[1] + r3 * np.sin(theta3) + rng.normal(0.0, 0.7, size=theta3.shape)

    # Spurious droplets / background condensation noise
    n_noise = 120
    xn = rng.uniform(-65, 65, size=n_noise)
    yn = rng.uniform(-65, 65, size=n_noise)

    x = np.concatenate([x1, x2, x3, xn])
    y = np.concatenate([y1, y2, y3, yn])
    label = (
        ["track_A"] * len(x1)
        + ["track_B"] * len(x2)
        + ["track_C"] * len(x3)
        + ["noise"] * len(xn)
    )

    return pd.DataFrame({"x_mm": x, "y_mm": y, "true_label": label})


def fit_line_pca(points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Orthogonal line fit via PCA; returns (mse, center, direction)."""
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]

    normal = np.array([-direction[1], direction[0]])
    orth_dist = np.abs(centered @ normal)
    mse = float(np.mean(orth_dist**2))
    return mse, center, direction


def fit_circle_scipy(points: np.ndarray) -> tuple[float, float, float, float]:
    """Circle fit with robust least-squares in scipy."""
    x = points[:, 0]
    y = points[:, 1]

    cx0, cy0 = points.mean(axis=0)
    r0 = np.mean(np.sqrt((x - cx0) ** 2 + (y - cy0) ** 2))

    def residuals(params: np.ndarray) -> np.ndarray:
        cx, cy, r = params
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return d - r

    result = least_squares(
        residuals,
        x0=np.array([cx0, cy0, max(r0, 1e-3)]),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
    )

    cx, cy, r = result.x
    r = abs(float(r))
    resid = residuals(np.array([cx, cy, r]))
    mse = float(np.mean(resid**2))
    return mse, float(cx), float(cy), r


def _inverse_softplus(y: float) -> float:
    # Stable inverse of softplus for y > 0.
    y = max(float(y), 1e-6)
    if y > 20.0:
        return y
    return float(np.log(np.expm1(y)))


def _softplus_np(x: float) -> float:
    # Stable numpy softplus: log(1 + exp(x))
    return float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))


def refine_circle_torch(
    points: np.ndarray,
    cx0: float,
    cy0: float,
    r0: float,
    steps: int = 240,
    lr: float = 0.06,
) -> tuple[float, float, float, float]:
    """Refine circle params with torch + smooth L1 loss (robust to outliers)."""
    pts = torch.tensor(points, dtype=torch.float32)

    params = torch.nn.Parameter(
        torch.tensor([cx0, cy0, _inverse_softplus(max(r0, 1e-3))], dtype=torch.float32)
    )
    opt = torch.optim.Adam([params], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        cx, cy, raw_r = params[0], params[1], params[2]
        r = torch.nn.functional.softplus(raw_r) + 1e-6
        d = torch.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2 + 1e-8)
        resid = d - r
        loss = torch.nn.functional.smooth_l1_loss(resid, torch.zeros_like(resid))
        loss.backward()
        opt.step()

    cx, cy, raw_r = params.detach().cpu().numpy()
    r = _softplus_np(float(raw_r)) + 1e-6
    d = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    mse = float(np.mean((d - r) ** 2))
    return float(cx), float(cy), r, mse


def bic(n: int, mse: float, k: int) -> float:
    safe_mse = max(mse, 1e-12)
    return float(n * np.log(safe_mse) + k * np.log(max(n, 2)))


def estimate_momentum(radius_mm: float, b_tesla: float = 0.8) -> float:
    """p ~= 0.3 * B * r (GeV/c), with r in meters."""
    radius_m = radius_mm / 1000.0
    return 0.3 * b_tesla * radius_m


def analyze_tracks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = df[["x_mm", "y_mm"]].to_numpy()

    scaler = StandardScaler()
    points_z = scaler.fit_transform(points)

    # Density-based clustering in standardized coordinate space
    labels = DBSCAN(eps=0.11, min_samples=10).fit_predict(points_z)
    df_labeled = df.copy()
    df_labeled["cluster_id"] = labels

    estimates: list[TrackEstimate] = []
    for cluster_id in sorted(c for c in np.unique(labels) if c >= 0):
        cluster_pts = points[labels == cluster_id]
        n = len(cluster_pts)
        if n < 25:
            continue

        line_mse, _, _ = fit_line_pca(cluster_pts)
        circle_mse_s, cx, cy, r = fit_circle_scipy(cluster_pts)
        cx_t, cy_t, r_t, circle_mse = refine_circle_torch(cluster_pts, cx, cy, r)

        line_b = bic(n=n, mse=line_mse, k=2)
        circle_b = bic(n=n, mse=circle_mse, k=3)

        # Guardrail: huge-radius circles are physically close to straight tracks.
        if r_t > 300.0:
            chosen = "line"
        else:
            chosen = "circle" if circle_b < line_b else "line"

        radius_mm: Optional[float]
        momentum: Optional[float]
        if chosen == "circle":
            radius_mm = r_t
            momentum = estimate_momentum(radius_mm)
        else:
            radius_mm = None
            momentum = None

        _ = (cx_t, cy_t)  # kept for readability/traceability
        estimates.append(
            TrackEstimate(
                cluster_id=int(cluster_id),
                n_points=int(n),
                chosen_model=chosen,
                line_bic=line_b,
                circle_bic=circle_b,
                radius_mm=radius_mm,
                momentum_gev_c=momentum,
            )
        )

    result_df = pd.DataFrame([e.__dict__ for e in estimates])
    if not result_df.empty:
        result_df = result_df.sort_values(by=["cluster_id"]).reset_index(drop=True)
    return df_labeled, result_df


def main() -> None:
    np.set_printoptions(precision=3, suppress=True)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    df = simulate_cloud_chamber(seed=42)
    labeled_df, summary = analyze_tracks(df)

    print("=== Cloud Chamber MVP ===")
    print(f"Total droplets: {len(df)}")
    print(f"Detected clusters (excluding noise): {len(summary)}")

    cluster_sizes = (
        labeled_df[labeled_df["cluster_id"] >= 0]
        .groupby("cluster_id")
        .size()
        .rename("count")
        .reset_index()
    )
    print("\nCluster sizes:")
    if cluster_sizes.empty:
        print("No track-like cluster found.")
    else:
        print(cluster_sizes.to_string(index=False))

    print("\nTrack estimates:")
    if summary.empty:
        print("No valid track estimate.")
    else:
        show = summary.copy()
        show["line_bic"] = show["line_bic"].map(lambda v: f"{v:.2f}")
        show["circle_bic"] = show["circle_bic"].map(lambda v: f"{v:.2f}")
        show["radius_mm"] = show["radius_mm"].map(lambda v: None if pd.isna(v) else round(float(v), 3))
        show["momentum_gev_c"] = show["momentum_gev_c"].map(
            lambda v: None if pd.isna(v) else round(float(v), 5)
        )
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
