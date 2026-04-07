"""Vertex detector MVP: simulate tracks and reconstruct the primary vertex."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.linalg import lstsq
from sklearn.metrics import mean_squared_error


@dataclass
class Track:
    """Single track represented by noisy hit points on detector layers."""

    hits: np.ndarray  # shape: (n_hits, 2)
    label: str


@dataclass
class FittedTrack:
    """TLS-fitted line parameters for one track."""

    point: np.ndarray  # one point on line, shape: (2,)
    direction: np.ndarray  # unit vector, shape: (2,)
    label: str


def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= 1e-12:
        raise ValueError("zero-length vector encountered")
    return v / norm


def fit_track_tls(hits: np.ndarray, label: str) -> FittedTrack:
    """Fit a 2D line using total least squares (SVD/PCA form)."""
    if hits.ndim != 2 or hits.shape[1] != 2 or hits.shape[0] < 2:
        raise ValueError("hits must have shape (n>=2, 2)")

    centroid = hits.mean(axis=0)
    centered = hits - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = unit(vh[0])
    return FittedTrack(point=centroid, direction=direction, label=label)


def simulate_event(
    rng: np.random.Generator,
    n_primary: int = 12,
    n_outlier: int = 4,
    layer_radii: tuple[float, ...] = (12.0, 24.0, 36.0, 48.0),
    sigma_hit: float = 0.015,
) -> tuple[np.ndarray, list[Track]]:
    """Generate one synthetic event with primary and outlier tracks."""
    true_vertex = np.array([0.08, -0.05], dtype=float)
    secondary_vertex = np.array([1.4, -1.2], dtype=float)
    tracks: list[Track] = []

    for i in range(n_primary):
        phi = rng.uniform(0.0, 2.0 * np.pi)
        direction = np.array([np.cos(phi), np.sin(phi)], dtype=float)
        hits = []
        for r in layer_radii:
            ideal = true_vertex + r * direction
            noisy = ideal + rng.normal(0.0, sigma_hit, size=2)
            hits.append(noisy)
        tracks.append(Track(hits=np.asarray(hits), label=f"primary_{i:02d}"))

    for i in range(n_outlier):
        phi = rng.uniform(0.0, 2.0 * np.pi)
        direction = np.array([np.cos(phi), np.sin(phi)], dtype=float)
        hits = []
        for r in layer_radii:
            ideal = secondary_vertex + r * direction
            noisy = ideal + rng.normal(0.0, sigma_hit * 1.4, size=2)
            hits.append(noisy)
        tracks.append(Track(hits=np.asarray(hits), label=f"outlier_{i:02d}"))

    return true_vertex, tracks


def build_constraints(
    fitted_tracks: list[FittedTrack],
    sigma_track: float = 0.030,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Create per-track normal-form constraints n^T v = n^T p."""
    normals = []
    points = []
    sigmas = []
    labels = []

    for trk in fitted_tracks:
        u = unit(trk.direction)
        n = np.array([-u[1], u[0]], dtype=float)
        normals.append(n)
        points.append(trk.point)
        sigmas.append(sigma_track)
        labels.append(trk.label)

    return np.asarray(normals), np.asarray(points), np.asarray(sigmas), labels


def solve_vertex_wls(
    normals: np.ndarray,
    points: np.ndarray,
    sigmas: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve weighted least squares for vertex in 2D."""
    n_tracks = normals.shape[0]
    if weights is None:
        weights = np.ones(n_tracks, dtype=float)

    inv_sigma = 1.0 / np.clip(sigmas, 1e-12, None)
    A = normals * inv_sigma[:, None]
    b = np.sum(normals * points, axis=1) * inv_sigma

    w_sqrt = np.sqrt(np.clip(weights, 1e-12, None))
    Aw = A * w_sqrt[:, None]
    bw = b * w_sqrt

    v_hat, _, _, _ = lstsq(Aw, bw)

    hessian = Aw.T @ Aw
    cov = np.linalg.pinv(hessian)
    residuals = np.sum(normals * (v_hat[None, :] - points), axis=1)
    return v_hat, residuals, cov


def robust_vertex_irls(
    normals: np.ndarray,
    points: np.ndarray,
    sigmas: np.ndarray,
    iters: int = 8,
    cauchy_c: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """IRLS with Cauchy weights for outlier-robust vertex fitting."""
    weights = np.ones(normals.shape[0], dtype=float)
    vertex, residuals, cov = solve_vertex_wls(normals, points, sigmas, weights)

    for _ in range(iters):
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = max(1e-6, 1.4826 * mad)
        z = residuals / (cauchy_c * scale)
        weights = 1.0 / (1.0 + z**2)
        vertex, residuals, cov = solve_vertex_wls(normals, points, sigmas, weights)

    return vertex, residuals, cov, weights


def torch_refine_vertex(
    init_vertex: np.ndarray,
    normals: np.ndarray,
    points: np.ndarray,
    steps: int = 120,
    lr: float = 0.08,
    delta: float = 0.04,
) -> np.ndarray:
    """Refine the vertex with pseudo-Huber loss in PyTorch."""
    v = torch.tensor(init_vertex, dtype=torch.float64, requires_grad=True)
    n = torch.tensor(normals, dtype=torch.float64)
    p = torch.tensor(points, dtype=torch.float64)

    optimizer = torch.optim.Adam([v], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        residuals = torch.sum(n * (v.unsqueeze(0) - p), dim=1)
        loss = torch.sum(delta**2 * (torch.sqrt(1.0 + (residuals / delta) ** 2) - 1.0))
        loss.backward()
        optimizer.step()

    return v.detach().cpu().numpy()


def evaluate_solution(
    name: str,
    estimate: np.ndarray,
    true_vertex: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, float]:
    """Return compact metrics for one estimator."""
    distance = float(np.linalg.norm(estimate - true_vertex))
    rmse = float(np.sqrt(mean_squared_error(np.zeros_like(residuals), residuals)))
    return {
        "method": name,
        "x": float(estimate[0]),
        "y": float(estimate[1]),
        "error_to_truth": distance,
        "residual_rmse": rmse,
    }


def main() -> None:
    rng = np.random.default_rng(42)

    true_vertex, tracks = simulate_event(rng)
    fitted_tracks = [fit_track_tls(t.hits, t.label) for t in tracks]
    normals, points, sigmas, labels = build_constraints(fitted_tracks)

    v_ols, r_ols, cov_ols = solve_vertex_wls(normals, points, sigmas)
    v_irls, r_irls, cov_irls, weights = robust_vertex_irls(normals, points, sigmas)
    v_torch = torch_refine_vertex(v_irls, normals, points)
    r_torch = np.sum(normals * (v_torch[None, :] - points), axis=1)

    summary = pd.DataFrame(
        [
            evaluate_solution("OLS", v_ols, true_vertex, r_ols),
            evaluate_solution("IRLS", v_irls, true_vertex, r_irls),
            evaluate_solution("IRLS+Torch", v_torch, true_vertex, r_torch),
        ]
    )

    track_table = pd.DataFrame(
        {
            "track": labels,
            "residual_ols": r_ols,
            "residual_irls": r_irls,
            "weight_irls": weights,
        }
    ).sort_values("weight_irls", ascending=True)

    np.set_printoptions(precision=6, suppress=True)
    print("=== Vertex Detector MVP ===")
    print(f"True vertex        : {true_vertex}")
    print(f"OLS vertex         : {v_ols}")
    print(f"IRLS vertex        : {v_irls}")
    print(f"IRLS+Torch vertex  : {v_torch}")
    print("\nSummary metrics:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    print("\nTrack residuals and robust weights (small weight => likely outlier):")
    print(track_table.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    print("\nApprox covariance (OLS):")
    print(cov_ols)
    print("Approx covariance (IRLS final weighted):")
    print(cov_irls)


if __name__ == "__main__":
    main()
