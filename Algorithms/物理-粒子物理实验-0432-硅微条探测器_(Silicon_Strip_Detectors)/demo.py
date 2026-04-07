"""Silicon strip detector MVP.

This demo simulates straight charged-particle tracks crossing several silicon-strip
layers, builds strip clusters with threshold logic, and reconstructs tracks with a
robust linear model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_squared_error


@dataclass(frozen=True)
class DetectorConfig:
    seed: int = 432
    n_events: int = 400
    z_layers_mm: tuple[float, ...] = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
    n_strips: int = 1024
    pitch_mm: float = 0.08
    charge_sigma_mm: float = 0.05
    mip_charge_e: float = 24000.0
    noise_sigma_e: float = 900.0
    threshold_e: float = 3200.0
    x0_range_mm: tuple[float, float] = (-12.0, 12.0)
    slope_sigma: float = 0.018
    outlier_prob: float = 0.03
    outlier_sigma_mm: float = 1.0
    min_hits_for_track: int = 3


def strip_centers_mm(cfg: DetectorConfig) -> np.ndarray:
    indices = np.arange(cfg.n_strips)
    center_offset = (cfg.n_strips - 1) / 2.0
    return (indices - center_offset) * cfg.pitch_mm


def find_contiguous_runs(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    boundaries = np.where(np.diff(indices) > 1)[0] + 1
    return np.split(indices, boundaries)


def simulate_layer_hit(
    true_x_mm: float,
    cfg: DetectorConfig,
    centers_mm: np.ndarray,
    rng: np.random.Generator,
) -> float | None:
    half_width = 0.5 * cfg.n_strips * cfg.pitch_mm
    if true_x_mm < -half_width or true_x_mm > half_width:
        return None

    window_mask = np.abs(centers_mm - true_x_mm) <= 3.0 * cfg.pitch_mm
    window_indices = np.where(window_mask)[0]
    if window_indices.size == 0:
        return None

    local_centers = centers_mm[window_indices]
    profile = norm.pdf(local_centers, loc=true_x_mm, scale=cfg.charge_sigma_mm)
    if not np.any(profile > 0):
        return None

    expected_charge = cfg.mip_charge_e * profile / profile.sum()
    measured_charge = expected_charge + rng.normal(
        loc=0.0,
        scale=cfg.noise_sigma_e,
        size=expected_charge.shape,
    )

    above_threshold_local = np.where(measured_charge > cfg.threshold_e)[0]
    if above_threshold_local.size == 0:
        return None

    clusters_local = find_contiguous_runs(above_threshold_local)
    best_cluster = max(clusters_local, key=lambda c: measured_charge[c].sum())
    q = measured_charge[best_cluster]
    x = local_centers[best_cluster]
    return float(np.average(x, weights=q))


def fit_track_ransac(
    z_hits_mm: np.ndarray,
    x_hits_mm: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float] | None:
    if z_hits_mm.size < 3:
        return None

    model = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=2,
        residual_threshold=0.20,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )

    try:
        model.fit(z_hits_mm.reshape(-1, 1), x_hits_mm)
    except ValueError:
        return None

    estimator = model.estimator_
    slope = float(estimator.coef_[0])
    intercept = float(estimator.intercept_)
    return intercept, slope


def run_mvp(cfg: DetectorConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)
    centers_mm = strip_centers_mm(cfg)
    z_layers = np.asarray(cfg.z_layers_mm, dtype=float)

    event_rows: list[dict[str, float | int | bool]] = []
    position_errors_um: list[float] = []
    fit_residuals_um: list[float] = []
    fit_truth_x0: list[float] = []
    fit_pred_x0: list[float] = []
    fit_truth_slope: list[float] = []
    fit_pred_slope: list[float] = []

    total_possible_hits = cfg.n_events * z_layers.size
    total_detected_hits = 0
    reconstructed_tracks = 0

    for event_id in range(cfg.n_events):
        true_x0 = float(rng.uniform(*cfg.x0_range_mm))
        true_slope = float(rng.normal(loc=0.0, scale=cfg.slope_sigma))
        true_positions = true_x0 + true_slope * z_layers

        z_hits: list[float] = []
        x_hits: list[float] = []
        for z_mm, x_true_mm in zip(z_layers, true_positions):
            measured_x = simulate_layer_hit(x_true_mm, cfg, centers_mm, rng)
            if measured_x is None:
                continue

            if rng.random() < cfg.outlier_prob:
                measured_x += float(rng.normal(loc=0.0, scale=cfg.outlier_sigma_mm))

            z_hits.append(float(z_mm))
            x_hits.append(measured_x)
            total_detected_hits += 1
            position_errors_um.append((measured_x - x_true_mm) * 1e3)

        z_arr = np.asarray(z_hits, dtype=float)
        x_arr = np.asarray(x_hits, dtype=float)
        fit = fit_track_ransac(z_arr, x_arr, rng)

        reconstructed = fit is not None and len(z_hits) >= cfg.min_hits_for_track
        fit_x0 = np.nan
        fit_slope = np.nan

        if reconstructed:
            reconstructed_tracks += 1
            fit_x0, fit_slope = fit
            fit_truth_x0.append(true_x0)
            fit_pred_x0.append(fit_x0)
            fit_truth_slope.append(true_slope)
            fit_pred_slope.append(fit_slope)

            residuals = x_arr - (fit_x0 + fit_slope * z_arr)
            fit_residuals_um.extend((residuals * 1e3).tolist())

        event_rows.append(
            {
                "event_id": event_id,
                "true_x0_mm": true_x0,
                "true_slope": true_slope,
                "n_hits": len(z_hits),
                "reconstructed": reconstructed,
                "fit_x0_mm": fit_x0,
                "fit_slope": fit_slope,
                "x0_error_um": (fit_x0 - true_x0) * 1e3 if reconstructed else np.nan,
                "slope_error_mrad": (fit_slope - true_slope) * 1e3 if reconstructed else np.nan,
            }
        )

    events_df = pd.DataFrame(event_rows)

    x0_rmse_um = (
        np.sqrt(mean_squared_error(fit_truth_x0, fit_pred_x0)) * 1e3
        if fit_truth_x0
        else float("nan")
    )
    slope_rmse_mrad = (
        np.sqrt(mean_squared_error(fit_truth_slope, fit_pred_slope)) * 1e3
        if fit_truth_slope
        else float("nan")
    )

    metrics = {
        "events": float(cfg.n_events),
        "layers": float(z_layers.size),
        "hit_efficiency": total_detected_hits / total_possible_hits,
        "track_reco_efficiency": reconstructed_tracks / cfg.n_events,
        "position_resolution_um": float(np.std(position_errors_um, ddof=1)),
        "fit_residual_rms_um": float(np.std(fit_residuals_um, ddof=1)),
        "x0_rmse_um": float(x0_rmse_um),
        "slope_rmse_mrad": float(slope_rmse_mrad),
    }
    return events_df, metrics


def main() -> None:
    cfg = DetectorConfig()
    events_df, metrics = run_mvp(cfg)

    print("=== Silicon Strip Detector MVP ===")
    print(f"events={int(metrics['events'])}, layers={int(metrics['layers'])}")
    print(f"hit_efficiency={metrics['hit_efficiency']:.4f}")
    print(f"track_reco_efficiency={metrics['track_reco_efficiency']:.4f}")
    print(f"position_resolution_um={metrics['position_resolution_um']:.2f}")
    print(f"fit_residual_rms_um={metrics['fit_residual_rms_um']:.2f}")
    print(f"x0_rmse_um={metrics['x0_rmse_um']:.2f}")
    print(f"slope_rmse_mrad={metrics['slope_rmse_mrad']:.3f}")

    preview_cols = [
        "event_id",
        "n_hits",
        "reconstructed",
        "true_x0_mm",
        "fit_x0_mm",
        "x0_error_um",
    ]
    print("\nEvent preview (first 8 rows):")
    print(events_df[preview_cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
