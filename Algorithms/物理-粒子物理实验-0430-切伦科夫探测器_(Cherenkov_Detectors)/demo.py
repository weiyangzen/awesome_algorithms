"""Minimal MVP for Cherenkov detectors (ring reconstruction + PID).

This script simulates a compact RICH-like detector workflow:
1. Generate mixed-species events (pi/K/p) at fixed momentum.
2. Simulate Cherenkov photons on a ring plus sensor background hits.
3. Reconstruct ring parameters with robust circle fitting.
4. Invert ring radius to beta and mass, then perform simple PID.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


MASS_GEV = {
    "pion": 0.13957,
    "kaon": 0.49367,
    "proton": 0.93827,
}


@dataclass(frozen=True)
class CherenkovDetectorConfig:
    seed: int = 430
    n_events: int = 420

    # Beam / particle setup
    momentum_gev: float = 3.0
    species: tuple[str, ...] = ("pion", "kaon", "proton")
    species_probs: tuple[float, ...] = (0.45, 0.35, 0.20)

    # Radiator + geometry
    refractive_index: float = 1.03
    mirror_to_sensor_mm: float = 1200.0

    # Photon statistics and detector response
    n0_photons: float = 720.0
    sigma_radial_mm: float = 5.5
    ring_center_sigma_mm: float = 1.5
    background_mean_hits: float = 6.0
    sensor_half_size_mm: float = 260.0

    # Ring-fit quality cuts
    min_points_for_fit: int = 8
    min_inliers: int = 10
    inlier_residual_mm: float = 8.5
    min_inlier_fraction: float = 0.30
    robust_f_scale_mm: float = 7.5

    # PID likelihood width floor
    pid_sigma_floor_mm: float = 2.2


def beta_from_p_mass(momentum_gev: float, mass_gev: float) -> float:
    return float(momentum_gev / np.sqrt(momentum_gev**2 + mass_gev**2))


def theta_cherenkov_rad(beta: float, refractive_index: float) -> float:
    x = beta * refractive_index
    if x <= 1.0:
        return 0.0
    arg = np.clip(1.0 / x, -1.0, 1.0)
    return float(np.arccos(arg))


def expected_radius_mm(theta_rad: float, path_mm: float) -> float:
    return float(path_mm * np.tan(theta_rad))


def simulate_event(
    species: str,
    cfg: CherenkovDetectorConfig,
    rng: np.random.Generator,
) -> dict[str, float | int | str | np.ndarray | bool]:
    mass = MASS_GEV[species]
    beta_true = beta_from_p_mass(cfg.momentum_gev, mass)
    theta_true = theta_cherenkov_rad(beta_true, cfg.refractive_index)
    emits = theta_true > 0.0

    cx_true = float(rng.normal(0.0, cfg.ring_center_sigma_mm))
    cy_true = float(rng.normal(0.0, cfg.ring_center_sigma_mm))
    radius_true = expected_radius_mm(theta_true, cfg.mirror_to_sensor_mm)

    mean_signal = cfg.n0_photons * (np.sin(theta_true) ** 2)
    n_signal = int(rng.poisson(mean_signal)) if emits else 0

    if n_signal > 0:
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n_signal)
        radial = radius_true + rng.normal(0.0, cfg.sigma_radial_mm, size=n_signal)
        xs_sig = cx_true + radial * np.cos(phi)
        ys_sig = cy_true + radial * np.sin(phi)
    else:
        xs_sig = np.empty(0, dtype=float)
        ys_sig = np.empty(0, dtype=float)

    n_bg = int(rng.poisson(cfg.background_mean_hits))
    xs_bg = rng.uniform(-cfg.sensor_half_size_mm, cfg.sensor_half_size_mm, size=n_bg)
    ys_bg = rng.uniform(-cfg.sensor_half_size_mm, cfg.sensor_half_size_mm, size=n_bg)

    xs = np.concatenate([xs_sig, xs_bg])
    ys = np.concatenate([ys_sig, ys_bg])

    if xs.size > 1:
        order = rng.permutation(xs.size)
        xs = xs[order]
        ys = ys[order]

    return {
        "species": species,
        "mass_true_gev": mass,
        "beta_true": beta_true,
        "theta_true_rad": theta_true,
        "radius_true_mm": radius_true,
        "emits_true": emits,
        "cx_true": cx_true,
        "cy_true": cy_true,
        "n_signal": n_signal,
        "n_background": n_bg,
        "x_hits": xs,
        "y_hits": ys,
    }


def fit_circle_robust(
    x_hits: np.ndarray,
    y_hits: np.ndarray,
    cfg: CherenkovDetectorConfig,
) -> dict[str, float | np.ndarray | bool] | None:
    if x_hits.size < 3:
        return None

    cx0 = float(np.mean(x_hits))
    cy0 = float(np.mean(y_hits))
    r0 = float(np.median(np.sqrt((x_hits - cx0) ** 2 + (y_hits - cy0) ** 2)))
    r0 = max(r0, 1.0)

    def residuals(params: np.ndarray) -> np.ndarray:
        cx, cy, radius = params
        dist = np.sqrt((x_hits - cx) ** 2 + (y_hits - cy) ** 2)
        return dist - radius

    bounds = (
        np.array([-cfg.sensor_half_size_mm, -cfg.sensor_half_size_mm, 0.0], dtype=float),
        np.array([cfg.sensor_half_size_mm, cfg.sensor_half_size_mm, 1.5 * cfg.sensor_half_size_mm], dtype=float),
    )

    result = least_squares(
        residuals,
        x0=np.array([cx0, cy0, r0], dtype=float),
        loss="soft_l1",
        f_scale=cfg.robust_f_scale_mm,
        bounds=bounds,
        max_nfev=800,
    )

    if not result.success:
        return None

    cx, cy, radius = result.x
    res = residuals(result.x)
    return {
        "cx": float(cx),
        "cy": float(cy),
        "radius_mm": float(radius),
        "residuals": res,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
    }


def classify_species(
    has_ring: bool,
    radius_rec_mm: float,
    n_inliers: int,
    cfg: CherenkovDetectorConfig,
) -> tuple[str, dict[str, float]]:
    score: dict[str, float] = {}

    if not has_ring:
        for sp in cfg.species:
            theta = theta_cherenkov_rad(
                beta_from_p_mass(cfg.momentum_gev, MASS_GEV[sp]),
                cfg.refractive_index,
            )
            score[sp] = 0.0 if theta == 0.0 else 30.0
        best = min(score, key=score.get)
        return best, score

    sigma_eff = max(cfg.sigma_radial_mm / np.sqrt(max(n_inliers, 1)), cfg.pid_sigma_floor_mm)

    for sp in cfg.species:
        beta_h = beta_from_p_mass(cfg.momentum_gev, MASS_GEV[sp])
        theta_h = theta_cherenkov_rad(beta_h, cfg.refractive_index)
        if theta_h <= 0.0:
            score[sp] = 1e6
            continue
        radius_h = expected_radius_mm(theta_h, cfg.mirror_to_sensor_mm)
        score[sp] = ((radius_rec_mm - radius_h) / sigma_eff) ** 2

    best = min(score, key=score.get)
    return best, score


def reconstruct_event(
    event: dict[str, float | int | str | np.ndarray | bool],
    cfg: CherenkovDetectorConfig,
) -> dict[str, float | int | bool | str]:
    x_hits = event["x_hits"]
    y_hits = event["y_hits"]
    if not isinstance(x_hits, np.ndarray) or not isinstance(y_hits, np.ndarray):
        raise TypeError("x_hits/y_hits must be numpy arrays")

    total_hits = int(x_hits.size)
    if total_hits < cfg.min_points_for_fit:
        pred, score = classify_species(False, np.nan, 0, cfg)
        return {
            "total_hits": total_hits,
            "fit_success": False,
            "has_ring": False,
            "n_inliers": 0,
            "inlier_fraction": 0.0,
            "residual_rms_mm": np.nan,
            "radius_rec_mm": np.nan,
            "theta_rec_rad": np.nan,
            "beta_rec": np.nan,
            "mass_rec_gev": np.nan,
            "pred_species": pred,
            "pid_score_best": float(score[pred]),
        }

    fit = fit_circle_robust(x_hits, y_hits, cfg)
    if fit is None:
        pred, score = classify_species(False, np.nan, 0, cfg)
        return {
            "total_hits": total_hits,
            "fit_success": False,
            "has_ring": False,
            "n_inliers": 0,
            "inlier_fraction": 0.0,
            "residual_rms_mm": np.nan,
            "radius_rec_mm": np.nan,
            "theta_rec_rad": np.nan,
            "beta_rec": np.nan,
            "mass_rec_gev": np.nan,
            "pred_species": pred,
            "pid_score_best": float(score[pred]),
        }

    residuals = fit["residuals"]
    if not isinstance(residuals, np.ndarray):
        raise TypeError("residuals must be numpy arrays")

    inlier_mask = np.abs(residuals) <= cfg.inlier_residual_mm
    n_inliers = int(np.sum(inlier_mask))
    inlier_fraction = n_inliers / max(total_hits, 1)
    residual_rms = float(np.sqrt(np.mean((residuals[inlier_mask]) ** 2))) if n_inliers > 0 else float("inf")

    has_ring = bool(
        n_inliers >= cfg.min_inliers
        and inlier_fraction >= cfg.min_inlier_fraction
        and residual_rms <= cfg.inlier_residual_mm
    )

    if has_ring:
        radius_rec = float(fit["radius_mm"])
        theta_rec = float(np.arctan2(radius_rec, cfg.mirror_to_sensor_mm))
        beta_rec = float(1.0 / (cfg.refractive_index * np.cos(theta_rec)))
        beta_rec = min(beta_rec, 0.999999)
        mass_rec = float(cfg.momentum_gev * np.sqrt(max(0.0, 1.0 / (beta_rec**2) - 1.0)))
    else:
        radius_rec = np.nan
        theta_rec = np.nan
        beta_rec = np.nan
        mass_rec = np.nan

    pred, score = classify_species(has_ring, radius_rec, n_inliers, cfg)
    return {
        "total_hits": total_hits,
        "fit_success": True,
        "has_ring": has_ring,
        "n_inliers": n_inliers,
        "inlier_fraction": float(inlier_fraction),
        "residual_rms_mm": float(residual_rms) if np.isfinite(residual_rms) else np.nan,
        "radius_rec_mm": radius_rec,
        "theta_rec_rad": theta_rec,
        "beta_rec": beta_rec,
        "mass_rec_gev": mass_rec,
        "pred_species": pred,
        "pid_score_best": float(score[pred]),
    }


def run_mvp(cfg: CherenkovDetectorConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)

    rows: list[dict[str, float | int | str | bool]] = []
    for event_id in range(cfg.n_events):
        species = str(rng.choice(cfg.species, p=cfg.species_probs))
        event = simulate_event(species, cfg, rng)
        reco = reconstruct_event(event, cfg)

        emits_true = bool(event["emits_true"])
        has_ring = bool(reco["has_ring"])

        theta_true = float(event["theta_true_rad"])
        radius_true = float(event["radius_true_mm"])
        beta_true = float(event["beta_true"])
        mass_true = float(event["mass_true_gev"])

        theta_rec = float(reco["theta_rec_rad"]) if np.isfinite(reco["theta_rec_rad"]) else np.nan
        radius_rec = float(reco["radius_rec_mm"]) if np.isfinite(reco["radius_rec_mm"]) else np.nan
        beta_rec = float(reco["beta_rec"]) if np.isfinite(reco["beta_rec"]) else np.nan
        mass_rec = float(reco["mass_rec_gev"]) if np.isfinite(reco["mass_rec_gev"]) else np.nan

        rows.append(
            {
                "event_id": event_id,
                "species_true": species,
                "species_pred": str(reco["pred_species"]),
                "pid_correct": bool(species == reco["pred_species"]),
                "emits_true": emits_true,
                "has_ring": has_ring,
                "n_signal": int(event["n_signal"]),
                "n_background": int(event["n_background"]),
                "n_hits_total": int(reco["total_hits"]),
                "n_inliers": int(reco["n_inliers"]),
                "inlier_fraction": float(reco["inlier_fraction"]),
                "residual_rms_mm": float(reco["residual_rms_mm"]),
                "theta_true_mrad": theta_true * 1e3,
                "theta_rec_mrad": theta_rec * 1e3 if np.isfinite(theta_rec) else np.nan,
                "theta_error_mrad": (theta_rec - theta_true) * 1e3 if has_ring else np.nan,
                "radius_true_mm": radius_true,
                "radius_rec_mm": radius_rec,
                "radius_error_mm": radius_rec - radius_true if has_ring else np.nan,
                "beta_true": beta_true,
                "beta_rec": beta_rec,
                "beta_error": beta_rec - beta_true if has_ring else np.nan,
                "mass_true_gev": mass_true,
                "mass_rec_gev": mass_rec,
                "mass_error_gev": mass_rec - mass_true if has_ring else np.nan,
            }
        )

    df = pd.DataFrame(rows)

    emits_mask = df["emits_true"] == True
    noemit_mask = df["emits_true"] == False
    ring_mask = df["has_ring"] == True
    emit_and_ring = emits_mask & ring_mask

    theta_err = df.loc[emit_and_ring, "theta_error_mrad"].to_numpy(dtype=float)
    radius_err = df.loc[emit_and_ring, "radius_error_mm"].to_numpy(dtype=float)
    beta_err = df.loc[emit_and_ring, "beta_error"].to_numpy(dtype=float)

    metrics = {
        "events": float(cfg.n_events),
        "emit_fraction_true": float(np.mean(df["emits_true"])),
        "ring_reco_efficiency_emit": float(np.mean(df.loc[emits_mask, "has_ring"])) if np.any(emits_mask) else float("nan"),
        "false_ring_rate_noemit": float(np.mean(df.loc[noemit_mask, "has_ring"])) if np.any(noemit_mask) else float("nan"),
        "pid_accuracy_overall": float(np.mean(df["pid_correct"])),
        "pid_accuracy_emit": float(np.mean(df.loc[emits_mask, "pid_correct"])) if np.any(emits_mask) else float("nan"),
        "theta_resolution_mrad": float(np.std(theta_err, ddof=1)) if theta_err.size > 1 else float("nan"),
        "radius_rmse_mm": float(np.sqrt(np.mean(radius_err**2))) if radius_err.size > 0 else float("nan"),
        "beta_rmse": float(np.sqrt(np.mean(beta_err**2))) if beta_err.size > 0 else float("nan"),
    }

    return df, metrics


def main() -> None:
    cfg = CherenkovDetectorConfig()
    events_df, metrics = run_mvp(cfg)

    print("=== Cherenkov Detector MVP (Ring Fit + PID) ===")
    print(
        f"events={int(metrics['events'])}, p={cfg.momentum_gev:.2f} GeV/c, "
        f"n={cfg.refractive_index:.4f}, L={cfg.mirror_to_sensor_mm:.1f} mm"
    )
    print(f"emit_fraction_true={metrics['emit_fraction_true']:.4f}")
    print(f"ring_reco_efficiency_emit={metrics['ring_reco_efficiency_emit']:.4f}")
    print(f"false_ring_rate_noemit={metrics['false_ring_rate_noemit']:.4f}")
    print(f"pid_accuracy_overall={metrics['pid_accuracy_overall']:.4f}")
    print(f"pid_accuracy_emit={metrics['pid_accuracy_emit']:.4f}")
    print(f"theta_resolution_mrad={metrics['theta_resolution_mrad']:.3f}")
    print(f"radius_rmse_mm={metrics['radius_rmse_mm']:.3f}")
    print(f"beta_rmse={metrics['beta_rmse']:.6f}")

    preview_cols = [
        "event_id",
        "species_true",
        "species_pred",
        "emits_true",
        "has_ring",
        "n_signal",
        "n_hits_total",
        "n_inliers",
        "theta_error_mrad",
    ]
    print("\nEvent preview (first 10 rows):")
    print(events_df[preview_cols].head(10).to_string(index=False))

    # Lightweight deterministic sanity checks for CI-like validation.
    assert metrics["ring_reco_efficiency_emit"] > 0.75
    assert metrics["false_ring_rate_noemit"] < 0.20
    assert metrics["pid_accuracy_overall"] > 0.75
    assert metrics["theta_resolution_mrad"] < 8.0

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
