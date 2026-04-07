"""Minimal runnable MVP for calorimeter energy reconstruction.

This demo builds a toy ECAL+HCAL detector response model, then compares:
1) physics-inspired baseline reconstruction (inverse sampling fractions), and
2) a transparent linear calibration model trained on simulated data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SAMPLING_FRACTION_ECAL = 0.28
SAMPLING_FRACTION_HCAL = 0.12


@dataclass(frozen=True)
class DetectorConfig:
    """Toy calorimeter detector configuration."""

    sf_ecal: float = SAMPLING_FRACTION_ECAL
    sf_hcal: float = SAMPLING_FRACTION_HCAL
    noise_ecal: float = 0.025
    noise_hcal: float = 0.060


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _normalize_three_parts(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """Normalize non-negative parts to sum <= 1 while keeping proportions."""
    parts = np.clip(np.array([a, b, c], dtype=float), 0.0, None)
    total = float(parts.sum())
    if total <= 1.0:
        return float(parts[0]), float(parts[1]), float(parts[2])
    scaled = parts / total
    return float(scaled[0]), float(scaled[1]), float(scaled[2])


def sample_shower_fractions(particle_type: str, rng: np.random.Generator) -> Tuple[float, float, float]:
    """Return (em_fraction, had_fraction, leakage_fraction)."""
    if particle_type == "electron":
        em = _clip01(rng.normal(0.93, 0.03))
        leakage = _clip01(rng.normal(0.03, 0.012))
    elif particle_type == "pion":
        em = _clip01(rng.normal(0.35, 0.09))
        leakage = _clip01(rng.normal(0.12, 0.04))
    else:
        raise ValueError(f"Unsupported particle_type: {particle_type}")

    had = max(0.0, 1.0 - em - leakage)
    return _normalize_three_parts(em, had, leakage)


def simulate_event(
    event_id: int,
    particle_type: str,
    true_energy: float,
    cfg: DetectorConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Simulate one event's layer-level sampled energies."""
    em_frac, had_frac, leakage_frac = sample_shower_fractions(particle_type, rng)

    # Longitudinal split in each subsystem.
    ecal_front_ratio = float(np.clip(rng.normal(0.70, 0.06), 0.45, 0.90))
    hcal_front_ratio = float(np.clip(rng.normal(0.55, 0.08), 0.30, 0.85))

    e_dep_ecal = true_energy * em_frac
    e_dep_hcal = true_energy * had_frac

    mean_ecal_l1 = cfg.sf_ecal * e_dep_ecal * ecal_front_ratio
    mean_ecal_l2 = cfg.sf_ecal * e_dep_ecal * (1.0 - ecal_front_ratio)
    mean_hcal_l1 = cfg.sf_hcal * e_dep_hcal * hcal_front_ratio
    mean_hcal_l2 = cfg.sf_hcal * e_dep_hcal * (1.0 - hcal_front_ratio)

    # Stochastic + constant noise in sampled signal units.
    ecal_l1 = max(0.0, mean_ecal_l1 + rng.normal(0.0, cfg.noise_ecal * np.sqrt(max(mean_ecal_l1, 1.0))))
    ecal_l2 = max(0.0, mean_ecal_l2 + rng.normal(0.0, cfg.noise_ecal * np.sqrt(max(mean_ecal_l2, 1.0))))
    hcal_l1 = max(0.0, mean_hcal_l1 + rng.normal(0.0, cfg.noise_hcal * np.sqrt(max(mean_hcal_l1, 1.0))))
    hcal_l2 = max(0.0, mean_hcal_l2 + rng.normal(0.0, cfg.noise_hcal * np.sqrt(max(mean_hcal_l2, 1.0))))

    return {
        "event_id": float(event_id),
        "particle_type": 0.0 if particle_type == "electron" else 1.0,
        "true_energy": float(true_energy),
        "ecal_l1": float(ecal_l1),
        "ecal_l2": float(ecal_l2),
        "hcal_l1": float(hcal_l1),
        "hcal_l2": float(hcal_l2),
        "leakage_true": float(true_energy * leakage_frac),
    }


def build_dataset(n_events: int, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible mixed electron/pion dataset."""
    if n_events <= 0:
        raise ValueError("n_events must be positive")

    rng = np.random.default_rng(seed)
    cfg = DetectorConfig()
    rows = []

    for event_id in range(n_events):
        particle_type = "electron" if rng.random() < 0.5 else "pion"
        true_energy = float(rng.uniform(20.0, 300.0))
        rows.append(simulate_event(event_id, particle_type, true_energy, cfg, rng))

    df = pd.DataFrame(rows)
    total_signal = df["ecal_l1"] + df["ecal_l2"] + df["hcal_l1"] + df["hcal_l2"]
    df["signal_total"] = total_signal
    df["ecal_total"] = df["ecal_l1"] + df["ecal_l2"]
    df["hcal_total"] = df["hcal_l1"] + df["hcal_l2"]
    df["ecal_signal_ratio"] = df["ecal_total"] / (df["signal_total"] + 1e-9)
    df["longitudinal_ratio"] = (df["ecal_l2"] + df["hcal_l2"]) / (df["signal_total"] + 1e-9)
    return df


def baseline_reconstruction(df: pd.DataFrame, cfg: DetectorConfig) -> np.ndarray:
    """Simple inverse-sampling-fraction reconstruction."""
    return (
        df["ecal_total"].to_numpy(dtype=float) / cfg.sf_ecal
        + df["hcal_total"].to_numpy(dtype=float) / cfg.sf_hcal
    )


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Select transparent physics-inspired features."""
    feature_columns = [
        "ecal_l1",
        "ecal_l2",
        "hcal_l1",
        "hcal_l2",
        "ecal_signal_ratio",
        "longitudinal_ratio",
        "particle_type",
    ]
    return df[feature_columns].to_numpy(dtype=float)


def train_calibration_model(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Train a linear calibration pipeline."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1e-3)),
        ]
    )
    model.fit(x_train, y_train)
    return model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return core calorimeter performance metrics."""
    residual = y_pred - y_true
    rel_residual = residual / np.clip(y_true, 1e-9, None)

    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    mape = float(np.mean(np.abs(rel_residual))) * 100.0
    bias_pct = float(np.mean(rel_residual)) * 100.0
    resolution_pct = float(np.std(rel_residual)) * 100.0
    slope, intercept = np.polyfit(y_true, y_pred, deg=1)
    r2 = float(1.0 - np.sum(residual**2) / np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "MAE_GeV": mae,
        "RMSE_GeV": rmse,
        "MAPE_%": mape,
        "Bias_%": bias_pct,
        "Resolution_%": resolution_pct,
        "LinearitySlope": float(slope),
        "LinearityIntercept": float(intercept),
        "R2": r2,
    }


def binned_resolution_table(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    bins: Iterable[float],
) -> pd.DataFrame:
    """Compare relative-resolution (%) in energy bins for two methods."""
    bins_arr = np.array(list(bins), dtype=float)
    if bins_arr.ndim != 1 or bins_arr.size < 2:
        raise ValueError("bins must contain at least two edges")

    rel_a = (y_pred_a - y_true) / np.clip(y_true, 1e-9, None)
    rel_b = (y_pred_b - y_true) / np.clip(y_true, 1e-9, None)

    std_a, _, _ = binned_statistic(y_true, rel_a, statistic="std", bins=bins_arr)
    std_b, _, _ = binned_statistic(y_true, rel_b, statistic="std", bins=bins_arr)
    cnt, _, _ = binned_statistic(y_true, y_true, statistic="count", bins=bins_arr)

    rows = []
    for i in range(len(bins_arr) - 1):
        left = bins_arr[i]
        right = bins_arr[i + 1]
        rows.append(
            {
                "EnergyBin_GeV": f"[{left:.0f}, {right:.0f})",
                "Count": int(cnt[i]),
                "BaselineResolution_%": float(std_a[i] * 100.0) if np.isfinite(std_a[i]) else np.nan,
                "CalibratedResolution_%": float(std_b[i] * 100.0) if np.isfinite(std_b[i]) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Run a deterministic calorimeter calibration experiment."""
    cfg = DetectorConfig()
    df = build_dataset(n_events=2500, seed=42)

    train_df, test_df = train_test_split(df, test_size=0.30, random_state=13, shuffle=True)
    x_train = build_feature_matrix(train_df)
    y_train = train_df["true_energy"].to_numpy(dtype=float)
    x_test = build_feature_matrix(test_df)
    y_test = test_df["true_energy"].to_numpy(dtype=float)

    baseline_test = baseline_reconstruction(test_df, cfg)
    model = train_calibration_model(x_train, y_train)
    calibrated_test = model.predict(x_test)

    baseline_metrics = evaluate_predictions(y_test, baseline_test)
    calibrated_metrics = evaluate_predictions(y_test, calibrated_test)
    metrics_df = pd.DataFrame(
        [
            {"Method": "Baseline", **baseline_metrics},
            {"Method": "LinearCalibrated", **calibrated_metrics},
        ]
    )

    preview_df = test_df[["true_energy", "ecal_total", "hcal_total"]].copy().reset_index(drop=True).head(12)
    preview_df["baseline_E"] = baseline_test[: len(preview_df)]
    preview_df["calibrated_E"] = calibrated_test[: len(preview_df)]
    preview_df["baseline_rel_err_%"] = (preview_df["baseline_E"] - preview_df["true_energy"]) / preview_df[
        "true_energy"
    ] * 100.0
    preview_df["calibrated_rel_err_%"] = (
        (preview_df["calibrated_E"] - preview_df["true_energy"]) / preview_df["true_energy"] * 100.0
    )

    resolution_df = binned_resolution_table(
        y_true=y_test,
        y_pred_a=baseline_test,
        y_pred_b=calibrated_test,
        bins=[20, 50, 80, 120, 170, 230, 300],
    )

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    print("=== Calorimeter Energy Reconstruction Demo ===")
    print(f"Total events: {len(df)}, train: {len(train_df)}, test: {len(test_df)}")
    print()
    print("Performance summary (lower is better except R2 and slope~1):")
    print(metrics_df.to_string(index=False))
    print()
    print("Prediction preview (first 12 test events):")
    print(preview_df.to_string(index=False))
    print()
    print("Binned relative resolution comparison (%):")
    print(resolution_df.to_string(index=False))


if __name__ == "__main__":
    main()
