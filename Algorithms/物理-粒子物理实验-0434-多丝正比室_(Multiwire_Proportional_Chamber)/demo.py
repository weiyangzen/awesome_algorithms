"""Minimal MVP for Multiwire Proportional Chamber (MWPC) track reconstruction.

Pipeline:
1. Simulate one straight-track crossing several MWPC planes.
2. Build three-wire charge sharing (left/center/right) per plane.
3. Reconstruct hit x-position by charge centroid.
4. Fit global track x(z)=a*z+b with weighted OLS and weighted Huber.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MWPCConfig:
    n_planes: int = 12
    plane_gap_cm: float = 1.0
    wire_pitch_cm: float = 0.20
    charge_spread_cm: float = 0.07
    gain_mean: float = 12000.0
    gain_rel_sigma: float = 0.22
    electronics_noise: float = 130.0
    outlier_shift_cm: float = 0.32


def weighted_linear_fit(z: np.ndarray, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted least squares for x = a*z + b."""
    design = np.column_stack([z, np.ones_like(z)])
    sqrt_w = np.sqrt(weight)
    lhs = design * sqrt_w[:, None]
    rhs = x * sqrt_w
    params, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return params


def weighted_huber_fit(
    z: np.ndarray,
    x: np.ndarray,
    base_weight: np.ndarray,
    k: float = 1.5,
    max_iter: int = 20,
) -> dict[str, np.ndarray | float | int]:
    """Iteratively reweighted fit with Huber influence function."""
    params = weighted_linear_fit(z, x, base_weight)
    robust_weight = np.ones_like(x)

    for it in range(1, max_iter + 1):
        pred = params[0] * z + params[1]
        residual = x - pred

        # Robust scale estimate from MAD.
        mad = np.median(np.abs(residual - np.median(residual)))
        scale = max(1e-6, 1.4826 * mad)

        norm_res = np.abs(residual) / (k * scale)
        robust_weight = np.where(norm_res <= 1.0, 1.0, 1.0 / norm_res)

        weight = base_weight * robust_weight
        new_params = weighted_linear_fit(z, x, weight)

        if np.linalg.norm(new_params - params) < 1e-12:
            params = new_params
            return {
                "a": float(params[0]),
                "b": float(params[1]),
                "iterations": it,
                "robust_weight": robust_weight,
            }

        params = new_params

    return {
        "a": float(params[0]),
        "b": float(params[1]),
        "iterations": max_iter,
        "robust_weight": robust_weight,
    }


def simulate_event(
    cfg: MWPCConfig,
    true_a: float,
    true_b: float,
    seed: int = 414,
) -> dict[str, np.ndarray | int]:
    """Generate one synthetic MWPC event with one injected outlier plane."""
    rng = np.random.default_rng(seed)

    plane_id = np.arange(cfg.n_planes)
    z = plane_id * cfg.plane_gap_cm
    x_true = true_a * z + true_b

    wire_index = np.round(x_true / cfg.wire_pitch_cm).astype(int)
    x_center = wire_index * cfg.wire_pitch_cm
    x_left = x_center - cfg.wire_pitch_cm
    x_right = x_center + cfg.wire_pitch_cm

    gain = cfg.gain_mean * rng.lognormal(mean=0.0, sigma=cfg.gain_rel_sigma, size=cfg.n_planes)

    q_left = gain * np.exp(-0.5 * ((x_true - x_left) / cfg.charge_spread_cm) ** 2)
    q_center = gain * np.exp(-0.5 * ((x_true - x_center) / cfg.charge_spread_cm) ** 2)
    q_right = gain * np.exp(-0.5 * ((x_true - x_right) / cfg.charge_spread_cm) ** 2)

    q_left += rng.normal(0.0, cfg.electronics_noise, size=cfg.n_planes)
    q_center += rng.normal(0.0, cfg.electronics_noise, size=cfg.n_planes)
    q_right += rng.normal(0.0, cfg.electronics_noise, size=cfg.n_planes)

    q_left = np.clip(q_left, 1.0, None)
    q_center = np.clip(q_center, 1.0, None)
    q_right = np.clip(q_right, 1.0, None)

    q_total = q_left + q_center + q_right

    x_centroid = (q_left * x_left + q_center * x_center + q_right * x_right) / q_total

    snr = q_total / (np.sqrt(3.0) * cfg.electronics_noise + 1e-9)
    sigma_x = cfg.wire_pitch_cm / np.sqrt(12.0 * (1.0 + 0.35 * snr))
    sigma_x = np.clip(sigma_x, 0.015, 0.09)

    x_meas = x_centroid.copy()
    outlier_plane = int(rng.integers(0, cfg.n_planes))
    outlier_sign = int(rng.choice([-1, 1]))
    x_meas[outlier_plane] += outlier_sign * cfg.outlier_shift_cm

    return {
        "plane_id": plane_id,
        "z": z,
        "x_true": x_true,
        "x_center": x_center,
        "x_meas": x_meas,
        "sigma_x": sigma_x,
        "q_left": q_left,
        "q_center": q_center,
        "q_right": q_right,
        "q_total": q_total,
        "outlier_plane": outlier_plane,
    }


def evaluate(
    event: dict[str, np.ndarray | int],
    a_hat: float,
    b_hat: float,
    true_a: float,
    true_b: float,
    label: str,
    iteration: int,
) -> dict[str, float | str]:
    z = event["z"]  # type: ignore[index]
    x_true = event["x_true"]  # type: ignore[index]
    x_meas = event["x_meas"]  # type: ignore[index]
    sigma_x = event["sigma_x"]  # type: ignore[index]

    x_pred = a_hat * z + b_hat  # type: ignore[operator]

    truth_rmse = float(np.sqrt(np.mean((x_pred - x_true) ** 2)))
    hit_rmse = float(np.sqrt(np.mean((x_pred - x_meas) ** 2)))
    chi2_ndf = float(np.sum(((x_meas - x_pred) / sigma_x) ** 2) / (len(z) - 2))  # type: ignore[arg-type]

    outlier_plane = int(event["outlier_plane"])
    outlier_abs_residual = float(abs(x_meas[outlier_plane] - x_pred[outlier_plane]))  # type: ignore[index]

    return {
        "method": label,
        "iterations": float(iteration),
        "slope_error": abs(a_hat - true_a),
        "intercept_error_cm": abs(b_hat - true_b),
        "truth_rmse_cm": truth_rmse,
        "hit_rmse_cm": hit_rmse,
        "chi2_ndf": chi2_ndf,
        "outlier_abs_residual_cm": outlier_abs_residual,
    }


def main() -> None:
    cfg = MWPCConfig()

    true_a = 0.095
    true_b = -0.52
    event = simulate_event(cfg, true_a=true_a, true_b=true_b, seed=414)

    z = event["z"]  # type: ignore[index]
    x_meas = event["x_meas"]  # type: ignore[index]
    sigma_x = event["sigma_x"]  # type: ignore[index]
    base_weight = 1.0 / (sigma_x**2)

    ols_params = weighted_linear_fit(z, x_meas, base_weight)
    huber = weighted_huber_fit(z, x_meas, base_weight, k=1.5, max_iter=20)

    metrics = [
        evaluate(
            event,
            a_hat=float(ols_params[0]),
            b_hat=float(ols_params[1]),
            true_a=true_a,
            true_b=true_b,
            label="Weighted-OLS",
            iteration=1,
        ),
        evaluate(
            event,
            a_hat=float(huber["a"]),
            b_hat=float(huber["b"]),
            true_a=true_a,
            true_b=true_b,
            label="Weighted-Huber",
            iteration=int(huber["iterations"]),
        ),
    ]

    plane_table = pd.DataFrame(
        {
            "plane": event["plane_id"],
            "z_cm": z,
            "x_true_cm": event["x_true"],
            "x_meas_cm": x_meas,
            "sigma_x_cm": sigma_x,
            "q_total": event["q_total"],
        }
    )

    metric_table = pd.DataFrame(metrics)

    print("MWPC MVP: single-track reconstruction with charge centroid")
    print(f"Injected outlier plane: {event['outlier_plane']}")
    print("\nPer-plane summary:")
    print(plane_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nFit metrics:")
    print(metric_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    ols_truth_rmse = float(metrics[0]["truth_rmse_cm"])
    huber_truth_rmse = float(metrics[1]["truth_rmse_cm"])
    huber_slope_error = float(metrics[1]["slope_error"])
    huber_intercept_error = float(metrics[1]["intercept_error_cm"])

    assert huber_truth_rmse <= ols_truth_rmse + 1e-9
    assert huber_truth_rmse < 0.08
    assert huber_slope_error < 0.02
    assert huber_intercept_error < 0.12

    print("All checks passed.")


if __name__ == "__main__":
    main()
