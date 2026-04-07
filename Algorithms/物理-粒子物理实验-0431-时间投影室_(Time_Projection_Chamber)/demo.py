"""Minimal MVP for Time Projection Chamber (TPC) track reconstruction.

Pipeline:
1. Simulate one 3D straight track with TPC-like readout (x/y pads + drift time).
2. Convert drift time back to z using calibrated drift velocity.
3. Fit x(z), y(z) with OLS and robust Huber losses.
4. Apply a short PyTorch pseudo-Huber refinement.
5. Report reconstruction quality and outlier-detection metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error


@dataclass(frozen=True)
class TPCConfig:
    n_hits: int = 84
    z_min_cm: float = 6.0
    z_max_cm: float = 95.0
    drift_velocity_cm_per_ns: float = 0.052
    pad_pitch_cm: float = 0.22
    pad_sigma_cm: float = 0.028
    time_sigma_ns: float = 0.75
    diffusion_sigma_ns: float = 0.60
    outlier_fraction: float = 0.14
    outlier_residual_threshold_cm: float = 0.33
    huber_f_scale: float = 1.2
    torch_delta: float = 1.0
    torch_steps: int = 60
    torch_lr: float = 0.002
    torch_anchor_lambda: float = 0.12


def line_xy(z: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate x(z), y(z) for params=[kx,bx,ky,by]."""
    kx, bx, ky, by = params
    x = kx * z + bx
    y = ky * z + by
    return x, y


def linear_xy_fit(z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS fit for x(z)=kx*z+bx and y(z)=ky*z+by."""
    design = np.column_stack([z, np.ones_like(z)])
    p_x, *_ = np.linalg.lstsq(design, x, rcond=None)
    p_y, *_ = np.linalg.lstsq(design, y, rcond=None)
    return np.array([p_x[0], p_x[1], p_y[0], p_y[1]], dtype=np.float64)


def huber_xy_fit(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    initial: np.ndarray,
    scale_xy_cm: float,
    f_scale: float,
) -> np.ndarray:
    """Robust Huber fit over concatenated x/y residuals."""

    def residuals(p: np.ndarray) -> np.ndarray:
        x_pred, y_pred = line_xy(z, p)
        rx = (x_pred - x) / scale_xy_cm
        ry = (y_pred - y) / scale_xy_cm
        return np.concatenate([rx, ry])

    result = least_squares(
        residuals,
        x0=initial,
        loss="huber",
        f_scale=f_scale,
        max_nfev=400,
    )
    return result.x.astype(np.float64)


def torch_refine(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    initial: np.ndarray,
    scale_xy_cm: float,
    delta: float,
    steps: int,
    lr: float,
    anchor_lambda: float,
) -> tuple[np.ndarray, float]:
    """Small pseudo-Huber refinement with PyTorch autograd."""
    z_t = torch.tensor(z, dtype=torch.float64)
    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)

    p0 = torch.tensor(initial, dtype=torch.float64)
    p = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([p], lr=lr)

    final_loss = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)

        x_pred = p[0] * z_t + p[1]
        y_pred = p[2] * z_t + p[3]
        residual = torch.cat([(x_pred - x_t), (y_pred - y_t)]) / scale_xy_cm

        pseudo_huber = delta * delta * (torch.sqrt(1.0 + (residual / delta) ** 2) - 1.0)
        anchor = anchor_lambda * torch.mean((p - p0) ** 2)
        loss = torch.mean(pseudo_huber) + anchor
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    return p.detach().cpu().numpy(), final_loss


def simulate_event(
    cfg: TPCConfig,
    true_params: np.ndarray,
    seed: int = 431,
) -> dict[str, np.ndarray]:
    """Create one synthetic TPC event with inliers and outliers."""
    rng = np.random.default_rng(seed)

    z_true = np.sort(rng.uniform(cfg.z_min_cm, cfg.z_max_cm, size=cfg.n_hits))
    x_true, y_true = line_xy(z_true, true_params)

    # Pad readout: quantization + electronics noise.
    x_meas = np.round(x_true / cfg.pad_pitch_cm) * cfg.pad_pitch_cm
    y_meas = np.round(y_true / cfg.pad_pitch_cm) * cfg.pad_pitch_cm
    x_meas += rng.normal(0.0, cfg.pad_sigma_cm, size=cfg.n_hits)
    y_meas += rng.normal(0.0, cfg.pad_sigma_cm, size=cfg.n_hits)

    # Time readout with z-dependent diffusion broadening.
    base_t = z_true / cfg.drift_velocity_cm_per_ns
    sigma_t = cfg.time_sigma_ns + cfg.diffusion_sigma_ns * np.sqrt(z_true / cfg.z_max_cm)
    t_meas = base_t + rng.normal(0.0, sigma_t, size=cfg.n_hits)

    n_outliers = int(round(cfg.n_hits * cfg.outlier_fraction))
    outlier_idx = rng.choice(cfg.n_hits, size=n_outliers, replace=False)
    is_outlier = np.zeros(cfg.n_hits, dtype=bool)
    is_outlier[outlier_idx] = True

    x_lo, x_hi = float(np.min(x_true) - 3.2), float(np.max(x_true) + 3.2)
    y_lo, y_hi = float(np.min(y_true) - 3.2), float(np.max(y_true) + 3.2)

    x_meas[outlier_idx] = rng.uniform(x_lo, x_hi, size=n_outliers)
    y_meas[outlier_idx] = rng.uniform(y_lo, y_hi, size=n_outliers)

    t_lo = cfg.z_min_cm / cfg.drift_velocity_cm_per_ns
    t_hi = cfg.z_max_cm / cfg.drift_velocity_cm_per_ns
    t_meas[outlier_idx] = rng.uniform(t_lo, t_hi, size=n_outliers)

    z_meas = np.clip(
        t_meas * cfg.drift_velocity_cm_per_ns,
        cfg.z_min_cm,
        cfg.z_max_cm,
    )

    return {
        "z_true": z_true,
        "x_true": x_true,
        "y_true": y_true,
        "z_meas": z_meas,
        "x_meas": x_meas,
        "y_meas": y_meas,
        "is_outlier": is_outlier,
    }


def evaluate_method(
    method: str,
    params: np.ndarray,
    event: dict[str, np.ndarray],
    true_params: np.ndarray,
    cfg: TPCConfig,
) -> dict[str, float | str]:
    """Compute fit and outlier metrics for one method."""
    z_true = event["z_true"]
    z_meas = event["z_meas"]
    x_true = event["x_true"]
    y_true = event["y_true"]
    x_meas = event["x_meas"]
    y_meas = event["y_meas"]
    is_outlier = event["is_outlier"]
    is_inlier = ~is_outlier

    x_pred_true, y_pred_true = line_xy(z_true, params)
    x_pred_meas, y_pred_meas = line_xy(z_meas, params)

    mse_x = mean_squared_error(x_true[is_inlier], x_pred_true[is_inlier])
    mse_y = mean_squared_error(y_true[is_inlier], y_pred_true[is_inlier])
    transverse_rmse_inlier = float(np.sqrt(0.5 * (mse_x + mse_y)))

    residual_meas = np.sqrt((x_pred_meas - x_meas) ** 2 + (y_pred_meas - y_meas) ** 2)
    outlier_hat = residual_meas > cfg.outlier_residual_threshold_cm

    outlier_recall = float(np.mean(outlier_hat[is_outlier]))
    outlier_precision = float(np.sum(outlier_hat & is_outlier) / max(np.sum(outlier_hat), 1))
    inlier_keep_rate = float(np.mean(~outlier_hat[is_inlier]))

    return {
        "method": method,
        "slope_error_x": float(abs(params[0] - true_params[0])),
        "slope_error_y": float(abs(params[2] - true_params[2])),
        "intercept_error_x_cm": float(abs(params[1] - true_params[1])),
        "intercept_error_y_cm": float(abs(params[3] - true_params[3])),
        "transverse_rmse_inlier_cm": transverse_rmse_inlier,
        "mean_residual_meas_cm": float(np.mean(residual_meas)),
        "outlier_recall": outlier_recall,
        "outlier_precision": outlier_precision,
        "inlier_keep_rate": inlier_keep_rate,
    }


def main() -> None:
    cfg = TPCConfig()

    # True single-track model (roughly beam-aligned in z).
    true_params = np.array([0.083, -4.25, -0.057, 2.75], dtype=np.float64)

    event = simulate_event(cfg=cfg, true_params=true_params, seed=431)

    z_meas = event["z_meas"]
    x_meas = event["x_meas"]
    y_meas = event["y_meas"]

    ols_params = linear_xy_fit(z_meas, x_meas, y_meas)

    scale_xy = cfg.pad_pitch_cm / np.sqrt(12.0)
    huber_params = huber_xy_fit(
        z=z_meas,
        x=x_meas,
        y=y_meas,
        initial=ols_params,
        scale_xy_cm=scale_xy,
        f_scale=cfg.huber_f_scale,
    )

    torch_params, torch_loss = torch_refine(
        z=z_meas,
        x=x_meas,
        y=y_meas,
        initial=huber_params,
        scale_xy_cm=scale_xy,
        delta=cfg.torch_delta,
        steps=cfg.torch_steps,
        lr=cfg.torch_lr,
        anchor_lambda=cfg.torch_anchor_lambda,
    )

    rows = [
        evaluate_method("OLS", ols_params, event, true_params, cfg),
        evaluate_method("Huber", huber_params, event, true_params, cfg),
        evaluate_method("Huber+Torch", torch_params, event, true_params, cfg),
    ]
    df = pd.DataFrame(rows)

    print("TPC MVP: single-track 3D reconstruction via drift time")
    print(f"Total hits: {cfg.n_hits}, injected outliers: {int(np.sum(event['is_outlier']))}")
    print(f"Torch final pseudo-Huber loss: {torch_loss:.6f}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    huber = rows[1]
    refined = rows[2]

    assert huber["transverse_rmse_inlier_cm"] < 0.28
    assert huber["outlier_recall"] >= 0.70
    assert refined["slope_error_x"] < 0.010
    assert refined["slope_error_y"] < 0.010

    print("All checks passed.")


if __name__ == "__main__":
    main()
