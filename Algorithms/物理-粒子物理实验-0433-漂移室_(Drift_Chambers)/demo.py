"""Minimal MVP for Drift Chamber track reconstruction.

Pipeline:
1. Simulate one straight charged-particle track crossing layered drift cells.
2. Convert drift time to drift distance.
3. Resolve left-right ambiguity by iterative sign assignment.
4. Fit line x(y) = a*y + b with robust (Huber) least squares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass(frozen=True)
class ChamberConfig:
    n_layers: int = 10
    layer_gap_cm: float = 1.2
    wire_pitch_cm: float = 1.0
    drift_velocity_cm_per_ns: float = 0.04
    time_sigma_ns: float = 0.8
    outlier_time_bias_ns: float = 5.0


def linear_fit(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Least-squares fit for x = a*y + b."""
    design = np.column_stack([y, np.ones_like(y)])
    params, *_ = np.linalg.lstsq(design, x, rcond=None)
    return params


def robust_fit(y: np.ndarray, x: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """Huber-loss robust fit for x = a*y + b."""

    def residual(params: np.ndarray) -> np.ndarray:
        a, b = params
        return a * y + b - x

    result = least_squares(residual, x0=initial, loss="huber", f_scale=0.15)
    return result.x


def simulate_event(
    cfg: ChamberConfig,
    true_a: float,
    true_b: float,
    seed: int = 7,
) -> dict[str, np.ndarray | int | float]:
    """Generate one synthetic drift-chamber event."""
    rng = np.random.default_rng(seed)

    layer_ids = np.arange(cfg.n_layers)
    y = layer_ids * cfg.layer_gap_cm

    # Stagger neighboring layers to improve left-right discrimination.
    offsets = np.where(layer_ids % 2 == 0, 0.0, 0.5 * cfg.wire_pitch_cm)

    x_true = true_a * y + true_b

    wire_index = np.round((x_true - offsets) / cfg.wire_pitch_cm).astype(int)
    x_wire = offsets + wire_index * cfg.wire_pitch_cm

    d_true = np.abs(x_true - x_wire)
    sign_true = np.where(x_true >= x_wire, 1, -1)

    time_meas = d_true / cfg.drift_velocity_cm_per_ns
    time_meas += rng.normal(0.0, cfg.time_sigma_ns, size=cfg.n_layers)
    time_meas = np.clip(time_meas, 0.0, None)

    outlier_layer = int(rng.integers(0, cfg.n_layers))
    time_meas[outlier_layer] += cfg.outlier_time_bias_ns

    d_meas = time_meas * cfg.drift_velocity_cm_per_ns
    d_meas = np.clip(d_meas, 0.0, 0.9 * cfg.wire_pitch_cm)

    return {
        "y": y,
        "x_true": x_true,
        "x_wire": x_wire,
        "d_true": d_true,
        "d_meas": d_meas,
        "sign_true": sign_true,
        "outlier_layer": outlier_layer,
    }


def reconstruct_track(
    y: np.ndarray,
    x_wire: np.ndarray,
    d_meas: np.ndarray,
    use_robust: bool,
    max_iter: int = 15,
) -> dict[str, np.ndarray | float | int]:
    """Iteratively resolve left-right ambiguity and fit track parameters."""
    params = linear_fit(y, x_wire)
    fit_fn = robust_fit if use_robust else (lambda yy, xx, initial: linear_fit(yy, xx))

    signs = np.ones_like(y, dtype=int)
    x_hit = x_wire.copy()

    for it in range(1, max_iter + 1):
        a, b = params
        pred = a * y + b

        signs = np.where(pred >= x_wire, 1, -1)
        x_hit = x_wire + signs * d_meas

        new_params = fit_fn(y, x_hit, params)

        if np.linalg.norm(new_params - params) < 1e-10:
            params = new_params
            return {
                "a": float(params[0]),
                "b": float(params[1]),
                "signs": signs,
                "x_hit": x_hit,
                "iterations": it,
            }

        params = new_params

    return {
        "a": float(params[0]),
        "b": float(params[1]),
        "signs": signs,
        "x_hit": x_hit,
        "iterations": max_iter,
    }


def evaluate_result(
    result: dict[str, np.ndarray | float | int],
    event: dict[str, np.ndarray | int | float],
    true_a: float,
    true_b: float,
    label: str,
) -> dict[str, float | str]:
    y = event["y"]  # type: ignore[index]
    x_true = event["x_true"]  # type: ignore[index]
    x_wire = event["x_wire"]  # type: ignore[index]
    sign_true = event["sign_true"]  # type: ignore[index]

    a_hat = float(result["a"])
    b_hat = float(result["b"])
    sign_hat = result["signs"]  # type: ignore[index]

    x_pred = a_hat * y + b_hat  # type: ignore[operator]
    rmse_cm = float(np.sqrt(np.mean((x_pred - x_true) ** 2)))

    lr_acc = float(np.mean(sign_hat == sign_true))

    x_hit = result["x_hit"]  # type: ignore[index]
    residual_cm = float(np.sqrt(np.mean((x_pred - x_hit) ** 2)))
    wire_only_rmse = float(np.sqrt(np.mean((x_pred - x_wire) ** 2)))

    return {
        "method": label,
        "iterations": float(result["iterations"]),
        "slope_error": abs(a_hat - true_a),
        "intercept_error_cm": abs(b_hat - true_b),
        "track_rmse_cm": rmse_cm,
        "lr_sign_accuracy": lr_acc,
        "fit_residual_cm": residual_cm,
        "wire_center_rmse_cm": wire_only_rmse,
    }


def main() -> None:
    cfg = ChamberConfig()

    # Ground-truth straight track.
    true_a = 0.22
    true_b = -0.35

    event = simulate_event(cfg=cfg, true_a=true_a, true_b=true_b, seed=7)

    ols = reconstruct_track(
        y=event["y"],  # type: ignore[index]
        x_wire=event["x_wire"],  # type: ignore[index]
        d_meas=event["d_meas"],  # type: ignore[index]
        use_robust=False,
    )
    huber = reconstruct_track(
        y=event["y"],  # type: ignore[index]
        x_wire=event["x_wire"],  # type: ignore[index]
        d_meas=event["d_meas"],  # type: ignore[index]
        use_robust=True,
    )

    rows = [
        evaluate_result(ols, event, true_a=true_a, true_b=true_b, label="Iterative-OLS"),
        evaluate_result(huber, event, true_a=true_a, true_b=true_b, label="Iterative-Huber"),
    ]

    df = pd.DataFrame(rows)

    print("Drift Chamber MVP: single-track reconstruction")
    print(f"Injected outlier layer index: {event['outlier_layer']}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    best = rows[1]

    assert best["track_rmse_cm"] < 0.14
    assert best["lr_sign_accuracy"] >= 0.7
    assert best["slope_error"] < 0.03
    assert best["intercept_error_cm"] < 0.12

    print("All checks passed.")


if __name__ == "__main__":
    main()
