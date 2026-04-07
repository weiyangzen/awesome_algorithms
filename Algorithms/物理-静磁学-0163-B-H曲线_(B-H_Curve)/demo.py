"""Minimal runnable MVP for B-H hysteresis curve simulation.

This script builds a major B-H loop with a transparent branch-based model:
- B = mu0 * (H + M)
- M follows direction-dependent target branches with first-order relaxation.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BHParams:
    """Parameters for a simple ferromagnetic hysteresis approximation."""

    mu0: float = 4.0e-7 * np.pi  # Vacuum permeability (H/m)
    ms: float = 1.6e6  # Saturation magnetization (A/m)
    a: float = 260.0  # Shape factor controlling transition steepness (A/m)
    hc: float = 180.0  # Coercive field shift in branch model (A/m)
    beta: float = 0.16  # Relaxation factor toward branch target in each step


def build_h_trajectory(h_max: float, points_per_leg: int = 600) -> np.ndarray:
    """Build one major-loop excitation trajectory: -Hmax -> Hmax -> -Hmax."""
    up = np.linspace(-h_max, h_max, points_per_leg, endpoint=False)
    down = np.linspace(h_max, -h_max, points_per_leg + 1)
    return np.concatenate([up, down])


def target_magnetization(h_value: float, direction: float, params: BHParams) -> float:
    """Direction-dependent branch target of magnetization.

    direction > 0: ascending branch, shifted by +Hc in field threshold.
    direction < 0: descending branch, shifted by -Hc in field threshold.
    """
    return params.ms * np.tanh((h_value - direction * params.hc) / params.a)


def simulate_bh_curve(h_path: np.ndarray, params: BHParams) -> tuple[np.ndarray, np.ndarray]:
    """Simulate magnetization M(H) and magnetic flux density B(H)."""
    m = np.zeros_like(h_path, dtype=float)
    b = np.zeros_like(h_path, dtype=float)

    # Start from negative saturation to form a full major loop.
    m[0] = -params.ms
    b[0] = params.mu0 * (h_path[0] + m[0])

    for i in range(1, len(h_path)):
        dh = h_path[i] - h_path[i - 1]
        direction = 1.0 if dh >= 0.0 else -1.0
        m_target = target_magnetization(h_path[i], direction, params)
        m[i] = m[i - 1] + params.beta * (m_target - m[i - 1])
        b[i] = params.mu0 * (h_path[i] + m[i])

    return m, b


def segment_crossings(x: np.ndarray, y: np.ndarray, y_target: float = 0.0) -> list[float]:
    """Find x values where y crosses y_target using linear interpolation."""
    values: list[float] = []
    y_shift = y - y_target
    for i in range(len(y_shift) - 1):
        y0 = y_shift[i]
        y1 = y_shift[i + 1]
        if y0 == 0.0:
            values.append(float(x[i]))
            continue
        if y0 * y1 < 0.0:
            t = y0 / (y0 - y1)
            x_cross = x[i] + t * (x[i + 1] - x[i])
            values.append(float(x_cross))
    return values


def slope_near_origin(h: np.ndarray, b: np.ndarray, ratio: float = 0.08) -> float:
    """Estimate differential permeability dB/dH near H=0 via linear fit."""
    h_lim = np.max(np.abs(h)) * ratio
    mask = np.abs(h) <= h_lim
    if np.count_nonzero(mask) < 3:
        return float("nan")
    coeff = np.polyfit(h[mask], b[mask], deg=1)
    return float(coeff[0])


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Compute \u222b y dx using explicit trapezoid rule for NumPy compatibility."""
    dx = x[1:] - x[:-1]
    y_mid = 0.5 * (y[1:] + y[:-1])
    return float(np.sum(dx * y_mid))


def compute_metrics(h: np.ndarray, m: np.ndarray, b: np.ndarray, params: BHParams) -> dict[str, float]:
    """Extract physically interpretable metrics from one loop."""
    b_at_h0 = segment_crossings(b, h, y_target=0.0)
    h_at_b0 = segment_crossings(h, b, y_target=0.0)

    br_pos = max(b_at_h0) if b_at_h0 else float("nan")
    br_neg = min(b_at_h0) if b_at_h0 else float("nan")
    hc_pos = max(h_at_b0) if h_at_b0 else float("nan")
    hc_neg = min(h_at_b0) if h_at_b0 else float("nan")

    # Closed-loop integral ∮ H dB or ∮ B dH are proportional to loss.
    # Here we use |∮ B dH| as a scalar hysteresis-area indicator.
    loop_area = abs(trapezoid_integral(b, h))

    return {
        "Hmax_A_per_m": float(np.max(np.abs(h))),
        "Bsat_T": float(np.max(np.abs(b))),
        "Br_pos_T": float(br_pos),
        "Br_neg_T": float(br_neg),
        "Hc_pos_A_per_m": float(hc_pos),
        "Hc_neg_A_per_m": float(hc_neg),
        "mu_diff_origin_H_per_m": slope_near_origin(h, b),
        "loop_area_BdH": loop_area,
        "sat_ratio_absM_over_Ms": float(np.max(np.abs(m)) / params.ms),
    }


def amplitude_sweep(h_max_values: list[float], params: BHParams) -> pd.DataFrame:
    """Run multiple amplitudes and summarize loop characteristics."""
    rows: list[dict[str, float]] = []
    for h_max in h_max_values:
        h = build_h_trajectory(h_max=h_max, points_per_leg=500)
        m, b = simulate_bh_curve(h, params)
        metrics = compute_metrics(h, m, b, params)
        rows.append(metrics)

    df = pd.DataFrame(rows).sort_values("Hmax_A_per_m").reset_index(drop=True)
    return df


def main() -> None:
    params = BHParams()
    base_dir = Path(__file__).resolve().parent

    # Main major-loop run.
    h_main = build_h_trajectory(h_max=1200.0, points_per_leg=700)
    m_main, b_main = simulate_bh_curve(h_main, params)
    main_metrics = compute_metrics(h_main, m_main, b_main, params)

    # Sweep to show approach to saturation and loop-area evolution.
    sweep_df = amplitude_sweep([300.0, 500.0, 800.0, 1200.0, 1600.0], params)

    # Save reproducible artifacts.
    sample_df = pd.DataFrame(
        {
            "H_A_per_m": h_main,
            "M_A_per_m": m_main,
            "B_T": b_main,
        }
    )
    sample_path = base_dir / "bh_curve_samples.csv"
    sweep_path = base_dir / "bh_sweep_summary.csv"
    sample_df.to_csv(sample_path, index=False)
    sweep_df.to_csv(sweep_path, index=False)

    # Print concise report.
    print("B-H Curve MVP report")
    print("=" * 60)
    print(
        "Parameters: "
        f"Ms={params.ms:.0f} A/m, a={params.a:.1f} A/m, "
        f"Hc={params.hc:.1f} A/m, beta={params.beta:.2f}"
    )
    print("\nMain-loop metrics:")
    for key, value in main_metrics.items():
        print(f"  {key:24s}: {value: .6e}")

    print("\nAmplitude sweep summary:")
    print(
        sweep_df.to_string(
            index=False,
            float_format=lambda x: f"{x: .6e}",
        )
    )

    # Minimal sanity checks for automated validation.
    assert np.isfinite(sweep_df.to_numpy()).all(), "Non-finite values found in sweep summary"
    assert (sweep_df["sat_ratio_absM_over_Ms"] <= 1.02).all(), "Saturation ratio is unexpectedly high"
    assert (sweep_df["loop_area_BdH"] >= 0.0).all(), "Loop area should be non-negative"

    print(f"\nSaved: {sample_path.name}, {sweep_path.name}")


if __name__ == "__main__":
    main()
