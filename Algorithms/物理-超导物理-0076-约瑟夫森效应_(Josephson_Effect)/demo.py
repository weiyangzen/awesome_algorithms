"""Josephson Effect MVP based on the overdamped RSJ model.

This script runs without interactive input and prints a concise report:
1) I-V curve under pure DC bias.
2) I-V curve under DC + AC drive.
3) Detected Shapiro steps (voltage plateaus near n*Omega).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RSJConfig:
    """Numerical parameters in normalized units."""

    dt: float = 0.05
    n_steps: int = 12_000
    burn_in_steps: int = 3_000


def simulate_average_voltage(
    i_dc: float,
    i_ac: float,
    omega: float,
    cfg: RSJConfig,
    phase0: float = 0.0,
) -> float:
    """Integrate dphi/dtau = i_dc + i_ac*sin(omega*tau) - sin(phi).

    Returns normalized average voltage <dphi/dtau> over post-burn-in steps.
    """
    if cfg.n_steps <= cfg.burn_in_steps:
        raise ValueError("n_steps must be larger than burn_in_steps")
    if cfg.dt <= 0.0:
        raise ValueError("dt must be positive")

    phi = float(phase0)
    dphi_sum = 0.0
    sample_count = 0

    for k in range(cfg.n_steps):
        tau = k * cfg.dt
        drive = i_dc + i_ac * math.sin(omega * tau)
        dphi = (drive - math.sin(phi)) * cfg.dt
        phi += dphi

        if k >= cfg.burn_in_steps:
            dphi_sum += dphi
            sample_count += 1

    return dphi_sum / (sample_count * cfg.dt)


def sweep_iv_curve(
    currents: np.ndarray,
    i_ac: float,
    omega: float,
    cfg: RSJConfig,
) -> np.ndarray:
    """Compute normalized I-V curve for a current grid."""
    voltages = np.empty_like(currents, dtype=float)
    for i, cur in enumerate(currents):
        voltages[i] = simulate_average_voltage(
            i_dc=float(cur),
            i_ac=i_ac,
            omega=omega,
            cfg=cfg,
        )
    return voltages


def _contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive index ranges where mask is True and contiguous."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    split_points = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, split_points + 1]
    ends = np.r_[split_points, idx.size - 1]
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def detect_shapiro_steps(
    currents: np.ndarray,
    voltages: np.ndarray,
    omega: float,
    max_step_index: int = 4,
    voltage_tolerance: float = 0.08,
    slope_tolerance: float = 0.15,
    min_points: int = 3,
) -> list[dict[str, float]]:
    """Detect plateaus near V = n*omega with low local slope."""
    if currents.shape != voltages.shape:
        raise ValueError("currents and voltages must have the same shape")

    slope = np.gradient(voltages, currents)
    detected: list[dict[str, float]] = []

    for n in range(max_step_index + 1):
        target_v = n * omega
        near_target = np.abs(voltages - target_v) <= voltage_tolerance
        nearly_flat = np.abs(slope) <= slope_tolerance
        mask = near_target & nearly_flat

        for left, right in _contiguous_true_segments(mask):
            count = right - left + 1
            if count < min_points:
                continue

            i_left = float(currents[left])
            i_right = float(currents[right])
            v_mean = float(np.mean(voltages[left : right + 1]))
            dv_di_mean = float(np.mean(np.abs(slope[left : right + 1])))

            detected.append(
                {
                    "n": float(n),
                    "target_v": float(target_v),
                    "i_left": i_left,
                    "i_right": i_right,
                    "v_mean": v_mean,
                    "mean_abs_dv_di": dv_di_mean,
                    "points": float(count),
                }
            )

    return detected


def summarize_curve_samples(
    currents: np.ndarray,
    voltages: np.ndarray,
    sample_count: int = 8,
) -> list[tuple[float, float]]:
    """Pick a small set of representative (I, V) pairs for terminal output."""
    sample_count = max(2, sample_count)
    idx = np.linspace(0, len(currents) - 1, sample_count, dtype=int)
    return [(float(currents[i]), float(voltages[i])) for i in idx]


def estimate_switching_current(
    currents: np.ndarray,
    voltages: np.ndarray,
    threshold: float = 0.05,
) -> float:
    """Estimate Ic as first current where average voltage exceeds threshold."""
    above = np.flatnonzero(voltages > threshold)
    if above.size == 0:
        return float("nan")
    return float(currents[int(above[0])])


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    cfg = RSJConfig(dt=0.05, n_steps=12_000, burn_in_steps=3_000)
    currents = np.linspace(0.0, 2.5, 121)

    # Case A: pure DC bias (i_ac = 0)
    v_dc = sweep_iv_curve(currents, i_ac=0.0, omega=0.0, cfg=cfg)

    # Case B: DC + AC drive, used to expose Shapiro steps
    ac_amplitude = 1.2
    ac_frequency = 0.5
    v_ac = sweep_iv_curve(currents, i_ac=ac_amplitude, omega=ac_frequency, cfg=cfg)

    steps = detect_shapiro_steps(
        currents=currents,
        voltages=v_ac,
        omega=ac_frequency,
        max_step_index=4,
        voltage_tolerance=0.08,
        slope_tolerance=0.15,
        min_points=3,
    )

    ic_est = estimate_switching_current(currents, v_dc, threshold=0.05)

    print("=== Josephson Effect MVP (Overdamped RSJ) ===")
    print(f"grid_points={len(currents)}, dt={cfg.dt}, steps={cfg.n_steps}, burn_in={cfg.burn_in_steps}")
    print(f"AC drive: i_ac={ac_amplitude:.3f}, omega={ac_frequency:.3f}")
    print()

    print("[DC only] representative I-V samples:")
    for cur, vol in summarize_curve_samples(currents, v_dc):
        print(f"  I={cur:>6.3f}, <V>={vol:>8.5f}")
    print(f"Estimated switching current Ic (threshold 0.05): {ic_est:.4f}")
    print()

    print("[DC + AC] representative I-V samples:")
    for cur, vol in summarize_curve_samples(currents, v_ac):
        print(f"  I={cur:>6.3f}, <V>={vol:>8.5f}")
    print()

    print("Detected Shapiro steps (plateaus near V = n*omega):")
    if not steps:
        print("  None detected with current thresholds.")
    else:
        for step in steps:
            print(
                "  "
                f"n={int(step['n'])}, target={step['target_v']:.3f}, "
                f"I in [{step['i_left']:.3f}, {step['i_right']:.3f}], "
                f"mean V={step['v_mean']:.4f}, "
                f"mean |dV/dI|={step['mean_abs_dv_di']:.4f}, "
                f"points={int(step['points'])}"
            )


if __name__ == "__main__":
    main()
