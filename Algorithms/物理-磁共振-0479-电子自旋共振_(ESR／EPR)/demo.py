"""Minimal runnable MVP for Electron Spin Resonance (ESR/EPR).

This script simulates a CW-EPR first-derivative spectrum in magnetic-field
sweep mode, detects resonance lines, and estimates:
- resonance field B0 (mT)
- g factor via h*nu = g*mu_B*B0
- linewidth from derivative peak-to-peak spacing
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks as scipy_find_peaks
except Exception:  # pragma: no cover - fallback for limited environments
    scipy_find_peaks = None


PLANCK_CONSTANT_J_S = 6.62607015e-34
BOHR_MAGNETON_J_T = 9.2740100783e-24


@dataclass(frozen=True)
class ESRComponent:
    """Ground-truth ESR component for synthetic spectrum generation."""

    name: str
    amplitude: float
    g_factor: float
    hwhm_mt: float


def resonance_field_mt(microwave_freq_ghz: float, g_factor: float) -> float:
    """Convert g-factor to resonance field B0 in mT using h*nu=g*mu_B*B."""
    freq_hz = microwave_freq_ghz * 1e9
    b_tesla = PLANCK_CONSTANT_J_S * freq_hz / (g_factor * BOHR_MAGNETON_J_T)
    return 1e3 * b_tesla


def estimate_g_factor(microwave_freq_ghz: float, field_mt: float) -> float:
    """Estimate g-factor from resonance field in mT."""
    freq_hz = microwave_freq_ghz * 1e9
    b_tesla = field_mt * 1e-3
    if b_tesla <= 0:
        return float("nan")
    return PLANCK_CONSTANT_J_S * freq_hz / (BOHR_MAGNETON_J_T * b_tesla)


def lorentzian_derivative(
    field_mt: np.ndarray,
    center_mt: float,
    hwhm_mt: float,
    amplitude: float,
) -> np.ndarray:
    """First derivative of Lorentzian absorption with respect to magnetic field."""
    x = (field_mt - center_mt) / hwhm_mt
    return amplitude * (-2.0 * x) / (hwhm_mt * (1.0 + x * x) ** 2)


def generate_synthetic_cw_esr(
    field_mt: np.ndarray,
    components: list[ESRComponent],
    microwave_freq_ghz: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate clean/noisy first-derivative CW-EPR signal and truth table."""
    rng = np.random.default_rng(seed)
    clean = np.zeros_like(field_mt, dtype=np.float64)

    truth_rows: list[dict[str, float | str]] = []
    for comp in components:
        center = resonance_field_mt(microwave_freq_ghz, comp.g_factor)
        clean += lorentzian_derivative(
            field_mt=field_mt,
            center_mt=center,
            hwhm_mt=comp.hwhm_mt,
            amplitude=comp.amplitude,
        )
        truth_rows.append(
            {
                "component": comp.name,
                "amplitude": comp.amplitude,
                "g_true": comp.g_factor,
                "center_field_mt_true": center,
                "hwhm_mt_true": comp.hwhm_mt,
                "linewidth_fwhm_mt_true": 2.0 * comp.hwhm_mt,
                "linewidth_pp_mt_true": 2.0 * comp.hwhm_mt / np.sqrt(3.0),
            }
        )

    noisy = clean + rng.normal(loc=0.0, scale=noise_std, size=field_mt.size)
    truth_df = pd.DataFrame(truth_rows).sort_values("center_field_mt_true").reset_index(drop=True)
    return noisy, clean, truth_df


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Simple smoothing filter for robust peak/trough detection."""
    if window <= 1:
        return signal.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(signal, kernel, mode="same")


def _fallback_find_peaks(signal: np.ndarray, min_height: float, min_distance_bins: int) -> np.ndarray:
    """Local-max detector used when scipy is unavailable."""
    candidates: list[int] = []
    for i in range(1, signal.size - 1):
        if signal[i] >= min_height and signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            if candidates and i - candidates[-1] < min_distance_bins:
                if signal[i] > signal[candidates[-1]]:
                    candidates[-1] = i
            else:
                candidates.append(i)
    return np.array(candidates, dtype=np.int64)


def find_local_extrema(
    signal: np.ndarray,
    min_separation_bins: int,
    prominence_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find local maxima and minima from one-dimensional signal."""
    dynamic_scale = float(np.max(np.abs(signal)))
    min_height = max(prominence_ratio * dynamic_scale, 1e-12)

    if scipy_find_peaks is not None:
        peak_idx, _ = scipy_find_peaks(
            signal,
            height=min_height,
            distance=max(1, min_separation_bins),
            prominence=min_height * 0.5,
        )
        trough_idx, _ = scipy_find_peaks(
            -signal,
            height=min_height,
            distance=max(1, min_separation_bins),
            prominence=min_height * 0.5,
        )
    else:
        peak_idx = _fallback_find_peaks(signal, min_height=min_height, min_distance_bins=min_separation_bins)
        trough_idx = _fallback_find_peaks(-signal, min_height=min_height, min_distance_bins=min_separation_bins)

    if peak_idx.size == 0:
        peak_idx = np.array([int(np.argmax(signal))], dtype=np.int64)
    if trough_idx.size == 0:
        trough_idx = np.array([int(np.argmin(signal))], dtype=np.int64)

    return peak_idx, trough_idx


def _interp_zero_cross(x1: float, y1: float, x2: float, y2: float) -> float:
    """Linear interpolation for y=0 crossing between two adjacent samples."""
    if y2 == y1:
        return x1
    alpha = -y1 / (y2 - y1)
    return x1 + alpha * (x2 - x1)


def detect_esr_lines(
    field_mt: np.ndarray,
    derivative_signal: np.ndarray,
    microwave_freq_ghz: float,
    max_lines: int,
    min_separation_mt: float,
    prominence_ratio: float,
    search_window_mt: float,
    smoothing_window: int,
) -> pd.DataFrame:
    """Detect ESR lines from derivative spectrum and estimate line parameters."""
    if max_lines < 1:
        raise ValueError("max_lines must be >= 1")

    smooth = moving_average(derivative_signal, window=smoothing_window)
    field_step_mt = float(np.abs(field_mt[1] - field_mt[0]))
    min_sep_bins = max(1, int(round(min_separation_mt / max(field_step_mt, 1e-12))))
    search_bins = max(2, int(round(search_window_mt / max(field_step_mt, 1e-12))))

    peak_idx, trough_idx = find_local_extrema(
        signal=smooth,
        min_separation_bins=min_sep_bins,
        prominence_ratio=prominence_ratio,
    )

    # For Lorentzian derivative (with positive amplitude), resonance center has sign change + -> -.
    crossing_anchor = np.where((smooth[:-1] > 0.0) & (smooth[1:] <= 0.0))[0]

    candidates: list[dict[str, float | int]] = []
    for i in crossing_anchor:
        left_candidates = peak_idx[(peak_idx <= i) & (peak_idx >= i - search_bins)]
        right_candidates = trough_idx[(trough_idx >= i + 1) & (trough_idx <= i + search_bins)]
        if left_candidates.size == 0 or right_candidates.size == 0:
            continue

        # Use nearest extrema around the crossing to avoid cross-line pairing.
        left_idx = int(left_candidates[-1])
        right_idx = int(right_candidates[0])
        if right_idx <= left_idx:
            continue

        center_field = _interp_zero_cross(
            float(field_mt[i]),
            float(smooth[i]),
            float(field_mt[i + 1]),
            float(smooth[i + 1]),
        )
        linewidth_pp_mt = float(field_mt[right_idx] - field_mt[left_idx])
        if linewidth_pp_mt <= 0:
            continue
        linewidth_fwhm_mt = float(np.sqrt(3.0) * linewidth_pp_mt)

        score = float(abs(smooth[left_idx]) + abs(smooth[right_idx]))
        candidates.append(
            {
                "center_field_mt": center_field,
                "g_estimate": estimate_g_factor(microwave_freq_ghz, center_field),
                "linewidth_pp_mt": linewidth_pp_mt,
                "linewidth_fwhm_mt": linewidth_fwhm_mt,
                "left_peak_field_mt": float(field_mt[left_idx]),
                "right_trough_field_mt": float(field_mt[right_idx]),
                "peak_to_peak_signal": float(smooth[left_idx] - smooth[right_idx]),
                "score": score,
            }
        )

    if not candidates:
        return pd.DataFrame(
            columns=[
                "rank",
                "center_field_mt",
                "g_estimate",
                "linewidth_pp_mt",
                "linewidth_fwhm_mt",
                "left_peak_field_mt",
                "right_trough_field_mt",
                "peak_to_peak_signal",
                "score",
            ]
        )

    df = pd.DataFrame(candidates).sort_values("score", ascending=False).reset_index(drop=True)

    selected_rows: list[pd.Series] = []
    for _, row in df.iterrows():
        if len(selected_rows) >= max_lines:
            break
        center = float(row["center_field_mt"])
        too_close = any(abs(center - float(existing["center_field_mt"])) < min_separation_mt for existing in selected_rows)
        if not too_close:
            selected_rows.append(row)

    selected = pd.DataFrame(selected_rows)
    selected = selected.sort_values("center_field_mt").reset_index(drop=True)
    selected.insert(0, "rank", np.arange(1, selected.shape[0] + 1, dtype=int))
    return selected


def build_comparison_table(truth_df: pd.DataFrame, est_df: pd.DataFrame) -> pd.DataFrame:
    """Match each true ESR component with nearest unmatched estimated line."""
    remaining = est_df.copy()
    rows: list[dict[str, float | str]] = []

    for _, truth in truth_df.iterrows():
        if remaining.empty:
            rows.append(
                {
                    "component": truth["component"],
                    "field_true_mt": truth["center_field_mt_true"],
                    "field_est_mt": np.nan,
                    "field_abs_error_mt": np.nan,
                    "g_true": truth["g_true"],
                    "g_est": np.nan,
                    "g_abs_error": np.nan,
                    "g_error_ppm": np.nan,
                    "fwhm_true_mt": truth["linewidth_fwhm_mt_true"],
                    "fwhm_est_mt": np.nan,
                    "fwhm_rel_error_pct": np.nan,
                }
            )
            continue

        errors = np.abs(remaining["center_field_mt"].to_numpy() - truth["center_field_mt_true"])
        best_local_idx = int(np.argmin(errors))
        best = remaining.iloc[best_local_idx]

        g_abs_error = abs(float(best["g_estimate"]) - float(truth["g_true"]))
        g_error_ppm = 1e6 * g_abs_error / max(float(truth["g_true"]), 1e-12)

        fwhm_rel_error_pct = 100.0 * abs(float(best["linewidth_fwhm_mt"]) - float(truth["linewidth_fwhm_mt_true"]))
        fwhm_rel_error_pct /= max(float(truth["linewidth_fwhm_mt_true"]), 1e-12)

        rows.append(
            {
                "component": truth["component"],
                "field_true_mt": float(truth["center_field_mt_true"]),
                "field_est_mt": float(best["center_field_mt"]),
                "field_abs_error_mt": float(abs(best["center_field_mt"] - truth["center_field_mt_true"])),
                "g_true": float(truth["g_true"]),
                "g_est": float(best["g_estimate"]),
                "g_abs_error": float(g_abs_error),
                "g_error_ppm": float(g_error_ppm),
                "fwhm_true_mt": float(truth["linewidth_fwhm_mt_true"]),
                "fwhm_est_mt": float(best["linewidth_fwhm_mt"]),
                "fwhm_rel_error_pct": float(fwhm_rel_error_pct),
            }
        )

        remaining = remaining.drop(index=best.name)

    return pd.DataFrame(rows)


def estimate_snr(signal: np.ndarray, exclusion_center_idx: int, exclusion_half_width: int = 50) -> tuple[float, float]:
    """Estimate peak SNR (linear + dB) using robust median absolute signal floor."""
    amplitude = np.abs(signal)

    left = max(0, exclusion_center_idx - exclusion_half_width)
    right = min(signal.size, exclusion_center_idx + exclusion_half_width + 1)

    mask = np.ones(signal.size, dtype=bool)
    mask[left:right] = False
    noise_floor = np.median(amplitude[mask])

    if noise_floor <= 0:
        return float("inf"), float("inf")

    snr_linear = float(amplitude[exclusion_center_idx] / noise_floor)
    snr_db = 20.0 * np.log10(max(snr_linear, 1e-12))
    return snr_linear, float(snr_db)


def main() -> None:
    seed = 479
    microwave_freq_ghz = 9.50
    field_mt = np.linspace(332.0, 347.0, 4096)

    components = [
        ESRComponent(name="radical_A", amplitude=1.00, g_factor=2.0035, hwhm_mt=0.19),
        ESRComponent(name="radical_B", amplitude=0.72, g_factor=2.0082, hwhm_mt=0.26),
        ESRComponent(name="radical_C", amplitude=0.55, g_factor=1.9986, hwhm_mt=0.17),
    ]

    noise_std = 0.035
    derivative_noisy, derivative_clean, truth_df = generate_synthetic_cw_esr(
        field_mt=field_mt,
        components=components,
        microwave_freq_ghz=microwave_freq_ghz,
        noise_std=noise_std,
        seed=seed,
    )

    est_df = detect_esr_lines(
        field_mt=field_mt,
        derivative_signal=derivative_noisy,
        microwave_freq_ghz=microwave_freq_ghz,
        max_lines=3,
        min_separation_mt=0.45,
        prominence_ratio=0.11,
        search_window_mt=0.75,
        smoothing_window=11,
    )

    comparison_df = build_comparison_table(truth_df, est_df)

    dominant_idx = int(np.argmax(np.abs(derivative_noisy)))
    snr_linear, snr_db = estimate_snr(derivative_noisy, exclusion_center_idx=dominant_idx)

    mean_field_error = float(np.nanmean(comparison_df["field_abs_error_mt"]))
    mean_g_error_ppm = float(np.nanmean(comparison_df["g_error_ppm"]))
    mean_fwhm_rel_error = float(np.nanmean(comparison_df["fwhm_rel_error_pct"]))

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:0.6f}")

    print("=== Synthetic CW-ESR setup ===")
    print(f"seed={seed}, microwave_freq_ghz={microwave_freq_ghz:.3f}")
    print(f"field_range_mt=[{field_mt[0]:.2f}, {field_mt[-1]:.2f}], n_points={field_mt.size}")
    print(f"noise_std={noise_std:.4f}")
    print()

    print("=== Ground-truth ESR components ===")
    print(truth_df.to_string(index=False))
    print()

    print("=== Detected ESR lines ===")
    if est_df.empty:
        print("No line detected.")
    else:
        cols = [
            "rank",
            "center_field_mt",
            "g_estimate",
            "linewidth_pp_mt",
            "linewidth_fwhm_mt",
            "peak_to_peak_signal",
            "score",
        ]
        print(est_df[cols].to_string(index=False))
    print()

    print("=== Matching: truth vs estimate ===")
    print(comparison_df.to_string(index=False))
    print()

    print("=== Aggregate quality metrics ===")
    print(f"mean_abs_field_error_mt: {mean_field_error:.5f}")
    print(f"mean_g_error_ppm: {mean_g_error_ppm:.2f}")
    print(f"mean_fwhm_relative_error_pct: {mean_fwhm_rel_error:.2f}")
    print(f"dominant_abs_signal_snr_linear: {snr_linear:.2f}")
    print(f"dominant_abs_signal_snr_db: {snr_db:.2f}")


if __name__ == "__main__":
    main()
