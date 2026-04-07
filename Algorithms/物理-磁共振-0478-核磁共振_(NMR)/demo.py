"""Minimal runnable MVP for Nuclear Magnetic Resonance (NMR).

This script simulates a complex free induction decay (FID), computes an
NMR-like frequency spectrum via FFT, detects resonance peaks, and estimates
T2* for each peak from linewidth (FWHM) in the power spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks as scipy_find_peaks
except Exception:  # pragma: no cover - fallback for limited environments
    scipy_find_peaks = None


@dataclass(frozen=True)
class ResonanceComponent:
    """Ground-truth resonance component used for synthetic FID generation."""

    name: str
    amplitude: float
    frequency_hz: float
    t2_star_s: float
    phase_rad: float


def generate_synthetic_fid(
    time_s: np.ndarray,
    components: list[ResonanceComponent],
    noise_std: float,
    seed: int,
) -> np.ndarray:
    """Generate complex-valued FID from damped rotating exponentials."""
    rng = np.random.default_rng(seed)
    fid = np.zeros_like(time_s, dtype=np.complex128)

    for comp in components:
        envelope = np.exp(-time_s / comp.t2_star_s)
        phase = 2.0 * np.pi * comp.frequency_hz * time_s + comp.phase_rad
        fid += comp.amplitude * envelope * np.exp(1j * phase)

    noise = noise_std * (rng.normal(size=time_s.size) + 1j * rng.normal(size=time_s.size))
    return fid + noise


def apodize_and_fft(
    fid: np.ndarray,
    dt_s: float,
    line_broadening_hz: float,
    zero_fill_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exponential apodization and compute centered FFT spectrum."""
    n = fid.size
    if zero_fill_factor < 1:
        raise ValueError("zero_fill_factor must be >= 1")

    time_s = np.arange(n, dtype=np.float64) * dt_s
    window = np.exp(-np.pi * line_broadening_hz * time_s)
    windowed = fid * window

    n_fft_target = max(n * zero_fill_factor, n)
    n_fft = 1 << int(np.ceil(np.log2(n_fft_target)))

    spectrum = np.fft.fftshift(np.fft.fft(windowed, n=n_fft))
    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=dt_s))
    return freq_hz, spectrum


def _fallback_find_peaks(
    signal: np.ndarray,
    min_height: float,
    min_distance_bins: int,
) -> np.ndarray:
    """Simple local-max peak detector for environments without scipy."""
    candidates: list[int] = []
    for i in range(1, signal.size - 1):
        if signal[i] >= min_height and signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            if candidates and i - candidates[-1] < min_distance_bins:
                if signal[i] > signal[candidates[-1]]:
                    candidates[-1] = i
            else:
                candidates.append(i)
    return np.array(candidates, dtype=np.int64)


def detect_resonance_peaks(
    freq_hz: np.ndarray,
    spectrum: np.ndarray,
    max_peaks: int,
    min_separation_hz: float,
    prominence_ratio: float,
) -> pd.DataFrame:
    """Detect dominant peaks in magnitude spectrum and return sorted table."""
    if max_peaks < 1:
        raise ValueError("max_peaks must be >= 1")

    magnitude = np.abs(spectrum)
    if magnitude.size < 3:
        raise ValueError("spectrum size must be >= 3")

    freq_step = float(np.abs(freq_hz[1] - freq_hz[0]))
    min_distance_bins = max(1, int(round(min_separation_hz / max(freq_step, 1e-12))))
    min_height = float(prominence_ratio * np.max(magnitude))

    if scipy_find_peaks is not None:
        peak_idx, _ = scipy_find_peaks(
            magnitude,
            height=min_height,
            distance=min_distance_bins,
            prominence=min_height * 0.5,
        )
    else:
        peak_idx = _fallback_find_peaks(magnitude, min_height=min_height, min_distance_bins=min_distance_bins)

    if peak_idx.size == 0:
        peak_idx = np.array([int(np.argmax(magnitude))], dtype=np.int64)

    ranked = peak_idx[np.argsort(magnitude[peak_idx])[::-1]]
    ranked = ranked[:max_peaks]

    df = pd.DataFrame(
        {
            "rank": np.arange(1, ranked.size + 1, dtype=int),
            "peak_index": ranked,
            "frequency_hz": freq_hz[ranked],
            "magnitude": magnitude[ranked],
        }
    )
    return df.sort_values("rank", ascending=True).reset_index(drop=True)


def _interpolate_half_height_crossing(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    y_target: float,
) -> float:
    """Linear interpolation of x at y_target between two points."""
    if y2 == y1:
        return x1
    alpha = (y_target - y1) / (y2 - y1)
    return x1 + alpha * (x2 - x1)


def estimate_t2_star_from_linewidth(
    freq_hz: np.ndarray,
    spectrum: np.ndarray,
    peak_index: int,
    line_broadening_hz: float,
) -> float:
    """Estimate T2* from power-spectrum FWHM with line-broadening correction."""
    power = np.abs(spectrum) ** 2
    peak_power = float(power[peak_index])
    if peak_power <= 0:
        return float("nan")
    half_power = 0.5 * peak_power

    left = peak_index
    while left > 0 and power[left] >= half_power:
        left -= 1
    if left == 0 and power[left] >= half_power:
        return float("nan")

    right = peak_index
    while right < power.size - 1 and power[right] >= half_power:
        right += 1
    if right == power.size - 1 and power[right] >= half_power:
        return float("nan")

    left_cross = _interpolate_half_height_crossing(
        float(freq_hz[left]),
        float(power[left]),
        float(freq_hz[left + 1]),
        float(power[left + 1]),
        half_power,
    )
    right_cross = _interpolate_half_height_crossing(
        float(freq_hz[right - 1]),
        float(power[right - 1]),
        float(freq_hz[right]),
        float(power[right]),
        half_power,
    )

    fwhm_hz = max(right_cross - left_cross, 0.0)
    intrinsic_width_hz = fwhm_hz - line_broadening_hz
    if intrinsic_width_hz <= 1e-9:
        return float("nan")

    return 1.0 / (np.pi * intrinsic_width_hz)


def estimate_snr(spectrum: np.ndarray, dominant_index: int, exclusion_bins: int = 20) -> float:
    """Estimate spectrum SNR from dominant peak against robust noise floor."""
    magnitude = np.abs(spectrum)
    left = max(0, dominant_index - exclusion_bins)
    right = min(magnitude.size, dominant_index + exclusion_bins + 1)

    mask = np.ones(magnitude.size, dtype=bool)
    mask[left:right] = False
    noise_floor = np.median(magnitude[mask])
    if noise_floor <= 0:
        return float("inf")

    return float(magnitude[dominant_index] / noise_floor)


def build_comparison_table(
    truth: list[ResonanceComponent],
    estimated: pd.DataFrame,
) -> pd.DataFrame:
    """Match each true peak with nearest unmatched estimated peak by frequency."""
    remaining = estimated.copy()
    rows: list[dict[str, float | str]] = []

    for comp in truth:
        if remaining.empty:
            rows.append(
                {
                    "component": comp.name,
                    "true_freq_hz": comp.frequency_hz,
                    "est_freq_hz": np.nan,
                    "abs_freq_error_hz": np.nan,
                    "true_t2_star_s": comp.t2_star_s,
                    "est_t2_star_s": np.nan,
                    "t2_rel_error_pct": np.nan,
                }
            )
            continue

        freq_errors = np.abs(remaining["frequency_hz"].to_numpy() - comp.frequency_hz)
        best_local_idx = int(np.argmin(freq_errors))
        best_row = remaining.iloc[best_local_idx]
        est_t2 = float(best_row.get("t2_star_estimate_s", np.nan))

        t2_rel_error = np.nan
        if np.isfinite(est_t2) and comp.t2_star_s > 0:
            t2_rel_error = 100.0 * abs(est_t2 - comp.t2_star_s) / comp.t2_star_s

        rows.append(
            {
                "component": comp.name,
                "true_freq_hz": comp.frequency_hz,
                "est_freq_hz": float(best_row["frequency_hz"]),
                "abs_freq_error_hz": float(abs(best_row["frequency_hz"] - comp.frequency_hz)),
                "true_t2_star_s": comp.t2_star_s,
                "est_t2_star_s": est_t2,
                "t2_rel_error_pct": t2_rel_error,
            }
        )

        remaining = remaining.drop(index=best_row.name)

    return pd.DataFrame(rows)


def main() -> None:
    seed = 478
    sample_rate_hz = 2048.0
    n_points = 2048
    noise_std = 0.035

    dt_s = 1.0 / sample_rate_hz
    time_s = np.arange(n_points, dtype=np.float64) * dt_s

    true_components = [
        ResonanceComponent("peak_A", amplitude=1.00, frequency_hz=120.0, t2_star_s=0.300, phase_rad=0.10),
        ResonanceComponent("peak_B", amplitude=0.70, frequency_hz=-210.0, t2_star_s=0.140, phase_rad=0.85),
        ResonanceComponent("peak_C", amplitude=0.45, frequency_hz=410.0, t2_star_s=0.090, phase_rad=-0.50),
    ]

    fid = generate_synthetic_fid(time_s=time_s, components=true_components, noise_std=noise_std, seed=seed)

    line_broadening_hz = 1.5
    freq_hz, spectrum = apodize_and_fft(
        fid=fid,
        dt_s=dt_s,
        line_broadening_hz=line_broadening_hz,
        zero_fill_factor=4,
    )

    peak_df = detect_resonance_peaks(
        freq_hz=freq_hz,
        spectrum=spectrum,
        max_peaks=3,
        min_separation_hz=40.0,
        prominence_ratio=0.08,
    )

    t2_estimates: list[float] = []
    for peak_idx in peak_df["peak_index"].to_numpy():
        t2_estimates.append(
            estimate_t2_star_from_linewidth(
                freq_hz=freq_hz,
                spectrum=spectrum,
                peak_index=int(peak_idx),
                line_broadening_hz=line_broadening_hz,
            )
        )
    peak_df["t2_star_estimate_s"] = t2_estimates

    dominant_peak_idx = int(peak_df.iloc[0]["peak_index"])
    snr_linear = estimate_snr(spectrum=spectrum, dominant_index=dominant_peak_idx)
    snr_db = 20.0 * np.log10(max(snr_linear, 1e-12))

    comparison_df = build_comparison_table(true_components, peak_df)

    mean_freq_error = float(np.nanmean(comparison_df["abs_freq_error_hz"]))
    mean_t2_rel_error = float(np.nanmean(comparison_df["t2_rel_error_pct"]))

    truth_df = pd.DataFrame(
        {
            "component": [c.name for c in true_components],
            "amplitude": [c.amplitude for c in true_components],
            "frequency_hz": [c.frequency_hz for c in true_components],
            "t2_star_s": [c.t2_star_s for c in true_components],
            "linewidth_hz_approx": [1.0 / (np.pi * c.t2_star_s) for c in true_components],
        }
    )

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:0.6f}")

    print("=== Synthetic NMR FID setup ===")
    print(f"seed={seed}, sample_rate_hz={sample_rate_hz:.1f}, n_points={n_points}, duration_s={time_s[-1]:.3f}")
    print(f"noise_std={noise_std:.4f}, zero_fill_factor=4, line_broadening_hz={line_broadening_hz:.1f}")
    print()

    print("=== Ground-truth resonance components ===")
    print(truth_df.to_string(index=False))
    print()

    print("=== Detected spectrum peaks ===")
    print(peak_df[["rank", "frequency_hz", "magnitude", "t2_star_estimate_s"]].to_string(index=False))
    print()

    print("=== Matching: truth vs estimate ===")
    print(comparison_df.to_string(index=False))
    print()

    print("=== Aggregate quality metrics ===")
    print(f"mean_abs_frequency_error_hz: {mean_freq_error:.4f}")
    print(f"mean_t2_relative_error_pct: {mean_t2_rel_error:.2f}")
    print(f"dominant_peak_snr_linear: {snr_linear:.2f}")
    print(f"dominant_peak_snr_db: {snr_db:.2f}")


if __name__ == "__main__":
    main()
