"""Minimal runnable MVP for mode locking (PHYS-0472)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


C_LIGHT = 299_792_458.0  # m/s


@dataclass(frozen=True)
class CaseMetrics:
    """Summary metrics for one phase configuration."""

    peak: float
    mean: float
    par: float
    fwhm_s: float


def cavity_mode_spacing(length_m: float, refractive_index: float) -> float:
    """Return longitudinal mode spacing Δf = c / (2nL)."""
    return C_LIGHT / (2.0 * refractive_index * length_m)


def build_modes(num_modes: int, sigma_ratio: float = 0.22) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetric mode indices and a Gaussian spectral amplitude envelope."""
    if num_modes < 3:
        raise ValueError("num_modes must be >= 3 for a meaningful multi-mode example.")
    if num_modes % 2 == 0:
        raise ValueError("num_modes must be odd to keep symmetric indices around 0.")

    half = num_modes // 2
    mode_indices = np.arange(-half, half + 1, dtype=float)
    sigma = max(1.0, sigma_ratio * num_modes)
    amplitudes = np.exp(-0.5 * (mode_indices / sigma) ** 2)
    amplitudes /= np.linalg.norm(amplitudes)
    return mode_indices, amplitudes


def synthesize_envelope(
    mode_indices: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    delta_f_hz: float,
    time_s: np.ndarray,
) -> np.ndarray:
    """Synthesize complex envelope E(t) = sum_m A_m exp(i(2π mΔf t + φ_m))."""
    if mode_indices.shape != amplitudes.shape or phases.shape != amplitudes.shape:
        raise ValueError("mode_indices, amplitudes, and phases must have the same shape.")

    phase_matrix = 2.0 * np.pi * np.outer(mode_indices * delta_f_hz, time_s)
    phase_matrix += phases[:, None]
    field = np.sum(amplitudes[:, None] * np.exp(1j * phase_matrix), axis=0)
    return field


def intensity_from_field(field: np.ndarray) -> np.ndarray:
    """Compute intensity and normalize to unit mean for fair comparison."""
    intensity = np.abs(field) ** 2
    mean_intensity = float(np.mean(intensity))
    if mean_intensity <= 0.0:
        raise ValueError("Mean intensity must be positive.")
    return intensity / mean_intensity


def estimate_fwhm(time_s: np.ndarray, intensity: np.ndarray) -> float:
    """Estimate full width at half maximum near the global peak (seconds)."""
    if len(time_s) < 3 or len(intensity) < 3:
        return float("nan")

    dt = float(np.mean(np.diff(time_s)))
    peak_idx = int(np.argmax(intensity))
    center = len(intensity) // 2

    # Pulse trains are periodic; roll so the target peak is away from boundaries.
    working = np.roll(intensity, center - peak_idx)
    half_level = 0.5 * float(working[center])

    left = center
    while left > 0 and working[left] >= half_level:
        left -= 1

    right = center
    last = len(working) - 1
    while right < last and working[right] >= half_level:
        right += 1

    if left == 0 or right == last:
        return float("nan")

    def linear_cross(x0: float, y0: float, x1: float, y1: float, y: float) -> float:
        if y1 == y0:
            return x0
        return x0 + (y - y0) * (x1 - x0) / (y1 - y0)

    left_x = linear_cross(
        float(left),
        float(working[left]),
        float(left + 1),
        float(working[left + 1]),
        half_level,
    )
    right_x = linear_cross(
        float(right - 1),
        float(working[right - 1]),
        float(right),
        float(working[right]),
        half_level,
    )
    return (right_x - left_x) * dt


def summarize_case(time_s: np.ndarray, intensity: np.ndarray) -> CaseMetrics:
    """Compute compact scalar indicators for one simulated case."""
    peak = float(np.max(intensity))
    mean = float(np.mean(intensity))
    par = peak / mean
    fwhm_s = estimate_fwhm(time_s, intensity)
    return CaseMetrics(peak=peak, mean=mean, par=par, fwhm_s=fwhm_s)


def save_csv(path: Path, time_s: np.ndarray, i_random: np.ndarray, i_locked: np.ndarray) -> None:
    """Save simulation time series as CSV."""
    data = np.column_stack((time_s, i_random, i_locked))
    header = "time_s,intensity_random_phase,intensity_locked_phase"
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.10e")


def main() -> None:
    # Baseline cavity and simulation settings.
    length_m = 1.2
    refractive_index = 1.5
    num_modes = 81
    periods_to_show = 6
    samples_per_period = 1200

    delta_f_hz = cavity_mode_spacing(length_m, refractive_index)
    repetition_period_s = 1.0 / delta_f_hz

    mode_indices, amplitudes = build_modes(num_modes=num_modes)
    num_samples = periods_to_show * samples_per_period
    time_s = np.linspace(0.0, periods_to_show * repetition_period_s, num_samples, endpoint=False)

    rng = np.random.default_rng(47)
    phases_random = rng.uniform(0.0, 2.0 * np.pi, size=amplitudes.shape)
    phases_locked = np.zeros_like(amplitudes)

    field_random = synthesize_envelope(mode_indices, amplitudes, phases_random, delta_f_hz, time_s)
    field_locked = synthesize_envelope(mode_indices, amplitudes, phases_locked, delta_f_hz, time_s)

    intensity_random = intensity_from_field(field_random)
    intensity_locked = intensity_from_field(field_locked)

    metrics_random = summarize_case(time_s, intensity_random)
    metrics_locked = summarize_case(time_s, intensity_locked)

    csv_path = Path(__file__).with_name("mode_locking_timeseries.csv")
    save_csv(csv_path, time_s, intensity_random, intensity_locked)

    print("Mode Locking MVP (PHYS-0472)")
    print(f"Cavity length L = {length_m:.3f} m, refractive index n = {refractive_index:.3f}")
    print(f"Mode spacing Δf = {delta_f_hz:.3e} Hz")
    print(f"Pulse repetition period T_rep = {repetition_period_s * 1e9:.3f} ns")
    print(f"Number of modes = {num_modes}")
    print("")
    print("Case: random phase (unlocked)")
    print(f"  peak intensity      = {metrics_random.peak:.3f}")
    print(f"  mean intensity      = {metrics_random.mean:.3f}")
    print(f"  PAR (peak/mean)     = {metrics_random.par:.3f}")
    if np.isnan(metrics_random.fwhm_s):
        print("  FWHM                = NaN (half-maximum crossing not found)")
    else:
        print(f"  FWHM                = {metrics_random.fwhm_s * 1e12:.3f} ps")

    print("")
    print("Case: locked phase (mode-locked)")
    print(f"  peak intensity      = {metrics_locked.peak:.3f}")
    print(f"  mean intensity      = {metrics_locked.mean:.3f}")
    print(f"  PAR (peak/mean)     = {metrics_locked.par:.3f}")
    if np.isnan(metrics_locked.fwhm_s):
        print("  FWHM                = NaN (half-maximum crossing not found)")
    else:
        print(f"  FWHM                = {metrics_locked.fwhm_s * 1e12:.3f} ps")

    print("")
    if not np.isnan(metrics_locked.fwhm_s):
        duty_cycle = metrics_locked.fwhm_s / repetition_period_s
        print(f"Locked-case duty cycle ≈ {duty_cycle * 100:.3f}%")
    print(f"CSV written to: {csv_path.name}")


if __name__ == "__main__":
    main()
