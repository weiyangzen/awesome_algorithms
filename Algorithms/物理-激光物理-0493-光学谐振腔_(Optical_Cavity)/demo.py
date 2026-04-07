"""Minimal MVP for Optical Cavity (two-mirror Fabry-Perot).

Implemented features:
- Round-trip ABCD matrix for two-mirror cavity
- Stability check by g-parameters and matrix trace
- Self-consistent Gaussian mode q-parameter
- FSR/finesse/linewidth estimates
- Airy transmission spectrum + FWHM extraction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

LIGHT_SPEED = 299_792_458.0  # m/s


@dataclass(frozen=True)
class CavityConfig:
    case_name: str
    length_m: float
    wavelength_m: float
    mirror1_radius_m: float
    mirror2_radius_m: float
    reflectivity1: float
    reflectivity2: float
    spectrum_points: int = 20_001
    spectrum_span_fsr: float = 1.2


@dataclass(frozen=True)
class CavitySummary:
    case_name: str
    stable: bool
    g1: float
    g2: float
    g_product: float
    trace_half: float
    fsr_hz: float
    finesse: float
    linewidth_hz: float
    transmission_peak: float
    transmission_fwhm_hz: float
    waist_radius_m: float
    waist_position_from_m1_m: float
    spot_radius_m1_m: float
    mode_index: int


def mirror_matrix(radius_m: float) -> np.ndarray:
    """ABCD matrix of a spherical mirror reflection."""
    if math.isinf(radius_m):
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    return np.array([[1.0, 0.0], [-2.0 / radius_m, 1.0]], dtype=float)


def propagation_matrix(length_m: float) -> np.ndarray:
    """ABCD matrix of free-space propagation."""
    return np.array([[1.0, length_m], [0.0, 1.0]], dtype=float)


def roundtrip_abcd(cfg: CavityConfig) -> np.ndarray:
    """One full round trip from mirror-1 plane back to mirror-1 plane."""
    m1 = mirror_matrix(cfg.mirror1_radius_m)
    m2 = mirror_matrix(cfg.mirror2_radius_m)
    p = propagation_matrix(cfg.length_m)
    return m1 @ p @ m2 @ p


def g_parameter(length_m: float, radius_m: float) -> float:
    """Compute g = 1 - L/R, with g=1 for planar mirror (R=inf)."""
    if math.isinf(radius_m):
        return 1.0
    return 1.0 - length_m / radius_m


def compute_g_parameters(cfg: CavityConfig) -> tuple[float, float, float]:
    g1 = g_parameter(cfg.length_m, cfg.mirror1_radius_m)
    g2 = g_parameter(cfg.length_m, cfg.mirror2_radius_m)
    return g1, g2, g1 * g2


def solve_self_consistent_q(a: float, b: float, c: float, d: float) -> complex | None:
    """Solve q = (Aq+B)/(Cq+D) and return physical root with Im(q)>0."""
    if abs(c) < 1e-14:
        return None

    coeffs = np.array([c, d - a, -b], dtype=complex)
    roots = np.roots(coeffs)
    candidates = [root for root in roots if root.imag > 1e-14]
    if not candidates:
        return None
    return max(candidates, key=lambda z: z.imag)


def gaussian_beam_metrics_from_q(q: complex, wavelength_m: float) -> tuple[float, float, float]:
    """Convert q-parameter into spot radius, waist radius and waist location."""
    inv_q = 1.0 / q
    if inv_q.imag >= 0.0:
        raise ValueError("Unphysical q: Im(1/q) must be negative for finite beam radius.")

    spot_radius_m = math.sqrt(-wavelength_m / (math.pi * inv_q.imag))
    z_rayleigh = q.imag
    waist_radius_m = math.sqrt(wavelength_m * z_rayleigh / math.pi)
    waist_position_from_m1_m = -q.real
    return spot_radius_m, waist_radius_m, waist_position_from_m1_m


def estimate_fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate full width at half maximum with linear interpolation."""
    peak_idx = int(np.argmax(y))
    peak = float(y[peak_idx])
    if peak_idx == 0 or peak_idx == len(y) - 1:
        return float("nan")

    half = 0.5 * peak
    left_candidates = np.where(y[:peak_idx] <= half)[0]
    right_candidates = np.where(y[peak_idx + 1 :] <= half)[0]
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return float("nan")

    left_idx = int(left_candidates[-1])
    right_idx = int(peak_idx + 1 + right_candidates[0])

    left_x1, left_x2 = float(x[left_idx]), float(x[left_idx + 1])
    left_y1, left_y2 = float(y[left_idx]), float(y[left_idx + 1])
    if abs(left_y2 - left_y1) < 1e-15:
        left_cross = left_x1
    else:
        left_cross = left_x1 + (half - left_y1) * (left_x2 - left_x1) / (left_y2 - left_y1)

    right_x1, right_x2 = float(x[right_idx - 1]), float(x[right_idx])
    right_y1, right_y2 = float(y[right_idx - 1]), float(y[right_idx])
    if abs(right_y2 - right_y1) < 1e-15:
        right_cross = right_x2
    else:
        right_cross = right_x1 + (half - right_y1) * (right_x2 - right_x1) / (right_y2 - right_y1)

    return right_cross - left_cross


def transmission_spectrum(
    cfg: CavityConfig,
    fsr_hz: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Generate Airy transmission around a nearest longitudinal resonance."""
    optical_freq = LIGHT_SPEED / cfg.wavelength_m
    mode_index = int(round(optical_freq / fsr_hz))
    mode_freq = mode_index * fsr_hz

    span_hz = cfg.spectrum_span_fsr * fsr_hz
    freqs = np.linspace(mode_freq - span_hz, mode_freq + span_hz, cfg.spectrum_points)

    sqrt_r_prod = math.sqrt(cfg.reflectivity1 * cfg.reflectivity2)
    phase = 4.0 * math.pi * cfg.length_m * freqs / LIGHT_SPEED

    numerator = (1.0 - cfg.reflectivity1) * (1.0 - cfg.reflectivity2)
    denominator = (1.0 - sqrt_r_prod) ** 2 + 4.0 * sqrt_r_prod * np.sin(0.5 * phase) ** 2
    transmission = numerator / denominator
    return freqs, transmission, float(np.max(transmission)), mode_index


def analyze_cavity(cfg: CavityConfig) -> CavitySummary:
    """Compute stability, Gaussian mode, and spectrum-derived metrics."""
    mat = roundtrip_abcd(cfg)
    a, b, c, d = (float(mat[0, 0]), float(mat[0, 1]), float(mat[1, 0]), float(mat[1, 1]))

    g1, g2, g_product = compute_g_parameters(cfg)
    trace_half = 0.5 * (a + d)
    stable = (0.0 < g_product < 1.0) and (abs(trace_half) < 1.0)

    fsr_hz = LIGHT_SPEED / (2.0 * cfg.length_m)
    finesse = math.pi * (cfg.reflectivity1 * cfg.reflectivity2) ** 0.25 / (
        1.0 - math.sqrt(cfg.reflectivity1 * cfg.reflectivity2)
    )
    linewidth_hz = fsr_hz / finesse

    freqs, transmission, t_peak, mode_index = transmission_spectrum(cfg, fsr_hz)
    fwhm_hz = estimate_fwhm(freqs, transmission)

    if stable:
        q = solve_self_consistent_q(a, b, c, d)
        if q is None:
            raise RuntimeError("Stable cavity should have a physical q root, but none was found.")
        spot_m1, waist_radius, waist_pos = gaussian_beam_metrics_from_q(q, cfg.wavelength_m)
    else:
        spot_m1 = float("nan")
        waist_radius = float("nan")
        waist_pos = float("nan")

    return CavitySummary(
        case_name=cfg.case_name,
        stable=stable,
        g1=g1,
        g2=g2,
        g_product=g_product,
        trace_half=trace_half,
        fsr_hz=fsr_hz,
        finesse=finesse,
        linewidth_hz=linewidth_hz,
        transmission_peak=t_peak,
        transmission_fwhm_hz=fwhm_hz,
        waist_radius_m=waist_radius,
        waist_position_from_m1_m=waist_pos,
        spot_radius_m1_m=spot_m1,
        mode_index=mode_index,
    )


def print_summary_table(summaries: list[CavitySummary]) -> None:
    rows = []
    for s in summaries:
        rows.append(
            {
                "case": s.case_name,
                "stable": s.stable,
                "g1": s.g1,
                "g2": s.g2,
                "g_product": s.g_product,
                "trace_half": s.trace_half,
                "fsr_MHz": s.fsr_hz / 1e6,
                "finesse": s.finesse,
                "linewidth_kHz": s.linewidth_hz / 1e3,
                "transmission_peak": s.transmission_peak,
                "fwhm_kHz": s.transmission_fwhm_hz / 1e3,
                "waist_radius_um": s.waist_radius_m * 1e6,
                "waist_position_cm": s.waist_position_from_m1_m * 100.0,
                "spot_mirror1_um": s.spot_radius_m1_m * 1e6,
                "mode_index": s.mode_index,
            }
        )
    table = pd.DataFrame(rows)
    with pd.option_context("display.precision", 6, "display.width", 180):
        print(table.to_string(index=False))


def main() -> None:
    print("Optical cavity MVP: two-mirror Fabry-Perot (ABCD + q + Airy spectrum)")

    stable_cfg = CavityConfig(
        case_name="stable_reference",
        length_m=0.25,
        wavelength_m=1064e-9,
        mirror1_radius_m=0.30,
        mirror2_radius_m=0.40,
        reflectivity1=0.985,
        reflectivity2=0.990,
    )
    unstable_cfg = CavityConfig(
        case_name="unstable_reference",
        length_m=0.80,
        wavelength_m=1064e-9,
        mirror1_radius_m=0.30,
        mirror2_radius_m=0.40,
        reflectivity1=0.985,
        reflectivity2=0.990,
    )

    summaries = [analyze_cavity(stable_cfg), analyze_cavity(unstable_cfg)]
    print_summary_table(summaries)

    stable_summary = next(s for s in summaries if s.case_name == "stable_reference")
    unstable_summary = next(s for s in summaries if s.case_name == "unstable_reference")

    assert stable_summary.stable
    assert not unstable_summary.stable

    assert math.isfinite(stable_summary.waist_radius_m) and stable_summary.waist_radius_m > 0.0
    assert math.isfinite(stable_summary.spot_radius_m1_m) and stable_summary.spot_radius_m1_m > 0.0

    assert 0.0 < stable_summary.transmission_peak <= 1.0 + 1e-9
    assert math.isfinite(stable_summary.transmission_fwhm_hz) and stable_summary.transmission_fwhm_hz > 0.0

    linewidth_rel_error = abs(
        stable_summary.transmission_fwhm_hz - stable_summary.linewidth_hz
    ) / stable_summary.linewidth_hz
    assert linewidth_rel_error < 0.08

    print(
        "Checks passed: stability classification and linewidth consistency "
        f"(relative error={linewidth_rel_error:.4%})."
    )


if __name__ == "__main__":
    main()
