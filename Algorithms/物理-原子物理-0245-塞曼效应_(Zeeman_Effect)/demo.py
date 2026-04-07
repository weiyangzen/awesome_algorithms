"""Minimal runnable MVP for Zeeman effect.

This script computes Zeeman splitting for two transitions:
1) Normal Zeeman example: 1P1 -> 1S0 (triplet splitting)
2) Anomalous Zeeman example: 2P3/2 -> 2S1/2

It validates linear-in-B behavior, expected component counts, and produces
compact tables plus synthetic broadened spectra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.constants import c, h, physical_constants
from scipy.signal import find_peaks

MU_B = physical_constants["Bohr magneton"][0]  # J/T


@dataclass(frozen=True)
class TermLevel:
    """Atomic term level in LS coupling."""

    label: str
    L: float
    S: float
    J: float


@dataclass(frozen=True)
class Transition:
    """Spectral transition with unperturbed center wavelength."""

    name: str
    upper: TermLevel
    lower: TermLevel
    lambda0_nm: float


def is_half_integer(x: float, tol: float = 1e-12) -> bool:
    return abs(2.0 * x - round(2.0 * x)) < tol


def validate_level(level: TermLevel) -> None:
    if level.L < 0 or level.S < 0 or level.J < 0:
        raise ValueError(f"{level.label}: L, S, J must be >= 0")
    if not (is_half_integer(level.L) and is_half_integer(level.S) and is_half_integer(level.J)):
        raise ValueError(f"{level.label}: L, S, J must be integer or half-integer")
    j_min = abs(level.L - level.S)
    j_max = level.L + level.S
    if level.J < j_min - 1e-12 or level.J > j_max + 1e-12:
        raise ValueError(f"{level.label}: J violates |L-S| <= J <= L+S")


def mj_values(J: float) -> np.ndarray:
    """Return m_J = -J, -J+1, ..., J as float array."""
    n2 = int(round(2.0 * J))
    return np.arange(-n2, n2 + 1, 2, dtype=np.int64) / 2.0


def lande_g(level: TermLevel) -> float:
    """Landé g factor in LS coupling.

    For J=0, the Zeeman first-order shift is zero because m_J is only 0.
    """
    validate_level(level)
    J = level.J
    if J == 0.0:
        return 0.0
    L = level.L
    S = level.S
    numerator = J * (J + 1.0) + S * (S + 1.0) - L * (L + 1.0)
    denominator = 2.0 * J * (J + 1.0)
    return 1.0 + numerator / denominator


def zeeman_shift_frequency(level: TermLevel, B_t: float, m_j: float) -> float:
    """Frequency shift (Hz): Delta nu = (mu_B / h) * g_J * m_J * B."""
    g_j = lande_g(level)
    return (MU_B / h) * g_j * m_j * B_t


def polarization_from_delta_m(delta_m: float) -> str:
    if np.isclose(delta_m, 0.0):
        return "pi"
    if np.isclose(delta_m, 1.0):
        return "sigma+"
    if np.isclose(delta_m, -1.0):
        return "sigma-"
    raise ValueError(f"Unexpected delta_m={delta_m}")


def zeeman_components(transition: Transition, B_t: float) -> pd.DataFrame:
    """Enumerate allowed Zeeman components for a transition.

    Selection rule used in this MVP: Delta m_J = 0, ±1.
    """
    if B_t < 0:
        raise ValueError("B_t must be non-negative")

    nu0_hz = c / (transition.lambda0_nm * 1e-9)

    rows = []
    for m_u in mj_values(transition.upper.J):
        for m_l in mj_values(transition.lower.J):
            delta_m = m_u - m_l
            if not any(np.isclose(delta_m, q) for q in (-1.0, 0.0, 1.0)):
                continue

            shift_u = zeeman_shift_frequency(transition.upper, B_t, m_u)
            shift_l = zeeman_shift_frequency(transition.lower, B_t, m_l)
            delta_nu = shift_u - shift_l
            nu = nu0_hz + delta_nu
            lam_nm = (c / nu) * 1e9

            rows.append(
                {
                    "transition": transition.name,
                    "m_upper": m_u,
                    "m_lower": m_l,
                    "delta_m": delta_m,
                    "polarization": polarization_from_delta_m(delta_m),
                    "delta_nu_GHz": delta_nu / 1e9,
                    "lambda_nm": lam_nm,
                    "weight": 1.0,
                }
            )

    if not rows:
        raise RuntimeError("No Zeeman components generated")

    df = pd.DataFrame(rows)
    return df.sort_values(["delta_nu_GHz", "m_upper", "m_lower"]).reset_index(drop=True)


def collapse_degenerate_components(df: pd.DataFrame, rounding_digits: int = 6) -> pd.DataFrame:
    """Merge nearly degenerate components by rounded frequency shifts."""
    grouped = (
        df.assign(delta_nu_key=df["delta_nu_GHz"].round(rounding_digits))
        .groupby(["transition", "delta_nu_key", "polarization"], as_index=False)
        .agg(
            delta_nu_GHz=("delta_nu_GHz", "mean"),
            lambda_nm=("lambda_nm", "mean"),
            multiplicity=("weight", "count"),
            weight=("weight", "sum"),
        )
        .sort_values("delta_nu_GHz")
        .reset_index(drop=True)
    )
    return grouped


def gaussian_profile(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z)


def synthesize_spectrum(
    lines_df: pd.DataFrame,
    lambda0_nm: float,
    span_pm: float = 120.0,
    fwhm_pm: float = 3.0,
    points: int = 4000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a synthetic broadened line profile in wavelength domain."""
    span_nm = span_pm * 1e-3
    sigma_nm = (fwhm_pm * 1e-3) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    x = np.linspace(lambda0_nm - span_nm, lambda0_nm + span_nm, points)
    y = np.zeros_like(x)

    for row in lines_df.itertuples(index=False):
        y += float(row.weight) * gaussian_profile(x, float(row.lambda_nm), sigma_nm)

    y /= float(np.max(y))
    return x, y


def check_normal_triplet_spacing(normal_collapsed: pd.DataFrame, B_t: float, tol_rel: float = 3e-4) -> None:
    """Check that the normal Zeeman triplet has equal spacing in frequency shift."""
    shifts = np.sort(normal_collapsed["delta_nu_GHz"].to_numpy(dtype=float))
    if shifts.size != 3:
        raise AssertionError(f"Expected 3 components, got {shifts.size}")

    expected_step_ghz = (MU_B / h) * B_t / 1e9
    left_step = shifts[1] - shifts[0]
    right_step = shifts[2] - shifts[1]

    assert np.isclose(left_step, expected_step_ghz, rtol=tol_rel), (
        f"left_step mismatch: {left_step} vs {expected_step_ghz}"
    )
    assert np.isclose(right_step, expected_step_ghz, rtol=tol_rel), (
        f"right_step mismatch: {right_step} vs {expected_step_ghz}"
    )


def main() -> None:
    # Normal Zeeman example (S=0 -> g=1 for J=1): triplet splitting.
    normal_transition = Transition(
        name="1P1_to_1S0",
        upper=TermLevel(label="1P1", L=1.0, S=0.0, J=1.0),
        lower=TermLevel(label="1S0", L=0.0, S=0.0, J=0.0),
        lambda0_nm=500.0,
    )

    # Anomalous Zeeman example (different Landé g factors).
    anomalous_transition = Transition(
        name="2P3/2_to_2S1/2",
        upper=TermLevel(label="2P3/2", L=1.0, S=0.5, J=1.5),
        lower=TermLevel(label="2S1/2", L=0.0, S=0.5, J=0.5),
        lambda0_nm=589.0,
    )

    B_main = 0.8

    # Component tables
    normal_df = zeeman_components(normal_transition, B_t=B_main)
    anomalous_df = zeeman_components(anomalous_transition, B_t=B_main)

    normal_collapsed = collapse_degenerate_components(normal_df)
    anomalous_collapsed = collapse_degenerate_components(anomalous_df)

    # Linear-in-B verification for normal triplet
    normal_halfB = collapse_degenerate_components(zeeman_components(normal_transition, B_t=0.4))
    normal_fullB = collapse_degenerate_components(zeeman_components(normal_transition, B_t=0.8))
    max_shift_half = np.max(np.abs(normal_halfB["delta_nu_GHz"].to_numpy(dtype=float)))
    max_shift_full = np.max(np.abs(normal_fullB["delta_nu_GHz"].to_numpy(dtype=float)))
    linearity_ratio = max_shift_full / max_shift_half

    # B=0 sanity check
    zero_field = collapse_degenerate_components(zeeman_components(anomalous_transition, B_t=0.0))
    max_zero_shift = np.max(np.abs(zero_field["delta_nu_GHz"].to_numpy(dtype=float)))

    # Synthetic spectrum and peak counting for normal triplet
    x_nm, y = synthesize_spectrum(normal_collapsed, lambda0_nm=normal_transition.lambda0_nm, fwhm_pm=2.0)
    peaks, _ = find_peaks(y, height=0.2, distance=80)
    peak_count = int(peaks.size)

    # Structured outputs
    print("Zeeman effect MVP")
    print(f"Bohr magneton mu_B = {MU_B:.9e} J/T")
    print(f"B field = {B_main:.3f} T")
    print(
        "Landé g factors: "
        f"g(1P1)={lande_g(normal_transition.upper):.6f}, "
        f"g(1S0)={lande_g(normal_transition.lower):.6f}, "
        f"g(2P3/2)={lande_g(anomalous_transition.upper):.6f}, "
        f"g(2S1/2)={lande_g(anomalous_transition.lower):.6f}"
    )

    print("\n[Normal Zeeman collapsed components]")
    print(normal_collapsed[["delta_nu_GHz", "lambda_nm", "polarization", "multiplicity"]].to_string(index=False))

    print("\n[Anomalous Zeeman collapsed components]")
    print(anomalous_collapsed[["delta_nu_GHz", "lambda_nm", "polarization", "multiplicity"]].to_string(index=False))

    print("\n[Checks]")
    print(f"normal_component_count={normal_collapsed.shape[0]}")
    print(f"anomalous_component_count={anomalous_collapsed.shape[0]}")
    print(f"linearity_ratio_B0.8_over_B0.4={linearity_ratio:.6f}")
    print(f"zero_field_max_shift_GHz={max_zero_shift:.3e}")
    print(f"normal_spectrum_peak_count={peak_count}")

    # Assertions
    check_normal_triplet_spacing(normal_collapsed, B_t=B_main)
    assert normal_collapsed.shape[0] == 3, "Normal Zeeman must collapse to 3 components"
    assert anomalous_df.shape[0] == 6, f"Expected 6 raw anomalous transitions, got {anomalous_df.shape[0]}"
    assert anomalous_collapsed.shape[0] >= 4, "Anomalous case should have at least 4 distinct shifts"
    assert np.isclose(linearity_ratio, 2.0, rtol=2e-3), f"Linearity failed: ratio={linearity_ratio}"
    assert max_zero_shift < 1e-9, f"Zero-field shifts should vanish, got {max_zero_shift} GHz"
    assert peak_count == 3, f"Expected 3 peaks for normal triplet spectrum, got {peak_count}"

    print("All checks passed.")


if __name__ == "__main__":
    main()
