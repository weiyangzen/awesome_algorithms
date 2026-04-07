"""Minimal runnable MVP for the Kronig-Penney model.

This demo uses the dimensionless delta-barrier Kronig-Penney equation:
    cos(k a) = F(x) = cos(x) + P * sin(x) / x
where x = alpha * a and E~ = x^2 in reduced units (hbar^2/2m = 1, a = 1).
Allowed bands satisfy |F(x)| <= 1, and forbidden gaps satisfy |F(x)| > 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq


EPS = 1e-12


@dataclass
class BandInfo:
    band_index: int
    x_left: float
    x_right: float
    e_left: float
    e_right: float
    e_width: float
    k_min: float
    k_max: float
    edge_residual: float
    max_allowed_violation: float


@dataclass
class GapInfo:
    gap_index: int
    prev_band: int
    next_band: int
    e_start: float
    e_end: float
    e_width: float
    midpoint_forbidden_margin: float


def kronig_penney_rhs(x: np.ndarray | float, p: float) -> np.ndarray:
    """Return F(x) = cos(x) + p * sin(x)/x with stable sinc handling near x=0."""
    x_arr = np.asarray(x, dtype=float)
    sin_over_x = np.sinc(x_arr / np.pi)  # np.sinc(y)=sin(pi*y)/(pi*y)
    return np.cos(x_arr) + p * sin_over_x


def allowed_mask(rhs: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Mask of allowed energies: |F(x)| <= 1."""
    return np.abs(rhs) <= (1.0 + tol)


def find_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous True segments as inclusive index pairs."""
    segments: list[tuple[int, int]] = []
    n = mask.size
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        segments.append((i, j))
        i = j + 1
    return segments


def refine_edge(x0: float, x1: float, p: float) -> float:
    """Refine boundary root of |F(x)| - 1 = 0 in [x0, x1] when possible."""

    def h(z: float) -> float:
        return abs(float(kronig_penney_rhs(z, p))) - 1.0

    h0 = h(x0)
    h1 = h(x1)
    if abs(h0) < EPS:
        return float(x0)
    if abs(h1) < EPS:
        return float(x1)
    if h0 * h1 < 0.0:
        return float(brentq(h, x0, x1, xtol=1e-12, rtol=1e-12, maxiter=200))
    return float(x0 if abs(h0) < abs(h1) else x1)


def analyze_bands(p: float, x_max: float, n_grid: int) -> tuple[list[BandInfo], list[GapInfo]]:
    """Scan x-space, identify allowed bands, and refine band edges."""
    x = np.linspace(1e-4, x_max, n_grid)
    rhs = kronig_penney_rhs(x, p)
    mask = allowed_mask(rhs, tol=1e-10)
    segments = find_true_segments(mask)

    bands: list[BandInfo] = []
    for band_id, (left_idx, right_idx) in enumerate(segments, start=1):
        x_left = float(x[left_idx])
        x_right = float(x[right_idx])

        if left_idx > 0:
            x_left = refine_edge(float(x[left_idx - 1]), float(x[left_idx]), p)
        if right_idx + 1 < x.size:
            x_right = refine_edge(float(x[right_idx]), float(x[right_idx + 1]), p)

        sample_x = np.linspace(x_left, x_right, 256)
        sample_rhs = kronig_penney_rhs(sample_x, p)
        clipped = np.clip(sample_rhs, -1.0, 1.0)
        k_vals = np.arccos(clipped)

        e_left = x_left * x_left
        e_right = x_right * x_right
        edge_res = max(
            abs(abs(float(kronig_penney_rhs(x_left, p))) - 1.0),
            abs(abs(float(kronig_penney_rhs(x_right, p))) - 1.0),
        )
        max_violation = float(np.max(np.maximum(np.abs(sample_rhs) - 1.0, 0.0)))

        bands.append(
            BandInfo(
                band_index=band_id,
                x_left=x_left,
                x_right=x_right,
                e_left=e_left,
                e_right=e_right,
                e_width=e_right - e_left,
                k_min=float(np.min(k_vals)),
                k_max=float(np.max(k_vals)),
                edge_residual=float(edge_res),
                max_allowed_violation=max_violation,
            )
        )

    gaps: list[GapInfo] = []
    for i in range(len(bands) - 1):
        prev_band = bands[i]
        next_band = bands[i + 1]
        e_start = prev_band.e_right
        e_end = next_band.e_left
        x_mid = 0.5 * (prev_band.x_right + next_band.x_left)
        margin = abs(float(kronig_penney_rhs(x_mid, p))) - 1.0
        gaps.append(
            GapInfo(
                gap_index=i + 1,
                prev_band=prev_band.band_index,
                next_band=next_band.band_index,
                e_start=e_start,
                e_end=e_end,
                e_width=e_end - e_start,
                midpoint_forbidden_margin=float(margin),
            )
        )

    return bands, gaps


def bands_to_dataframe(bands: list[BandInfo]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "band": b.band_index,
                "x_left": b.x_left,
                "x_right": b.x_right,
                "E_left": b.e_left,
                "E_right": b.e_right,
                "E_width": b.e_width,
                "k_min": b.k_min,
                "k_max": b.k_max,
                "edge_residual": b.edge_residual,
                "allowed_violation": b.max_allowed_violation,
            }
            for b in bands
        ]
    )


def gaps_to_dataframe(gaps: list[GapInfo]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gap": g.gap_index,
                "between_bands": f"{g.prev_band}-{g.next_band}",
                "E_start": g.e_start,
                "E_end": g.e_end,
                "E_width": g.e_width,
                "midpoint_forbidden_margin": g.midpoint_forbidden_margin,
            }
            for g in gaps
        ]
    )


def main() -> None:
    # Dimensionless parameters for a compact, reproducible MVP.
    p = 6.0
    x_max = 12.0 * np.pi
    n_grid = 80_000

    bands, gaps = analyze_bands(p=p, x_max=x_max, n_grid=n_grid)
    band_df = bands_to_dataframe(bands)
    gap_df = gaps_to_dataframe(gaps)

    if band_df.empty:
        raise RuntimeError("No allowed band found. Increase x_max or adjust p.")

    max_edge_residual = float(band_df["edge_residual"].max())
    max_allowed_violation = float(band_df["allowed_violation"].max())
    min_band_width = float(band_df["E_width"].min())
    gap_count = int(len(gap_df))
    min_gap_width = float(gap_df["E_width"].min()) if gap_count > 0 else 0.0
    min_forbidden_margin = (
        float(gap_df["midpoint_forbidden_margin"].min()) if gap_count > 0 else 0.0
    )

    checks = {
        "at least 4 bands detected": len(bands) >= 4,
        "all bands have positive width": min_band_width > 0.0,
        "max edge residual < 1e-8": max_edge_residual < 1e-8,
        "allowed violation < 1e-9": max_allowed_violation < 1e-9,
        "at least 1 forbidden gap": gap_count >= 1,
        "minimum gap width > 1e-4": min_gap_width > 1e-4 if gap_count > 0 else False,
        "mid-gap forbidden margin > 1e-3": min_forbidden_margin > 1e-3 if gap_count > 0 else False,
    }

    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    print("=== Kronig-Penney Model MVP (PHYS-0436) ===")
    print(f"Parameters: P={p:.3f}, a=1 (dimensionless), x_max={x_max:.3f}, n_grid={n_grid}")
    print("\nAllowed bands:")
    print(band_df.to_string(index=False))

    if gap_count > 0:
        print("\nForbidden gaps:")
        print(gap_df.to_string(index=False))
    else:
        print("\nForbidden gaps: none detected in the scanned range.")

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
