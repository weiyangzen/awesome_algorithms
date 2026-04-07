"""Minimal runnable MVP for atomic hyperfine structure.

This script models zero-field hyperfine energies with the standard
magnetic-dipole (A) and electric-quadrupole (B) constants:

E(F) = (A/2) * K + B * [ (3/4)K(K+1) - I(I+1)J(J+1) ] / [2I(2I-1)J(2J-1)]
K = F(F+1) - I(I+1) - J(J+1)

It prints level tables and runs deterministic checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import electron_volt, h


@dataclass(frozen=True)
class HyperfineCase:
    label: str
    I: float
    J: float
    A_MHz: float
    B_MHz: float


def allowed_f_values(I: float, J: float) -> list[float]:
    """Return allowed F values: |I-J|, |I-J|+1, ..., I+J."""
    f_min = abs(I - J)
    f_max = I + J
    count = int(round(f_max - f_min)) + 1
    return [round(f_min + k, 10) for k in range(count)]


def k_value(F: float, I: float, J: float) -> float:
    return F * (F + 1.0) - I * (I + 1.0) - J * (J + 1.0)


def hyperfine_energy_mhz(F: float, I: float, J: float, A_MHz: float, B_MHz: float) -> tuple[float, float, float]:
    """Return (E_A, E_B, E_total) in MHz."""
    K = k_value(F=F, I=I, J=J)
    e_a = 0.5 * A_MHz * K

    has_quadrupole = (I >= 1.0) and (J >= 1.0) and (abs(B_MHz) > 0.0)
    if not has_quadrupole:
        e_b = 0.0
    else:
        numerator = 0.75 * K * (K + 1.0) - I * (I + 1.0) * J * (J + 1.0)
        denominator = 2.0 * I * (2.0 * I - 1.0) * J * (2.0 * J - 1.0)
        e_b = B_MHz * numerator / denominator

    return e_a, e_b, e_a + e_b


def build_levels_table(case: HyperfineCase) -> pd.DataFrame:
    rows = []
    for F in allowed_f_values(case.I, case.J):
        e_a, e_b, e_total = hyperfine_energy_mhz(
            F=F,
            I=case.I,
            J=case.J,
            A_MHz=case.A_MHz,
            B_MHz=case.B_MHz,
        )
        rows.append(
            {
                "F": F,
                "g_F": int(round(2.0 * F + 1.0)),
                "K": k_value(F, case.I, case.J),
                "E_A_MHz": e_a,
                "E_B_MHz": e_b,
                "E_total_MHz": e_total,
            }
        )
    df = pd.DataFrame(rows).sort_values("F").reset_index(drop=True)
    df["E_total_GHz"] = df["E_total_MHz"] / 1.0e3
    df["E_total_J"] = df["E_total_MHz"] * 1.0e6 * h
    df["E_total_eV"] = df["E_total_J"] / electron_volt
    return df


def build_intervals_table(levels_df: pd.DataFrame, A_MHz: float, B_MHz: float) -> pd.DataFrame:
    rows = []
    for idx in range(len(levels_df) - 1):
        f_low = float(levels_df.loc[idx, "F"])
        f_high = float(levels_df.loc[idx + 1, "F"])
        e_low = float(levels_df.loc[idx, "E_total_MHz"])
        e_high = float(levels_df.loc[idx + 1, "E_total_MHz"])
        delta = e_high - e_low

        expected = np.nan
        if np.isclose(B_MHz, 0.0):
            expected = A_MHz * (f_low + 1.0)

        rows.append(
            {
                "transition": f"F={f_low:g} -> F={f_high:g}",
                "DeltaE_MHz": delta,
                "Expected_Lande_MHz_if_B0": expected,
            }
        )
    return pd.DataFrame(rows)


def weighted_centroid_mhz(levels_df: pd.DataFrame) -> float:
    weights = levels_df["g_F"].to_numpy(dtype=float)
    energies = levels_df["E_total_MHz"].to_numpy(dtype=float)
    return float(np.sum(weights * energies) / np.sum(weights))


def run_checks() -> None:
    # Check 1: hydrogen 1S1/2 should have two levels split exactly by A (B=0).
    hydrogen = HyperfineCase(
        label="Hydrogen 1S1/2",
        I=0.5,
        J=0.5,
        A_MHz=1420.405751768,
        B_MHz=0.0,
    )
    h_levels = build_levels_table(hydrogen)
    assert len(h_levels) == 2, "Hydrogen 1S1/2 should produce F=0,1 only"
    split = float(h_levels.loc[1, "E_total_MHz"] - h_levels.loc[0, "E_total_MHz"])
    assert np.isclose(split, hydrogen.A_MHz, rtol=0.0, atol=1e-9), (
        f"Hydrogen splitting mismatch: {split} vs {hydrogen.A_MHz}"
    )

    # Check 2: weighted centroid should be zero for both A-term and A+B term formula.
    rb87_like = HyperfineCase(
        label="I=3/2, J=3/2 example",
        I=1.5,
        J=1.5,
        A_MHz=84.7185,
        B_MHz=12.4965,
    )
    rb_levels = build_levels_table(rb87_like)
    assert abs(weighted_centroid_mhz(h_levels)) < 1e-12
    assert abs(weighted_centroid_mhz(rb_levels)) < 1e-12

    # Check 3: Landé interval rule must hold exactly when B=0.
    synthetic_b0 = HyperfineCase(
        label="Synthetic B=0 check",
        I=1.5,
        J=1.5,
        A_MHz=100.0,
        B_MHz=0.0,
    )
    levels_b0 = build_levels_table(synthetic_b0)
    intervals_b0 = build_intervals_table(levels_b0, synthetic_b0.A_MHz, synthetic_b0.B_MHz)
    expected = np.array([100.0, 200.0, 300.0], dtype=float)
    got = intervals_b0["DeltaE_MHz"].to_numpy(dtype=float)
    assert np.allclose(got, expected, rtol=0.0, atol=1e-12), f"Landé rule failed: got={got}"


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)

    cases = [
        HyperfineCase(
            label="Hydrogen 1S1/2 (B=0)",
            I=0.5,
            J=0.5,
            A_MHz=1420.405751768,
            B_MHz=0.0,
        ),
        HyperfineCase(
            label="I=3/2, J=3/2 with A/B",
            I=1.5,
            J=1.5,
            A_MHz=84.7185,
            B_MHz=12.4965,
        ),
    ]

    print("Hyperfine Structure MVP")
    print("Formula: E(F)=E_A+E_B, with explicit A/B terms (zero magnetic field).")
    print()

    for case in cases:
        levels = build_levels_table(case)
        intervals = build_intervals_table(levels, case.A_MHz, case.B_MHz)
        centroid = weighted_centroid_mhz(levels)

        print(f"[Case] {case.label}")
        print(f"  I={case.I:g}, J={case.J:g}, A={case.A_MHz:.9f} MHz, B={case.B_MHz:.9f} MHz")
        print("  Levels:")
        print(levels.to_string(index=False, float_format=lambda x: f"{x:.9f}"))
        print("  Adjacent intervals:")
        print(intervals.to_string(index=False, float_format=lambda x: f"{x:.9f}"))
        print(f"  Weighted centroid = {centroid:.12e} MHz")
        print()

    run_checks()
    print("All checks passed.")


if __name__ == "__main__":
    main()
