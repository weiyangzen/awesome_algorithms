"""Lamb Shift MVP for hydrogen-like levels.

This script implements a compact, auditable approximation pipeline:
1) one-loop self-energy (A40 + leading logarithm),
2) one-loop vacuum polarization for S states,
3) a fitted effective higher-order S-state remainder that matches
   the 2S1/2-2P1/2 experimental splitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import alpha, c, h, m_e, pi
from scipy.optimize import root_scalar


@dataclass(frozen=True)
class HydrogenLevel:
    label: str
    n: int
    l: int
    j: float
    bethe_log: float


# Bethe logarithms used in the one-loop A40 terms.
LEVELS: tuple[HydrogenLevel, ...] = (
    HydrogenLevel(label="1S1/2", n=1, l=0, j=0.5, bethe_log=2.984128556),
    HydrogenLevel(label="2S1/2", n=2, l=0, j=0.5, bethe_log=2.811769893),
    HydrogenLevel(label="2P1/2", n=2, l=1, j=0.5, bethe_log=-0.030016709),
    HydrogenLevel(label="2P3/2", n=2, l=1, j=1.5, bethe_log=-0.030016709),
)

TARGET_2S_2P12_SPLITTING_MHZ = 1057.844


def _find_level(label: str) -> HydrogenLevel:
    for level in LEVELS:
        if level.label == label:
            return level
    raise KeyError(f"Unknown level label: {label}")


def one_loop_prefactor_hz(n: int, z: int = 1) -> float:
    """Return common one-loop Lamb-shift prefactor in Hz.

    prefactor = (alpha/pi) * (Z*alpha)^4 * (m_e*c^2/h) / n^3
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if z <= 0:
        raise ValueError("z must be positive.")
    return (alpha / pi) * (z * alpha) ** 4 * (m_e * c**2 / h) / (n**3)


def a40_term(level: HydrogenLevel) -> float:
    """Return the one-loop A40 coefficient for selected hydrogen levels."""
    if level.l == 0 and np.isclose(level.j, 0.5):
        return (10.0 / 9.0) - (4.0 / 3.0) * level.bethe_log
    if level.l == 1 and np.isclose(level.j, 0.5):
        return (-1.0 / 6.0) - (4.0 / 3.0) * level.bethe_log
    if level.l == 1 and np.isclose(level.j, 1.5):
        return (1.0 / 12.0) - (4.0 / 3.0) * level.bethe_log
    raise ValueError(f"Unsupported level for A40 approximation: {level}")


def one_loop_self_energy_hz(level: HydrogenLevel, z: int = 1) -> float:
    """Compute one-loop self-energy contribution in Hz."""
    term = a40_term(level)
    if level.l == 0:
        term += (4.0 / 3.0) * np.log((1.0 / (z * alpha)) ** 2)
    return one_loop_prefactor_hz(level.n, z=z) * term


def one_loop_vacuum_polarization_hz(level: HydrogenLevel, z: int = 1) -> float:
    """Compute a minimal one-loop vacuum polarization term in Hz.

    Uses a simple leading-order S-state contribution:
    DeltaE_VP(ns) ~ -(4/15) * prefactor.
    """
    if level.l != 0:
        return 0.0
    return -(4.0 / 15.0) * one_loop_prefactor_hz(level.n, z=z)


def effective_higher_order_remainder_hz(level: HydrogenLevel, c_s_mhz: float) -> float:
    """Model unaccounted higher-order terms as C_s/n^3 for S states only."""
    if level.l != 0:
        return 0.0
    return (c_s_mhz * 1.0e6) / (level.n**3)


def total_shift_hz(level: HydrogenLevel, z: int = 1, c_s_mhz: float = 0.0) -> float:
    """Total modeled Lamb shift in Hz for a given level."""
    return (
        one_loop_self_energy_hz(level, z=z)
        + one_loop_vacuum_polarization_hz(level, z=z)
        + effective_higher_order_remainder_hz(level, c_s_mhz=c_s_mhz)
    )


def splitting_mhz(level_a: HydrogenLevel, level_b: HydrogenLevel, c_s_mhz: float = 0.0) -> float:
    """Return (level_a - level_b) in MHz."""
    return (total_shift_hz(level_a, c_s_mhz=c_s_mhz) - total_shift_hz(level_b, c_s_mhz=c_s_mhz)) / 1.0e6


def fit_effective_remainder_c_s_mhz(target_splitting_mhz: float) -> float:
    """Fit the S-state remainder constant so 2S1/2-2P1/2 matches target."""
    level_2s = _find_level("2S1/2")
    level_2p12 = _find_level("2P1/2")

    def objective(c_s_mhz: float) -> float:
        return splitting_mhz(level_2s, level_2p12, c_s_mhz=c_s_mhz) - target_splitting_mhz

    left, right = -200.0, 200.0
    f_left, f_right = objective(left), objective(right)
    if f_left * f_right > 0:
        raise RuntimeError("Failed to bracket root for effective remainder constant.")

    result = root_scalar(objective, bracket=[left, right], method="brentq", xtol=1e-12, rtol=1e-12)
    if not result.converged:
        raise RuntimeError("Root solving for effective remainder did not converge.")
    return float(result.root)


def build_contribution_table(c_s_mhz: float) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for level in LEVELS:
        se_mhz = one_loop_self_energy_hz(level) / 1.0e6
        vp_mhz = one_loop_vacuum_polarization_hz(level) / 1.0e6
        rem_mhz = effective_higher_order_remainder_hz(level, c_s_mhz=c_s_mhz) / 1.0e6
        total_mhz = se_mhz + vp_mhz + rem_mhz
        rows.append(
            {
                "level": level.label,
                "n": level.n,
                "l": level.l,
                "j": level.j,
                "self_energy_MHz": se_mhz,
                "vacuum_polarization_MHz": vp_mhz,
                "effective_remainder_MHz": rem_mhz,
                "total_shift_MHz": total_mhz,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    level_1s = _find_level("1S1/2")
    level_2s = _find_level("2S1/2")
    level_2p12 = _find_level("2P1/2")
    level_2p32 = _find_level("2P3/2")

    one_loop_split = splitting_mhz(level_2s, level_2p12, c_s_mhz=0.0)
    fitted_c_s_mhz = fit_effective_remainder_c_s_mhz(TARGET_2S_2P12_SPLITTING_MHZ)
    fitted_split = splitting_mhz(level_2s, level_2p12, c_s_mhz=fitted_c_s_mhz)
    p_fine_structure = splitting_mhz(level_2p32, level_2p12, c_s_mhz=fitted_c_s_mhz)

    one_loop_table = build_contribution_table(c_s_mhz=0.0)
    fitted_table = build_contribution_table(c_s_mhz=fitted_c_s_mhz)

    one_loop_1s = float(one_loop_table.loc[one_loop_table["level"] == "1S1/2", "total_shift_MHz"].iloc[0])
    one_loop_2s = float(one_loop_table.loc[one_loop_table["level"] == "2S1/2", "total_shift_MHz"].iloc[0])
    s_ratio = one_loop_1s / one_loop_2s

    summary_df = pd.DataFrame(
        [
            {
                "quantity": "2S1/2-2P1/2 one-loop (MHz)",
                "value": one_loop_split,
            },
            {
                "quantity": "2S1/2-2P1/2 target (MHz)",
                "value": TARGET_2S_2P12_SPLITTING_MHZ,
            },
            {
                "quantity": "fitted C_s (MHz)",
                "value": fitted_c_s_mhz,
            },
            {
                "quantity": "2S1/2-2P1/2 fitted (MHz)",
                "value": fitted_split,
            },
            {
                "quantity": "2P3/2-2P1/2 fitted (MHz)",
                "value": p_fine_structure,
            },
            {
                "quantity": "1S/2S one-loop ratio",
                "value": s_ratio,
            },
        ]
    )

    print("=== One-Loop Lamb Shift Contributions (C_s = 0) ===")
    print(one_loop_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print("=== Fitted Model Contributions (with effective higher-order remainder) ===")
    print(fitted_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print("=== Key Splittings and Fitted Constant ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Core sanity checks for this MVP.
    assert 900.0 < one_loop_split < 1200.0, "One-loop 2S-2P splitting left expected scale window."
    assert abs(fitted_split - TARGET_2S_2P12_SPLITTING_MHZ) < 1e-8, "Fitted splitting mismatch too large."
    assert 7.0 < s_ratio < 8.6, "1S/2S one-loop ratio should stay close to n^-3 scaling (~8)."
    assert fitted_c_s_mhz > 0.0, "Expected positive effective higher-order remainder for this setup."
    assert abs(float(total_shift_hz(level_2p12, c_s_mhz=fitted_c_s_mhz) / 1.0e6)) < 50.0, (
        "2P1/2 shift should remain much smaller than S-state shifts in this approximation."
    )
    assert float(total_shift_hz(level_1s, c_s_mhz=fitted_c_s_mhz) / 1.0e6) > float(
        total_shift_hz(level_2s, c_s_mhz=fitted_c_s_mhz) / 1.0e6
    ), "1S Lamb shift should be larger than 2S in this model."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
