"""Minimal runnable MVP for hydrogenic fine structure.

The script computes hydrogen-like bound-state energies with:
1) Non-relativistic baseline energy E_n^(0)
2) First-order fine-structure correction Delta E_fs(n, j)
3) Exact Dirac binding energy as a cross-check

It then validates expected physical trends and prints a compact report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import alpha, c, electron_volt, h, m_e, m_p


@dataclass(frozen=True)
class QuantumState:
    n: int
    l: int
    j: float


def l_to_symbol(l: int) -> str:
    symbols = ["s", "p", "d", "f", "g", "h", "i"]
    if 0 <= l < len(symbols):
        return symbols[l]
    return f"l{l}"


def state_label(state: QuantumState) -> str:
    two_j = int(round(2.0 * state.j))
    return f"{state.n}{l_to_symbol(state.l)}_{two_j}/2"


def reduced_mass_hydrogen() -> float:
    """Reduced mass mu = m_e * m_p / (m_e + m_p)."""
    return (m_e * m_p) / (m_e + m_p)


def nonrel_binding_energy_eV(n: int, z: int, mu: float) -> float:
    """Hydrogen-like non-relativistic bound-state energy (negative, in eV)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    factor = (z * alpha) ** 2
    e_joule = -0.5 * mu * c * c * factor / float(n * n)
    return e_joule / electron_volt


def fine_structure_shift_eV(n: int, j: float, z: int, mu: float) -> float:
    """First-order fine-structure correction in eV.

    Delta E_fs = E_n^(0) * (Z*alpha)^2 / n * (1/(j+1/2) - 3/(4n)).
    """
    if j < 0.5:
        raise ValueError("j must be >= 1/2")
    e0 = nonrel_binding_energy_eV(n=n, z=z, mu=mu)
    bracket = (1.0 / (j + 0.5)) - (3.0 / (4.0 * n))
    return e0 * ((z * alpha) ** 2) * bracket / float(n)


def dirac_binding_energy_eV(n: int, j: float, z: int, mu: float) -> float:
    """Exact Dirac binding energy (rest mass removed), in eV."""
    if j < 0.5:
        raise ValueError("j must be >= 1/2")

    zj = z * alpha
    gamma_term = (j + 0.5) ** 2 - zj * zj
    if gamma_term <= 0.0:
        raise ValueError("Invalid (Z, j) combination for real Dirac spectrum")

    delta_j = (j + 0.5) - np.sqrt(gamma_term)
    denom = float(n) - delta_j
    e_total_joule = mu * c * c / np.sqrt(1.0 + (zj * zj) / (denom * denom))
    e_bind_joule = e_total_joule - mu * c * c
    return e_bind_joule / electron_volt


def enumerate_hydrogenic_states(n_max: int) -> list[QuantumState]:
    if n_max < 1:
        raise ValueError("n_max must be >= 1")

    states: list[QuantumState] = []
    for n in range(1, n_max + 1):
        for l in range(0, n):
            if l == 0:
                states.append(QuantumState(n=n, l=l, j=0.5))
                continue
            states.append(QuantumState(n=n, l=l, j=l - 0.5))
            states.append(QuantumState(n=n, l=l, j=l + 0.5))
    return states


def build_energy_table(n_max: int, z: int, use_reduced_mass: bool) -> pd.DataFrame:
    mu = reduced_mass_hydrogen() if use_reduced_mass else m_e
    rows = []
    for st in enumerate_hydrogenic_states(n_max=n_max):
        e0 = nonrel_binding_energy_eV(n=st.n, z=z, mu=mu)
        dfs = fine_structure_shift_eV(n=st.n, j=st.j, z=z, mu=mu)
        e_approx = e0 + dfs
        e_dirac = dirac_binding_energy_eV(n=st.n, j=st.j, z=z, mu=mu)
        rows.append(
            {
                "state": state_label(st),
                "n": st.n,
                "l": st.l,
                "j": st.j,
                "E0_eV": e0,
                "DeltaE_fs_eV": dfs,
                "E_approx_eV": e_approx,
                "E_dirac_eV": e_dirac,
                "abs_err_eV": abs(e_approx - e_dirac),
            }
        )

    df = pd.DataFrame(rows).sort_values(["n", "l", "j"]).reset_index(drop=True)
    return df


def compute_2p_splitting_eV(z: int, use_reduced_mass: bool) -> float:
    mu = reduced_mass_hydrogen() if use_reduced_mass else m_e
    e_j12 = nonrel_binding_energy_eV(2, z, mu) + fine_structure_shift_eV(2, 0.5, z, mu)
    e_j32 = nonrel_binding_energy_eV(2, z, mu) + fine_structure_shift_eV(2, 1.5, z, mu)
    return float(abs(e_j12 - e_j32))


def main() -> None:
    n_max = 4

    # Hydrogen (with reduced mass) table for reporting.
    df_h = build_energy_table(n_max=n_max, z=1, use_reduced_mass=True)

    split_eV_h = compute_2p_splitting_eV(z=1, use_reduced_mass=True)
    split_h_ghz = (split_eV_h * electron_volt / h) / 1.0e9

    # Infinite nuclear mass mode for clean Z^4 scaling verification.
    split_eV_z1_inf = compute_2p_splitting_eV(z=1, use_reduced_mass=False)
    split_eV_z2_inf = compute_2p_splitting_eV(z=2, use_reduced_mass=False)
    z4_ratio = split_eV_z2_inf / split_eV_z1_inf

    # 2s1/2 and 2p1/2 degeneracy in this model (no Lamb shift).
    e_2s12 = df_h.loc[df_h["state"] == "2s_1/2", "E_approx_eV"].iloc[0]
    e_2p12 = df_h.loc[df_h["state"] == "2p_1/2", "E_approx_eV"].iloc[0]
    e_2p32 = df_h.loc[df_h["state"] == "2p_3/2", "E_approx_eV"].iloc[0]

    max_abs_err = float(df_h["abs_err_eV"].max())

    print("Fine-structure MVP (hydrogenic)")
    print(f"alpha={alpha:.12f}")
    print(f"n_max={n_max}")
    print()

    print("Energy table (first 10 rows):")
    print(df_h.head(10).to_string(index=False))
    print()

    print("Key diagnostics:")
    print(f"  max |E_approx - E_dirac| = {max_abs_err:.3e} eV")
    print(f"  2p splitting (H, reduced mass) = {split_eV_h:.9e} eV")
    print(f"  2p splitting (H, reduced mass) = {split_h_ghz:.6f} GHz")
    print(f"  E(2s_1/2) - E(2p_1/2) = {e_2s12 - e_2p12:.3e} eV")
    print(f"  Z^4 scaling ratio split(Z=2)/split(Z=1) = {z4_ratio:.8f}")

    # Assertions: deterministic and physics-based.
    assert max_abs_err < 1.0e-7, f"Approx vs Dirac error too large: {max_abs_err}"
    assert abs(e_2s12 - e_2p12) < 1.0e-12, "2s_1/2 and 2p_1/2 should be degenerate in this model"
    assert e_2p12 < e_2p32, "Expected E(2p_1/2) < E(2p_3/2) for hydrogen fine structure"
    assert np.isclose(z4_ratio, 16.0, rtol=2.0e-4), f"Z^4 scaling failed: ratio={z4_ratio}"
    assert 8.0 < split_h_ghz < 13.0, f"Unexpected 2p splitting frequency scale: {split_h_ghz} GHz"

    print("All checks passed.")


if __name__ == "__main__":
    main()
