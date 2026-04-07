"""Minimal runnable MVP for hydrogen hyperfine splitting (1S1/2).

Model:
- Two coupled spins: proton I=1/2 and electron J=1/2.
- Hamiltonian H = A * (I dot J) + Zeeman terms along Bz.
- A is set by the 21 cm transition frequency at zero field.

This script prints level tables and transition frequencies, then runs
non-interactive assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.constants import h, physical_constants

# Experimental reference for hydrogen 1S hyperfine transition (~21 cm line)
NU_21CM_HZ = 1_420_405_751.768
A_HFS_J = h * NU_21CM_HZ

MU_B = physical_constants["Bohr magneton"][0]
MU_N = physical_constants["nuclear magneton"][0]
G_J = abs(physical_constants["electron g factor"][0])
G_P = physical_constants["proton g factor"][0]

# Spin-1/2 operators in units of hbar (dimensionless eigenvalues +/-1/2)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)
SX = 0.5 * SIGMA_X
SY = 0.5 * SIGMA_Y
SZ = 0.5 * SIGMA_Z


def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)


def build_hamiltonian(B_t: float) -> np.ndarray:
    """Construct 4x4 hyperfine Hamiltonian for magnetic field B_t (Tesla)."""
    h_hyperfine = A_HFS_J * (kron(SX, SX) + kron(SY, SY) + kron(SZ, SZ))

    # Electron Zeeman: + g_J * mu_B * m_J * B
    # Proton Zeeman:   - g_P * mu_N * m_I * B
    h_zeeman = (G_J * MU_B * B_t) * kron(I2, SZ) - (G_P * MU_N * B_t) * kron(SZ, I2)

    h_total = h_hyperfine + h_zeeman
    if not np.allclose(h_total, h_total.conj().T, rtol=0.0, atol=1e-18):
        raise AssertionError("Hamiltonian is not Hermitian")
    return h_total


def diagonalize_levels(B_t: float) -> pd.DataFrame:
    """Return eigen-energies and expectation values of m_I, m_J, m_F."""
    h_total = build_hamiltonian(B_t)
    eigvals, eigvecs = np.linalg.eigh(h_total)

    op_iz = kron(SZ, I2)
    op_jz = kron(I2, SZ)
    op_fz = op_iz + op_jz

    rows = []
    for idx in range(eigvals.size):
        vec = eigvecs[:, idx]
        m_i = np.real(np.vdot(vec, op_iz @ vec))
        m_j = np.real(np.vdot(vec, op_jz @ vec))
        m_f = np.real(np.vdot(vec, op_fz @ vec))
        rows.append(
            {
                "state": idx,
                "energy_J": float(np.real(eigvals[idx])),
                "energy_MHz": float(np.real(eigvals[idx]) / h / 1e6),
                "mI_expect": float(m_i),
                "mJ_expect": float(m_j),
                "mF_expect": float(m_f),
            }
        )

    return pd.DataFrame(rows).sort_values("energy_J").reset_index(drop=True)


def classify_branches(levels_df: pd.DataFrame) -> dict[str, float]:
    """Classify the four branches by m_F expectation and energy ordering."""
    work = levels_df.copy()

    idx_p1 = (work["mF_expect"] - 1.0).abs().idxmin()
    e_p1 = float(work.loc[idx_p1, "energy_J"])
    work = work.drop(index=idx_p1)

    idx_m1 = (work["mF_expect"] + 1.0).abs().idxmin()
    e_m1 = float(work.loc[idx_m1, "energy_J"])
    work = work.drop(index=idx_m1)

    # Remaining two states are m_F=0 branches: lower (mostly singlet-like), upper (triplet-like)
    zero_sorted = work.sort_values("energy_J")
    e_0_low = float(zero_sorted.iloc[0]["energy_J"])
    e_0_high = float(zero_sorted.iloc[1]["energy_J"])

    return {
        "mF_plus1": e_p1,
        "mF_minus1": e_m1,
        "mF0_low": e_0_low,
        "mF0_high": e_0_high,
    }


def breit_rabi_mf0_analytic(B_t: float) -> tuple[float, float]:
    """Analytic m_F=0 energies from the exact 2x2 block diagonalization."""
    base = -A_HFS_J / 4.0
    delta = (G_J * MU_B + G_P * MU_N) * B_t
    split = 0.5 * np.sqrt(A_HFS_J * A_HFS_J + delta * delta)
    return (base - split, base + split)


def mF_pm1_analytic(B_t: float, m_sign: int) -> float:
    """Analytic energies for m_F=+1 or m_F=-1 pure states."""
    if m_sign not in (-1, 1):
        raise ValueError("m_sign must be -1 or +1")
    zeeman_coeff = 0.5 * (G_J * MU_B - G_P * MU_N) * B_t
    return A_HFS_J / 4.0 + m_sign * zeeman_coeff


def build_transition_table(B_values_t: np.ndarray) -> pd.DataFrame:
    """Compute branch transition frequencies from mF0_low to triplet branches."""
    rows = []
    for B_t in B_values_t:
        if np.isclose(B_t, 0.0):
            nu_center = NU_21CM_HZ
            rows.append(
                {
                    "B_mT": B_t * 1e3,
                    "nu_sigma_minus_MHz": nu_center / 1e6,
                    "nu_pi_MHz": nu_center / 1e6,
                    "nu_sigma_plus_MHz": nu_center / 1e6,
                }
            )
            continue

        levels = diagonalize_levels(B_t)
        br = classify_branches(levels)

        nu_sigma_minus = (br["mF_minus1"] - br["mF0_low"]) / h
        nu_pi = (br["mF0_high"] - br["mF0_low"]) / h
        nu_sigma_plus = (br["mF_plus1"] - br["mF0_low"]) / h

        rows.append(
            {
                "B_mT": B_t * 1e3,
                "nu_sigma_minus_MHz": nu_sigma_minus / 1e6,
                "nu_pi_MHz": nu_pi / 1e6,
                "nu_sigma_plus_MHz": nu_sigma_plus / 1e6,
            }
        )

    return pd.DataFrame(rows)


def run_checks() -> None:
    """Consistency checks for zero-field, analytic formulas, and weak-field behavior."""
    # 1) Zero-field spectrum: singlet (-3A/4) + triplet (+A/4 x3)
    levels0 = diagonalize_levels(0.0)
    e0 = levels0["energy_J"].to_numpy(dtype=float)

    assert np.isclose(e0[0], -0.75 * A_HFS_J, rtol=0.0, atol=1e-30)
    assert np.allclose(e0[1:], 0.25 * A_HFS_J, rtol=0.0, atol=1e-30)

    nu_derived = (np.mean(e0[1:]) - e0[0]) / h
    assert np.isclose(nu_derived, NU_21CM_HZ, rtol=0.0, atol=1e-9)

    # 2) Compare numerical diagonalization against analytic branches at finite B
    B_test = 3.0e-3
    levels_test = diagonalize_levels(B_test)
    br = classify_branches(levels_test)

    e_mf0_low_exact, e_mf0_high_exact = breit_rabi_mf0_analytic(B_test)
    e_p1_exact = mF_pm1_analytic(B_test, +1)
    e_m1_exact = mF_pm1_analytic(B_test, -1)

    assert np.isclose(br["mF0_low"], e_mf0_low_exact, rtol=0.0, atol=1e-28)
    assert np.isclose(br["mF0_high"], e_mf0_high_exact, rtol=0.0, atol=1e-28)
    assert np.isclose(br["mF_plus1"], e_p1_exact, rtol=0.0, atol=1e-28)
    assert np.isclose(br["mF_minus1"], e_m1_exact, rtol=0.0, atol=1e-28)

    # 3) Weak-field first-order Zeeman shift for sigma branches
    B_small = 1.0e-5
    levels_small = diagonalize_levels(B_small)
    br_small = classify_branches(levels_small)
    nu_minus = (br_small["mF_minus1"] - br_small["mF0_low"]) / h
    nu_plus = (br_small["mF_plus1"] - br_small["mF0_low"]) / h

    slope_hz_per_t = 0.5 * (G_J * MU_B - G_P * MU_N) / h
    expected_minus = NU_21CM_HZ - slope_hz_per_t * B_small
    expected_plus = NU_21CM_HZ + slope_hz_per_t * B_small

    # First-order approximation ignores the small quadratic correction from mF=0 mixing.
    assert np.isclose(nu_minus, expected_minus, rtol=0.0, atol=20.0)
    assert np.isclose(nu_plus, expected_plus, rtol=0.0, atol=20.0)


def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    B_demo = 1.0e-3
    levels_demo = diagonalize_levels(B_demo)

    B_values = np.array([0.0, 0.001, 0.002, 0.005, 0.010], dtype=float)
    transition_table = build_transition_table(B_values)

    print("Hydrogen hyperfine splitting MVP (1S1/2)")
    print(f"nu_21cm_reference = {NU_21CM_HZ:.3f} Hz")
    print(f"A_hfs = h * nu = {A_HFS_J:.9e} J")
    print(f"g_J={G_J:.12f}, g_p={G_P:.12f}, mu_B={MU_B:.9e} J/T, mu_N={MU_N:.9e} J/T")

    print(f"\n[Eigen-levels at B={B_demo*1e3:.3f} mT]")
    print(levels_demo.to_string(index=False, float_format=lambda x: f"{x: .9f}"))

    print("\n[Transition frequencies from mF0_low branch]")
    print(transition_table.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    run_checks()
    print("All checks passed.")


if __name__ == "__main__":
    main()
