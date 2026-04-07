"""Minimal runnable MVP for spin-orbit coupling in hydrogenic atoms.

This script demonstrates:
1) source-level construction/diagonalization of L·S in uncoupled basis,
2) hydrogenic spin-orbit shift DeltaE_so(n,l,j,Z) in atomic units,
3) Z^4 scaling validation with sklearn/scipy/(optional torch),
4) tabular spectrum output for quick inspection.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.constants import physical_constants
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None


# Atomic-unit speed of light c = 1/alpha.
C_AU = 137.035999084
HARTREE_TO_EV = float(physical_constants["Hartree energy in eV"][0])


def validate_n_l_j(n: int, l: int, j: float) -> None:
    if n < 1:
        raise ValueError("n must be >= 1")
    if l < 0 or l >= n:
        raise ValueError("l must satisfy 0 <= l < n")
    allowed = allowed_j_values(l)
    if all(abs(j - a) > 1e-12 for a in allowed):
        raise ValueError(f"invalid j={j} for l={l}, allowed={allowed}")


def allowed_j_values(l: int) -> list[float]:
    if l < 0:
        raise ValueError("l must be >= 0")
    if l == 0:
        return [0.5]
    return [l - 0.5, l + 0.5]


def ls_expectation(l: int, j: float, s: float = 0.5) -> float:
    return 0.5 * (j * (j + 1.0) - l * (l + 1.0) - s * (s + 1.0))


def radial_inverse_r3_expectation_hydrogenic(n: int, l: int, z: int) -> float:
    """<r^-3> for hydrogen-like atom in atomic units (l > 0).

    Formula:
        <r^-3> = Z^3 / [n^3 * l * (l + 1/2) * (l + 1)].
    """
    if z < 1:
        raise ValueError("z must be >= 1")
    if n < 1:
        raise ValueError("n must be >= 1")
    if l <= 0:
        raise ValueError("formula is only valid for l > 0")
    if l >= n:
        raise ValueError("l must satisfy l < n")
    denom = (n**3) * l * (l + 0.5) * (l + 1.0)
    return (z**3) / denom


def spin_orbit_shift_au(n: int, l: int, j: float, z: int, c_au: float = C_AU) -> float:
    """Hydrogenic spin-orbit energy correction in Hartree.

    DeltaE_so = [Z / (2 c^2)] * <r^-3> * <L·S>.
    For l=0, L=0 => DeltaE_so = 0.
    """
    validate_n_l_j(n=n, l=l, j=j)
    if z < 1:
        raise ValueError("z must be >= 1")
    if l == 0:
        return 0.0
    inv_r3 = radial_inverse_r3_expectation_hydrogenic(n=n, l=l, z=z)
    xi = (z / (2.0 * c_au**2)) * inv_r3
    return xi * ls_expectation(l=l, j=j)


def angular_momentum_matrices(j: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Jx, Jy, Jz) for angular momentum quantum number j."""
    m_values = np.arange(j, -j - 1.0, -1.0, dtype=np.float64)
    dim = m_values.size

    jp = np.zeros((dim, dim), dtype=np.complex128)
    jm = np.zeros((dim, dim), dtype=np.complex128)
    jj1 = j * (j + 1.0)

    for col, m in enumerate(m_values):
        # J+ |j,m> -> |j,m+1>. In descending m ordering, row index is col-1.
        if col > 0:
            coeff = np.sqrt(max(jj1 - m * (m + 1.0), 0.0))
            jp[col - 1, col] = coeff
        # J- |j,m> -> |j,m-1>. In descending m ordering, row index is col+1.
        if col < dim - 1:
            coeff = np.sqrt(max(jj1 - m * (m - 1.0), 0.0))
            jm[col + 1, col] = coeff

    jx = 0.5 * (jp + jm)
    jy = -0.5j * (jp - jm)
    jz = np.diag(m_values.astype(np.complex128))
    return jx, jy, jz


def l_dot_s_matrix_uncoupled(l: int, s: float = 0.5) -> np.ndarray:
    """Build L·S matrix in uncoupled basis |l,m_l> ⊗ |s,m_s>."""
    lx, ly, lz = angular_momentum_matrices(float(l))
    sx, sy, sz = angular_momentum_matrices(float(s))

    i_l = np.eye(lx.shape[0], dtype=np.complex128)
    i_s = np.eye(sx.shape[0], dtype=np.complex128)

    lx_full = np.kron(lx, i_s)
    ly_full = np.kron(ly, i_s)
    lz_full = np.kron(lz, i_s)
    sx_full = np.kron(i_l, sx)
    sy_full = np.kron(i_l, sy)
    sz_full = np.kron(i_l, sz)

    return lx_full @ sx_full + ly_full @ sy_full + lz_full @ sz_full


def expected_ls_eigenvalues(l: int) -> np.ndarray:
    values: list[float] = []
    for j in allowed_j_values(l):
        eig = ls_expectation(l=l, j=j)
        multiplicity = int(round(2.0 * j + 1.0))
        values.extend([eig] * multiplicity)
    return np.array(sorted(values), dtype=np.float64)


def fit_scaling_sklearn(z_values: np.ndarray, shifts: np.ndarray) -> float:
    x = (z_values.astype(np.float64) ** 4).reshape(-1, 1)
    model = LinearRegression(fit_intercept=False)
    model.fit(x, shifts)
    return float(model.coef_[0])


def fit_scaling_curve_fit(z_values: np.ndarray, shifts: np.ndarray) -> float:
    def model(x: np.ndarray, kappa: float) -> np.ndarray:
        return kappa * (x**4)

    kappa, _ = curve_fit(model, z_values.astype(np.float64), shifts, p0=np.array([1.0e-7], dtype=np.float64))
    return float(kappa[0])


def fit_scaling_torch_optional(z_values: np.ndarray, shifts: np.ndarray) -> Optional[float]:
    if torch is None:
        return None

    x = torch.tensor((z_values.astype(np.float64) ** 4), dtype=torch.float64)
    y = torch.tensor(shifts.astype(np.float64), dtype=torch.float64)
    # Closed-form least squares for y = kappa * x (no intercept).
    denom = torch.dot(x, x)
    if torch.abs(denom).item() == 0.0:
        return 0.0
    kappa = torch.dot(x, y) / denom
    return float(kappa.cpu().item())


def build_spectrum_table(z_values: np.ndarray, n_values: list[int]) -> pd.DataFrame:
    rows = []
    for z in z_values:
        zi = int(z)
        for n in n_values:
            for l in range(1, n):
                for j in allowed_j_values(l):
                    e0 = -(zi**2) / (2.0 * (n**2))
                    delta = spin_orbit_shift_au(n=n, l=l, j=j, z=zi)
                    total = e0 + delta
                    rows.append(
                        {
                            "Z": zi,
                            "n": n,
                            "l": l,
                            "j": j,
                            "degeneracy_2j1": int(round(2.0 * j + 1.0)),
                            "E0_au": e0,
                            "DeltaE_SO_au": delta,
                            "DeltaE_SO_meV": delta * HARTREE_TO_EV * 1.0e3,
                            "E_total_au": total,
                        }
                    )
    table = pd.DataFrame(rows).sort_values(["Z", "n", "l", "j"]).reset_index(drop=True)
    return table


def build_doublet_splitting_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (z, n, l), group in table.groupby(["Z", "n", "l"], sort=True):
        if group.shape[0] < 2:
            continue
        split = float(group["E_total_au"].max() - group["E_total_au"].min())
        rows.append(
            {
                "Z": int(z),
                "n": int(n),
                "l": int(l),
                "split_au": split,
                "split_meV": split * HARTREE_TO_EV * 1.0e3,
            }
        )
    return pd.DataFrame(rows).sort_values(["Z", "n", "l"]).reset_index(drop=True)


def main() -> None:
    # 1) L·S matrix sanity check for l=1.
    ls_mat = l_dot_s_matrix_uncoupled(l=1)
    ls_eig = np.linalg.eigvalsh(ls_mat).real
    ls_expected = expected_ls_eigenvalues(l=1)
    max_ls_err = float(np.max(np.abs(np.sort(ls_eig) - ls_expected)))

    # 2) Build compact spectrum table.
    z_values = np.arange(1, 7, dtype=np.int64)
    n_values = [2, 3, 4]
    spectrum = build_spectrum_table(z_values=z_values, n_values=n_values)
    splits = build_doublet_splitting_table(spectrum)

    # 3) Check Z^4 scaling on a fixed state (n=2, l=1, j=3/2).
    target_n = 2
    target_l = 1
    target_j = 1.5
    target_shifts = np.array(
        [spin_orbit_shift_au(n=target_n, l=target_l, j=target_j, z=int(z)) for z in z_values],
        dtype=np.float64,
    )
    expected_kappa = float(target_shifts[0] / (z_values[0] ** 4))
    kappa_sk = fit_scaling_sklearn(z_values=z_values.astype(np.float64), shifts=target_shifts)
    kappa_cf = fit_scaling_curve_fit(z_values=z_values.astype(np.float64), shifts=target_shifts)
    kappa_torch = fit_scaling_torch_optional(z_values=z_values.astype(np.float64), shifts=target_shifts)

    # 4) Hydrogen n=2, l=1 splitting check (j=1/2 vs 3/2).
    h_2p_j32 = spin_orbit_shift_au(n=2, l=1, j=1.5, z=1)
    h_2p_j12 = spin_orbit_shift_au(n=2, l=1, j=0.5, z=1)
    h_2p_split = h_2p_j32 - h_2p_j12

    print("Spin-Orbit Coupling MVP (hydrogenic atom, atomic units)")
    print(f"LdotS_eigen_max_error_l1={max_ls_err:.3e}")
    print(f"kappa_expected={expected_kappa:.12e}")
    print(f"kappa_sklearn={kappa_sk:.12e}")
    print(f"kappa_curve_fit={kappa_cf:.12e}")
    if kappa_torch is None:
        print("kappa_torch=unavailable")
    else:
        print(f"kappa_torch={kappa_torch:.12e}")
    print(f"H_2p_j3/2_shift_au={h_2p_j32:.12e}")
    print(f"H_2p_j1/2_shift_au={h_2p_j12:.12e}")
    print(f"H_2p_doublet_split_au={h_2p_split:.12e}")
    print(f"H_2p_doublet_split_meV={h_2p_split * HARTREE_TO_EV * 1.0e3:.6f}")
    print("spectrum_preview=")
    print(spectrum.head(10).to_string(index=False))
    print("doublet_splitting_preview=")
    print(splits.head(10).to_string(index=False))

    # Core correctness checks.
    assert spin_orbit_shift_au(n=2, l=0, j=0.5, z=1) == 0.0, "l=0 state should have zero SO shift"
    assert max_ls_err < 1.0e-12, f"L·S eigenvalue mismatch: {max_ls_err}"

    unique_vals, counts = np.unique(np.round(np.sort(ls_eig), decimals=12), return_counts=True)
    assert unique_vals.size == 2, "l=1 L·S should have two distinct eigenvalues"
    assert set(counts.tolist()) == {2, 4}, f"unexpected L·S degeneracies: counts={counts.tolist()}"

    assert h_2p_j32 > 0.0 and h_2p_j12 < 0.0, "2p j-split sign pattern is wrong"
    assert h_2p_split > 0.0, "2p fine-structure splitting must be positive"

    assert abs(kappa_sk - expected_kappa) < 1.0e-18, "sklearn Z^4 scaling mismatch"
    assert abs(kappa_cf - expected_kappa) < 1.0e-18, "curve_fit Z^4 scaling mismatch"
    if kappa_torch is not None:
        assert abs(kappa_torch - expected_kappa) < 1.0e-18, "torch Z^4 scaling mismatch"

    assert not spectrum.empty and not splits.empty, "output tables should not be empty"

    print("All checks passed.")


if __name__ == "__main__":
    main()
