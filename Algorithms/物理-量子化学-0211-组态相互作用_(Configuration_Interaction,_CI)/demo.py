"""Minimal, runnable Configuration Interaction (CI) MVP.

This demo implements a tiny fermionic CI solver from scratch:
- determinant basis from bitstrings,
- second-quantized Hamiltonian action via creation/annihilation operators,
- explicit CI matrix build and diagonalization.

No interactive input is required.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import numpy as np

Determinant = int


def determinant_from_occupied(occupied_orbitals: Sequence[int]) -> Determinant:
    det = 0
    for orb in occupied_orbitals:
        det |= 1 << orb
    return det


def occupied_orbitals(det: Determinant, n_orb: int) -> List[int]:
    return [p for p in range(n_orb) if (det >> p) & 1]


def format_determinant(det: Determinant, n_orb: int) -> str:
    bits = "".join("1" if (det >> p) & 1 else "0" for p in reversed(range(n_orb)))
    occ = occupied_orbitals(det, n_orb)
    return f"|{bits}> occ={occ}"


def apply_annihilation(det: Determinant, orb: int) -> Optional[Tuple[Determinant, float]]:
    if ((det >> orb) & 1) == 0:
        return None
    lower_mask = (1 << orb) - 1
    parity = (det & lower_mask).bit_count()
    sign = -1.0 if parity % 2 else 1.0
    new_det = det & ~(1 << orb)
    return new_det, sign


def apply_creation(det: Determinant, orb: int) -> Optional[Tuple[Determinant, float]]:
    if ((det >> orb) & 1) == 1:
        return None
    lower_mask = (1 << orb) - 1
    parity = (det & lower_mask).bit_count()
    sign = -1.0 if parity % 2 else 1.0
    new_det = det | (1 << orb)
    return new_det, sign


def one_body_transition(det: Determinant, p: int, q: int) -> Optional[Tuple[Determinant, float]]:
    # a_p^\dagger a_q acts right-to-left on |det>.
    ann = apply_annihilation(det, q)
    if ann is None:
        return None
    det1, s1 = ann
    cre = apply_creation(det1, p)
    if cre is None:
        return None
    det2, s2 = cre
    return det2, s1 * s2


def two_body_transition(
    det: Determinant,
    p: int,
    q: int,
    r: int,
    s: int,
) -> Optional[Tuple[Determinant, float]]:
    # a_p^\dagger a_q^\dagger a_s a_r acts right-to-left on |det>.
    ann_r = apply_annihilation(det, r)
    if ann_r is None:
        return None
    det1, s1 = ann_r

    ann_s = apply_annihilation(det1, s)
    if ann_s is None:
        return None
    det2, s2 = ann_s

    cre_q = apply_creation(det2, q)
    if cre_q is None:
        return None
    det3, s3 = cre_q

    cre_p = apply_creation(det3, p)
    if cre_p is None:
        return None
    det4, s4 = cre_p

    return det4, s1 * s2 * s3 * s4


def ci_matrix_element(
    det_i: Determinant,
    det_j: Determinant,
    h1: np.ndarray,
    g2: np.ndarray,
    n_orb: int,
) -> float:
    value = 0.0

    for p in range(n_orb):
        for q in range(n_orb):
            trans = one_body_transition(det_j, p, q)
            if trans is None:
                continue
            det_k, sign = trans
            if det_k == det_i:
                value += float(h1[p, q]) * sign

    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    trans = two_body_transition(det_j, p, q, r, s)
                    if trans is None:
                        continue
                    det_k, sign = trans
                    if det_k == det_i:
                        value += 0.25 * float(g2[p, q, r, s]) * sign

    return value


def build_determinants(n_orb: int, n_elec: int) -> List[Determinant]:
    if n_orb <= 0:
        raise ValueError("n_orb must be > 0")
    if not (0 < n_elec <= n_orb):
        raise ValueError("n_elec must satisfy 0 < n_elec <= n_orb")
    return [determinant_from_occupied(c) for c in combinations(range(n_orb), n_elec)]


def build_cis_subspace(reference_det: Determinant, n_orb: int) -> List[Determinant]:
    occ = occupied_orbitals(reference_det, n_orb)
    virt = [p for p in range(n_orb) if p not in occ]

    singles = set()
    for i in occ:
        for a in virt:
            ann = apply_annihilation(reference_det, i)
            if ann is None:
                continue
            det1, _ = ann
            cre = apply_creation(det1, a)
            if cre is None:
                continue
            det2, _ = cre
            singles.add(det2)

    return [reference_det] + sorted(singles)


def build_ci_hamiltonian(
    determinants: Sequence[Determinant],
    h1: np.ndarray,
    g2: np.ndarray,
    n_orb: int,
) -> np.ndarray:
    n_det = len(determinants)
    h_ci = np.zeros((n_det, n_det), dtype=float)

    for i, det_i in enumerate(determinants):
        for j, det_j in enumerate(determinants):
            h_ci[i, j] = ci_matrix_element(det_i, det_j, h1, g2, n_orb)

    return h_ci


def make_toy_integrals(n_orb: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    h_raw = rng.normal(loc=0.0, scale=0.20, size=(n_orb, n_orb))
    h1 = 0.5 * (h_raw + h_raw.T)
    for p in range(n_orb):
        h1[p, p] += 0.6 * p

    v = rng.normal(loc=0.0, scale=0.05, size=(n_orb, n_orb, n_orb, n_orb))

    # Build antisymmetrized two-electron integrals <pq||rs> with pair symmetry.
    g2 = v - np.transpose(v, (0, 1, 3, 2))
    g2 = 0.5 * (g2 - np.transpose(g2, (1, 0, 2, 3)))
    g2 = 0.5 * (g2 - np.transpose(g2, (0, 1, 3, 2)))
    g2 = 0.5 * (g2 + np.transpose(g2, (2, 3, 0, 1)))

    return h1, g2


def diagonalize(h_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if h_mat.ndim != 2 or h_mat.shape[0] != h_mat.shape[1]:
        raise ValueError("Hamiltonian must be square.")
    if not np.all(np.isfinite(h_mat)):
        raise ValueError("Hamiltonian contains non-finite values.")
    eigvals, eigvecs = np.linalg.eigh(h_mat)
    return eigvals, eigvecs


def top_coefficients(
    coeffs: np.ndarray,
    determinants: Sequence[Determinant],
    n_orb: int,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    pairs = []
    for idx, amp in enumerate(coeffs):
        pairs.append((format_determinant(determinants[idx], n_orb), float(amp)))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_k]


def main() -> None:
    n_orb = 6
    n_elec = 3

    h1, g2 = make_toy_integrals(n_orb=n_orb, seed=42)

    all_dets = build_determinants(n_orb=n_orb, n_elec=n_elec)

    ref_orbs = list(np.argsort(np.diag(h1))[:n_elec])
    reference_det = determinant_from_occupied(ref_orbs)
    cis_dets = build_cis_subspace(reference_det=reference_det, n_orb=n_orb)

    h_fci = build_ci_hamiltonian(all_dets, h1, g2, n_orb)
    h_cis = build_ci_hamiltonian(cis_dets, h1, g2, n_orb)

    e_fci, c_fci = diagonalize(h_fci)
    e_cis, _ = diagonalize(h_cis)

    e0_fci = float(e_fci[0])
    e0_cis = float(e_cis[0])

    ref_idx = all_dets.index(reference_det)
    e_ref = float(h_fci[ref_idx, ref_idx])

    herm_res_fci = float(np.linalg.norm(h_fci - h_fci.T))
    gs_vec = c_fci[:, 0]
    eig_res_fci = float(np.linalg.norm(h_fci @ gs_vec - e0_fci * gs_vec))

    print("=== Configuration Interaction MVP (Toy Spin-Orbital Model) ===")
    print(f"n_orb={n_orb}, n_elec={n_elec}")
    print(f"FCI determinant count: {len(all_dets)}")
    print(f"CIS determinant count: {len(cis_dets)}")
    print(f"Reference determinant: {format_determinant(reference_det, n_orb)}")

    print("\n--- Energies ---")
    print(f"Reference diagonal energy   : {e_ref: .10f}")
    print(f"CIS ground-state energy     : {e0_cis: .10f}")
    print(f"FCI ground-state energy     : {e0_fci: .10f}")
    print(f"Correlation gain (Ref-FCI)  : {e_ref - e0_fci: .10f}")
    print(f"CIS-into-FCI improvement    : {e0_cis - e0_fci: .10f}")

    print("\n--- Numerical checks ---")
    print(f"Hermiticity residual ||H-H^T||: {herm_res_fci:.3e}")
    print(f"Eigen residual ||Hc-Ec||      : {eig_res_fci:.3e}")

    check_variational = e0_fci <= e0_cis + 1e-10 and e0_fci <= e_ref + 1e-10
    check_hermitian = herm_res_fci < 1e-9
    check_eigenpair = eig_res_fci < 1e-8

    print("\n--- Variational checks ---")
    print(f"E_FCI <= E_CIS and E_ref: {check_variational}")
    print(f"Hamiltonian is symmetric : {check_hermitian}")
    print(f"Ground eigenpair residual: {check_eigenpair}")

    print("\n--- Top FCI amplitudes ---")
    for det_label, amp in top_coefficients(gs_vec, all_dets, n_orb=n_orb, top_k=5):
        print(f"{det_label:30s} coeff={amp:+.6f}")


if __name__ == "__main__":
    main()
