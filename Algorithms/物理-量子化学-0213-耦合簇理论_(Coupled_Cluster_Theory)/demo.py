"""Minimal runnable MVP for Coupled Cluster Theory.

This script demonstrates projected CCSD equations in a tiny fixed-electron
spin-orbital space. The implementation is explicit and auditable:
- build a second-quantized Hamiltonian in determinant basis,
- construct single/double excitation operators,
- solve projected equations <Phi_mu|exp(-T) H exp(T)|Phi0> = 0,
- compare CC energy with FCI energy on the same toy model.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.optimize import root

Determinant = int


@dataclass(frozen=True)
class Excitation:
    name: str
    occ: Tuple[int, ...]
    virt: Tuple[int, ...]
    target_det: Determinant


def determinant_from_occupied(occupied_orbitals: Sequence[int]) -> Determinant:
    det = 0
    for orb in occupied_orbitals:
        det |= 1 << orb
    return det


def occupied_orbitals(det: Determinant, n_orb: int) -> List[int]:
    return [p for p in range(n_orb) if (det >> p) & 1]


def format_determinant(det: Determinant, n_orb: int) -> str:
    bits = "".join("1" if (det >> p) & 1 else "0" for p in reversed(range(n_orb)))
    return f"|{bits}>"


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
    # a_p^† a_q acts right-to-left on |det>.
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
    # a_p^† a_q^† a_s a_r acts right-to-left on |det>.
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


def build_hamiltonian(
    basis: Sequence[Determinant],
    h1: np.ndarray,
    g2: np.ndarray,
    n_orb: int,
) -> np.ndarray:
    dim = len(basis)
    h_mat = np.zeros((dim, dim), dtype=float)
    for i, det_i in enumerate(basis):
        for j, det_j in enumerate(basis):
            h_mat[i, j] = ci_matrix_element(det_i, det_j, h1, g2, n_orb)
    return h_mat


def apply_excitation(
    det: Determinant,
    occ_from: Sequence[int],
    virt_to: Sequence[int],
) -> Optional[Tuple[Determinant, float]]:
    """Apply tau = a_a^†...a_b^† a_j...a_i to |det>.

    Convention:
    - occ_from sorted ascending, annihilation applied in that order.
    - virt_to sorted ascending, creation applied in reverse order.
    This corresponds to the usual excitation operator ordering.
    """
    if len(occ_from) != len(virt_to):
        raise ValueError("excitation rank mismatch")

    current = det
    sign = 1.0

    for i in occ_from:
        out = apply_annihilation(current, i)
        if out is None:
            return None
        current, s = out
        sign *= s

    for a in reversed(virt_to):
        out = apply_creation(current, a)
        if out is None:
            return None
        current, s = out
        sign *= s

    return current, sign


def enumerate_ccsd_excitations(reference_det: Determinant, n_orb: int) -> List[Excitation]:
    occ = occupied_orbitals(reference_det, n_orb)
    virt = [p for p in range(n_orb) if p not in occ]

    ex_list: List[Excitation] = []

    for i in occ:
        for a in virt:
            res = apply_excitation(reference_det, (i,), (a,))
            if res is None:
                continue
            target_det, _ = res
            ex_list.append(
                Excitation(
                    name=f"S: {i}->{a}",
                    occ=(i,),
                    virt=(a,),
                    target_det=target_det,
                )
            )

    for i, j in combinations(occ, 2):
        for a, b in combinations(virt, 2):
            res = apply_excitation(reference_det, (i, j), (a, b))
            if res is None:
                continue
            target_det, _ = res
            ex_list.append(
                Excitation(
                    name=f"D: {i},{j}->{a},{b}",
                    occ=(i, j),
                    virt=(a, b),
                    target_det=target_det,
                )
            )

    return ex_list


def build_excitation_operator(
    excitation: Excitation,
    basis: Sequence[Determinant],
    index_of: Dict[Determinant, int],
) -> np.ndarray:
    dim = len(basis)
    op = np.zeros((dim, dim), dtype=float)

    for col, det_j in enumerate(basis):
        out = apply_excitation(det_j, excitation.occ, excitation.virt)
        if out is None:
            continue
        det_k, sign = out
        row = index_of.get(det_k)
        if row is not None:
            op[row, col] = sign

    return op


def make_toy_integrals(n_orb: int, seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    eps = np.linspace(-1.2, 0.9, n_orb)
    h1 = np.diag(eps)

    mix = rng.normal(loc=0.0, scale=0.03, size=(n_orb, n_orb))
    mix = 0.5 * (mix + mix.T)
    np.fill_diagonal(mix, 0.0)
    h1 = h1 + mix

    v = rng.normal(loc=0.0, scale=0.08, size=(n_orb, n_orb, n_orb, n_orb))

    g2 = v - np.transpose(v, (0, 1, 3, 2))
    g2 = 0.5 * (g2 - np.transpose(g2, (1, 0, 2, 3)))
    g2 = 0.5 * (g2 + np.transpose(g2, (2, 3, 0, 1)))

    return h1, g2


def assemble_cluster_operator(t_amp: np.ndarray, tau_ops: Sequence[np.ndarray]) -> np.ndarray:
    if len(t_amp) != len(tau_ops):
        raise ValueError("amplitude/operator size mismatch")
    dim = tau_ops[0].shape[0]
    t_mat = np.zeros((dim, dim), dtype=float)
    for coeff, op in zip(t_amp, tau_ops):
        t_mat += float(coeff) * op
    return t_mat


def cc_energy_and_residual(
    t_amp: np.ndarray,
    h_mat: np.ndarray,
    tau_ops: Sequence[np.ndarray],
    ref_idx: int,
    proj_indices: Sequence[int],
) -> Tuple[float, np.ndarray]:
    t_mat = assemble_cluster_operator(t_amp, tau_ops)
    u = expm(t_mat)
    u_inv = expm(-t_mat)

    h_bar = u_inv @ h_mat @ u

    energy = float(np.real(h_bar[ref_idx, ref_idx]))
    residual = np.array([float(np.real(h_bar[idx, ref_idx])) for idx in proj_indices], dtype=float)
    return energy, residual


def initial_guess(h_mat: np.ndarray, ref_idx: int, proj_indices: Sequence[int]) -> np.ndarray:
    t0 = np.zeros(len(proj_indices), dtype=float)
    e_ref = float(h_mat[ref_idx, ref_idx])
    for k, idx in enumerate(proj_indices):
        coupling = float(h_mat[idx, ref_idx])
        denom = e_ref - float(h_mat[idx, idx])
        if abs(denom) > 1e-10:
            t0[k] = -coupling / denom
    return t0


def solve_projected_cc(
    h_mat: np.ndarray,
    tau_ops: Sequence[np.ndarray],
    ref_idx: int,
    proj_indices: Sequence[int],
) -> Tuple[np.ndarray, bool, str, float, np.ndarray, np.ndarray]:
    t0 = initial_guess(h_mat, ref_idx, proj_indices)

    def residual_only(t_amp: np.ndarray) -> np.ndarray:
        _, res = cc_energy_and_residual(t_amp, h_mat, tau_ops, ref_idx, proj_indices)
        return res

    sol = root(residual_only, t0, method="hybr", tol=1e-12)
    if (not sol.success) or np.linalg.norm(sol.fun) > 1e-10:
        sol = root(residual_only, t0, method="lm", tol=1e-12)

    t_opt = np.array(sol.x, dtype=float)
    energy, residual = cc_energy_and_residual(t_opt, h_mat, tau_ops, ref_idx, proj_indices)
    return t_opt, bool(sol.success), str(sol.message), energy, residual, t0


def main() -> None:
    n_orb = 4
    n_elec = 2

    h1, g2 = make_toy_integrals(n_orb=n_orb, seed=7)

    basis = build_determinants(n_orb=n_orb, n_elec=n_elec)
    index_of = {det: idx for idx, det in enumerate(basis)}

    h_mat = build_hamiltonian(basis, h1, g2, n_orb)
    herm_res = float(np.linalg.norm(h_mat - h_mat.T))

    eigvals, _ = np.linalg.eigh(h_mat)
    e_fci = float(eigvals[0])

    ref_occ = [int(x) for x in np.argsort(np.diag(h1))[:n_elec]]
    reference_det = determinant_from_occupied(ref_occ)
    ref_idx = index_of[reference_det]

    excitations = enumerate_ccsd_excitations(reference_det, n_orb=n_orb)
    proj_indices = [index_of[ex.target_det] for ex in excitations]
    tau_ops = [build_excitation_operator(ex, basis, index_of) for ex in excitations]

    t_opt, success, message, e_cc, residual, t0 = solve_projected_cc(
        h_mat=h_mat,
        tau_ops=tau_ops,
        ref_idx=ref_idx,
        proj_indices=proj_indices,
    )

    res_norm = float(np.linalg.norm(residual))

    print("=== Coupled Cluster Theory MVP (Projected CCSD on Toy Space) ===")
    print(f"n_orb={n_orb}, n_elec={n_elec}, determinant_dim={len(basis)}")
    print(f"Reference determinant: {format_determinant(reference_det, n_orb)} occ={ref_occ}")
    print(f"Excitation count (S+D): {len(excitations)}")

    print("\n--- Solver status ---")
    print(f"Nonlinear solver success : {success}")
    print(f"Solver message           : {message}")
    print(f"Initial residual norm    : {np.linalg.norm(cc_energy_and_residual(t0, h_mat, tau_ops, ref_idx, proj_indices)[1]):.3e}")
    print(f"Final residual norm      : {res_norm:.3e}")

    print("\n--- Energies ---")
    print(f"CC projected energy      : {e_cc: .10f}")
    print(f"FCI benchmark energy     : {e_fci: .10f}")
    print(f"|E_CC - E_FCI|           : {abs(e_cc - e_fci):.3e}")

    print("\n--- Numerical checks ---")
    print(f"Hermiticity ||H-H^T||    : {herm_res:.3e}")
    print(f"Residual below 1e-8      : {res_norm < 1e-8}")
    print(f"CC close to FCI (1e-7)   : {abs(e_cc - e_fci) < 1e-7}")

    print("\n--- Final amplitudes ---")
    for ex, amp in zip(excitations, t_opt):
        print(f"{ex.name:15s} target={format_determinant(ex.target_det, n_orb):6s} t={amp:+.8f}")


if __name__ == "__main__":
    main()
