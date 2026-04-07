"""Minimal runnable MVP for Many-Body Perturbation Theory (PHYS-0213).

This demo implements a non-interacting reference + two-body perturbation model
in a finite fermionic Fock space:
1) build H0 and interaction V in a fixed-N Slater determinant basis
2) compute Rayleigh-Schrödinger coefficients E0, E1, E2 from matrix elements
3) compare MBPT(2) energy E(λ) = E0 + λE1 + λ^2E2 against exact diagonalization
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.linalg import eigh


@dataclass(frozen=True)
class MBPTConfig:
    """Configuration for a finite-basis MBPT(2) toy model."""

    n_orbitals: int = 6
    n_particles: int = 3

    orbital_energies: tuple[float, ...] = (-1.55, -1.00, -0.45, 0.20, 0.95, 1.70)

    interaction_strength: float = 0.24
    interaction_range: float = 1.30

    lambda_max: float = 0.55
    n_lambda: int = 12
    quality_lambda: float = 0.35


def validate_config(cfg: MBPTConfig) -> None:
    if cfg.n_orbitals < 4:
        raise ValueError("n_orbitals must be >= 4")
    if not (1 <= cfg.n_particles < cfg.n_orbitals):
        raise ValueError("n_particles must satisfy 1 <= n_particles < n_orbitals")
    if len(cfg.orbital_energies) != cfg.n_orbitals:
        raise ValueError("orbital_energies length must equal n_orbitals")
    if cfg.interaction_strength <= 0.0:
        raise ValueError("interaction_strength must be positive")
    if cfg.interaction_range <= 0.0:
        raise ValueError("interaction_range must be positive")
    if cfg.lambda_max <= 0.0:
        raise ValueError("lambda_max must be positive")
    if cfg.n_lambda < 5:
        raise ValueError("n_lambda must be >= 5")
    if cfg.quality_lambda <= 0.0 or cfg.quality_lambda > cfg.lambda_max:
        raise ValueError("quality_lambda must be in (0, lambda_max]")


def build_fock_basis(cfg: MBPTConfig) -> list[int]:
    """Return fixed-N determinants as bitstrings."""
    basis: list[int] = []
    for occ in combinations(range(cfg.n_orbitals), cfg.n_particles):
        state = 0
        for p in occ:
            state |= 1 << p
        basis.append(state)
    return basis


def orbital_occupations(state: int, n_orbitals: int) -> list[int]:
    return [p for p in range(n_orbitals) if (state >> p) & 1]


def _fermion_phase_before(state: int, orbital: int) -> int:
    lower_mask = (1 << orbital) - 1
    parity = (state & lower_mask).bit_count() % 2
    return -1 if parity else 1


def annihilate(state: int, orbital: int) -> tuple[int, int] | None:
    if not ((state >> orbital) & 1):
        return None
    sign = _fermion_phase_before(state, orbital)
    return state ^ (1 << orbital), sign


def create(state: int, orbital: int) -> tuple[int, int] | None:
    if (state >> orbital) & 1:
        return None
    sign = _fermion_phase_before(state, orbital)
    return state | (1 << orbital), sign


def apply_two_body_operator(
    state: int,
    p: int,
    q: int,
    r: int,
    s: int,
) -> tuple[int, int] | None:
    """Apply a_p^\\dagger a_q^\\dagger a_s a_r to |state>."""
    out = annihilate(state, r)
    if out is None:
        return None
    cur_state, sign = out

    out = annihilate(cur_state, s)
    if out is None:
        return None
    cur_state, sign2 = out
    sign *= sign2

    out = create(cur_state, q)
    if out is None:
        return None
    cur_state, sign2 = out
    sign *= sign2

    out = create(cur_state, p)
    if out is None:
        return None
    cur_state, sign2 = out
    sign *= sign2

    return cur_state, sign


def build_antisymmetrized_interaction(cfg: MBPTConfig) -> np.ndarray:
    """Construct <pq||rs> from a smooth base interaction tensor."""
    n = cfg.n_orbitals
    raw = np.zeros((n, n, n, n), dtype=float)

    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    distance = abs(p - r) + abs(q - s)
                    raw[p, q, r, s] = cfg.interaction_strength * np.exp(
                        -distance / cfg.interaction_range
                    )

    # Hermitian symmetrization: V_{pqrs} = V_{rspq}
    raw = 0.5 * (raw + raw.transpose(2, 3, 0, 1))

    # Antisymmetrization on bra and ket index pairs.
    g = (
        raw
        - raw.transpose(0, 1, 3, 2)
        - raw.transpose(1, 0, 2, 3)
        + raw.transpose(1, 0, 3, 2)
    )

    # Re-enforce Hermiticity after antisymmetrization.
    g = 0.5 * (g + g.transpose(2, 3, 0, 1))
    return g


def build_h0_matrix(cfg: MBPTConfig, basis: list[int]) -> np.ndarray:
    eps = np.asarray(cfg.orbital_energies, dtype=float)
    dim = len(basis)
    h0 = np.zeros((dim, dim), dtype=float)

    for i, state in enumerate(basis):
        occ = orbital_occupations(state, cfg.n_orbitals)
        h0[i, i] = float(np.sum(eps[occ]))

    return h0


def build_interaction_matrix(
    basis: list[int],
    interaction_tensor: np.ndarray,
) -> np.ndarray:
    """Build many-body matrix of V = 1/4 sum_{pqrs}<pq||rs>a†_p a†_q a_s a_r."""
    dim = len(basis)
    n_orbitals = interaction_tensor.shape[0]
    basis_index = {state: i for i, state in enumerate(basis)}

    v_mat = np.zeros((dim, dim), dtype=float)

    for col, state in enumerate(basis):
        for p in range(n_orbitals):
            for q in range(n_orbitals):
                for r in range(n_orbitals):
                    for s in range(n_orbitals):
                        v_pqrs = interaction_tensor[p, q, r, s]
                        if abs(v_pqrs) < 1.0e-12:
                            continue

                        transitioned = apply_two_body_operator(state, p, q, r, s)
                        if transitioned is None:
                            continue

                        new_state, sign = transitioned
                        row = basis_index[new_state]
                        v_mat[row, col] += 0.25 * v_pqrs * sign

    return 0.5 * (v_mat + v_mat.T)


def compute_mbpt_coefficients(
    h0_diag: np.ndarray,
    v_mat: np.ndarray,
    reference_index: int,
) -> tuple[float, float, float, float]:
    """Return E0, E1, E2 and minimum denominator distance for MBPT(2)."""
    e0 = float(h0_diag[reference_index])
    e1 = float(v_mat[reference_index, reference_index])

    e2 = 0.0
    min_abs_denom = float("inf")

    for n in range(h0_diag.size):
        if n == reference_index:
            continue

        denom = e0 - float(h0_diag[n])
        min_abs_denom = min(min_abs_denom, abs(denom))
        if abs(denom) < 1.0e-12:
            raise ValueError("Near-degenerate denominator encountered; MBPT(2) unstable")

        coupling = float(v_mat[n, reference_index])
        e2 += (coupling * coupling) / denom

    return e0, e1, e2, min_abs_denom


def exact_ground_energy(h0: np.ndarray, v_mat: np.ndarray, lam: float) -> float:
    evals = eigh(h0 + lam * v_mat, eigvals_only=True, check_finite=True)
    return float(evals[0])


def run_mbpt(cfg: MBPTConfig) -> dict[str, object]:
    validate_config(cfg)

    basis = build_fock_basis(cfg)
    g = build_antisymmetrized_interaction(cfg)
    h0 = build_h0_matrix(cfg, basis)
    v_mat = build_interaction_matrix(basis, g)

    h0_diag = np.diag(h0)
    ref_idx = int(np.argmin(h0_diag))
    e0, e1, e2, min_abs_denom = compute_mbpt_coefficients(h0_diag, v_mat, ref_idx)

    lambda_grid = np.linspace(0.0, cfg.lambda_max, cfg.n_lambda, dtype=float)

    rows: list[dict[str, float]] = []
    for lam in lambda_grid:
        e_exact = exact_ground_energy(h0, v_mat, float(lam))
        e_mbpt2 = e0 + lam * e1 + (lam * lam) * e2
        rows.append(
            {
                "lambda": float(lam),
                "E_exact": e_exact,
                "E_MBPT2": float(e_mbpt2),
                "abs_error": float(abs(e_mbpt2 - e_exact)),
            }
        )

    energy_table = pd.DataFrame(rows)

    antisym_pq = float(np.max(np.abs(g + g.transpose(1, 0, 2, 3))))
    antisym_rs = float(np.max(np.abs(g + g.transpose(0, 1, 3, 2))))
    hermitian_tensor = float(np.max(np.abs(g - g.transpose(2, 3, 0, 1))))
    v_hermitian = float(np.max(np.abs(v_mat - v_mat.T)))

    quality_mask = energy_table["lambda"] <= cfg.quality_lambda
    max_quality_error = float(energy_table.loc[quality_mask, "abs_error"].max())
    max_global_error = float(energy_table["abs_error"].max())

    monotonic_exact = bool(np.all(np.diff(energy_table["E_exact"].to_numpy()) > -1.0e-12))

    coefficients = pd.DataFrame(
        {
            "term": ["E0", "E1", "E2"],
            "value": [e0, e1, e2],
        }
    )

    summary = {
        "basis_dimension": float(len(basis)),
        "reference_index": float(ref_idx),
        "reference_h0_energy": e0,
        "E1": e1,
        "E2": e2,
        "min_abs_denominator": min_abs_denom,
        "antisymmetry_error_pq": antisym_pq,
        "antisymmetry_error_rs": antisym_rs,
        "tensor_hermitian_error": hermitian_tensor,
        "interaction_matrix_hermitian_error": v_hermitian,
        "max_error_lambda_le_quality": max_quality_error,
        "max_error_all_lambdas": max_global_error,
        "error_at_lambda_0": float(energy_table.iloc[0]["abs_error"]),
        "monotonic_exact_energy_vs_lambda": float(1.0 if monotonic_exact else 0.0),
    }

    return {
        "cfg": cfg,
        "basis": basis,
        "interaction_tensor": g,
        "h0": h0,
        "v_mat": v_mat,
        "energy_table": energy_table,
        "coefficients": coefficients,
        "summary": summary,
    }


def main() -> None:
    cfg = MBPTConfig()
    result = run_mbpt(cfg)

    coeffs = result["coefficients"]
    energy_table = result["energy_table"]
    summary = result["summary"]

    checks = {
        "Tensor antisymmetry (pq) error < 1e-12": summary["antisymmetry_error_pq"] < 1.0e-12,
        "Tensor antisymmetry (rs) error < 1e-12": summary["antisymmetry_error_rs"] < 1.0e-12,
        "Tensor Hermitian error < 1e-12": summary["tensor_hermitian_error"] < 1.0e-12,
        "Interaction matrix Hermitian error < 1e-12": summary["interaction_matrix_hermitian_error"] < 1.0e-12,
        "Minimum MBPT denominator > 0.1": summary["min_abs_denominator"] > 0.1,
        "Second-order coefficient E2 < 0": summary["E2"] < 0.0,
        f"Max |MBPT2-exact| for lambda <= {cfg.quality_lambda:.2f} < 5e-3": summary[
            "max_error_lambda_le_quality"
        ]
        < 5.0e-3,
        "Error at lambda=0 < 1e-12": summary["error_at_lambda_0"] < 1.0e-12,
        "All reported values finite": bool(
            np.isfinite(coeffs["value"].to_numpy()).all()
            and np.isfinite(energy_table[["E_exact", "E_MBPT2", "abs_error"]].to_numpy()).all()
        ),
    }

    pd.set_option("display.float_format", lambda x: f"{x:.8f}")

    print("=== Many-Body Perturbation Theory MVP (PHYS-0213) ===")
    print(
        f"orbitals={cfg.n_orbitals}, particles={cfg.n_particles}, "
        f"basis_dim={int(summary['basis_dimension'])}, "
        f"lambda in [0, {cfg.lambda_max:.2f}] with {cfg.n_lambda} points"
    )
    print(
        f"interaction: strength={cfg.interaction_strength:.4f}, "
        f"range={cfg.interaction_range:.4f}, quality_lambda={cfg.quality_lambda:.2f}"
    )

    print("\nMBPT coefficients:")
    print(coeffs.to_string(index=False))

    print("\nEnergy comparison (exact vs MBPT2):")
    print(energy_table.to_string(index=False))

    summary_table = pd.DataFrame(
        {
            "quantity": [
                "reference H0 energy",
                "E1 coefficient",
                "E2 coefficient",
                "min |E0-En| denominator",
                "tensor antisymmetry error (pq)",
                "tensor antisymmetry error (rs)",
                "tensor Hermitian error",
                "interaction matrix Hermitian error",
                f"max error (lambda <= {cfg.quality_lambda:.2f})",
                "max error (all lambda)",
                "error at lambda=0",
                "exact E(lambda) monotonicity flag",
            ],
            "value": [
                summary["reference_h0_energy"],
                summary["E1"],
                summary["E2"],
                summary["min_abs_denominator"],
                summary["antisymmetry_error_pq"],
                summary["antisymmetry_error_rs"],
                summary["tensor_hermitian_error"],
                summary["interaction_matrix_hermitian_error"],
                summary["max_error_lambda_le_quality"],
                summary["max_error_all_lambdas"],
                summary["error_at_lambda_0"],
                summary["monotonic_exact_energy_vs_lambda"],
            ],
        }
    )

    print("\nMBPT summary:")
    print(summary_table.to_string(index=False))

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'PASS' if ok else 'FAIL'}")

    all_passed = all(checks.values())
    print(f"\nValidation: {'PASS' if all_passed else 'FAIL'}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
