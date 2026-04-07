"""Quantum Error Correction MVP: 3-qubit bit-flip code.

This script implements an end-to-end QEC pipeline without quantum SDKs:
1) Encode 1 logical qubit into 3 physical qubits (repetition code).
2) Apply independent X noise on each physical qubit.
3) Perform stabilizer-syndrome recovery using projectors of Z0Z1 and Z1Z2.
4) Decode back to one qubit and evaluate logical failure.

It also verifies the analytic logical error law for single-qubit bit-flip noise:
    p_L = 3 p^2 - 2 p^3
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


@dataclass(frozen=True)
class QECConfig:
    """Configuration for the deterministic MVP experiment."""

    p_grid: tuple[float, ...] = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
    state_theta: float = 1.10
    state_phi: float = 0.70


def normalize_state(psi: np.ndarray) -> np.ndarray:
    """Normalize a statevector."""
    norm = float(np.linalg.norm(psi))
    if norm <= 0.0:
        raise ValueError("State norm must be positive.")
    return psi / norm


def bloch_state(theta: float, phi: float) -> np.ndarray:
    """Build |psi>=cos(theta/2)|0>+e^{i phi}sin(theta/2)|1>."""
    alpha = np.cos(theta / 2.0)
    beta = np.exp(1j * phi) * np.sin(theta / 2.0)
    return normalize_state(np.array([alpha, beta], dtype=np.complex128))


def pure_density(psi: np.ndarray) -> np.ndarray:
    """Return |psi><psi|."""
    return np.outer(psi, psi.conj())


def kron_all(ops: list[np.ndarray]) -> np.ndarray:
    """Kronecker product of an operator list."""
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def x_on(qubit: int, n_qubits: int = 3) -> np.ndarray:
    """Pauli-X on target qubit (q0 is most significant)."""
    ops = [I2] * n_qubits
    ops[qubit] = X
    return kron_all(ops)


def z_on(qubit: int, n_qubits: int = 3) -> np.ndarray:
    """Pauli-Z on target qubit (q0 is most significant)."""
    ops = [I2] * n_qubits
    ops[qubit] = Z
    return kron_all(ops)


def build_encoder_decoder() -> tuple[np.ndarray, np.ndarray]:
    """Build encode isometry E (8x2) and decoder D=E^dagger (2x8).

    E maps:
      |0> -> |000>
      |1> -> |111>
    """
    ket000 = np.zeros(8, dtype=np.complex128)
    ket111 = np.zeros(8, dtype=np.complex128)
    ket000[0] = 1.0
    ket111[7] = 1.0

    e = np.stack([ket000, ket111], axis=1)  # shape (8, 2)
    d = e.conj().T
    return e, d


def encode_density(rho_logical: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Encode one-qubit density matrix into 3-qubit code space."""
    return e @ rho_logical @ e.conj().T


def bitflip_channel_three_qubits(rho: np.ndarray, p: float) -> np.ndarray:
    """Independent physical X noise on each qubit.

    rho -> sum_e Prob(e) X^e rho X^e, where e in {0,1}^3.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1].")

    out = np.zeros_like(rho)
    for e0, e1, e2 in product((0, 1), repeat=3):
        wt = e0 + e1 + e2
        prob = (p**wt) * ((1.0 - p) ** (3 - wt))

        op = kron_all([
            X if e0 else I2,
            X if e1 else I2,
            X if e2 else I2,
        ])
        out += prob * (op @ rho @ op.conj().T)
    return out


def bitflip_channel_one_qubit(rho: np.ndarray, p: float) -> np.ndarray:
    """Uncoded one-qubit bit-flip channel for baseline comparison."""
    return (1.0 - p) * rho + p * (X @ rho @ X)


def build_stabilizer_projectors() -> dict[tuple[int, int], np.ndarray]:
    """Projectors for syndromes of S1=Z0Z1 and S2=Z1Z2."""
    i8 = np.eye(8, dtype=np.complex128)
    s1 = z_on(0) @ z_on(1)
    s2 = z_on(1) @ z_on(2)

    projectors: dict[tuple[int, int], np.ndarray] = {}
    for a in (+1, -1):
        for b in (+1, -1):
            p_ab = 0.25 * (i8 + a * s1) @ (i8 + b * s2)
            projectors[(a, b)] = p_ab
    return projectors


def build_recovery_corrections() -> dict[tuple[int, int], np.ndarray]:
    """Syndrome -> correction map for 3-qubit bit-flip code."""
    i8 = np.eye(8, dtype=np.complex128)
    return {
        (+1, +1): i8,      # no error (or triple error class)
        (-1, +1): x_on(0), # likely error on q0
        (-1, -1): x_on(1), # likely error on q1
        (+1, -1): x_on(2), # likely error on q2
    }


def recover_density(
    rho: np.ndarray,
    projectors: dict[tuple[int, int], np.ndarray],
    corrections: dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Apply syndrome measurement + conditional correction."""
    out = np.zeros_like(rho)
    syndrome_probs: dict[tuple[int, int], float] = {}

    for syndrome, projector in projectors.items():
        branch = projector @ rho @ projector
        branch_prob = float(np.real(np.trace(branch)))
        syndrome_probs[syndrome] = max(branch_prob, 0.0)

        corr = corrections[syndrome]
        out += corr @ branch @ corr.conj().T

    return out, syndrome_probs


def decode_density(rho_encoded: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Decode 3-qubit code-state density matrix back to one qubit."""
    rho = d @ rho_encoded @ d.conj().T
    rho = 0.5 * (rho + rho.conj().T)  # enforce Hermitian numerically
    tr = float(np.real(np.trace(rho)))
    if tr <= 0.0:
        raise ValueError("Decoded state has non-positive trace.")
    return rho / tr


def pure_state_fidelity(psi: np.ndarray, rho: np.ndarray) -> float:
    """Fidelity F(|psi>, rho) = <psi|rho|psi>."""
    return float(np.real(np.vdot(psi, rho @ psi)))


def logical_failure_probability_from_zero(
    p: float,
    e: np.ndarray,
    d: np.ndarray,
    projectors: dict[tuple[int, int], np.ndarray],
    corrections: dict[tuple[int, int], np.ndarray],
) -> float:
    """Logical bit-flip failure probability for input |0>."""
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    rho0 = pure_density(psi0)

    rho_enc = encode_density(rho0, e)
    rho_noisy = bitflip_channel_three_qubits(rho_enc, p)
    rho_rec, _ = recover_density(rho_noisy, projectors, corrections)
    rho_out = decode_density(rho_rec, d)

    # For |0> input, population on |1> is logical failure probability.
    return float(np.real(rho_out[1, 1]))


def validate_projectors(projectors: dict[tuple[int, int], np.ndarray]) -> None:
    """Check orthogonality and completeness of syndrome projectors."""
    keys = list(projectors.keys())
    i8 = np.eye(8, dtype=np.complex128)

    p_sum = np.zeros((8, 8), dtype=np.complex128)
    for key in keys:
        p_sum += projectors[key]
    assert np.allclose(p_sum, i8, atol=1e-12), "Projectors do not sum to identity."

    for a in keys:
        pa = projectors[a]
        assert np.allclose(pa @ pa, pa, atol=1e-12), f"Projector {a} not idempotent."
        for b in keys:
            if a == b:
                continue
            pb = projectors[b]
            assert np.allclose(pa @ pb, 0.0, atol=1e-12), f"Projectors {a} and {b} not orthogonal."


def run_mvp(config: QECConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run deterministic QEC experiment and return tables + diagnostics."""
    e, d = build_encoder_decoder()
    projectors = build_stabilizer_projectors()
    corrections = build_recovery_corrections()

    validate_projectors(projectors)

    # Demonstration on a generic superposition state.
    psi = bloch_state(config.state_theta, config.state_phi)
    rho_logical = pure_density(psi)

    p_demo = 0.10
    rho_enc = encode_density(rho_logical, e)
    rho_noisy = bitflip_channel_three_qubits(rho_enc, p_demo)
    rho_rec, syndrome_probs = recover_density(rho_noisy, projectors, corrections)
    rho_out = decode_density(rho_rec, d)

    rho_uncoded = bitflip_channel_one_qubit(rho_logical, p_demo)
    f_uncoded = pure_state_fidelity(psi, rho_uncoded)
    f_coded = pure_state_fidelity(psi, rho_out)

    syndrome_rows = []
    for syndrome in [(+1, +1), (-1, +1), (-1, -1), (+1, -1)]:
        syndrome_rows.append(
            {
                "syndrome_(s1,s2)": str(syndrome),
                "probability": syndrome_probs[syndrome],
            }
        )
    syndrome_df = pd.DataFrame(syndrome_rows)

    # Logical error curve: compare exact simulation with theory p_L.
    rows = []
    max_abs_theory_gap = 0.0
    for p in config.p_grid:
        p_coded = logical_failure_probability_from_zero(p, e, d, projectors, corrections)
        p_theory = 3.0 * (p**2) - 2.0 * (p**3)
        gap = abs(p_coded - p_theory)
        max_abs_theory_gap = max(max_abs_theory_gap, gap)

        rows.append(
            {
                "p": p,
                "uncoded_fail": p,
                "coded_fail_sim": p_coded,
                "coded_fail_theory": p_theory,
                "improvement(uncoded-coding)": p - p_coded,
                "abs_theory_gap": gap,
            }
        )

    curve_df = pd.DataFrame(rows)

    diagnostics = {
        "syndrome_prob_sum_error": float(abs(syndrome_df["probability"].sum() - 1.0)),
        "fidelity_uncoded_at_p0.10": float(f_uncoded),
        "fidelity_coded_at_p0.10": float(f_coded),
        "max_abs_theory_gap": float(max_abs_theory_gap),
    }

    # Hard checks for MVP correctness.
    assert diagnostics["syndrome_prob_sum_error"] < 1e-12, "Syndrome probabilities must sum to 1."
    assert diagnostics["max_abs_theory_gap"] < 1e-12, "Simulated logical error must match theory."

    # For p < 0.5, 3-qubit bit-flip code should improve logical bit-flip probability.
    for _, row in curve_df.iterrows():
        if row["p"] < 0.5:
            assert row["coded_fail_sim"] <= row["uncoded_fail"] + 1e-12

    return syndrome_df, curve_df, diagnostics


def main() -> None:
    config = QECConfig()
    syndrome_df, curve_df, diagnostics = run_mvp(config)

    print("Quantum Error Correction MVP (3-qubit bit-flip code)")
    print("=" * 72)
    print("Pipeline: encode -> independent X noise -> syndrome recovery -> decode")
    print()
    print("Syndrome distribution at p=0.10 for one generic input state:")
    print(syndrome_df.to_string(index=False, float_format=lambda x: f"{x:.12f}"))
    print()
    print("Logical failure comparison (input |0>):")
    print(curve_df.to_string(index=False, float_format=lambda x: f"{x:.12f}"))
    print()
    print("Diagnostics:")
    print(f"  syndrome_prob_sum_error : {diagnostics['syndrome_prob_sum_error']:.3e}")
    print(f"  fidelity_uncoded@p=0.10 : {diagnostics['fidelity_uncoded_at_p0.10']:.12f}")
    print(f"  fidelity_coded@p=0.10   : {diagnostics['fidelity_coded_at_p0.10']:.12f}")
    print(f"  max_abs_theory_gap      : {diagnostics['max_abs_theory_gap']:.3e}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
