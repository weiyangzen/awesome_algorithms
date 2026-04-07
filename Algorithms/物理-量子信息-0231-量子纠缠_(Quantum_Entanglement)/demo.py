"""Quantum entanglement MVP.

This script implements a minimal, source-traceable workflow for two-qubit
entanglement analysis:
- construct canonical quantum states (Bell, product, Werner),
- quantify entanglement with entropy and concurrence,
- test Bell nonlocality with CHSH value,
- estimate CHSH from finite-shot sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


I2 = np.eye(2, dtype=np.complex128)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


@dataclass(frozen=True)
class EntanglementParams:
    """Config for the demonstration run."""

    shots_per_setting: int = 6000
    seed: int = 2026


def normalize_axis(v: np.ndarray) -> np.ndarray:
    """Normalize a 3D measurement axis."""

    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise ValueError("Measurement axis norm must be positive.")
    return np.asarray(v, dtype=np.float64) / n


def density_from_ket(ket: np.ndarray) -> np.ndarray:
    """Convert state vector |psi> to density matrix |psi><psi|."""

    ket = np.asarray(ket, dtype=np.complex128).reshape(-1)
    return np.outer(ket, ket.conj())


def bell_phi_plus_density() -> np.ndarray:
    """Return density matrix of |Phi+> = (|00> + |11>) / sqrt(2)."""

    ket = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    return density_from_ket(ket)


def product_00_density() -> np.ndarray:
    """Return density matrix of separable |00>."""

    ket = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    return density_from_ket(ket)


def werner_state_density(p: float) -> np.ndarray:
    """Werner state: rho = p|Phi+><Phi+| + (1-p)I/4, 0<=p<=1."""

    if not (0.0 <= p <= 1.0):
        raise ValueError("Werner parameter p must be in [0, 1].")
    bell = bell_phi_plus_density()
    return p * bell + (1.0 - p) * np.eye(4, dtype=np.complex128) / 4.0


def partial_trace_two_qubit(rho: np.ndarray, keep: int) -> np.ndarray:
    """Partial trace for a 2-qubit density matrix.

    Args:
        rho: shape (4, 4)
        keep: 0 keeps qubit A (trace B), 1 keeps qubit B (trace A)
    """

    rho = np.asarray(rho, dtype=np.complex128).reshape(2, 2, 2, 2)
    if keep == 0:
        return np.trace(rho, axis1=1, axis2=3)
    if keep == 1:
        return np.trace(rho, axis1=0, axis2=2)
    raise ValueError("keep must be 0 or 1.")


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Compute entropy S(rho) = -Tr(rho log2 rho)."""

    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(np.real(eigvals), 0.0, 1.0)
    positive = eigvals[eigvals > 1e-12]
    if positive.size == 0:
        return 0.0
    entropy = float(-np.sum(positive * np.log2(positive)))
    return max(0.0, entropy)


def concurrence(rho: np.ndarray) -> float:
    """Wootters concurrence for a 2-qubit mixed state."""

    yy = np.kron(SIGMA_Y, SIGMA_Y)
    r_matrix = rho @ yy @ rho.conj() @ yy
    eigvals = np.linalg.eigvals(r_matrix)
    eigvals_real = np.clip(np.real(eigvals), 0.0, None)
    lambdas = np.sort(np.sqrt(eigvals_real))[::-1]
    c_val = float(max(0.0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]))
    return c_val


def spin_operator(axis: np.ndarray) -> np.ndarray:
    """Pauli spin operator sigma(axis) = ax*X + ay*Y + az*Z."""

    a = normalize_axis(axis)
    return a[0] * SIGMA_X + a[1] * SIGMA_Y + a[2] * SIGMA_Z


def correlator(rho: np.ndarray, axis_a: np.ndarray, axis_b: np.ndarray) -> float:
    """Expectation E(a,b) = <sigma_a \\otimes sigma_b>."""

    op = np.kron(spin_operator(axis_a), spin_operator(axis_b))
    return float(np.real(np.trace(rho @ op)))


def chsh_value(
    rho: np.ndarray,
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> float:
    """CHSH S = |E(a0,b0)+E(a0,b1)+E(a1,b0)-E(a1,b1)|."""

    e00 = correlator(rho, a0, b0)
    e01 = correlator(rho, a0, b1)
    e10 = correlator(rho, a1, b0)
    e11 = correlator(rho, a1, b1)
    return float(abs(e00 + e01 + e10 - e11))


def local_projector(axis: np.ndarray, outcome: int) -> np.ndarray:
    """Projector for spin outcome ±1 on given axis."""

    if outcome not in (-1, 1):
        raise ValueError("outcome must be -1 or +1.")
    return 0.5 * (I2 + outcome * spin_operator(axis))


def joint_outcome_probabilities(rho: np.ndarray, axis_a: np.ndarray, axis_b: np.ndarray) -> np.ndarray:
    """Probabilities of outcomes (++,+-,-+,--) for settings (a,b)."""

    outcomes = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    probs = []
    for oa, ob in outcomes:
        pa = local_projector(axis_a, oa)
        pb = local_projector(axis_b, ob)
        p = float(np.real(np.trace(rho @ np.kron(pa, pb))))
        probs.append(max(0.0, p))

    probs_arr = np.asarray(probs, dtype=np.float64)
    total = float(np.sum(probs_arr))
    if total <= 0.0:
        raise RuntimeError("Invalid probability normalization.")
    probs_arr /= total
    return probs_arr


def sampled_correlator(
    rho: np.ndarray,
    axis_a: np.ndarray,
    axis_b: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> float:
    """Monte Carlo estimate of correlator from finite projective measurements."""

    if shots <= 0:
        raise ValueError("shots must be > 0")

    probs = joint_outcome_probabilities(rho, axis_a, axis_b)
    outcomes = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.int8)
    draw = rng.choice(4, size=shots, p=probs)
    picked = outcomes[draw]
    products = picked[:, 0] * picked[:, 1]
    return float(np.mean(products))


def sampled_chsh_value(
    rho: np.ndarray,
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    shots_per_setting: int,
    rng: np.random.Generator,
) -> float:
    """Finite-shot CHSH estimate."""

    e00 = sampled_correlator(rho, a0, b0, shots_per_setting, rng)
    e01 = sampled_correlator(rho, a0, b1, shots_per_setting, rng)
    e10 = sampled_correlator(rho, a1, b0, shots_per_setting, rng)
    e11 = sampled_correlator(rho, a1, b1, shots_per_setting, rng)
    return float(abs(e00 + e01 + e10 - e11))


def evaluate_state(
    state_name: str,
    rho: np.ndarray,
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    params: EntanglementParams,
    rng: np.random.Generator,
) -> dict[str, str | float | bool]:
    """Compute entanglement and CHSH diagnostics for one state."""

    reduced_a = partial_trace_two_qubit(rho, keep=0)
    entropy_a = von_neumann_entropy(reduced_a)
    conc = concurrence(rho)
    s_theory = chsh_value(rho, a0, a1, b0, b1)
    s_sampled = sampled_chsh_value(rho, a0, a1, b0, b1, params.shots_per_setting, rng)

    return {
        "state": state_name,
        "entanglement_entropy_A": entropy_a,
        "concurrence": conc,
        "chsh_theory": s_theory,
        "chsh_sampled": s_sampled,
        "entangled_by_concurrence": bool(conc > 1e-6),
        "violates_chsh": bool(s_theory > 2.0 + 1e-9),
    }


def main() -> None:
    params = EntanglementParams(shots_per_setting=6000, seed=2026)
    rng = np.random.default_rng(params.seed)

    # Near-optimal settings for |Phi+> that achieve Tsirelson bound 2*sqrt(2).
    a0 = np.array([0.0, 0.0, 1.0])
    a1 = np.array([1.0, 0.0, 0.0])
    b0 = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    b1 = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)

    states = {
        "Bell_PhiPlus": bell_phi_plus_density(),
        "Product_00": product_00_density(),
        "Werner_p0.60": werner_state_density(0.60),
        "Werner_p0.80": werner_state_density(0.80),
    }

    rows = [
        evaluate_state(name, rho, a0, a1, b0, b1, params, rng)
        for name, rho in states.items()
    ]

    summary = pd.DataFrame(rows)
    print("=== Quantum Entanglement MVP (2-qubit) ===")
    print(f"shots_per_setting={params.shots_per_setting}, seed={params.seed}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Sweep Werner parameter to show entanglement/nonlocality thresholds.
    p_grid = np.linspace(0.0, 1.0, 6)
    sweep_rows = []
    for p in p_grid:
        rho_w = werner_state_density(float(p))
        sweep_rows.append(
            {
                "p": float(p),
                "concurrence": concurrence(rho_w),
                "chsh_theory": chsh_value(rho_w, a0, a1, b0, b1),
            }
        )
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df["entangled"] = sweep_df["concurrence"] > 1e-6
    sweep_df["chsh_violation"] = sweep_df["chsh_theory"] > 2.0 + 1e-9

    print("\n=== Werner Sweep ===")
    print(sweep_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Sanity checks on physically expected behavior.
    bell_row = summary.loc[summary["state"] == "Bell_PhiPlus"].iloc[0]
    prod_row = summary.loc[summary["state"] == "Product_00"].iloc[0]
    w06_row = summary.loc[summary["state"] == "Werner_p0.60"].iloc[0]
    w08_row = summary.loc[summary["state"] == "Werner_p0.80"].iloc[0]

    if bell_row["entanglement_entropy_A"] < 0.999:
        raise AssertionError("Bell state should have near-maximal subsystem entropy.")
    if bell_row["concurrence"] < 0.999:
        raise AssertionError("Bell state concurrence should be near 1.")
    if bell_row["chsh_theory"] <= 2.7:
        raise AssertionError("Bell state should strongly violate CHSH.")

    if prod_row["entanglement_entropy_A"] > 1e-8:
        raise AssertionError("Product state entropy should be near 0.")
    if prod_row["concurrence"] > 1e-8:
        raise AssertionError("Product state concurrence should be near 0.")
    if prod_row["chsh_theory"] > 2.0 + 1e-8:
        raise AssertionError("Product state must satisfy CHSH <= 2.")

    if w06_row["concurrence"] <= 0.0:
        raise AssertionError("Werner p=0.60 should still be entangled (concurrence > 0).")
    if w06_row["chsh_theory"] > 2.0 + 1e-8:
        raise AssertionError("Werner p=0.60 should not violate CHSH for these settings.")

    if w08_row["chsh_theory"] <= 2.0:
        raise AssertionError("Werner p=0.80 should violate CHSH.")


if __name__ == "__main__":
    main()
