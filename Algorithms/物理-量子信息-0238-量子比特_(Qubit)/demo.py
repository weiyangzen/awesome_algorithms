"""Qubit MVP via explicit state-vector and density-matrix simulation.

This script demonstrates:
- preparing a single-qubit pure state from Bloch-sphere angles,
- converting between ket, density matrix, and Bloch vector,
- computing measurement probabilities with Born's rule,
- sampling finite-shot measurements in X/Y/Z bases,
- applying simple unitary gates and re-evaluating observables.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ATOL = 1e-10
EPS = 1e-12

I2 = np.eye(2, dtype=np.complex128)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

AXES = {
    "X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
}


@dataclass(frozen=True)
class QubitParams:
    """Configuration for one non-interactive MVP run."""

    theta: float = np.pi / 3.0
    phi: float = np.pi / 4.0
    shots_per_axis: int = 5000
    seed: int = 2026
    gate_rz_angle: float = np.pi / 3.0


def normalize_state(ket: np.ndarray) -> np.ndarray:
    """Normalize a ket vector."""

    ket = np.asarray(ket, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(ket))
    if norm <= EPS:
        raise ValueError("State norm must be positive.")
    return ket / norm


def ket_from_angles(theta: float, phi: float) -> np.ndarray:
    """Build |psi> = cos(theta/2)|0> + exp(i*phi) sin(theta/2)|1>."""

    ket = np.array(
        [
            np.cos(theta / 2.0),
            np.exp(1j * phi) * np.sin(theta / 2.0),
        ],
        dtype=np.complex128,
    )
    return normalize_state(ket)


def density_from_ket(ket: np.ndarray) -> np.ndarray:
    """Convert pure-state ket to density matrix rho = |psi><psi|."""

    ket_n = normalize_state(ket)
    return np.outer(ket_n, ket_n.conj())


def is_valid_density(rho: np.ndarray, atol: float = ATOL) -> tuple[bool, dict[str, float]]:
    """Check basic physical constraints of a single-qubit density matrix."""

    rho = np.asarray(rho, dtype=np.complex128)
    hermitian_dev = float(np.max(np.abs(rho - rho.conj().T)))
    trace_dev = float(abs(np.trace(rho) - 1.0))
    eigvals = np.linalg.eigvalsh(rho)
    min_eig = float(np.min(np.real(eigvals)))

    valid = hermitian_dev < atol and trace_dev < atol and min_eig >= -atol
    diagnostics = {
        "hermitian_max_dev": hermitian_dev,
        "trace_dev": trace_dev,
        "min_eigenvalue": min_eig,
    }
    return valid, diagnostics


def bloch_vector_from_density(rho: np.ndarray) -> np.ndarray:
    """Return Bloch vector r where rho = (I + r·sigma)/2."""

    rx = float(np.real(np.trace(rho @ SIGMA_X)))
    ry = float(np.real(np.trace(rho @ SIGMA_Y)))
    rz = float(np.real(np.trace(rho @ SIGMA_Z)))
    return np.array([rx, ry, rz], dtype=np.float64)


def projector(axis: np.ndarray, outcome: int) -> np.ndarray:
    """Projector for spin outcome +1/-1 along a normalized axis."""

    if outcome not in (-1, 1):
        raise ValueError("Outcome must be +1 or -1.")
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n <= EPS:
        raise ValueError("Axis norm must be positive.")
    axis = axis / n
    sigma_n = axis[0] * SIGMA_X + axis[1] * SIGMA_Y + axis[2] * SIGMA_Z
    return 0.5 * (I2 + outcome * sigma_n)


def probabilities_for_axis(rho: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    """Return probabilities P(+1), P(-1) for projective measurement on axis."""

    p_plus = float(np.real(np.trace(rho @ projector(axis, +1))))
    p_minus = float(np.real(np.trace(rho @ projector(axis, -1))))
    probs = np.clip(np.array([p_plus, p_minus], dtype=np.float64), 0.0, 1.0)
    total = float(np.sum(probs))
    if total <= EPS:
        raise RuntimeError("Invalid probability normalization.")
    probs = probs / total
    return float(probs[0]), float(probs[1])


def sampled_expectation(
    p_plus: float,
    p_minus: float,
    shots: int,
    rng: np.random.Generator,
) -> tuple[float, int, int]:
    """Sample ±1 outcomes and estimate expectation E = <m>."""

    if shots <= 0:
        raise ValueError("shots must be > 0")
    outcomes = rng.choice([1, -1], size=shots, p=[p_plus, p_minus])
    plus_count = int(np.sum(outcomes == 1))
    minus_count = int(np.sum(outcomes == -1))
    exp_val = float(np.mean(outcomes))
    return exp_val, plus_count, minus_count


def hadamard() -> np.ndarray:
    """Single-qubit Hadamard gate."""

    return (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    """Single-qubit phase rotation around Z axis."""

    return np.array(
        [
            [np.exp(-1j * theta / 2.0), 0.0],
            [0.0, np.exp(1j * theta / 2.0)],
        ],
        dtype=np.complex128,
    )


def apply_unitary(rho: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    """Unitary density-matrix evolution: rho' = U rho U^dagger."""

    if unitary.shape != (2, 2):
        raise ValueError("Only 2x2 unitary is supported in this MVP.")
    rho_next = unitary @ rho @ unitary.conj().T
    return np.asarray(rho_next, dtype=np.complex128)


def make_axis_table(
    rho: np.ndarray,
    shots: int,
    rng: np.random.Generator,
    state_name: str,
) -> pd.DataFrame:
    """Collect theory + sampling results for X/Y/Z measurements."""

    rows: list[dict[str, float | str | int]] = []
    bloch = bloch_vector_from_density(rho)

    for axis_name, axis_vec in AXES.items():
        p_plus, p_minus = probabilities_for_axis(rho, axis_vec)
        sampled_e, n_plus, n_minus = sampled_expectation(p_plus, p_minus, shots, rng)
        theory_e = float(p_plus - p_minus)
        born_from_bloch = float(np.dot(bloch, axis_vec))

        rows.append(
            {
                "state": state_name,
                "axis": axis_name,
                "P(+1)_theory": p_plus,
                "P(-1)_theory": p_minus,
                "E_theory": theory_e,
                "E_sampled": sampled_e,
                "N_plus": n_plus,
                "N_minus": n_minus,
                "E_from_bloch": born_from_bloch,
            }
        )

    return pd.DataFrame(rows)


def summarize_state(rho: np.ndarray, name: str) -> pd.DataFrame:
    """Small summary for density matrix validity and Bloch norm."""

    valid, diagnostics = is_valid_density(rho)
    bloch = bloch_vector_from_density(rho)
    purity = float(np.real(np.trace(rho @ rho)))
    return pd.DataFrame(
        [
            {
                "state": name,
                "valid_density": bool(valid),
                "bloch_rx": bloch[0],
                "bloch_ry": bloch[1],
                "bloch_rz": bloch[2],
                "bloch_norm": float(np.linalg.norm(bloch)),
                "purity": purity,
                "hermitian_max_dev": diagnostics["hermitian_max_dev"],
                "trace_dev": diagnostics["trace_dev"],
                "min_eigenvalue": diagnostics["min_eigenvalue"],
            }
        ]
    )


def main() -> None:
    params = QubitParams()
    rng = np.random.default_rng(params.seed)

    ket0 = ket_from_angles(params.theta, params.phi)
    rho0 = density_from_ket(ket0)

    # Example gate sequence: H followed by Rz(theta_g).
    unitary = rz(params.gate_rz_angle) @ hadamard()
    rho1 = apply_unitary(rho0, unitary)

    summary = pd.concat(
        [
            summarize_state(rho0, "initial"),
            summarize_state(rho1, "after_H_then_Rz"),
        ],
        ignore_index=True,
    )

    axis_table = pd.concat(
        [
            make_axis_table(rho0, params.shots_per_axis, rng, "initial"),
            make_axis_table(rho1, params.shots_per_axis, rng, "after_H_then_Rz"),
        ],
        ignore_index=True,
    )

    print("=== Qubit MVP (single qubit) ===")
    print(
        f"theta={params.theta:.6f}, phi={params.phi:.6f}, "
        f"shots_per_axis={params.shots_per_axis}, seed={params.seed}"
    )
    print()
    print("[1] State diagnostics")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print("[2] Measurement table (X/Y/Z)")
    print(axis_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Core sanity checks.
    assert bool(summary["valid_density"].all()), "Density validity check failed."
    assert float(np.max(np.abs(axis_table["E_theory"] - axis_table["E_from_bloch"]))) < 1e-10
    assert float(np.max(np.abs(axis_table["P(+1)_theory"] + axis_table["P(-1)_theory"] - 1.0))) < 1e-10

    # Sampling checks: bounded fluctuations, should be reasonably close at 5000 shots.
    sampling_err = np.abs(axis_table["E_sampled"] - axis_table["E_theory"])
    assert float(np.max(sampling_err)) < 0.08, "Sampling deviation too large."

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
