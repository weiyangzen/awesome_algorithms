"""Quantum teleportation MVP via explicit 3-qubit statevector simulation.

Protocol (Alice owns q0,q1; Bob owns q2):
1) Prepare unknown state |psi> on q0 and Bell pair on q1-q2.
2) Alice applies CNOT(q0->q1), H(q0), then measures (m0, m1).
3) Bob receives two classical bits and applies Z^m0 X^m1 to recover |psi>.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-12
I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
KET0 = np.array([1.0, 0.0], dtype=np.complex128)


@dataclass(frozen=True)
class TeleportationConfig:
    """Configuration for deterministic and random validation."""

    state_specs: tuple[tuple[str, float, float], ...] = (
        ("|0>", 0.0, 0.0),
        ("|1>", np.pi, 0.0),
        ("|+>", np.pi / 2.0, 0.0),
        ("|->", np.pi / 2.0, np.pi),
        ("complex", 1.234, 2.468),
    )
    n_random_states: int = 128
    random_seed: int = 2026


def normalize_state(state: np.ndarray) -> np.ndarray:
    """Return normalized statevector."""
    norm = float(np.linalg.norm(state))
    if norm <= 0.0:
        raise ValueError("State norm must be positive.")
    return state / norm


def bloch_state(theta: float, phi: float) -> np.ndarray:
    """Build |psi>=cos(theta/2)|0>+exp(i*phi)sin(theta/2)|1>."""
    alpha = np.cos(theta / 2.0)
    beta = np.exp(1j * phi) * np.sin(theta / 2.0)
    return normalize_state(np.array([alpha, beta], dtype=np.complex128))


def apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply a 2x2 gate to one qubit in an n-qubit state (q0 is most significant)."""
    if qubit < 0 or qubit >= n_qubits:
        raise ValueError("qubit out of range.")
    ops = [I2] * n_qubits
    ops[qubit] = gate
    full = ops[0]
    for op in ops[1:]:
        full = np.kron(full, op)
    return full @ state


def apply_cnot(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    """Apply CNOT using explicit basis-index permutation (no quantum SDK)."""
    if control == target:
        raise ValueError("control and target must differ.")
    dim = 1 << n_qubits
    out = np.zeros_like(state)
    control_shift = n_qubits - 1 - control
    target_shift = n_qubits - 1 - target
    for basis_index in range(dim):
        amplitude = state[basis_index]
        if abs(amplitude) < EPS:
            continue
        if (basis_index >> control_shift) & 1:
            mapped_index = basis_index ^ (1 << target_shift)
        else:
            mapped_index = basis_index
        out[mapped_index] += amplitude
    return out


def build_pre_measurement_state(psi: np.ndarray) -> np.ndarray:
    """Return the 3-qubit state right before Alice's measurement."""
    state = np.kron(np.kron(psi, KET0), KET0)  # |psi>_q0 |0>_q1 |0>_q2
    state = apply_single_qubit_gate(state, H, qubit=1, n_qubits=3)
    state = apply_cnot(state, control=1, target=2, n_qubits=3)  # Bell pair on q1-q2
    state = apply_cnot(state, control=0, target=1, n_qubits=3)
    state = apply_single_qubit_gate(state, H, qubit=0, n_qubits=3)
    return normalize_state(state)


def conditional_bob_state(pre_measurement_state: np.ndarray, m0: int, m1: int) -> tuple[np.ndarray, float]:
    """Return Bob's conditional state and measurement probability for outcome (m0,m1)."""
    idx0 = (m0 << 2) | (m1 << 1) | 0
    idx1 = (m0 << 2) | (m1 << 1) | 1
    amp0 = pre_measurement_state[idx0]
    amp1 = pre_measurement_state[idx1]
    prob = float(abs(amp0) ** 2 + abs(amp1) ** 2)
    if prob <= EPS:
        raise ValueError("Measurement branch has near-zero probability.")
    bob_state = np.array([amp0, amp1], dtype=np.complex128) / np.sqrt(prob)
    return bob_state, prob


def apply_bob_correction(bob_state: np.ndarray, m0: int, m1: int) -> np.ndarray:
    """Apply Bob's Pauli corrections: Z^m0 X^m1."""
    corrected = bob_state.copy()
    if m1 == 1:
        corrected = X @ corrected
    if m0 == 1:
        corrected = Z @ corrected
    return normalize_state(corrected)


def state_fidelity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Pure-state fidelity |<reference|candidate>|^2."""
    return float(abs(np.vdot(reference, candidate)) ** 2)


def run_protocol_for_state(label: str, psi: np.ndarray) -> list[dict[str, float | int | str]]:
    """Run all 4 measurement branches for one input state."""
    pre = build_pre_measurement_state(psi)
    rows: list[dict[str, float | int | str]] = []
    for m0 in (0, 1):
        for m1 in (0, 1):
            bob_raw, prob = conditional_bob_state(pre, m0, m1)
            bob_fixed = apply_bob_correction(bob_raw, m0, m1)
            rows.append(
                {
                    "state_label": label,
                    "m0": m0,
                    "m1": m1,
                    "probability": prob,
                    "fidelity_before": state_fidelity(psi, bob_raw),
                    "fidelity_after": state_fidelity(psi, bob_fixed),
                }
            )
    return rows


def run_mvp(config: TeleportationConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run deterministic table + random checks for teleportation correctness."""
    all_rows: list[dict[str, float | int | str]] = []
    for label, theta, phi in config.state_specs:
        psi = bloch_state(theta, phi)
        all_rows.extend(run_protocol_for_state(label, psi))
    report = pd.DataFrame(all_rows)

    group_probs = report.groupby("state_label", sort=False)["probability"].sum()
    max_prob_sum_error = float(np.max(np.abs(group_probs.to_numpy(dtype=np.float64) - 1.0)))
    max_prob_uniform_error = float(
        np.max(np.abs(report["probability"].to_numpy(dtype=np.float64) - 0.25))
    )
    min_fidelity_after = float(np.min(report["fidelity_after"].to_numpy(dtype=np.float64)))
    max_fidelity_loss_after = float(
        np.max(1.0 - report["fidelity_after"].to_numpy(dtype=np.float64))
    )

    rng = np.random.default_rng(config.random_seed)
    random_max_loss = 0.0
    random_prob_sum_error = 0.0
    for _ in range(config.n_random_states):
        u = rng.random()
        theta = np.arccos(1.0 - 2.0 * u)  # uniform on Bloch sphere
        phi = 2.0 * np.pi * rng.random()
        psi = bloch_state(theta, phi)
        pre = build_pre_measurement_state(psi)

        prob_sum = 0.0
        for m0 in (0, 1):
            for m1 in (0, 1):
                bob_raw, prob = conditional_bob_state(pre, m0, m1)
                bob_fixed = apply_bob_correction(bob_raw, m0, m1)
                prob_sum += prob
                fidelity_after = state_fidelity(psi, bob_fixed)
                random_max_loss = max(random_max_loss, 1.0 - fidelity_after)
        random_prob_sum_error = max(random_prob_sum_error, abs(prob_sum - 1.0))

    diagnostics = {
        "deterministic_max_prob_sum_error": max_prob_sum_error,
        "deterministic_max_uniform_prob_error": max_prob_uniform_error,
        "deterministic_min_fidelity_after": min_fidelity_after,
        "deterministic_max_fidelity_loss_after": max_fidelity_loss_after,
        "random_states_checked": float(config.n_random_states),
        "random_max_fidelity_loss_after": float(random_max_loss),
        "random_max_prob_sum_error": float(random_prob_sum_error),
    }
    return report, diagnostics


def run_checks(report: pd.DataFrame, diagnostics: dict[str, float]) -> None:
    """Assert protocol invariants."""
    assert not report.empty, "Report should not be empty."
    assert diagnostics["deterministic_max_prob_sum_error"] < 1e-12, "Probability sum check failed."
    assert diagnostics["deterministic_max_uniform_prob_error"] < 1e-12, "Outcome probabilities not uniform."
    assert diagnostics["deterministic_min_fidelity_after"] > 1.0 - 1e-12, "Deterministic fidelity check failed."
    assert (
        diagnostics["deterministic_max_fidelity_loss_after"] < 1e-12
    ), "Deterministic correction does not fully recover |psi|."
    assert (
        diagnostics["random_max_fidelity_loss_after"] < 1e-11
    ), "Random-state teleportation fidelity is too low."
    assert diagnostics["random_max_prob_sum_error"] < 1e-12, "Random branch probability sum check failed."


def main() -> None:
    config = TeleportationConfig()
    report, diagnostics = run_mvp(config)
    run_checks(report, diagnostics)

    print("Quantum Teleportation MVP (explicit statevector, 3 qubits)")
    print()
    print(report.to_string(index=False, float_format=lambda v: f"{v: .12f}"))
    print()
    print("Diagnostics:")
    print(f"  deterministic_max_prob_sum_error     : {diagnostics['deterministic_max_prob_sum_error']:.3e}")
    print(f"  deterministic_max_uniform_prob_error : {diagnostics['deterministic_max_uniform_prob_error']:.3e}")
    print(f"  deterministic_min_fidelity_after     : {diagnostics['deterministic_min_fidelity_after']:.12f}")
    print(f"  deterministic_max_fidelity_loss_after: {diagnostics['deterministic_max_fidelity_loss_after']:.3e}")
    print(f"  random_states_checked                : {int(diagnostics['random_states_checked'])}")
    print(f"  random_max_fidelity_loss_after       : {diagnostics['random_max_fidelity_loss_after']:.3e}")
    print(f"  random_max_prob_sum_error            : {diagnostics['random_max_prob_sum_error']:.3e}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
