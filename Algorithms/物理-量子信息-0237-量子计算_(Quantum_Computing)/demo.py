"""Runnable MVP for PHYS-0236: quantum computing with a tiny state-vector simulator.

The script demonstrates two canonical gate-model tasks:
1) Bell-state preparation (entanglement)
2) 2-qubit Grover search (amplitude amplification)

No interactive input is required.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from scipy.linalg import expm


I2 = np.eye(2, dtype=np.complex128)
X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
H2 = (1.0 / math.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def zero_state(n_qubits: int) -> np.ndarray:
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0 + 0.0j
    return state


def validate_state_vector(state: np.ndarray, atol: float = 1e-10) -> None:
    if state.ndim != 1:
        raise ValueError("State vector must be 1-D")
    dim = int(state.shape[0])
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError("State vector length must be a power of 2")
    norm = float(np.vdot(state, state).real)
    if not np.isclose(norm, 1.0, atol=atol):
        raise ValueError(f"State vector is not normalized: norm={norm:.12f}")


def kron_all(ops: list[np.ndarray]) -> np.ndarray:
    if not ops:
        raise ValueError("ops must be non-empty")
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def single_qubit_operator(gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    if gate.shape != (2, 2):
        raise ValueError("single-qubit gate must have shape (2, 2)")
    if not (0 <= qubit < n_qubits):
        raise ValueError("qubit index out of range")
    ops = [I2 for _ in range(n_qubits)]
    ops[qubit] = gate
    return kron_all(ops)


def apply_unitary(state: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    if unitary.shape != (state.size, state.size):
        raise ValueError("Unitary shape mismatch with state dimension")
    return unitary @ state


def cnot_operator(n_qubits: int, control: int, target: int) -> np.ndarray:
    if not (0 <= control < n_qubits and 0 <= target < n_qubits):
        raise ValueError("control/target index out of range")
    if control == target:
        raise ValueError("control and target must be different")

    dim = 2**n_qubits
    mat = np.zeros((dim, dim), dtype=np.complex128)

    for src in range(dim):
        bits = list(f"{src:0{n_qubits}b}")
        if bits[control] == "1":
            bits[target] = "0" if bits[target] == "1" else "1"
        dst = int("".join(bits), 2)
        mat[dst, src] = 1.0 + 0.0j

    return mat


def rz_gate(theta: float) -> np.ndarray:
    # Rz(theta) = exp(-i * theta/2 * Z), built with scipy.linalg.expm.
    generator = -0.5j * theta * Z2
    return expm(generator)


def state_probabilities(state: np.ndarray) -> np.ndarray:
    probs = np.abs(state) ** 2
    total = float(np.sum(probs))
    if total <= 0:
        raise ValueError("Invalid probability total")
    return probs / total


def sample_counts(probs: np.ndarray, shots: int, seed: int) -> np.ndarray:
    if shots <= 0:
        raise ValueError("shots must be positive")
    rng = np.random.default_rng(seed)
    samples = rng.choice(len(probs), size=shots, p=probs)
    return np.bincount(samples, minlength=len(probs))


def probability_table(probs: np.ndarray, counts: np.ndarray) -> pd.DataFrame:
    n_qubits = int(math.log2(len(probs)))
    labels = [f"|{i:0{n_qubits}b}>" for i in range(len(probs))]
    df = pd.DataFrame(
        {
            "basis": labels,
            "probability": probs,
            "counts": counts,
        }
    )
    return df


def hadamard_layer(n_qubits: int) -> np.ndarray:
    return kron_all([H2 for _ in range(n_qubits)])


def grover_oracle(target_index: int, n_qubits: int) -> np.ndarray:
    dim = 2**n_qubits
    if not (0 <= target_index < dim):
        raise ValueError("target_index out of range")
    oracle = np.eye(dim, dtype=np.complex128)
    oracle[target_index, target_index] = -1.0 + 0.0j
    return oracle


def diffusion_operator(n_qubits: int) -> np.ndarray:
    dim = 2**n_qubits
    s = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)
    return 2.0 * np.outer(s, np.conjugate(s)) - np.eye(dim, dtype=np.complex128)


def bell_state_experiment(shots: int = 2000, seed: int = 123) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    n_qubits = 2
    state = zero_state(n_qubits)

    state = apply_unitary(state, single_qubit_operator(H2, qubit=0, n_qubits=n_qubits))
    state = apply_unitary(state, single_qubit_operator(rz_gate(theta=math.pi / 3.0), qubit=0, n_qubits=n_qubits))
    state = apply_unitary(state, cnot_operator(n_qubits=n_qubits, control=0, target=1))

    validate_state_vector(state)
    probs = state_probabilities(state)
    counts = sample_counts(probs, shots=shots, seed=seed)
    table = probability_table(probs, counts)
    return state, probs, table


def grover_experiment(
    target_bits: str = "10", shots: int = 2000, seed: int = 321
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    n_qubits = 2
    target_index = int(target_bits, 2)

    state = zero_state(n_qubits)
    state = apply_unitary(state, hadamard_layer(n_qubits))

    oracle = grover_oracle(target_index=target_index, n_qubits=n_qubits)
    diffuser = diffusion_operator(n_qubits=n_qubits)

    state = apply_unitary(state, oracle)
    state = apply_unitary(state, diffuser)

    validate_state_vector(state)
    probs = state_probabilities(state)
    counts = sample_counts(probs, shots=shots, seed=seed)
    table = probability_table(probs, counts)
    return state, probs, table, target_index


def torch_norm_check(state: np.ndarray) -> float:
    tensor = torch.from_numpy(state)
    probs = tensor.real * tensor.real + tensor.imag * tensor.imag
    return float(torch.sum(probs).item())


def print_table(title: str, table: pd.DataFrame) -> None:
    print(title)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()


def main() -> None:
    bell_state, bell_probs, bell_table = bell_state_experiment()
    grover_state, grover_probs, grover_table, target_index = grover_experiment()

    print("=== Quantum Computing MVP: Bell + Grover ===")
    print()

    print_table("[Bell state] computational-basis distribution", bell_table)
    bell_correlation_prob = float(bell_probs[0] + bell_probs[3])
    print(f"Bell correlation P(00 or 11): {bell_correlation_prob:.6f}")
    print(f"Bell torch norm check: {torch_norm_check(bell_state):.6f}")
    print()

    print_table("[Grover, target |10>] distribution after 1 iteration", grover_table)
    grover_target_prob = float(grover_probs[target_index])
    print(f"Grover target probability: {grover_target_prob:.6f}")
    print(f"Grover torch norm check: {torch_norm_check(grover_state):.6f}")
    print()

    if bell_correlation_prob < 0.999:
        raise RuntimeError("Bell-state correlation is lower than expected")
    if grover_target_prob < 0.999:
        raise RuntimeError("Grover amplification failed")

    print("MVP checks passed.")


if __name__ == "__main__":
    main()
