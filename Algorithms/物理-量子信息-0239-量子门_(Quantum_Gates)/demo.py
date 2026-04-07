"""Quantum gates MVP via explicit state-vector simulation.

This script demonstrates:
- single-qubit gates (X, Y, Z, H, S, T, Rx, Rz),
- two-qubit CNOT construction,
- Bell-state preparation from gate sequence,
- basic gate identities and unitarity checks.

Run: uv run python demo.py
"""

from __future__ import annotations

import numpy as np


EPS = 1e-12
ATOL = 1e-10

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
S = np.array([[1.0, 0.0], [0.0, 1j]], dtype=np.complex128)
T = np.array([[1.0, 0.0], [0.0, np.exp(1j * np.pi / 4.0)]], dtype=np.complex128)


def rx(theta: float) -> np.ndarray:
    """Return single-qubit Rx rotation matrix."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return c * I2 - 1j * s * X


def rz(theta: float) -> np.ndarray:
    """Return single-qubit Rz rotation matrix."""
    return np.array(
        [
            [np.exp(-1j * theta / 2.0), 0.0],
            [0.0, np.exp(1j * theta / 2.0)],
        ],
        dtype=np.complex128,
    )


def normalize_state(state: np.ndarray) -> np.ndarray:
    """Return normalized statevector."""
    norm = float(np.linalg.norm(state))
    if norm <= EPS:
        raise ValueError("State norm must be positive.")
    return state / norm


def basis_state(bits: str) -> np.ndarray:
    """Return computational basis state |bits> for bitstring like '00' or '1'."""
    if len(bits) == 0:
        raise ValueError("bits must be non-empty.")
    if any(ch not in "01" for ch in bits):
        raise ValueError("bits must be a binary string.")

    n_qubits = len(bits)
    dim = 1 << n_qubits
    index = int(bits, 2)
    state = np.zeros(dim, dtype=np.complex128)
    state[index] = 1.0
    return state


def kron_all(operators: list[np.ndarray]) -> np.ndarray:
    """Kronecker product of operators in left-to-right qubit order."""
    if not operators:
        raise ValueError("operators must be non-empty.")
    out = operators[0]
    for op in operators[1:]:
        out = np.kron(out, op)
    return out


def single_qubit_gate_matrix(gate: np.ndarray, n_qubits: int, target: int) -> np.ndarray:
    """Expand a 2x2 gate to full n-qubit unitary (q0 is most significant bit)."""
    if gate.shape != (2, 2):
        raise ValueError("gate must be 2x2.")
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1.")
    if target < 0 or target >= n_qubits:
        raise ValueError("target qubit out of range.")

    operators = [I2 for _ in range(n_qubits)]
    operators[target] = gate
    return kron_all(operators)


def cnot_matrix(n_qubits: int, control: int, target: int) -> np.ndarray:
    """Construct CNOT matrix by explicit basis-index mapping."""
    if n_qubits < 2:
        raise ValueError("CNOT requires n_qubits >= 2.")
    if control == target:
        raise ValueError("control and target must differ.")
    if not (0 <= control < n_qubits and 0 <= target < n_qubits):
        raise ValueError("control/target out of range.")

    dim = 1 << n_qubits
    unitary = np.zeros((dim, dim), dtype=np.complex128)

    c_shift = n_qubits - 1 - control
    t_shift = n_qubits - 1 - target

    for col in range(dim):
        if ((col >> c_shift) & 1) == 1:
            row = col ^ (1 << t_shift)
        else:
            row = col
        unitary[row, col] = 1.0

    return unitary


def apply_unitary(state: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    """Apply a unitary to a statevector."""
    if unitary.shape[0] != unitary.shape[1]:
        raise ValueError("unitary must be square.")
    if unitary.shape[1] != state.shape[0]:
        raise ValueError("Dimension mismatch between unitary and state.")
    return normalize_state(unitary @ state)


def is_unitary(unitary: np.ndarray, atol: float = ATOL) -> tuple[bool, float]:
    """Check unitarity and return (is_unitary, max_abs_deviation)."""
    if unitary.shape[0] != unitary.shape[1]:
        return False, float("inf")
    ident = np.eye(unitary.shape[0], dtype=np.complex128)
    err = unitary.conj().T @ unitary - ident
    max_abs_dev = float(np.max(np.abs(err)))
    return max_abs_dev < atol, max_abs_dev


def probabilities(state: np.ndarray) -> np.ndarray:
    """Return computational-basis probabilities."""
    probs = np.abs(state) ** 2
    probs = np.clip(probs.real, 0.0, 1.0)
    total = float(np.sum(probs))
    if total <= EPS:
        raise ValueError("Probability sum is zero.")
    return probs / total


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """Pure-state fidelity |<psi|phi>|^2."""
    psi_n = normalize_state(psi)
    phi_n = normalize_state(phi)
    return float(abs(np.vdot(psi_n, phi_n)) ** 2)


def state_to_dirac(state: np.ndarray, tol: float = 1e-9) -> str:
    """Render nonzero amplitudes as a compact Dirac-style string."""
    dim = state.shape[0]
    n_qubits = int(np.log2(dim))
    terms: list[str] = []
    for index, amp in enumerate(state):
        if abs(amp) < tol:
            continue
        bits = format(index, f"0{n_qubits}b")
        terms.append(f"({amp.real:+.4f}{amp.imag:+.4f}j)|{bits}>")
    return " + ".join(terms) if terms else "0"


def run_single_qubit_examples() -> dict[str, np.ndarray | float]:
    """Run minimal single-qubit gate examples."""
    ket0 = basis_state("0")
    ket1 = basis_state("1")
    ket_plus = normalize_state(ket0 + ket1)

    after_h = apply_unitary(ket0, H)
    after_x = apply_unitary(ket0, X)
    after_z_plus = apply_unitary(ket_plus, Z)

    theta = np.pi / 3.0
    after_rx = apply_unitary(ket0, rx(theta))

    return {
        "ket0": ket0,
        "ket1": ket1,
        "ket_plus": ket_plus,
        "after_h": after_h,
        "after_x": after_x,
        "after_z_plus": after_z_plus,
        "after_rx": after_rx,
        "rx_theta": theta,
    }


def run_bell_state_example() -> dict[str, np.ndarray | float]:
    """Prepare Bell state: |00> --H(q0)--> --CNOT(q0,q1)--> (|00>+|11>)/sqrt(2)."""
    ket00 = basis_state("00")
    u_h_q0 = single_qubit_gate_matrix(H, n_qubits=2, target=0)
    u_cnot = cnot_matrix(n_qubits=2, control=0, target=1)

    after_h = apply_unitary(ket00, u_h_q0)
    bell = apply_unitary(after_h, u_cnot)

    bell_ideal = normalize_state(basis_state("00") + basis_state("11"))
    probs = probabilities(bell)

    return {
        "after_h": after_h,
        "bell": bell,
        "bell_ideal": bell_ideal,
        "bell_fidelity": fidelity(bell, bell_ideal),
        "bell_probs": probs,
    }


def run_identity_checks() -> dict[str, float]:
    """Return matrix-norm residuals for standard gate identities."""
    cnot_2q = cnot_matrix(2, 0, 1)
    i4 = np.eye(4, dtype=np.complex128)

    checks = {
        "||H^2 - I||_max": float(np.max(np.abs(H @ H - I2))),
        "||X^2 - I||_max": float(np.max(np.abs(X @ X - I2))),
        "||Z^2 - I||_max": float(np.max(np.abs(Z @ Z - I2))),
        "||HZH - X||_max": float(np.max(np.abs(H @ Z @ H - X))),
        "||S^2 - Z||_max": float(np.max(np.abs(S @ S - Z))),
        "||T^2 - S||_max": float(np.max(np.abs(T @ T - S))),
        "||CNOT^2 - I||_max": float(np.max(np.abs(cnot_2q @ cnot_2q - i4))),
    }
    return checks


def main() -> None:
    gate_bank = {
        "I": I2,
        "X": X,
        "Y": Y,
        "Z": Z,
        "H": H,
        "S": S,
        "T": T,
        "Rx(pi/5)": rx(np.pi / 5.0),
        "Rz(pi/7)": rz(np.pi / 7.0),
        "CNOT(2q)": cnot_matrix(2, 0, 1),
    }

    print("Quantum Gates MVP (explicit linear-algebra simulation)")
    print()
    print("[1] Unitarity checks")
    for name, gate in gate_bank.items():
        ok, dev = is_unitary(gate)
        print(f"  {name:<10} unitary={str(ok):<5} max_dev={dev:.3e}")
        assert ok, f"Gate {name} failed unitarity check (max_dev={dev})."

    single = run_single_qubit_examples()
    bell = run_bell_state_example()
    checks = run_identity_checks()

    print()
    print("[2] Single-qubit examples")
    print(f"  H|0>      = {state_to_dirac(single['after_h'])}")
    print(f"  X|0>      = {state_to_dirac(single['after_x'])}")
    print(f"  Z|+>      = {state_to_dirac(single['after_z_plus'])}")
    print(f"  Rx(theta)|0>, theta={single['rx_theta']:.4f}")
    print(f"            = {state_to_dirac(single['after_rx'])}")

    print()
    print("[3] Bell-state example")
    print(f"  After H(q0): {state_to_dirac(bell['after_h'])}")
    print(f"  Bell state : {state_to_dirac(bell['bell'])}")
    print(f"  Fidelity to (|00>+|11>)/sqrt(2): {bell['bell_fidelity']:.12f}")
    print(f"  Probabilities [|00>,|01>,|10>,|11>]: {np.array2string(bell['bell_probs'], precision=6)}")

    print()
    print("[4] Gate identity residuals")
    for key, val in checks.items():
        print(f"  {key:<20} {val:.3e}")

    ket0 = single["ket0"]
    ket1 = single["ket1"]
    ket_plus = single["ket_plus"]
    ket_minus = normalize_state(ket0 - ket1)

    assert fidelity(single["after_h"], ket_plus) > 1.0 - 1e-12
    assert fidelity(single["after_x"], ket1) > 1.0 - 1e-12
    assert fidelity(single["after_z_plus"], ket_minus) > 1.0 - 1e-12

    expected_bell_probs = np.array([0.5, 0.0, 0.0, 0.5], dtype=np.float64)
    assert bell["bell_fidelity"] > 1.0 - 1e-12
    assert np.max(np.abs(bell["bell_probs"] - expected_bell_probs)) < 1e-12

    for key, val in checks.items():
        assert val < 1e-10, f"Identity check failed: {key} residual={val}"

    print("All checks passed.")


if __name__ == "__main__":
    main()
