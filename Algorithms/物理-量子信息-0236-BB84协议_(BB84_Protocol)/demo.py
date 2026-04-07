"""BB84 protocol MVP with explicit single-qubit state simulation.

The script models:
- Alice state preparation in Z/X bases,
- optional Eve intercept-resend attack,
- channel bit-flip noise,
- Bob basis measurement,
- basis sifting, QBER estimation, and key extraction.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-12
KET0 = np.array([1.0, 0.0], dtype=np.complex128)
KET1 = np.array([0.0, 1.0], dtype=np.complex128)
KET_PLUS = (KET0 + KET1) / np.sqrt(2.0)
KET_MINUS = (KET0 - KET1) / np.sqrt(2.0)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


@dataclass(frozen=True)
class BB84Params:
    """Configuration for one BB84 simulation run."""

    n_qubits: int = 4096
    channel_bit_flip: float = 0.01
    eve_intercept_prob: float = 0.0
    sample_fraction: float = 0.20
    seed: int = 2026


def normalize_state(state: np.ndarray) -> np.ndarray:
    """Return a normalized statevector."""

    norm = float(np.linalg.norm(state))
    if norm <= 0.0:
        raise ValueError("State norm must be positive.")
    return state / norm


def state_from_bit_basis(bit: int, basis: int) -> np.ndarray:
    """Prepare |0>,|1> in Z basis or |+>,|-> in X basis."""

    if basis not in (0, 1):
        raise ValueError("Basis must be 0 (Z) or 1 (X).")
    if bit not in (0, 1):
        raise ValueError("Bit must be 0 or 1.")

    if basis == 0:
        return KET0.copy() if bit == 0 else KET1.copy()
    return KET_PLUS.copy() if bit == 0 else KET_MINUS.copy()


def measurement_probabilities(state: np.ndarray, basis: int) -> np.ndarray:
    """Return measurement outcome probabilities [p(0), p(1)]."""

    state = normalize_state(state)

    if basis == 0:
        p0 = float(abs(state[0]) ** 2)
        p1 = float(abs(state[1]) ** 2)
    elif basis == 1:
        p0 = float(abs(np.vdot(KET_PLUS, state)) ** 2)
        p1 = float(abs(np.vdot(KET_MINUS, state)) ** 2)
    else:
        raise ValueError("Basis must be 0 (Z) or 1 (X).")

    probs = np.array([p0, p1], dtype=np.float64)
    probs = np.clip(probs, 0.0, 1.0)
    total = float(np.sum(probs))
    if total <= EPS:
        raise RuntimeError("Degenerate measurement probabilities.")
    return probs / total


def measure_state(state: np.ndarray, basis: int, rng: np.random.Generator) -> tuple[int, np.ndarray]:
    """Projectively measure a qubit in the chosen basis."""

    probs = measurement_probabilities(state, basis)
    outcome = int(rng.choice(2, p=probs))
    post_state = state_from_bit_basis(outcome, basis)
    return outcome, post_state


def apply_channel_noise(state: np.ndarray, bit_flip_prob: float, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Apply independent Pauli-X bit-flip channel."""

    if bit_flip_prob < 0.0 or bit_flip_prob > 1.0:
        raise ValueError("bit_flip_prob must be in [0, 1].")

    if rng.random() < bit_flip_prob:
        return normalize_state(PAULI_X @ state), 1
    return state.copy(), 0


def binary_entropy(p: float) -> float:
    """Binary entropy H2(p) in bits."""

    if p <= 0.0 or p >= 1.0:
        return 0.0
    p_clip = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(-(p_clip * np.log2(p_clip) + (1.0 - p_clip) * np.log2(1.0 - p_clip)))


def privacy_amplification(raw_key: np.ndarray, out_len: int, rng: np.random.Generator) -> np.ndarray:
    """Universal hashing by random binary matrix multiplication (mod 2)."""

    if out_len <= 0 or raw_key.size == 0:
        return np.empty(0, dtype=np.uint8)

    hash_matrix = rng.integers(0, 2, size=(out_len, raw_key.size), dtype=np.uint8)
    projected = (hash_matrix.astype(np.uint16) @ raw_key.astype(np.uint16)) & 1
    return projected.astype(np.uint8)


def simulate_bb84(params: BB84Params) -> dict[str, float | int | np.ndarray]:
    """Run one BB84 simulation and return diagnostics/results."""

    if params.n_qubits < 100:
        raise ValueError("n_qubits must be >= 100.")

    rng = np.random.default_rng(params.seed)
    n = params.n_qubits

    alice_bits = rng.integers(0, 2, size=n, dtype=np.uint8)
    alice_bases = rng.integers(0, 2, size=n, dtype=np.uint8)
    bob_bases = rng.integers(0, 2, size=n, dtype=np.uint8)

    bob_bits = np.empty(n, dtype=np.uint8)
    eve_intercepted = 0
    channel_flips = 0

    for i in range(n):
        state = state_from_bit_basis(int(alice_bits[i]), int(alice_bases[i]))

        if rng.random() < params.eve_intercept_prob:
            eve_intercepted += 1
            eve_basis = int(rng.integers(0, 2))
            eve_bit, _ = measure_state(state, eve_basis, rng)
            state = state_from_bit_basis(eve_bit, eve_basis)

        state, flipped = apply_channel_noise(state, params.channel_bit_flip, rng)
        channel_flips += flipped

        bob_bit, _ = measure_state(state, int(bob_bases[i]), rng)
        bob_bits[i] = bob_bit

    sift_mask = alice_bases == bob_bases
    alice_sift = alice_bits[sift_mask]
    bob_sift = bob_bits[sift_mask]
    sifted_len = int(alice_sift.size)
    if sifted_len < 20:
        raise RuntimeError("Sifted key too short; increase n_qubits.")

    sample_size = max(1, int(params.sample_fraction * sifted_len))
    sample_size = min(sample_size, sifted_len - 1)

    perm = rng.permutation(sifted_len)
    sample_idx = perm[:sample_size]
    keep_idx = perm[sample_size:]

    sample_alice = alice_sift[sample_idx]
    sample_bob = bob_sift[sample_idx]
    qber_sample = float(np.mean(sample_alice != sample_bob))

    raw_alice = alice_sift[keep_idx]
    raw_bob = bob_sift[keep_idx]
    raw_len = int(raw_alice.size)
    qber_raw_before_ec = float(np.mean(raw_alice != raw_bob)) if raw_len > 0 else 0.0

    leak_ec_bits = int(np.ceil(raw_len * binary_entropy(qber_sample)))

    secret_fraction = max(0.0, 1.0 - 2.0 * binary_entropy(qber_sample))
    final_key_len = int(np.floor(secret_fraction * raw_len))

    corrected_key = raw_alice.copy()
    final_key = privacy_amplification(corrected_key, final_key_len, rng)

    qber_total_sifted = float(np.mean(alice_sift != bob_sift))

    return {
        "sifted_len": sifted_len,
        "sample_size": sample_size,
        "raw_len": raw_len,
        "qber_sample": qber_sample,
        "qber_raw_before_ec": qber_raw_before_ec,
        "qber_total_sifted": qber_total_sifted,
        "secret_fraction": float(secret_fraction),
        "leak_ec_bits": leak_ec_bits,
        "final_key_len": final_key_len,
        "eve_intercepted": eve_intercepted,
        "channel_flips": channel_flips,
        "final_key": final_key,
    }


def summarize_result(name: str, params: BB84Params, result: dict[str, float | int | np.ndarray]) -> pd.DataFrame:
    """Create a concise table for one scenario."""

    return pd.DataFrame(
        [
            {"metric": "scenario", "value": name},
            {"metric": "n_qubits", "value": params.n_qubits},
            {"metric": "eve_intercept_prob", "value": f"{params.eve_intercept_prob:.2f}"},
            {"metric": "channel_bit_flip", "value": f"{params.channel_bit_flip:.3f}"},
            {"metric": "sifted_len", "value": result["sifted_len"]},
            {"metric": "sample_size", "value": result["sample_size"]},
            {"metric": "raw_len", "value": result["raw_len"]},
            {"metric": "qber_sample", "value": f"{result['qber_sample']:.3%}"},
            {"metric": "qber_raw_before_ec", "value": f"{result['qber_raw_before_ec']:.3%}"},
            {"metric": "qber_total_sifted", "value": f"{result['qber_total_sifted']:.3%}"},
            {"metric": "secret_fraction", "value": f"{result['secret_fraction']:.3f}"},
            {"metric": "leak_ec_bits", "value": result["leak_ec_bits"]},
            {"metric": "final_key_len", "value": result["final_key_len"]},
            {"metric": "eve_intercepted", "value": result["eve_intercepted"]},
            {"metric": "channel_flips", "value": result["channel_flips"]},
        ]
    )


def run_sanity_checks(clean: dict[str, float | int | np.ndarray], attacked: dict[str, float | int | np.ndarray]) -> None:
    """Check expected BB84 behavior."""

    if clean["qber_sample"] >= 0.08:
        raise AssertionError("Clean scenario QBER unexpectedly high.")
    if attacked["qber_sample"] <= 0.18:
        raise AssertionError("Intercept-resend QBER unexpectedly low.")
    if attacked["final_key_len"] != 0:
        raise AssertionError("Attacked scenario should produce zero secure key with these settings.")
    if clean["final_key_len"] <= 64:
        raise AssertionError("Clean scenario should produce a non-trivial secure key.")


def main() -> None:
    clean_params = BB84Params(eve_intercept_prob=0.0, seed=2026)
    attacked_params = BB84Params(eve_intercept_prob=1.0, seed=2026)

    clean = simulate_bb84(clean_params)
    attacked = simulate_bb84(attacked_params)
    run_sanity_checks(clean, attacked)

    print("=== BB84 Protocol MVP (explicit qubit-state simulation) ===")
    print(summarize_result("clean_channel", clean_params, clean).to_string(index=False))
    print()
    print(summarize_result("intercept_resend", attacked_params, attacked).to_string(index=False))
    print("All checks passed.")


if __name__ == "__main__":
    main()
