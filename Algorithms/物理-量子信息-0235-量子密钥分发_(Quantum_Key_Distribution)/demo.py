"""BB84 quantum key distribution MVP.

This script provides a minimal, source-traceable simulation of BB84 QKD with:
- random basis preparation/measurement,
- optional intercept-resend eavesdropping,
- basis sifting and QBER estimation,
- simple error-correction leakage accounting,
- privacy amplification via universal binary hashing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BB84Params:
    """Configuration for one BB84 simulation run."""

    n_qubits: int = 4096
    channel_bit_flip: float = 0.01
    eve_intercept_prob: float = 0.0
    sample_fraction: float = 0.2
    seed: int = 7


def binary_entropy(p: float) -> float:
    """Binary entropy H2(p) in bits, numerically stable at boundaries."""

    if p <= 0.0 or p >= 1.0:
        return 0.0
    p_clip = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(-(p_clip * np.log2(p_clip) + (1.0 - p_clip) * np.log2(1.0 - p_clip)))


def privacy_amplification(raw_key: np.ndarray, out_len: int, rng: np.random.Generator) -> np.ndarray:
    """Compress key with a random binary universal hash matrix (mod 2)."""

    if out_len <= 0 or raw_key.size == 0:
        return np.empty(0, dtype=np.uint8)

    hash_matrix = rng.integers(0, 2, size=(out_len, raw_key.size), dtype=np.uint8)
    projected = (hash_matrix.astype(np.uint16) @ raw_key.astype(np.uint16)) & 1
    return projected.astype(np.uint8)


def simulate_bb84(params: BB84Params) -> dict[str, float | int | np.ndarray]:
    """Run one BB84 protocol simulation and return diagnostics."""

    if params.n_qubits < 100:
        raise ValueError("n_qubits must be >= 100 for a meaningful QBER estimate.")

    rng = np.random.default_rng(params.seed)
    n = params.n_qubits

    # Alice prepares random bits in random bases (0: Z basis, 1: X basis).
    alice_bits = rng.integers(0, 2, size=n, dtype=np.uint8)
    alice_bases = rng.integers(0, 2, size=n, dtype=np.uint8)

    # The signal initially carries Alice's prepared states.
    signal_bits = alice_bits.copy()
    signal_bases = alice_bases.copy()

    # Eve optional intercept-resend attack.
    eve_mask = rng.random(n) < params.eve_intercept_prob
    eve_bases = rng.integers(0, 2, size=n, dtype=np.uint8)
    eve_measured = np.empty(n, dtype=np.uint8)
    eve_basis_match = eve_bases == signal_bases
    eve_measured[eve_basis_match] = signal_bits[eve_basis_match]
    mismatch_count = int(np.count_nonzero(~eve_basis_match))
    if mismatch_count > 0:
        eve_measured[~eve_basis_match] = rng.integers(0, 2, size=mismatch_count, dtype=np.uint8)

    signal_bits[eve_mask] = eve_measured[eve_mask]
    signal_bases[eve_mask] = eve_bases[eve_mask]

    # Physical channel noise: independent bit-flip errors.
    flip_mask = rng.random(n) < params.channel_bit_flip
    signal_bits = signal_bits ^ flip_mask.astype(np.uint8)

    # Bob measures with random bases.
    bob_bases = rng.integers(0, 2, size=n, dtype=np.uint8)
    bob_bits = np.empty(n, dtype=np.uint8)
    bob_basis_match = bob_bases == signal_bases
    bob_bits[bob_basis_match] = signal_bits[bob_basis_match]
    bob_mismatch_count = int(np.count_nonzero(~bob_basis_match))
    if bob_mismatch_count > 0:
        bob_bits[~bob_basis_match] = rng.integers(0, 2, size=bob_mismatch_count, dtype=np.uint8)

    # Public basis announcement and sifting.
    sift_mask = alice_bases == bob_bases
    alice_sift = alice_bits[sift_mask]
    bob_sift = bob_bits[sift_mask]
    sifted_len = int(alice_sift.size)
    if sifted_len < 20:
        raise RuntimeError("Sifted key is too short; increase n_qubits.")

    # Parameter estimation from a random public sample of sifted bits.
    sample_size = max(1, int(params.sample_fraction * sifted_len))
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

    # Minimal EC accounting: disclose approximately n*H2(QBER) bits, then correct to Alice's key.
    leak_ec_bits = int(np.ceil(raw_len * binary_entropy(qber_sample)))
    corrected_key = raw_alice.copy()

    # Asymptotic BB84 secret fraction bound under one-way EC.
    secret_fraction = max(0.0, 1.0 - 2.0 * binary_entropy(qber_sample))
    final_key_len = int(np.floor(secret_fraction * raw_len))
    final_key = privacy_amplification(corrected_key, final_key_len, rng)

    qber_total_sifted = float(np.mean(alice_sift != bob_sift))

    return {
        "alice_bits": alice_bits,
        "alice_bases": alice_bases,
        "bob_bases": bob_bases,
        "bob_bits": bob_bits,
        "sifted_len": sifted_len,
        "sample_size": sample_size,
        "raw_len": raw_len,
        "qber_sample": qber_sample,
        "qber_raw_before_ec": qber_raw_before_ec,
        "qber_total_sifted": qber_total_sifted,
        "secret_fraction": float(secret_fraction),
        "leak_ec_bits": leak_ec_bits,
        "final_key_len": final_key_len,
        "eve_intercepted": int(np.count_nonzero(eve_mask)),
        "channel_flips": int(np.count_nonzero(flip_mask)),
        "final_key": final_key,
    }


def summarize_result(name: str, params: BB84Params, result: dict[str, float | int | np.ndarray]) -> pd.DataFrame:
    """Create a concise tabular summary for printing."""

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


def main() -> None:
    clean_params = BB84Params(eve_intercept_prob=0.0, seed=7)
    attacked_params = BB84Params(eve_intercept_prob=1.0, seed=7)

    clean = simulate_bb84(clean_params)
    attacked = simulate_bb84(attacked_params)

    print("=== BB84 QKD MVP ===")
    print(summarize_result("clean_channel", clean_params, clean).to_string(index=False))
    print()
    print(summarize_result("intercept_resend", attacked_params, attacked).to_string(index=False))

    # Basic sanity checks: eavesdropping should markedly increase QBER and kill key rate.
    if clean["qber_sample"] >= 0.08:
        raise AssertionError("Clean-channel QBER unexpectedly high; check simulation logic.")
    if attacked["qber_sample"] <= 0.18:
        raise AssertionError("Intercept-resend QBER unexpectedly low; check basis handling.")
    if attacked["final_key_len"] != 0:
        raise AssertionError("Attacked scenario should produce zero secure key in this setup.")
    if clean["final_key_len"] <= 64:
        raise AssertionError("Clean scenario should yield a non-trivial secure key.")


if __name__ == "__main__":
    main()
