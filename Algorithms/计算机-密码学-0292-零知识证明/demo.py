"""Minimal Schnorr Sigma-protocol MVP for zero-knowledge proofs.

This demo is intentionally educational:
- It uses a tiny prime-order subgroup so every step is easy to audit.
- It demonstrates completeness, soundness intuition, HVZK simulation,
  and special-soundness extraction in source-level code.
- It is NOT production cryptography.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class SchnorrParams:
    """Public parameters for a Schnorr-style Sigma protocol."""

    p: int = 23  # Prime modulus for multiplicative group Z_p^*
    q: int = 11  # Prime subgroup order (q | p-1)
    g: int = 2   # Generator of subgroup of order q

    def __post_init__(self) -> None:
        if self.p <= 2 or self.q <= 2:
            raise ValueError("p and q must be > 2.")
        if (self.p - 1) % self.q != 0:
            raise ValueError("q must divide p-1.")
        if not (1 < self.g < self.p):
            raise ValueError("g must satisfy 1 < g < p.")
        if pow(self.g, self.q, self.p) != 1:
            raise ValueError("g must have order dividing q.")
        if pow(self.g, 1, self.p) == 1:
            raise ValueError("g cannot be identity.")


@dataclass(frozen=True)
class Transcript:
    """One Sigma-protocol transcript tuple (t, e, s)."""

    t: int  # Commitment
    e: int  # Challenge bit
    s: int  # Response in Z_q



def mod_inv(value: int, mod: int) -> int:
    value %= mod
    if value == 0:
        raise ZeroDivisionError("Cannot invert 0 modulo prime.")
    return pow(value, mod - 2, mod)



def public_key(params: SchnorrParams, secret_x: int) -> int:
    if not (0 < secret_x < params.q):
        raise ValueError("secret_x must satisfy 0 < x < q.")
    return pow(params.g, secret_x, params.p)



def verify_transcript(params: SchnorrParams, y: int, tr: Transcript) -> bool:
    if tr.e not in (0, 1):
        return False
    if not (0 <= tr.s < params.q):
        return False
    if not (1 <= tr.t < params.p):
        return False

    lhs = pow(params.g, tr.s, params.p)
    rhs = (tr.t * pow(y, tr.e, params.p)) % params.p
    return lhs == rhs



def honest_transcript(params: SchnorrParams, secret_x: int, rng: np.random.Generator) -> Transcript:
    r = int(rng.integers(0, params.q))
    t = pow(params.g, r, params.p)
    e = int(rng.integers(0, 2))
    s = (r + e * secret_x) % params.q
    tr = Transcript(t=t, e=e, s=s)
    if not verify_transcript(params, public_key(params, secret_x), tr):
        raise RuntimeError("Internal error: honest transcript must verify.")
    return tr



def simulated_transcript(params: SchnorrParams, y: int, rng: np.random.Generator) -> Transcript:
    e = int(rng.integers(0, 2))
    s = int(rng.integers(0, params.q))

    y_to_e = pow(y, e, params.p)
    t = (pow(params.g, s, params.p) * mod_inv(y_to_e, params.p)) % params.p

    tr = Transcript(t=t, e=e, s=s)
    if not verify_transcript(params, y, tr):
        raise RuntimeError("Internal error: simulated transcript must verify.")
    return tr



def cheating_round_without_secret(
    params: SchnorrParams,
    y: int,
    rng: np.random.Generator,
) -> tuple[Transcript, bool]:
    guessed_e = int(rng.integers(0, 2))
    s = int(rng.integers(0, params.q))

    # Precompute commitment for guessed challenge only.
    t = (pow(params.g, s, params.p) * mod_inv(pow(y, guessed_e, params.p), params.p)) % params.p

    actual_e = int(rng.integers(0, 2))
    tr = Transcript(t=t, e=actual_e, s=s)
    ok = verify_transcript(params, y, tr)
    return tr, ok



def run_honest_session(
    params: SchnorrParams,
    secret_x: int,
    rounds: int,
    rng: np.random.Generator,
) -> bool:
    y = public_key(params, secret_x)
    for _ in range(rounds):
        tr = honest_transcript(params, secret_x, rng)
        if not verify_transcript(params, y, tr):
            return False
    return True



def estimate_cheating_rates(
    params: SchnorrParams,
    y: int,
    rounds: int,
    single_round_trials: int,
    session_trials: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    single_success = 0
    for _ in range(single_round_trials):
        _, ok = cheating_round_without_secret(params, y, rng)
        single_success += int(ok)

    session_success = 0
    for _ in range(session_trials):
        ok_all = True
        for _ in range(rounds):
            _, ok = cheating_round_without_secret(params, y, rng)
            if not ok:
                ok_all = False
                break
        session_success += int(ok_all)

    return single_success / single_round_trials, session_success / session_trials



def t_distribution(
    transcripts: Sequence[Transcript],
    challenge: int,
    p: int,
) -> np.ndarray:
    values = [tr.t for tr in transcripts if tr.e == challenge]
    if not values:
        return np.zeros(p, dtype=float)
    counts = np.bincount(np.array(values, dtype=int), minlength=p).astype(float)
    return counts / counts.sum()



def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())



def extract_secret_from_two_challenges(
    params: SchnorrParams,
    secret_x: int,
    rng: np.random.Generator,
) -> int:
    r = int(rng.integers(0, params.q))
    t = pow(params.g, r, params.p)

    tr0 = Transcript(t=t, e=0, s=r % params.q)
    tr1 = Transcript(t=t, e=1, s=(r + secret_x) % params.q)
    y = public_key(params, secret_x)

    if not (verify_transcript(params, y, tr0) and verify_transcript(params, y, tr1)):
        raise RuntimeError("Extractor precondition failed: transcripts must be valid.")

    return (tr1.s - tr0.s) % params.q



def main() -> None:
    rng = np.random.default_rng(20260407)

    params = SchnorrParams(p=23, q=11, g=2)
    secret_x = 7
    y = public_key(params, secret_x)

    rounds = 8
    honest_ok = run_honest_session(params, secret_x, rounds=rounds, rng=rng)

    single_round_rate, session_rate = estimate_cheating_rates(
        params=params,
        y=y,
        rounds=rounds,
        single_round_trials=4000,
        session_trials=4000,
        rng=rng,
    )

    n_samples = 4000
    real_transcripts: List[Transcript] = [honest_transcript(params, secret_x, rng) for _ in range(n_samples)]
    sim_transcripts: List[Transcript] = [simulated_transcript(params, y, rng) for _ in range(n_samples)]

    real_all_valid = all(verify_transcript(params, y, tr) for tr in real_transcripts)
    sim_all_valid = all(verify_transcript(params, y, tr) for tr in sim_transcripts)

    dist_e0 = l1_distance(
        t_distribution(real_transcripts, challenge=0, p=params.p),
        t_distribution(sim_transcripts, challenge=0, p=params.p),
    )
    dist_e1 = l1_distance(
        t_distribution(real_transcripts, challenge=1, p=params.p),
        t_distribution(sim_transcripts, challenge=1, p=params.p),
    )

    extracted_x = extract_secret_from_two_challenges(params, secret_x, rng)

    print("Schnorr Sigma-protocol Zero-Knowledge MVP")
    print(f"Params: p={params.p}, q={params.q}, g={params.g}")
    print(f"Secret x={secret_x}, Public y=g^x mod p={y}")

    print("\nCase 1 | Honest prover completeness")
    print(f"- rounds={rounds}, all accepted={honest_ok}")

    print("\nCase 2 | Cheating prover (no witness)")
    print(f"- empirical single-round success={single_round_rate:.4f}, theory≈0.5000")
    print(
        f"- empirical {rounds}-round session success={session_rate:.4f}, "
        f"theory≈{(0.5 ** rounds):.4f}"
    )

    print("\nCase 3 | HVZK simulator sanity")
    print(f"- all real transcripts verify={real_all_valid}")
    print(f"- all simulated transcripts verify={sim_all_valid}")
    print(f"- L1 distance of t-distribution | e=0: {dist_e0:.4f}")
    print(f"- L1 distance of t-distribution | e=1: {dist_e1:.4f}")

    print("\nCase 4 | Special soundness extractor")
    print(f"- extracted x from two valid challenges on same commitment: {extracted_x}")

    assert honest_ok
    assert 0.40 <= single_round_rate <= 0.60
    assert 0.0 <= session_rate <= 0.02
    assert real_all_valid and sim_all_valid
    assert dist_e0 < 0.25 and dist_e1 < 0.25
    assert extracted_x == secret_x

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
