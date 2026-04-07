"""Knapsack cryptanalysis MVP (Merkle-Hellman + lattice attack).

This script demonstrates:
1) Building a classic Merkle-Hellman knapsack cryptosystem instance.
2) Encrypting a random binary message.
3) Recovering the plaintext directly from (public key, ciphertext)
   via a low-density subset-sum lattice embedding + LLL reduction.

No interactive input is required.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Vector = List[int]
Matrix = List[Vector]


@dataclass
class MerkleHellmanKeypair:
    private_superincreasing: List[int]
    modulus: int
    multiplier: int
    public_key: List[int]


@dataclass
class AttackResult:
    seed: int
    keypair: MerkleHellmanKeypair
    plaintext_bits: List[int]
    ciphertext: int
    recovered_bits: List[int]
    reduced_basis: Matrix


def dot_f64(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def gram_schmidt(basis: Matrix) -> Tuple[List[List[float]], List[List[float]], List[float]]:
    n = len(basis)
    dim = len(basis[0])
    mu = [[0.0] * n for _ in range(n)]
    b_star = [[0.0] * dim for _ in range(n)]
    b_star_norm = [0.0] * n

    for i in range(n):
        b_star[i] = [float(x) for x in basis[i]]
        for j in range(i):
            if b_star_norm[j] < 1e-18:
                mu[i][j] = 0.0
                continue
            mu[i][j] = dot_f64([float(x) for x in basis[i]], b_star[j]) / b_star_norm[j]
            proj = mu[i][j]
            for k in range(dim):
                b_star[i][k] -= proj * b_star[j][k]
        b_star_norm[i] = dot_f64(b_star[i], b_star[i])

    return mu, b_star, b_star_norm


def lll_reduce(basis: Matrix, delta: float = 0.75) -> Matrix:
    """Educational LLL implementation for small dimensions.

    The basis is interpreted as row vectors.
    """
    if not basis:
        return []

    reduced = [row[:] for row in basis]
    n = len(reduced)
    dim = len(reduced[0])
    mu, _b_star, b_star_norm = gram_schmidt(reduced)

    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            q = int(round(mu[k][j]))
            if q != 0:
                for c in range(dim):
                    reduced[k][c] -= q * reduced[j][c]

        mu, _b_star, b_star_norm = gram_schmidt(reduced)

        left = b_star_norm[k]
        right = (delta - mu[k][k - 1] ** 2) * b_star_norm[k - 1]
        if left + 1e-12 >= right:
            k += 1
        else:
            reduced[k], reduced[k - 1] = reduced[k - 1], reduced[k]
            mu, _b_star, b_star_norm = gram_schmidt(reduced)
            k = max(k - 1, 1)

    return reduced


def random_superincreasing_sequence(n: int, rng: random.Random) -> List[int]:
    seq: List[int] = []
    total = 0
    for _ in range(n):
        nxt = total + rng.randint(2, 12)
        seq.append(nxt)
        total += nxt
    return seq


def generate_merkle_hellman_keypair(n: int, rng: random.Random) -> MerkleHellmanKeypair:
    private_w = random_superincreasing_sequence(n, rng)
    total = sum(private_w)

    # Keep modulus much larger than sum(w) to induce low-density public subset-sum.
    modulus = rng.randint(total * 40, total * 80)

    while True:
        multiplier = rng.randint(2, modulus - 2)
        if math.gcd(multiplier, modulus) == 1:
            break

    public_key = [(multiplier * wi) % modulus for wi in private_w]
    return MerkleHellmanKeypair(private_w, modulus, multiplier, public_key)


def encrypt_bits(public_key: Sequence[int], bits: Sequence[int]) -> int:
    if len(public_key) != len(bits):
        raise ValueError("public_key and bits must have the same length")
    return sum(w * b for w, b in zip(public_key, bits))


def decrypt_with_private(keypair: MerkleHellmanKeypair, ciphertext: int) -> List[int]:
    inv_multiplier = pow(keypair.multiplier, -1, keypair.modulus)
    transformed = (ciphertext * inv_multiplier) % keypair.modulus

    bits = [0] * len(keypair.private_superincreasing)
    remain = transformed
    for i in range(len(keypair.private_superincreasing) - 1, -1, -1):
        wi = keypair.private_superincreasing[i]
        if wi <= remain:
            bits[i] = 1
            remain -= wi

    if remain != 0:
        raise ValueError("private-key decryption failed; invalid ciphertext for this key")
    return bits


def subset_sum_density(weights: Sequence[int]) -> float:
    return len(weights) / math.log2(max(weights))


def build_lattice_embedding(weights: Sequence[int], target: int, scale: int = 1) -> Matrix:
    """Build the Lagarias-Odlyzko style integer embedding basis.

    Basis vectors (rows):
    - For i in [0, n-1]: e_i * 2 with final coord 2*scale*weights[i]
    - Last row: all ones in first n coords, final coord 2*scale*target

    If x in {0,1}^n is a valid solution, then:
      v = sum_i x_i * row_i - last_row = (2x-1, ..., 2x_n-1, 0)
    which is very short (first n entries are +-1).
    """
    n = len(weights)
    basis: Matrix = []

    for i in range(n):
        row = [0] * (n + 1)
        row[i] = 2
        row[-1] = 2 * scale * int(weights[i])
        basis.append(row)

    last = [1] * n + [2 * scale * int(target)]
    basis.append(last)
    return basis


def vector_norm_sq(v: Sequence[int]) -> int:
    return sum(x * x for x in v)


def decode_pm1_vector(candidate: Sequence[int], weights: Sequence[int], target: int) -> Optional[List[int]]:
    if candidate[-1] != 0:
        return None

    front = candidate[:-1]
    if any(x not in (-1, 1) for x in front):
        return None

    bits_a = [(x + 1) // 2 for x in front]
    bits_b = [(1 - x) // 2 for x in front]

    for bits in (bits_a, bits_b):
        if sum(w * b for w, b in zip(weights, bits)) == target:
            return bits
    return None


def recover_bits_via_lll(weights: Sequence[int], target: int, scale: int = 1) -> Tuple[Optional[List[int]], Matrix]:
    basis = build_lattice_embedding(weights, target, scale=scale)
    reduced = lll_reduce(basis, delta=0.75)

    # Candidate pool: reduced rows, their negations, and pairwise +/- combinations.
    candidates: List[List[int]] = []
    for row in reduced:
        candidates.append(row)
        candidates.append([-x for x in row])

    m = len(reduced)
    for i in range(m):
        for j in range(i + 1, m):
            add_vec = [reduced[i][k] + reduced[j][k] for k in range(len(reduced[i]))]
            sub_vec = [reduced[i][k] - reduced[j][k] for k in range(len(reduced[i]))]
            candidates.append(add_vec)
            candidates.append(sub_vec)
            candidates.append([-x for x in add_vec])
            candidates.append([-x for x in sub_vec])

    candidates.sort(key=vector_norm_sq)

    for vec in candidates:
        bits = decode_pm1_vector(vec, weights, target)
        if bits is not None:
            return bits, reduced

    return None, reduced


def find_attackable_instance(n: int = 10, max_trials: int = 120) -> AttackResult:
    """Search deterministic seeds until the lattice attack succeeds."""
    for seed in range(max_trials):
        rng = random.Random(20260407 + seed)
        keypair = generate_merkle_hellman_keypair(n, rng)
        plaintext_bits = [rng.randint(0, 1) for _ in range(n)]

        if all(b == 0 for b in plaintext_bits):
            plaintext_bits[rng.randrange(n)] = 1

        ciphertext = encrypt_bits(keypair.public_key, plaintext_bits)
        recovered, reduced = recover_bits_via_lll(keypair.public_key, ciphertext, scale=1)

        if recovered == plaintext_bits:
            return AttackResult(
                seed=seed,
                keypair=keypair,
                plaintext_bits=plaintext_bits,
                ciphertext=ciphertext,
                recovered_bits=recovered,
                reduced_basis=reduced,
            )

    raise RuntimeError("LLL attack did not succeed within trial budget")


def bits_to_str(bits: Sequence[int]) -> str:
    return "".join(str(b) for b in bits)


def main() -> None:
    result = find_attackable_instance(n=10, max_trials=120)
    keypair = result.keypair

    # Sanity check: private-key decryption should match plaintext.
    private_recovered = decrypt_with_private(keypair, result.ciphertext)
    assert private_recovered == result.plaintext_bits

    print("Knapsack Cryptanalysis Demo (Merkle-Hellman + LLL)")
    print("=" * 72)
    print(f"selected_seed        : {result.seed}")
    print(f"dimension(n)         : {len(keypair.public_key)}")
    print(f"subset-sum density   : {subset_sum_density(keypair.public_key):.4f}")
    print(f"modulus              : {keypair.modulus}")
    print(f"multiplier           : {keypair.multiplier}")
    print(f"ciphertext           : {result.ciphertext}")
    print(f"plaintext bits       : {bits_to_str(result.plaintext_bits)}")
    print(f"recovered (attack)   : {bits_to_str(result.recovered_bits)}")

    shortest = min(result.reduced_basis, key=vector_norm_sq)
    print(f"shortest LLL vector  : {shortest}")
    print(f"shortest norm^2      : {vector_norm_sq(shortest)}")

    assert result.recovered_bits == result.plaintext_bits
    print("status               : SUCCESS (ciphertext solved via lattice cryptanalysis)")


if __name__ == "__main__":
    main()
