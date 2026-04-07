"""Toy Groth16-style zk-SNARK MVP for relation x * w = y over a finite field.

This implementation is educational and intentionally simplified:
- It models G1/G2/GT group elements as field exponents.
- It models pairing as multiplication in the same finite field.
- It demonstrates setup/prove/verify data flow, not production security.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import random


@dataclass(frozen=True)
class CircuitSpec:
    """One-constraint circuit: x * w = y with variables [1, x, y, w]."""

    u: tuple[int, ...]  # Left linear form coefficients
    v: tuple[int, ...]  # Right linear form coefficients
    w: tuple[int, ...]  # Output linear form coefficients
    public_indices: tuple[int, ...]  # Includes constant-1 slot
    private_indices: tuple[int, ...]


@dataclass(frozen=True)
class TrustedSetup:
    """Toy proving/verifying key material."""

    prime: int
    tau: int
    alpha: int
    beta: int
    gamma: int
    delta: int
    circuit: CircuitSpec
    ic_public: tuple[int, ...]
    k_private: tuple[int, ...]


@dataclass(frozen=True)
class Proof:
    """Toy Groth16 proof tuple."""

    A: int
    B: int
    C: int


def mod_inv(a: int, prime: int) -> int:
    """Multiplicative inverse in F_p (prime field)."""
    a %= prime
    if a == 0:
        raise ValueError("inverse of zero does not exist")
    return pow(a, prime - 2, prime)


def sample_nonzero(rng: random.Random, prime: int) -> int:
    return rng.randrange(1, prime)


def dot_mod(coeffs: tuple[int, ...], values: tuple[int, ...], prime: int) -> int:
    if len(coeffs) != len(values):
        raise ValueError("length mismatch in dot_mod")
    acc = 0
    for c, v in zip(coeffs, values):
        acc = (acc + (c % prime) * (v % prime)) % prime
    return acc


def pairing(a: int, b: int, prime: int) -> int:
    """Toy bilinear map in exponent space (educational stand-in)."""
    return (a % prime) * (b % prime) % prime


def build_multiplication_circuit() -> CircuitSpec:
    # Variable order: z = [1, x, y, w]
    # Constraint: (x) * (w) = (y)
    u = (0, 1, 0, 0)
    v = (0, 0, 0, 1)
    w = (0, 0, 1, 0)
    public_indices = (0, 1, 2)
    private_indices = (3,)
    return CircuitSpec(u=u, v=v, w=w, public_indices=public_indices, private_indices=private_indices)


def trusted_setup(seed: int = 2026, prime: int = 2_147_483_647) -> TrustedSetup:
    """Create toy trusted setup material for the one-constraint circuit."""
    rng = random.Random(seed)
    circuit = build_multiplication_circuit()

    tau = sample_nonzero(rng, prime)
    alpha = sample_nonzero(rng, prime)
    beta = sample_nonzero(rng, prime)
    gamma = sample_nonzero(rng, prime)
    delta = sample_nonzero(rng, prime)

    inv_gamma = mod_inv(gamma, prime)
    inv_delta = mod_inv(delta, prime)

    # IC_j = (beta*u_j(tau) + alpha*v_j(tau) + w_j(tau)) / gamma  for public slots.
    ic_public_list: list[int] = []
    for j in circuit.public_indices:
        numerator = (
            beta * circuit.u[j] + alpha * circuit.v[j] + circuit.w[j]
        ) % prime
        ic_public_list.append((numerator * inv_gamma) % prime)

    # K_j = (beta*u_j(tau) + alpha*v_j(tau) + w_j(tau)) / delta for private slots.
    # Store as full-length vector for easy dot product against assignment.
    k_private = [0] * len(circuit.u)
    for j in circuit.private_indices:
        numerator = (
            beta * circuit.u[j] + alpha * circuit.v[j] + circuit.w[j]
        ) % prime
        k_private[j] = (numerator * inv_delta) % prime

    return TrustedSetup(
        prime=prime,
        tau=tau,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        circuit=circuit,
        ic_public=tuple(ic_public_list),
        k_private=tuple(k_private),
    )


def make_assignment(x: int, y: int, w: int, prime: int) -> tuple[int, ...]:
    """Build full variable assignment z = [1, x, y, w] in F_p."""
    return (1 % prime, x % prime, y % prime, w % prime)


def prove(setup: TrustedSetup, x: int, y: int, witness_w: int, seed: int = 99) -> Proof:
    """Generate toy proof for statement 'know w such that x*w=y (mod p)'."""
    p = setup.prime
    z = make_assignment(x=x, y=y, w=witness_w, prime=p)

    a_eval = dot_mod(setup.circuit.u, z, p)
    b_eval = dot_mod(setup.circuit.v, z, p)

    # For this one-constraint toy circuit, no H-query term is needed in C.
    private_term = dot_mod(setup.k_private, z, p)

    rng = random.Random(seed)
    r = rng.randrange(0, p)
    s = rng.randrange(0, p)

    A = (setup.alpha + a_eval + r * setup.delta) % p
    B = (setup.beta + b_eval + s * setup.delta) % p

    # Groth16-style randomized C construction (toy algebra in exponent space).
    C = (private_term + A * s + B * r - r * s * setup.delta) % p

    return Proof(A=A, B=B, C=C)


def verify(setup: TrustedSetup, x: int, y: int, proof: Proof) -> bool:
    """Verify toy pairing equation for public statement (x, y)."""
    p = setup.prime
    x_mod = x % p
    y_mod = y % p

    # vk_x = IC_0 + x * IC_1 + y * IC_2
    if len(setup.ic_public) != 3:
        raise ValueError("expected exactly 3 public IC terms: [1, x, y]")
    vk_x = (
        setup.ic_public[0]
        + x_mod * setup.ic_public[1]
        + y_mod * setup.ic_public[2]
    ) % p

    lhs = pairing(proof.A, proof.B, p)
    rhs = (
        pairing(setup.alpha, setup.beta, p)
        + pairing(vk_x, setup.gamma, p)
        + pairing(proof.C, setup.delta, p)
    ) % p
    return lhs == rhs


def main() -> None:
    setup = trusted_setup(seed=2026)

    # Public statement: x * w = y (mod p), with private witness w.
    x = 13
    witness_w = 21
    y = (x * witness_w) % setup.prime

    proof = prove(setup, x=x, y=y, witness_w=witness_w, seed=99)
    ok_valid = verify(setup, x=x, y=y, proof=proof)

    bad_public_y = (y + 1) % setup.prime
    ok_bad_public = verify(setup, x=x, y=bad_public_y, proof=proof)

    bad_proof = replace(proof, C=(proof.C + 1234567) % setup.prime)
    ok_tampered = verify(setup, x=x, y=y, proof=bad_proof)

    bad_witness_proof = prove(setup, x=x, y=y, witness_w=witness_w + 5, seed=99)
    ok_bad_witness = verify(setup, x=x, y=y, proof=bad_witness_proof)

    print("Toy zk-SNARK (Groth16-style) demo")
    print(f"prime field modulus p = {setup.prime}")
    print(f"public statement: x={x}, y={y}")
    print("\nVerification results:")
    print(f"- valid proof: {ok_valid}")
    print(f"- same proof but wrong public y: {ok_bad_public}")
    print(f"- tampered proof element C: {ok_tampered}")
    print(f"- proof generated from wrong witness: {ok_bad_witness}")


if __name__ == "__main__":
    main()
