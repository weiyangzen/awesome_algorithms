"""Toy zk-STARK MVP for a Fibonacci AIR over a prime field.

This script is intentionally educational:
- It demonstrates a transparent STARK-like flow (AIR -> Merkle commitments
  -> Fiat-Shamir challenges -> FRI-style folding checks).
- It is NOT production cryptography and does not provide zero-knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import List

import numpy as np


FIELD_PRIME = 2**61 - 1


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1) == 0)


def hash_pair(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(left + right).digest()


def hash_field_element(value: int) -> bytes:
    v = value % FIELD_PRIME
    return hashlib.sha256(v.to_bytes(16, byteorder="big", signed=False)).digest()


def build_merkle_tree(values: List[int]) -> List[List[bytes]]:
    if not values:
        raise ValueError("Merkle tree requires at least one value.")
    if not is_power_of_two(len(values)):
        raise ValueError("Merkle tree requires power-of-two leaves in this MVP.")

    level = [hash_field_element(v) for v in values]
    tree: List[List[bytes]] = [level]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            nxt.append(hash_pair(level[i], level[i + 1]))
        tree.append(nxt)
        level = nxt
    return tree


def merkle_root(tree: List[List[bytes]]) -> bytes:
    return tree[-1][0]


def merkle_path(tree: List[List[bytes]], index: int) -> List[bytes]:
    if index < 0 or index >= len(tree[0]):
        raise IndexError("Leaf index out of range for Merkle path.")

    idx = index
    path: List[bytes] = []
    for level in tree[:-1]:
        sibling_index = idx ^ 1
        path.append(level[sibling_index])
        idx //= 2
    return path


def verify_merkle_path(root: bytes, value: int, index: int, path: List[bytes]) -> bool:
    h = hash_field_element(value)
    idx = index
    for sibling in path:
        if idx % 2 == 0:
            h = hash_pair(h, sibling)
        else:
            h = hash_pair(sibling, h)
        idx //= 2
    return h == root


def derive_beta(layer_root: bytes, round_index: int) -> int:
    blob = b"beta" + layer_root + round_index.to_bytes(4, "big")
    value = int.from_bytes(hashlib.sha256(blob).digest(), "big")
    return (value % (FIELD_PRIME - 1)) + 1


def derive_queries(layer_root: bytes, round_index: int, max_q: int, count: int) -> List[int]:
    if max_q <= 0:
        return []
    count = min(count, max_q)

    queries: List[int] = []
    seen = set()
    nonce = 0
    while len(queries) < count:
        blob = (
            b"query"
            + layer_root
            + round_index.to_bytes(4, "big")
            + nonce.to_bytes(4, "big")
        )
        q = int.from_bytes(hashlib.sha256(blob).digest(), "big") % max_q
        if q not in seen:
            queries.append(q)
            seen.add(q)
        nonce += 1
    return queries


def fibonacci_trace(length: int, a0: int, a1: int) -> List[int]:
    if length < 2:
        raise ValueError("Trace length must be >= 2.")
    values = np.zeros(length, dtype=object)
    values[0] = a0 % FIELD_PRIME
    values[1] = a1 % FIELD_PRIME
    for i in range(2, length):
        values[i] = (values[i - 1] + values[i - 2]) % FIELD_PRIME
    return [int(v) for v in values.tolist()]


def compute_air_constraints(trace: List[int], a0: int, a1: int) -> List[int]:
    n = len(trace)
    if n < 4:
        raise ValueError("This MVP expects trace length >= 4.")
    constraints = [0] * n

    for i in range(n - 2):
        constraints[i] = (trace[i + 2] - trace[i + 1] - trace[i]) % FIELD_PRIME
    constraints[n - 2] = (trace[0] - a0) % FIELD_PRIME
    constraints[n - 1] = (trace[1] - a1) % FIELD_PRIME
    return constraints


def fold_layer(values: List[int], beta: int) -> List[int]:
    if len(values) % 2 != 0:
        raise ValueError("FRI folding requires an even number of values.")
    out = []
    for i in range(0, len(values), 2):
        out.append((values[i] + beta * values[i + 1]) % FIELD_PRIME)
    return out


def solve_linear_system_mod(mat: List[List[int]], vec: List[int], mod: int) -> List[int]:
    n = len(mat)
    aug = [row[:] + [vec[i] % mod] for i, row in enumerate(mat)]

    for col in range(n):
        pivot = None
        for row in range(col, n):
            if aug[row][col] % mod != 0:
                pivot = row
                break
        if pivot is None:
            raise ValueError("Matrix is singular modulo prime.")

        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        inv_pivot = pow(aug[col][col], mod - 2, mod)
        for j in range(col, n + 1):
            aug[col][j] = (aug[col][j] * inv_pivot) % mod

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col] % mod
            if factor == 0:
                continue
            for j in range(col, n + 1):
                aug[row][j] = (aug[row][j] - factor * aug[col][j]) % mod

    return [aug[i][n] % mod for i in range(n)]


def interpolate_degree(values: List[int]) -> int:
    n = len(values)
    xs = list(range(n))

    vandermonde: List[List[int]] = []
    for x in xs:
        row = [1]
        for _ in range(1, n):
            row.append((row[-1] * x) % FIELD_PRIME)
        vandermonde.append(row)

    coeffs = solve_linear_system_mod(vandermonde, values, FIELD_PRIME)
    for degree in range(n - 1, -1, -1):
        if coeffs[degree] % FIELD_PRIME != 0:
            return degree
    return -1


@dataclass(frozen=True)
class StarkConfig:
    trace_len: int = 32
    a0: int = 1
    a1: int = 1
    degree_bound: int = 2
    queries_per_round: int = 3

    def __post_init__(self) -> None:
        if not is_power_of_two(self.trace_len):
            raise ValueError("trace_len must be a power of two.")
        if self.trace_len < 4:
            raise ValueError("trace_len must be >= 4.")
        if self.queries_per_round <= 0:
            raise ValueError("queries_per_round must be positive.")


@dataclass
class QueryOpening:
    q: int
    a: int
    b: int
    c: int
    path_a: List[bytes]
    path_b: List[bytes]
    path_c: List[bytes]


@dataclass
class FriRoundProof:
    beta: int
    layer_root: bytes
    next_root: bytes
    openings: List[QueryOpening]


@dataclass
class StarkProof:
    trace: List[int]
    trace_root: bytes
    constraints: List[int]
    constraint_root: bytes
    constraint_degree: int
    degree_bound: int
    fri_rounds: List[FriRoundProof]
    final_value: int


def prove(trace: List[int], cfg: StarkConfig) -> StarkProof:
    if len(trace) != cfg.trace_len:
        raise ValueError("Trace length does not match config.")

    trace_tree = build_merkle_tree(trace)
    trace_root = merkle_root(trace_tree)

    constraints = compute_air_constraints(trace, cfg.a0, cfg.a1)
    constraint_tree = build_merkle_tree(constraints)
    constraint_root = merkle_root(constraint_tree)

    constraint_degree = interpolate_degree(constraints)

    fri_rounds: List[FriRoundProof] = []
    layer_values = constraints
    layer_tree = constraint_tree

    round_index = 0
    while len(layer_values) > 1:
        layer_root = merkle_root(layer_tree)
        beta = derive_beta(layer_root, round_index)
        next_values = fold_layer(layer_values, beta)
        next_tree = build_merkle_tree(next_values)
        next_root = merkle_root(next_tree)

        max_q = len(layer_values) // 2
        query_count = min(cfg.queries_per_round, max_q)
        query_indices = derive_queries(layer_root, round_index, max_q, query_count)

        openings: List[QueryOpening] = []
        for q in query_indices:
            i0 = 2 * q
            i1 = i0 + 1
            openings.append(
                QueryOpening(
                    q=q,
                    a=layer_values[i0],
                    b=layer_values[i1],
                    c=next_values[q],
                    path_a=merkle_path(layer_tree, i0),
                    path_b=merkle_path(layer_tree, i1),
                    path_c=merkle_path(next_tree, q),
                )
            )

        fri_rounds.append(
            FriRoundProof(
                beta=beta,
                layer_root=layer_root,
                next_root=next_root,
                openings=openings,
            )
        )

        layer_values = next_values
        layer_tree = next_tree
        round_index += 1

    return StarkProof(
        trace=trace,
        trace_root=trace_root,
        constraints=constraints,
        constraint_root=constraint_root,
        constraint_degree=constraint_degree,
        degree_bound=cfg.degree_bound,
        fri_rounds=fri_rounds,
        final_value=layer_values[0],
    )


def verify(proof: StarkProof, cfg: StarkConfig) -> bool:
    if len(proof.trace) != cfg.trace_len:
        return False

    reconstructed_trace_root = merkle_root(build_merkle_tree(proof.trace))
    if reconstructed_trace_root != proof.trace_root:
        return False

    t = np.array(proof.trace, dtype=object)
    transition = (t[2:] - t[1:-1] - t[:-2]) % FIELD_PRIME
    if any(int(v) != 0 for v in transition.tolist()):
        return False
    if proof.trace[0] % FIELD_PRIME != cfg.a0 % FIELD_PRIME:
        return False
    if proof.trace[1] % FIELD_PRIME != cfg.a1 % FIELD_PRIME:
        return False

    expected_constraints = compute_air_constraints(proof.trace, cfg.a0, cfg.a1)
    if expected_constraints != proof.constraints:
        return False

    reconstructed_constraint_root = merkle_root(build_merkle_tree(proof.constraints))
    if reconstructed_constraint_root != proof.constraint_root:
        return False

    recomputed_degree = interpolate_degree(proof.constraints)
    if recomputed_degree != proof.constraint_degree:
        return False
    if proof.constraint_degree > proof.degree_bound:
        return False

    layer_values = proof.constraints
    layer_root = proof.constraint_root

    for round_index, round_proof in enumerate(proof.fri_rounds):
        if round_proof.layer_root != layer_root:
            return False

        expected_beta = derive_beta(layer_root, round_index)
        if round_proof.beta != expected_beta:
            return False

        next_values = fold_layer(layer_values, expected_beta)
        next_root = merkle_root(build_merkle_tree(next_values))
        if round_proof.next_root != next_root:
            return False

        max_q = len(layer_values) // 2
        expected_q_count = min(cfg.queries_per_round, max_q)
        expected_queries = derive_queries(
            layer_root,
            round_index,
            max_q,
            expected_q_count,
        )
        given_queries = [o.q for o in round_proof.openings]
        if given_queries != expected_queries:
            return False

        for opening in round_proof.openings:
            i0 = 2 * opening.q
            i1 = i0 + 1

            if opening.a != layer_values[i0] or opening.b != layer_values[i1]:
                return False
            if opening.c != next_values[opening.q]:
                return False

            if not verify_merkle_path(layer_root, opening.a, i0, opening.path_a):
                return False
            if not verify_merkle_path(layer_root, opening.b, i1, opening.path_b):
                return False
            if not verify_merkle_path(next_root, opening.c, opening.q, opening.path_c):
                return False

            if (opening.a + expected_beta * opening.b) % FIELD_PRIME != opening.c:
                return False

        layer_values = next_values
        layer_root = next_root

    if len(layer_values) != 1:
        return False
    if layer_values[0] != proof.final_value:
        return False

    # For a valid Fibonacci AIR instance, all constraints should be zero,
    # so the final folded value should also be zero.
    if proof.final_value % FIELD_PRIME != 0:
        return False

    return True


def main() -> None:
    cfg = StarkConfig(trace_len=32, a0=1, a1=1, degree_bound=2, queries_per_round=3)

    print("=== Toy zk-STARK MVP (educational) ===")
    print(f"Field prime: {FIELD_PRIME}")
    print(
        f"Config: trace_len={cfg.trace_len}, degree_bound={cfg.degree_bound}, "
        f"queries_per_round={cfg.queries_per_round}"
    )

    honest_trace = fibonacci_trace(cfg.trace_len, cfg.a0, cfg.a1)
    honest_proof = prove(honest_trace, cfg)
    honest_ok = verify(honest_proof, cfg)

    print("\n[Case 1] Honest trace")
    print(f"  trace_root      = {honest_proof.trace_root.hex()[:24]}...")
    print(f"  constraint_root = {honest_proof.constraint_root.hex()[:24]}...")
    print(f"  constraint_deg  = {honest_proof.constraint_degree}")
    print(f"  final_fri_value = {honest_proof.final_value}")
    print(f"  verify          = {honest_ok}")

    tampered_trace = honest_trace.copy()
    tampered_trace[10] = (tampered_trace[10] + 12345) % FIELD_PRIME
    tampered_proof = prove(tampered_trace, cfg)
    tampered_ok = verify(tampered_proof, cfg)

    print("\n[Case 2] Tampered trace")
    print(f"  trace_root      = {tampered_proof.trace_root.hex()[:24]}...")
    print(f"  constraint_root = {tampered_proof.constraint_root.hex()[:24]}...")
    print(f"  constraint_deg  = {tampered_proof.constraint_degree}")
    print(f"  final_fri_value = {tampered_proof.final_value}")
    print(f"  verify          = {tampered_ok}")

    if not honest_ok:
        raise AssertionError("Honest proof should verify, but it failed.")
    if tampered_ok:
        raise AssertionError("Tampered proof should fail verification, but it passed.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
