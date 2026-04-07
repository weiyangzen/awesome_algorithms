"""Index Calculus discrete logarithm (minimal runnable MVP).

Solve g^x ≡ h (mod p) in Z_p^* for small teaching-sized parameters.
This MVP intentionally stays with Python standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Relation:
    """A relation k = sum(e_i * log_g(prime_i)) over modulus (p-1)."""

    k: int
    exponents: tuple[int, ...]


def prime_sieve(limit: int) -> list[int]:
    """Return all primes <= limit using a small sieve."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    i = 2
    while i * i <= limit:
        if is_prime[i]:
            j = i * i
            while j <= limit:
                is_prime[j] = False
                j += i
        i += 1
    return [i for i, ok in enumerate(is_prime) if ok]


def factor_integer(n: int) -> dict[int, int]:
    """Prime factorization by trial division (good enough for MVP-sized n)."""
    if n <= 0:
        raise ValueError("n must be positive.")
    x = n
    factors: dict[int, int] = {}
    d = 2
    while d * d <= x:
        while x % d == 0:
            factors[d] = factors.get(d, 0) + 1
            x //= d
        d += 1 if d == 2 else 2
    if x > 1:
        factors[x] = factors.get(x, 0) + 1
    return factors


def factor_over_base(value: int, factor_base: list[int]) -> tuple[int, ...] | None:
    """Factor value over a fixed factor base. Return exponent vector if smooth."""
    if value <= 0:
        return None
    x = value
    exponents = [0] * len(factor_base)
    for idx, prime in enumerate(factor_base):
        while x % prime == 0:
            exponents[idx] += 1
            x //= prime
    if x != 1:
        return None
    return tuple(exponents)


def collect_relations(
    p: int,
    g: int,
    group_order: int,
    factor_base: list[int],
    needed: int,
    seed: int,
    max_trials: int,
) -> list[Relation]:
    """Collect k such that g^k mod p is factor-base smooth."""
    if needed <= 0:
        return []

    rng = random.Random(seed)
    relations: list[Relation] = []
    seen_vectors: set[tuple[int, ...]] = set()

    for _ in range(max_trials):
        k = rng.randrange(1, group_order)
        y = pow(g, k, p)
        exponents = factor_over_base(y, factor_base)
        if exponents is None:
            continue
        if exponents in seen_vectors:
            continue
        seen_vectors.add(exponents)
        relations.append(Relation(k=k % group_order, exponents=exponents))
        if len(relations) >= needed:
            return relations

    raise RuntimeError(
        "Unable to collect enough smooth relations. "
        "Try a larger smooth_bound or max_trials."
    )


def gauss_jordan_mod(
    matrix: list[list[int]],
    rhs: list[int],
    prime_mod: int,
) -> tuple[list[int] | None, int]:
    """Solve linear system over GF(prime_mod) with free vars set to 0.

    Returns (solution, rank). If inconsistent, solution is None.
    """
    if not matrix:
        return [], 0

    m = len(matrix)
    n = len(matrix[0])
    a = [[v % prime_mod for v in row] for row in matrix]
    b = [v % prime_mod for v in rhs]

    pivot_row = 0
    pivot_cols: list[int] = []

    for col in range(n):
        pivot = None
        for row in range(pivot_row, m):
            if a[row][col] % prime_mod != 0:
                pivot = row
                break
        if pivot is None:
            continue

        a[pivot_row], a[pivot] = a[pivot], a[pivot_row]
        b[pivot_row], b[pivot] = b[pivot], b[pivot_row]

        inv = pow(a[pivot_row][col], -1, prime_mod)
        a[pivot_row] = [(v * inv) % prime_mod for v in a[pivot_row]]
        b[pivot_row] = (b[pivot_row] * inv) % prime_mod

        for row in range(m):
            if row == pivot_row:
                continue
            factor = a[row][col] % prime_mod
            if factor == 0:
                continue
            a[row] = [
                (a[row][j] - factor * a[pivot_row][j]) % prime_mod for j in range(n)
            ]
            b[row] = (b[row] - factor * b[pivot_row]) % prime_mod

        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == m:
            break

    for row in range(m):
        if all(v % prime_mod == 0 for v in a[row]) and b[row] % prime_mod != 0:
            return None, len(pivot_cols)

    solution = [0] * n
    for row, col in enumerate(pivot_cols):
        solution[col] = b[row] % prime_mod

    return solution, len(pivot_cols)


def crt_pairwise(residues: list[int], moduli: list[int]) -> int:
    """Chinese Remainder Theorem for pairwise coprime moduli."""
    if len(residues) != len(moduli):
        raise ValueError("residues/moduli length mismatch")

    x = 0
    m = 1
    for ai, mi in zip(residues, moduli):
        delta = (ai - x) % mi
        inv = pow(m, -1, mi)
        t = (delta * inv) % mi
        x += m * t
        m *= mi
    return x % m


def find_target_smooth_multiple(
    h: int,
    g: int,
    p: int,
    group_order: int,
    factor_base: list[int],
    max_tries: int,
) -> tuple[int, tuple[int, ...]]:
    """Find t such that h * g^t (mod p) is factor-base smooth."""
    tries = min(max_tries, group_order)
    for t in range(tries):
        value = (h * pow(g, t, p)) % p
        exponents = factor_over_base(value, factor_base)
        if exponents is not None:
            return t, exponents
    raise RuntimeError(
        "Failed to find smooth target multiple; increase smooth_bound or max_tries."
    )


def index_calculus_discrete_log(
    p: int,
    g: int,
    h: int,
    smooth_bound: int = 31,
    relation_surplus: int = 8,
    seed: int = 2026,
) -> tuple[int, dict[str, int]]:
    """Compute x such that g^x ≡ h (mod p) using a teaching-level Index Calculus."""
    if p <= 2:
        raise ValueError("p must be an odd prime for this MVP.")
    if not (1 <= g < p and 1 <= h < p):
        raise ValueError("g and h must lie in [1, p-1].")

    n = p - 1
    if pow(g, n, p) != 1:
        raise ValueError("g is not in Z_p^*.")

    factors = factor_integer(n)
    if any(exp != 1 for exp in factors.values()):
        raise ValueError(
            "This MVP supports square-free p-1 only (for simple CRT over prime moduli)."
        )

    prime_moduli = sorted(factors.keys())
    factor_base = [q for q in prime_sieve(smooth_bound) if q < p]
    if not factor_base:
        raise ValueError("Factor base is empty; increase smooth_bound.")

    # Keep trying with more relations until all per-prime systems are full rank.
    logs_by_prime: dict[int, list[int]] = {}
    relation_count = len(factor_base) + relation_surplus

    for round_idx in range(6):
        relations = collect_relations(
            p=p,
            g=g,
            group_order=n,
            factor_base=factor_base,
            needed=relation_count,
            seed=seed + round_idx,
            max_trials=250000,
        )

        logs_by_prime.clear()
        success = True
        for modulus in prime_moduli:
            matrix = [[e % modulus for e in rel.exponents] for rel in relations]
            rhs = [rel.k % modulus for rel in relations]
            solution, rank = gauss_jordan_mod(matrix, rhs, modulus)
            if solution is None or rank < len(factor_base):
                success = False
                break
            logs_by_prime[modulus] = solution

        if success:
            break
        relation_count += max(2, len(factor_base) // 2)
    else:
        raise RuntimeError(
            "Relation matrix did not reach full rank; try a larger smooth_bound."
        )

    t, target_exponents = find_target_smooth_multiple(
        h=h,
        g=g,
        p=p,
        group_order=n,
        factor_base=factor_base,
        max_tries=n,
    )

    residues: list[int] = []
    for modulus in prime_moduli:
        logs = logs_by_prime[modulus]
        rhs = sum(e * logs[i] for i, e in enumerate(target_exponents)) % modulus
        x_mod = (rhs - (t % modulus)) % modulus
        residues.append(x_mod)

    x = crt_pairwise(residues, prime_moduli) % n
    if pow(g, x, p) != h:
        raise RuntimeError("Recovered logarithm failed verification.")

    details = {
        "group_order": n,
        "factor_base_size": len(factor_base),
        "relations_used": relation_count,
        "target_shift_t": t,
    }
    return x, details


def main() -> None:
    # Fixed, reproducible demo instance in Z_p^*.
    p = 1019
    g = 2
    secret_x = 321
    h = pow(g, secret_x, p)

    recovered_x, details = index_calculus_discrete_log(
        p=p,
        g=g,
        h=h,
        smooth_bound=31,
        relation_surplus=10,
        seed=2026,
    )

    print("指数积分法 (Index Calculus) MVP")
    print(f"p={p}, g={g}, h={h}, n=p-1={details['group_order']}")
    print(
        "factor_base_size="
        f"{details['factor_base_size']}, relations_used={details['relations_used']}"
    )
    print(f"target_shift_t={details['target_shift_t']}")
    print(f"恢复出的 x = {recovered_x}")
    print(f"真实 x(模 {details['group_order']}) = {secret_x % details['group_order']}")
    print(f"校验 g^x mod p = {pow(g, recovered_x, p)}")


if __name__ == "__main__":
    main()
