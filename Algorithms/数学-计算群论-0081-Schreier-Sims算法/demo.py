"""Schreier-Sims algorithm MVP for small permutation groups.

This script is intentionally transparent:
- It implements core permutation operations directly;
- Builds stabilizer-chain transversals via Schreier trees;
- Uses deterministic Schreier-Sims refinement (no black-box package);
- Verifies group order on several known examples.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import permutations
from math import prod
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

Permutation = Tuple[int, ...]


@dataclass
class SchreierSimsResult:
    base: List[int]
    strong_generators: List[Permutation]
    transversals: List[Dict[int, Permutation]]
    orbit_sizes: List[int]
    order: int
    rounds: int


def identity_perm(n: int) -> Permutation:
    return tuple(range(n))


def is_identity(p: Permutation) -> bool:
    return all(i == p[i] for i in range(len(p)))


def compose(p: Permutation, q: Permutation) -> Permutation:
    """Return permutation for applying p then q (right action convention)."""
    return tuple(q[p[i]] for i in range(len(p)))


def inverse(p: Permutation) -> Permutation:
    inv = [0] * len(p)
    for i, x in enumerate(p):
        inv[x] = i
    return tuple(inv)


def perm_from_cycles(n: int, cycles: Sequence[Sequence[int]]) -> Permutation:
    p = list(range(n))
    for cyc in cycles:
        if len(cyc) < 2:
            continue
        for i in range(len(cyc)):
            p[cyc[i]] = cyc[(i + 1) % len(cyc)]
    return tuple(p)


def fixes_prefix(g: Permutation, base: Sequence[int], level: int) -> bool:
    return all(g[base[j]] == base[j] for j in range(level))


def dedup_non_identity(perms: Iterable[Permutation]) -> List[Permutation]:
    seen = set()
    out: List[Permutation] = []
    for p in perms:
        if is_identity(p) or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def level_generators(
    strong_generators: Sequence[Permutation], base: Sequence[int], level: int
) -> List[Permutation]:
    return [g for g in strong_generators if fixes_prefix(g, base, level)]


def orbit_with_transversal(
    beta: int, generators: Sequence[Permutation], n: int
) -> Dict[int, Permutation]:
    reps: Dict[int, Permutation] = {beta: identity_perm(n)}
    q: Deque[int] = deque([beta])
    while q:
        x = q.popleft()
        ux = reps[x]
        for g in generators:
            y = g[x]
            if y not in reps:
                reps[y] = compose(ux, g)
                q.append(y)
    return reps


def build_transversals(
    strong_generators: Sequence[Permutation], base: Sequence[int], n: int
) -> List[Dict[int, Permutation]]:
    transversals: List[Dict[int, Permutation]] = []
    for level, beta in enumerate(base):
        gens = level_generators(strong_generators, base, level)
        reps = orbit_with_transversal(beta, gens, n)
        transversals.append(reps)
    return transversals


def sift(
    g: Permutation,
    base: Sequence[int],
    transversals: Sequence[Dict[int, Permutation]],
) -> Tuple[Optional[int], Permutation]:
    residue = g
    for level, beta in enumerate(base):
        image = residue[beta]
        rep = transversals[level].get(image)
        if rep is None:
            return level, residue
        residue = compose(residue, inverse(rep))
    if is_identity(residue):
        return None, residue
    return len(base), residue


def schreier_generators_for_level(
    level: int,
    strong_generators: Sequence[Permutation],
    base: Sequence[int],
    n: int,
) -> List[Permutation]:
    gens = level_generators(strong_generators, base, level)
    beta = base[level]
    reps = orbit_with_transversal(beta, gens, n)
    out: List[Permutation] = []
    for alpha, u_alpha in reps.items():
        for g in gens:
            alpha_g = g[alpha]
            u_alpha_g = reps[alpha_g]
            s = compose(compose(u_alpha, g), inverse(u_alpha_g))
            if not is_identity(s):
                out.append(s)
    return out


def schreier_sims(
    n: int,
    generators: Sequence[Permutation],
    base: Optional[Sequence[int]] = None,
    max_rounds: int = 50,
) -> SchreierSimsResult:
    if n <= 0:
        raise ValueError("n must be positive")

    if base is None:
        base = list(range(n))
    if len(base) != n or sorted(base) != list(range(n)):
        raise ValueError("This MVP expects base to be a permutation of [0..n-1]")

    strong_generators = dedup_non_identity(generators)
    if not strong_generators:
        transversals = [{b: identity_perm(n)} for b in base]
        return SchreierSimsResult(
            base=list(base),
            strong_generators=[],
            transversals=transversals,
            orbit_sizes=[1] * len(base),
            order=1,
            rounds=0,
        )

    rounds = 0
    for rounds in range(1, max_rounds + 1):
        transversals = build_transversals(strong_generators, base, n)
        changed = False
        for level in range(len(base)):
            schreier_gens = schreier_generators_for_level(
                level, strong_generators, base, n
            )
            for cand in schreier_gens:
                fail_level, residue = sift(cand, base, transversals)
                if fail_level is not None and (not is_identity(residue)):
                    if residue not in strong_generators:
                        strong_generators.append(residue)
                        changed = True
        if not changed:
            break
    else:
        raise RuntimeError("Schreier-Sims did not stabilize within max_rounds")

    transversals = build_transversals(strong_generators, base, n)
    orbit_sizes = [len(t) for t in transversals]
    order = prod(orbit_sizes)
    return SchreierSimsResult(
        base=list(base),
        strong_generators=strong_generators,
        transversals=transversals,
        orbit_sizes=orbit_sizes,
        order=order,
        rounds=rounds,
    )


def group_closure(n: int, generators: Sequence[Permutation]) -> List[Permutation]:
    e = identity_perm(n)
    group = {e}
    q: Deque[Permutation] = deque([e])
    gens = dedup_non_identity(generators)
    while q:
        x = q.popleft()
        for g in gens:
            y = compose(x, g)
            if y not in group:
                group.add(y)
                q.append(y)
    return list(group)


def find_non_member(n: int, group: Sequence[Permutation]) -> Optional[Permutation]:
    group_set = set(group)
    for p in permutations(range(n)):
        if p not in group_set:
            return p
    return None


def run_case(
    name: str,
    n: int,
    generators: Sequence[Permutation],
    expected_order: Optional[int] = None,
) -> None:
    result = schreier_sims(n=n, generators=generators, base=list(range(n)))
    brute_group = group_closure(n, generators)
    brute_order = len(brute_group)

    print(f"\n[{name}]")
    print(f"  strong generators: {len(result.strong_generators)}")
    print(f"  orbit sizes      : {result.orbit_sizes}")
    print(f"  chain order      : {result.order}")
    print(f"  brute-force order: {brute_order}")
    print(f"  rounds           : {result.rounds}")

    if expected_order is not None:
        assert result.order == expected_order, (
            f"{name}: expected {expected_order}, got {result.order}"
        )
    assert result.order == brute_order, f"{name}: chain/bruteforce mismatch"

    # Positive membership check: pick one known group element and sift it.
    probe_member = brute_group[len(brute_group) // 2]
    fail_level, residue = sift(probe_member, result.base, result.transversals)
    assert fail_level is None and is_identity(residue), f"{name}: member check failed"

    # Negative membership check: pick a permutation outside the group, if exists.
    probe_non_member = find_non_member(n, brute_group)
    if probe_non_member is not None:
        fail_level, residue = sift(probe_non_member, result.base, result.transversals)
        assert fail_level is not None or (not is_identity(residue)), (
            f"{name}: non-member unexpectedly accepted"
        )

    print("  checks           : passed")


def main() -> None:
    # S4 = < (0 1), (0 1 2 3) >, |S4| = 24
    s4_gens = [
        perm_from_cycles(4, [(0, 1)]),
        perm_from_cycles(4, [(0, 1, 2, 3)]),
    ]
    run_case("S4", n=4, generators=s4_gens, expected_order=24)

    # D4 (symmetries of square), |D4| = 8
    d4_gens = [
        perm_from_cycles(4, [(0, 1, 2, 3)]),  # rotation
        perm_from_cycles(4, [(1, 3)]),  # reflection across axis through 0 and 2
    ]
    run_case("D4", n=4, generators=d4_gens, expected_order=8)

    # A5 = < (0 1 2), (0 1 2 3 4) >, |A5| = 60
    a5_gens = [
        perm_from_cycles(5, [(0, 1, 2)]),
        perm_from_cycles(5, [(0, 1, 2, 3, 4)]),
    ]
    run_case("A5", n=5, generators=a5_gens, expected_order=60)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
