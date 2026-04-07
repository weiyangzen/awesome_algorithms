"""Pollard's Rho discrete logarithm algorithm (minimal runnable MVP).

Solve for x in g^x = h (mod p) for a known group order n.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
import random


@dataclass(frozen=True)
class State:
    """State for the random walk: x = g^a * h^b (mod p)."""

    x: int
    a: int
    b: int


def _egcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x1, y1 = _egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def mod_inverse(a: int, n: int) -> int:
    a %= n
    g, x, _ = _egcd(a, n)
    if g != 1:
        raise ValueError(f"{a} has no modular inverse modulo {n}")
    return x % n


def solve_linear_congruence(a: int, b: int, n: int) -> list[int]:
    """Solve a*x ≡ b (mod n), return all solutions in [0, n-1]."""
    a %= n
    b %= n
    d = gcd(a, n)
    if b % d != 0:
        return []

    a1, b1, n1 = a // d, b // d, n // d
    x0 = (mod_inverse(a1, n1) * b1) % n1
    return [(x0 + k * n1) % n for k in range(d)]


def rho_step(state: State, g: int, h: int, p: int, n: int) -> State:
    """Partition by x mod 3 and update (x, a, b)."""
    if state.x % 3 == 0:
        return State((state.x * h) % p, state.a, (state.b + 1) % n)
    if state.x % 3 == 1:
        return State((state.x * state.x) % p, (2 * state.a) % n, (2 * state.b) % n)
    return State((state.x * g) % p, (state.a + 1) % n, state.b)


def pollards_rho_discrete_log(
    g: int,
    h: int,
    p: int,
    order: int | None = None,
    max_steps: int | None = None,
    restarts: int = 32,
    seed: int = 2026,
) -> int:
    """Find x such that g^x ≡ h (mod p) using Pollard's Rho + Floyd detection."""
    if p <= 2:
        raise ValueError("p must be an odd prime in this MVP setting.")
    if not (1 <= g < p and 1 <= h < p):
        raise ValueError("g and h must be in [1, p-1].")

    n = (p - 1) if order is None else order
    if n <= 0:
        raise ValueError("group order must be positive.")
    if pow(g, n, p) != 1:
        raise ValueError("provided order is inconsistent: g^order mod p != 1.")
    if max_steps is None:
        max_steps = 3 * n

    rng = random.Random(seed)

    for _ in range(restarts):
        a0 = rng.randrange(n)
        b0 = rng.randrange(n)
        x0 = (pow(g, a0, p) * pow(h, b0, p)) % p
        tortoise = State(x0, a0, b0)
        hare = rho_step(rho_step(tortoise, g, h, p, n), g, h, p, n)

        for _ in range(max_steps):
            if tortoise.x == hare.x:
                # g^(a_t-a_h) = h^(b_h-b_t) = g^(x*(b_h-b_t))
                s = (tortoise.a - hare.a) % n
                r = (hare.b - tortoise.b) % n
                if r == 0:
                    break
                for candidate in solve_linear_congruence(r, s, n):
                    if pow(g, candidate, p) == h:
                        return candidate
                break

            tortoise = rho_step(tortoise, g, h, p, n)
            hare = rho_step(rho_step(hare, g, h, p, n), g, h, p, n)

    raise RuntimeError("Pollard's Rho failed to find a logarithm; increase restarts.")


def main() -> None:
    # Demo on Z_p^* with prime p.
    p = 1019
    g = 2
    order = p - 1
    secret_x = 123
    h = pow(g, secret_x, p)

    recovered_x = pollards_rho_discrete_log(g=g, h=h, p=p, order=order)

    print("Pollard's Rho离散对数算法 MVP")
    print(f"p={p}, g={g}, h={h}, group_order={order}")
    print(f"恢复出的 x = {recovered_x}")
    print(f"校验 g^x mod p = {pow(g, recovered_x, p)}")
    print(f"真实 x(模 {order}) = {secret_x % order}")


if __name__ == "__main__":
    main()
