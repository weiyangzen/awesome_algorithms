"""BGV/BFV-style toy MVP.

This demo is intentionally educational:
- Ring: R_q = Z_q[x] / (x^n + 1)
- Message space: R_t
- Uses RLWE-style public-key encryption structure
- Supports homomorphic add/mul (mul returns degree-2 ciphertext)

It omits production features (RNS/NTT, relinearization keys, modulus chain),
but keeps the source-level algebraic dataflow explicit and runnable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

Poly = np.ndarray
Ciphertext = List[Poly]


@dataclass(frozen=True)
class Params:
    """Toy parameters for a deterministic MVP run."""

    n: int = 8
    t: int = 17
    q: int = 17 * 4096  # q % t == 0 for clean toy decryption modulo t
    noise_bound: int = 1
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.n <= 0 or (self.n & (self.n - 1)) != 0:
            raise ValueError("n must be a positive power of two")
        if self.t <= 1:
            raise ValueError("t must be > 1")
        if self.q <= self.t:
            raise ValueError("q must be larger than t")
        if self.q % self.t != 0:
            raise ValueError("this toy demo requires q % t == 0")
        if self.noise_bound < 0:
            raise ValueError("noise_bound must be >= 0")


def mod_q(poly: Poly, q: int) -> Poly:
    return np.mod(poly, q).astype(np.int64)


def poly_add(a: Poly, b: Poly, q: int) -> Poly:
    return mod_q(a + b, q)


def poly_mul_negacyclic(a: Poly, b: Poly, q: int) -> Poly:
    """Multiply in Z_q[x]/(x^n + 1)."""

    n = int(a.shape[0])
    conv = np.convolve(a, b)
    out = np.zeros(n, dtype=np.int64)
    out += conv[:n]
    for k in range(n, 2 * n - 1):
        out[k - n] -= conv[k]  # x^n == -1
    return mod_q(out, q)


def sample_small(rng: np.random.Generator, n: int, bound: int) -> Poly:
    return rng.integers(-bound, bound + 1, size=n, dtype=np.int64)


def zero_poly(n: int) -> Poly:
    return np.zeros(n, dtype=np.int64)


class ToyBGVBFV:
    """A compact RLWE-style toy scheme for BGV/BFV concept demonstration.

    Design choice in this MVP:
    - Noise terms are multiplied by t so that decryption modulo t stays exact.
    - Multiplication outputs degree-2 ciphertext (no relinearization step).
    """

    def __init__(self, params: Params):
        self.p = params
        self.rng = np.random.default_rng(self.p.seed)
        self.sk, self.pk = self._keygen()

    def _keygen(self) -> Tuple[Poly, Tuple[Poly, Poly]]:
        n, q, t = self.p.n, self.p.q, self.p.t
        s = sample_small(self.rng, n, self.p.noise_bound)
        a = self.rng.integers(0, q, size=n, dtype=np.int64)
        e = sample_small(self.rng, n, self.p.noise_bound)

        # b = -a*s + t*e (mod q)
        b = mod_q(-poly_mul_negacyclic(a, s, q) + t * e, q)
        return s, (a, b)

    def encrypt(self, m: Sequence[int]) -> Ciphertext:
        m_arr = np.array(m, dtype=np.int64)
        if m_arr.shape != (self.p.n,):
            raise ValueError(f"message must have shape ({self.p.n},)")

        q, t, n = self.p.q, self.p.t, self.p.n
        a, b = self.pk

        r = sample_small(self.rng, n, self.p.noise_bound)
        e1 = sample_small(self.rng, n, self.p.noise_bound)
        e2 = sample_small(self.rng, n, self.p.noise_bound)

        m_lift = mod_q(m_arr, q)
        c0 = poly_add(poly_add(poly_mul_negacyclic(b, r, q), t * e1, q), m_lift, q)
        c1 = poly_add(poly_mul_negacyclic(a, r, q), t * e2, q)
        return [c0, c1]

    def decrypt(self, ct: Ciphertext) -> Poly:
        if len(ct) == 0:
            raise ValueError("ciphertext must not be empty")

        q, t, n = self.p.q, self.p.t, self.p.n
        acc = zero_poly(n)

        s_pow = zero_poly(n)
        s_pow[0] = 1  # s^0

        for i, c_i in enumerate(ct):
            if c_i.shape != (n,):
                raise ValueError(f"ciphertext component {i} has invalid shape {c_i.shape}")
            if i == 0:
                term = c_i
            else:
                s_pow = poly_mul_negacyclic(s_pow, self.sk, q)
                term = poly_mul_negacyclic(c_i, s_pow, q)
            acc = poly_add(acc, term, q)

        return np.mod(acc, t).astype(np.int64)

    def add(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        q, n = self.p.q, self.p.n
        L = max(len(ct1), len(ct2))
        out: Ciphertext = []
        z = zero_poly(n)
        for i in range(L):
            a_i = ct1[i] if i < len(ct1) else z
            b_i = ct2[i] if i < len(ct2) else z
            out.append(poly_add(a_i, b_i, q))
        return out

    def mul(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        if len(ct1) != 2 or len(ct2) != 2:
            raise ValueError("toy mul currently supports fresh 2-component ciphertext only")

        q = self.p.q
        c00, c01 = ct1
        c10, c11 = ct2

        # Degree-2 ciphertext in secret s: d0 + d1*s + d2*s^2
        d0 = poly_mul_negacyclic(c00, c10, q)
        d1 = poly_add(
            poly_mul_negacyclic(c00, c11, q),
            poly_mul_negacyclic(c01, c10, q),
            q,
        )
        d2 = poly_mul_negacyclic(c01, c11, q)
        return [d0, d1, d2]


def run_demo() -> None:
    params = Params()
    scheme = ToyBGVBFV(params)

    m1 = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int64) % params.t
    m2 = np.array([5, 8, 9, 7, 9, 3, 2, 3], dtype=np.int64) % params.t

    ct1 = scheme.encrypt(m1)
    ct2 = scheme.encrypt(m2)

    dec1 = scheme.decrypt(ct1)
    dec2 = scheme.decrypt(ct2)

    assert np.array_equal(dec1, m1), "decrypt(encrypt(m1)) failed"
    assert np.array_equal(dec2, m2), "decrypt(encrypt(m2)) failed"

    ct_add = scheme.add(ct1, ct2)
    dec_add = scheme.decrypt(ct_add)
    exp_add = np.mod(m1 + m2, params.t)
    assert np.array_equal(dec_add, exp_add), "homomorphic add failed"

    ct_mul = scheme.mul(ct1, ct2)
    dec_mul = scheme.decrypt(ct_mul)
    exp_mul = poly_mul_negacyclic(m1, m2, params.t)
    assert np.array_equal(dec_mul, exp_mul), "homomorphic mul failed"

    print("=== Toy BGV/BFV MVP ===")
    print(f"params: n={params.n}, q={params.q}, t={params.t}, noise_bound={params.noise_bound}")
    print("m1:", m1.tolist())
    print("m2:", m2.tolist())
    print("decrypt(enc(m1)):", dec1.tolist())
    print("decrypt(enc(m2)):", dec2.tolist())
    print("add result:", dec_add.tolist())
    print("add expected:", exp_add.tolist())
    print("mul result:", dec_mul.tolist())
    print("mul expected:", exp_mul.tolist())
    print("ciphertext lengths: fresh=2, after mul=", len(ct_mul))
    print("note: relinearization and modulus switching are intentionally omitted in this toy MVP")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
