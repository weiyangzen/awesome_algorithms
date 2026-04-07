"""Toy CKKS MVP (educational, source-visible, no cryptography black box).

This script intentionally implements a compact CKKS-like pipeline:
1) key generation on an RLWE-style relation,
2) encode/encrypt/decrypt for approximate real vectors,
3) homomorphic addition,
4) homomorphic ciphertext-ciphertext multiplication + rescale.

Important: this is a teaching MVP, not production cryptography.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class CKKSParams:
    ring_dim: int
    modulus_q: int
    scale: int
    sigma: float
    a_bound: int
    seed: int


@dataclass(frozen=True)
class PublicKey:
    b: List[int]
    a: List[int]


@dataclass(frozen=True)
class SecretKey:
    s: List[int]
    s_square: List[int]


@dataclass(frozen=True)
class Ciphertext:
    components: List[List[int]]
    scale: float


def ensure_power_of_two(value: int) -> None:
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f"ring_dim must be power-of-two, got {value}.")


def center_lift(value: int, q: int) -> int:
    reduced = int(value) % q
    half = q // 2
    if reduced > half:
        reduced -= q
    return reduced


def vec_add_mod(left: Sequence[int], right: Sequence[int], q: int) -> List[int]:
    if len(left) != len(right):
        raise ValueError("vector length mismatch in vec_add_mod")
    return [(int(x) + int(y)) % q for x, y in zip(left, right)]


def vec_mul_mod(left: Sequence[int], right: Sequence[int], q: int) -> List[int]:
    if len(left) != len(right):
        raise ValueError("vector length mismatch in vec_mul_mod")
    return [(int(x) * int(y)) % q for x, y in zip(left, right)]


def vec_center(values: Sequence[int], q: int) -> List[int]:
    return [center_lift(v, q) for v in values]


def sample_ternary(rng: np.random.Generator, length: int) -> List[int]:
    return [int(x) for x in rng.integers(-1, 2, size=length)]


def sample_error(rng: np.random.Generator, length: int, sigma: float) -> List[int]:
    noise = rng.normal(loc=0.0, scale=sigma, size=length)
    return [int(np.rint(x)) for x in noise]


def sample_small_uniform(rng: np.random.Generator, length: int, bound: int) -> List[int]:
    if bound < 1:
        raise ValueError(f"a_bound must be >= 1, got {bound}.")
    return [int(x) for x in rng.integers(-bound, bound + 1, size=length)]


def encode_real(values: Sequence[float], params: CKKSParams) -> List[int]:
    if len(values) > params.ring_dim:
        raise ValueError(
            f"input length {len(values)} exceeds ring_dim={params.ring_dim}."
        )
    padded = np.zeros(params.ring_dim, dtype=np.float64)
    padded[: len(values)] = np.asarray(values, dtype=np.float64)
    scaled = np.rint(padded * params.scale).astype(np.int64)
    return [int(x) % params.modulus_q for x in scaled]


def decode_real(
    plaintext: Sequence[int], params: CKKSParams, slots: int, scale: float
) -> np.ndarray:
    if slots > params.ring_dim:
        raise ValueError(f"slots {slots} exceeds ring_dim={params.ring_dim}.")
    centered = np.array(vec_center(plaintext, params.modulus_q), dtype=np.float64)
    return centered[:slots] / float(scale)


def keygen(params: CKKSParams, rng: np.random.Generator) -> tuple[PublicKey, SecretKey]:
    s = sample_ternary(rng, params.ring_dim)
    a = sample_small_uniform(rng, params.ring_dim, params.a_bound)
    e = sample_error(rng, params.ring_dim, params.sigma)
    a_times_s = vec_mul_mod(a, s, params.modulus_q)
    minus_a_times_s = [(-x) % params.modulus_q for x in a_times_s]
    b = vec_add_mod(minus_a_times_s, e, params.modulus_q)
    s_square = vec_mul_mod(s, s, params.modulus_q)
    return PublicKey(b=b, a=a), SecretKey(s=s, s_square=s_square)


def encrypt(
    public_key: PublicKey,
    message: Sequence[int],
    params: CKKSParams,
    rng: np.random.Generator,
) -> Ciphertext:
    if len(message) != params.ring_dim:
        raise ValueError("encoded plaintext length must equal ring_dim.")
    u = sample_ternary(rng, params.ring_dim)
    e1 = sample_error(rng, params.ring_dim, params.sigma)
    e2 = sample_error(rng, params.ring_dim, params.sigma)

    bu = vec_mul_mod(public_key.b, u, params.modulus_q)
    au = vec_mul_mod(public_key.a, u, params.modulus_q)
    c0 = vec_add_mod(vec_add_mod(bu, e1, params.modulus_q), message, params.modulus_q)
    c1 = vec_add_mod(au, e2, params.modulus_q)
    return Ciphertext(components=[c0, c1], scale=float(params.scale))


def decrypt(ciphertext: Ciphertext, secret_key: SecretKey, params: CKKSParams) -> List[int]:
    q = params.modulus_q
    if len(ciphertext.components) == 2:
        c0, c1 = ciphertext.components
        return vec_add_mod(c0, vec_mul_mod(c1, secret_key.s, q), q)
    if len(ciphertext.components) == 3:
        c0, c1, c2 = ciphertext.components
        part_01 = vec_add_mod(c0, vec_mul_mod(c1, secret_key.s, q), q)
        return vec_add_mod(part_01, vec_mul_mod(c2, secret_key.s_square, q), q)
    raise ValueError("ciphertext must have 2 or 3 components in this MVP.")


def add_ciphertexts(left: Ciphertext, right: Ciphertext, params: CKKSParams) -> Ciphertext:
    if len(left.components) != len(right.components):
        raise ValueError("ciphertext arity mismatch in add_ciphertexts.")
    if abs(left.scale - right.scale) > 1e-9:
        raise ValueError("ciphertext scales must match for addition.")
    q = params.modulus_q
    summed = [vec_add_mod(a, b, q) for a, b in zip(left.components, right.components)]
    return Ciphertext(components=summed, scale=left.scale)


def mul_ciphertexts(left: Ciphertext, right: Ciphertext, params: CKKSParams) -> Ciphertext:
    if len(left.components) != 2 or len(right.components) != 2:
        raise ValueError("this MVP supports multiplication for 2-component ciphertexts only.")
    q = params.modulus_q
    a0, a1 = left.components
    b0, b1 = right.components

    d0 = vec_mul_mod(a0, b0, q)
    d1_left = vec_mul_mod(a0, b1, q)
    d1_right = vec_mul_mod(a1, b0, q)
    d1 = vec_add_mod(d1_left, d1_right, q)
    d2 = vec_mul_mod(a1, b1, q)
    return Ciphertext(components=[d0, d1, d2], scale=left.scale * right.scale)


def rescale(ciphertext: Ciphertext, factor: float, params: CKKSParams) -> Ciphertext:
    if factor <= 0:
        raise ValueError("rescale factor must be positive.")
    q = params.modulus_q
    rescaled_components: List[List[int]] = []
    for comp in ciphertext.components:
        centered = vec_center(comp, q)
        divided = [int(np.rint(v / factor)) % q for v in centered]
        rescaled_components.append(divided)
    return Ciphertext(components=rescaled_components, scale=ciphertext.scale / factor)


def max_abs_error(observed: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(observed - expected)))


def digest_vector(values: Sequence[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    payload = ",".join(f"{v:.10f}" for v in arr)
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    params = CKKSParams(
        ring_dim=8,
        modulus_q=2**50,
        scale=2**20,
        sigma=1.0,
        a_bound=8,
        seed=2026,
    )
    ensure_power_of_two(params.ring_dim)
    rng = np.random.default_rng(params.seed)

    # Fixed deterministic inputs for reproducible validation.
    x = np.array([0.125, -0.5, 1.125, 0.75], dtype=np.float64)
    y = np.array([1.5, -0.25, 0.5, -1.25], dtype=np.float64)

    pk, sk = keygen(params, rng)
    mx = encode_real(x, params)
    my = encode_real(y, params)
    ctx = encrypt(pk, mx, params, rng)
    cty = encrypt(pk, my, params, rng)

    # Decrypt original x.
    x_plain = decrypt(ctx, sk, params)
    x_hat = decode_real(x_plain, params, slots=len(x), scale=ctx.scale)

    # Homomorphic addition: Enc(x) + Enc(y) -> x + y.
    ct_add = add_ciphertexts(ctx, cty, params)
    add_plain = decrypt(ct_add, sk, params)
    add_hat = decode_real(add_plain, params, slots=len(x), scale=ct_add.scale)

    # Homomorphic multiplication + rescale: Enc(x) * Enc(y) -> x * y.
    ct_mul = mul_ciphertexts(ctx, cty, params)
    ct_mul_rescaled = rescale(ct_mul, factor=params.scale, params=params)
    mul_plain = decrypt(ct_mul_rescaled, sk, params)
    mul_hat = decode_real(
        mul_plain, params, slots=len(x), scale=ct_mul_rescaled.scale
    )

    target_add = x + y
    target_mul = x * y

    err_x = max_abs_error(x_hat, x)
    err_add = max_abs_error(add_hat, target_add)
    err_mul = max_abs_error(mul_hat, target_mul)

    print("=== CKKS Toy MVP (CS-0154) ===")
    print(
        f"params: ring_dim={params.ring_dim}, q=2^{params.modulus_q.bit_length()-1}, "
        f"scale={params.scale}, sigma={params.sigma}, a_bound={params.a_bound}"
    )
    print(f"x           = {np.array2string(x, precision=6)}")
    print(f"y           = {np.array2string(y, precision=6)}")
    print(f"dec(x)      = {np.array2string(x_hat, precision=6)}")
    print(f"dec(x+y)    = {np.array2string(add_hat, precision=6)}")
    print(f"dec(x*y)    = {np.array2string(mul_hat, precision=6)}")
    print()
    print(f"max|dec(x)-x|      = {err_x:.6e}")
    print(f"max|dec(x+y)-(x+y)|= {err_add:.6e}")
    print(f"max|dec(x*y)-(x*y)|= {err_mul:.6e}")
    print(f"final_scale        = {ct_mul_rescaled.scale:.1f}")
    print(f"fingerprint        = {digest_vector(mul_hat)}")

    if len(ct_mul.components) != 3:
        raise RuntimeError("ciphertext multiplication should produce 3 components.")
    if abs(ct_mul_rescaled.scale - params.scale) > 1e-9:
        raise RuntimeError("rescale did not restore expected scale.")

    if err_x > 5e-5 or err_add > 5e-5 or err_mul > 2e-4:
        raise RuntimeError(
            "CKKS toy demo exceeded error budget; adjust parameters or implementation."
        )


if __name__ == "__main__":
    main()
