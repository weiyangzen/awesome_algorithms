"""CS-0132 Diffie-Hellman 密钥交换：最小可运行 MVP。

实现内容:
1) 使用 RFC 3526 MODP Group 14（2048-bit 素数）做基础 DH 交换。
2) 对公钥做基本合法性检查，避免明显无效输入。
3) 用 SHA-256 从共享秘密派生会话密钥。
4) 提供 MITM 演示，说明“仅 DH 不认证”会被中间人攻击。

运行方式:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import numpy as np

# RFC 3526, 2048-bit MODP Group (group 14)
P_HEX = (
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
    "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
    "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
    "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF"
)


@dataclass(frozen=True)
class DHParameters:
    """Diffie-Hellman 参数集合。"""

    p: int
    g: int
    q: int


@dataclass(frozen=True)
class DHParty:
    """参与方的私钥与公钥。"""

    private_key: int
    public_key: int


def _int_to_bytes(x: int) -> bytes:
    if x < 0:
        raise ValueError("x must be non-negative")
    nbytes = max(1, (x.bit_length() + 7) // 8)
    return x.to_bytes(nbytes, "big")


def load_default_params() -> DHParameters:
    p = int(P_HEX, 16)
    g = 2
    # group 14 是安全素数群，q=(p-1)/2
    q = (p - 1) // 2
    return DHParameters(p=p, g=g, q=q)


def sample_private_key(rng: np.random.Generator, params: DHParameters, bits: int = 320) -> int:
    """采样私钥 x in [2, q-2]。"""
    if bits < 32:
        raise ValueError("bits must be >= 32")

    raw = int.from_bytes(rng.bytes((bits + 7) // 8), "big")
    return 2 + (raw % (params.q - 3))


def compute_public_key(private_key: int, params: DHParameters) -> int:
    return pow(params.g, private_key, params.p)


def validate_public_key(public_key: int, params: DHParameters) -> None:
    """对对端公钥做基础校验，防止明显无效输入。"""
    if not (2 <= public_key <= params.p - 2):
        raise ValueError("Public key out of valid range")

    # 在安全素数群中，要求落在 q 阶子群。
    if pow(public_key, params.q, params.p) != 1:
        raise ValueError("Public key not in expected subgroup")


def compute_shared_secret(peer_public_key: int, own_private_key: int, params: DHParameters) -> int:
    validate_public_key(peer_public_key, params)
    return pow(peer_public_key, own_private_key, params.p)


def kdf_sha256(shared_secret: int, context: bytes = b"CS-0132-DH-MVP") -> bytes:
    """从 DH 共享秘密派生 32-byte 会话密钥。"""
    return sha256(_int_to_bytes(shared_secret) + context).digest()


def create_party(rng: np.random.Generator, params: DHParameters) -> DHParty:
    priv = sample_private_key(rng, params)
    pub = compute_public_key(priv, params)
    return DHParty(private_key=priv, public_key=pub)


def run_single_exchange(rng: np.random.Generator, params: DHParameters) -> tuple[int, bytes]:
    """执行一次 Alice-Bob DH，返回共享秘密与会话密钥。"""
    alice = create_party(rng, params)
    bob = create_party(rng, params)

    shared_alice = compute_shared_secret(bob.public_key, alice.private_key, params)
    shared_bob = compute_shared_secret(alice.public_key, bob.private_key, params)

    if shared_alice != shared_bob:
        raise AssertionError("DH shared secret mismatch")

    session_key = kdf_sha256(shared_alice)
    return shared_alice, session_key


def run_regression(rounds: int = 200, seed: int = 20260407) -> None:
    print("[Case 1] DH 一致性回归")
    params = load_default_params()
    rng = np.random.default_rng(seed)

    t0 = perf_counter()
    for _ in range(rounds):
        _, session_key = run_single_exchange(rng, params)
        if len(session_key) != 32:
            raise AssertionError("Unexpected key length")
    t1 = perf_counter()

    print(f"rounds={rounds}, seed={seed}, elapsed={t1 - t0:.4f}s: passed")


def run_mitm_demo(seed: int = 7) -> None:
    """演示未认证 DH 可被中间人攻击。"""
    print("\n[Case 2] MITM 风险演示（未认证 DH）")

    params = load_default_params()
    rng = np.random.default_rng(seed)

    alice = create_party(rng, params)
    bob = create_party(rng, params)

    # Mallory 伪造两条链路上的公钥
    mallory_to_alice = create_party(rng, params)
    mallory_to_bob = create_party(rng, params)

    # Alice 收到的是 Mallory 的公钥（而非 Bob 的）
    alice_shared = compute_shared_secret(mallory_to_alice.public_key, alice.private_key, params)
    # Bob 收到的是 Mallory 的另一个公钥（而非 Alice 的）
    bob_shared = compute_shared_secret(mallory_to_bob.public_key, bob.private_key, params)

    # Mallory 分别与 Alice/Bob 建立两份不同共享秘密
    mallory_shared_with_alice = compute_shared_secret(alice.public_key, mallory_to_alice.private_key, params)
    mallory_shared_with_bob = compute_shared_secret(bob.public_key, mallory_to_bob.private_key, params)

    if alice_shared != mallory_shared_with_alice:
        raise AssertionError("MITM chain Alice-Mallory mismatch")
    if bob_shared != mallory_shared_with_bob:
        raise AssertionError("MITM chain Mallory-Bob mismatch")
    if alice_shared == bob_shared:
        raise AssertionError("Unexpected equality; MITM demo failed")

    alice_key = kdf_sha256(alice_shared)
    bob_key = kdf_sha256(bob_shared)
    ma_key = kdf_sha256(mallory_shared_with_alice)
    mb_key = kdf_sha256(mallory_shared_with_bob)

    print(f"Alice session key (hex, first 16): {alice_key.hex()[:16]}")
    print(f"Bob   session key (hex, first 16): {bob_key.hex()[:16]}")
    print(f"Mallory<->Alice key prefix:        {ma_key.hex()[:16]}")
    print(f"Mallory<->Bob   key prefix:        {mb_key.hex()[:16]}")
    print("结论: Alice 与 Bob 并未共享同一把密钥，说明 DH 必须配合认证机制（如签名/证书）。")


def run_invalid_key_test() -> None:
    print("\n[Case 3] 无效公钥拦截")
    params = load_default_params()
    rng = np.random.default_rng(99)
    alice = create_party(rng, params)

    invalid_keys = [0, 1, params.p - 1, params.p, params.p + 1]
    blocked = 0
    for k in invalid_keys:
        try:
            _ = compute_shared_secret(k, alice.private_key, params)
        except ValueError:
            blocked += 1

    if blocked != len(invalid_keys):
        raise AssertionError("Some invalid public keys were not blocked")
    print(f"blocked_invalid_keys={blocked}/{len(invalid_keys)}: passed")


def main() -> None:
    run_regression()
    run_mitm_demo()
    run_invalid_key_test()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
