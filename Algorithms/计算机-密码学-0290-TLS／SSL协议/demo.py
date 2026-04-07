"""Minimal runnable MVP for TLS/SSL protocol internals.

This is an educational, TLS-like implementation that shows source-level flow:
- ephemeral Diffie-Hellman key agreement
- TLS 1.2 style PRF (HMAC-SHA256 based)
- key block expansion into traffic keys
- Finished verification data derivation
- record protection + integrity check + tamper detection

It is NOT a production TLS stack.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass

import numpy as np


# A large prime modulus for finite-field DH in this MVP.
DH_P = (1 << 521) - 1
DH_G = 5
DH_BYTES = (DH_P.bit_length() + 7) // 8

RANDOM_LEN = 32
MASTER_SECRET_LEN = 48
VERIFY_DATA_LEN = 12

MAC_KEY_LEN = 32
ENC_KEY_LEN = 16
IV_LEN = 12
TAG_LEN = 32  # HMAC-SHA256 digest size


@dataclass(frozen=True)
class TrafficKeys:
    mac_key: bytes
    enc_key: bytes
    iv: bytes


@dataclass(frozen=True)
class TLSSession:
    client_random: bytes
    server_random: bytes
    master_secret: bytes
    client_to_server: TrafficKeys
    server_to_client: TrafficKeys
    transcript_hash: bytes


def hmac_sha256(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("XOR inputs must have the same length.")
    return bytes(a ^ b for a, b in zip(left, right))


def tls12_p_hash(secret: bytes, seed: bytes, out_len: int) -> bytes:
    """TLS 1.2 P_hash with HMAC-SHA256."""
    output = bytearray()
    a = seed
    while len(output) < out_len:
        a = hmac_sha256(secret, a)
        output.extend(hmac_sha256(secret, a + seed))
    return bytes(output[:out_len])


def tls12_prf(secret: bytes, label: bytes, seed: bytes, out_len: int) -> bytes:
    """TLS 1.2 PRF(secret, label, seed) = P_SHA256(secret, label || seed)."""
    return tls12_p_hash(secret, label + seed, out_len)


def stream_xor_encrypt(enc_key: bytes, nonce: bytes, data: bytes) -> bytes:
    """Toy stream cipher (SHA256-based keystream) for MVP record protection."""
    keystream = bytearray()
    counter = 0
    while len(keystream) < len(data):
        block = hashlib.sha256(enc_key + nonce + counter.to_bytes(8, "big")).digest()
        keystream.extend(block)
        counter += 1
    return xor_bytes(data, bytes(keystream[: len(data)]))


def generate_dh_keypair() -> tuple[int, int]:
    private = secrets.randbits(256) + 2
    public = pow(DH_G, private, DH_P)
    return private, public


def validate_peer_public(peer_public: int) -> None:
    if not isinstance(peer_public, int):
        raise TypeError("Peer DH public key must be int.")
    if peer_public <= 1 or peer_public >= DH_P - 1:
        raise ValueError("Peer DH public key out of valid range.")


def dh_shared_secret(local_private: int, peer_public: int) -> int:
    validate_peer_public(peer_public)
    return pow(peer_public, local_private, DH_P)


def derive_master_secret(shared_secret: int, client_random: bytes, server_random: bytes) -> bytes:
    pre_master = shared_secret.to_bytes(DH_BYTES, "big")
    return tls12_prf(pre_master, b"master secret", client_random + server_random, MASTER_SECRET_LEN)


def derive_traffic_keys(master_secret: bytes, client_random: bytes, server_random: bytes) -> tuple[TrafficKeys, TrafficKeys]:
    block_len = 2 * (MAC_KEY_LEN + ENC_KEY_LEN + IV_LEN)
    key_block = tls12_prf(
        master_secret,
        b"key expansion",
        server_random + client_random,
        block_len,
    )
    offset = 0
    c_mac = key_block[offset : offset + MAC_KEY_LEN]
    offset += MAC_KEY_LEN
    s_mac = key_block[offset : offset + MAC_KEY_LEN]
    offset += MAC_KEY_LEN
    c_key = key_block[offset : offset + ENC_KEY_LEN]
    offset += ENC_KEY_LEN
    s_key = key_block[offset : offset + ENC_KEY_LEN]
    offset += ENC_KEY_LEN
    c_iv = key_block[offset : offset + IV_LEN]
    offset += IV_LEN
    s_iv = key_block[offset : offset + IV_LEN]

    return (
        TrafficKeys(mac_key=c_mac, enc_key=c_key, iv=c_iv),
        TrafficKeys(mac_key=s_mac, enc_key=s_key, iv=s_iv),
    )


def encode_int(n: int) -> bytes:
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return len(raw).to_bytes(2, "big") + raw


def build_transcript(client_random: bytes, server_random: bytes, client_pub: int, server_pub: int) -> bytes:
    return b"".join(
        [
            b"ClientHello",
            client_random,
            b"ServerHello",
            server_random,
            b"ClientKeyShare",
            encode_int(client_pub),
            b"ServerKeyShare",
            encode_int(server_pub),
        ]
    )


def finished_verify_data(master_secret: bytes, label: bytes, transcript: bytes) -> bytes:
    transcript_hash = hashlib.sha256(transcript).digest()
    return tls12_prf(master_secret, label, transcript_hash, VERIFY_DATA_LEN)


def protect_record(keys: TrafficKeys, seq: int, plaintext: bytes) -> bytes:
    header = seq.to_bytes(8, "big") + len(plaintext).to_bytes(2, "big")
    tag = hmac_sha256(keys.mac_key, header + plaintext)
    payload = plaintext + tag
    nonce = keys.iv + seq.to_bytes(8, "big")
    return stream_xor_encrypt(keys.enc_key, nonce, payload)


def unprotect_record(keys: TrafficKeys, seq: int, ciphertext: bytes) -> bytes:
    nonce = keys.iv + seq.to_bytes(8, "big")
    payload = stream_xor_encrypt(keys.enc_key, nonce, ciphertext)
    if len(payload) < TAG_LEN:
        raise ValueError("Ciphertext too short.")
    plaintext = payload[:-TAG_LEN]
    recv_tag = payload[-TAG_LEN:]
    header = seq.to_bytes(8, "big") + len(plaintext).to_bytes(2, "big")
    expect_tag = hmac_sha256(keys.mac_key, header + plaintext)
    if not hmac.compare_digest(recv_tag, expect_tag):
        raise ValueError("Record MAC verification failed.")
    return plaintext


def perform_handshake() -> TLSSession:
    client_random = secrets.token_bytes(RANDOM_LEN)
    server_random = secrets.token_bytes(RANDOM_LEN)
    client_priv, client_pub = generate_dh_keypair()
    server_priv, server_pub = generate_dh_keypair()

    client_shared = dh_shared_secret(client_priv, server_pub)
    server_shared = dh_shared_secret(server_priv, client_pub)
    if client_shared != server_shared:
        raise RuntimeError("DH shared secret mismatch.")

    client_master = derive_master_secret(client_shared, client_random, server_random)
    server_master = derive_master_secret(server_shared, client_random, server_random)
    if not hmac.compare_digest(client_master, server_master):
        raise RuntimeError("Master secret mismatch.")

    client_keys_view = derive_traffic_keys(client_master, client_random, server_random)
    server_keys_view = derive_traffic_keys(server_master, client_random, server_random)
    if client_keys_view != server_keys_view:
        raise RuntimeError("Traffic key block mismatch.")
    client_to_server, server_to_client = client_keys_view

    transcript = build_transcript(client_random, server_random, client_pub, server_pub)
    client_finished = finished_verify_data(client_master, b"client finished", transcript)
    server_expect_client_finished = finished_verify_data(server_master, b"client finished", transcript)
    if not hmac.compare_digest(client_finished, server_expect_client_finished):
        raise RuntimeError("Client Finished verify_data mismatch.")

    server_finished = finished_verify_data(server_master, b"server finished", transcript)
    client_expect_server_finished = finished_verify_data(client_master, b"server finished", transcript)
    if not hmac.compare_digest(server_finished, client_expect_server_finished):
        raise RuntimeError("Server Finished verify_data mismatch.")

    return TLSSession(
        client_random=client_random,
        server_random=server_random,
        master_secret=client_master,
        client_to_server=client_to_server,
        server_to_client=server_to_client,
        transcript_hash=hashlib.sha256(transcript).digest(),
    )


def short_hex(data: bytes, keep: int = 12) -> str:
    hx = data.hex()
    if len(hx) <= keep * 2:
        return hx
    return f"{hx[:keep]}...{hx[-keep:]}"


def main() -> None:
    print("=== TLS/SSL Protocol MVP (educational, TLS-1.2-like) ===")

    timings_ms: list[float] = []
    session: TLSSession | None = None
    for _ in range(5):
        t0 = time.perf_counter()
        session = perform_handshake()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    if session is None:
        raise RuntimeError("Handshake did not produce a session.")

    stats = np.array(timings_ms, dtype=np.float64)
    print(f"Handshake runs: {len(timings_ms)}")
    print(f"Handshake time mean/std (ms): {stats.mean():.3f}/{stats.std(ddof=0):.3f}")
    print(f"Master secret prefix: {short_hex(session.master_secret, keep=10)}")
    print(f"Transcript hash: {session.transcript_hash.hex()}")

    client_plain = b"GET /resource HTTP/1.1\r\nHost: internal.local\r\n\r\n"
    c_record = protect_record(session.client_to_server, seq=0, plaintext=client_plain)
    server_seen = unprotect_record(session.client_to_server, seq=0, ciphertext=c_record)
    if server_seen != client_plain:
        raise RuntimeError("Server decrypted payload mismatch.")

    server_plain = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"
    s_record = protect_record(session.server_to_client, seq=0, plaintext=server_plain)
    client_seen = unprotect_record(session.server_to_client, seq=0, ciphertext=s_record)
    if client_seen != server_plain:
        raise RuntimeError("Client decrypted payload mismatch.")

    print(f"Client->Server ciphertext bytes: {len(c_record)}")
    print(f"Server->Client ciphertext bytes: {len(s_record)}")
    print("Application data exchange: PASS")

    tampered = bytearray(c_record)
    tampered[len(tampered) // 2] ^= 0x01
    tamper_rejected = False
    try:
        _ = unprotect_record(session.client_to_server, seq=0, ciphertext=bytes(tampered))
    except ValueError:
        tamper_rejected = True
    print(f"Tamper detection: {tamper_rejected}")
    if not tamper_rejected:
        raise RuntimeError("Tampered record was not rejected.")

    print("All TLS/SSL MVP checks passed.")


if __name__ == "__main__":
    main()
