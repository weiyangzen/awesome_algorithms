"""Educational One-Time Pad (OTP) MVP.

This script demonstrates:
1) correct OTP encrypt/decrypt,
2) wrong-key decryption failure,
3) why key reuse breaks OTP security.
"""

from __future__ import annotations

import secrets


def xor_bytes(left: bytes, right: bytes) -> bytes:
    """Return byte-wise XOR for two equal-length byte strings."""
    if len(left) != len(right):
        raise ValueError("length mismatch: OTP requires equal-length operands")
    return bytes(a ^ b for a, b in zip(left, right))


def generate_key(length: int) -> bytes:
    """Generate a random one-time key of the requested byte length."""
    if length < 0:
        raise ValueError("length must be non-negative")
    return secrets.token_bytes(length)


def otp_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt plaintext with OTP (C = P XOR K)."""
    return xor_bytes(plaintext, key)


def otp_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """Decrypt ciphertext with OTP (P = C XOR K)."""
    return xor_bytes(ciphertext, key)


def to_hex(data: bytes, max_chars: int = 48) -> str:
    """Pretty-print bytes as truncated hexadecimal string."""
    text = data.hex()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def main() -> None:
    message = b"OTP demo for CS-0155: perfect secrecy needs one-time random key."

    key = generate_key(len(message))
    ciphertext = otp_encrypt(message, key)
    recovered = otp_decrypt(ciphertext, key)

    # Create a guaranteed-different key by flipping one bit.
    wrong_key = bytes([key[0] ^ 0x01]) + key[1:] if key else key
    wrong_recovered = otp_decrypt(ciphertext, wrong_key)

    # Key-reuse counterexample: C1 XOR C2 leaks P1 XOR P2.
    p1 = b"attack at dawn!"
    p2 = b"defend at dusk!"
    reused_key = generate_key(len(p1))
    c1 = otp_encrypt(p1, reused_key)
    c2 = otp_encrypt(p2, reused_key)

    xor_cipher = xor_bytes(c1, c2)
    xor_plain = xor_bytes(p1, p2)

    # If attacker knows p1 plus (c1, c2), they can recover p2 exactly.
    recovered_p2 = xor_bytes(xor_bytes(c1, c2), p1)

    ok_recover = recovered == message
    ok_wrong_key = wrong_recovered != message
    ok_reuse_leak = xor_cipher == xor_plain
    ok_known_plaintext_attack = recovered_p2 == p2

    print("=== One-Time Pad (OTP) MVP ===")
    print(f"message: {message.decode('utf-8')}")
    print(f"key(hex): {to_hex(key)}")
    print(f"ciphertext(hex): {to_hex(ciphertext)}")
    print(f"decrypt with correct key: {ok_recover}")
    print(f"decrypt with wrong key equals original: {not ok_wrong_key}")
    print()
    print("=== Key Reuse Risk Demo ===")
    print(f"p1: {p1.decode('utf-8')}")
    print(f"p2: {p2.decode('utf-8')}")
    print(f"c1(hex): {to_hex(c1)}")
    print(f"c2(hex): {to_hex(c2)}")
    print(f"C1 XOR C2 == P1 XOR P2: {ok_reuse_leak}")
    print(f"recover p2 from (p1, c1, c2): {ok_known_plaintext_attack}")

    if not (ok_recover and ok_wrong_key and ok_reuse_leak and ok_known_plaintext_attack):
        raise RuntimeError("OTP demo checks failed")


if __name__ == "__main__":
    main()
