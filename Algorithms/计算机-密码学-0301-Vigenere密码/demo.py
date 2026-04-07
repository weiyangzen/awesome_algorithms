"""Minimal runnable MVP for Vigenere cipher (CS-0156)."""

from __future__ import annotations

from collections import Counter


ALPHABET_SIZE = 26


def _key_to_shifts(key: str) -> list[int]:
    """Convert key letters to 0-25 shifts; ignore non-letters."""
    shifts = [ord(ch.upper()) - ord("A") for ch in key if ch.isalpha()]
    if not shifts:
        raise ValueError("key must contain at least one alphabetic character")
    return shifts


def _shift_char(ch: str, shift: int) -> str:
    """Shift one alphabetic character by shift positions, preserving case."""
    if not ch.isalpha():
        return ch
    base = ord("A") if ch.isupper() else ord("a")
    offset = ord(ch) - base
    return chr(base + (offset + shift) % ALPHABET_SIZE)


def vigenere_encrypt(plaintext: str, key: str) -> str:
    """Encrypt plaintext using Vigenere cipher."""
    shifts = _key_to_shifts(key)
    out: list[str] = []
    key_idx = 0

    for ch in plaintext:
        if ch.isalpha():
            shift = shifts[key_idx % len(shifts)]
            out.append(_shift_char(ch, shift))
            key_idx += 1
        else:
            out.append(ch)

    return "".join(out)


def vigenere_decrypt(ciphertext: str, key: str) -> str:
    """Decrypt ciphertext using Vigenere cipher."""
    shifts = _key_to_shifts(key)
    out: list[str] = []
    key_idx = 0

    for ch in ciphertext:
        if ch.isalpha():
            shift = shifts[key_idx % len(shifts)]
            out.append(_shift_char(ch, -shift))
            key_idx += 1
        else:
            out.append(ch)

    return "".join(out)


def coincidence_index(text: str) -> float:
    """Compute index of coincidence over alphabetic letters only."""
    letters = [ch.upper() for ch in text if ch.isalpha()]
    n = len(letters)
    if n < 2:
        return 0.0

    counts = Counter(letters)
    numerator = sum(c * (c - 1) for c in counts.values())
    denominator = n * (n - 1)
    return numerator / denominator


def main() -> None:
    # Textbook sanity check.
    textbook_plain = "ATTACKATDAWN"
    textbook_key = "LEMON"
    textbook_cipher = vigenere_encrypt(textbook_plain, textbook_key)
    assert textbook_cipher == "LXFOPVEFRNHR"
    assert vigenere_decrypt(textbook_cipher, textbook_key) == textbook_plain

    message = "Vigenere cipher demo for CS-0156. Keep this text secret; it is classic, not modern-safe."
    key = "BlueRiver"

    cipher = vigenere_encrypt(message, key)
    recovered = vigenere_decrypt(cipher, key)

    # Deterministic cipher: same plaintext + same key => same ciphertext.
    cipher_again = vigenere_encrypt(message, key)

    wrong_key_recovered = vigenere_decrypt(cipher, "BlueRover")

    ok_roundtrip = recovered == message
    ok_deterministic = cipher_again == cipher
    ok_wrong_key = wrong_key_recovered != message

    ic_plain = coincidence_index(message)
    ic_cipher = coincidence_index(cipher)

    print("=== Vigenere Cipher MVP (CS-0156) ===")
    print(f"key: {key}")
    print(f"plaintext: {message}")
    print(f"ciphertext: {cipher}")
    print(f"decrypted: {recovered}")
    print(f"roundtrip ok: {ok_roundtrip}")
    print(f"deterministic same-key check: {ok_deterministic}")
    print(f"wrong-key recovery equals original: {not ok_wrong_key}")
    print(f"IC(plaintext): {ic_plain:.4f}")
    print(f"IC(ciphertext): {ic_cipher:.4f}")

    if not (ok_roundtrip and ok_deterministic and ok_wrong_key):
        raise RuntimeError("Vigenere MVP checks failed")


if __name__ == "__main__":
    main()
