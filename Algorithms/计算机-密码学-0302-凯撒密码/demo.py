"""Caesar cipher MVP: encrypt/decrypt + brute-force cracking demo."""

from __future__ import annotations

from collections import Counter
from math import inf
from typing import Dict, List, Tuple

ENGLISH_FREQ: Dict[str, float] = {
    "A": 8.167,
    "B": 1.492,
    "C": 2.782,
    "D": 4.253,
    "E": 12.702,
    "F": 2.228,
    "G": 2.015,
    "H": 6.094,
    "I": 6.966,
    "J": 0.153,
    "K": 0.772,
    "L": 4.025,
    "M": 2.406,
    "N": 6.749,
    "O": 7.507,
    "P": 1.929,
    "Q": 0.095,
    "R": 5.987,
    "S": 6.327,
    "T": 9.056,
    "U": 2.758,
    "V": 0.978,
    "W": 2.360,
    "X": 0.150,
    "Y": 1.974,
    "Z": 0.074,
}


def normalize_shift(shift: int) -> int:
    """Map arbitrary integer shift into [0, 25]."""
    return shift % 26


def shift_char(ch: str, shift: int) -> str:
    """Shift one alphabetic character, keep non-letters unchanged."""
    k = normalize_shift(shift)

    if "A" <= ch <= "Z":
        return chr((ord(ch) - ord("A") + k) % 26 + ord("A"))
    if "a" <= ch <= "z":
        return chr((ord(ch) - ord("a") + k) % 26 + ord("a"))
    return ch


def caesar_encrypt(text: str, shift: int) -> str:
    """Encrypt text with Caesar cipher."""
    return "".join(shift_char(ch, shift) for ch in text)


def caesar_decrypt(cipher_text: str, shift: int) -> str:
    """Decrypt text with Caesar cipher."""
    return caesar_encrypt(cipher_text, -shift)


def chi_square_english_score(text: str) -> float:
    """Lower score means text is closer to typical English letter distribution."""
    letters = [ch for ch in text.upper() if "A" <= ch <= "Z"]
    n = len(letters)
    if n == 0:
        return inf

    counts = Counter(letters)
    score = 0.0
    for letter, freq in ENGLISH_FREQ.items():
        observed = counts.get(letter, 0)
        expected = n * (freq / 100.0)
        if expected > 0:
            score += (observed - expected) ** 2 / expected
    return score


def brute_force_caesar(cipher_text: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
    """Try all 26 shifts, return best candidates sorted by chi-square score."""
    candidates: List[Tuple[int, float, str]] = []
    for shift in range(26):
        plain = caesar_decrypt(cipher_text, shift)
        score = chi_square_english_score(plain)
        candidates.append((shift, score, plain))

    candidates.sort(key=lambda item: item[1])
    return candidates[:top_k]


def main() -> None:
    print("=== Caesar Cipher MVP ===")

    plaintext = "Attack at dawn! Meet at 09:30 near Gate-B."
    shift = 7

    ciphertext = caesar_encrypt(plaintext, shift)
    recovered = caesar_decrypt(ciphertext, shift)

    print("\n[Known-key roundtrip]")
    print(f"Plaintext : {plaintext}")
    print(f"Shift key : {shift}")
    print(f"Ciphertext: {ciphertext}")
    print(f"Recovered : {recovered}")
    print(f"Roundtrip OK: {recovered == plaintext}")

    unknown_plain = "Defend the east wall at sunrise and hold position"
    unknown_shift = 11
    unknown_cipher = caesar_encrypt(unknown_plain, unknown_shift)

    print("\n[Unknown-key cracking]")
    print(f"Ciphertext: {unknown_cipher}")
    print("Top brute-force candidates (shift, score, plaintext):")

    best = brute_force_caesar(unknown_cipher, top_k=5)
    for s, score, candidate in best:
        print(f"  shift={s:2d}, score={score:8.3f}, text={candidate}")

    if best:
        print("\nBest guess:")
        print(f"  shift={best[0][0]}, plaintext={best[0][2]}")


if __name__ == "__main__":
    main()
