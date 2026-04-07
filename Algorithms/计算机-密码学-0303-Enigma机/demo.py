"""Enigma machine MVP demo.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A2I: Dict[str, int] = {ch: i for i, ch in enumerate(ALPHABET)}


@dataclass
class Rotor:
    """Single Enigma rotor with wiring, notch, position, and ring setting."""

    wiring: str
    notch: str
    position: int = 0
    ring_setting: int = 0

    def __post_init__(self) -> None:
        self.forward_map = [A2I[ch] for ch in self.wiring]
        self.backward_map = [0] * 26
        for i, v in enumerate(self.forward_map):
            self.backward_map[v] = i

    def step(self) -> None:
        self.position = (self.position + 1) % 26

    def at_notch(self) -> bool:
        return ALPHABET[self.position] == self.notch

    def encode_forward(self, idx: int) -> int:
        shifted = (idx + self.position - self.ring_setting) % 26
        wired = self.forward_map[shifted]
        return (wired - self.position + self.ring_setting) % 26

    def encode_backward(self, idx: int) -> int:
        shifted = (idx + self.position - self.ring_setting) % 26
        wired = self.backward_map[shifted]
        return (wired - self.position + self.ring_setting) % 26


class Plugboard:
    """Bidirectional swaps, e.g. pairs=['AV', 'BS']."""

    def __init__(self, pairs: Iterable[str]) -> None:
        mapping = {c: c for c in ALPHABET}
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"Invalid plugboard pair: {pair}")
            a, b = pair[0].upper(), pair[1].upper()
            if a == b:
                raise ValueError(f"Plugboard pair cannot map same letter: {pair}")
            if mapping[a] != a or mapping[b] != b:
                raise ValueError(f"Plugboard collision in pair: {pair}")
            mapping[a], mapping[b] = b, a
        self.mapping = mapping

    def swap(self, ch: str) -> str:
        return self.mapping[ch]


class EnigmaMachine:
    """Three-rotor Enigma I style machine."""

    def __init__(self, rotors: List[Rotor], reflector: str, plugboard: Plugboard) -> None:
        if len(rotors) != 3:
            raise ValueError("MVP expects exactly 3 rotors: [left, middle, right].")
        self.rotors = rotors
        self.reflector = [A2I[ch] for ch in reflector]
        self.plugboard = plugboard

    def set_positions(self, positions: str) -> None:
        if len(positions) != 3:
            raise ValueError("positions must be a 3-letter string")
        for rotor, ch in zip(self.rotors, positions.upper()):
            rotor.position = A2I[ch]

    def _step_rotors(self) -> None:
        left, middle, right = self.rotors
        step_left = middle.at_notch()
        step_middle = right.at_notch() or middle.at_notch()

        if step_left:
            left.step()
        if step_middle:
            middle.step()
        right.step()

    def _encode_letter(self, ch: str) -> str:
        self._step_rotors()

        c = self.plugboard.swap(ch)
        idx = A2I[c]

        # Right -> middle -> left
        for rotor in reversed(self.rotors):
            idx = rotor.encode_forward(idx)

        idx = self.reflector[idx]

        # Left -> middle -> right
        for rotor in self.rotors:
            idx = rotor.encode_backward(idx)

        out = ALPHABET[idx]
        return self.plugboard.swap(out)

    def process(self, text: str) -> str:
        out_chars = []
        for ch in text:
            up = ch.upper()
            if up in A2I:
                out_chars.append(self._encode_letter(up))
            else:
                out_chars.append(ch)
        return "".join(out_chars)


def letter_frequency(text: str) -> np.ndarray:
    """Return normalized 26-dim frequency vector for A-Z."""
    letters = [ch for ch in text if ch in A2I]
    if not letters:
        return np.zeros(26, dtype=float)
    arr = np.array([A2I[ch] for ch in letters], dtype=int)
    counts = np.bincount(arr, minlength=26).astype(float)
    return counts / counts.sum()


def build_default_machine() -> EnigmaMachine:
    # Historical rotor wirings for Enigma I.
    rotor_specs = {
        "I": ("EKMFLGDQVZNTOWYHXUSPAIBRCJ", "Q"),
        "II": ("AJDKSIRUXBLHWTMCQGZNPYFVOE", "E"),
        "III": ("BDFHJLCPRTXVZNYEIWGAKMUSQO", "V"),
    }
    reflector_b = "YRUHQSLDPXNGOKMIEBFZCWVJAT"

    left = Rotor(*rotor_specs["I"], ring_setting=0)
    middle = Rotor(*rotor_specs["II"], ring_setting=0)
    right = Rotor(*rotor_specs["III"], ring_setting=0)

    plugboard = Plugboard(["PO", "ML", "IU", "KJ", "NH", "YT", "GB", "VF", "RE", "DC"])
    return EnigmaMachine(rotors=[left, middle, right], reflector=reflector_b, plugboard=plugboard)


def main() -> None:
    machine = build_default_machine()
    start_pos = "MCK"

    plaintext = "ENIGMA MACHINE DEMO, 1939: ATTACK AT DAWN!"

    machine.set_positions(start_pos)
    ciphertext = machine.process(plaintext)

    machine.set_positions(start_pos)
    decrypted = machine.process(ciphertext)

    if decrypted != plaintext.upper():
        raise RuntimeError("Encrypt/decrypt symmetry check failed.")

    freq = letter_frequency(ciphertext)
    top_idx = np.argsort(freq)[-5:][::-1]
    top5 = [(ALPHABET[i], float(freq[i])) for i in top_idx]

    print("=== Enigma MVP ===")
    print(f"Start positions: {start_pos}")
    print(f"Plaintext : {plaintext.upper()}")
    print(f"Ciphertext: {ciphertext}")
    print(f"Decrypted : {decrypted}")
    print("Top-5 ciphertext letter frequencies:")
    for ch, p in top5:
        print(f"  {ch}: {p:.4f}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
