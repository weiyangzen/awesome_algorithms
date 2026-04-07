"""Minimal runnable MVP for Huffman Coding (CS-0063).

This demo builds a Huffman tree from character frequencies, then
encodes and decodes text to verify correctness and report compression
statistics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import math
from typing import Dict, Optional

import numpy as np


@dataclass
class HuffmanNode:
    """Node in a Huffman tree."""

    freq: int
    symbol: Optional[str] = None
    left: Optional["HuffmanNode"] = None
    right: Optional["HuffmanNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def build_frequency_table(text: str) -> Dict[str, int]:
    """Count symbol frequencies in input text."""
    if not text:
        raise ValueError("text must not be empty")
    return dict(Counter(text))


def build_huffman_tree(freq_table: Dict[str, int]) -> HuffmanNode:
    """Greedily build a Huffman tree with a min-heap."""
    if not freq_table:
        raise ValueError("frequency table must not be empty")

    heap = []
    order = 0
    for symbol, freq in sorted(freq_table.items(), key=lambda kv: (kv[1], kv[0])):
        heapq.heappush(heap, (freq, order, HuffmanNode(freq=freq, symbol=symbol)))
        order += 1

    while len(heap) > 1:
        freq1, _, node1 = heapq.heappop(heap)
        freq2, _, node2 = heapq.heappop(heap)
        merged = HuffmanNode(freq=freq1 + freq2, left=node1, right=node2)
        heapq.heappush(heap, (merged.freq, order, merged))
        order += 1

    return heap[0][2]


def build_codebook(root: HuffmanNode) -> Dict[str, str]:
    """Traverse tree and produce prefix-free bit codes."""
    codebook: Dict[str, str] = {}

    def dfs(node: HuffmanNode, prefix: str) -> None:
        if node.is_leaf:
            if node.symbol is None:
                raise RuntimeError("leaf node without symbol")
            # Single-symbol text gets one valid bit code.
            codebook[node.symbol] = prefix or "0"
            return
        if node.left is None or node.right is None:
            raise RuntimeError("invalid internal node")
        dfs(node.left, prefix + "0")
        dfs(node.right, prefix + "1")

    dfs(root, "")
    return codebook


def encode_text(text: str, codebook: Dict[str, str]) -> str:
    """Encode text as a bit-string represented by '0'/'1' characters."""
    return "".join(codebook[ch] for ch in text)


def decode_bits(bits: str, root: HuffmanNode) -> str:
    """Decode bit-string back to original text."""
    if root.is_leaf:
        if root.symbol is None:
            raise RuntimeError("invalid single-node tree")
        return root.symbol * len(bits)

    out_chars = []
    node = root
    for bit in bits:
        if bit == "0":
            if node.left is None:
                raise RuntimeError("invalid path on bit 0")
            node = node.left
        elif bit == "1":
            if node.right is None:
                raise RuntimeError("invalid path on bit 1")
            node = node.right
        else:
            raise ValueError(f"invalid bit: {bit!r}")

        if node.is_leaf:
            if node.symbol is None:
                raise RuntimeError("leaf node without symbol")
            out_chars.append(node.symbol)
            node = root

    if node is not root:
        raise RuntimeError("incomplete bit-string ending at internal node")
    return "".join(out_chars)


def assert_prefix_free(codebook: Dict[str, str]) -> None:
    """Check that no code is the prefix of another code."""
    codes = list(codebook.values())
    for i, code_i in enumerate(codes):
        for j, code_j in enumerate(codes):
            if i == j:
                continue
            if code_j.startswith(code_i):
                raise RuntimeError("codebook is not prefix-free")


def expected_bits_per_symbol(freq_table: Dict[str, int], codebook: Dict[str, str]) -> float:
    """Compute expected code length E[L] based on empirical frequencies."""
    symbols = list(freq_table.keys())
    freqs = np.array([freq_table[s] for s in symbols], dtype=np.float64)
    probs = freqs / np.sum(freqs)
    lengths = np.array([len(codebook[s]) for s in symbols], dtype=np.float64)
    return float(np.sum(probs * lengths))


def shannon_entropy_bits(freq_table: Dict[str, int]) -> float:
    """Compute Shannon entropy H(X) from empirical distribution."""
    freqs = np.array(list(freq_table.values()), dtype=np.float64)
    probs = freqs / np.sum(freqs)
    return float(-np.sum(probs * np.log2(probs)))


def main() -> None:
    print("Huffman Coding MVP (CS-0063)")
    print("=" * 72)

    text = (
        "huffman coding is a greedy algorithm. "
        "it builds an optimal prefix code for known symbol frequencies."
    )

    freq_table = build_frequency_table(text)
    root = build_huffman_tree(freq_table)
    codebook = build_codebook(root)
    assert_prefix_free(codebook)

    encoded_bits = encode_text(text, codebook)
    decoded_text = decode_bits(encoded_bits, root)
    if decoded_text != text:
        raise RuntimeError("round-trip decode mismatch")

    n_symbols = len(freq_table)
    fixed_bits_per_symbol = max(1, math.ceil(math.log2(n_symbols)))
    fixed_bits_total = fixed_bits_per_symbol * len(text)
    huffman_bits_total = len(encoded_bits)

    avg_len = expected_bits_per_symbol(freq_table, codebook)
    entropy = shannon_entropy_bits(freq_table)

    print(f"input text length (chars): {len(text)}")
    print(f"unique symbols: {n_symbols}")
    print(f"fixed-length bits/symbol: {fixed_bits_per_symbol}")
    print(f"fixed-length total bits: {fixed_bits_total}")
    print(f"huffman total bits: {huffman_bits_total}")
    print(f"compression ratio vs fixed: {huffman_bits_total / fixed_bits_total:.4f}")
    print(f"empirical entropy H(X): {entropy:.4f} bits/symbol")
    print(f"expected Huffman length E[L]: {avg_len:.4f} bits/symbol")
    print("-" * 72)

    top_items = sorted(freq_table.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    print("Top symbol -> (freq, code):")
    for symbol, freq in top_items:
        shown = repr(symbol)
        print(f"  {shown:>6} -> freq={freq:>3}, code={codebook[symbol]}")

    if not (huffman_bits_total < fixed_bits_total):
        raise RuntimeError("Huffman coding failed to beat fixed-length code on this sample")
    if not (entropy <= avg_len < entropy + 1.0 + 1e-12):
        raise RuntimeError("Huffman average length should satisfy H <= E[L] < H+1")

    print("=" * 72)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
