"""Educational rainbow-table attack MVP.

This script demonstrates how rainbow tables can crack unsalted password hashes.
It intentionally uses a tiny password space so the full workflow is observable:
1) build chains with round-dependent reduction functions,
2) store only chain start/end points,
3) recover plaintext candidates from target hashes.

No interactive input is required.
"""

from __future__ import annotations

import hashlib
import random
import string
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RainbowConfig:
    """Configuration for the toy rainbow-table system."""

    alphabet: str = "abcdef0123456789"
    password_length: int = 4
    chain_length: int = 40
    chain_count: int = 1600


def sha1_hex(text: str) -> str:
    """Return lowercase SHA-1 hex digest for a plaintext password."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def int_to_password(value: int, alphabet: str, length: int) -> str:
    """Map an integer into a fixed-length password over `alphabet`."""
    base = len(alphabet)
    chars: List[str] = []
    x = value
    for _ in range(length):
        x, r = divmod(x, base)
        chars.append(alphabet[r])
    return "".join(chars)


def reduce_hash(hash_hex: str, round_index: int, cfg: RainbowConfig) -> str:
    """Round-dependent reduction: hash -> plausible plaintext.

    A rainbow table needs a *family* of reduction functions R_i; here we derive
    that family by mixing round index into the big integer represented by hash.
    """
    hash_value = int(hash_hex, 16)
    salt_like_offset = (round_index + 1) * 0x9E3779B1
    mixed = hash_value ^ salt_like_offset

    # Map into the finite password space.
    space_size = len(cfg.alphabet) ** cfg.password_length
    return int_to_password(mixed % space_size, cfg.alphabet, cfg.password_length)


def chain_end(start_plain: str, cfg: RainbowConfig) -> str:
    """Run one chain and return the final plaintext endpoint."""
    plain = start_plain
    for i in range(cfg.chain_length):
        digest = sha1_hex(plain)
        plain = reduce_hash(digest, i, cfg)
    return plain


def build_rainbow_table(cfg: RainbowConfig, seed: int = 20260407) -> Dict[str, List[str]]:
    """Build an endpoint->start list mapping.

    We keep a list because multiple chains may collide at the same endpoint.
    """
    random.seed(seed)
    space_size = len(cfg.alphabet) ** cfg.password_length

    table: Dict[str, List[str]] = {}
    for _ in range(cfg.chain_count):
        start = int_to_password(random.randrange(space_size), cfg.alphabet, cfg.password_length)
        end = chain_end(start, cfg)
        table.setdefault(end, []).append(start)

    return table


def regenerate_chain_find_target(start_plain: str, target_hash: str, cfg: RainbowConfig) -> Optional[str]:
    """Regenerate one chain and check whether target hash appears in it."""
    plain = start_plain
    for i in range(cfg.chain_length):
        digest = sha1_hex(plain)
        if digest == target_hash:
            return plain
        plain = reduce_hash(digest, i, cfg)
    return None


def crack_hash(target_hash: str, table: Dict[str, List[str]], cfg: RainbowConfig) -> Optional[str]:
    """Attempt to crack one hash with the rainbow table."""
    # Try every possible alignment of target hash inside a chain.
    for pos in range(cfg.chain_length - 1, -1, -1):
        h = target_hash
        candidate_endpoint = ""

        # Complete the chain tail from position `pos` to chain end.
        for j in range(pos, cfg.chain_length):
            candidate_plain = reduce_hash(h, j, cfg)
            candidate_endpoint = candidate_plain
            h = sha1_hex(candidate_plain)

        if candidate_endpoint not in table:
            continue

        # Possible hit: regenerate each colliding chain from stored start point.
        for start_plain in table[candidate_endpoint]:
            cracked = regenerate_chain_find_target(start_plain, target_hash, cfg)
            if cracked is not None:
                return cracked

    return None


def password_space_size(cfg: RainbowConfig) -> int:
    return len(cfg.alphabet) ** cfg.password_length


def sample_passwords(cfg: RainbowConfig) -> Sequence[str]:
    """Deterministic samples for demonstration (both easy and random-like)."""
    return (
        "a1b2",
        "dead",
        "beef",
        "0f0f",
        "c0de",
        "face",
        "1234",
        "9a9a",
    )


def evaluate_attack(cfg: RainbowConfig, table: Dict[str, List[str]]) -> Tuple[int, int, List[Tuple[str, str, Optional[str]]]]:
    """Run attack on fixed samples and return (successes, total, details)."""
    details: List[Tuple[str, str, Optional[str]]] = []

    for plain in sample_passwords(cfg):
        target_hash = sha1_hex(plain)
        cracked = crack_hash(target_hash, table, cfg)
        details.append((plain, target_hash, cracked))

    successes = sum(1 for plain, _, cracked in details if cracked == plain)
    return successes, len(details), details


def main() -> None:
    cfg = RainbowConfig()

    print("=== Rainbow Table Attack MVP ===")
    print(f"alphabet size: {len(cfg.alphabet)}")
    print(f"password length: {cfg.password_length}")
    print(f"password space size: {password_space_size(cfg)}")
    print(f"chain length (t): {cfg.chain_length}")
    print(f"chain count (m): {cfg.chain_count}")

    t0 = time.perf_counter()
    table = build_rainbow_table(cfg)
    t1 = time.perf_counter()

    endpoint_count = len(table)
    total_stored_starts = sum(len(v) for v in table.values())
    collision_count = total_stored_starts - endpoint_count

    print("\n[Build]")
    print(f"table build time: {(t1 - t0):.4f}s")
    print(f"distinct endpoints: {endpoint_count}")
    print(f"endpoint collisions: {collision_count}")

    successes, total, details = evaluate_attack(cfg, table)

    print("\n[Attack Samples]")
    for plain, target_hash, cracked in details:
        ok = cracked == plain
        print(
            f"plain={plain:>4} hash={target_hash[:12]}... "
            f"cracked={str(cracked):>4} success={ok}"
        )

    success_rate = successes / total if total else 0.0
    print("\n[Summary]")
    print(f"successes: {successes}/{total}")
    print(f"success rate: {success_rate:.2%}")
    print("note: misses are expected because rainbow tables trade coverage for space/time.")


if __name__ == "__main__":
    main()
