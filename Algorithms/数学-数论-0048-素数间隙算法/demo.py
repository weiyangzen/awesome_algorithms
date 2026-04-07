"""Prime gap algorithm MVP.

This script computes prime numbers up to a fixed bound using the Sieve of
Eratosthenes, derives adjacent prime gaps, and prints a compact report.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from statistics import mean
from typing import List, Sequence


@dataclass(frozen=True)
class GapRecord:
    """A single adjacent-prime gap record."""

    left_prime: int
    right_prime: int
    gap: int
    normalized_by_log: float


def sieve_primes(limit: int) -> List[int]:
    """Return all primes <= limit using a compact bytearray sieve."""
    if limit < 2:
        return []

    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0:2] = b"\x00\x00"

    p = 2
    while p * p <= limit:
        if is_prime[p]:
            start = p * p
            step = p
            is_prime[start : limit + 1 : step] = b"\x00" * (((limit - start) // step) + 1)
        p += 1

    return [i for i in range(2, limit + 1) if is_prime[i]]


def build_gap_records(primes: Sequence[int]) -> List[GapRecord]:
    """Build adjacent-prime gap records from an ordered prime sequence."""
    records: List[GapRecord] = []
    for i in range(1, len(primes)):
        left = primes[i - 1]
        right = primes[i]
        gap = right - left
        denom = log(left) if left > 1 else 1.0
        records.append(
            GapRecord(
                left_prime=left,
                right_prime=right,
                gap=gap,
                normalized_by_log=gap / denom,
            )
        )
    return records


def summarize(records: Sequence[GapRecord]) -> dict:
    """Produce summary statistics for the gap list."""
    if not records:
        return {
            "count": 0,
            "max_gap": 0,
            "avg_gap": 0.0,
            "avg_norm": 0.0,
            "record": None,
            "top5": [],
        }

    max_record = max(records, key=lambda r: r.gap)
    top5 = sorted(records, key=lambda r: r.gap, reverse=True)[:5]
    return {
        "count": len(records),
        "max_gap": max_record.gap,
        "avg_gap": mean(r.gap for r in records),
        "avg_norm": mean(r.normalized_by_log for r in records),
        "record": max_record,
        "top5": top5,
    }


def main() -> None:
    """Run a fixed, non-interactive demo for prime gap analysis."""
    limit = 200_000
    primes = sieve_primes(limit)
    records = build_gap_records(primes)
    report = summarize(records)

    print("Prime Gap Algorithm MVP")
    print(f"Upper bound: {limit}")
    print(f"Prime count: {len(primes)}")
    print(f"Gap count: {report['count']}")
    print(f"Average gap: {report['avg_gap']:.4f}")
    print(f"Average normalized gap (gap / ln(p)): {report['avg_norm']:.4f}")

    best = report["record"]
    if best is not None:
        print(
            "Largest observed gap: "
            f"{best.gap} between {best.left_prime} and {best.right_prime} "
            f"(gap/ln(p)={best.normalized_by_log:.4f})"
        )

    print("Top 5 gaps in scanned range:")
    for idx, item in enumerate(report["top5"], start=1):
        print(
            f"  {idx}. gap={item.gap:>2} | "
            f"({item.left_prime}, {item.right_prime}) | "
            f"gap/ln(p)={item.normalized_by_log:.4f}"
        )


if __name__ == "__main__":
    main()
