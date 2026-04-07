"""Minimal runnable MVP for the working set page replacement algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class StepRecord:
    """One access step in the simulation."""

    step: int
    page: int
    hit: bool
    expired: Tuple[int, ...]
    evicted: Optional[int]
    resident: Tuple[int, ...]
    faults: int


class WorkingSetPager:
    """Working set algorithm with an optional hard frame cap.

    - `delta`: working-set window size in number of references.
    - `max_frames`: hard memory cap used in this MVP to keep memory bounded.
    """

    def __init__(self, delta: int, max_frames: Optional[int] = None) -> None:
        if delta <= 0:
            raise ValueError("delta must be positive")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")

        self.delta = delta
        self.max_frames = max_frames
        self.time = 0
        self.resident: Set[int] = set()
        self.last_seen: Dict[int, int] = {}
        self.faults = 0
        self.records: List[StepRecord] = []

    def _expire_stale_pages(self) -> List[int]:
        """Drop pages not referenced within the last `delta` references."""

        threshold = self.time - self.delta
        expired = [p for p in self.resident if self.last_seen.get(p, 0) < threshold]
        for page in expired:
            self.resident.remove(page)
        return sorted(expired)

    def access(self, page: int) -> None:
        """Process one page reference."""

        self.time += 1
        expired = self._expire_stale_pages()

        hit = page in self.resident
        evicted: Optional[int] = None

        if not hit:
            self.faults += 1
            if self.max_frames is not None and len(self.resident) >= self.max_frames:
                # Fallback policy when the working set still does not fit: evict oldest page.
                evicted = min(self.resident, key=lambda p: self.last_seen[p])
                self.resident.remove(evicted)
            self.resident.add(page)

        self.last_seen[page] = self.time

        self.records.append(
            StepRecord(
                step=self.time,
                page=page,
                hit=hit,
                expired=tuple(expired),
                evicted=evicted,
                resident=tuple(sorted(self.resident)),
                faults=self.faults,
            )
        )


def simulate_working_set(
    reference_string: Sequence[int], delta: int, max_frames: Optional[int]
) -> WorkingSetPager:
    pager = WorkingSetPager(delta=delta, max_frames=max_frames)
    for page in reference_string:
        pager.access(page)
    return pager


def simulate_lru_faults(reference_string: Sequence[int], capacity: int) -> int:
    if capacity <= 0:
        raise ValueError("capacity must be positive")

    time = 0
    faults = 0
    resident: Set[int] = set()
    last_seen: Dict[int, int] = {}

    for page in reference_string:
        time += 1
        if page in resident:
            last_seen[page] = time
            continue

        faults += 1
        if len(resident) >= capacity:
            victim = min(resident, key=lambda p: last_seen[p])
            resident.remove(victim)

        resident.add(page)
        last_seen[page] = time

    return faults


def format_records(records: Sequence[StepRecord]) -> str:
    header = (
        f"{'step':>4} {'page':>4} {'hit':>4} {'expired':>12} "
        f"{'evicted':>7} {'resident':>20} {'faults':>6}"
    )
    lines = [header, "-" * len(header)]

    for r in records:
        lines.append(
            f"{r.step:>4} "
            f"{r.page:>4} "
            f"{('Y' if r.hit else 'N'):>4} "
            f"{str(list(r.expired)):>12} "
            f"{str(r.evicted) if r.evicted is not None else '-':>7} "
            f"{str(list(r.resident)):>20} "
            f"{r.faults:>6}"
        )

    return "\n".join(lines)


def main() -> None:
    # Deterministic demo reference string (no interactive input required).
    reference_string = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    delta = 4
    max_frames = 4

    pager = simulate_working_set(
        reference_string=reference_string,
        delta=delta,
        max_frames=max_frames,
    )
    lru_faults = simulate_lru_faults(reference_string, capacity=max_frames)

    print("Working Set Algorithm MVP")
    print(f"reference_string = {reference_string}")
    print(f"delta = {delta}, max_frames = {max_frames}")
    print()
    print(format_records(pager.records))
    print()
    print(f"Working Set faults: {pager.faults}")
    print(f"LRU faults (same frame cap): {lru_faults}")


if __name__ == "__main__":
    main()
