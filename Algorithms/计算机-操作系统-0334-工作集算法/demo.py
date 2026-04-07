"""Minimal runnable MVP for the Working Set page replacement algorithm."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set

import numpy as np


@dataclass
class StepRecord:
    """One access step snapshot for inspection."""

    t: int
    page: int
    hit: bool
    evicted: Optional[int]
    resident_old_to_new: List[int]
    ws_before: List[int]
    ws_after: List[int]


class WorkingSetPager:
    """
    Working Set based pager.

    - window_size (Delta): references in the recent window define working set.
    - frame_limit: maximum number of resident pages.
    """

    def __init__(self, frame_limit: int, window_size: int) -> None:
        if frame_limit <= 0:
            raise ValueError("frame_limit must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self.frame_limit = frame_limit
        self.window_size = window_size

        self.t = 0
        self.hits = 0
        self.faults = 0
        self.evictions = 0

        self.resident: Set[int] = set()
        self.last_access: Dict[int, int] = {}
        self._history: Deque[int] = deque()
        self._history_counter: Counter[int] = Counter()

    def _working_set(self) -> Set[int]:
        return set(self._history_counter.keys())

    def _push_history(self, page: int) -> None:
        if len(self._history) >= self.window_size:
            old_page = self._history.popleft()
            self._history_counter[old_page] -= 1
            if self._history_counter[old_page] == 0:
                del self._history_counter[old_page]
        self._history.append(page)
        self._history_counter[page] += 1

    def _choose_victim(self, ws_before: Set[int]) -> int:
        stale_pages = [p for p in self.resident if p not in ws_before]
        candidate_pages = stale_pages if stale_pages else list(self.resident)
        return min(candidate_pages, key=lambda p: self.last_access[p])

    def access(self, page: int) -> StepRecord:
        self.t += 1
        ws_before = self._working_set()

        hit = page in self.resident
        evicted: Optional[int] = None

        if hit:
            self.hits += 1
        else:
            self.faults += 1
            if len(self.resident) >= self.frame_limit:
                victim = self._choose_victim(ws_before)
                self.resident.remove(victim)
                evicted = victim
                self.evictions += 1
            self.resident.add(page)

        self.last_access[page] = self.t
        self._push_history(page)
        ws_after = self._working_set()

        resident_old_to_new = sorted(self.resident, key=lambda p: self.last_access[p])

        return StepRecord(
            t=self.t,
            page=page,
            hit=hit,
            evicted=evicted,
            resident_old_to_new=resident_old_to_new,
            ws_before=sorted(ws_before),
            ws_after=sorted(ws_after),
        )

    def simulate(self, reference_string: Iterable[int]) -> dict:
        steps: List[StepRecord] = []
        for page in reference_string:
            steps.append(self.access(page))

        total = self.hits + self.faults
        fault_rate = (self.faults / total) if total else 0.0
        hit_rate = (self.hits / total) if total else 0.0

        ws_sizes = np.array([len(s.ws_after) for s in steps], dtype=np.float64)
        avg_ws_size = float(np.mean(ws_sizes)) if ws_sizes.size else 0.0
        max_ws_size = int(np.max(ws_sizes)) if ws_sizes.size else 0

        return {
            "frame_limit": self.frame_limit,
            "window_size": self.window_size,
            "total": total,
            "hits": self.hits,
            "faults": self.faults,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "fault_rate": fault_rate,
            "avg_ws_size": avg_ws_size,
            "max_ws_size": max_ws_size,
            "steps": steps,
        }


def run_sanity_checks() -> None:
    case_a = WorkingSetPager(frame_limit=3, window_size=3)
    result_a = case_a.simulate([1, 2, 3, 1])
    assert result_a["faults"] == 3
    assert result_a["hits"] == 1

    case_b = WorkingSetPager(frame_limit=2, window_size=2)
    result_b = case_b.simulate([1, 2, 3])
    assert result_b["faults"] == 3
    assert result_b["evictions"] == 1


def main() -> None:
    run_sanity_checks()

    reference_string = [1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 3, 4, 5, 6, 2, 1, 2, 7, 2, 1, 8, 2, 1]
    frame_limit = 4
    window_size = 5

    pager = WorkingSetPager(frame_limit=frame_limit, window_size=window_size)
    result = pager.simulate(reference_string)

    print("Working Set Page Replacement Demo")
    print(f"Reference string: {reference_string}")
    print(f"Frame limit    : {frame_limit}")
    print(f"Window size Δ  : {window_size}")
    print()
    print("Step | Ref | Result | Evicted | Resident(old->new) | WS(before) | WS(after)")
    print("-----+-----+--------+---------+---------------------+------------+----------")

    for step in result["steps"]:
        result_text = "HIT" if step.hit else "FAULT"
        evicted_text = "-" if step.evicted is None else str(step.evicted)
        print(
            f"{step.t:>4} | {step.page:>3} | {result_text:>6} | {evicted_text:>7} | "
            f"{str(step.resident_old_to_new):>19} | {str(step.ws_before):>10} | {step.ws_after}"
        )

    print()
    print(
        "Summary: "
        f"total={result['total']}, hits={result['hits']}, faults={result['faults']}, "
        f"evictions={result['evictions']}, hit_rate={result['hit_rate']:.2%}, "
        f"fault_rate={result['fault_rate']:.2%}, avg_ws_size={result['avg_ws_size']:.2f}, "
        f"max_ws_size={result['max_ws_size']}"
    )


if __name__ == "__main__":
    main()
