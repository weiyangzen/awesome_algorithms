"""FIFO page replacement MVP with deterministic demo and lightweight analysis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Deque

import numpy as np
import pandas as pd


@dataclass
class ReplacementResult:
    algorithm: str
    frame_count: int
    reference: list[int]
    hits: int
    faults: int
    events: list[dict[str, Any]]
    evictions: int = 0

    @property
    def hit_ratio(self) -> float:
        total = len(self.reference)
        return float(self.hits / total) if total else 0.0

    @property
    def fault_ratio(self) -> float:
        total = len(self.reference)
        return float(self.faults / total) if total else 0.0


def _validate_inputs(reference: list[int], frame_count: int) -> None:
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0")
    if any((not isinstance(p, Integral)) or p < 0 for p in reference):
        raise ValueError("reference pages must be non-negative integers")


def fifo_page_replacement(reference: list[int], frame_count: int) -> ReplacementResult:
    """Run FIFO replacement and return detailed event trace."""
    _validate_inputs(reference, frame_count)

    queue: Deque[int] = deque()
    resident: set[int] = set()

    hits = 0
    faults = 0
    evictions = 0
    events: list[dict[str, Any]] = []

    for step, page in enumerate(reference):
        replaced: int | None = None

        if page in resident:
            hits += 1
            action = "hit"
        else:
            faults += 1
            if len(queue) < frame_count:
                queue.append(page)
                resident.add(page)
                action = "fault_fill"
            else:
                replaced = queue.popleft()
                resident.remove(replaced)
                queue.append(page)
                resident.add(page)
                action = "fault_replace"
                evictions += 1

        events.append(
            {
                "step": step,
                "page": page,
                "action": action,
                "replaced": "-" if replaced is None else str(replaced),
                "frames_oldest_to_newest": str(list(queue)),
                "is_fault": int(action != "hit"),
            }
        )

    return ReplacementResult(
        algorithm="FIFO",
        frame_count=frame_count,
        reference=reference,
        hits=hits,
        faults=faults,
        events=events,
        evictions=evictions,
    )


def lru_page_replacement(reference: list[int], frame_count: int) -> ReplacementResult:
    """Small baseline used only for side-by-side comparison in the demo output."""
    _validate_inputs(reference, frame_count)

    frames: list[int] = []
    last_used: dict[int, int] = {}
    hits = 0
    faults = 0
    events: list[dict[str, Any]] = []

    for step, page in enumerate(reference):
        replaced: int | None = None

        if page in frames:
            hits += 1
            action = "hit"
        else:
            faults += 1
            if len(frames) < frame_count:
                frames.append(page)
                action = "fault_fill"
            else:
                victim = min(frames, key=lambda p: last_used[p])
                victim_idx = frames.index(victim)
                replaced = victim
                frames[victim_idx] = page
                action = "fault_replace"

        last_used[page] = step
        events.append(
            {
                "step": step,
                "page": page,
                "action": action,
                "replaced": "-" if replaced is None else str(replaced),
                "frames": str(frames),
                "is_fault": int(action != "hit"),
            }
        )

    return ReplacementResult(
        algorithm="LRU-baseline",
        frame_count=frame_count,
        reference=reference,
        hits=hits,
        faults=faults,
        events=events,
    )


def summarize_algorithms(results: list[ReplacementResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "algorithm": r.algorithm,
                "frame_count": r.frame_count,
                "ref_len": len(r.reference),
                "hits": r.hits,
                "faults": r.faults,
                "hit_ratio": r.hit_ratio,
                "fault_ratio": r.fault_ratio,
            }
        )

    return pd.DataFrame(rows).sort_values(by=["faults", "algorithm"]).reset_index(
        drop=True
    )


def run_random_fifo_anomaly_scan(
    small_frames: int = 3,
    large_frames: int = 4,
    trials: int = 120,
    ref_len: int = 50,
    page_kinds: int = 8,
    seed: int = 20260407,
) -> pd.DataFrame:
    """Search Belady anomaly cases for FIFO on random references."""
    if small_frames >= large_frames:
        raise ValueError("small_frames must be < large_frames")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for _ in range(trials):
        ref = rng.integers(0, page_kinds, size=ref_len, endpoint=False).tolist()
        faults_small = fifo_page_replacement(ref, small_frames).faults
        faults_large = fifo_page_replacement(ref, large_frames).faults

        rows.append(
            {
                f"faults_{small_frames}": faults_small,
                f"faults_{large_frames}": faults_large,
                "anomaly": int(faults_large > faults_small),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    reference = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2]
    frame_count = 3

    fifo_result = fifo_page_replacement(reference, frame_count)
    lru_result = lru_page_replacement(reference, frame_count)

    assert fifo_result.hits + fifo_result.faults == len(reference)
    assert fifo_result.faults == 10, f"FIFO faults changed: {fifo_result.faults}"
    assert fifo_result.evictions == max(fifo_result.faults - frame_count, 0)
    assert all(
        event["action"] in {"hit", "fault_fill", "fault_replace"}
        for event in fifo_result.events
    )

    summary_df = summarize_algorithms([fifo_result, lru_result])
    fifo_events_df = pd.DataFrame(fifo_result.events)

    print("=== FIFO Page Replacement Demo ===")
    print(f"reference = {reference}")
    print(f"frame_count = {frame_count}")
    print()
    print("Algorithm summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"FIFO internals: evictions={fifo_result.evictions}")
    print()
    print("FIFO event trace (first 10 steps):")
    show_cols = [
        "step",
        "page",
        "action",
        "replaced",
        "frames_oldest_to_newest",
    ]
    print(fifo_events_df[show_cols].head(10).to_string(index=False))
    print()

    belady_reference = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    fifo_3 = fifo_page_replacement(belady_reference, 3)
    fifo_4 = fifo_page_replacement(belady_reference, 4)

    assert fifo_3.faults == 9, f"Unexpected FIFO-3 faults: {fifo_3.faults}"
    assert fifo_4.faults == 10, f"Unexpected FIFO-4 faults: {fifo_4.faults}"
    assert fifo_4.faults > fifo_3.faults, "Belady anomaly demo failed"

    print("Belady anomaly check (classic reference):")
    print(f"reference = {belady_reference}")
    print(f"FIFO faults with 3 frames = {fifo_3.faults}")
    print(f"FIFO faults with 4 frames = {fifo_4.faults}")
    print()

    random_df = run_random_fifo_anomaly_scan()
    means = random_df.mean(axis=0)
    anomaly_rate = float(means["anomaly"])

    print("Random workload scan (120 trials):")
    print(
        "avg_faults_3={:.3f}, avg_faults_4={:.3f}, anomaly_rate={:.2%}".format(
            float(means["faults_3"]),
            float(means["faults_4"]),
            anomaly_rate,
        )
    )

    assert len(random_df) == 120
    assert 0.0 <= anomaly_rate <= 1.0
    print("All assertions passed.")


if __name__ == "__main__":
    main()
