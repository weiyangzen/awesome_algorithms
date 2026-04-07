"""LFU page replacement MVP with deterministic demo and lightweight analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    tie_break_rule: str = ""

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
    if any(p < 0 for p in reference):
        raise ValueError("reference pages must be non-negative integers")


def lfu_page_replacement(reference: list[int], frame_count: int) -> ReplacementResult:
    """Run LFU replacement with LRU tie-break and return detailed transitions."""
    _validate_inputs(reference, frame_count)

    frames: list[int | None] = [None] * frame_count
    freq: dict[int, int] = {}
    last_used: dict[int, int] = {}

    hits = 0
    faults = 0
    evictions = 0
    events: list[dict[str, Any]] = []

    for step, page in enumerate(reference):
        replaced: int | None = None
        victim_freq: int | None = None

        if page in frames:
            hits += 1
            freq[page] += 1
            last_used[page] = step
            action = "hit"
        else:
            faults += 1

            if None in frames:
                slot = frames.index(None)
                frames[slot] = page
                freq[page] = 1
                last_used[page] = step
                action = "fault_fill"
            else:
                # LFU core: pick the lowest frequency page.
                # Tie-break: among same frequency, evict the least recently used page.
                slot, victim = min(
                    enumerate(frames),
                    key=lambda item: (
                        freq[item[1]],
                        last_used[item[1]],
                        item[0],
                    ),
                )
                assert victim is not None

                replaced = victim
                victim_freq = freq[victim]
                del freq[victim]
                del last_used[victim]

                frames[slot] = page
                freq[page] = 1
                last_used[page] = step
                action = "fault_replace"
                evictions += 1

        frame_view = [p for p in frames if p is not None]
        freq_view = {p: freq[p] for p in frame_view}

        events.append(
            {
                "step": step,
                "page": page,
                "action": action,
                "replaced": replaced,
                "victim_freq": victim_freq,
                "frames": str(frames),
                "freq": str(freq_view),
                "is_fault": int(action != "hit"),
            }
        )

    return ReplacementResult(
        algorithm="LFU",
        frame_count=frame_count,
        reference=reference,
        hits=hits,
        faults=faults,
        events=events,
        evictions=evictions,
        tie_break_rule="LRU when frequency ties",
    )


def fifo_page_replacement(reference: list[int], frame_count: int) -> ReplacementResult:
    _validate_inputs(reference, frame_count)

    frames: list[int] = []
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
                replaced = frames.pop(0)
                frames.append(page)
                action = "fault_replace"

        events.append(
            {
                "step": step,
                "page": page,
                "action": action,
                "replaced": replaced,
                "frames": str(frames),
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
    )


def lru_page_replacement(reference: list[int], frame_count: int) -> ReplacementResult:
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
                "replaced": replaced,
                "frames": str(frames),
                "is_fault": int(action != "hit"),
            }
        )

    return ReplacementResult(
        algorithm="LRU",
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


def run_random_workload_comparison(
    frame_count: int = 4,
    trials: int = 120,
    ref_len: int = 60,
    page_kinds: int = 10,
    seed: int = 20260407,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(trials):
        ref = rng.integers(0, page_kinds, size=ref_len, endpoint=False).tolist()

        rows.append(
            {
                "lfu_faults": lfu_page_replacement(ref, frame_count).faults,
                "fifo_faults": fifo_page_replacement(ref, frame_count).faults,
                "lru_faults": lru_page_replacement(ref, frame_count).faults,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    reference = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2]
    frame_count = 3

    lfu_result = lfu_page_replacement(reference, frame_count)
    fifo_result = fifo_page_replacement(reference, frame_count)
    lru_result = lru_page_replacement(reference, frame_count)

    assert lfu_result.hits + lfu_result.faults == len(reference)
    assert lfu_result.faults == 8, f"LFU faults changed: {lfu_result.faults}"
    assert lfu_result.evictions >= 1
    assert all(
        event["action"] in {"hit", "fault_fill", "fault_replace"}
        for event in lfu_result.events
    )

    summary_df = summarize_algorithms([lfu_result, fifo_result, lru_result])

    lfu_events_df = pd.DataFrame(lfu_result.events)
    show_cols = [
        "step",
        "page",
        "action",
        "replaced",
        "victim_freq",
        "frames",
        "freq",
    ]

    print("=== LFU Page Replacement Demo ===")
    print(f"reference = {reference}")
    print(f"frame_count = {frame_count}")
    print()
    print("Algorithm summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(
        "LFU internals: "
        f"evictions={lfu_result.evictions}, "
        f"tie_break_rule='{lfu_result.tie_break_rule}'"
    )
    print()
    print("LFU event trace (first 10 steps):")
    print(lfu_events_df[show_cols].head(10).to_string(index=False))
    print()

    random_df = run_random_workload_comparison()
    means = random_df.mean(axis=0)

    print("Random workload average faults (120 trials):")
    print(
        f"LFU={means['lfu_faults']:.3f}, "
        f"FIFO={means['fifo_faults']:.3f}, "
        f"LRU={means['lru_faults']:.3f}"
    )

    random_ref_len = 60
    assert len(random_df) == 120
    assert all(0.0 <= float(v) <= random_ref_len for v in means.values), (
        f"Mean faults out of bounds: {means.to_dict()}"
    )

    print("All assertions passed.")


if __name__ == "__main__":
    main()
