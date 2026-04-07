"""Clock page replacement MVP with deterministic demo and lightweight analysis."""

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
    hand_end: int = 0
    second_chance_clears: int = 0
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
    if any(p < 0 for p in reference):
        raise ValueError("reference pages must be non-negative integers")


def clock_page_replacement(reference: list[int], frame_count: int) -> ReplacementResult:
    """Run Clock replacement and return detailed state transitions."""
    _validate_inputs(reference, frame_count)

    frames: list[int | None] = [None] * frame_count
    use_bits = [0] * frame_count
    hand = 0

    hits = 0
    faults = 0
    evictions = 0
    second_chance_clears = 0
    events: list[dict[str, Any]] = []

    for step, page in enumerate(reference):
        hand_before = hand

        if page in frames:
            idx = frames.index(page)
            use_bits[idx] = 1
            hits += 1
            action = "hit"
            replaced = None
            scanned = 0
        else:
            faults += 1
            replaced = None
            scanned = 0

            while True:
                scanned += 1
                current = frames[hand]

                if current is None:
                    frames[hand] = page
                    use_bits[hand] = 1
                    action = "fault_fill"
                    hand = (hand + 1) % frame_count
                    break

                if use_bits[hand] == 0:
                    replaced = current
                    frames[hand] = page
                    use_bits[hand] = 1
                    action = "fault_replace"
                    evictions += 1
                    hand = (hand + 1) % frame_count
                    break

                use_bits[hand] = 0
                second_chance_clears += 1
                hand = (hand + 1) % frame_count

        events.append(
            {
                "step": step,
                "page": page,
                "action": action,
                "replaced": replaced,
                "scanned_slots": scanned,
                "hand_before": hand_before,
                "hand_after": hand,
                "frames": str(frames),
                "use_bits": str(use_bits),
                "is_fault": int(action != "hit"),
            }
        )

    return ReplacementResult(
        algorithm="Clock",
        frame_count=frame_count,
        reference=reference,
        hits=hits,
        faults=faults,
        events=events,
        hand_end=hand,
        second_chance_clears=second_chance_clears,
        evictions=evictions,
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

        clock_faults = clock_page_replacement(ref, frame_count).faults
        fifo_faults = fifo_page_replacement(ref, frame_count).faults
        lru_faults = lru_page_replacement(ref, frame_count).faults

        rows.append(
            {
                "clock_faults": clock_faults,
                "fifo_faults": fifo_faults,
                "lru_faults": lru_faults,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    reference = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2]
    frame_count = 3

    clock_result = clock_page_replacement(reference, frame_count)
    fifo_result = fifo_page_replacement(reference, frame_count)
    lru_result = lru_page_replacement(reference, frame_count)

    assert clock_result.hits + clock_result.faults == len(reference)
    assert clock_result.faults == 9, f"Clock faults changed: {clock_result.faults}"
    assert clock_result.second_chance_clears >= 1
    assert all(event["action"] in {"hit", "fault_fill", "fault_replace"} for event in clock_result.events)

    summary_df = summarize_algorithms([clock_result, fifo_result, lru_result])

    clock_events_df = pd.DataFrame(clock_result.events)
    show_cols = [
        "step",
        "page",
        "action",
        "replaced",
        "scanned_slots",
        "hand_before",
        "hand_after",
        "frames",
        "use_bits",
    ]

    print("=== Clock Page Replacement Demo ===")
    print(f"reference = {reference}")
    print(f"frame_count = {frame_count}")
    print()
    print("Algorithm summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(
        "Clock internals: "
        f"hand_end={clock_result.hand_end}, "
        f"second_chance_clears={clock_result.second_chance_clears}, "
        f"evictions={clock_result.evictions}"
    )
    print()
    print("Clock event trace (first 10 steps):")
    print(clock_events_df[show_cols].head(10).to_string(index=False))
    print()

    random_df = run_random_workload_comparison()
    means = random_df.mean(axis=0)
    print("Random workload average faults (120 trials):")
    print(
        f"Clock={means['clock_faults']:.3f}, "
        f"FIFO={means['fifo_faults']:.3f}, "
        f"LRU={means['lru_faults']:.3f}"
    )

    # Keep assertions weak and robust: only check numeric sanity bounds.
    random_ref_len = 60
    assert len(random_df) == 120
    assert all(0.0 <= float(v) <= random_ref_len for v in means.values), (
        f"Mean faults out of bounds: {means.to_dict()}"
    )

    print("All assertions passed.")


if __name__ == "__main__":
    main()
