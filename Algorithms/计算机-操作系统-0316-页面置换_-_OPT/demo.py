"""OPT page replacement MVP with deterministic demo and lightweight analysis."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import inf
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


def _build_future_positions(reference: list[int]) -> dict[int, deque[int]]:
    positions: dict[int, deque[int]] = defaultdict(deque)
    for idx, page in enumerate(reference):
        positions[page].append(idx)
    return positions


def opt_page_replacement(reference: list[int], frame_count: int) -> ReplacementResult:
    """Run OPT (Belady) replacement and return detailed transitions.

    Victim rule on fault with full frames:
    - Evict the page whose next use is farthest in the future.
    - If a page is never used again, its next use is treated as infinity.
    - Deterministic tie-break: choose the smaller frame slot index.
    """

    _validate_inputs(reference, frame_count)

    frames: list[int] = []
    future_positions = _build_future_positions(reference)

    hits = 0
    faults = 0
    events: list[dict[str, Any]] = []

    for step, page in enumerate(reference):
        # Remove current index so remaining queue starts from the true "future".
        if future_positions[page]:
            future_positions[page].popleft()

        replaced: int | None = None
        replaced_next_use: float | None = None

        if page in frames:
            hits += 1
            action = "hit"
        else:
            faults += 1

            if len(frames) < frame_count:
                frames.append(page)
                action = "fault_fill"
            else:
                # OPT core: inspect each resident page's next future access.
                candidates: list[tuple[int, int, float]] = []
                for slot, resident in enumerate(frames):
                    queue = future_positions.get(resident)
                    next_use = float(queue[0]) if queue else inf
                    candidates.append((slot, resident, next_use))

                victim_slot, victim_page, victim_next_use = max(
                    candidates,
                    key=lambda item: (item[2], -item[0]),
                )

                replaced = victim_page
                replaced_next_use = victim_next_use
                frames[victim_slot] = page
                action = "fault_replace"

        events.append(
            {
                "step": step,
                "page": page,
                "action": action,
                "replaced": replaced,
                "replaced_next_use": replaced_next_use,
                "frames": str(frames),
                "is_fault": int(action != "hit"),
            }
        )

    return ReplacementResult(
        algorithm="OPT",
        frame_count=frame_count,
        reference=reference,
        hits=hits,
        faults=faults,
        events=events,
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
                victim_slot = frames.index(victim)
                replaced = victim
                frames[victim_slot] = page
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
    for result in results:
        rows.append(
            {
                "algorithm": result.algorithm,
                "frame_count": result.frame_count,
                "ref_len": len(result.reference),
                "hits": result.hits,
                "faults": result.faults,
                "hit_ratio": result.hit_ratio,
                "fault_ratio": result.fault_ratio,
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
        reference = rng.integers(0, page_kinds, size=ref_len, endpoint=False).tolist()

        opt_faults = opt_page_replacement(reference, frame_count).faults
        fifo_faults = fifo_page_replacement(reference, frame_count).faults
        lru_faults = lru_page_replacement(reference, frame_count).faults

        rows.append(
            {
                "opt_faults": opt_faults,
                "fifo_faults": fifo_faults,
                "lru_faults": lru_faults,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    reference = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2]
    frame_count = 3

    opt_result = opt_page_replacement(reference, frame_count)
    fifo_result = fifo_page_replacement(reference, frame_count)
    lru_result = lru_page_replacement(reference, frame_count)

    summary_df = summarize_algorithms([opt_result, fifo_result, lru_result])

    print("OPT Page Replacement Demo")
    print(f"Reference string: {reference}")
    print(f"Frame count     : {frame_count}")
    print()
    print("Summary (fixed reference)")
    print(summary_df.to_string(index=False))
    print()

    print("OPT trace")
    print("Step | Ref | Result | Replaced(next_use) | Frames")
    print("-----+-----+--------+--------------------+----------------")
    for event in opt_result.events:
        result_text = "HIT" if event["action"] == "hit" else "FAULT"
        if event["replaced"] is None:
            replaced_text = "-"
        else:
            next_use = event["replaced_next_use"]
            next_use_text = "never" if next_use == inf else str(int(next_use))
            replaced_text = f"{event['replaced']}({next_use_text})"

        print(
            f"{event['step']:>4} | {event['page']:>3} | {result_text:>6} | "
            f"{replaced_text:>18} | {event['frames']}"
        )

    print()
    random_df = run_random_workload_comparison()
    means = random_df.mean().to_frame(name="mean_faults")
    print("Random workload mean faults (trials=120, ref_len=60, frame_count=4)")
    print(means.to_string())

    # Basic correctness checks for deterministic and random settings.
    assert opt_result.hits + opt_result.faults == len(reference)
    assert opt_result.faults <= fifo_result.faults
    assert opt_result.faults <= lru_result.faults

    # By definition, OPT should not be worse than FIFO/LRU on each trial.
    assert bool((random_df["opt_faults"] <= random_df["fifo_faults"]).all())
    assert bool((random_df["opt_faults"] <= random_df["lru_faults"]).all())

    print()
    print("All assertions passed.")


if __name__ == "__main__":
    main()
