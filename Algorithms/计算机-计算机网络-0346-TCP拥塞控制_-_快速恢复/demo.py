"""Minimal runnable MVP for TCP fast recovery congestion control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

EVENT_ACK_NEW = "ACK_NEW"
EVENT_DUP_ACK = "DUP_ACK"

STATE_CONGESTION_AVOIDANCE = "congestion_avoidance"
STATE_FAST_RECOVERY = "fast_recovery"


@dataclass
class StepRecord:
    step: int
    event: str
    state_before: str
    state_after: str
    cwnd_before: int
    cwnd_after: int
    ssthresh_before: int
    ssthresh_after: int
    dup_ack_count_before: int
    dup_ack_count_after: int
    note: str


def validate_inputs(init_cwnd: int, init_ssthresh: int, events: Iterable[str]) -> List[str]:
    """Validate simulation inputs and return normalized event list."""
    if init_cwnd < 2:
        raise ValueError("init_cwnd must be >= 2 for this fast-recovery MVP")
    if init_ssthresh < 2:
        raise ValueError("init_ssthresh must be >= 2")

    normalized = [str(e).strip().upper() for e in events]
    if not normalized:
        raise ValueError("events must not be empty")

    allowed = {EVENT_ACK_NEW, EVENT_DUP_ACK}
    for idx, event in enumerate(normalized, start=1):
        if event not in allowed:
            raise ValueError(f"event at step {idx} is invalid: {event}")

    return normalized


def additive_increase(cwnd: int) -> int:
    """Simplified congestion-avoidance growth used by this MVP."""
    return cwnd + 1


def enter_fast_recovery(cwnd: int) -> Tuple[int, int]:
    """Apply Reno triple-duplicate-ACK entry rule."""
    new_ssthresh = max(cwnd // 2, 2)
    new_cwnd = new_ssthresh + 3
    return new_cwnd, new_ssthresh


def simulate_tcp_fast_recovery(
    init_cwnd: int,
    init_ssthresh: int,
    events: Iterable[str],
) -> List[StepRecord]:
    """Simulate Reno-style fast recovery on an ACK/dupACK event stream."""
    event_seq = validate_inputs(init_cwnd, init_ssthresh, events)

    cwnd = int(init_cwnd)
    ssthresh = int(init_ssthresh)
    state = STATE_CONGESTION_AVOIDANCE
    dup_ack_count = 0

    records: List[StepRecord] = []

    for step, event in enumerate(event_seq, start=1):
        cwnd_before = cwnd
        ssthresh_before = ssthresh
        state_before = state
        dup_before = dup_ack_count

        if state_before == STATE_CONGESTION_AVOIDANCE:
            if event == EVENT_ACK_NEW:
                dup_ack_count = 0
                cwnd = additive_increase(cwnd_before)
                note = "additive increase in congestion avoidance"
            else:
                dup_ack_count += 1
                if dup_ack_count == 3:
                    cwnd, ssthresh = enter_fast_recovery(cwnd_before)
                    state = STATE_FAST_RECOVERY
                    note = "triple dup ACK -> fast retransmit + enter fast recovery"
                else:
                    note = f"dup ACK count = {dup_ack_count}, waiting for triple dup ACK"

        elif state_before == STATE_FAST_RECOVERY:
            if event == EVENT_DUP_ACK:
                dup_ack_count += 1
                cwnd = cwnd_before + 1
                note = "additional dup ACK inflates cwnd by 1 in fast recovery"
            else:
                cwnd = ssthresh
                state = STATE_CONGESTION_AVOIDANCE
                dup_ack_count = 0
                note = "new ACK -> deflate cwnd to ssthresh and exit fast recovery"
        else:
            raise ValueError(f"unknown state: {state_before}")

        records.append(
            StepRecord(
                step=step,
                event=event,
                state_before=state_before,
                state_after=state,
                cwnd_before=cwnd_before,
                cwnd_after=cwnd,
                ssthresh_before=ssthresh_before,
                ssthresh_after=ssthresh,
                dup_ack_count_before=dup_before,
                dup_ack_count_after=dup_ack_count,
                note=note,
            )
        )

    return records


def extract_series(
    records: Sequence[StepRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract numpy arrays from records for checks and display."""
    cwnd_after = np.array([r.cwnd_after for r in records], dtype=int)
    ssthresh_after = np.array([r.ssthresh_after for r in records], dtype=int)
    state_after = np.array([r.state_after for r in records], dtype=object)
    dup_after = np.array([r.dup_ack_count_after for r in records], dtype=int)
    return cwnd_after, ssthresh_after, state_after, dup_after


def run_demo() -> None:
    """Run a deterministic fast-recovery demo with assertions."""
    print("=== TCP Fast Recovery MVP Demo ===")

    init_cwnd = 12
    init_ssthresh = 20
    events = [
        EVENT_ACK_NEW,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
    ]

    records = simulate_tcp_fast_recovery(
        init_cwnd=init_cwnd,
        init_ssthresh=init_ssthresh,
        events=events,
    )

    cwnd_after, ssthresh_after, state_after, dup_after = extract_series(records)

    expected_cwnd_after = np.array([13, 13, 13, 9, 10, 11, 6, 7, 8, 8, 8, 7, 4, 5], dtype=int)
    expected_ssthresh_after = np.array([20, 20, 20, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4], dtype=int)
    expected_state_after = np.array(
        [
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_FAST_RECOVERY,
            STATE_FAST_RECOVERY,
            STATE_FAST_RECOVERY,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_FAST_RECOVERY,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
        ],
        dtype=object,
    )
    expected_dup_after = np.array([0, 1, 2, 3, 4, 5, 0, 0, 0, 1, 2, 3, 0, 0], dtype=int)

    if not np.array_equal(cwnd_after, expected_cwnd_after):
        raise AssertionError(f"unexpected cwnd sequence: {cwnd_after}")
    if not np.array_equal(ssthresh_after, expected_ssthresh_after):
        raise AssertionError(f"unexpected ssthresh sequence: {ssthresh_after}")
    if not np.array_equal(state_after, expected_state_after):
        raise AssertionError(f"unexpected state sequence: {state_after}")
    if not np.array_equal(dup_after, expected_dup_after):
        raise AssertionError(f"unexpected dup-ACK counter sequence: {dup_after}")

    fr_entries = [
        r
        for r in records
        if r.state_before == STATE_CONGESTION_AVOIDANCE and r.state_after == STATE_FAST_RECOVERY
    ]
    if len(fr_entries) != 2:
        raise AssertionError(f"expected 2 fast-recovery entries, got {len(fr_entries)}")

    for rec in fr_entries:
        if rec.dup_ack_count_after != 3:
            raise AssertionError(f"step {rec.step}: fast recovery must start at dup_ack_count=3")
        if rec.cwnd_after != rec.ssthresh_after + 3:
            raise AssertionError(f"step {rec.step}: cwnd must inflate to ssthresh+3 at FR entry")

    for rec in records:
        if rec.state_before == STATE_FAST_RECOVERY and rec.event == EVENT_DUP_ACK:
            if rec.cwnd_after != rec.cwnd_before + 1:
                raise AssertionError(f"step {rec.step}: duplicate ACK in FR must increase cwnd by 1")
        if rec.state_before == STATE_FAST_RECOVERY and rec.event == EVENT_ACK_NEW:
            if rec.cwnd_after != rec.ssthresh_before:
                raise AssertionError(f"step {rec.step}: ACK exit from FR must deflate cwnd to ssthresh")
            if rec.state_after != STATE_CONGESTION_AVOIDANCE:
                raise AssertionError(f"step {rec.step}: ACK exit from FR must return to CA")
            if rec.dup_ack_count_after != 0:
                raise AssertionError(f"step {rec.step}: duplicate ACK counter must reset after recovery")

    print("Step timeline:")
    for rec in records:
        print(
            f"  step={rec.step:2d} event={rec.event:7s} "
            f"state={rec.state_before:20s}->{rec.state_after:20s} "
            f"dupAck={rec.dup_ack_count_before}->{rec.dup_ack_count_after} "
            f"cwnd={rec.cwnd_before:2d}->{rec.cwnd_after:2d} "
            f"ssthresh={rec.ssthresh_before:2d}->{rec.ssthresh_after:2d}"
        )

    print("cwnd_after sequence:", cwnd_after.tolist())
    print("ssthresh_after sequence:", ssthresh_after.tolist())
    print("state_after sequence:", state_after.tolist())
    print("dup_ack_count_after sequence:", dup_after.tolist())
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
