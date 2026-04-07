"""Minimal runnable MVP for TCP Reno congestion control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

EVENT_ACK_NEW = "ACK_NEW"
EVENT_DUP_ACK = "DUP_ACK"
EVENT_TIMEOUT = "TIMEOUT"

STATE_SLOW_START = "slow_start"
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
    retransmissions_before: int
    retransmissions_after: int
    note: str


def validate_inputs(init_cwnd: int, init_ssthresh: int, events: Iterable[str]) -> List[str]:
    """Validate inputs and normalize event sequence."""
    if init_cwnd < 2:
        raise ValueError("init_cwnd must be >= 2")
    if init_ssthresh < 2:
        raise ValueError("init_ssthresh must be >= 2")

    normalized = [str(e).strip().upper() for e in events]
    if not normalized:
        raise ValueError("events must not be empty")

    allowed = {EVENT_ACK_NEW, EVENT_DUP_ACK, EVENT_TIMEOUT}
    for idx, event in enumerate(normalized, start=1):
        if event not in allowed:
            raise ValueError(f"event at step {idx} is invalid: {event}")
    return normalized


def enter_fast_recovery(cwnd_before: int) -> Tuple[int, int]:
    """Triple-duplicate-ACK transition used by Reno."""
    new_ssthresh = max(cwnd_before // 2, 2)
    new_cwnd = new_ssthresh + 3
    return new_cwnd, new_ssthresh


def simulate_tcp_reno(
    init_cwnd: int,
    init_ssthresh: int,
    events: Iterable[str],
) -> List[StepRecord]:
    """Simulate an educational Reno state machine over a discrete event stream."""
    event_seq = validate_inputs(init_cwnd=init_cwnd, init_ssthresh=init_ssthresh, events=events)

    cwnd = int(init_cwnd)
    ssthresh = int(init_ssthresh)
    state = STATE_CONGESTION_AVOIDANCE
    dup_ack_count = 0
    retransmissions = 0

    records: List[StepRecord] = []

    for step, event in enumerate(event_seq, start=1):
        cwnd_before = cwnd
        ssthresh_before = ssthresh
        state_before = state
        dup_before = dup_ack_count
        retrans_before = retransmissions

        if event == EVENT_TIMEOUT:
            ssthresh = max(cwnd_before // 2, 2)
            cwnd = 1
            state = STATE_SLOW_START
            dup_ack_count = 0
            note = "RTO timeout: cwnd->1, update ssthresh, restart slow start"

        elif state_before == STATE_FAST_RECOVERY:
            if event == EVENT_DUP_ACK:
                dup_ack_count += 1
                cwnd = cwnd_before + 1
                note = "additional dup ACK in fast recovery inflates cwnd by 1"
            elif event == EVENT_ACK_NEW:
                # Reno exits fast recovery once any new ACK arrives.
                cwnd = ssthresh_before
                state = STATE_CONGESTION_AVOIDANCE
                dup_ack_count = 0
                note = "new ACK ends fast recovery; deflate cwnd to ssthresh"
            else:
                raise ValueError(f"event {event} is invalid in fast recovery")

        elif state_before in {STATE_CONGESTION_AVOIDANCE, STATE_SLOW_START}:
            if event == EVENT_ACK_NEW:
                dup_ack_count = 0
                if state_before == STATE_SLOW_START:
                    cwnd = cwnd_before * 2
                    if cwnd >= ssthresh_before:
                        state = STATE_CONGESTION_AVOIDANCE
                        note = "slow start reaches ssthresh, switch to congestion avoidance"
                    else:
                        note = "slow start exponential growth"
                else:
                    cwnd = cwnd_before + 1
                    note = "additive increase in congestion avoidance"
            elif event == EVENT_DUP_ACK:
                dup_ack_count += 1
                if dup_ack_count == 3:
                    cwnd, ssthresh = enter_fast_recovery(cwnd_before=cwnd_before)
                    state = STATE_FAST_RECOVERY
                    retransmissions += 1
                    note = "triple dup ACK: fast retransmit and enter fast recovery"
                else:
                    note = f"dup ACK count={dup_ack_count}, waiting for triple dup ACK"
            else:
                raise ValueError(f"event {event} is invalid in {state_before}")

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
                retransmissions_before=retrans_before,
                retransmissions_after=retransmissions,
                note=note,
            )
        )

    return records


def extract_series(
    records: Sequence[StepRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract deterministic numpy series from simulation records."""
    cwnd_after = np.array([r.cwnd_after for r in records], dtype=int)
    ssthresh_after = np.array([r.ssthresh_after for r in records], dtype=int)
    state_after = np.array([r.state_after for r in records], dtype=object)
    dup_after = np.array([r.dup_ack_count_after for r in records], dtype=int)
    retrans_after = np.array([r.retransmissions_after for r in records], dtype=int)
    return cwnd_after, ssthresh_after, state_after, dup_after, retrans_after


def records_to_dataframe(records: Sequence[StepRecord]) -> pd.DataFrame:
    """Build a readable pandas table from simulation records."""
    return pd.DataFrame(
        {
            "step": [r.step for r in records],
            "event": [r.event for r in records],
            "state_before": [r.state_before for r in records],
            "state_after": [r.state_after for r in records],
            "cwnd_before": [r.cwnd_before for r in records],
            "cwnd_after": [r.cwnd_after for r in records],
            "ssthresh_after": [r.ssthresh_after for r in records],
            "dupAck_after": [r.dup_ack_count_after for r in records],
            "retrans_after": [r.retransmissions_after for r in records],
            "note": [r.note for r in records],
        }
    )


def run_demo() -> None:
    """Run deterministic Reno demo and assertions."""
    print("=== TCP Reno MVP Demo ===")

    init_cwnd = 12
    init_ssthresh = 18
    events = [
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_ACK_NEW,
        EVENT_TIMEOUT,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_ACK_NEW,
    ]

    records = simulate_tcp_reno(init_cwnd=init_cwnd, init_ssthresh=init_ssthresh, events=events)
    cwnd_after, ssthresh_after, state_after, dup_after, retrans_after = extract_series(records)

    expected_cwnd_after = np.array(
        [13, 14, 14, 14, 10, 11, 7, 8, 8, 8, 7, 4, 1, 2, 3, 4, 4, 4, 5, 2],
        dtype=int,
    )
    expected_ssthresh_after = np.array(
        [18, 18, 18, 18, 7, 7, 7, 7, 7, 7, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2],
        dtype=int,
    )
    expected_state_after = np.array(
        [
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_FAST_RECOVERY,
            STATE_FAST_RECOVERY,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_FAST_RECOVERY,
            STATE_CONGESTION_AVOIDANCE,
            STATE_SLOW_START,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_FAST_RECOVERY,
            STATE_CONGESTION_AVOIDANCE,
        ],
        dtype=object,
    )
    expected_dup_after = np.array(
        [0, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 0],
        dtype=int,
    )
    expected_retrans_after = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3],
        dtype=int,
    )

    if not np.array_equal(cwnd_after, expected_cwnd_after):
        raise AssertionError(f"unexpected cwnd sequence: {cwnd_after}")
    if not np.array_equal(ssthresh_after, expected_ssthresh_after):
        raise AssertionError(f"unexpected ssthresh sequence: {ssthresh_after}")
    if not np.array_equal(state_after, expected_state_after):
        raise AssertionError(f"unexpected state sequence: {state_after}")
    if not np.array_equal(dup_after, expected_dup_after):
        raise AssertionError(f"unexpected dup-ACK sequence: {dup_after}")
    if not np.array_equal(retrans_after, expected_retrans_after):
        raise AssertionError(f"unexpected retransmission sequence: {retrans_after}")

    fr_entries = [
        r
        for r in records
        if r.state_before != STATE_FAST_RECOVERY and r.state_after == STATE_FAST_RECOVERY
    ]
    if len(fr_entries) != 3:
        raise AssertionError(f"expected 3 fast-recovery entries, got {len(fr_entries)}")

    for rec in fr_entries:
        if rec.dup_ack_count_after != 3:
            raise AssertionError(f"step {rec.step}: FR must start at triple dup ACK")
        if rec.cwnd_after != rec.ssthresh_after + 3:
            raise AssertionError(f"step {rec.step}: FR entry must set cwnd=ssthresh+3")

    fr_ack_exits = [
        r for r in records if r.state_before == STATE_FAST_RECOVERY and r.event == EVENT_ACK_NEW
    ]
    if len(fr_ack_exits) != 3:
        raise AssertionError(f"expected 3 ACK-based fast-recovery exits, got {len(fr_ack_exits)}")
    for rec in fr_ack_exits:
        if rec.state_after != STATE_CONGESTION_AVOIDANCE:
            raise AssertionError(f"step {rec.step}: Reno should exit fast recovery on new ACK")
        if rec.cwnd_after != rec.ssthresh_before:
            raise AssertionError(f"step {rec.step}: Reno exit should deflate cwnd to ssthresh")

    timeout_steps = [r for r in records if r.event == EVENT_TIMEOUT]
    if len(timeout_steps) != 1:
        raise AssertionError(f"expected exactly one timeout event, got {len(timeout_steps)}")
    timeout_rec = timeout_steps[0]
    if timeout_rec.cwnd_after != 1 or timeout_rec.state_after != STATE_SLOW_START:
        raise AssertionError("timeout must reset cwnd to 1 and enter slow_start")

    df = records_to_dataframe(records)
    print("\n--- Step timeline ---")
    print(df.to_string(index=False))
    print("\nAll checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
