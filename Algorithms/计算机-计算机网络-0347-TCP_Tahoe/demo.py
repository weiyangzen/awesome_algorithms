"""Minimal runnable MVP for TCP Tahoe congestion control."""

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
    ca_ack_counter_before: int
    ca_ack_counter_after: int
    retransmissions_before: int
    retransmissions_after: int
    note: str


def validate_inputs(init_cwnd: int, init_ssthresh: int, events: Iterable[str]) -> List[str]:
    """Validate inputs and normalize events."""
    if init_cwnd < 1:
        raise ValueError("init_cwnd must be >= 1")
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


def multiplicative_decrease(cwnd: int) -> int:
    """Tahoe multiplicative decrease for ssthresh update."""
    return max(cwnd // 2, 2)


def simulate_tcp_tahoe(
    init_cwnd: int,
    init_ssthresh: int,
    events: Iterable[str],
) -> List[StepRecord]:
    """Simulate a compact TCP Tahoe state machine over an event sequence."""
    event_seq = validate_inputs(init_cwnd=init_cwnd, init_ssthresh=init_ssthresh, events=events)

    cwnd = int(init_cwnd)
    ssthresh = int(init_ssthresh)
    state = STATE_SLOW_START
    dup_ack_count = 0
    ca_ack_counter = 0
    retransmissions = 0

    records: List[StepRecord] = []

    for step, event in enumerate(event_seq, start=1):
        state_before = state
        cwnd_before = cwnd
        ssthresh_before = ssthresh
        dup_before = dup_ack_count
        ca_before = ca_ack_counter
        retrans_before = retransmissions

        if event == EVENT_TIMEOUT:
            ssthresh = multiplicative_decrease(cwnd_before)
            cwnd = 1
            state = STATE_SLOW_START
            dup_ack_count = 0
            ca_ack_counter = 0
            retransmissions += 1
            note = "RTO timeout: multiplicative decrease and restart with cwnd=1"

        elif event == EVENT_DUP_ACK:
            dup_ack_count += 1
            if dup_ack_count == 3:
                ssthresh = multiplicative_decrease(cwnd_before)
                cwnd = 1
                state = STATE_SLOW_START
                dup_ack_count = 0
                ca_ack_counter = 0
                retransmissions += 1
                note = "triple DUP_ACK: fast retransmit then Tahoe fallback to slow start"
            else:
                note = f"DUP_ACK observed, count={dup_ack_count}, waiting for triple DUP_ACK"

        elif event == EVENT_ACK_NEW:
            dup_ack_count = 0
            if state_before == STATE_SLOW_START:
                cwnd = cwnd_before + 1
                if cwnd >= ssthresh_before:
                    state = STATE_CONGESTION_AVOIDANCE
                    ca_ack_counter = 0
                    note = "slow start reaches ssthresh, switch to congestion avoidance"
                else:
                    note = "slow start growth: cwnd += 1 per ACK"
            elif state_before == STATE_CONGESTION_AVOIDANCE:
                ca_ack_counter += 1
                if ca_ack_counter >= cwnd_before:
                    cwnd = cwnd_before + 1
                    ca_ack_counter = 0
                    note = "congestion avoidance additive increase: +1 after cwnd ACKs"
                else:
                    note = "congestion avoidance accumulating ACK credit"
            else:
                raise ValueError(f"unknown state: {state_before}")
        else:
            raise ValueError(f"unsupported event: {event}")

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
                ca_ack_counter_before=ca_before,
                ca_ack_counter_after=ca_ack_counter,
                retransmissions_before=retrans_before,
                retransmissions_after=retransmissions,
                note=note,
            )
        )

    return records


def extract_series(
    records: Sequence[StepRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract numpy arrays for deterministic checks."""
    cwnd_after = np.array([r.cwnd_after for r in records], dtype=int)
    ssthresh_after = np.array([r.ssthresh_after for r in records], dtype=int)
    state_after = np.array([r.state_after for r in records], dtype=object)
    dup_after = np.array([r.dup_ack_count_after for r in records], dtype=int)
    ca_after = np.array([r.ca_ack_counter_after for r in records], dtype=int)
    retrans_after = np.array([r.retransmissions_after for r in records], dtype=int)
    return cwnd_after, ssthresh_after, state_after, dup_after, ca_after, retrans_after


def records_to_dataframe(records: Sequence[StepRecord]) -> pd.DataFrame:
    """Convert records to a readable timeline table."""
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
            "caAckCounter_after": [r.ca_ack_counter_after for r in records],
            "retrans_after": [r.retransmissions_after for r in records],
            "note": [r.note for r in records],
        }
    )


def run_demo() -> None:
    """Run deterministic Tahoe demo with assertions."""
    print("=== TCP Tahoe MVP Demo ===")

    init_cwnd = 1
    init_ssthresh = 8
    events = [
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_TIMEOUT,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_DUP_ACK,
        EVENT_ACK_NEW,
        EVENT_ACK_NEW,
        EVENT_DUP_ACK,
        EVENT_DUP_ACK,
        EVENT_ACK_NEW,
    ]

    records = simulate_tcp_tahoe(
        init_cwnd=init_cwnd,
        init_ssthresh=init_ssthresh,
        events=events,
    )

    cwnd_after, ssthresh_after, state_after, dup_after, ca_after, retrans_after = extract_series(
        records
    )

    expected_cwnd_after = np.array(
        [2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 1, 2, 3, 4, 4, 1, 2, 2, 2, 3, 3, 3, 3, 3],
        dtype=int,
    )
    expected_ssthresh_after = np.array(
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        dtype=int,
    )
    expected_state_after = np.array(
        [
            STATE_SLOW_START,
            STATE_SLOW_START,
            STATE_SLOW_START,
            STATE_SLOW_START,
            STATE_SLOW_START,
            STATE_SLOW_START,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_SLOW_START,
            STATE_SLOW_START,
            STATE_SLOW_START,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_SLOW_START,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
            STATE_CONGESTION_AVOIDANCE,
        ],
        dtype=object,
    )
    expected_dup_after = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 2, 0],
        dtype=int,
    )
    expected_ca_after = np.array(
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 2],
        dtype=int,
    )
    expected_retrans_after = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
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
    if not np.array_equal(ca_after, expected_ca_after):
        raise AssertionError(f"unexpected CA-ACK-credit sequence: {ca_after}")
    if not np.array_equal(retrans_after, expected_retrans_after):
        raise AssertionError(f"unexpected retransmission sequence: {retrans_after}")

    triple_dup_fallbacks = [
        r
        for r in records
        if r.event == EVENT_DUP_ACK and r.dup_ack_count_before == 2 and r.retransmissions_after == 1
    ]
    if len(triple_dup_fallbacks) != 1:
        raise AssertionError(
            f"expected exactly one triple-dup fallback, got {len(triple_dup_fallbacks)}"
        )
    triple_dup_rec = triple_dup_fallbacks[0]
    if triple_dup_rec.cwnd_after != 1 or triple_dup_rec.state_after != STATE_SLOW_START:
        raise AssertionError("Tahoe must set cwnd=1 and re-enter slow start on triple DUP_ACK")
    if triple_dup_rec.ssthresh_after != 4:
        raise AssertionError("expected ssthresh=4 after first multiplicative decrease")

    timeout_recs = [r for r in records if r.event == EVENT_TIMEOUT]
    if len(timeout_recs) != 1:
        raise AssertionError(f"expected exactly one timeout event, got {len(timeout_recs)}")
    timeout_rec = timeout_recs[0]
    if timeout_rec.cwnd_after != 1 or timeout_rec.state_after != STATE_SLOW_START:
        raise AssertionError("timeout must force Tahoe back to slow start with cwnd=1")
    if timeout_rec.ssthresh_after != 2:
        raise AssertionError("expected ssthresh=2 after timeout multiplicative decrease")

    if any(r.ssthresh_after < 2 for r in records):
        raise AssertionError("ssthresh must never be below 2")
    if any(r.cwnd_after < 1 for r in records):
        raise AssertionError("cwnd must never be below 1")

    table = records_to_dataframe(records)
    print("Step timeline:")
    with pd.option_context("display.max_colwidth", 60, "display.width", 180):
        print(table.to_string(index=False))

    print("cwnd_after sequence:", cwnd_after.tolist())
    print("ssthresh_after sequence:", ssthresh_after.tolist())
    print("state_after sequence:", state_after.tolist())
    print("dup_ack_count_after sequence:", dup_after.tolist())
    print("ca_ack_counter_after sequence:", ca_after.tolist())
    print("retransmissions_after sequence:", retrans_after.tolist())
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
