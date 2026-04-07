"""Minimal runnable MVP for TCP fast retransmit congestion control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

ACK_NEW = "NEW_ACK"
ACK_DUP = "DUP_ACK"
ACK_STALE = "STALE_ACK"


@dataclass
class StepRecord:
    step: int
    ack_no: int
    ack_kind: str
    highest_ack_before: int
    highest_ack_after: int
    dup_ack_count_before: int
    dup_ack_count_after: int
    cwnd_before: int
    cwnd_after: int
    ssthresh_before: int
    ssthresh_after: int
    in_fast_retransmit_before: bool
    in_fast_retransmit_after: bool
    retransmissions_before: int
    retransmissions_after: int
    retransmitted_seq: int | None
    note: str


def validate_inputs(init_cwnd: int, init_ssthresh: int, ack_stream: Iterable[int]) -> List[int]:
    """Validate simulation inputs and return normalized ACK stream."""
    if init_cwnd < 2:
        raise ValueError("init_cwnd must be >= 2")
    if init_ssthresh < 2:
        raise ValueError("init_ssthresh must be >= 2")

    normalized = [int(x) for x in ack_stream]
    if not normalized:
        raise ValueError("ack_stream must not be empty")
    if any(x <= 0 for x in normalized):
        raise ValueError("ack numbers must be positive integers")

    return normalized


def classify_ack(ack_no: int, highest_ack: int) -> str:
    """Classify an ACK against current cumulative ACK frontier."""
    if ack_no > highest_ack:
        return ACK_NEW
    if ack_no == highest_ack:
        return ACK_DUP
    return ACK_STALE


def on_triple_duplicate_ack(cwnd: int) -> Tuple[int, int]:
    """Apply Reno-style fast retransmit entry update."""
    new_ssthresh = max(cwnd // 2, 2)
    new_cwnd = new_ssthresh + 3
    return new_cwnd, new_ssthresh


def simulate_tcp_fast_retransmit(
    init_cwnd: int,
    init_ssthresh: int,
    initial_highest_ack: int,
    ack_stream: Iterable[int],
) -> List[StepRecord]:
    """Simulate fast retransmit trigger on cumulative ACK stream."""
    ack_seq = validate_inputs(init_cwnd=init_cwnd, init_ssthresh=init_ssthresh, ack_stream=ack_stream)
    if initial_highest_ack <= 0:
        raise ValueError("initial_highest_ack must be positive")

    highest_ack = int(initial_highest_ack)
    dup_ack_count = 0
    cwnd = int(init_cwnd)
    ssthresh = int(init_ssthresh)
    in_fast_retransmit = False
    retransmissions = 0

    records: List[StepRecord] = []

    for step, ack_no in enumerate(ack_seq, start=1):
        highest_before = highest_ack
        dup_before = dup_ack_count
        cwnd_before = cwnd
        ssthresh_before = ssthresh
        fast_before = in_fast_retransmit
        retrans_before = retransmissions
        retransmitted_seq: int | None = None

        ack_kind = classify_ack(ack_no=ack_no, highest_ack=highest_before)

        if ack_kind == ACK_NEW:
            highest_ack = ack_no
            dup_ack_count = 0
            if in_fast_retransmit:
                cwnd = ssthresh
                in_fast_retransmit = False
                note = "new ACK advances cumulative ACK and closes fast-retransmit episode"
            else:
                note = "new cumulative ACK advances send window"

        elif ack_kind == ACK_DUP:
            dup_ack_count += 1
            if dup_ack_count == 3:
                retransmissions += 1
                retransmitted_seq = highest_before
                cwnd, ssthresh = on_triple_duplicate_ack(cwnd=cwnd_before)
                in_fast_retransmit = True
                note = "third duplicate ACK: retransmit missing segment immediately"
            elif in_fast_retransmit:
                cwnd = cwnd_before + 1
                note = "extra duplicate ACK while retransmitting: keep pipe non-empty"
            else:
                note = f"duplicate ACK count={dup_ack_count}, waiting for fast retransmit threshold"

        else:
            note = "stale ACK (below highest cumulative ACK), ignored"

        records.append(
            StepRecord(
                step=step,
                ack_no=ack_no,
                ack_kind=ack_kind,
                highest_ack_before=highest_before,
                highest_ack_after=highest_ack,
                dup_ack_count_before=dup_before,
                dup_ack_count_after=dup_ack_count,
                cwnd_before=cwnd_before,
                cwnd_after=cwnd,
                ssthresh_before=ssthresh_before,
                ssthresh_after=ssthresh,
                in_fast_retransmit_before=fast_before,
                in_fast_retransmit_after=in_fast_retransmit,
                retransmissions_before=retrans_before,
                retransmissions_after=retransmissions,
                retransmitted_seq=retransmitted_seq,
                note=note,
            )
        )

    return records


def extract_series(
    records: Sequence[StepRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract numpy series for deterministic checks."""
    highest_after = np.array([r.highest_ack_after for r in records], dtype=int)
    dup_after = np.array([r.dup_ack_count_after for r in records], dtype=int)
    cwnd_after = np.array([r.cwnd_after for r in records], dtype=int)
    ssthresh_after = np.array([r.ssthresh_after for r in records], dtype=int)
    fast_after = np.array([r.in_fast_retransmit_after for r in records], dtype=bool)
    retrans_after = np.array([r.retransmissions_after for r in records], dtype=int)
    return highest_after, dup_after, cwnd_after, ssthresh_after, fast_after, retrans_after


def records_to_dataframe(records: Sequence[StepRecord]) -> pd.DataFrame:
    """Convert simulation records into a readable table."""
    return pd.DataFrame(
        {
            "step": [r.step for r in records],
            "ack_no": [r.ack_no for r in records],
            "ack_kind": [r.ack_kind for r in records],
            "highest_ack": [f"{r.highest_ack_before}->{r.highest_ack_after}" for r in records],
            "dup_ack_count": [f"{r.dup_ack_count_before}->{r.dup_ack_count_after}" for r in records],
            "cwnd": [f"{r.cwnd_before}->{r.cwnd_after}" for r in records],
            "ssthresh": [f"{r.ssthresh_before}->{r.ssthresh_after}" for r in records],
            "in_fast_retx": [f"{int(r.in_fast_retransmit_before)}->{int(r.in_fast_retransmit_after)}" for r in records],
            "retransmitted_seq": [r.retransmitted_seq for r in records],
            "retransmissions": [f"{r.retransmissions_before}->{r.retransmissions_after}" for r in records],
            "note": [r.note for r in records],
        }
    )


def run_demo() -> None:
    """Run deterministic fast retransmit demo and assertions."""
    print("=== TCP Fast Retransmit MVP Demo ===")

    init_cwnd = 10
    init_ssthresh = 16
    initial_highest_ack = 1
    ack_stream = [2, 3, 3, 3, 3, 3, 7, 8, 8, 8, 8, 8, 12, 11, 13]

    records = simulate_tcp_fast_retransmit(
        init_cwnd=init_cwnd,
        init_ssthresh=init_ssthresh,
        initial_highest_ack=initial_highest_ack,
        ack_stream=ack_stream,
    )

    highest_after, dup_after, cwnd_after, ssthresh_after, fast_after, retrans_after = extract_series(records)

    expected_highest_after = np.array([2, 3, 3, 3, 3, 3, 7, 8, 8, 8, 8, 8, 12, 12, 13], dtype=int)
    expected_dup_after = np.array([0, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0, 0, 0], dtype=int)
    expected_cwnd_after = np.array([10, 10, 10, 10, 8, 9, 5, 5, 5, 5, 5, 6, 2, 2, 2], dtype=int)
    expected_ssthresh_after = np.array([16, 16, 16, 16, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2], dtype=int)
    expected_fast_after = np.array(
        [False, False, False, False, True, True, False, False, False, False, True, True, False, False, False],
        dtype=bool,
    )
    expected_retrans_after = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=int)

    if not np.array_equal(highest_after, expected_highest_after):
        raise AssertionError(f"unexpected highest_ack sequence: {highest_after}")
    if not np.array_equal(dup_after, expected_dup_after):
        raise AssertionError(f"unexpected dup_ack_count sequence: {dup_after}")
    if not np.array_equal(cwnd_after, expected_cwnd_after):
        raise AssertionError(f"unexpected cwnd sequence: {cwnd_after}")
    if not np.array_equal(ssthresh_after, expected_ssthresh_after):
        raise AssertionError(f"unexpected ssthresh sequence: {ssthresh_after}")
    if not np.array_equal(fast_after, expected_fast_after):
        raise AssertionError(f"unexpected in_fast_retransmit sequence: {fast_after}")
    if not np.array_equal(retrans_after, expected_retrans_after):
        raise AssertionError(f"unexpected retransmissions sequence: {retrans_after}")

    trigger_steps = [r.step for r in records if r.retransmitted_seq is not None]
    trigger_seqs = [r.retransmitted_seq for r in records if r.retransmitted_seq is not None]

    if trigger_steps != [5, 11]:
        raise AssertionError(f"unexpected fast-retransmit steps: {trigger_steps}")
    if trigger_seqs != [3, 8]:
        raise AssertionError(f"unexpected retransmitted sequence numbers: {trigger_seqs}")

    for rec in records:
        if rec.retransmitted_seq is not None:
            if rec.ack_kind != ACK_DUP or rec.dup_ack_count_after != 3:
                raise AssertionError(f"step {rec.step}: retransmit must be triggered by third duplicate ACK")
            if rec.retransmissions_after != rec.retransmissions_before + 1:
                raise AssertionError(f"step {rec.step}: retransmission counter must increase by 1")

        if rec.ack_kind == ACK_STALE and rec.retransmitted_seq is not None:
            raise AssertionError(f"step {rec.step}: stale ACK must not trigger retransmit")

        if rec.in_fast_retransmit_before and rec.ack_kind == ACK_NEW:
            if rec.cwnd_after != rec.ssthresh_before:
                raise AssertionError(f"step {rec.step}: new ACK after retransmit should deflate cwnd to ssthresh")

    table = records_to_dataframe(records)
    print("Step timeline:")
    with pd.option_context("display.max_colwidth", 60, "display.width", 170):
        print(table.to_string(index=False))

    print("highest_ack_after sequence:", highest_after.tolist())
    print("dup_ack_count_after sequence:", dup_after.tolist())
    print("cwnd_after sequence:", cwnd_after.tolist())
    print("ssthresh_after sequence:", ssthresh_after.tolist())
    print("in_fast_retransmit_after sequence:", fast_after.astype(int).tolist())
    print("retransmissions_after sequence:", retrans_after.tolist())
    print("fast_retransmit trigger steps:", trigger_steps)
    print("fast_retransmit retransmitted_seq:", trigger_seqs)
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
