"""Minimal runnable MVP for TCP slow start congestion control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np

PHASE_SLOW_START = "slow_start"
PHASE_CONGESTION_AVOIDANCE = "congestion_avoidance"


@dataclass
class RTTRecord:
    rtt: int
    event: str
    phase_before: str
    phase_after: str
    cwnd_before: int
    cwnd_after: int
    ssthresh_before: int
    ssthresh_after: int


def validate_inputs(
    init_cwnd: int,
    init_ssthresh: int,
    total_rtts: int,
    loss_rtts: Iterable[int],
) -> Set[int]:
    """Validate simulation inputs and return normalized loss RTT set."""
    if init_cwnd <= 0:
        raise ValueError("init_cwnd must be positive")
    if init_ssthresh < 2:
        raise ValueError("init_ssthresh must be >= 2")
    if total_rtts <= 0:
        raise ValueError("total_rtts must be positive")

    normalized = {int(x) for x in loss_rtts}
    for rtt in normalized:
        if not (1 <= rtt <= total_rtts):
            raise ValueError(f"loss RTT {rtt} out of valid range [1, {total_rtts}]")

    return normalized


def next_window_without_loss(cwnd: int, ssthresh: int, phase: str) -> Tuple[int, str]:
    """Evolve congestion window for one RTT when no loss occurs."""
    if phase == PHASE_SLOW_START:
        next_cwnd = min(cwnd * 2, ssthresh)
        next_phase = PHASE_CONGESTION_AVOIDANCE if next_cwnd >= ssthresh else PHASE_SLOW_START
        return next_cwnd, next_phase

    if phase == PHASE_CONGESTION_AVOIDANCE:
        return cwnd + 1, PHASE_CONGESTION_AVOIDANCE

    raise ValueError(f"unknown phase: {phase}")


def apply_loss_reaction(cwnd: int) -> Tuple[int, int, str]:
    """Apply Tahoe-style loss reaction: halve threshold and reset cwnd to 1."""
    new_ssthresh = max(cwnd // 2, 2)
    return 1, new_ssthresh, PHASE_SLOW_START


def simulate_tcp_slow_start(
    init_cwnd: int,
    init_ssthresh: int,
    total_rtts: int,
    loss_rtts: Iterable[int],
) -> List[RTTRecord]:
    """Simulate TCP slow start and basic congestion avoidance over RTT steps."""
    loss_set = validate_inputs(init_cwnd, init_ssthresh, total_rtts, loss_rtts)

    cwnd = int(init_cwnd)
    ssthresh = int(init_ssthresh)
    phase = PHASE_SLOW_START if cwnd < ssthresh else PHASE_CONGESTION_AVOIDANCE

    records: List[RTTRecord] = []

    for rtt in range(1, total_rtts + 1):
        cwnd_before = cwnd
        ssthresh_before = ssthresh
        phase_before = phase

        if rtt in loss_set:
            event = "loss"
            cwnd, ssthresh, phase = apply_loss_reaction(cwnd_before)
        else:
            event = "ack"
            cwnd, phase = next_window_without_loss(cwnd_before, ssthresh_before, phase_before)

        records.append(
            RTTRecord(
                rtt=rtt,
                event=event,
                phase_before=phase_before,
                phase_after=phase,
                cwnd_before=cwnd_before,
                cwnd_after=cwnd,
                ssthresh_before=ssthresh_before,
                ssthresh_after=ssthresh,
            )
        )

    return records


def extract_series(records: Sequence[RTTRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract numpy series from records for checks and display."""
    cwnd_after = np.array([r.cwnd_after for r in records], dtype=int)
    ssthresh_after = np.array([r.ssthresh_after for r in records], dtype=int)
    phase_after = np.array([r.phase_after for r in records], dtype=object)
    return cwnd_after, ssthresh_after, phase_after


def theoretical_slow_start_curve(init_cwnd: int, ssthresh: int, steps: int) -> np.ndarray:
    """Closed-form style curve for no-loss slow start under RTT discretization."""
    if steps < 0:
        raise ValueError("steps must be >= 0")

    curve: List[int] = []
    cwnd = int(init_cwnd)
    for _ in range(steps):
        cwnd = min(cwnd * 2, int(ssthresh))
        curve.append(cwnd)

    return np.array(curve, dtype=int)


def run_demo() -> None:
    """Run a deterministic TCP slow-start demo with assertions."""
    print("=== TCP Slow Start MVP Demo ===")

    init_cwnd = 1
    init_ssthresh = 16
    total_rtts = 12
    loss_rtts = {6, 11}

    records = simulate_tcp_slow_start(
        init_cwnd=init_cwnd,
        init_ssthresh=init_ssthresh,
        total_rtts=total_rtts,
        loss_rtts=loss_rtts,
    )

    cwnd_after, ssthresh_after, phase_after = extract_series(records)

    expected_cwnd_after = np.array([2, 4, 8, 16, 17, 1, 2, 4, 8, 9, 1, 2], dtype=int)
    expected_ssthresh_after = np.array([16, 16, 16, 16, 16, 8, 8, 8, 8, 8, 4, 4], dtype=int)
    expected_phase_after = np.array(
        [
            PHASE_SLOW_START,
            PHASE_SLOW_START,
            PHASE_SLOW_START,
            PHASE_CONGESTION_AVOIDANCE,
            PHASE_CONGESTION_AVOIDANCE,
            PHASE_SLOW_START,
            PHASE_SLOW_START,
            PHASE_SLOW_START,
            PHASE_CONGESTION_AVOIDANCE,
            PHASE_CONGESTION_AVOIDANCE,
            PHASE_SLOW_START,
            PHASE_SLOW_START,
        ],
        dtype=object,
    )

    if not np.array_equal(cwnd_after, expected_cwnd_after):
        raise AssertionError(f"unexpected cwnd sequence: {cwnd_after}")
    if not np.array_equal(ssthresh_after, expected_ssthresh_after):
        raise AssertionError(f"unexpected ssthresh sequence: {ssthresh_after}")
    if not np.array_equal(phase_after, expected_phase_after):
        raise AssertionError(f"unexpected phase sequence: {phase_after}")

    for rec in records:
        if rec.event == "loss":
            expected_threshold = max(rec.cwnd_before // 2, 2)
            if rec.cwnd_after != 1:
                raise AssertionError(f"RTT {rec.rtt}: loss must reset cwnd to 1")
            if rec.ssthresh_after != expected_threshold:
                raise AssertionError(
                    f"RTT {rec.rtt}: wrong ssthresh after loss, got {rec.ssthresh_after}, expected {expected_threshold}"
                )

    no_loss_records = simulate_tcp_slow_start(
        init_cwnd=1,
        init_ssthresh=16,
        total_rtts=4,
        loss_rtts=set(),
    )
    no_loss_cwnd, _, _ = extract_series(no_loss_records)
    no_loss_theory = theoretical_slow_start_curve(init_cwnd=1, ssthresh=16, steps=4)
    if not np.array_equal(no_loss_cwnd, no_loss_theory):
        raise AssertionError(
            f"no-loss curve mismatch: simulated={no_loss_cwnd}, theoretical={no_loss_theory}"
        )

    print("RTT timeline:")
    for rec in records:
        print(
            f"  rtt={rec.rtt:2d} event={rec.event:4s} "
            f"phase={rec.phase_before:20s}->{rec.phase_after:20s} "
            f"cwnd={rec.cwnd_before:2d}->{rec.cwnd_after:2d} "
            f"ssthresh={rec.ssthresh_before:2d}->{rec.ssthresh_after:2d}"
        )

    print("cwnd_after sequence:", cwnd_after.tolist())
    print("ssthresh_after sequence:", ssthresh_after.tolist())
    print("phase_after sequence:", phase_after.tolist())
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
