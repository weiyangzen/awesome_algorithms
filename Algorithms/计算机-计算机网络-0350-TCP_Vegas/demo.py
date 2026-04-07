"""Minimal runnable MVP for TCP Vegas (delay-based congestion control)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

STATE_SLOW_START = "SLOW_START"
STATE_CONGESTION_AVOIDANCE = "CONGESTION_AVOIDANCE"


@dataclass
class RoundRecord:
    round_id: int
    capacity_packets_per_rtt: float
    state_before: str
    state_after: str
    cwnd_before: float
    cwnd_after: float
    send_packets: float
    delivered_packets: float
    queue_packets: float
    rtt_ms: float
    base_rtt_ms: float
    expected_rate_packets_per_ms: float
    actual_rate_packets_per_ms: float
    diff_packets: float


@dataclass
class SimulationResult:
    records: List[RoundRecord]
    capacities: np.ndarray
    propagation_rtt_ms: float


class VegasController:
    """Compact educational TCP Vegas controller.

    Notes:
    - Uses queue-delay signal (`RTT - BaseRTT`) instead of packet loss.
    - This is a round-based teaching MVP, not a kernel-accurate TCP stack.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 3.0,
        gamma: float = 1.0,
        initial_cwnd_packets: float = 1.0,
        min_cwnd_packets: float = 1.0,
        max_cwnd_packets: float = 256.0,
    ) -> None:
        if alpha <= 0 or beta <= 0 or gamma <= 0:
            raise ValueError("alpha, beta, gamma must be positive")
        if alpha >= beta:
            raise ValueError("alpha must be smaller than beta")
        if initial_cwnd_packets <= 0:
            raise ValueError("initial_cwnd_packets must be positive")
        if min_cwnd_packets <= 0:
            raise ValueError("min_cwnd_packets must be positive")
        if max_cwnd_packets < min_cwnd_packets:
            raise ValueError("max_cwnd_packets must be >= min_cwnd_packets")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.min_cwnd_packets = float(min_cwnd_packets)
        self.max_cwnd_packets = float(max_cwnd_packets)

        self.state = STATE_SLOW_START
        self.cwnd_packets = float(initial_cwnd_packets)
        self.base_rtt_ms = float("inf")

    def observe_rtt(self, rtt_ms: float) -> float:
        if rtt_ms <= 0:
            raise ValueError("rtt_ms must be positive")
        self.base_rtt_ms = min(self.base_rtt_ms, float(rtt_ms))
        return self.base_rtt_ms

    def compute_expected_rate(self) -> float:
        if not np.isfinite(self.base_rtt_ms):
            raise ValueError("base RTT is not initialized")
        return self.cwnd_packets / self.base_rtt_ms

    def compute_diff_packets(self, rtt_ms: float) -> Tuple[float, float, float]:
        expected_rate = self.compute_expected_rate()

        # Vegas uses cwnd/RTT as actual throughput estimate for each RTT sample.
        actual_rate = self.cwnd_packets / max(rtt_ms, 1e-9)
        diff_packets = (expected_rate - actual_rate) * self.base_rtt_ms
        return expected_rate, actual_rate, diff_packets

    def update_window(self, diff_packets: float) -> None:
        if self.state == STATE_SLOW_START:
            if diff_packets > self.gamma:
                self.state = STATE_CONGESTION_AVOIDANCE
            else:
                self.cwnd_packets = min(self.max_cwnd_packets, self.cwnd_packets * 2.0)
            return

        if self.state == STATE_CONGESTION_AVOIDANCE:
            if diff_packets < self.alpha:
                self.cwnd_packets = min(self.max_cwnd_packets, self.cwnd_packets + 1.0)
            elif diff_packets > self.beta:
                self.cwnd_packets = max(self.min_cwnd_packets, self.cwnd_packets - 1.0)
            return

        raise ValueError(f"unknown Vegas state: {self.state}")


def validate_inputs(rounds: int, capacities: Sequence[float], propagation_rtt_ms: float) -> None:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if len(capacities) != rounds:
        raise ValueError("capacities length must equal rounds")
    if propagation_rtt_ms <= 0:
        raise ValueError("propagation_rtt_ms must be positive")
    if np.any(np.array(capacities, dtype=float) <= 0):
        raise ValueError("all capacities must be positive")


def build_capacity_schedule(rounds: int) -> np.ndarray:
    """Build a deterministic 3-phase bottleneck-capacity schedule."""
    if rounds < 6:
        raise ValueError("rounds must be at least 6 for the built-in schedule")

    caps = np.full(rounds, 30.0, dtype=float)
    seg1 = rounds // 3
    seg2 = (2 * rounds) // 3
    caps[seg1:seg2] = 20.0
    caps[seg2:] = 35.0
    return caps


def simulate_tcp_vegas(
    rounds: int,
    capacities: Sequence[float],
    propagation_rtt_ms: float = 50.0,
    alpha: float = 1.0,
    beta: float = 3.0,
    gamma: float = 1.0,
    initial_cwnd_packets: float = 1.0,
) -> SimulationResult:
    validate_inputs(rounds=rounds, capacities=capacities, propagation_rtt_ms=propagation_rtt_ms)

    controller = VegasController(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        initial_cwnd_packets=initial_cwnd_packets,
        min_cwnd_packets=1.0,
        max_cwnd_packets=256.0,
    )

    queue_packets = 0.0
    records: List[RoundRecord] = []

    for round_id in range(1, rounds + 1):
        capacity = float(capacities[round_id - 1])

        state_before = controller.state
        cwnd_before = controller.cwnd_packets
        send_packets = cwnd_before

        available_packets = queue_packets + send_packets
        delivered_packets = min(available_packets, capacity)
        queue_packets = max(0.0, available_packets - delivered_packets)

        queue_delay_ms = (queue_packets / max(capacity, 1e-9)) * propagation_rtt_ms
        rtt_ms = propagation_rtt_ms + queue_delay_ms

        base_rtt_ms = controller.observe_rtt(rtt_ms)
        expected_rate, actual_rate, diff_packets = controller.compute_diff_packets(rtt_ms=rtt_ms)
        controller.update_window(diff_packets=diff_packets)

        state_after = controller.state
        cwnd_after = controller.cwnd_packets

        records.append(
            RoundRecord(
                round_id=round_id,
                capacity_packets_per_rtt=capacity,
                state_before=state_before,
                state_after=state_after,
                cwnd_before=cwnd_before,
                cwnd_after=cwnd_after,
                send_packets=send_packets,
                delivered_packets=delivered_packets,
                queue_packets=queue_packets,
                rtt_ms=rtt_ms,
                base_rtt_ms=base_rtt_ms,
                expected_rate_packets_per_ms=expected_rate,
                actual_rate_packets_per_ms=actual_rate,
                diff_packets=diff_packets,
            )
        )

    return SimulationResult(
        records=records,
        capacities=np.array(capacities, dtype=float),
        propagation_rtt_ms=float(propagation_rtt_ms),
    )


def extract_series(records: Iterable[RoundRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    recs = list(records)
    delivered = np.array([r.delivered_packets for r in recs], dtype=float)
    queue = np.array([r.queue_packets for r in recs], dtype=float)
    rtt = np.array([r.rtt_ms for r in recs], dtype=float)
    diff = np.array([r.diff_packets for r in recs], dtype=float)
    states = np.array([r.state_after for r in recs], dtype=object)
    return delivered, queue, rtt, diff, states


def records_to_dataframe(records: Iterable[RoundRecord]) -> pd.DataFrame:
    recs = list(records)
    return pd.DataFrame(
        {
            "round": [r.round_id for r in recs],
            "capacity": [r.capacity_packets_per_rtt for r in recs],
            "state_before": [r.state_before for r in recs],
            "state_after": [r.state_after for r in recs],
            "cwnd_before": [r.cwnd_before for r in recs],
            "cwnd_after": [r.cwnd_after for r in recs],
            "delivered": [r.delivered_packets for r in recs],
            "queue": [r.queue_packets for r in recs],
            "rtt_ms": [r.rtt_ms for r in recs],
            "base_rtt_ms": [r.base_rtt_ms for r in recs],
            "diff_packets": [r.diff_packets for r in recs],
        }
    )


def run_checks(result: SimulationResult) -> None:
    delivered, queue, rtt, diff, states = extract_series(result.records)

    if STATE_SLOW_START not in states or STATE_CONGESTION_AVOIDANCE not in states:
        raise AssertionError("Vegas should visit both SLOW_START and CONGESTION_AVOIDANCE")

    early_avg = float(np.mean(delivered[6:14]))
    drop_avg = float(np.mean(delivered[20:30]))
    late_avg = float(np.mean(delivered[-8:]))

    if not (drop_avg < early_avg):
        raise AssertionError(f"expected drop_avg < early_avg, got {drop_avg:.3f} vs {early_avg:.3f}")
    if not (late_avg > drop_avg):
        raise AssertionError(f"expected late_avg > drop_avg, got {late_avg:.3f} vs {drop_avg:.3f}")

    if float(np.min(rtt)) < result.propagation_rtt_ms - 1e-9:
        raise AssertionError("RTT cannot be lower than propagation RTT in this queue model")

    if not np.isclose(float(np.min(rtt)), result.propagation_rtt_ms, atol=1e-9):
        raise AssertionError("base RTT sample should match propagation RTT in this deterministic setup")

    if float(np.max(queue)) <= 0.0:
        raise AssertionError("queue should be built at least once for Vegas delay sensing")

    cwnd_after = np.array([r.cwnd_after for r in result.records], dtype=float)
    if np.any(cwnd_after < 1.0) or np.any(cwnd_after > 256.0):
        raise AssertionError("cwnd leaves configured bounds [1, 256]")

    # After exiting slow start, Vegas should keep non-loss delay target around [alpha, beta] at least sometimes.
    in_target_band = np.sum((diff >= 1.0) & (diff <= 3.0))
    if in_target_band < 2:
        raise AssertionError(f"expected at least 2 rounds with diff in [alpha, beta], got {in_target_band}")


def print_report(result: SimulationResult) -> None:
    delivered, queue, rtt, diff, states = extract_series(result.records)
    df = records_to_dataframe(result.records)

    print("=== TCP Vegas MVP Simulation ===")
    print(f"rounds: {len(result.records)}")
    print(f"propagation_rtt_ms: {result.propagation_rtt_ms:.2f}")
    print(f"avg_delivered_packets: {np.mean(delivered):.2f}")
    print(f"max_queue_packets: {np.max(queue):.2f}")
    print(f"min_rtt_ms: {np.min(rtt):.2f}")
    print(f"max_rtt_ms: {np.max(rtt):.2f}")
    print(f"avg_diff_packets: {np.mean(diff):.2f}")
    print(f"states_seen: {sorted(set(states.tolist()))}")

    print("\n--- First 16 rounds ---")
    print(df.head(16).to_string(index=False))

    print("\n--- Last 8 rounds ---")
    print(df.tail(8).to_string(index=False))


def run_demo() -> None:
    rounds = 48
    propagation_rtt_ms = 50.0
    capacities = build_capacity_schedule(rounds=rounds)

    result = simulate_tcp_vegas(
        rounds=rounds,
        capacities=capacities,
        propagation_rtt_ms=propagation_rtt_ms,
        alpha=1.0,
        beta=3.0,
        gamma=1.0,
        initial_cwnd_packets=1.0,
    )
    run_checks(result)
    print_report(result)
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
