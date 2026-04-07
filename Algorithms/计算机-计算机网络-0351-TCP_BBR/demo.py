"""Minimal runnable MVP for TCP BBR (model-based congestion control)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Sequence, Tuple

import numpy as np

STATE_STARTUP = "STARTUP"
STATE_DRAIN = "DRAIN"
STATE_PROBE_BW = "PROBE_BW"
STATE_PROBE_RTT = "PROBE_RTT"


@dataclass
class RoundRecord:
    round_id: int
    capacity_packets_per_rtt: float
    state_before: str
    state_after: str
    pacing_gain: float
    pacing_rate_packets_per_ms: float
    send_packets: float
    delivered_packets: float
    queue_packets: float
    rtt_ms: float
    btlbw_packets_per_ms: float
    rtprop_ms: float
    bdp_packets: float
    cwnd_before: float
    cwnd_after: float


@dataclass
class SimulationResult:
    records: List[RoundRecord]
    base_rtt_ms: float
    capacities: np.ndarray


class BBRController:
    """A compact educational BBR-like controller.

    Notes:
    - This is an MVP approximation for teaching and reproducible simulation.
    - It keeps the core BBR ideas: model estimation (BtlBw/RTprop) + state machine.
    """

    def __init__(
        self,
        base_rtt_ms: float,
        startup_gain: float = 2.77,
        cwnd_gain: float = 2.0,
        bw_window: int = 8,
        rtprop_window_rounds: int = 20,
        probe_rtt_interval_rounds: int = 22,
        probe_rtt_duration_rounds: int = 2,
        min_cwnd_packets: float = 4.0,
    ) -> None:
        if base_rtt_ms <= 0:
            raise ValueError("base_rtt_ms must be positive")

        self.base_rtt_ms = float(base_rtt_ms)
        self.startup_gain = float(startup_gain)
        self.drain_gain = 1.0 / self.startup_gain
        self.cwnd_gain = float(cwnd_gain)
        self.bw_window = int(bw_window)
        self.rtprop_window_rounds = int(rtprop_window_rounds)
        self.probe_rtt_interval_rounds = int(probe_rtt_interval_rounds)
        self.probe_rtt_duration_rounds = int(probe_rtt_duration_rounds)
        self.min_cwnd_packets = float(min_cwnd_packets)

        self.probe_bw_gains = np.array([1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

        self.state = STATE_STARTUP
        self.probe_bw_index = 0
        self.probe_rtt_rounds_left = 0
        self.last_probe_rtt_round = 0

        self.bw_samples: Deque[float] = deque(maxlen=self.bw_window)
        self.rtt_samples: Deque[Tuple[int, float]] = deque()

        # Conservative initial model; updated quickly by ACK samples.
        self.btlbw_packets_per_ms = 0.50
        self.rtprop_ms = self.base_rtt_ms

        # Congestion window starts small and grows through ACK clocking.
        self.cwnd_packets = 8.0

        # Full pipe detection (BBR STARTUP exit logic approximation).
        self.full_bw = 0.0
        self.full_bw_count = 0
        self.full_bw_reached = False

    def bdp_packets(self) -> float:
        return max(self.min_cwnd_packets, self.btlbw_packets_per_ms * self.rtprop_ms)

    def current_pacing_gain(self) -> float:
        if self.state == STATE_STARTUP:
            return self.startup_gain
        if self.state == STATE_DRAIN:
            return self.drain_gain
        if self.state == STATE_PROBE_BW:
            return float(self.probe_bw_gains[self.probe_bw_index])
        if self.state == STATE_PROBE_RTT:
            return 1.0
        raise ValueError(f"unknown state: {self.state}")

    def pacing_rate_packets_per_ms(self) -> float:
        return max(1e-6, self.current_pacing_gain() * self.btlbw_packets_per_ms)

    def plan_send_packets(self) -> float:
        if self.state == STATE_PROBE_RTT:
            pacing_budget = self.min_cwnd_packets
        else:
            pacing_budget = max(
                self.min_cwnd_packets,
                self.pacing_rate_packets_per_ms() * self.rtprop_ms,
            )

        send_packets = min(self.cwnd_packets, pacing_budget)
        return max(1.0, send_packets)

    def maybe_enter_probe_rtt(self, round_id: int) -> None:
        should_enter = (
            self.state != STATE_PROBE_RTT
            and self.state != STATE_STARTUP
            and (round_id - self.last_probe_rtt_round) >= self.probe_rtt_interval_rounds
        )
        if should_enter:
            self.state = STATE_PROBE_RTT
            self.probe_rtt_rounds_left = self.probe_rtt_duration_rounds

    def _update_bandwidth_model(self, delivered_packets: float, rtt_ms: float) -> None:
        if delivered_packets <= 0:
            return
        sample_rate = delivered_packets / max(rtt_ms, 1e-6)
        self.bw_samples.append(sample_rate)
        self.btlbw_packets_per_ms = max(self.bw_samples)

    def _update_rtprop_model(self, round_id: int, rtt_ms: float) -> None:
        self.rtt_samples.append((round_id, rtt_ms))
        while self.rtt_samples and (round_id - self.rtt_samples[0][0]) >= self.rtprop_window_rounds:
            self.rtt_samples.popleft()
        self.rtprop_ms = min(sample for _, sample in self.rtt_samples)

    def _update_full_pipe_detection(self) -> None:
        if self.state != STATE_STARTUP:
            return

        if self.btlbw_packets_per_ms >= self.full_bw * 1.25:
            self.full_bw = self.btlbw_packets_per_ms
            self.full_bw_count = 0
            return

        self.full_bw_count += 1
        if self.full_bw_count >= 3:
            self.full_bw_reached = True
            self.state = STATE_DRAIN

    def _update_cwnd(self, delivered_packets: float) -> None:
        bdp = self.bdp_packets()
        target_cwnd = max(self.min_cwnd_packets, self.cwnd_gain * bdp)

        if self.state == STATE_PROBE_RTT:
            self.cwnd_packets = self.min_cwnd_packets
            return

        if self.state == STATE_STARTUP:
            # ACK clocked rapid growth until pipeline is judged full.
            self.cwnd_packets = min(target_cwnd * 2.0, self.cwnd_packets + max(delivered_packets, 1.0))
            return

        # DRAIN / PROBE_BW: converge to model target conservatively.
        self.cwnd_packets = min(target_cwnd, self.cwnd_packets + 0.5 * max(delivered_packets, 1.0))
        self.cwnd_packets = max(self.min_cwnd_packets, self.cwnd_packets)

    def _advance_states_after_ack(self, round_id: int, inflight_packets: float) -> None:
        bdp = self.bdp_packets()

        if self.state == STATE_DRAIN and inflight_packets <= bdp:
            self.state = STATE_PROBE_BW
            self.probe_bw_index = 0

        if self.state == STATE_PROBE_BW:
            self.probe_bw_index = (self.probe_bw_index + 1) % len(self.probe_bw_gains)

        if self.state == STATE_PROBE_RTT:
            self.probe_rtt_rounds_left -= 1
            if self.probe_rtt_rounds_left <= 0:
                self.state = STATE_PROBE_BW
                self.probe_bw_index = 0
                self.last_probe_rtt_round = round_id

    def on_round_end(
        self,
        round_id: int,
        delivered_packets: float,
        rtt_ms: float,
        inflight_packets: float,
    ) -> None:
        self._update_bandwidth_model(delivered_packets=delivered_packets, rtt_ms=rtt_ms)
        self._update_rtprop_model(round_id=round_id, rtt_ms=rtt_ms)
        self._update_full_pipe_detection()
        self._update_cwnd(delivered_packets=delivered_packets)
        self._advance_states_after_ack(round_id=round_id, inflight_packets=inflight_packets)


def validate_inputs(rounds: int, capacities: Sequence[float], base_rtt_ms: float) -> None:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if len(capacities) != rounds:
        raise ValueError("capacities length must equal rounds")
    if base_rtt_ms <= 0:
        raise ValueError("base_rtt_ms must be positive")
    if np.any(np.array(capacities, dtype=float) <= 0):
        raise ValueError("all capacities must be positive")


def build_capacity_schedule(rounds: int) -> np.ndarray:
    """Create a deterministic bottleneck-capacity schedule (packets per RTT)."""
    caps = np.full(rounds, 120.0, dtype=float)
    caps[18:30] = 80.0
    caps[30:] = 140.0
    return caps


def simulate_bbr(rounds: int, capacities: Sequence[float], base_rtt_ms: float = 50.0) -> SimulationResult:
    validate_inputs(rounds=rounds, capacities=capacities, base_rtt_ms=base_rtt_ms)

    controller = BBRController(base_rtt_ms=base_rtt_ms)
    queue_packets = 0.0
    records: List[RoundRecord] = []

    for round_id in range(1, rounds + 1):
        controller.maybe_enter_probe_rtt(round_id=round_id)

        state_before = controller.state
        cwnd_before = controller.cwnd_packets
        pacing_gain = controller.current_pacing_gain()
        pacing_rate = controller.pacing_rate_packets_per_ms()
        send_packets = controller.plan_send_packets()

        capacity = float(capacities[round_id - 1])

        available_packets = queue_packets + send_packets
        delivered_packets = min(available_packets, capacity)
        queue_packets = max(0.0, available_packets - delivered_packets)

        queue_delay_ms = (queue_packets / max(capacity, 1e-6)) * base_rtt_ms
        rtt_ms = base_rtt_ms + queue_delay_ms

        controller.on_round_end(
            round_id=round_id,
            delivered_packets=delivered_packets,
            rtt_ms=rtt_ms,
            inflight_packets=queue_packets,
        )

        state_after = controller.state
        bdp_packets = controller.bdp_packets()
        cwnd_after = controller.cwnd_packets

        records.append(
            RoundRecord(
                round_id=round_id,
                capacity_packets_per_rtt=capacity,
                state_before=state_before,
                state_after=state_after,
                pacing_gain=pacing_gain,
                pacing_rate_packets_per_ms=pacing_rate,
                send_packets=send_packets,
                delivered_packets=delivered_packets,
                queue_packets=queue_packets,
                rtt_ms=rtt_ms,
                btlbw_packets_per_ms=controller.btlbw_packets_per_ms,
                rtprop_ms=controller.rtprop_ms,
                bdp_packets=bdp_packets,
                cwnd_before=cwnd_before,
                cwnd_after=cwnd_after,
            )
        )

    return SimulationResult(records=records, base_rtt_ms=base_rtt_ms, capacities=np.array(capacities, dtype=float))


def extract_series(records: Iterable[RoundRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    recs = list(records)
    delivered = np.array([r.delivered_packets for r in recs], dtype=float)
    queue = np.array([r.queue_packets for r in recs], dtype=float)
    rtt = np.array([r.rtt_ms for r in recs], dtype=float)
    states = np.array([r.state_after for r in recs], dtype=object)
    return delivered, queue, rtt, states


def run_checks(result: SimulationResult) -> None:
    delivered, queue, rtt, states = extract_series(result.records)

    state_set = set(states.tolist())
    required_states = {STATE_STARTUP, STATE_DRAIN, STATE_PROBE_BW, STATE_PROBE_RTT}
    if not required_states.issubset(state_set):
        raise AssertionError(f"missing BBR states: required={required_states}, got={state_set}")

    # Capacity drop interval (rounds 19-30) should reduce throughput versus early rounds.
    early_avg = float(np.mean(delivered[6:16]))
    drop_avg = float(np.mean(delivered[20:30]))
    recover_avg = float(np.mean(delivered[34:44]))

    if not (drop_avg < early_avg):
        raise AssertionError(f"expected lower throughput during drop: early={early_avg:.2f}, drop={drop_avg:.2f}")
    if not (recover_avg > drop_avg):
        raise AssertionError(f"expected throughput recovery: drop={drop_avg:.2f}, recover={recover_avg:.2f}")

    if float(np.min(rtt)) < result.base_rtt_ms - 1e-9:
        raise AssertionError("RTT sample cannot be lower than base RTT in this queue model")

    if float(np.max(queue)) <= 0.0:
        raise AssertionError("queue should build at least once during probing")


def print_report(result: SimulationResult) -> None:
    delivered, queue, rtt, states = extract_series(result.records)

    print("=== TCP BBR MVP Simulation ===")
    print(f"rounds: {len(result.records)}")
    print(f"base_rtt_ms: {result.base_rtt_ms:.2f}")
    print(f"avg_delivered_packets: {np.mean(delivered):.2f}")
    print(f"max_queue_packets: {np.max(queue):.2f}")
    print(f"min_rtt_ms: {np.min(rtt):.2f}")
    print(f"max_rtt_ms: {np.max(rtt):.2f}")
    print(f"states_seen: {sorted(set(states.tolist()))}")

    print("\n--- First 18 rounds ---")
    print("rnd cap state_bef->aft gain send delivered queue rtt btlbw rtprop bdp cwnd_bef->aft")
    for rec in result.records[:18]:
        print(
            f"{rec.round_id:>3} {rec.capacity_packets_per_rtt:>4.0f} "
            f"{rec.state_before:>9s}->{rec.state_after:<9s} "
            f"{rec.pacing_gain:>4.2f} {rec.send_packets:>6.1f} {rec.delivered_packets:>8.1f} "
            f"{rec.queue_packets:>6.1f} {rec.rtt_ms:>6.2f} {rec.btlbw_packets_per_ms:>5.2f} "
            f"{rec.rtprop_ms:>6.2f} {rec.bdp_packets:>6.1f} {rec.cwnd_before:>6.1f}->{rec.cwnd_after:<6.1f}"
        )

    print("\n--- Last 8 rounds ---")
    for rec in result.records[-8:]:
        print(
            f"{rec.round_id:>3} {rec.capacity_packets_per_rtt:>4.0f} "
            f"{rec.state_before:>9s}->{rec.state_after:<9s} "
            f"{rec.pacing_gain:>4.2f} {rec.send_packets:>6.1f} {rec.delivered_packets:>8.1f} "
            f"{rec.queue_packets:>6.1f} {rec.rtt_ms:>6.2f} {rec.btlbw_packets_per_ms:>5.2f} "
            f"{rec.rtprop_ms:>6.2f} {rec.bdp_packets:>6.1f} {rec.cwnd_before:>6.1f}->{rec.cwnd_after:<6.1f}"
        )


def run_demo() -> None:
    rounds = 48
    base_rtt_ms = 50.0
    capacities = build_capacity_schedule(rounds=rounds)

    result = simulate_bbr(rounds=rounds, capacities=capacities, base_rtt_ms=base_rtt_ms)
    run_checks(result)
    print_report(result)
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
