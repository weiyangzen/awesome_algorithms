"""TCP congestion avoidance (AIMD) minimal runnable MVP.

This demo focuses on the congestion-avoidance phase only:
- ACK path: additive increase (roughly +1 MSS per RTT)
- LOSS path: multiplicative decrease (halve cwnd)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoundRecord:
    rtt_index: int
    event: str
    cwnd_before: float
    cwnd_after: float
    ssthresh: float
    throughput_mbps: float


class CongestionAvoidanceSimulator:
    def __init__(
        self,
        *,
        initial_cwnd: float = 10.0,
        initial_ssthresh: float = 64.0,
        mss_bytes: int = 1460,
        rtt_ms: float = 100.0,
        total_rtts: int = 40,
        loss_rtts: set[int] | None = None,
    ) -> None:
        if initial_cwnd <= 0:
            raise ValueError("initial_cwnd must be positive.")
        if initial_ssthresh <= 0:
            raise ValueError("initial_ssthresh must be positive.")
        if rtt_ms <= 0:
            raise ValueError("rtt_ms must be positive.")
        if total_rtts <= 0:
            raise ValueError("total_rtts must be positive.")

        self.cwnd = float(initial_cwnd)
        self.ssthresh = float(initial_ssthresh)
        self.mss_bytes = int(mss_bytes)
        self.rtt_ms = float(rtt_ms)
        self.total_rtts = int(total_rtts)
        self.loss_rtts = set() if loss_rtts is None else set(loss_rtts)

    def _on_ack_round(self) -> None:
        ack_count = max(1, int(self.cwnd))
        for _ in range(ack_count):
            self.cwnd += 1.0 / self.cwnd

    def _on_loss(self) -> None:
        self.ssthresh = max(self.cwnd / 2.0, 2.0)
        self.cwnd = self.ssthresh

    def _estimate_throughput_mbps(self) -> float:
        bytes_per_rtt = self.cwnd * self.mss_bytes
        bits_per_second = bytes_per_rtt * 8.0 / (self.rtt_ms / 1000.0)
        return bits_per_second / 1_000_000.0

    def run(self) -> list[RoundRecord]:
        records: list[RoundRecord] = []
        for rtt in range(1, self.total_rtts + 1):
            cwnd_before = self.cwnd
            if rtt in self.loss_rtts:
                event = "LOSS"
                self._on_loss()
            else:
                event = "ACK"
                self._on_ack_round()
            records.append(
                RoundRecord(
                    rtt_index=rtt,
                    event=event,
                    cwnd_before=cwnd_before,
                    cwnd_after=self.cwnd,
                    ssthresh=self.ssthresh,
                    throughput_mbps=self._estimate_throughput_mbps(),
                )
            )
        return records


def render_table(records: list[RoundRecord]) -> str:
    header = "RTT | EVENT | CWND_BEFORE | CWND_AFTER | SSTHRESH | THROUGHPUT(Mbps)"
    line = "-" * len(header)
    rows = [header, line]
    for rec in records:
        rows.append(
            f"{rec.rtt_index:>3} | "
            f"{rec.event:^5} | "
            f"{rec.cwnd_before:>11.3f} | "
            f"{rec.cwnd_after:>10.3f} | "
            f"{rec.ssthresh:>8.3f} | "
            f"{rec.throughput_mbps:>16.3f}"
        )
    return "\n".join(rows)


def summarize(records: list[RoundRecord]) -> str:
    avg_cwnd = sum(r.cwnd_after for r in records) / len(records)
    max_cwnd = max(r.cwnd_after for r in records)
    min_cwnd = min(r.cwnd_after for r in records)
    loss_count = sum(1 for r in records if r.event == "LOSS")
    avg_tp = sum(r.throughput_mbps for r in records) / len(records)
    return (
        "Summary\n"
        f"- RTT rounds: {len(records)}\n"
        f"- Loss events: {loss_count}\n"
        f"- cwnd after RTT: avg={avg_cwnd:.3f}, min={min_cwnd:.3f}, max={max_cwnd:.3f}\n"
        f"- throughput: avg={avg_tp:.3f} Mbps"
    )


def main() -> None:
    simulator = CongestionAvoidanceSimulator(
        initial_cwnd=10.0,
        initial_ssthresh=64.0,
        mss_bytes=1460,
        rtt_ms=100.0,
        total_rtts=40,
        loss_rtts={8, 18, 29, 36},
    )
    records = simulator.run()
    print("TCP Congestion Avoidance (AIMD) Simulation")
    print(render_table(records))
    print()
    print(summarize(records))


if __name__ == "__main__":
    main()
