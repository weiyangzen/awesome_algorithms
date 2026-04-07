"""滑动窗口协议（Go-Back-N）最小可运行 MVP。

运行方式：
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Dict, List


@dataclass
class Stats:
    """运行统计。"""

    sent_total: int = 0
    retransmissions: int = 0
    data_dropped: int = 0
    ack_dropped: int = 0
    data_accepted: int = 0
    out_of_order_seen: int = 0
    acks_received: int = 0
    timeouts: int = 0


@dataclass
class GoBackNSimulator:
    """离散时间 Go-Back-N 协议模拟器。"""

    total_frames: int
    window_size: int
    timeout_ticks: int
    propagation_delay: int
    data_loss_prob: float
    ack_loss_prob: float
    seed: int = 42

    base: int = 0
    next_seq: int = 0
    receiver_expected: int = 0
    tick: int = 0
    timer_start: int | None = None

    data_events: Dict[int, List[int]] = field(default_factory=dict)
    ack_events: Dict[int, List[int]] = field(default_factory=dict)
    attempts: Dict[int, int] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    stats: Stats = field(default_factory=Stats)

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)

    def _log(self, message: str) -> None:
        self.logs.append(message)

    def _schedule_event(self, bucket: Dict[int, List[int]], when: int, value: int) -> None:
        bucket.setdefault(when, []).append(value)

    def _send_data(self, seq: int, *, retransmit: bool) -> None:
        self.stats.sent_total += 1
        if retransmit:
            self.stats.retransmissions += 1

        self.attempts[seq] = self.attempts.get(seq, 0) + 1
        phase = "重传" if retransmit else "首次"

        if self.rng.random() < self.data_loss_prob:
            self.stats.data_dropped += 1
            self._log(
                f"T={self.tick:02d} 发送方{phase}发送 seq={seq}（尝试#{self.attempts[seq]}） -> DATA 丢失"
            )
            return

        arrival_tick = self.tick + self.propagation_delay
        self._schedule_event(self.data_events, arrival_tick, seq)
        self._log(
            f"T={self.tick:02d} 发送方{phase}发送 seq={seq}（尝试#{self.attempts[seq]}） -> "
            f"将在 T={arrival_tick:02d} 到达接收方"
        )

    def _process_data_arrivals(self) -> None:
        arrivals = sorted(self.data_events.pop(self.tick, []))
        for seq in arrivals:
            if seq == self.receiver_expected:
                self.receiver_expected += 1
                self.stats.data_accepted += 1
                ack = self.receiver_expected - 1
                self._log(
                    f"T={self.tick:02d} 接收方收到 seq={seq}，按序接收，累计 ACK={ack}"
                )
            else:
                self.stats.out_of_order_seen += 1
                ack = self.receiver_expected - 1
                self._log(
                    f"T={self.tick:02d} 接收方收到 seq={seq}，乱序丢弃，重复 ACK={ack}"
                )

            if self.rng.random() < self.ack_loss_prob:
                self.stats.ack_dropped += 1
                self._log(f"T={self.tick:02d} 接收方发出的 ACK={ack} 丢失")
            else:
                ack_arrival_tick = self.tick + self.propagation_delay
                self._schedule_event(self.ack_events, ack_arrival_tick, ack)
                self._log(
                    f"T={self.tick:02d} ACK={ack} 将在 T={ack_arrival_tick:02d} 到达发送方"
                )

    def _process_ack_arrivals(self) -> None:
        arrivals = sorted(self.ack_events.pop(self.tick, []))
        for ack in arrivals:
            self.stats.acks_received += 1
            if ack >= self.base:
                old_base = self.base
                self.base = ack + 1
                self._log(
                    f"T={self.tick:02d} 发送方收到 ACK={ack}，窗口左边界 {old_base} -> {self.base}"
                )
                if self.base == self.next_seq:
                    self.timer_start = None
                else:
                    self.timer_start = self.tick
            else:
                self._log(
                    f"T={self.tick:02d} 发送方收到 ACK={ack}（过期/重复），窗口保持 base={self.base}"
                )

    def _send_new_data_within_window(self) -> None:
        sent_now = 0
        while self.next_seq < self.total_frames and self.next_seq < self.base + self.window_size:
            self._send_data(self.next_seq, retransmit=False)
            self.next_seq += 1
            sent_now += 1

        if sent_now == 0:
            if self.next_seq >= self.total_frames:
                self._log(f"T={self.tick:02d} 无新数据可发（已发完全部帧）")
            else:
                self._log(
                    f"T={self.tick:02d} 发送窗口已满（base={self.base}, next={self.next_seq}, "
                    f"win={self.window_size}）"
                )

    def _maybe_timeout_and_retransmit(self) -> None:
        if self.base >= self.next_seq or self.timer_start is None:
            return
        if self.tick - self.timer_start < self.timeout_ticks:
            return

        self.stats.timeouts += 1
        self._log(
            f"T={self.tick:02d} !!! 超时触发：重传区间 [{self.base}, {self.next_seq - 1}]"
        )
        for seq in range(self.base, self.next_seq):
            self._send_data(seq, retransmit=True)
        self.timer_start = self.tick

    def run(self, max_ticks: int = 300) -> Stats:
        self._log(
            "配置: "
            f"total_frames={self.total_frames}, window_size={self.window_size}, "
            f"timeout_ticks={self.timeout_ticks}, propagation_delay={self.propagation_delay}, "
            f"data_loss_prob={self.data_loss_prob}, ack_loss_prob={self.ack_loss_prob}, seed={self.seed}"
        )

        while self.base < self.total_frames and self.tick < max_ticks:
            self._log(
                f"\n--- T={self.tick:02d} 状态: base={self.base}, next={self.next_seq}, "
                f"recv_expected={self.receiver_expected} ---"
            )

            self._process_data_arrivals()
            self._process_ack_arrivals()

            if self.base < self.next_seq and self.timer_start is None:
                self.timer_start = self.tick

            self._send_new_data_within_window()

            if self.base < self.next_seq and self.timer_start is None:
                self.timer_start = self.tick

            self._maybe_timeout_and_retransmit()

            self.tick += 1

        if self.base == self.total_frames:
            self._log(f"\n传输完成: T={self.tick:02d}，全部 {self.total_frames} 帧已确认")
        else:
            self._log(
                f"\n在最大 tick={max_ticks} 前未完成: base={self.base}, next={self.next_seq}"
            )

        return self.stats


def main() -> None:
    sim = GoBackNSimulator(
        total_frames=12,
        window_size=4,
        timeout_ticks=4,
        propagation_delay=1,
        data_loss_prob=0.25,
        ack_loss_prob=0.15,
        seed=7,
    )

    stats = sim.run(max_ticks=300)

    print("=" * 78)
    print("滑动窗口协议 (Go-Back-N) MVP 演示")
    print("=" * 78)
    for line in sim.logs:
        print(line)

    print("\n" + "=" * 78)
    print("统计汇总")
    print("=" * 78)
    print(f"发送总次数          : {stats.sent_total}")
    print(f"其中重传次数        : {stats.retransmissions}")
    print(f"数据包丢失次数      : {stats.data_dropped}")
    print(f"ACK 丢失次数        : {stats.ack_dropped}")
    print(f"接收方按序接收帧数  : {stats.data_accepted}")
    print(f"接收方看到乱序次数  : {stats.out_of_order_seen}")
    print(f"发送方收到 ACK 次数 : {stats.acks_received}")
    print(f"超时触发次数        : {stats.timeouts}")
    print(f"最终 base            : {sim.base}")
    print(f"最终 next_seq        : {sim.next_seq}")

    if sim.base != sim.total_frames:
        raise RuntimeError("模拟未完成，参数配置需要调整。")


if __name__ == "__main__":
    main()
