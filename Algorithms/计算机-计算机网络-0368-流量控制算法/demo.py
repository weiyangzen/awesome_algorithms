"""流量控制算法最小可运行示例（MVP）。

模型说明：
- 发送端根据接收端通告窗口（advertised window）进行发送。
- 链路按离散 RTT tick 前进，数据包在 1 个 tick 后到达接收端。
- 接收端有固定缓冲区，应用层按给定速率从缓冲区读取。

目标：演示流量控制如何避免接收缓冲区溢出，并观察吞吐与阻塞行为。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List


@dataclass
class TickRecord:
    tick: int
    sent: int
    acked: int
    unacked: int
    buffer_used: int
    advertised_window: int
    app_read: int
    blocked_by_flow_control: bool
    dropped: int


@dataclass
class SimulationResult:
    records: List[TickRecord]
    total_data: int
    total_sent: int
    total_acked: int
    total_dropped: int


class FlowControlSimulator:
    def __init__(
        self,
        total_data: int,
        link_capacity_per_tick: int,
        receiver_buffer_capacity: int,
        app_read_fn: Callable[[int], int],
        max_ticks: int = 200,
    ) -> None:
        self.total_data = total_data
        self.link_capacity_per_tick = link_capacity_per_tick
        self.receiver_buffer_capacity = receiver_buffer_capacity
        self.app_read_fn = app_read_fn
        self.max_ticks = max_ticks

    def run(self) -> SimulationResult:
        sent_total = 0
        acked_total = 0
        dropped_total = 0

        unacked_bytes = 0
        receiver_buffer_used = 0
        advertised_window = self.receiver_buffer_capacity

        # in_flight[t] 表示将在 tick=t 到达接收端的数据量
        in_flight: dict[int, int] = {}
        records: List[TickRecord] = []

        for tick in range(self.max_ticks):
            arriving = in_flight.pop(tick, 0)

            # 1) 接收端接收网络到达数据
            free_space = self.receiver_buffer_capacity - receiver_buffer_used
            accepted = min(arriving, free_space)
            dropped = arriving - accepted
            receiver_buffer_used += accepted
            dropped_total += dropped

            # 2) ACK 到达发送端（简化为同 tick 生效）
            acked_total += accepted
            unacked_bytes = max(0, unacked_bytes - accepted)

            # 3) 接收端应用读取缓冲数据
            app_read_budget = max(0, self.app_read_fn(tick))
            app_read = min(app_read_budget, receiver_buffer_used)
            receiver_buffer_used -= app_read

            # 4) 更新通告窗口
            advertised_window = self.receiver_buffer_capacity - receiver_buffer_used

            # 5) 发送端依据流量控制发送新数据
            remaining = self.total_data - sent_total
            window_allowance = max(0, advertised_window - unacked_bytes)
            send_now = min(self.link_capacity_per_tick, remaining, window_allowance)
            blocked = remaining > 0 and send_now == 0

            if send_now > 0:
                in_flight[tick + 1] = in_flight.get(tick + 1, 0) + send_now
                sent_total += send_now
                unacked_bytes += send_now

            records.append(
                TickRecord(
                    tick=tick,
                    sent=send_now,
                    acked=accepted,
                    unacked=unacked_bytes,
                    buffer_used=receiver_buffer_used,
                    advertised_window=advertised_window,
                    app_read=app_read,
                    blocked_by_flow_control=blocked,
                    dropped=dropped,
                )
            )

            # 结束条件：全部确认且网络在途为空
            if acked_total >= self.total_data and not in_flight:
                break

        return SimulationResult(
            records=records,
            total_data=self.total_data,
            total_sent=sent_total,
            total_acked=acked_total,
            total_dropped=dropped_total,
        )


def app_read_pattern(tick: int) -> int:
    """构造一个周期性读取速率，模拟接收应用忽快忽慢。"""
    pattern = [2000, 2000, 3000, 4000, 1500, 1000, 3500, 5000]
    return pattern[tick % len(pattern)]


def print_report(result: SimulationResult, link_capacity_per_tick: int) -> None:
    ticks = len(result.records)
    blocked_ticks = sum(1 for r in result.records if r.blocked_by_flow_control)
    utilization = 0.0
    if ticks > 0 and link_capacity_per_tick > 0:
        utilization = result.total_sent / (ticks * link_capacity_per_tick)

    print("=== Flow Control Simulation Summary ===")
    print(f"ticks: {ticks}")
    print(f"total_data: {result.total_data} bytes")
    print(f"total_sent: {result.total_sent} bytes")
    print(f"total_acked: {result.total_acked} bytes")
    print(f"total_dropped: {result.total_dropped} bytes")
    print(f"blocked_ticks_by_flow_control: {blocked_ticks}")
    print(f"link_utilization: {utilization:.2%}")

    print("\n--- First 15 ticks ---")
    print(
        "tick sent acked unacked buffer_used adv_win app_read blocked dropped"
    )
    for r in result.records[:15]:
        print(
            f"{r.tick:>4} {r.sent:>4} {r.acked:>5} {r.unacked:>7} "
            f"{r.buffer_used:>11} {r.advertised_window:>7} {r.app_read:>8} "
            f"{str(r.blocked_by_flow_control):>7} {r.dropped:>7}"
        )


def main() -> None:
    total_data = 120_000
    link_capacity_per_tick = 8_000
    receiver_buffer_capacity = 24_000

    simulator = FlowControlSimulator(
        total_data=total_data,
        link_capacity_per_tick=link_capacity_per_tick,
        receiver_buffer_capacity=receiver_buffer_capacity,
        app_read_fn=app_read_pattern,
        max_ticks=200,
    )

    result = simulator.run()
    print_report(result, link_capacity_per_tick)


if __name__ == "__main__":
    main()
