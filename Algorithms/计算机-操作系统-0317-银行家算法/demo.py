"""银行家算法（Banker's Algorithm）最小可运行 MVP。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyCheckResult:
    """安全性检测结果。"""

    is_safe: bool
    safe_sequence: list[int]
    work_trace: list[list[int]]
    pending_processes: list[int]


@dataclass
class RequestResult:
    """资源请求处理结果。"""

    granted: bool
    reason: str
    safety_check: SafetyCheckResult | None


class Banker:
    """银行家算法核心实现。"""

    def __init__(
        self,
        available: np.ndarray,
        max_demand: np.ndarray,
        allocation: np.ndarray,
    ) -> None:
        self.available = np.asarray(available, dtype=int).copy()
        self.max_demand = np.asarray(max_demand, dtype=int).copy()
        self.allocation = np.asarray(allocation, dtype=int).copy()

        if self.available.ndim != 1:
            raise ValueError("available must be a 1-D array")
        if self.max_demand.ndim != 2 or self.allocation.ndim != 2:
            raise ValueError("max_demand and allocation must be 2-D arrays")
        if self.max_demand.shape != self.allocation.shape:
            raise ValueError("max_demand and allocation must have the same shape")

        self.n_processes, self.n_resources = self.max_demand.shape
        if self.available.shape[0] != self.n_resources:
            raise ValueError("available length must equal resource dimension")
        if np.any(self.available < 0):
            raise ValueError("available must be non-negative")
        if np.any(self.max_demand < 0) or np.any(self.allocation < 0):
            raise ValueError("max_demand/allocation must be non-negative")
        if np.any(self.allocation > self.max_demand):
            raise ValueError("allocation cannot exceed max_demand")

        self.need = self.max_demand - self.allocation

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回当前状态快照，用于回归断言。"""
        return self.available.copy(), self.allocation.copy(), self.need.copy()

    def check_safety(self) -> SafetyCheckResult:
        """执行安全性检测，返回安全序列和工作向量轨迹。"""
        work = self.available.copy()
        finish = np.zeros(self.n_processes, dtype=bool)
        safe_sequence: list[int] = []
        work_trace: list[list[int]] = [work.tolist()]

        while len(safe_sequence) < self.n_processes:
            progress = False
            for pid in range(self.n_processes):
                if finish[pid]:
                    continue
                if np.all(self.need[pid] <= work):
                    work = work + self.allocation[pid]
                    finish[pid] = True
                    safe_sequence.append(pid)
                    work_trace.append(work.tolist())
                    progress = True

            if not progress:
                break

        pending = [pid for pid in range(self.n_processes) if not finish[pid]]
        return SafetyCheckResult(
            is_safe=len(safe_sequence) == self.n_processes,
            safe_sequence=safe_sequence,
            work_trace=work_trace,
            pending_processes=pending,
        )

    def request(self, pid: int, request_vector: np.ndarray) -> RequestResult:
        """处理单次资源请求。"""
        if not (0 <= pid < self.n_processes):
            raise ValueError(f"pid out of range: {pid}")

        request_arr = np.asarray(request_vector, dtype=int)
        if request_arr.shape != (self.n_resources,):
            raise ValueError(
                f"request shape must be ({self.n_resources},), got {request_arr.shape}"
            )
        if np.any(request_arr < 0):
            raise ValueError("request must be non-negative")

        if np.any(request_arr > self.need[pid]):
            return RequestResult(
                granted=False,
                reason="请求超过该进程剩余需求（request > need）",
                safety_check=None,
            )
        if np.any(request_arr > self.available):
            return RequestResult(
                granted=False,
                reason="系统可用资源不足（request > available）",
                safety_check=None,
            )

        self.available -= request_arr
        self.allocation[pid] += request_arr
        self.need[pid] -= request_arr

        safety = self.check_safety()
        if safety.is_safe:
            return RequestResult(
                granted=True,
                reason=f"请求已批准，安全序列示例: {safety.safe_sequence}",
                safety_check=safety,
            )

        # 试分配后不安全，回滚。
        self.available += request_arr
        self.allocation[pid] -= request_arr
        self.need[pid] += request_arr
        return RequestResult(
            granted=False,
            reason="试分配会进入不安全状态，已回滚",
            safety_check=safety,
        )


def _assert_snapshot_equal(
    lhs: tuple[np.ndarray, np.ndarray, np.ndarray],
    rhs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    for left, right in zip(lhs, rhs, strict=True):
        assert np.array_equal(left, right), f"state mismatch:\n{left}\n!=\n{right}"


def run_demo() -> None:
    """运行教材级 5 进程/3 资源案例并进行断言验证。"""
    available = np.array([3, 3, 2], dtype=int)
    max_demand = np.array(
        [
            [7, 5, 3],
            [3, 2, 2],
            [9, 0, 2],
            [2, 2, 2],
            [4, 3, 3],
        ],
        dtype=int,
    )
    allocation = np.array(
        [
            [0, 1, 0],
            [2, 0, 0],
            [3, 0, 2],
            [2, 1, 1],
            [0, 0, 2],
        ],
        dtype=int,
    )

    banker = Banker(available=available, max_demand=max_demand, allocation=allocation)

    print("=== Banker's Algorithm MVP ===")
    print("Initial available:", banker.available.tolist())
    print("Initial need matrix:\n", banker.need)

    initial_safety = banker.check_safety()
    assert initial_safety.is_safe, "initial state should be safe"
    print("Initial safe sequence:", initial_safety.safe_sequence)
    print("Initial work trace:", initial_safety.work_trace)

    # 场景 1：教材常见可批准请求（P1 请求 [1,0,2]）。
    req1 = np.array([1, 0, 2], dtype=int)
    res1 = banker.request(pid=1, request_vector=req1)
    assert res1.granted, f"expected granted request, got: {res1.reason}"
    assert np.array_equal(banker.allocation[1], np.array([3, 0, 2], dtype=int))
    assert np.array_equal(banker.need[1], np.array([0, 2, 0], dtype=int))
    print("Request #1 (P1, [1,0,2]):", res1.reason)

    # 场景 2：请求大于 available，直接拒绝且不改状态。
    snap_before_insufficient = banker.snapshot()
    req2 = np.array([3, 0, 0], dtype=int)  # 当前 available[0] 仅为 2
    res2 = banker.request(pid=4, request_vector=req2)
    assert not res2.granted
    assert "available" in res2.reason
    _assert_snapshot_equal(snap_before_insufficient, banker.snapshot())
    print("Request #2 (P4, [3,0,0]):", res2.reason)

    # 场景 3：请求不超过 available，但会导致不安全，必须回滚拒绝。
    snap_before_unsafe = banker.snapshot()
    req3 = np.array([0, 2, 0], dtype=int)
    res3 = banker.request(pid=0, request_vector=req3)
    assert not res3.granted
    assert res3.safety_check is not None
    assert not res3.safety_check.is_safe
    _assert_snapshot_equal(snap_before_unsafe, banker.snapshot())
    print("Request #3 (P0, [0,2,0]):", res3.reason)
    print("Unsafe check pending processes:", res3.safety_check.pending_processes)

    final_safety = banker.check_safety()
    assert final_safety.is_safe, "final state should remain safe after rejected unsafe request"
    print("Final safe sequence:", final_safety.safe_sequence)
    print("Final available:", banker.available.tolist())
    print("All assertions passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
