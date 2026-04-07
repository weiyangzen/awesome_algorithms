"""死锁检测算法 MVP（多实例资源，Work/Finish 迭代）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeadlockCase:
    """单个死锁检测输入快照。"""

    name: str
    processes: tuple[str, ...]
    resources: tuple[str, ...]
    allocation: tuple[tuple[int, ...], ...]
    request: tuple[tuple[int, ...], ...]
    available: tuple[int, ...]
    expected_deadlocked: tuple[str, ...]


@dataclass
class DetectionResult:
    """检测输出结果。"""

    case_name: str
    deadlocked_processes: list[str]
    progress_order: list[str]
    work_trace: list[np.ndarray]
    finish_mask: np.ndarray
    final_work: np.ndarray


def _to_matrix(
    rows: tuple[tuple[int, ...], ...], n: int, m: int, label: str
) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.int64)
    if arr.shape != (n, m):
        raise ValueError(f"{label} shape mismatch: expected {(n, m)}, got {arr.shape}")
    if np.any(arr < 0):
        raise ValueError(f"{label} contains negative values")
    return arr


def _to_vector(values: tuple[int, ...], m: int, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.shape != (m,):
        raise ValueError(f"{label} shape mismatch: expected {(m,)}, got {arr.shape}")
    if np.any(arr < 0):
        raise ValueError(f"{label} contains negative values")
    return arr


def validate_case(case: DeadlockCase) -> None:
    if len(set(case.processes)) != len(case.processes):
        raise ValueError(f"{case.name}: duplicated process names")
    if len(set(case.resources)) != len(case.resources):
        raise ValueError(f"{case.name}: duplicated resource names")

    unknown = set(case.expected_deadlocked) - set(case.processes)
    if unknown:
        raise ValueError(
            f"{case.name}: expected_deadlocked contains unknown processes: {sorted(unknown)}"
        )


def materialize_case(case: DeadlockCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_case(case)
    n = len(case.processes)
    m = len(case.resources)

    allocation = _to_matrix(case.allocation, n, m, "allocation")
    request = _to_matrix(case.request, n, m, "request")
    available = _to_vector(case.available, m, "available")
    return allocation, request, available


def detect_deadlock(case: DeadlockCase) -> DetectionResult:
    """执行 Work/Finish 死锁检测，返回死锁进程集合与轨迹。"""
    allocation, request, available = materialize_case(case)
    n = len(case.processes)

    work = available.copy()
    finish = np.all(allocation == 0, axis=1)
    progress_order: list[str] = []
    work_trace: list[np.ndarray] = [work.copy()]

    while True:
        progressed = False
        for i in range(n):
            if finish[i]:
                continue
            if np.all(request[i] <= work):
                work += allocation[i]
                finish[i] = True
                progress_order.append(case.processes[i])
                work_trace.append(work.copy())
                progressed = True
        if not progressed:
            break

    deadlocked_processes = [
        case.processes[i] for i in range(n) if not bool(finish[i])
    ]
    return DetectionResult(
        case_name=case.name,
        deadlocked_processes=deadlocked_processes,
        progress_order=progress_order,
        work_trace=work_trace,
        finish_mask=finish.copy(),
        final_work=work.copy(),
    )


def _format_matrix(
    mat: np.ndarray, row_labels: tuple[str, ...], col_labels: tuple[str, ...]
) -> str:
    lines = ["          " + " ".join(f"{c:>4}" for c in col_labels)]
    for i, row in enumerate(mat):
        values = " ".join(f"{int(v):>4}" for v in row)
        lines.append(f"{row_labels[i]:>8} {values}")
    return "\n".join(lines)


def _format_vector(vec: np.ndarray, labels: tuple[str, ...]) -> str:
    return ", ".join(f"{name}={int(value)}" for name, value in zip(labels, vec))


def print_result(case: DeadlockCase, result: DetectionResult) -> None:
    allocation, request, available = materialize_case(case)
    print(f"\n=== {case.name} ===")
    print(f"Processes: {', '.join(case.processes)}")
    print(f"Resources: {', '.join(case.resources)}")
    print("Allocation:")
    print(_format_matrix(allocation, case.processes, case.resources))
    print("Request:")
    print(_format_matrix(request, case.processes, case.resources))
    print(f"Available(initial): {_format_vector(available, case.resources)}")

    print("Progress order:", " -> ".join(result.progress_order) or "None")
    for step, work in enumerate(result.work_trace):
        print(f"Work after step {step}: {_format_vector(work, case.resources)}")

    print("Deadlocked processes:", result.deadlocked_processes or "None")
    print(
        "Finish mask:",
        ", ".join(
            f"{case.processes[i]}={bool(result.finish_mask[i])}"
            for i in range(len(case.processes))
        ),
    )
    print(f"Expected deadlocked: {list(case.expected_deadlocked)}")


def build_demo_cases() -> list[DeadlockCase]:
    full_deadlock = DeadlockCase(
        name="Case-Full-Deadlock",
        processes=("P0", "P1", "P2"),
        resources=("A", "B", "C"),
        allocation=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        request=((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        available=(0, 0, 0),
        expected_deadlocked=("P0", "P1", "P2"),
    )
    no_deadlock = DeadlockCase(
        name="Case-No-Deadlock",
        processes=("P0", "P1", "P2"),
        resources=("A", "B", "C"),
        allocation=((1, 0, 0), (0, 1, 0), (0, 0, 0)),
        request=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        available=(0, 0, 1),
        expected_deadlocked=(),
    )
    partial_deadlock = DeadlockCase(
        name="Case-Partial-Deadlock",
        processes=("P0", "P1", "P2"),
        resources=("A", "B"),
        allocation=((1, 0), (0, 1), (0, 0)),
        request=((0, 1), (1, 0), (0, 0)),
        available=(0, 0),
        expected_deadlocked=("P0", "P1"),
    )
    return [full_deadlock, no_deadlock, partial_deadlock]


def main() -> None:
    print("Deadlock Detection Algorithm Demo (Work/Finish)")
    cases = build_demo_cases()
    results: list[DetectionResult] = []

    for case in cases:
        result = detect_deadlock(case)
        print_result(case, result)
        results.append(result)

        assert result.deadlocked_processes == list(case.expected_deadlocked), (
            f"{case.name}: expected {list(case.expected_deadlocked)}, "
            f"got {result.deadlocked_processes}"
        )

    deadlocked_case_count = sum(1 for r in results if r.deadlocked_processes)
    total_deadlocked_processes = sum(len(r.deadlocked_processes) for r in results)
    print("\n=== Summary ===")
    print(f"Cases: {len(results)}")
    print(f"Cases with deadlock: {deadlocked_case_count}")
    print(f"Total deadlocked processes: {total_deadlocked_processes}")
    print("All assertions passed.")


if __name__ == "__main__":
    main()
