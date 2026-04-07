"""MVCC (Multi-Version Concurrency Control) minimal runnable MVP.

This script implements a tiny in-memory MVCC engine with snapshot reads:
- each committed write creates a new version (commit_ts, value)
- each transaction reads from its start snapshot (start_ts)
- write-write conflict at commit uses "first-committer-wins"
- aborted transaction can retry with a newer snapshot

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Operation:
    kind: str  # "R" | "W" | "C"
    item: Optional[str] = None
    value: Optional[int] = None


@dataclass(frozen=True)
class Version:
    commit_ts: int
    value: int
    writer: str


@dataclass
class TransactionState:
    base_tx: str
    name: str
    start_ts: int
    ops: Sequence[Operation]
    attempt: int
    pc: int = 0
    status: str = "active"  # active | committed | aborted_retry | aborted_final
    write_buffer: Dict[str, int] = field(default_factory=dict)
    read_history: List[Tuple[str, int, int]] = field(default_factory=list)  # (item, value, version_ts)


@dataclass(frozen=True)
class CommitRecord:
    tx_name: str
    commit_ts: int
    writes: Dict[str, int]


@dataclass
class WorkloadResult:
    name: str
    final_visible: Dict[str, int]
    versions: Dict[str, List[Version]]
    tx_status: Dict[str, str]
    retries: Dict[str, int]
    read_history_by_attempt: Dict[str, List[Tuple[str, int, int]]]
    commit_log: List[CommitRecord]
    event_log: List[str]


class MVCCEngine:
    """A tiny MVCC engine with snapshot reads and first-committer-wins checks."""

    def __init__(self, initial_data: Dict[str, int]) -> None:
        self._clock = 0
        self.versions: Dict[str, List[Version]] = {}
        self.event_log: List[str] = []
        self.commit_log: List[CommitRecord] = []

        for item, value in sorted(initial_data.items()):
            # Bootstrap versions are considered committed at ts=0.
            self.versions[item] = [Version(commit_ts=0, value=value, writer="BOOTSTRAP")]

    def _next_ts(self) -> int:
        self._clock += 1
        return self._clock

    def _ensure_item(self, item: str) -> None:
        if item not in self.versions:
            self.versions[item] = [Version(commit_ts=0, value=0, writer="BOOTSTRAP")]

    def begin(self, base_tx: str, ops: Sequence[Operation], attempt: int) -> TransactionState:
        start_ts = self._next_ts()
        name = base_tx if attempt == 0 else f"{base_tx}#retry{attempt}"
        self.event_log.append(f"BEGIN {name} snapshot_ts={start_ts}")
        return TransactionState(
            base_tx=base_tx,
            name=name,
            start_ts=start_ts,
            ops=ops,
            attempt=attempt,
        )

    def _visible_version(self, item: str, snapshot_ts: int) -> Version:
        self._ensure_item(item)
        for version in reversed(self.versions[item]):
            if version.commit_ts <= snapshot_ts:
                return version
        raise RuntimeError(f"No visible version for item={item}, snapshot_ts={snapshot_ts}")

    def read(self, tx: TransactionState, item: str) -> None:
        if item in tx.write_buffer:
            value = tx.write_buffer[item]
            tx.read_history.append((item, value, tx.start_ts))
            self.event_log.append(
                f"{tx.name} READ {item}={value} from own_write_buffer(snapshot_ts={tx.start_ts})"
            )
            return

        visible = self._visible_version(item, tx.start_ts)
        tx.read_history.append((item, visible.value, visible.commit_ts))
        self.event_log.append(
            f"{tx.name} READ {item}={visible.value} visible_version_ts={visible.commit_ts} "
            f"(snapshot_ts={tx.start_ts})"
        )

    def write(self, tx: TransactionState, item: str, value: int) -> None:
        tx.write_buffer[item] = value
        self.event_log.append(f"{tx.name} WRITE-BUFFER {item}={value}")

    def _has_write_write_conflict(self, tx: TransactionState) -> Tuple[bool, str]:
        for item in sorted(tx.write_buffer):
            latest_committed_ts = self.versions.get(item, [Version(0, 0, "BOOTSTRAP")])[-1].commit_ts
            if latest_committed_ts > tx.start_ts:
                return (
                    True,
                    f"item={item}, latest_committed_ts={latest_committed_ts} > snapshot_ts={tx.start_ts}",
                )
        return False, ""

    def commit(self, tx: TransactionState) -> bool:
        has_conflict, reason = self._has_write_write_conflict(tx)
        if has_conflict:
            tx.status = "aborted_retry"
            self.event_log.append(f"{tx.name} ABORT write_write_conflict ({reason})")
            return False

        commit_ts = self._next_ts()
        writes_sorted = dict(sorted(tx.write_buffer.items()))
        for item, value in writes_sorted.items():
            self._ensure_item(item)
            self.versions[item].append(Version(commit_ts=commit_ts, value=value, writer=tx.name))

        tx.status = "committed"
        self.commit_log.append(CommitRecord(tx_name=tx.name, commit_ts=commit_ts, writes=writes_sorted))
        self.event_log.append(f"{tx.name} COMMIT commit_ts={commit_ts} writes={writes_sorted}")
        return True

    def execute_step(self, tx: TransactionState) -> bool:
        if tx.status != "active":
            return False
        if tx.pc >= len(tx.ops):
            raise RuntimeError(f"{tx.name} program ended without terminal status")

        op = tx.ops[tx.pc]

        if op.kind == "R":
            if op.item is None:
                raise ValueError("Read operation requires item")
            self.read(tx, op.item)
            tx.pc += 1
            return True

        if op.kind == "W":
            if op.item is None or op.value is None:
                raise ValueError("Write operation requires item and value")
            self.write(tx, op.item, op.value)
            tx.pc += 1
            return True

        if op.kind == "C":
            self.commit(tx)
            tx.pc += 1
            return True

        raise ValueError(f"Unsupported operation kind: {op.kind}")


def run_workload(
    name: str,
    initial_data: Dict[str, int],
    programs: Dict[str, Sequence[Operation]],
    schedule: Sequence[str],
    max_retries: int = 1,
) -> WorkloadResult:
    engine = MVCCEngine(initial_data=initial_data)

    retries: Dict[str, int] = {tx: 0 for tx in programs}
    states: Dict[str, TransactionState] = {}
    attempt_states: Dict[str, TransactionState] = {}

    def begin_or_restart(tx_name: str) -> TransactionState:
        attempt = retries[tx_name]
        state = engine.begin(base_tx=tx_name, ops=programs[tx_name], attempt=attempt)
        states[tx_name] = state
        attempt_states[state.name] = state
        return state

    for step_id, tx_name in enumerate(schedule, start=1):
        if tx_name not in programs:
            raise KeyError(f"Unknown transaction in schedule: {tx_name}")

        if tx_name not in states:
            state = begin_or_restart(tx_name)
        else:
            state = states[tx_name]

        if state.status in {"committed", "aborted_final"}:
            engine.event_log.append(f"[step={step_id:02d}] SKIP {state.name} status={state.status}")
            continue

        if state.status == "aborted_retry":
            if retries[tx_name] >= max_retries:
                state.status = "aborted_final"
                engine.event_log.append(
                    f"[step={step_id:02d}] {state.name} no retry budget left -> aborted_final"
                )
                continue

            retries[tx_name] += 1
            state = begin_or_restart(tx_name)
            engine.event_log.append(
                f"[step={step_id:02d}] RESTART {tx_name} as {state.name} "
                f"(retry={retries[tx_name]})"
            )

        progressed = engine.execute_step(state)
        if not progressed:
            raise RuntimeError(f"No progress at step={step_id:02d} for {state.name}")

    unresolved = [
        tx_name
        for tx_name, state in states.items()
        if state.status not in {"committed", "aborted_final"}
    ]
    if unresolved:
        raise RuntimeError(
            "Schedule ended before all started transactions reached terminal status: "
            f"{unresolved}"
        )

    final_visible: Dict[str, int] = {
        item: versions[-1].value for item, versions in sorted(engine.versions.items())
    }

    return WorkloadResult(
        name=name,
        final_visible=final_visible,
        versions={k: list(vs) for k, vs in sorted(engine.versions.items())},
        tx_status={tx: state.status for tx, state in sorted(states.items())},
        retries=dict(sorted(retries.items())),
        read_history_by_attempt={
            attempt_name: list(state.read_history)
            for attempt_name, state in sorted(attempt_states.items())
        },
        commit_log=list(engine.commit_log),
        event_log=list(engine.event_log),
    )


def R(item: str) -> Operation:
    return Operation(kind="R", item=item)


def W(item: str, value: int) -> Operation:
    return Operation(kind="W", item=item, value=value)


def C() -> Operation:
    return Operation(kind="C")


def print_result(result: WorkloadResult) -> None:
    print(f"\n=== {result.name} ===")
    print(f"final_visible: {result.final_visible}")
    print(f"tx_status: {result.tx_status}")
    print(f"retries: {result.retries}")
    print("commit_log:")
    for record in result.commit_log:
        print(f"  {record.tx_name}@{record.commit_ts} writes={record.writes}")
    print("versions:")
    for item, versions in result.versions.items():
        rendered = ", ".join(
            f"(ts={v.commit_ts}, value={v.value}, writer={v.writer})" for v in versions
        )
        print(f"  {item}: [{rendered}]")
    print("event_log:")
    for line in result.event_log:
        print(f"  {line}")


def scenario_snapshot_read_stability() -> WorkloadResult:
    """Reader keeps old snapshot while later transaction reads new committed version."""
    programs = {
        "T1": [R("X"), R("X"), C()],
        "T2": [W("X", 20), C()],
        "T3": [R("X"), C()],
    }
    schedule = ["T1", "T2", "T2", "T1", "T1", "T3", "T3"]

    result = run_workload(
        name="Scenario A: Snapshot read stability",
        initial_data={"X": 10},
        programs=programs,
        schedule=schedule,
        max_retries=0,
    )

    t1_reads = result.read_history_by_attempt["T1"]
    t3_reads = result.read_history_by_attempt["T3"]
    assert t1_reads[0][1] == 10 and t1_reads[1][1] == 10, "T1 should see stable snapshot value 10"
    assert t3_reads[0][1] == 20, "T3 starts later and should observe new committed value 20"
    assert result.tx_status == {"T1": "committed", "T2": "committed", "T3": "committed"}
    assert result.final_visible["X"] == 20
    return result


def scenario_write_conflict_with_retry() -> WorkloadResult:
    """Two concurrent writers conflict; loser retries with newer snapshot."""
    programs = {
        "T1": [R("X"), W("X", 1), C()],
        "T2": [R("X"), W("X", 2), C()],
    }
    schedule = ["T1", "T2", "T2", "T2", "T1", "T1", "T1", "T1", "T1"]

    result = run_workload(
        name="Scenario B: Write-write conflict and retry",
        initial_data={"X": 0},
        programs=programs,
        schedule=schedule,
        max_retries=1,
    )

    assert any("ABORT write_write_conflict" in line for line in result.event_log), (
        "One transaction should abort due to write-write conflict"
    )
    assert result.retries["T1"] == 1, "T1 should retry exactly once"
    assert result.tx_status == {"T1": "committed", "T2": "committed"}
    assert result.final_visible["X"] == 1, "T1 retry should commit newer value after T2"
    return result


def main() -> None:
    result_a = scenario_snapshot_read_stability()
    print_result(result_a)

    result_b = scenario_write_conflict_with_retry()
    print_result(result_b)

    print("\nAll MVCC MVP checks passed.")


if __name__ == "__main__":
    main()
