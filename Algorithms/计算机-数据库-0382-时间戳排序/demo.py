"""Timestamp Ordering (TO) minimal runnable MVP.

This script demonstrates single-version timestamp ordering concurrency control:
- Basic TO (abort on obsolete write)
- Thomas write rule (ignore obsolete write)
- automatic restart with a newer timestamp

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Operation:
    kind: str  # "R" | "W" | "C"
    item: Optional[str] = None
    value: Optional[int] = None


@dataclass
class TransactionState:
    name: str
    ops: Sequence[Operation]
    ts: int
    pc: int = 0
    status: str = "active"  # active | committed | aborted_retry | aborted_final
    restart_count: int = 0


@dataclass
class WorkloadResult:
    name: str
    mode: str
    final_data: Dict[str, int]
    final_rts: Dict[str, int]
    final_wts: Dict[str, int]
    tx_status: Dict[str, str]
    restart_count: Dict[str, int]
    event_log: List[str]


class TimestampOrderingEngine:
    """Single-version Timestamp Ordering engine."""

    def __init__(self, initial_data: Dict[str, int], mode: str) -> None:
        if mode not in {"basic", "thomas"}:
            raise ValueError("mode must be 'basic' or 'thomas'.")

        self.mode = mode
        self.data: Dict[str, int] = dict(initial_data)
        self.rts: Dict[str, int] = {k: 0 for k in initial_data}
        self.wts: Dict[str, int] = {k: 0 for k in initial_data}

        self._timestamp_counter = 0
        self.tx_timestamp: Dict[str, int] = {}

    def begin_transaction(self, tx_name: str) -> int:
        self._timestamp_counter += 1
        ts = self._timestamp_counter
        self.tx_timestamp[tx_name] = ts
        return ts

    def restart_transaction(self, tx_name: str) -> int:
        return self.begin_transaction(tx_name)

    def _ensure_item(self, item: str) -> None:
        self.data.setdefault(item, 0)
        self.rts.setdefault(item, 0)
        self.wts.setdefault(item, 0)

    def read(self, tx_name: str, item: str) -> Tuple[str, Optional[int], str]:
        self._ensure_item(item)
        ts = self.tx_timestamp[tx_name]

        if ts < self.wts[item]:
            reason = f"ABORT READ {tx_name}.{item}: TS={ts} < WTS({item})={self.wts[item]}"
            return "ABORT", None, reason

        self.rts[item] = max(self.rts[item], ts)
        value = self.data[item]
        info = (
            f"READ {tx_name}.{item}={value} allowed: "
            f"TS={ts}, RTS({item})={self.rts[item]}, WTS({item})={self.wts[item]}"
        )
        return "OK", value, info

    def write(self, tx_name: str, item: str, value: int) -> Tuple[str, str]:
        self._ensure_item(item)
        ts = self.tx_timestamp[tx_name]

        if ts < self.rts[item]:
            reason = f"ABORT WRITE {tx_name}.{item}: TS={ts} < RTS({item})={self.rts[item]}"
            return "ABORT", reason

        if ts < self.wts[item]:
            if self.mode == "basic":
                reason = f"ABORT WRITE {tx_name}.{item}: TS={ts} < WTS({item})={self.wts[item]}"
                return "ABORT", reason

            info = (
                f"IGNORE WRITE {tx_name}.{item}={value}: "
                f"TS={ts} < WTS({item})={self.wts[item]} (Thomas rule)"
            )
            return "IGNORE", info

        self.data[item] = value
        self.wts[item] = ts
        info = (
            f"WRITE {tx_name}.{item}={value} applied: "
            f"TS={ts}, RTS({item})={self.rts[item]}, WTS({item})={self.wts[item]}"
        )
        return "OK", info


def run_workload(
    name: str,
    mode: str,
    programs: Dict[str, Sequence[Operation]],
    initial_data: Dict[str, int],
    schedule: Sequence[str],
    max_restarts: int,
) -> WorkloadResult:
    engine = TimestampOrderingEngine(initial_data=initial_data, mode=mode)

    states: Dict[str, TransactionState] = {}
    for tx_name in sorted(programs):
        states[tx_name] = TransactionState(
            name=tx_name,
            ops=programs[tx_name],
            ts=engine.begin_transaction(tx_name),
        )

    log: List[str] = []

    def is_finished(status: str) -> bool:
        return status in {"committed", "aborted_final"}

    for step_id, tx_name in enumerate(schedule, start=1):
        if tx_name not in states:
            raise KeyError(f"Unknown transaction in schedule: {tx_name}")

        state = states[tx_name]

        if is_finished(state.status):
            log.append(f"[{step_id:02d}] {tx_name} skipped ({state.status})")
            continue

        if state.status == "aborted_retry":
            if state.restart_count >= max_restarts:
                state.status = "aborted_final"
                log.append(
                    f"[{step_id:02d}] {tx_name} no restart budget left; finalize as aborted_final"
                )
                continue

            old_ts = state.ts
            state.ts = engine.restart_transaction(tx_name)
            state.pc = 0
            state.status = "active"
            state.restart_count += 1
            log.append(
                f"[{step_id:02d}] {tx_name} RESTART ts {old_ts} -> {state.ts} "
                f"(restart #{state.restart_count})"
            )

        if state.pc >= len(state.ops):
            raise RuntimeError(f"{tx_name} reached end of program without terminal status")

        op = state.ops[state.pc]

        if op.kind == "R":
            if op.item is None:
                raise ValueError("Read operation requires item")
            outcome, value, msg = engine.read(tx_name, op.item)
            if outcome == "ABORT":
                state.status = "aborted_retry"
                log.append(f"[{step_id:02d}] {msg}")
            else:
                state.pc += 1
                log.append(f"[{step_id:02d}] {msg}")
                _ = value  # keep explicit return contract usage
            continue

        if op.kind == "W":
            if op.item is None or op.value is None:
                raise ValueError("Write operation requires item and value")
            outcome, msg = engine.write(tx_name, op.item, op.value)
            if outcome == "ABORT":
                state.status = "aborted_retry"
                log.append(f"[{step_id:02d}] {msg}")
            elif outcome == "IGNORE":
                state.pc += 1
                log.append(f"[{step_id:02d}] {msg}")
            else:
                state.pc += 1
                log.append(f"[{step_id:02d}] {msg}")
            continue

        if op.kind == "C":
            state.pc += 1
            state.status = "committed"
            log.append(f"[{step_id:02d}] COMMIT {tx_name} with TS={state.ts}")
            continue

        raise ValueError(f"Unsupported operation kind: {op.kind}")

    unresolved = [s.name for s in states.values() if not is_finished(s.status)]
    if unresolved:
        raise RuntimeError(f"Schedule ended before transactions finished: {unresolved}")

    return WorkloadResult(
        name=name,
        mode=mode,
        final_data=dict(sorted(engine.data.items())),
        final_rts=dict(sorted(engine.rts.items())),
        final_wts=dict(sorted(engine.wts.items())),
        tx_status={tx: st.status for tx, st in sorted(states.items())},
        restart_count={tx: st.restart_count for tx, st in sorted(states.items())},
        event_log=log,
    )


def print_result(result: WorkloadResult) -> None:
    print(f"\n=== {result.name} (mode={result.mode}) ===")
    print(f"final_data: {result.final_data}")
    print(f"final_RTS : {result.final_rts}")
    print(f"final_WTS : {result.final_wts}")
    print(f"tx_status : {result.tx_status}")
    print(f"restarts  : {result.restart_count}")
    print("event_log:")
    for line in result.event_log:
        print(f"  {line}")


def scenario_basic_with_restart() -> WorkloadResult:
    programs = {
        "T1": [Operation("R", "X"), Operation("W", "X", 5), Operation("C")],
        "T2": [Operation("W", "X", 9), Operation("C")],
    }
    schedule = ["T1", "T2", "T2", "T1", "T1", "T1", "T1"]

    result = run_workload(
        name="A: Basic TO with restart",
        mode="basic",
        programs=programs,
        initial_data={"X": 0},
        schedule=schedule,
        max_restarts=2,
    )

    assert result.final_data["X"] == 5, "T1 should overwrite X after restart with newer TS"
    assert result.tx_status == {"T1": "committed", "T2": "committed"}
    assert result.restart_count["T1"] == 1
    assert any("ABORT WRITE T1.X" in x for x in result.event_log)
    return result


def scenario_thomas_rule() -> WorkloadResult:
    programs = {
        "T1": [Operation("R", "X"), Operation("W", "X", 5), Operation("C")],
        "T2": [Operation("W", "X", 9), Operation("C")],
    }
    schedule = ["T1", "T2", "T2", "T1", "T1"]

    result = run_workload(
        name="B: Thomas write rule",
        mode="thomas",
        programs=programs,
        initial_data={"X": 0},
        schedule=schedule,
        max_restarts=0,
    )

    assert result.final_data["X"] == 9, "Obsolete write should be ignored under Thomas rule"
    assert result.tx_status == {"T1": "committed", "T2": "committed"}
    assert result.restart_count["T1"] == 0
    assert any("IGNORE WRITE T1.X=5" in x for x in result.event_log)
    return result


def scenario_abort_by_rts_check() -> WorkloadResult:
    programs = {
        "T1": [Operation("W", "Y", 7), Operation("C")],
        "T2": [Operation("R", "Y"), Operation("C")],
    }
    schedule = ["T2", "T1", "T1", "T2"]

    result = run_workload(
        name="C: Abort on TS < RTS",
        mode="basic",
        programs=programs,
        initial_data={"Y": 0},
        schedule=schedule,
        max_restarts=0,
    )

    assert result.final_data["Y"] == 0, "T1 write must be rejected due to TS < RTS"
    assert result.tx_status == {"T1": "aborted_final", "T2": "committed"}
    assert any("TS=1 < RTS(Y)=2" in x for x in result.event_log)
    return result


def main() -> None:
    results = [
        scenario_basic_with_restart(),
        scenario_thomas_rule(),
        scenario_abort_by_rts_check(),
    ]
    for res in results:
        print_result(res)

    print("\nAll TO scenarios passed.")


if __name__ == "__main__":
    main()
