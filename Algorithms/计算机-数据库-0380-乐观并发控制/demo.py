"""Optimistic Concurrency Control (OCC) minimal runnable MVP.

This script demonstrates a tiny in-memory OCC engine with:
- read phase (read set + private write buffer)
- validation phase (backward validation with logical timestamps)
- write phase (atomic apply on successful validation)
- abort and bounded retry

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Operation:
    tx: str
    kind: str  # "R" | "W" | "C"
    item: Optional[str] = None
    value: Optional[int] = None


@dataclass
class TransactionState:
    base_tx: str
    name: str
    program: Sequence[Operation]
    start_ts: int
    pc: int = 0
    status: str = "active"  # active | committed | aborted
    read_set: Set[str] = field(default_factory=set)
    write_buffer: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CommittedTxn:
    tx_name: str
    commit_ts: int
    write_set: Set[str]


@dataclass
class WorkloadResult:
    name: str
    final_db: Dict[str, int]
    final_status: Dict[str, str]
    retries: Dict[str, int]
    commit_order: List[Tuple[str, int]]
    event_log: List[str]


class OCCEngine:
    """A tiny single-version OCC engine with logical timestamps."""

    def __init__(self, initial_db: Dict[str, int]) -> None:
        self.committed_db: Dict[str, int] = dict(initial_db)
        self.history: List[CommittedTxn] = []
        self.event_log: List[str] = []
        self._clock: int = 0

    def _next_ts(self) -> int:
        self._clock += 1
        return self._clock

    def begin(self, base_tx: str, program: Sequence[Operation], attempt: int) -> TransactionState:
        start_ts = self._next_ts()
        if attempt == 0:
            name = base_tx
        else:
            name = f"{base_tx}#retry{attempt}"
        self.event_log.append(f"BEGIN {name} start_ts={start_ts}")
        return TransactionState(base_tx=base_tx, name=name, program=program, start_ts=start_ts)

    def _read(self, tx: TransactionState, item: str) -> None:
        value = tx.write_buffer.get(item, self.committed_db.get(item))
        tx.read_set.add(item)
        self.event_log.append(f"{tx.name} READ {item}={value}")

    def _write(self, tx: TransactionState, item: str, value: int) -> None:
        tx.write_buffer[item] = value
        self.event_log.append(f"{tx.name} WRITE-BUFFER {item}={value}")

    def validate(self, tx: TransactionState) -> Tuple[bool, str]:
        my_write_keys = set(tx.write_buffer.keys())
        my_touch_set = tx.read_set | my_write_keys

        for committed in self.history:
            # Transactions committed before Ti started are not overlapping competitors.
            if committed.commit_ts <= tx.start_ts:
                continue
            conflict_keys = committed.write_set & my_touch_set
            if conflict_keys:
                keys_text = ",".join(sorted(conflict_keys))
                reason = (
                    f"conflict with {committed.tx_name} (commit_ts={committed.commit_ts}) "
                    f"on {{{keys_text}}}"
                )
                return False, reason

        return True, ""

    def commit(self, tx: TransactionState) -> bool:
        ok, reason = self.validate(tx)
        if not ok:
            tx.status = "aborted"
            self.event_log.append(f"{tx.name} ABORT validation_failed: {reason}")
            return False

        for item, value in sorted(tx.write_buffer.items()):
            self.committed_db[item] = value

        commit_ts = self._next_ts()
        self.history.append(
            CommittedTxn(
                tx_name=tx.name,
                commit_ts=commit_ts,
                write_set=set(tx.write_buffer.keys()),
            )
        )
        tx.status = "committed"
        self.event_log.append(
            f"{tx.name} COMMIT commit_ts={commit_ts} apply={dict(sorted(tx.write_buffer.items()))}"
        )
        return True

    def execute_step(self, tx: TransactionState) -> bool:
        if tx.status != "active":
            return False

        if tx.pc >= len(tx.program):
            raise RuntimeError(f"{tx.name} ran out of program but not finalized.")

        op = tx.program[tx.pc]
        if op.tx != tx.base_tx:
            raise ValueError(
                f"Operation owner mismatch for {tx.name}: expected {tx.base_tx}, got {op.tx}"
            )

        if op.kind == "R":
            if op.item is None:
                raise ValueError("Read operation must provide item.")
            self._read(tx, op.item)
            tx.pc += 1
            return True

        if op.kind == "W":
            if op.item is None or op.value is None:
                raise ValueError("Write operation must provide item and value.")
            self._write(tx, op.item, op.value)
            tx.pc += 1
            return True

        if op.kind == "C":
            self.commit(tx)
            tx.pc += 1
            return True

        raise ValueError(f"Unsupported operation kind: {op.kind}")


def transaction_rank(tx: str) -> int:
    m = re.search(r"(\d+)$", tx)
    return int(m.group(1)) if m else 0


def run_workload(
    name: str,
    programs: Dict[str, Sequence[Operation]],
    initial_db: Dict[str, int],
    max_retries: int = 1,
    max_rounds: int = 200,
) -> WorkloadResult:
    engine = OCCEngine(initial_db)
    tx_order = sorted(programs.keys(), key=transaction_rank)

    retries = {tx: 0 for tx in tx_order}
    states = {
        tx: engine.begin(base_tx=tx, program=programs[tx], attempt=0)
        for tx in tx_order
    }

    rounds = 0
    while True:
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError(f"Exceeded max_rounds={max_rounds}; scheduling did not converge.")

        all_final = all(
            states[tx].status == "committed"
            or (states[tx].status == "aborted" and retries[tx] >= max_retries)
            for tx in tx_order
        )
        if all_final:
            break

        progressed = False
        for tx in tx_order:
            state = states[tx]

            if state.status == "committed":
                continue

            if state.status == "aborted":
                if retries[tx] < max_retries:
                    retries[tx] += 1
                    states[tx] = engine.begin(base_tx=tx, program=programs[tx], attempt=retries[tx])
                    progressed = True
                continue

            progressed = engine.execute_step(state) or progressed

        if not progressed:
            raise RuntimeError("No progress in one full round.")

    final_status = {tx: states[tx].status for tx in tx_order}
    commit_order = [(h.tx_name, h.commit_ts) for h in engine.history]
    return WorkloadResult(
        name=name,
        final_db=engine.committed_db,
        final_status=final_status,
        retries=retries,
        commit_order=commit_order,
        event_log=engine.event_log,
    )


# ---------- operation builders ----------

def R(tx: str, item: str) -> Operation:
    return Operation(tx=tx, kind="R", item=item)


def W(tx: str, item: str, value: int) -> Operation:
    return Operation(tx=tx, kind="W", item=item, value=value)


def C(tx: str) -> Operation:
    return Operation(tx=tx, kind="C")


# ---------- demo ----------

def print_result(result: WorkloadResult) -> None:
    print(f"\n{result.name}")
    print("-" * len(result.name))
    for line in result.event_log:
        print(line)
    print(f"Final DB: {result.final_db}")
    print(f"Final Status: {result.final_status}")
    print(f"Retries: {result.retries}")
    print(f"Commit Order: {result.commit_order}")


def main() -> None:
    # Scenario 1: low conflict, both transactions commit once.
    workload_1 = {
        "T1": [R("T1", "A"), W("T1", "A", 10), C("T1")],
        "T2": [R("T2", "B"), W("T2", "B", 20), C("T2")],
    }
    result_1 = run_workload(
        name="Scenario 1: Low conflict commits",
        programs=workload_1,
        initial_db={"A": 0, "B": 0},
        max_retries=0,
    )

    assert result_1.final_status == {"T1": "committed", "T2": "committed"}
    assert result_1.retries == {"T1": 0, "T2": 0}
    assert result_1.final_db == {"A": 10, "B": 20}

    # Scenario 2: conflicting updates on X, one aborts then retries and commits.
    workload_2 = {
        "T1": [R("T1", "X"), W("T1", "X", 101), C("T1")],
        "T2": [R("T2", "X"), W("T2", "X", 102), C("T2")],
    }
    result_2 = run_workload(
        name="Scenario 2: Validation abort and retry",
        programs=workload_2,
        initial_db={"X": 100},
        max_retries=1,
    )

    assert result_2.final_status == {"T1": "committed", "T2": "committed"}
    assert result_2.retries == {"T1": 0, "T2": 1}
    assert result_2.final_db == {"X": 102}
    assert any("ABORT validation_failed" in line for line in result_2.event_log)
    assert any("BEGIN T2#retry1" in line for line in result_2.event_log)

    print_result(result_1)
    print_result(result_2)
    print("\nAll assertions passed. OCC MVP is working.")


if __name__ == "__main__":
    main()
