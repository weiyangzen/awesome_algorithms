"""Strict Two-Phase Locking (Strict 2PL) minimal runnable MVP.

The demo simulates transaction scheduling, lock acquisition/release,
waiting queues, and deadlock detection/rollback in pure Python.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import re
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Operation:
    tx: str
    kind: str  # "R" | "W" | "C"
    item: Optional[str] = None
    value: Optional[int] = None


@dataclass
class TransactionState:
    program: Sequence[Operation]
    pc: int = 0
    status: str = "active"  # active | waiting | committed | aborted
    write_buffer: Dict[str, int] = field(default_factory=dict)
    waiting_on: Optional[Tuple[str, str]] = None  # (item, lock_mode)


@dataclass(frozen=True)
class LockRequest:
    tx: str
    mode: str  # "S" | "X"


@dataclass
class LockState:
    shared: Set[str] = field(default_factory=set)
    exclusive: Optional[str] = None
    queue: Deque[LockRequest] = field(default_factory=deque)


@dataclass
class WorkloadResult:
    name: str
    final_db: Dict[str, int]
    tx_status: Dict[str, str]
    event_log: List[str]


class LockManager:
    """Lock manager for strict 2PL with per-item wait queues."""

    def __init__(self) -> None:
        self.lock_table: Dict[str, LockState] = {}

    def _get_state(self, item: str) -> LockState:
        if item not in self.lock_table:
            self.lock_table[item] = LockState()
        return self.lock_table[item]

    @staticmethod
    def _holders_blocking(state: LockState, tx: str, mode: str) -> Set[str]:
        blockers: Set[str] = set()
        if state.exclusive is not None and state.exclusive != tx:
            blockers.add(state.exclusive)
        if mode == "X":
            blockers.update(holder for holder in state.shared if holder != tx)
        return blockers

    def _can_grant(self, state: LockState, tx: str, mode: str, respect_queue: bool) -> bool:
        if respect_queue and state.queue and state.queue[0].tx != tx:
            return False
        return len(self._holders_blocking(state, tx, mode)) == 0

    @staticmethod
    def _grant(state: LockState, tx: str, mode: str) -> None:
        if mode == "S":
            if state.exclusive == tx:
                return
            state.shared.add(tx)
            return

        # mode == "X"
        state.shared.discard(tx)  # lock upgrade S->X if needed
        state.exclusive = tx

    @staticmethod
    def _remove_from_queue(state: LockState, tx: str) -> None:
        if not state.queue:
            return
        state.queue = deque(req for req in state.queue if req.tx != tx)

    def request_lock(self, tx: str, item: str, mode: str) -> bool:
        state = self._get_state(item)

        # Already has sufficient lock.
        if mode == "S" and (state.exclusive == tx or tx in state.shared):
            return True
        if mode == "X" and state.exclusive == tx:
            return True

        if self._can_grant(state, tx, mode, respect_queue=True):
            self._grant(state, tx, mode)
            self._remove_from_queue(state, tx)
            return True

        if not any(req.tx == tx for req in state.queue):
            state.queue.append(LockRequest(tx=tx, mode=mode))
        return False

    def _drain_queue(self, item: str, state: LockState) -> List[Tuple[str, str, str]]:
        granted: List[Tuple[str, str, str]] = []
        while state.queue:
            req = state.queue[0]
            if not self._can_grant(state, req.tx, req.mode, respect_queue=False):
                break
            self._grant(state, req.tx, req.mode)
            state.queue.popleft()
            granted.append((req.tx, item, req.mode))

            # An X-lock at queue head blocks everyone behind it.
            if req.mode == "X":
                break
        return granted

    def release_all(self, tx: str) -> List[Tuple[str, str, str]]:
        granted_after_release: List[Tuple[str, str, str]] = []
        to_delete: List[str] = []

        for item, state in self.lock_table.items():
            changed = False
            if state.exclusive == tx:
                state.exclusive = None
                changed = True
            if tx in state.shared:
                state.shared.remove(tx)
                changed = True
            if any(req.tx == tx for req in state.queue):
                self._remove_from_queue(state, tx)
                changed = True

            if changed:
                granted_after_release.extend(self._drain_queue(item, state))

            if state.exclusive is None and not state.shared and not state.queue:
                to_delete.append(item)

        for item in to_delete:
            del self.lock_table[item]

        return granted_after_release

    def build_wait_for_graph(self) -> Dict[str, Set[str]]:
        graph: Dict[str, Set[str]] = defaultdict(set)
        for state in self.lock_table.values():
            for req in state.queue:
                blockers = self._holders_blocking(state, req.tx, req.mode)
                if not blockers:
                    continue
                graph[req.tx].update(blockers)
                for blocker in blockers:
                    graph.setdefault(blocker, set())
        return graph


# ---------- deadlock utilities ----------

def find_cycle(graph: Dict[str, Set[str]]) -> Optional[List[str]]:
    """Return one cycle path (last node repeats the first), else None."""

    color: Dict[str, int] = {}
    stack: List[str] = []
    stack_index: Dict[str, int] = {}
    found_cycle: Optional[List[str]] = None

    def dfs(node: str) -> bool:
        nonlocal found_cycle
        color[node] = 1
        stack_index[node] = len(stack)
        stack.append(node)

        for nxt in graph.get(node, set()):
            state = color.get(nxt, 0)
            if state == 0:
                if dfs(nxt):
                    return True
            elif state == 1:
                start = stack_index[nxt]
                found_cycle = stack[start:] + [nxt]
                return True

        stack.pop()
        stack_index.pop(node, None)
        color[node] = 2
        return False

    for node in list(graph.keys()):
        if color.get(node, 0) == 0 and dfs(node):
            return found_cycle
    return None


def transaction_rank(tx: str) -> int:
    """Used for deterministic victim selection: larger suffix = younger."""
    m = re.search(r"(\d+)$", tx)
    return int(m.group(1)) if m else 0


def choose_victim(cycle: Iterable[str]) -> str:
    unique = list(dict.fromkeys(cycle))
    return max(unique, key=transaction_rank)


# ---------- scheduler ----------

def execute_step(
    tx: str,
    state: TransactionState,
    lock_mgr: LockManager,
    committed_db: Dict[str, int],
    event_log: List[str],
) -> bool:
    """Execute at most one operation for one transaction; return True if progressed."""
    if state.status in {"committed", "aborted"}:
        return False

    if state.pc >= len(state.program):
        raise RuntimeError(f"{tx} ran out of program but not finalized.")

    op = state.program[state.pc]
    if op.tx != tx:
        raise ValueError(f"Program op tx mismatch: expected {tx}, got {op.tx}")

    if op.kind == "R":
        if op.item is None:
            raise ValueError("Read operation must provide item.")
        granted = lock_mgr.request_lock(tx, op.item, "S")
        if not granted:
            state.status = "waiting"
            waiting = (op.item, "S")
            if state.waiting_on != waiting:
                event_log.append(f"{tx} waits S({op.item})")
            state.waiting_on = waiting
            return False

        state.status = "active"
        state.waiting_on = None
        value = state.write_buffer.get(op.item, committed_db.get(op.item))
        event_log.append(f"{tx} READ {op.item}={value}")
        state.pc += 1
        return True

    if op.kind == "W":
        if op.item is None or op.value is None:
            raise ValueError("Write operation must provide item and value.")
        granted = lock_mgr.request_lock(tx, op.item, "X")
        if not granted:
            state.status = "waiting"
            waiting = (op.item, "X")
            if state.waiting_on != waiting:
                event_log.append(f"{tx} waits X({op.item})")
            state.waiting_on = waiting
            return False

        state.status = "active"
        state.waiting_on = None
        state.write_buffer[op.item] = op.value
        event_log.append(f"{tx} WRITE-BUFFER {op.item}={op.value}")
        state.pc += 1
        return True

    if op.kind == "C":
        # Strict 2PL: release all locks only at commit.
        for item, value in state.write_buffer.items():
            committed_db[item] = value
        event_log.append(f"{tx} COMMIT apply={dict(sorted(state.write_buffer.items()))}")
        state.write_buffer.clear()

        state.status = "committed"
        state.waiting_on = None
        state.pc += 1

        grants = lock_mgr.release_all(tx)
        for g_tx, g_item, g_mode in grants:
            event_log.append(f"LOCK-GRANT {g_tx} gets {g_mode}({g_item}) after release")
        return True

    raise ValueError(f"Unsupported operation kind: {op.kind}")


def abort_transaction(
    tx: str,
    states: Dict[str, TransactionState],
    lock_mgr: LockManager,
    event_log: List[str],
) -> None:
    state = states[tx]
    if state.status in {"committed", "aborted"}:
        return

    state.status = "aborted"
    state.waiting_on = None
    state.write_buffer.clear()
    event_log.append(f"{tx} ABORT (deadlock victim)")

    grants = lock_mgr.release_all(tx)
    for g_tx, g_item, g_mode in grants:
        event_log.append(f"LOCK-GRANT {g_tx} gets {g_mode}({g_item}) after release")


def run_workload(
    name: str,
    programs: Dict[str, Sequence[Operation]],
    initial_db: Dict[str, int],
    max_rounds: int = 200,
) -> WorkloadResult:
    committed_db = dict(initial_db)
    states = {tx: TransactionState(program=ops) for tx, ops in programs.items()}
    tx_order = sorted(programs.keys(), key=transaction_rank)

    lock_mgr = LockManager()
    event_log: List[str] = [f"=== {name} ==="]

    rounds = 0
    while not all(s.status in {"committed", "aborted"} for s in states.values()):
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError(f"Exceeded max_rounds={max_rounds}; scheduling did not converge.")

        progressed = False
        for tx in tx_order:
            progressed = execute_step(tx, states[tx], lock_mgr, committed_db, event_log) or progressed

        wait_graph = lock_mgr.build_wait_for_graph()
        cycle = find_cycle(wait_graph)
        if cycle:
            victim = choose_victim(cycle[:-1])
            cycle_text = " -> ".join(cycle)
            event_log.append(f"DEADLOCK DETECTED: {cycle_text}; victim={victim}")
            abort_transaction(victim, states, lock_mgr, event_log)
            progressed = True

        if not progressed:
            raise RuntimeError("No progress and no deadlock cycle found.")

    tx_status = {tx: state.status for tx, state in states.items()}
    return WorkloadResult(name=name, final_db=committed_db, tx_status=tx_status, event_log=event_log)


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
    print(f"TX Status: {result.tx_status}")


def main() -> None:
    # Scenario 1: waiting but no deadlock.
    # T2 cannot read A until T1 commits, which demonstrates strictness (no dirty read).
    workload_1 = {
        "T1": [W("T1", "A", 100), W("T1", "B", 200), C("T1")],
        "T2": [R("T2", "A"), W("T2", "B", 300), C("T2")],
        "T3": [R("T3", "B"), C("T3")],
    }
    result_1 = run_workload("Scenario 1: Strictness and waiting", workload_1, {"A": 0, "B": 0})

    assert result_1.tx_status == {"T1": "committed", "T2": "committed", "T3": "committed"}
    assert result_1.final_db == {"A": 100, "B": 300}
    t1_commit_pos = next(i for i, e in enumerate(result_1.event_log) if "T1 COMMIT" in e)
    t2_read_pos = next(i for i, e in enumerate(result_1.event_log) if "T2 READ A=100" in e)
    assert t2_read_pos > t1_commit_pos

    # Scenario 2: deadlock under strict 2PL and deterministic victim rollback.
    workload_2 = {
        "T1": [W("T1", "X", 1), W("T1", "Y", 1), C("T1")],
        "T2": [W("T2", "Y", 2), W("T2", "X", 2), C("T2")],
    }
    result_2 = run_workload("Scenario 2: Deadlock and abort", workload_2, {"X": 0, "Y": 0})

    assert result_2.tx_status["T1"] == "committed"
    assert result_2.tx_status["T2"] == "aborted"
    assert result_2.final_db == {"X": 1, "Y": 1}
    assert any("DEADLOCK DETECTED" in line for line in result_2.event_log)

    print_result(result_1)
    print_result(result_2)
    print("\nAll assertions passed. Strict 2PL MVP is working.")


if __name__ == "__main__":
    main()
