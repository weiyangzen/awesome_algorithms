"""Deadlock detection MVP using wait-for graph cycle detection."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set


@dataclass(frozen=True)
class Operation:
    kind: str
    resource: Optional[str] = None
    note: str = ""


@dataclass
class Transaction:
    tx_id: str
    timestamp: int  # smaller means older
    program: List[Operation]
    pc: int = 0
    status: str = "ready"  # ready | waiting | committed
    wait_resource: Optional[str] = None
    held_locks: Set[str] = field(default_factory=set)
    restart_count: int = 0

    @property
    def done(self) -> bool:
        return self.status == "committed"


class DeadlockDetectingScheduler:
    """Exclusive lock scheduler with deadlock detection and victim restart."""

    def __init__(self, transactions: List[Transaction]) -> None:
        self.tx_by_id: Dict[str, Transaction] = {tx.tx_id: tx for tx in transactions}
        self.lock_owner: Dict[str, str] = {}
        self.waiting_queues: Dict[str, Deque[str]] = defaultdict(deque)
        self.event_log: List[str] = []
        self.commit_order: List[str] = []
        self.detected_cycles: List[List[str]] = []
        self.deadlock_resolutions: int = 0

    def run(self, max_rounds: int = 100) -> None:
        for round_idx in range(1, max_rounds + 1):
            progressed = False
            self._log(f"=== round {round_idx} ===")

            for tx_id in sorted(self.tx_by_id):
                tx = self.tx_by_id[tx_id]
                progressed = self._step(tx) or progressed

            # Periodic deadlock detection on wait-for graph.
            resolved = self._detect_and_resolve_deadlock()
            progressed = progressed or resolved

            if all(tx.done for tx in self.tx_by_id.values()):
                self._log("all transactions committed")
                return

            if not progressed:
                raise RuntimeError(
                    "scheduler made no progress and no deadlock was resolved"
                )

        raise RuntimeError("max rounds exceeded before all transactions committed")

    def _step(self, tx: Transaction) -> bool:
        if tx.status in {"waiting", "committed"}:
            return False

        if tx.pc >= len(tx.program):
            tx.status = "committed"
            self.commit_order.append(tx.tx_id)
            self._log(f"{tx.tx_id}: implicit commit at program end")
            return True

        op = tx.program[tx.pc]
        if op.kind == "lock":
            if op.resource is None:
                raise ValueError(f"{tx.tx_id}: lock op missing resource")
            return self._handle_lock(tx, op.resource)

        if op.kind == "work":
            detail = f" ({op.note})" if op.note else ""
            self._log(f"{tx.tx_id}: work{detail}")
            tx.pc += 1
            return True

        if op.kind == "unlock":
            if op.resource is None:
                raise ValueError(f"{tx.tx_id}: unlock op missing resource")
            self._release_one(tx, op.resource)
            tx.pc += 1
            self._log(f"{tx.tx_id}: unlock {op.resource}")
            return True

        if op.kind == "commit":
            self._commit(tx)
            return True

        raise ValueError(f"unsupported op kind: {op.kind}")

    def _handle_lock(self, tx: Transaction, resource: str) -> bool:
        owner = self.lock_owner.get(resource)
        if owner is None or owner == tx.tx_id:
            self.lock_owner[resource] = tx.tx_id
            tx.held_locks.add(resource)
            tx.pc += 1
            self._log(f"{tx.tx_id}: granted lock on {resource}")
            return True

        # Wait and let deadlock detector decide if recovery is needed.
        if not (tx.status == "waiting" and tx.wait_resource == resource):
            tx.status = "waiting"
            tx.wait_resource = resource
            if tx.tx_id not in self.waiting_queues[resource]:
                self.waiting_queues[resource].append(tx.tx_id)
            self._log(f"{tx.tx_id}: waits for {resource} held by {owner}")
        return False

    def _detect_and_resolve_deadlock(self) -> bool:
        graph = self.wait_for_graph()
        cycle = self.find_cycle(graph)
        if cycle is None:
            return False

        self.detected_cycles.append(cycle)
        victim = self.choose_victim(cycle)
        self.deadlock_resolutions += 1
        self._abort_and_restart(victim, cycle)
        return True

    @staticmethod
    def find_cycle(graph: Dict[str, Set[str]]) -> Optional[List[str]]:
        """Return one cycle path (last node repeats first), else None."""
        color: Dict[str, int] = {}
        stack: List[str] = []
        stack_index: Dict[str, int] = {}
        found: Optional[List[str]] = None

        def dfs(node: str) -> bool:
            nonlocal found
            color[node] = 1
            stack_index[node] = len(stack)
            stack.append(node)
            for nxt in graph.get(node, set()):
                nxt_color = color.get(nxt, 0)
                if nxt_color == 0:
                    if dfs(nxt):
                        return True
                elif nxt_color == 1:
                    start = stack_index[nxt]
                    found = stack[start:] + [nxt]
                    return True
            stack.pop()
            stack_index.pop(node, None)
            color[node] = 2
            return False

        for node in list(graph):
            if color.get(node, 0) == 0 and dfs(node):
                return found
        return None

    def choose_victim(self, cycle: List[str]) -> str:
        # Deterministic policy: restart the youngest transaction in the cycle.
        unique = list(dict.fromkeys(cycle))
        return max(unique, key=lambda tx_id: self.tx_by_id[tx_id].timestamp)

    def _abort_and_restart(self, tx_id: str, cycle: List[str]) -> None:
        tx = self.tx_by_id[tx_id]
        self._log(f"deadlock detected: {' -> '.join(cycle)}; victim={tx_id}")
        tx.restart_count += 1
        self._remove_from_all_waiting_queues(tx_id)
        self._release_all(tx)
        tx.pc = 0
        tx.status = "ready"
        tx.wait_resource = None
        self._log(f"{tx_id}: restart from program beginning")

    def _remove_from_all_waiting_queues(self, tx_id: str) -> None:
        for resource in list(self.waiting_queues):
            queue = self.waiting_queues[resource]
            if not queue:
                continue
            self.waiting_queues[resource] = deque(
                waiting_tx for waiting_tx in queue if waiting_tx != tx_id
            )

    def _commit(self, tx: Transaction) -> None:
        self._release_all(tx)
        tx.status = "committed"
        tx.pc += 1
        self.commit_order.append(tx.tx_id)
        self._log(f"{tx.tx_id}: commit")

    def _release_all(self, tx: Transaction) -> None:
        for resource in sorted(tx.held_locks):
            self._release_one(tx, resource)

    def _release_one(self, tx: Transaction, resource: str) -> None:
        if self.lock_owner.get(resource) == tx.tx_id:
            del self.lock_owner[resource]
        tx.held_locks.discard(resource)
        self._wake_waiter(resource)

    def _wake_waiter(self, resource: str) -> None:
        queue = self.waiting_queues[resource]
        while queue:
            tx_id = queue[0]
            tx = self.tx_by_id[tx_id]

            # Skip stale entries.
            if tx.status != "waiting" or tx.wait_resource != resource:
                queue.popleft()
                continue

            if resource in self.lock_owner:
                return

            queue.popleft()
            self.lock_owner[resource] = tx_id
            tx.held_locks.add(resource)
            tx.status = "ready"
            tx.wait_resource = None
            tx.pc += 1  # consume the blocked lock operation
            self._log(f"{tx_id}: wake up and granted lock on {resource}")
            return

    def wait_for_graph(self) -> Dict[str, Set[str]]:
        graph: Dict[str, Set[str]] = defaultdict(set)
        for tx in self.tx_by_id.values():
            if tx.status != "waiting" or tx.wait_resource is None:
                continue
            owner = self.lock_owner.get(tx.wait_resource)
            if owner is not None and owner != tx.tx_id:
                graph[tx.tx_id].add(owner)
                graph.setdefault(owner, set())
        return graph

    def _log(self, message: str) -> None:
        self.event_log.append(message)


def build_demo_transactions() -> List[Transaction]:
    # T1/T2 request A and B in opposite order to force a deadlock cycle.
    t1 = Transaction(
        tx_id="T1",
        timestamp=1,
        program=[
            Operation("lock", "A"),
            Operation("work", note="t1 uses A"),
            Operation("lock", "B"),
            Operation("work", note="t1 uses A+B"),
            Operation("unlock", "B"),
            Operation("unlock", "A"),
            Operation("commit"),
        ],
    )
    t2 = Transaction(
        tx_id="T2",
        timestamp=2,
        program=[
            Operation("lock", "B"),
            Operation("work", note="t2 uses B"),
            Operation("lock", "A"),
            Operation("work", note="t2 uses B+A"),
            Operation("unlock", "A"),
            Operation("unlock", "B"),
            Operation("commit"),
        ],
    )
    t3 = Transaction(
        tx_id="T3",
        timestamp=3,
        program=[
            Operation("lock", "C"),
            Operation("work", note="independent workload"),
            Operation("unlock", "C"),
            Operation("commit"),
        ],
    )
    return [t1, t2, t3]


def main() -> None:
    scheduler = DeadlockDetectingScheduler(build_demo_transactions())
    scheduler.run(max_rounds=100)

    for line in scheduler.event_log:
        print(line)

    print("\n--- summary ---")
    print("commit order:", " -> ".join(scheduler.commit_order))
    print("deadlock detections:", scheduler.deadlock_resolutions)
    for tx_id in sorted(scheduler.tx_by_id):
        tx = scheduler.tx_by_id[tx_id]
        print(
            f"{tx_id}: status={tx.status}, restarts={tx.restart_count}, "
            f"held_locks={sorted(tx.held_locks)}"
        )

    # Deterministic self-checks for this demo.
    assert all(tx.done for tx in scheduler.tx_by_id.values())
    assert scheduler.deadlock_resolutions >= 1
    assert scheduler.tx_by_id["T2"].restart_count >= 1
    assert scheduler.lock_owner == {}
    assert scheduler.wait_for_graph() == {}


if __name__ == "__main__":
    main()
