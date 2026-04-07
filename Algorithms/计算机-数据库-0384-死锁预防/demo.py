"""Wait-Die deadlock prevention: minimal runnable MVP."""

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


class WaitDieScheduler:
    """Exclusive-lock scheduler using wait-die deadlock prevention."""

    def __init__(self, transactions: List[Transaction]) -> None:
        self.tx_by_id: Dict[str, Transaction] = {tx.tx_id: tx for tx in transactions}
        self.lock_owner: Dict[str, str] = {}
        self.waiting_queues: Dict[str, Deque[str]] = defaultdict(deque)
        self.event_log: List[str] = []
        self.commit_order: List[str] = []

    def run(self, max_rounds: int = 100) -> None:
        for round_idx in range(1, max_rounds + 1):
            progressed = False
            self._log(f"=== round {round_idx} ===")

            for tx_id in sorted(self.tx_by_id):
                tx = self.tx_by_id[tx_id]
                progressed = self._step(tx) or progressed
                self.assert_wait_for_graph_acyclic()

            if all(tx.done for tx in self.tx_by_id.values()):
                self._log("all transactions committed")
                return

            if not progressed:
                raise RuntimeError("scheduler made no progress; unexpected under wait-die")

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
                raise ValueError(f"{tx.tx_id}: lock op requires resource")
            return self._handle_lock(tx, op.resource)

        if op.kind == "work":
            detail = f" ({op.note})" if op.note else ""
            self._log(f"{tx.tx_id}: work{detail}")
            tx.pc += 1
            return True

        if op.kind == "unlock":
            if op.resource is None:
                raise ValueError(f"{tx.tx_id}: unlock op requires resource")
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

        owner_tx = self.tx_by_id[owner]

        if tx.timestamp < owner_tx.timestamp:
            # older transaction waits for younger holder
            if not (tx.status == "waiting" and tx.wait_resource == resource):
                tx.status = "waiting"
                tx.wait_resource = resource
                self.waiting_queues[resource].append(tx.tx_id)
                self._log(
                    f"{tx.tx_id}: waits for {resource} held by {owner} "
                    f"(older waits, wait-die)"
                )
            return False

        # younger transaction dies and restarts
        self._abort_and_restart(tx, blocker=owner, resource=resource)
        return True

    def _abort_and_restart(self, tx: Transaction, blocker: str, resource: str) -> None:
        self._log(
            f"{tx.tx_id}: abort/restart because {resource} held by older {blocker} "
            f"(younger dies, wait-die)"
        )
        tx.restart_count += 1
        self._release_all(tx)
        tx.pc = 0
        tx.status = "ready"
        tx.wait_resource = None

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

            # Remove stale waiting entries.
            if tx.status != "waiting" or tx.wait_resource != resource:
                queue.popleft()
                continue

            if resource in self.lock_owner:
                return

            queue.popleft()
            self.lock_owner[resource] = tx.tx_id
            tx.held_locks.add(resource)
            tx.status = "ready"
            tx.wait_resource = None
            tx.pc += 1
            self._log(f"{tx.tx_id}: wake up and granted lock on {resource}")
            return

    def wait_for_graph(self) -> Dict[str, Set[str]]:
        graph: Dict[str, Set[str]] = defaultdict(set)
        for tx in self.tx_by_id.values():
            if tx.status != "waiting" or tx.wait_resource is None:
                continue
            owner = self.lock_owner.get(tx.wait_resource)
            if owner is not None and owner != tx.tx_id:
                graph[tx.tx_id].add(owner)
        return graph

    def assert_wait_for_graph_acyclic(self) -> None:
        graph = self.wait_for_graph()
        state: Dict[str, int] = {}  # 0 unseen, 1 visiting, 2 done

        def dfs(node: str) -> bool:
            state[node] = 1
            for nxt in graph.get(node, set()):
                if state.get(nxt, 0) == 1:
                    return True
                if state.get(nxt, 0) == 0 and dfs(nxt):
                    return True
            state[node] = 2
            return False

        for node in list(graph):
            if state.get(node, 0) == 0 and dfs(node):
                raise AssertionError(f"wait-for graph has cycle: {graph}")

    def _log(self, message: str) -> None:
        self.event_log.append(message)


def build_demo_transactions() -> List[Transaction]:
    # T1 and T2 intentionally request locks in opposite order to create potential deadlock.
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
            Operation("work", note="independent task"),
            Operation("unlock", "C"),
            Operation("commit"),
        ],
    )

    return [t1, t2, t3]


def main() -> None:
    transactions = build_demo_transactions()
    scheduler = WaitDieScheduler(transactions)
    scheduler.run(max_rounds=100)

    for line in scheduler.event_log:
        print(line)

    print("\n--- summary ---")
    print("commit order:", " -> ".join(scheduler.commit_order))
    for tx_id in sorted(scheduler.tx_by_id):
        tx = scheduler.tx_by_id[tx_id]
        print(
            f"{tx_id}: status={tx.status}, restarts={tx.restart_count}, "
            f"held_locks={sorted(tx.held_locks)}"
        )

    # Self-checks for this deterministic demo scenario.
    assert all(tx.done for tx in scheduler.tx_by_id.values())
    assert scheduler.tx_by_id["T2"].restart_count >= 1
    assert scheduler.lock_owner == {}
    assert scheduler.wait_for_graph() == {}


if __name__ == "__main__":
    main()
