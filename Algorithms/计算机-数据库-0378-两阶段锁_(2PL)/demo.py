"""Minimal runnable MVP for Two-Phase Locking (2PL).

This script implements a tiny strict-2PL scheduler with:
- shared/exclusive locks
- lock upgrade (S -> X)
- FIFO wait queues
- deadlock detection (wait-for graph cycle) and abort

Run:
    uv run python demo.py
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple


SHARED = "S"
EXCLUSIVE = "X"


@dataclass(frozen=True)
class Operation:
    kind: str  # "R" | "W" | "C"
    item: Optional[str] = None


@dataclass
class Txn:
    name: str
    ops: List[Operation]
    pc: int = 0
    state: str = "active"  # active | committed | aborted

    def current_op(self) -> Optional[Operation]:
        if self.pc >= len(self.ops):
            return None
        return self.ops[self.pc]


@dataclass
class GrantedLock:
    tx: str
    mode: str  # S | X


@dataclass
class LockRequest:
    tx: str
    item: str
    mode: str  # S | X
    upgrade: bool = False


@dataclass
class ItemLockState:
    granted: List[GrantedLock]
    waiting: Deque[LockRequest]


class LockManager:
    def __init__(self) -> None:
        self.table: Dict[str, ItemLockState] = defaultdict(
            lambda: ItemLockState(granted=[], waiting=deque())
        )
        self.held_by_tx: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.pending_by_tx: Dict[str, LockRequest] = {}
        self.phase_by_tx: Dict[str, str] = defaultdict(lambda: "growing")

    @staticmethod
    def _compatible(request_mode: str, existing_modes: List[str]) -> bool:
        if request_mode == SHARED:
            return all(m == SHARED for m in existing_modes)
        return len(existing_modes) == 0

    def _find_granted_lock(self, item: str, tx: str) -> Optional[GrantedLock]:
        for lock in self.table[item].granted:
            if lock.tx == tx:
                return lock
        return None

    def request_lock(self, tx: str, item: str, mode: str) -> Tuple[bool, str]:
        if self.phase_by_tx[tx] == "shrinking":
            raise RuntimeError(f"{tx} is in shrinking phase and cannot acquire new locks")

        state = self.table[item]
        held_mode = self.held_by_tx[tx].get(item)

        # Already holds enough lock.
        if held_mode == EXCLUSIVE or held_mode == mode:
            return True, f"{tx} already has {held_mode}({item})"

        # Handle upgrade S -> X.
        if held_mode == SHARED and mode == EXCLUSIVE:
            others = [l for l in state.granted if l.tx != tx]
            if not others and not state.waiting:
                own = self._find_granted_lock(item, tx)
                if own is None:
                    raise RuntimeError("internal error: missing own shared lock during upgrade")
                own.mode = EXCLUSIVE
                self.held_by_tx[tx][item] = EXCLUSIVE
                return True, f"{tx} upgrades S->{EXCLUSIVE} on {item}"

            req = LockRequest(tx=tx, item=item, mode=EXCLUSIVE, upgrade=True)
            # Prioritize upgrade request slightly to reduce upgrade starvation.
            state.waiting.appendleft(req)
            self.pending_by_tx[tx] = req
            return False, f"{tx} waits to upgrade lock on {item}"

        # FIFO fairness: if queue is not empty, queue this request.
        if state.waiting:
            req = LockRequest(tx=tx, item=item, mode=mode)
            state.waiting.append(req)
            self.pending_by_tx[tx] = req
            return False, f"{tx} waits for {mode}({item}) due to FIFO queue"

        existing_modes = [l.mode for l in state.granted if l.tx != tx]
        if self._compatible(mode, existing_modes):
            state.granted.append(GrantedLock(tx=tx, mode=mode))
            self.held_by_tx[tx][item] = mode
            return True, f"{tx} granted {mode}({item})"

        req = LockRequest(tx=tx, item=item, mode=mode)
        state.waiting.append(req)
        self.pending_by_tx[tx] = req
        return False, f"{tx} waits for {mode}({item})"

    def begin_shrinking(self, tx: str) -> None:
        self.phase_by_tx[tx] = "shrinking"

    def release_all(self, tx: str) -> List[str]:
        granted_now: List[str] = []

        # Remove pending request for this tx (e.g., abort path).
        self.pending_by_tx.pop(tx, None)

        touched_items = list(self.held_by_tx[tx].keys())
        for item in touched_items:
            state = self.table[item]
            state.granted = [g for g in state.granted if g.tx != tx]
            self.held_by_tx[tx].pop(item, None)
            granted_now.extend(self._try_grant_waiting(item))

        # Also remove this tx from wait queues if present without held locks.
        for item, state in self.table.items():
            before = len(state.waiting)
            state.waiting = deque([req for req in state.waiting if req.tx != tx])
            if len(state.waiting) != before:
                granted_now.extend(self._try_grant_waiting(item))

        return granted_now

    def _try_grant_waiting(self, item: str) -> List[str]:
        state = self.table[item]
        granted_events: List[str] = []

        while state.waiting:
            req = state.waiting[0]
            existing = [g.mode for g in state.granted if g.tx != req.tx]

            if not self._compatible(req.mode, existing):
                break

            state.waiting.popleft()
            self.pending_by_tx.pop(req.tx, None)

            if req.upgrade:
                own = self._find_granted_lock(item, req.tx)
                if own is None:
                    raise RuntimeError("internal error: upgrade requester has no existing lock")
                own.mode = EXCLUSIVE
                self.held_by_tx[req.tx][item] = EXCLUSIVE
                granted_events.append(f"{req.tx} upgrade on {item} granted from queue")
            else:
                state.granted.append(GrantedLock(tx=req.tx, mode=req.mode))
                self.held_by_tx[req.tx][item] = req.mode
                granted_events.append(f"{req.tx} granted {req.mode}({item}) from queue")

            # If we just granted an exclusive request, no more grants on this item now.
            if req.mode == EXCLUSIVE:
                break

        return granted_events

    def wait_for_graph(self) -> Dict[str, Set[str]]:
        graph: Dict[str, Set[str]] = defaultdict(set)
        for req in self.pending_by_tx.values():
            holders = self.table[req.item].granted
            for h in holders:
                if h.tx == req.tx:
                    continue
                if req.mode == EXCLUSIVE or h.mode == EXCLUSIVE:
                    graph[req.tx].add(h.tx)
            graph.setdefault(req.tx, set())
        return graph


def find_cycle(graph: Dict[str, Set[str]]) -> List[str]:
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in graph}
    parent: Dict[str, Optional[str]] = {n: None for n in graph}

    def dfs(u: str) -> List[str]:
        color[u] = GRAY
        for v in graph[u]:
            if color.get(v, WHITE) == WHITE:
                color[v] = WHITE
                parent[v] = u
                cycle = dfs(v)
                if cycle:
                    return cycle
            elif color.get(v) == GRAY:
                # reconstruct cycle u -> ... -> v
                path = [v]
                x = u
                while x is not None and x != v:
                    path.append(x)
                    x = parent.get(x)
                path.append(v)
                path.reverse()
                return path
        color[u] = BLACK
        return []

    for node in list(graph.keys()):
        if color[node] == WHITE:
            cycle = dfs(node)
            if cycle:
                return cycle
    return []


def precedence_graph(history: List[Tuple[str, str, str]]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)

    for i in range(len(history)):
        tx_i, op_i, item_i = history[i]
        graph.setdefault(tx_i, set())
        for j in range(i + 1, len(history)):
            tx_j, op_j, item_j = history[j]
            if tx_i == tx_j or item_i != item_j:
                continue
            if op_i == "R" and op_j == "R":
                continue
            graph[tx_i].add(tx_j)
            graph.setdefault(tx_j, set())

    return graph


def topo_sort(graph: Dict[str, Set[str]]) -> List[str]:
    indeg: Dict[str, int] = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            indeg[v] = indeg.get(v, 0) + 1

    queue = deque(sorted([u for u, d in indeg.items() if d == 0]))
    order: List[str] = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in sorted(graph.get(u, [])):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    if len(order) != len(indeg):
        return []
    return order


def run_schedule(transactions: List[Txn], db: Dict[str, int]) -> None:
    lm = LockManager()
    tx_map: Dict[str, Txn] = {t.name: t for t in transactions}
    tx_order = [t.name for t in transactions]
    write_delta = {name: (idx + 1) * 10 for idx, name in enumerate(tx_order)}

    event_log: List[str] = []
    history: List[Tuple[str, str, str]] = []  # (tx, op, item) for R/W only

    rounds = 0
    while True:
        rounds += 1
        if rounds > 200:
            raise RuntimeError("scheduler exceeded max rounds")

        if all(tx.state in {"committed", "aborted"} for tx in tx_map.values()):
            break

        made_progress = False

        for tx_name in tx_order:
            tx = tx_map[tx_name]
            if tx.state in {"committed", "aborted"}:
                continue

            if tx_name in lm.pending_by_tx:
                req = lm.pending_by_tx[tx_name]
                event_log.append(f"{tx_name} blocked on {req.mode}({req.item})")
                continue

            op = tx.current_op()
            if op is None:
                tx.state = "committed"
                continue

            if op.kind in {"R", "W"}:
                if op.item is None:
                    raise RuntimeError(f"{tx_name} operation {op.kind} missing item")

                needed_mode = SHARED if op.kind == "R" else EXCLUSIVE
                granted, msg = lm.request_lock(tx_name, op.item, needed_mode)
                event_log.append(msg)

                if not granted:
                    continue

                if op.kind == "R":
                    value = db[op.item]
                    event_log.append(f"{tx_name} executes R({op.item}) -> {value}")
                else:
                    db[op.item] += write_delta[tx_name]
                    event_log.append(f"{tx_name} executes W({op.item}) -> {db[op.item]}")

                history.append((tx_name, op.kind, op.item))
                tx.pc += 1
                made_progress = True

            elif op.kind == "C":
                lm.begin_shrinking(tx_name)
                granted_events = lm.release_all(tx_name)
                tx.pc += 1
                tx.state = "committed"
                event_log.append(f"{tx_name} commits and releases all locks")
                event_log.extend(granted_events)
                made_progress = True

            else:
                raise RuntimeError(f"Unknown operation kind: {op.kind}")

        if made_progress:
            continue

        # No progress -> check deadlock.
        wfg = lm.wait_for_graph()
        cycle = find_cycle(wfg)
        if cycle:
            # Abort a deterministic victim: lexicographically largest tx id in cycle.
            victim = sorted(set(cycle))[-1]
            tx_map[victim].state = "aborted"
            tx_map[victim].pc = len(tx_map[victim].ops)
            released = lm.release_all(victim)
            event_log.append(f"deadlock detected cycle={cycle}; abort {victim}")
            event_log.extend(released)
            continue

        raise RuntimeError("No progress and no deadlock; schedule cannot proceed")

    # Summaries
    print("=== Two-Phase Locking (2PL) MVP Demo ===")
    print("Initial transactions:")
    for tx in transactions:
        print(f"  {tx.name}:", " ".join(f"{op.kind}({op.item})" if op.item else op.kind for op in tx.ops))

    print("\nExecution log:")
    for i, line in enumerate(event_log, start=1):
        print(f"  {i:02d}. {line}")

    print("\nFinal transaction states:")
    for tx in transactions:
        print(f"  {tx.name}: {tx.state}")

    print("\nFinal database state:")
    for item in sorted(db):
        print(f"  {item} = {db[item]}")

    pg = precedence_graph(history)
    order = topo_sort(pg)
    edges = sorted((u, v) for u, vs in pg.items() for v in vs)

    print("\nConflict edges (precedence graph):")
    if edges:
        for u, v in edges:
            print(f"  {u} -> {v}")
    else:
        print("  (none)")

    if order:
        print("Serializable equivalent order:", " -> ".join(order))
    else:
        print("Serializable equivalent order: not found (cycle in precedence graph)")


def main() -> None:
    # Scenario intentionally creates lock contention and lock upgrade waiting.
    transactions = [
        Txn("T1", [Operation("R", "A"), Operation("W", "A"), Operation("C")]),
        Txn("T2", [Operation("R", "A"), Operation("W", "B"), Operation("C")]),
    ]
    db = {"A": 0, "B": 0}

    run_schedule(transactions, db)


if __name__ == "__main__":
    main()
