"""Two-Phase Commit (2PC) MVP simulation.

This script demonstrates a minimal 2PC protocol with:
- coordinator + participant state machines
- prepare / commit / abort message flow
- timeout handling
- coordinator crash and recovery behavior

The demo is deterministic and requires no interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence


class Vote(str, Enum):
    YES = "YES"
    NO = "NO"
    TIMEOUT = "TIMEOUT"


class TxnState(str, Enum):
    INIT = "INIT"
    READY = "READY"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


class Decision(str, Enum):
    COMMIT = "COMMIT"
    ABORT = "ABORT"


@dataclass(frozen=True)
class Operation:
    account: str
    delta: int


@dataclass(frozen=True)
class Transaction:
    txn_id: str
    operations_by_participant: Dict[str, List[Operation]]


class Participant:
    def __init__(self, name: str, balances: Dict[str, int]) -> None:
        self.name = name
        self.balances = dict(balances)
        self.crashed = False

        self.txn_state: Dict[str, TxnState] = {}
        self.local_decision: Dict[str, Decision] = {}
        self._snapshot_before_prepare: Dict[str, Dict[str, int]] = {}

    def set_crashed(self, crashed: bool) -> None:
        self.crashed = crashed

    def prepare(self, txn: Transaction) -> Vote:
        if self.crashed:
            return Vote.TIMEOUT

        txn_id = txn.txn_id
        current = self.txn_state.get(txn_id, TxnState.INIT)
        if current == TxnState.COMMITTED:
            return Vote.YES
        if current == TxnState.ABORTED:
            return Vote.NO

        ops = txn.operations_by_participant.get(self.name, [])
        trial = dict(self.balances)
        for op in ops:
            trial[op.account] = trial.get(op.account, 0) + op.delta
            if trial[op.account] < 0:
                self.txn_state[txn_id] = TxnState.ABORTED
                self.local_decision[txn_id] = Decision.ABORT
                return Vote.NO

        self._snapshot_before_prepare[txn_id] = dict(self.balances)
        self.txn_state[txn_id] = TxnState.READY
        return Vote.YES

    def apply_commit(self, txn: Transaction) -> bool:
        if self.crashed:
            return False

        txn_id = txn.txn_id
        state = self.txn_state.get(txn_id, TxnState.INIT)
        if state == TxnState.COMMITTED:
            return True
        if state != TxnState.READY:
            return False

        for op in txn.operations_by_participant.get(self.name, []):
            self.balances[op.account] = self.balances.get(op.account, 0) + op.delta

        self.txn_state[txn_id] = TxnState.COMMITTED
        self.local_decision[txn_id] = Decision.COMMIT
        self._snapshot_before_prepare.pop(txn_id, None)
        return True

    def apply_abort(self, txn_id: str) -> bool:
        if self.crashed:
            return False

        state = self.txn_state.get(txn_id, TxnState.INIT)
        if state == TxnState.COMMITTED:
            return False
        if state == TxnState.ABORTED:
            return True

        snapshot = self._snapshot_before_prepare.pop(txn_id, None)
        if snapshot is not None:
            self.balances = snapshot

        self.txn_state[txn_id] = TxnState.ABORTED
        self.local_decision[txn_id] = Decision.ABORT
        return True

    def blocked_transactions(self) -> List[str]:
        blocked: List[str] = []
        for txn_id, state in self.txn_state.items():
            if state == TxnState.READY and txn_id not in self.local_decision:
                blocked.append(txn_id)
        return blocked


class Coordinator:
    def __init__(self, participants: Dict[str, Participant]) -> None:
        self.participants = participants
        self.crashed = False
        self.decision_log: Dict[str, Decision] = {}
        self.event_log: List[str] = []

    def execute_2pc(self, txn: Transaction, crash_after_prepare: bool = False) -> str:
        if self.crashed:
            raise RuntimeError("Coordinator is crashed and cannot execute transactions.")

        txn_id = txn.txn_id
        self.event_log.append(f"{txn_id}:BEGIN_2PC")

        votes: Dict[str, Vote] = {}
        for name, participant in self.participants.items():
            vote = participant.prepare(txn)
            votes[name] = vote
        vote_summary = ", ".join(f"{k}={v.value}" for k, v in votes.items())
        self.event_log.append(f"{txn_id}:VOTES[{vote_summary}]")

        if crash_after_prepare:
            self.crashed = True
            self.event_log.append(f"{txn_id}:COORDINATOR_CRASH_AFTER_PREPARE")
            return "IN_DOUBT"

        all_yes = all(v == Vote.YES for v in votes.values())
        decision = Decision.COMMIT if all_yes else Decision.ABORT
        self.decision_log[txn_id] = decision
        self.event_log.append(f"{txn_id}:DECISION_{decision.value}")

        self._broadcast_decision(txn=txn, decision=decision)
        return decision.value

    def recover_and_finish(self, txn: Transaction) -> str:
        txn_id = txn.txn_id
        self.crashed = False

        decision = self.decision_log.get(txn_id)
        if decision is None:
            decision = Decision.ABORT
            self.decision_log[txn_id] = decision

        self.event_log.append(f"{txn_id}:RECOVERY_DECISION_{decision.value}")
        self._broadcast_decision(txn=txn, decision=decision)
        return decision.value

    def replay_decision(self, txn: Transaction) -> None:
        decision = self.decision_log.get(txn.txn_id)
        if decision is None:
            return
        self.event_log.append(f"{txn.txn_id}:REPLAY_{decision.value}")
        self._broadcast_decision(txn=txn, decision=decision)

    def _broadcast_decision(self, txn: Transaction, decision: Decision) -> None:
        for participant in self.participants.values():
            if decision == Decision.COMMIT:
                participant.apply_commit(txn)
            else:
                participant.apply_abort(txn.txn_id)


def total_balance(participants: Dict[str, Participant]) -> int:
    return sum(sum(p.balances.values()) for p in participants.values())


def cluster_snapshot(participants: Dict[str, Participant], txn_id: str) -> str:
    lines: List[str] = []
    for name in sorted(participants):
        p = participants[name]
        state = p.txn_state.get(txn_id, TxnState.INIT).value
        blocked = p.blocked_transactions()
        blocked_text = ",".join(blocked) if blocked else "-"
        lines.append(
            f"  {name}: balances={p.balances}, state[{txn_id}]={state}, blocked={blocked_text}, crashed={p.crashed}"
        )
    return "\n".join(lines)


def make_cluster() -> Dict[str, Participant]:
    return {
        "shard_A": Participant("shard_A", {"alice": 120}),
        "shard_B": Participant("shard_B", {"bob": 35}),
    }


def make_transfer_txn(txn_id: str, amount: int) -> Transaction:
    return Transaction(
        txn_id=txn_id,
        operations_by_participant={
            "shard_A": [Operation("alice", -amount)],
            "shard_B": [Operation("bob", amount)],
        },
    )


def run_case_happy_path() -> bool:
    print("\n=== CASE 1: 正常提交 (all YES -> COMMIT) ===")
    participants = make_cluster()
    coordinator = Coordinator(participants)
    txn = make_transfer_txn("T1", amount=40)

    before = total_balance(participants)
    decision = coordinator.execute_2pc(txn)
    after = total_balance(participants)

    print(f"decision={decision}, total_balance(before={before}, after={after})")
    print(cluster_snapshot(participants, txn.txn_id))

    passed = (
        decision == Decision.COMMIT.value
        and participants["shard_A"].balances["alice"] == 80
        and participants["shard_B"].balances["bob"] == 75
        and before == after
    )
    print(f"case_pass={passed}")
    return passed


def run_case_vote_abort() -> bool:
    print("\n=== CASE 2: 参与者投 NO -> 全局 ABORT ===")
    participants = make_cluster()
    coordinator = Coordinator(participants)
    txn = make_transfer_txn("T2", amount=500)

    before = total_balance(participants)
    decision = coordinator.execute_2pc(txn)
    after = total_balance(participants)

    print(f"decision={decision}, total_balance(before={before}, after={after})")
    print(cluster_snapshot(participants, txn.txn_id))

    passed = (
        decision == Decision.ABORT.value
        and participants["shard_A"].balances["alice"] == 120
        and participants["shard_B"].balances["bob"] == 35
        and before == after
    )
    print(f"case_pass={passed}")
    return passed


def run_case_coordinator_crash_blocking() -> bool:
    print("\n=== CASE 3: 协调者在 Prepare 后崩溃（阻塞）+ 恢复 ===")
    participants = make_cluster()
    coordinator = Coordinator(participants)
    txn = make_transfer_txn("T3", amount=30)

    before = total_balance(participants)
    decision1 = coordinator.execute_2pc(txn, crash_after_prepare=True)

    print(f"decision_before_recovery={decision1}")
    print("snapshot while coordinator is down:")
    print(cluster_snapshot(participants, txn.txn_id))

    blocked_any = any(p.blocked_transactions() for p in participants.values())
    decision2 = coordinator.recover_and_finish(txn)
    after = total_balance(participants)

    print(f"decision_after_recovery={decision2}, total_balance(before={before}, after={after})")
    print("snapshot after recovery:")
    print(cluster_snapshot(participants, txn.txn_id))

    passed = (
        decision1 == "IN_DOUBT"
        and blocked_any
        and decision2 == Decision.ABORT.value
        and participants["shard_A"].balances["alice"] == 120
        and participants["shard_B"].balances["bob"] == 35
        and before == after
    )
    print(f"case_pass={passed}")
    return passed


def run_case_timeout_and_replay() -> bool:
    print("\n=== CASE 4: Prepare 超时 -> ABORT，参与者恢复后重放决议 ===")
    participants = make_cluster()
    coordinator = Coordinator(participants)
    txn = make_transfer_txn("T4", amount=20)

    participants["shard_B"].set_crashed(True)
    before = total_balance(participants)
    decision = coordinator.execute_2pc(txn)

    print(f"decision={decision} (shard_B timeout during prepare)")
    print("snapshot before shard_B recovery:")
    print(cluster_snapshot(participants, txn.txn_id))

    participants["shard_B"].set_crashed(False)
    coordinator.replay_decision(txn)
    after = total_balance(participants)

    print("snapshot after shard_B recovery + decision replay:")
    print(cluster_snapshot(participants, txn.txn_id))

    passed = (
        decision == Decision.ABORT.value
        and participants["shard_A"].balances["alice"] == 120
        and participants["shard_B"].balances["bob"] == 35
        and before == after
    )
    print(f"case_pass={passed}")
    return passed


def print_event_logs(logs: Sequence[str]) -> None:
    print("\nEvent log example entries:")
    for item in logs:
        print(f"  - {item}")


def main() -> None:
    print("2PC MVP demo: coordinator/participant protocol simulation")

    results = [
        run_case_happy_path(),
        run_case_vote_abort(),
        run_case_coordinator_crash_blocking(),
        run_case_timeout_and_replay(),
    ]

    # Build one standalone log sample for display.
    sample_participants = make_cluster()
    sample_coordinator = Coordinator(sample_participants)
    sample_txn = make_transfer_txn("T_LOG", amount=10)
    sample_coordinator.execute_2pc(sample_txn)
    print_event_logs(sample_coordinator.event_log)

    all_passed = all(results)
    print(f"\nGlobal checks pass: {all_passed}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
