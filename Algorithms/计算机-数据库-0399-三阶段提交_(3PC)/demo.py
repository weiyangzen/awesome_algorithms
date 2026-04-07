"""Three-Phase Commit (3PC) MVP simulation.

This script demonstrates a deterministic 3PC protocol with:
- coordinator + participant finite state machines
- can-commit / pre-commit / do-commit message flow
- timeout-driven local transition behavior
- coordinator crash points around phase boundaries

The demo is self-contained and requires no interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence


class Vote(str, Enum):
    YES = "YES"
    NO = "NO"
    TIMEOUT = "TIMEOUT"


class ParticipantState(str, Enum):
    INIT = "INIT"
    WAIT = "WAIT"
    PRECOMMIT = "PRECOMMIT"
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

        self.state: Dict[str, ParticipantState] = {}
        self.local_decision: Dict[str, Decision] = {}
        self.pending_txn: Dict[str, Transaction] = {}

    def set_crashed(self, crashed: bool) -> None:
        self.crashed = crashed

    def can_commit(self, txn: Transaction) -> Vote:
        if self.crashed:
            return Vote.TIMEOUT

        txn_id = txn.txn_id
        current = self.state.get(txn_id, ParticipantState.INIT)
        if current == ParticipantState.COMMITTED:
            return Vote.YES
        if current == ParticipantState.ABORTED:
            return Vote.NO

        trial = dict(self.balances)
        for op in txn.operations_by_participant.get(self.name, []):
            trial[op.account] = trial.get(op.account, 0) + op.delta
            if trial[op.account] < 0:
                self.state[txn_id] = ParticipantState.ABORTED
                self.local_decision[txn_id] = Decision.ABORT
                self.pending_txn.pop(txn_id, None)
                return Vote.NO

        self.pending_txn[txn_id] = txn
        self.state[txn_id] = ParticipantState.WAIT
        return Vote.YES

    def receive_precommit(self, txn_id: str) -> bool:
        if self.crashed:
            return False

        current = self.state.get(txn_id, ParticipantState.INIT)
        if current == ParticipantState.PRECOMMIT:
            return True
        if current != ParticipantState.WAIT:
            return False

        self.state[txn_id] = ParticipantState.PRECOMMIT
        return True

    def receive_do_commit(self, txn_id: str) -> bool:
        if self.crashed:
            return False

        current = self.state.get(txn_id, ParticipantState.INIT)
        if current == ParticipantState.COMMITTED:
            return True
        if current not in (ParticipantState.PRECOMMIT, ParticipantState.WAIT):
            return False

        txn = self.pending_txn.get(txn_id)
        if txn is None:
            return False

        for op in txn.operations_by_participant.get(self.name, []):
            self.balances[op.account] = self.balances.get(op.account, 0) + op.delta

        self.state[txn_id] = ParticipantState.COMMITTED
        self.local_decision[txn_id] = Decision.COMMIT
        self.pending_txn.pop(txn_id, None)
        return True

    def receive_abort(self, txn_id: str) -> bool:
        if self.crashed:
            return False

        current = self.state.get(txn_id, ParticipantState.INIT)
        if current == ParticipantState.COMMITTED:
            return False
        if current == ParticipantState.ABORTED:
            return True

        self.state[txn_id] = ParticipantState.ABORTED
        self.local_decision[txn_id] = Decision.ABORT
        self.pending_txn.pop(txn_id, None)
        return True

    def on_timeout(self, txn_id: str) -> bool:
        """3PC timeout rule under bounded-delay assumption.

        WAIT      -> ABORT
        PRECOMMIT -> COMMIT
        """
        if self.crashed:
            return False

        current = self.state.get(txn_id, ParticipantState.INIT)
        if current == ParticipantState.WAIT:
            return self.receive_abort(txn_id)
        if current == ParticipantState.PRECOMMIT:
            return self.receive_do_commit(txn_id)
        return False

    def blocked_transactions(self) -> List[str]:
        blocked: List[str] = []
        for txn_id, state in self.state.items():
            if state in (ParticipantState.WAIT, ParticipantState.PRECOMMIT) and txn_id not in self.local_decision:
                blocked.append(txn_id)
        return blocked


class Coordinator:
    def __init__(self, participants: Dict[str, Participant]) -> None:
        self.participants = participants
        self.crashed = False
        self.decision_log: Dict[str, Decision] = {}
        self.event_log: List[str] = []

    def execute_3pc(self, txn: Transaction, crash_point: str | None = None) -> str:
        if self.crashed:
            raise RuntimeError("Coordinator is crashed and cannot execute transactions.")

        txn_id = txn.txn_id
        self.event_log.append(f"{txn_id}:BEGIN_3PC")

        votes: Dict[str, Vote] = {}
        for name, participant in self.participants.items():
            votes[name] = participant.can_commit(txn)
        vote_summary = ", ".join(f"{k}={v.value}" for k, v in votes.items())
        self.event_log.append(f"{txn_id}:PHASE1_CAN_COMMIT[{vote_summary}]")

        if crash_point == "after_cancommit":
            self.crashed = True
            self.event_log.append(f"{txn_id}:COORDINATOR_CRASH_AFTER_PHASE1")
            return "IN_DOUBT_CANCOMMIT"

        if not all(v == Vote.YES for v in votes.values()):
            self.decision_log[txn_id] = Decision.ABORT
            self.event_log.append(f"{txn_id}:DECISION_ABORT_FROM_PHASE1")
            self._broadcast_abort(txn_id)
            return Decision.ABORT.value

        acks: Dict[str, bool] = {}
        for name, participant in self.participants.items():
            acks[name] = participant.receive_precommit(txn_id)
        ack_summary = ", ".join(f"{k}={'ACK' if ok else 'TIMEOUT'}" for k, ok in acks.items())
        self.event_log.append(f"{txn_id}:PHASE2_PRECOMMIT[{ack_summary}]")

        if crash_point == "after_precommit":
            self.crashed = True
            self.event_log.append(f"{txn_id}:COORDINATOR_CRASH_AFTER_PHASE2")
            return "IN_DOUBT_PRECOMMIT"

        if not all(acks.values()):
            self.decision_log[txn_id] = Decision.ABORT
            self.event_log.append(f"{txn_id}:DECISION_ABORT_FROM_PHASE2")
            self._broadcast_abort(txn_id)
            return Decision.ABORT.value

        self.decision_log[txn_id] = Decision.COMMIT
        self.event_log.append(f"{txn_id}:DECISION_COMMIT")
        self._broadcast_commit(txn_id)
        return Decision.COMMIT.value

    def observe_and_record_decision(self, txn_id: str) -> str:
        """Recover decision from participant end states in this toy simulator."""
        existing = self.decision_log.get(txn_id)
        if existing is not None:
            return existing.value

        states = [p.state.get(txn_id, ParticipantState.INIT) for p in self.participants.values()]
        if states and all(s == ParticipantState.COMMITTED for s in states):
            self.decision_log[txn_id] = Decision.COMMIT
            self.event_log.append(f"{txn_id}:RECOVERED_DECISION_COMMIT")
            return Decision.COMMIT.value

        if all(s in (ParticipantState.ABORTED, ParticipantState.INIT) for s in states):
            self.decision_log[txn_id] = Decision.ABORT
            self.event_log.append(f"{txn_id}:RECOVERED_DECISION_ABORT")
            return Decision.ABORT.value

        self.event_log.append(f"{txn_id}:RECOVERED_DECISION_UNKNOWN")
        return "UNKNOWN"

    def replay_decision(self, txn: Transaction) -> None:
        decision = self.decision_log.get(txn.txn_id)
        if decision is None:
            return
        self.event_log.append(f"{txn.txn_id}:REPLAY_{decision.value}")
        if decision == Decision.COMMIT:
            self._broadcast_commit(txn.txn_id)
        else:
            self._broadcast_abort(txn.txn_id)

    def _broadcast_commit(self, txn_id: str) -> None:
        for participant in self.participants.values():
            participant.receive_do_commit(txn_id)

    def _broadcast_abort(self, txn_id: str) -> None:
        for participant in self.participants.values():
            participant.receive_abort(txn_id)


def total_balance(participants: Dict[str, Participant]) -> int:
    return sum(sum(p.balances.values()) for p in participants.values())


def cluster_snapshot(participants: Dict[str, Participant], txn_id: str) -> str:
    lines: List[str] = []
    for name in sorted(participants):
        p = participants[name]
        state = p.state.get(txn_id, ParticipantState.INIT).value
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
    print("\n=== CASE 1: 正常路径 (all YES -> PRECOMMIT -> COMMIT) ===")
    participants = make_cluster()
    coordinator = Coordinator(participants)
    txn = make_transfer_txn("T1", amount=40)

    before = total_balance(participants)
    decision = coordinator.execute_3pc(txn)
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
    print("\n=== CASE 2: 阶段一否决 (NO -> ABORT) ===")
    participants = make_cluster()
    coordinator = Coordinator(participants)
    txn = make_transfer_txn("T2", amount=500)

    before = total_balance(participants)
    decision = coordinator.execute_3pc(txn)
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


def run_case_crash_after_cancommit_timeout_abort() -> bool:
    print("\n=== CASE 3: 协调者在阶段一后崩溃，参与者超时 ABORT ===")
    participants = make_cluster()
    coordinator = Coordinator(participants)
    txn = make_transfer_txn("T3", amount=20)

    before = total_balance(participants)
    decision_before = coordinator.execute_3pc(txn, crash_point="after_cancommit")

    blocked_before = any(p.blocked_transactions() for p in participants.values())
    for participant in participants.values():
        participant.on_timeout(txn.txn_id)
    after = total_balance(participants)

    print(f"decision_before_timeout={decision_before}")
    print(cluster_snapshot(participants, txn.txn_id))

    passed = (
        decision_before == "IN_DOUBT_CANCOMMIT"
        and blocked_before
        and all(p.state[txn.txn_id] == ParticipantState.ABORTED for p in participants.values())
        and participants["shard_A"].balances["alice"] == 120
        and participants["shard_B"].balances["bob"] == 35
        and before == after
    )
    print(f"case_pass={passed}")
    return passed


def run_case_crash_after_precommit_timeout_commit() -> bool:
    print("\n=== CASE 4: 协调者在 PRECOMMIT 后崩溃，参与者超时 COMMIT ===")
    participants = make_cluster()
    coordinator = Coordinator(participants)
    txn = make_transfer_txn("T4", amount=30)

    before = total_balance(participants)
    decision_before = coordinator.execute_3pc(txn, crash_point="after_precommit")

    blocked_before = any(p.blocked_transactions() for p in participants.values())
    for participant in participants.values():
        participant.on_timeout(txn.txn_id)

    coordinator.crashed = False
    recovered_decision = coordinator.observe_and_record_decision(txn.txn_id)
    coordinator.replay_decision(txn)
    after = total_balance(participants)

    print(
        f"decision_before_timeout={decision_before}, recovered_decision={recovered_decision}, "
        f"total_balance(before={before}, after={after})"
    )
    print(cluster_snapshot(participants, txn.txn_id))

    passed = (
        decision_before == "IN_DOUBT_PRECOMMIT"
        and blocked_before
        and recovered_decision == Decision.COMMIT.value
        and all(p.state[txn.txn_id] == ParticipantState.COMMITTED for p in participants.values())
        and participants["shard_A"].balances["alice"] == 90
        and participants["shard_B"].balances["bob"] == 65
        and before == after
    )
    print(f"case_pass={passed}")
    return passed


def print_event_logs(logs: Sequence[str]) -> None:
    print("\nEvent log example entries:")
    for item in logs:
        print(f"  - {item}")


def main() -> None:
    print("3PC MVP demo: coordinator/participant protocol simulation")

    results = [
        run_case_happy_path(),
        run_case_vote_abort(),
        run_case_crash_after_cancommit_timeout_abort(),
        run_case_crash_after_precommit_timeout_commit(),
    ]

    sample_participants = make_cluster()
    sample_coordinator = Coordinator(sample_participants)
    sample_txn = make_transfer_txn("T_LOG", amount=10)
    sample_coordinator.execute_3pc(sample_txn)
    print_event_logs(sample_coordinator.event_log)

    all_passed = all(results)
    print(f"\nGlobal checks pass: {all_passed}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
