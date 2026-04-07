"""Byzantine fault tolerance MVP (PBFT-style) for CS-0317.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib

import numpy as np


def digest_request(payload: str) -> str:
    """Return a short stable digest for a client request."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class ReplicaState:
    """Local PBFT-related state for one replica."""

    replica_id: str
    is_byzantine: bool = False
    accepted_preprepare: tuple[int, int, str] | None = None
    prepare_senders: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    commit_senders: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    prepared_digest: str | None = None
    committed_digest: str | None = None
    executed_digest: str | None = None


class PBFTSimulator:
    """Deterministic PBFT-style simulation with one Byzantine replica."""

    def __init__(
        self,
        replica_ids: list[str],
        f: int,
        primary_id: str,
        byzantine_ids: set[str],
        view: int = 1,
        seq: int = 10,
    ) -> None:
        if len(set(replica_ids)) != len(replica_ids):
            raise ValueError(f"Duplicate replica ids are not allowed: {replica_ids}")
        if primary_id not in replica_ids:
            raise ValueError(f"Primary {primary_id} must be in replica ids: {replica_ids}")
        if len(replica_ids) < 3 * f + 1:
            raise ValueError(f"Need n >= 3f+1, got n={len(replica_ids)}, f={f}")

        self.replica_ids = replica_ids
        self.index = {rid: i for i, rid in enumerate(replica_ids)}
        self.f = f
        self.primary_id = primary_id
        self.view = view
        self.seq = seq
        self.prepared_threshold = 2 * f
        self.committed_threshold = 2 * f + 1
        self.replicas: dict[str, ReplicaState] = {
            rid: ReplicaState(replica_id=rid, is_byzantine=(rid in byzantine_ids))
            for rid in replica_ids
        }

        n = len(replica_ids)
        self.prepare_good = np.zeros((n, n), dtype=int)
        self.prepare_bad = np.zeros((n, n), dtype=int)
        self.commit_good = np.zeros((n, n), dtype=int)
        self.commit_bad = np.zeros((n, n), dtype=int)
        self.event_log: list[str] = []

    def _is_honest(self, rid: str) -> bool:
        return not self.replicas[rid].is_byzantine

    def _matrix_set(self, matrix: np.ndarray, receiver: str, sender: str) -> None:
        matrix[self.index[receiver], self.index[sender]] = 1

    def deliver_preprepare(self, sender: str, receiver: str, digest: str) -> None:
        """Deliver one pre-prepare message to a receiver."""
        state = self.replicas[receiver]
        if sender != self.primary_id:
            self.event_log.append(f"PRE-PREPARE ignored at {receiver}: sender {sender} is not primary")
            return

        if state.accepted_preprepare is None:
            state.accepted_preprepare = (self.view, self.seq, digest)
            self.event_log.append(
                f"{receiver} accepted PRE-PREPARE(v={self.view},n={self.seq},d={digest})"
            )
        elif state.accepted_preprepare[2] != digest:
            self.event_log.append(
                f"{receiver} rejected conflicting PRE-PREPARE digest={digest}, "
                f"keeps {state.accepted_preprepare[2]}"
            )

    def deliver_prepare(self, sender: str, receiver: str, digest: str, good_digest: str) -> None:
        """Deliver one prepare vote to a receiver and count it when valid."""
        state = self.replicas[receiver]
        if state.accepted_preprepare is None:
            self.event_log.append(f"PREPARE ignored at {receiver}: no accepted PRE-PREPARE yet")
            return
        if state.accepted_preprepare[2] != digest:
            self.event_log.append(
                f"PREPARE ignored at {receiver}: digest mismatch {digest} != {state.accepted_preprepare[2]}"
            )
            return

        state.prepare_senders[digest].add(sender)
        if digest == good_digest:
            self._matrix_set(self.prepare_good, receiver, sender)
        else:
            self._matrix_set(self.prepare_bad, receiver, sender)

    def deliver_commit(self, sender: str, receiver: str, digest: str, good_digest: str) -> None:
        """Deliver one commit vote to a receiver and count it when valid."""
        state = self.replicas[receiver]
        if state.accepted_preprepare is None:
            self.event_log.append(f"COMMIT ignored at {receiver}: no accepted PRE-PREPARE yet")
            return
        if state.accepted_preprepare[2] != digest:
            self.event_log.append(
                f"COMMIT ignored at {receiver}: digest mismatch {digest} != {state.accepted_preprepare[2]}"
            )
            return

        state.commit_senders[digest].add(sender)
        if digest == good_digest:
            self._matrix_set(self.commit_good, receiver, sender)
        else:
            self._matrix_set(self.commit_bad, receiver, sender)

    def broadcast_preprepare(self, digest: str) -> None:
        for receiver in self.replica_ids:
            self.deliver_preprepare(self.primary_id, receiver, digest)

    def broadcast_prepare_honest(self, digest: str, only_honest_sender: bool = True) -> None:
        """All honest replicas with accepted pre-prepare broadcast prepare(digest)."""
        for sender in self.replica_ids:
            if only_honest_sender and not self._is_honest(sender):
                continue
            if self.replicas[sender].accepted_preprepare is None:
                continue
            for receiver in self.replica_ids:
                self.deliver_prepare(sender, receiver, digest, digest)

    def broadcast_prepare_byzantine(self, good_digest: str, bad_digest: str) -> None:
        """Byzantine replicas send conflicting prepare digests to different receivers."""
        byzantine_ids = [rid for rid in self.replica_ids if not self._is_honest(rid)]
        for b in byzantine_ids:
            for receiver in self.replica_ids:
                if receiver in (self.primary_id, b):
                    sent_digest = good_digest
                elif receiver == "R1":
                    sent_digest = bad_digest
                else:
                    sent_digest = good_digest
                self.deliver_prepare(b, receiver, sent_digest, good_digest)

    def update_prepared(self, digest: str) -> None:
        for rid in self.replica_ids:
            state = self.replicas[rid]
            if state.accepted_preprepare is None:
                continue
            votes = len(state.prepare_senders[digest])
            if votes >= self.prepared_threshold:
                state.prepared_digest = digest
                self.event_log.append(f"{rid} reached PREPARED for digest={digest}, votes={votes}")

    def broadcast_commit_honest(self, digest: str) -> None:
        for sender in self.replica_ids:
            state = self.replicas[sender]
            if not self._is_honest(sender):
                continue
            if state.prepared_digest != digest:
                continue
            for receiver in self.replica_ids:
                self.deliver_commit(sender, receiver, digest, digest)

    def broadcast_commit_byzantine(self, good_digest: str, bad_digest: str) -> None:
        byzantine_ids = [rid for rid in self.replica_ids if not self._is_honest(rid)]
        for b in byzantine_ids:
            for receiver in self.replica_ids:
                if receiver == "R2":
                    sent_digest = bad_digest
                else:
                    sent_digest = good_digest
                self.deliver_commit(b, receiver, sent_digest, good_digest)

    def update_committed_and_execute(self, digest: str) -> None:
        for rid in self.replica_ids:
            state = self.replicas[rid]
            if state.prepared_digest != digest:
                continue
            votes = len(state.commit_senders[digest])
            if votes >= self.committed_threshold:
                state.committed_digest = digest
                state.executed_digest = digest
                self.event_log.append(f"{rid} reached COMMITTED for digest={digest}, votes={votes}")

    def assert_safety(self, good_digest: str, bad_digest: str) -> None:
        honest_ids = [rid for rid in self.replica_ids if self._is_honest(rid)]
        committed_honest = [
            rid for rid in honest_ids if self.replicas[rid].committed_digest == good_digest
        ]
        assert len(committed_honest) >= self.committed_threshold, (
            f"Need at least {self.committed_threshold} honest commits, got {len(committed_honest)}: "
            f"{committed_honest}"
        )

        for rid in honest_ids:
            state = self.replicas[rid]
            assert state.prepared_digest != bad_digest, f"{rid} should not be prepared on bad digest"
            assert state.committed_digest != bad_digest, f"{rid} should not commit bad digest"
            assert state.executed_digest == good_digest, f"{rid} should execute good digest"


def print_matrix(title: str, matrix: np.ndarray, replica_ids: list[str]) -> None:
    print(f"\n=== {title} ===")
    header = "recv\\send " + " ".join(f"{rid:>3}" for rid in replica_ids)
    print(header)
    print("-" * len(header))
    for i, rid in enumerate(replica_ids):
        row = " ".join(f"{int(v):>3}" for v in matrix[i].tolist())
        print(f"{rid:>9} {row}")


def print_replica_summary(sim: PBFTSimulator, good_digest: str) -> None:
    print("\n=== Replica Summary ===")
    header = f"{'replica':<8} {'role':<10} {'prepared':<18} {'committed':<18} {'executed':<18}"
    print(header)
    print("-" * len(header))
    for rid in sim.replica_ids:
        st = sim.replicas[rid]
        role = "byzantine" if st.is_byzantine else "honest"
        prepared = st.prepared_digest or "-"
        committed = st.committed_digest or "-"
        executed = st.executed_digest or "-"
        print(f"{rid:<8} {role:<10} {prepared:<18} {committed:<18} {executed:<18}")

    honest_ids = [rid for rid in sim.replica_ids if sim._is_honest(rid)]
    success = [
        rid for rid in honest_ids if sim.replicas[rid].committed_digest == good_digest
    ]
    print(
        f"\nHonest committed replicas: {success} "
        f"(need >= {sim.committed_threshold} for f={sim.f})."
    )


def main() -> None:
    replica_ids = ["R0", "R1", "R2", "R3"]
    sim = PBFTSimulator(
        replica_ids=replica_ids,
        f=1,
        primary_id="R0",
        byzantine_ids={"R3"},
        view=1,
        seq=10,
    )

    request = "transfer:alice->bob:10"
    good_digest = digest_request(request)
    bad_digest = digest_request(request + ":tampered")

    sim.broadcast_preprepare(good_digest)
    sim.broadcast_prepare_honest(good_digest)
    sim.broadcast_prepare_byzantine(good_digest=good_digest, bad_digest=bad_digest)
    sim.update_prepared(good_digest)

    sim.broadcast_commit_honest(good_digest)
    sim.broadcast_commit_byzantine(good_digest=good_digest, bad_digest=bad_digest)
    sim.update_committed_and_execute(good_digest)

    sim.assert_safety(good_digest=good_digest, bad_digest=bad_digest)

    print("=== PBFT Byzantine Fault Tolerance Demo ===")
    print(
        f"n={len(replica_ids)}, f={sim.f}, primary={sim.primary_id}, "
        f"prepared_threshold={sim.prepared_threshold}, committed_threshold={sim.committed_threshold}"
    )
    print(f"request={request}")
    print(f"good_digest={good_digest}, bad_digest={bad_digest}")
    print("\nAll safety assertions passed.")

    print_matrix("Prepare Matrix (good digest accepted)", sim.prepare_good, replica_ids)
    print_matrix("Prepare Matrix (bad digest accepted)", sim.prepare_bad, replica_ids)
    print_matrix("Commit Matrix (good digest accepted)", sim.commit_good, replica_ids)
    print_matrix("Commit Matrix (bad digest accepted)", sim.commit_bad, replica_ids)
    print_replica_summary(sim, good_digest=good_digest)

    print("\nSample event log (first 10 lines):")
    for line in sim.event_log[:10]:
        print(f"- {line}")
    if len(sim.event_log) > 10:
        print(f"- ... ({len(sim.event_log) - 10} more lines)")

    print("\nAll checks passed for CS-0317 (拜占庭容错).")


if __name__ == "__main__":
    main()
