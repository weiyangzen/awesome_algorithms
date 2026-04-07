"""Minimal runnable Multi-Paxos MVP.

This demo focuses on:
1) Stable leader proposing multiple slots after one Phase 1.
2) Stale leader rejection.
3) New leader takeover with higher ballot while preserving safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(order=True, frozen=True)
class ProposalID:
    """Total-order proposal identifier: (epoch, proposer_id)."""

    epoch: int
    proposer_id: str


@dataclass
class AcceptorState:
    promised_id: Optional[ProposalID] = None
    accepted: Dict[int, Tuple[ProposalID, str]] = field(default_factory=dict)


class Acceptor:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.state = AcceptorState()

    def on_prepare(self, proposal_id: ProposalID) -> Tuple[bool, Dict[int, Tuple[ProposalID, str]]]:
        promised = self.state.promised_id
        if promised is None or proposal_id >= promised:
            self.state.promised_id = proposal_id
            return True, dict(self.state.accepted)
        return False, dict(self.state.accepted)

    def on_accept(self, proposal_id: ProposalID, slot: int, value: str) -> bool:
        promised = self.state.promised_id
        if promised is None or proposal_id >= promised:
            self.state.promised_id = proposal_id
            self.state.accepted[slot] = (proposal_id, value)
            return True
        return False


class Cluster:
    def __init__(self, acceptor_count: int = 3) -> None:
        if acceptor_count < 3 or acceptor_count % 2 == 0:
            raise ValueError("acceptor_count must be an odd number >= 3")
        self.acceptors: List[Acceptor] = [Acceptor(f"A{i}") for i in range(acceptor_count)]

    @property
    def quorum_size(self) -> int:
        return len(self.acceptors) // 2 + 1

    def compute_chosen_values(self) -> Dict[int, str]:
        """A value is chosen for slot s if accepted by a quorum."""
        slot_value_votes: Dict[int, Dict[str, int]] = {}
        for acceptor in self.acceptors:
            for slot, (_, value) in acceptor.state.accepted.items():
                slot_value_votes.setdefault(slot, {})
                slot_value_votes[slot][value] = slot_value_votes[slot].get(value, 0) + 1

        chosen: Dict[int, str] = {}
        for slot, value_counts in slot_value_votes.items():
            for value, count in value_counts.items():
                if count >= self.quorum_size:
                    chosen[slot] = value
                    break
        return chosen


class MultiPaxosLeader:
    def __init__(self, cluster: Cluster, leader_id: str, epoch: int) -> None:
        self.cluster = cluster
        self.leader_id = leader_id
        self.proposal_id = ProposalID(epoch=epoch, proposer_id=leader_id)
        self.active = False
        self.decided_values: Dict[int, str] = {}

    def phase1_prepare(self) -> None:
        promises: List[Dict[int, Tuple[ProposalID, str]]] = []
        for acceptor in self.cluster.acceptors:
            ok, accepted_map = acceptor.on_prepare(self.proposal_id)
            if ok:
                promises.append(accepted_map)

        if len(promises) < self.cluster.quorum_size:
            raise RuntimeError("Phase1 failed: no quorum promises")

        recovered: Dict[int, Tuple[ProposalID, str]] = {}
        for accepted_map in promises:
            for slot, (pid, value) in accepted_map.items():
                old = recovered.get(slot)
                if old is None or pid > old[0]:
                    recovered[slot] = (pid, value)

        self.decided_values = {slot: value for slot, (_, value) in recovered.items()}
        self.active = True

    def _phase2_accept(self, slot: int, value: str) -> bool:
        accepted_count = 0
        for acceptor in self.cluster.acceptors:
            if acceptor.on_accept(self.proposal_id, slot, value):
                accepted_count += 1
        if accepted_count >= self.cluster.quorum_size:
            self.decided_values[slot] = value
            return True
        return False

    def propose(self, value: str) -> int:
        if not self.active:
            self.phase1_prepare()

        slot = 1
        while slot in self.decided_values:
            slot += 1

        if not self._phase2_accept(slot, value):
            raise RuntimeError(f"Phase2 failed for slot={slot}")
        return slot


def main() -> None:
    cluster = Cluster(acceptor_count=3)

    print("=== Multi-Paxos MVP Demo ===")
    leader1 = MultiPaxosLeader(cluster=cluster, leader_id="L1", epoch=1)
    leader1.phase1_prepare()
    print(f"Leader {leader1.leader_id} active with ballot={leader1.proposal_id}")
    s1 = leader1.propose("SET x=1")
    s2 = leader1.propose("SET y=2")
    print(f"L1 proposed slot {s1} -> SET x=1")
    print(f"L1 proposed slot {s2} -> SET y=2")

    stale_leader = MultiPaxosLeader(cluster=cluster, leader_id="OLD", epoch=0)
    try:
        stale_leader.propose("SHOULD_FAIL")
        raise AssertionError("Stale leader unexpectedly succeeded")
    except RuntimeError as exc:
        print(f"Stale leader rejected as expected: {exc}")

    leader2 = MultiPaxosLeader(cluster=cluster, leader_id="L2", epoch=2)
    leader2.phase1_prepare()
    print(f"Leader {leader2.leader_id} active with ballot={leader2.proposal_id}")
    s3 = leader2.propose("SET z=3")
    print(f"L2 proposed slot {s3} -> SET z=3")

    chosen = cluster.compute_chosen_values()
    print("Chosen log:")
    for slot in sorted(chosen):
        print(f"  slot {slot}: {chosen[slot]}")

    assert chosen.get(1) == "SET x=1", "slot 1 safety violation"
    assert chosen.get(2) == "SET y=2", "slot 2 safety violation"
    assert chosen.get(3) == "SET z=3", "slot 3 progress violation"
    print("Safety check passed.")


if __name__ == "__main__":
    main()
