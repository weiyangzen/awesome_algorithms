"""Minimal runnable MVP for Raft consensus algorithm.

This demo models a tiny Raft cluster with deterministic time simulation:
- leader election (RequestVote)
- log replication (AppendEntries)
- commit + state machine apply
- leader crash and recovery

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LogEntry:
    term: int
    command: str


@dataclass
class Node:
    node_id: int
    current_term: int = 0
    voted_for: Optional[int] = None
    role: str = "follower"  # follower | candidate | leader
    log: List[LogEntry] = field(default_factory=list)
    commit_index: int = -1
    last_applied: int = -1
    leader_id: Optional[int] = None
    election_timeout: int = 0
    election_elapsed: int = 0
    active: bool = True
    state_machine: Dict[str, int] = field(default_factory=dict)
    applied_commands: List[str] = field(default_factory=list)


class RaftCluster:
    """A compact educational Raft simulator (single process, discrete-time ticks)."""

    def __init__(
        self,
        node_ids: List[int],
        seed: int = 2026,
        heartbeat_interval: int = 2,
    ) -> None:
        if len(node_ids) < 3 or len(node_ids) % 2 == 0:
            raise ValueError("Raft demo expects an odd cluster size >= 3.")
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive.")

        self.rng = random.Random(seed)
        self.heartbeat_interval = heartbeat_interval
        self.time_tick = 0
        self.node_ids = sorted(node_ids)
        self.nodes: Dict[int, Node] = {nid: Node(node_id=nid) for nid in self.node_ids}
        self.leader_id: Optional[int] = None

        # Valid for current leader term.
        self.next_index: Dict[int, int] = {}
        self.match_index: Dict[int, int] = {}

        self.events: List[str] = []

        for node in self.nodes.values():
            self._reset_election_timer(node)

    def majority(self) -> int:
        return len(self.node_ids) // 2 + 1

    def active_node_ids(self) -> List[int]:
        return [nid for nid in self.node_ids if self.nodes[nid].active]

    def _last_log_index_term(self, node: Node) -> Tuple[int, int]:
        if not node.log:
            return -1, 0
        idx = len(node.log) - 1
        return idx, node.log[idx].term

    def _reset_election_timer(self, node: Node) -> None:
        node.election_elapsed = 0
        node.election_timeout = self.rng.randint(5, 9)

    def _become_follower(self, node: Node, new_term: int, leader_id: Optional[int]) -> None:
        if new_term < node.current_term:
            return
        node.current_term = new_term
        node.role = "follower"
        node.voted_for = None
        node.leader_id = leader_id
        self._reset_election_timer(node)

    def _is_candidate_log_up_to_date(
        self,
        candidate_last_index: int,
        candidate_last_term: int,
        follower_last_index: int,
        follower_last_term: int,
    ) -> bool:
        if candidate_last_term != follower_last_term:
            return candidate_last_term > follower_last_term
        return candidate_last_index >= follower_last_index

    def request_vote(
        self,
        target_id: int,
        term: int,
        candidate_id: int,
        candidate_last_index: int,
        candidate_last_term: int,
    ) -> bool:
        target = self.nodes[target_id]
        if not target.active:
            return False

        if term < target.current_term:
            return False

        if term > target.current_term:
            self._become_follower(target, term, leader_id=None)

        follower_last_index, follower_last_term = self._last_log_index_term(target)
        up_to_date = self._is_candidate_log_up_to_date(
            candidate_last_index,
            candidate_last_term,
            follower_last_index,
            follower_last_term,
        )

        if (target.voted_for is None or target.voted_for == candidate_id) and up_to_date:
            target.voted_for = candidate_id
            self._reset_election_timer(target)
            return True

        return False

    def start_election(self, candidate_id: int) -> bool:
        candidate = self.nodes[candidate_id]
        if not candidate.active:
            return False

        candidate.role = "candidate"
        candidate.current_term += 1
        candidate.voted_for = candidate_id
        candidate.leader_id = None
        self._reset_election_timer(candidate)

        votes = 1
        last_idx, last_term = self._last_log_index_term(candidate)

        for nid in self.node_ids:
            if nid == candidate_id:
                continue
            granted = self.request_vote(
                target_id=nid,
                term=candidate.current_term,
                candidate_id=candidate_id,
                candidate_last_index=last_idx,
                candidate_last_term=last_term,
            )
            if granted:
                votes += 1

        self.events.append(
            f"t={self.time_tick}: node {candidate_id} starts election term={candidate.current_term}, votes={votes}"
        )

        if votes >= self.majority():
            candidate.role = "leader"
            candidate.leader_id = candidate_id
            self.leader_id = candidate_id

            self.next_index = {}
            self.match_index = {}
            leader_last_index = len(candidate.log) - 1
            for nid in self.node_ids:
                self.next_index[nid] = len(candidate.log)
                self.match_index[nid] = -1
            self.match_index[candidate_id] = leader_last_index

            self.events.append(
                f"t={self.time_tick}: node {candidate_id} becomes leader for term {candidate.current_term}"
            )
            self.send_heartbeats()
            return True

        # For this MVP, fallback to follower and wait for the next timeout.
        candidate.role = "follower"
        candidate.voted_for = None
        self._reset_election_timer(candidate)
        return False

    def append_entries(
        self,
        target_id: int,
        term: int,
        leader_id: int,
        prev_log_index: int,
        prev_log_term: int,
        entries: List[LogEntry],
        leader_commit: int,
    ) -> Tuple[bool, int]:
        target = self.nodes[target_id]
        if not target.active:
            return False, target.current_term

        if term < target.current_term:
            return False, target.current_term

        if term > target.current_term or target.role != "follower":
            self._become_follower(target, term, leader_id=leader_id)
        else:
            target.leader_id = leader_id
            self._reset_election_timer(target)

        if prev_log_index >= len(target.log):
            return False, target.current_term

        if prev_log_index >= 0 and target.log[prev_log_index].term != prev_log_term:
            return False, target.current_term

        insert_at = prev_log_index + 1
        for i, entry in enumerate(entries):
            local_index = insert_at + i
            if local_index < len(target.log):
                if target.log[local_index].term != entry.term:
                    target.log = target.log[:local_index]
                    target.log.append(entry)
            else:
                target.log.append(entry)

        if leader_commit > target.commit_index:
            target.commit_index = min(leader_commit, len(target.log) - 1)
            self._apply_entries(target)

        return True, target.current_term

    def _apply_command(self, sm: Dict[str, int], command: str) -> None:
        tokens = command.split()
        if len(tokens) != 3:
            raise ValueError(f"Unsupported command format: {command}")

        op, key, value_raw = tokens
        value = int(value_raw)
        if op == "set":
            sm[key] = value
        elif op == "add":
            sm[key] = sm.get(key, 0) + value
        else:
            raise ValueError(f"Unsupported command op: {op}")

    def _apply_entries(self, node: Node) -> None:
        while node.last_applied < node.commit_index:
            node.last_applied += 1
            entry = node.log[node.last_applied]
            self._apply_command(node.state_machine, entry.command)
            node.applied_commands.append(entry.command)

    def _replicate_to_follower(self, follower_id: int) -> bool:
        if self.leader_id is None:
            return False
        leader = self.nodes[self.leader_id]
        if not leader.active or leader.role != "leader":
            return False

        if follower_id == self.leader_id:
            return True
        follower = self.nodes[follower_id]
        if not follower.active:
            return False

        if follower_id not in self.next_index:
            self.next_index[follower_id] = len(leader.log)
            self.match_index[follower_id] = -1

        while True:
            next_idx = self.next_index[follower_id]
            prev_idx = next_idx - 1
            prev_term = leader.log[prev_idx].term if prev_idx >= 0 else 0
            entries = leader.log[next_idx:]

            ok, follower_term = self.append_entries(
                target_id=follower_id,
                term=leader.current_term,
                leader_id=self.leader_id,
                prev_log_index=prev_idx,
                prev_log_term=prev_term,
                entries=entries,
                leader_commit=leader.commit_index,
            )

            if ok:
                self.match_index[follower_id] = prev_idx + len(entries)
                self.next_index[follower_id] = self.match_index[follower_id] + 1
                return True

            if follower_term > leader.current_term:
                self.events.append(
                    f"t={self.time_tick}: leader {self.leader_id} steps down due to higher term {follower_term}"
                )
                self._become_follower(leader, follower_term, leader_id=None)
                self.leader_id = None
                return False

            if self.next_index[follower_id] == 0:
                return False
            self.next_index[follower_id] -= 1

    def _advance_commit_index(self) -> None:
        if self.leader_id is None:
            return
        leader = self.nodes[self.leader_id]
        if leader.role != "leader" or not leader.active:
            return

        # Raft commit rule: only commit entries from current term by majority.
        for idx in range(len(leader.log) - 1, leader.commit_index, -1):
            if leader.log[idx].term != leader.current_term:
                continue
            replicated = 0
            for nid in self.node_ids:
                if self.match_index.get(nid, -1) >= idx:
                    replicated += 1
            if replicated >= self.majority():
                leader.commit_index = idx
                self._apply_entries(leader)
                self.events.append(
                    f"t={self.time_tick}: leader {self.leader_id} commits log index {idx}"
                )
                return

    def send_heartbeats(self) -> None:
        if self.leader_id is None:
            return
        leader = self.nodes[self.leader_id]
        if not leader.active or leader.role != "leader":
            return

        for nid in self.node_ids:
            if nid == self.leader_id:
                continue
            self._replicate_to_follower(nid)

    def client_submit(self, command: str) -> int:
        if self.leader_id is None:
            raise RuntimeError("No active leader; cannot submit command.")

        leader = self.nodes[self.leader_id]
        if not leader.active or leader.role != "leader":
            raise RuntimeError("Current leader is unavailable.")

        leader.log.append(LogEntry(term=leader.current_term, command=command))
        self.match_index[self.leader_id] = len(leader.log) - 1

        replicated_targets = []
        for nid in self.node_ids:
            if nid == self.leader_id:
                continue
            ok = self._replicate_to_follower(nid)
            if ok:
                replicated_targets.append(nid)

        self._advance_commit_index()
        self.send_heartbeats()

        committed_idx = leader.commit_index
        self.events.append(
            f"t={self.time_tick}: client command='{command}' append_by={self.leader_id}, committed_index={committed_idx}, replicated_to={replicated_targets}"
        )
        return committed_idx

    def tick(self) -> None:
        self.time_tick += 1

        if self.leader_id is not None:
            leader = self.nodes[self.leader_id]
            if (not leader.active) or leader.role != "leader":
                self.leader_id = None

        if self.leader_id is not None and self.time_tick % self.heartbeat_interval == 0:
            self.send_heartbeats()

        for nid in self.node_ids:
            node = self.nodes[nid]
            if not node.active:
                continue
            if node.role == "leader":
                continue

            node.election_elapsed += 1
            if node.election_elapsed >= node.election_timeout:
                elected = self.start_election(nid)
                if elected:
                    break

    def run_until_leader(self, max_ticks: int = 200) -> int:
        for _ in range(max_ticks):
            if self.leader_id is not None and self.nodes[self.leader_id].active:
                return self.leader_id
            self.tick()
        raise RuntimeError("No leader elected within max_ticks.")

    def crash_node(self, node_id: int) -> None:
        node = self.nodes[node_id]
        node.active = False
        if node.role == "leader":
            self.leader_id = None
        node.role = "follower"
        node.leader_id = None
        self.events.append(f"t={self.time_tick}: node {node_id} crashed")

    def recover_node(self, node_id: int) -> None:
        node = self.nodes[node_id]
        node.active = True
        node.role = "follower"
        node.leader_id = None
        self._reset_election_timer(node)
        self.events.append(f"t={self.time_tick}: node {node_id} recovered")

    def assert_safety(self) -> None:
        committed_logs: List[List[LogEntry]] = []
        committed_sms: List[Dict[str, int]] = []

        for nid in self.active_node_ids():
            node = self.nodes[nid]
            prefix = node.log[: node.commit_index + 1]
            committed_logs.append(prefix)
            committed_sms.append(dict(node.state_machine))

        if not committed_logs:
            raise AssertionError("No active nodes to validate.")

        base_log = committed_logs[0]
        base_sm = committed_sms[0]
        for i in range(1, len(committed_logs)):
            if committed_logs[i] != base_log:
                raise AssertionError("Committed log prefixes diverged across active nodes.")
            if committed_sms[i] != base_sm:
                raise AssertionError("State machine snapshots diverged across active nodes.")

    def print_cluster_summary(self) -> None:
        print("\n=== Cluster Summary ===")
        print(f"time_tick={self.time_tick}")
        print(f"leader_id={self.leader_id}")
        print(f"majority={self.majority()}")

        for nid in self.node_ids:
            n = self.nodes[nid]
            print(
                f"node={nid} active={n.active} role={n.role} term={n.current_term} "
                f"log_len={len(n.log)} commit_index={n.commit_index} "
                f"last_applied={n.last_applied} state_machine={n.state_machine}"
            )

    def print_recent_events(self, keep_last: int = 20) -> None:
        print("\n=== Recent Events ===")
        for line in self.events[-keep_last:]:
            print(line)


def main() -> None:
    cluster = RaftCluster(node_ids=[1, 2, 3, 4, 5], seed=7, heartbeat_interval=2)

    # 1) Initial leader election.
    leader1 = cluster.run_until_leader(max_ticks=200)

    # 2) Submit commands on first leader.
    idx1 = cluster.client_submit("set x 1")
    idx2 = cluster.client_submit("set y 10")
    idx3 = cluster.client_submit("add x 4")

    assert idx1 >= 0 and idx2 >= idx1 and idx3 >= idx2

    # 3) Crash the first leader and elect a new one.
    cluster.crash_node(leader1)
    leader2 = cluster.run_until_leader(max_ticks=200)
    assert leader2 != leader1

    # 4) Continue serving writes under new leader.
    cluster.client_submit("add y -3")
    cluster.client_submit("set z 8")

    # 5) Recover old leader and allow catch-up.
    cluster.recover_node(leader1)
    for _ in range(10):
        cluster.tick()
    cluster.send_heartbeats()

    # 6) Safety checks: committed prefix and state machine must converge on active nodes.
    cluster.assert_safety()

    leader_now = cluster.leader_id
    if leader_now is None:
        raise RuntimeError("Expected an active leader at the end of demo.")

    final_sm = cluster.nodes[leader_now].state_machine
    expected = {"x": 5, "y": 7, "z": 8}
    if final_sm != expected:
        raise AssertionError(f"Unexpected final state machine: got {final_sm}, want {expected}")

    cluster.print_cluster_summary()
    cluster.print_recent_events(keep_last=25)

    print("\nAll assertions passed. Raft MVP demo is working.")


if __name__ == "__main__":
    main()
