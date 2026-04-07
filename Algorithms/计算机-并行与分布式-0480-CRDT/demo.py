"""CRDT MVP: state-based OR-Set with deterministic simulation.

This script demonstrates:
1) OR-Set conflict resolution under concurrent add/remove.
2) CRDT merge laws: commutative, associative, idempotent.
3) Eventual consistency after anti-entropy synchronization.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

Tag = Tuple[str, int]


@dataclass
class ORSet:
    """Observed-Remove Set (state-based CRDT)."""

    replica_id: str
    adds: Dict[str, Set[Tag]] = field(default_factory=dict)
    tombstones: Set[Tag] = field(default_factory=set)
    _counter: int = 0

    def _next_tag(self) -> Tag:
        self._counter += 1
        return (self.replica_id, self._counter)

    def add(self, element: str) -> Tag:
        tag = self._next_tag()
        self.adds.setdefault(element, set()).add(tag)
        return tag

    def remove(self, element: str) -> Set[Tag]:
        observed = self.live_tags(element)
        if observed:
            self.tombstones.update(observed)
        return observed

    def live_tags(self, element: str) -> Set[Tag]:
        return set(self.adds.get(element, set()) - self.tombstones)

    def contains(self, element: str) -> bool:
        return bool(self.live_tags(element))

    def value(self) -> Set[str]:
        return {element for element in self.adds if self.contains(element)}

    def merge(self, other: "ORSet") -> None:
        for element, tags in other.adds.items():
            self.adds.setdefault(element, set()).update(tags)
        self.tombstones.update(other.tombstones)
        self._counter = max(self._counter, self._max_local_tag_counter())

    def _max_local_tag_counter(self) -> int:
        max_seen = self._counter
        for tags in self.adds.values():
            for rid, seq in tags:
                if rid == self.replica_id and seq > max_seen:
                    max_seen = seq
        for rid, seq in self.tombstones:
            if rid == self.replica_id and seq > max_seen:
                max_seen = seq
        return max_seen

    def clone(self) -> "ORSet":
        return ORSet(
            replica_id=self.replica_id,
            adds={k: set(v) for k, v in self.adds.items()},
            tombstones=set(self.tombstones),
            _counter=self._counter,
        )

    def canonical_state(self) -> Tuple[Tuple[str, Tuple[Tag, ...]], ...]:
        add_part: List[Tuple[str, Tuple[Tag, ...]]] = []
        for element in sorted(self.adds):
            add_part.append((element, tuple(sorted(self.adds[element]))))
        remove_part = ("__tombstones__", tuple(sorted(self.tombstones)))
        return tuple(add_part + [remove_part])


def _sync_all(replicas: Iterable[ORSet]) -> None:
    reps = list(replicas)
    for i in range(len(reps)):
        for j in range(len(reps)):
            if i != j:
                reps[i].merge(reps[j])


def prove_merge_laws() -> None:
    x = ORSet("X")
    y = ORSet("Y")
    z = ORSet("Z")

    x.add("alpha")
    x.add("beta")

    y.add("beta")
    y.add("gamma")

    z.add("gamma")
    z.add("delta")

    # Make states non-trivial with a remove that only sees local tags.
    y.remove("beta")

    # Commutative: X merge Y == Y merge X
    xy = x.clone()
    xy.merge(y)
    yx = y.clone()
    yx.merge(x)
    assert xy.canonical_state() == yx.canonical_state(), "Merge is not commutative"

    # Associative: (X merge Y) merge Z == X merge (Y merge Z)
    left = x.clone()
    left.merge(y)
    left.merge(z)

    right = x.clone()
    yz = y.clone()
    yz.merge(z)
    right.merge(yz)

    assert left.canonical_state() == right.canonical_state(), "Merge is not associative"

    # Idempotent: X merge X == X
    idem = x.clone()
    idem.merge(x)
    assert idem.canonical_state() == x.canonical_state(), "Merge is not idempotent"


def run_deterministic_scenario() -> pd.DataFrame:
    a = ORSet("A")
    b = ORSet("B")
    c = ORSet("C")

    events: List[dict[str, object]] = []

    def record(step: int, replica: ORSet, op: str, element: str, note: str) -> None:
        events.append(
            {
                "step": step,
                "replica": replica.replica_id,
                "op": op,
                "element": element,
                "note": note,
                "value": "{" + ", ".join(sorted(replica.value())) + "}",
                "live_tag_count": int(sum(len(replica.live_tags(e)) for e in replica.adds)),
                "tombstone_count": len(replica.tombstones),
            }
        )

    step = 0

    step += 1
    tag_a1 = a.add("apple")
    record(step, a, "add", "apple", f"tag={tag_a1}")

    step += 1
    tag_b1 = b.add("banana")
    record(step, b, "add", "banana", f"tag={tag_b1}")

    step += 1
    a.merge(b)
    record(step, a, "merge", "*", "A <- B")

    step += 1
    removed = a.remove("banana")
    record(step, a, "remove", "banana", f"observed={sorted(removed)}")

    step += 1
    tag_b2 = b.add("banana")
    record(step, b, "add", "banana", f"concurrent tag={tag_b2}")

    step += 1
    tag_c1 = c.add("cherry")
    record(step, c, "add", "cherry", f"tag={tag_c1}")

    step += 1
    c.merge(a)
    record(step, c, "merge", "*", "C <- A")

    step += 1
    b.merge(c)
    record(step, b, "merge", "*", "B <- C")

    step += 1
    _sync_all([a, b, c])
    record(step, a, "sync", "*", "all-to-all anti-entropy finished")
    record(step, b, "sync", "*", "all-to-all anti-entropy finished")
    record(step, c, "sync", "*", "all-to-all anti-entropy finished")

    # After full sync, all replicas must converge.
    av = a.value()
    bv = b.value()
    cv = c.value()
    assert av == bv == cv, "Replicas did not converge"

    # Key semantic check:
    # - banana tag B:1 was observed and removed by A;
    # - banana tag B:2 was concurrent and survives.
    assert "banana" in av, "Concurrent add should survive observed-remove in OR-Set"
    assert "apple" in av and "cherry" in av

    return pd.DataFrame(events)


def run_random_gossip(seed: int = 20260407, steps: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    replicas = {name: ORSet(name) for name in ("A", "B", "C")}
    names = np.array(list(replicas.keys()))
    universe = np.array(["k0", "k1", "k2", "k3", "k4"]) 

    rows: List[dict[str, object]] = []

    for step in range(1, steps + 1):
        op = str(rng.choice(np.array(["add", "remove", "merge"]), p=[0.50, 0.25, 0.25]))

        if op in {"add", "remove"}:
            who = str(rng.choice(names))
            replica = replicas[who]
            element = str(rng.choice(universe))

            if op == "add":
                tag = replica.add(element)
                note = f"tag={tag}"
            else:
                removed = replica.remove(element)
                note = f"removed={len(removed)}"

            rows.append(
                {
                    "step": step,
                    "op": op,
                    "replica": who,
                    "peer": "-",
                    "element": element,
                    "note": note,
                }
            )
        else:
            dst = str(rng.choice(names))
            src_candidates = [n for n in names.tolist() if n != dst]
            src = str(rng.choice(np.array(src_candidates)))
            replicas[dst].merge(replicas[src])
            rows.append(
                {
                    "step": step,
                    "op": "merge",
                    "replica": dst,
                    "peer": src,
                    "element": "*",
                    "note": f"{dst}<-{src}",
                }
            )

    _sync_all(list(replicas.values()))

    final_values = {name: replicas[name].value() for name in replicas}
    base = next(iter(final_values.values()))
    for name, value in final_values.items():
        assert value == base, f"Replica {name} diverged after full sync"

    rows.append(
        {
            "step": steps + 1,
            "op": "final_sync",
            "replica": "ALL",
            "peer": "ALL",
            "element": "*",
            "note": "converged",
        }
    )

    return pd.DataFrame(rows)


def main() -> None:
    prove_merge_laws()

    scenario_df = run_deterministic_scenario()
    random_df = run_random_gossip()

    print("=== CRDT OR-Set MVP ===")
    print("Deterministic scenario trace:")
    print(scenario_df.to_string(index=False))
    print()

    print("Random gossip trace (last 15 rows):")
    print(random_df.tail(15).to_string(index=False))
    print()

    op_summary = random_df.groupby("op").size().rename("count").reset_index()
    print("Random gossip operation summary:")
    print(op_summary.to_string(index=False))
    print()

    print("All assertions passed.")


if __name__ == "__main__":
    main()
