"""DFA minimization MVP using reachable-state pruning + Hopcroft partition refinement.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple


@dataclass(frozen=True)
class DFA:
    states: FrozenSet[str]
    alphabet: FrozenSet[str]
    transition: Dict[Tuple[str, str], str]
    start: str
    finals: FrozenSet[str]


def validate_dfa(dfa: DFA) -> None:
    if dfa.start not in dfa.states:
        raise ValueError("start state is not in states")
    if not dfa.finals.issubset(dfa.states):
        raise ValueError("final states must be subset of states")
    if not dfa.alphabet:
        raise ValueError("alphabet must be non-empty")

    for q in dfa.states:
        for a in dfa.alphabet:
            key = (q, a)
            if key not in dfa.transition:
                raise ValueError(f"missing transition for ({q}, {a})")
            nxt = dfa.transition[key]
            if nxt not in dfa.states:
                raise ValueError(f"transition ({q}, {a}) -> {nxt} leaves states")


def reachable_states(dfa: DFA) -> FrozenSet[str]:
    seen: Set[str] = {dfa.start}
    queue: deque[str] = deque([dfa.start])
    while queue:
        q = queue.popleft()
        for a in dfa.alphabet:
            nxt = dfa.transition[(q, a)]
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return frozenset(seen)


def prune_unreachable(dfa: DFA) -> DFA:
    reach = reachable_states(dfa)
    new_transition: Dict[Tuple[str, str], str] = {}
    for q in reach:
        for a in dfa.alphabet:
            new_transition[(q, a)] = dfa.transition[(q, a)]
    return DFA(
        states=reach,
        alphabet=dfa.alphabet,
        transition=new_transition,
        start=dfa.start,
        finals=frozenset(q for q in dfa.finals if q in reach),
    )


def _predecessors(dfa: DFA, block: FrozenSet[str], symbol: str) -> FrozenSet[str]:
    preds = {q for q in dfa.states if dfa.transition[(q, symbol)] in block}
    return frozenset(preds)


def hopcroft_minimize(dfa: DFA) -> Tuple[DFA, Dict[str, str], List[FrozenSet[str]]]:
    """Return minimized DFA, old->new state map, and final partition blocks."""
    validate_dfa(dfa)
    dfa = prune_unreachable(dfa)

    finals = frozenset(dfa.finals)
    nonfinals = frozenset(dfa.states - dfa.finals)

    partition: List[FrozenSet[str]] = []
    if finals:
        partition.append(finals)
    if nonfinals:
        partition.append(nonfinals)

    worklist: List[FrozenSet[str]] = []
    if finals and nonfinals:
        worklist.append(finals if len(finals) <= len(nonfinals) else nonfinals)
    elif finals:
        worklist.append(finals)
    elif nonfinals:
        worklist.append(nonfinals)

    while worklist:
        a_block = worklist.pop()
        for sym in dfa.alphabet:
            x = _predecessors(dfa, a_block, sym)
            new_partition: List[FrozenSet[str]] = []

            for y in partition:
                inter = frozenset(y & x)
                diff = frozenset(y - x)
                if inter and diff:
                    new_partition.extend([inter, diff])

                    if y in worklist:
                        worklist.remove(y)
                        worklist.extend([inter, diff])
                    else:
                        worklist.append(inter if len(inter) <= len(diff) else diff)
                else:
                    new_partition.append(y)

            partition = new_partition

    ordered_blocks = sorted(partition, key=lambda b: (len(b), sorted(b)[0]))

    block_name: Dict[FrozenSet[str], str] = {}
    for i, block in enumerate(ordered_blocks):
        block_name[block] = f"Q{i}"

    state_to_new: Dict[str, str] = {}
    for block, name in block_name.items():
        for q in block:
            state_to_new[q] = name

    min_states = frozenset(block_name.values())
    min_start = state_to_new[dfa.start]
    min_finals = frozenset(state_to_new[q] for q in dfa.finals)

    min_trans: Dict[Tuple[str, str], str] = {}
    for block, name in block_name.items():
        rep = next(iter(block))
        for sym in dfa.alphabet:
            tgt_old = dfa.transition[(rep, sym)]
            min_trans[(name, sym)] = state_to_new[tgt_old]

    minimized = DFA(
        states=min_states,
        alphabet=dfa.alphabet,
        transition=min_trans,
        start=min_start,
        finals=min_finals,
    )
    validate_dfa(minimized)
    return minimized, state_to_new, ordered_blocks


def accepts(dfa: DFA, word: str) -> bool:
    q = dfa.start
    for ch in word:
        if ch not in dfa.alphabet:
            raise ValueError(f"symbol {ch!r} not in alphabet")
        q = dfa.transition[(q, ch)]
    return q in dfa.finals


def words_over(alphabet: Iterable[str], max_len: int) -> List[str]:
    symbols = sorted(set(alphabet))
    out = [""]
    for k in range(1, max_len + 1):
        for tup in product(symbols, repeat=k):
            out.append("".join(tup))
    return out


def equivalent_on_bounded_words(dfa1: DFA, dfa2: DFA, max_len: int = 8) -> bool:
    if dfa1.alphabet != dfa2.alphabet:
        raise ValueError("cannot compare DFAs with different alphabets")
    for w in words_over(dfa1.alphabet, max_len=max_len):
        if accepts(dfa1, w) != accepts(dfa2, w):
            print(f"mismatch on word {w!r}")
            return False
    return True


def build_demo_dfa() -> DFA:
    """Build a non-minimal DFA with 8 states; 2 states are unreachable.

    Language: all binary strings ending with "01".
    Minimal DFA has 3 states over alphabet {0,1}.
    """
    states = frozenset({"A", "B", "C", "D", "E", "F", "U", "UA"})
    alphabet = frozenset({"0", "1"})
    finals = frozenset({"C", "F", "UA"})
    start = "A"

    trans: Dict[Tuple[str, str], str] = {
        # Reachable core (A-F): intentionally duplicated equivalent states
        ("A", "0"): "B",
        ("A", "1"): "D",
        ("B", "0"): "E",
        ("B", "1"): "C",
        ("C", "0"): "E",
        ("C", "1"): "D",
        ("D", "0"): "E",
        ("D", "1"): "A",
        ("E", "0"): "B",
        ("E", "1"): "F",
        ("F", "0"): "B",
        ("F", "1"): "A",
        # Unreachable pair (kept only to verify pruning)
        ("U", "0"): "U",
        ("U", "1"): "UA",
        ("UA", "0"): "U",
        ("UA", "1"): "UA",
    }

    dfa = DFA(states=states, alphabet=alphabet, transition=trans, start=start, finals=finals)
    validate_dfa(dfa)
    return dfa


def summarize_dfa(dfa: DFA, name: str) -> None:
    print(f"=== {name} ===")
    print(f"states ({len(dfa.states)}): {sorted(dfa.states)}")
    print(f"alphabet: {sorted(dfa.alphabet)}")
    print(f"start: {dfa.start}")
    print(f"finals: {sorted(dfa.finals)}")

    for q in sorted(dfa.states):
        row = []
        for a in sorted(dfa.alphabet):
            row.append(f"δ({q},{a})={dfa.transition[(q, a)]}")
        print("  " + ", ".join(row))
    print()


def main() -> None:
    original = build_demo_dfa()
    summarize_dfa(original, "Original DFA")

    pruned = prune_unreachable(original)
    print(f"reachable states from start: {sorted(pruned.states)}")
    print(f"pruned {len(original.states) - len(pruned.states)} unreachable states\n")

    minimized, mapping, partition = hopcroft_minimize(original)

    print("final partition blocks (equivalence classes):")
    for i, block in enumerate(partition):
        print(f"  block#{i}: {sorted(block)}")
    print()

    print("old -> minimized state mapping:")
    for q in sorted(mapping):
        print(f"  {q:>2} -> {mapping[q]}")
    print()

    summarize_dfa(minimized, "Minimized DFA")

    checks = ["", "0", "1", "01", "101", "1001", "110", "1101", "11101", "11111"]
    print("sample acceptance checks (language: ending with '01'):")
    for w in checks:
        print(f"  w={w!r:<8} accepted={accepts(minimized, w)}")
    print()

    eq_ok = equivalent_on_bounded_words(original, minimized, max_len=8)
    print(f"bounded equivalence test (all words len<=8): {eq_ok}")
    if not eq_ok:
        raise RuntimeError("minimized DFA is not language-equivalent on bounded test")


if __name__ == "__main__":
    main()
