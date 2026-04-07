"""子集构造法（Subset Construction）最小可运行示例。

目标：
1. 构造一个含 epsilon 转移的 NFA；
2. 用子集构造法将其转换成 DFA；
3. 对比 NFA 与 DFA 在多组输入上的识别结果，验证等价性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

EPSILON = "ε"


@dataclass(frozen=True)
class NFA:
    states: Set[str]
    alphabet: Set[str]
    transitions: Dict[Tuple[str, str], Set[str]]
    start_state: str
    accept_states: Set[str]


@dataclass(frozen=True)
class DFA:
    states: Set[str]
    alphabet: Set[str]
    transitions: Dict[Tuple[str, str], str]
    start_state: str
    accept_states: Set[str]
    subset_mapping: Dict[str, frozenset[str]]


def add_transition(
    transitions: Dict[Tuple[str, str], Set[str]], src: str, symbol: str, dst: str
) -> None:
    transitions.setdefault((src, symbol), set()).add(dst)


def format_subset(subset: frozenset[str]) -> str:
    if not subset:
        return "{}"
    return "{" + ", ".join(sorted(subset)) + "}"


def epsilon_closure(nfa: NFA, states: Iterable[str]) -> frozenset[str]:
    """计算状态集合的 epsilon 闭包。"""
    closure = set(states)
    stack = list(states)

    while stack:
        state = stack.pop()
        for nxt in nfa.transitions.get((state, EPSILON), set()):
            if nxt not in closure:
                closure.add(nxt)
                stack.append(nxt)

    return frozenset(closure)


def move(nfa: NFA, states: Iterable[str], symbol: str) -> set[str]:
    """从状态集合出发，沿 symbol 进行一次转移。"""
    reached: set[str] = set()
    for state in states:
        reached.update(nfa.transitions.get((state, symbol), set()))
    return reached


def subset_construction(nfa: NFA) -> tuple[DFA, List[str]]:
    """将 epsilon-NFA 转换为 DFA，并返回构造追踪日志。"""
    start_subset = epsilon_closure(nfa, [nfa.start_state])

    subset_to_name: Dict[frozenset[str], str] = {start_subset: "D0"}
    name_to_subset: Dict[str, frozenset[str]] = {"D0": start_subset}
    worklist: List[frozenset[str]] = [start_subset]

    dfa_transitions: Dict[Tuple[str, str], str] = {}
    dfa_accept_states: Set[str] = set()
    trace: List[str] = []

    if start_subset & nfa.accept_states:
        dfa_accept_states.add("D0")

    while worklist:
        current_subset = worklist.pop(0)
        current_name = subset_to_name[current_subset]
        trace.append(
            f"处理 {current_name} = {format_subset(current_subset)}"
        )

        for symbol in sorted(nfa.alphabet):
            moved = move(nfa, current_subset, symbol)
            next_subset = epsilon_closure(nfa, moved)

            # 空集状态在子集构造法中是合法 DFA 状态（可作陷阱状态）。
            if next_subset not in subset_to_name:
                next_name = f"D{len(subset_to_name)}"
                subset_to_name[next_subset] = next_name
                name_to_subset[next_name] = next_subset
                worklist.append(next_subset)
                trace.append(
                    f"  发现新状态 {next_name} = {format_subset(next_subset)}"
                )

                if next_subset & nfa.accept_states:
                    dfa_accept_states.add(next_name)
            else:
                next_name = subset_to_name[next_subset]

            dfa_transitions[(current_name, symbol)] = next_name
            trace.append(f"  转移: {current_name} --{symbol}--> {next_name}")

    dfa = DFA(
        states=set(name_to_subset.keys()),
        alphabet=set(nfa.alphabet),
        transitions=dfa_transitions,
        start_state="D0",
        accept_states=dfa_accept_states,
        subset_mapping=name_to_subset,
    )
    return dfa, trace


def simulate_nfa(nfa: NFA, text: str) -> bool:
    current = epsilon_closure(nfa, [nfa.start_state])

    for ch in text:
        if ch not in nfa.alphabet:
            raise ValueError(f"输入符号 '{ch}' 不在 NFA 字母表中")
        current = epsilon_closure(nfa, move(nfa, current, ch))

    return bool(current & nfa.accept_states)


def simulate_dfa(dfa: DFA, text: str) -> bool:
    state = dfa.start_state

    for ch in text:
        if ch not in dfa.alphabet:
            raise ValueError(f"输入符号 '{ch}' 不在 DFA 字母表中")
        state = dfa.transitions[(state, ch)]

    return state in dfa.accept_states


def build_sample_nfa() -> NFA:
    """构造识别正则 (a|b)*ab 的 epsilon-NFA（Thompson 风格）。"""
    states = {f"q{i}" for i in range(10)}
    alphabet = {"a", "b"}
    transitions: Dict[Tuple[str, str], Set[str]] = {}

    # (a|b)* 部分
    add_transition(transitions, "q0", EPSILON, "q1")
    add_transition(transitions, "q0", EPSILON, "q7")
    add_transition(transitions, "q1", EPSILON, "q2")
    add_transition(transitions, "q1", EPSILON, "q4")
    add_transition(transitions, "q2", "a", "q3")
    add_transition(transitions, "q4", "b", "q5")
    add_transition(transitions, "q3", EPSILON, "q6")
    add_transition(transitions, "q5", EPSILON, "q6")
    add_transition(transitions, "q6", EPSILON, "q1")
    add_transition(transitions, "q6", EPSILON, "q7")

    # 末尾拼接 "ab"
    add_transition(transitions, "q7", "a", "q8")
    add_transition(transitions, "q8", "b", "q9")

    return NFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        start_state="q0",
        accept_states={"q9"},
    )


def generate_random_strings(
    rng: np.random.Generator,
    count: int,
    max_len: int,
    alphabet: Tuple[str, ...],
) -> List[str]:
    results: List[str] = []
    alpha_arr = np.array(alphabet)

    for _ in range(count):
        length = int(rng.integers(0, max_len + 1))
        if length == 0:
            results.append("")
            continue
        chars = rng.choice(alpha_arr, size=length)
        results.append("".join(chars.tolist()))

    return results


def verify_equivalence(nfa: NFA, dfa: DFA, samples: List[str]) -> None:
    mismatches: List[str] = []

    for s in samples:
        nfa_ok = simulate_nfa(nfa, s)
        dfa_ok = simulate_dfa(dfa, s)
        if nfa_ok != dfa_ok:
            mismatches.append(f"样例 '{s}': NFA={nfa_ok}, DFA={dfa_ok}")

    if mismatches:
        raise AssertionError("发现不一致:\n" + "\n".join(mismatches))


def sorted_dfa_states(states: Iterable[str]) -> List[str]:
    return sorted(states, key=lambda name: int(name[1:]))


def main() -> None:
    nfa = build_sample_nfa()
    dfa, trace = subset_construction(nfa)

    print("=== 子集构造法 Demo ===")
    print("目标语言: (a|b)*ab")
    print(
        f"NFA 状态数={len(nfa.states)}, DFA 状态数={len(dfa.states)}, "
        f"DFA 接受态={sorted_dfa_states(dfa.accept_states)}"
    )

    print("\n[构造追踪]")
    for line in trace:
        print(line)

    print("\n[DFA 状态映射]")
    for d_state in sorted_dfa_states(dfa.states):
        subset = dfa.subset_mapping[d_state]
        acc = " (accept)" if d_state in dfa.accept_states else ""
        print(f"{d_state} = {format_subset(subset)}{acc}")

    print("\n[DFA 转移表]")
    for d_state in sorted_dfa_states(dfa.states):
        row = []
        for symbol in sorted(dfa.alphabet):
            row.append(f"{symbol}->{dfa.transitions[(d_state, symbol)]}")
        print(f"{d_state}: " + ", ".join(row))

    fixed_cases = [
        "",
        "a",
        "b",
        "ab",
        "abb",
        "aab",
        "baab",
        "baba",
        "abab",
        "bbab",
    ]

    rng = np.random.default_rng(2026)
    random_cases = generate_random_strings(rng, count=30, max_len=8, alphabet=("a", "b"))
    samples = sorted(set(fixed_cases + random_cases), key=lambda s: (len(s), s))

    verify_equivalence(nfa, dfa, samples)

    print("\n[识别样例]")
    for s in samples[:20]:
        accepted = simulate_dfa(dfa, s)
        tag = "ACCEPT" if accepted else "REJECT"
        shown = s if s else "ε"
        print(f"{shown:>8} -> {tag}")

    print(
        f"\n等价性验证通过: 共 {len(samples)} 个样例，NFA 与 DFA 结果完全一致。"
    )


if __name__ == "__main__":
    main()
