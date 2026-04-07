"""Thompson construction MVP: regex -> epsilon-NFA -> matching demo."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

EPSILON = "ε"
CONCAT = "."


@dataclass
class Fragment:
    start: int
    accept: int


def is_literal(ch: str) -> bool:
    return ch not in {"(", ")", "|", "*", CONCAT}


def insert_concat_ops(regex: str) -> str:
    if not regex:
        return regex

    out: List[str] = []
    prev: Optional[str] = None

    for ch in regex:
        if prev is not None:
            prev_is_atom = is_literal(prev) or prev in {")", "*"}
            curr_starts_atom = is_literal(ch) or ch == "("
            if prev_is_atom and curr_starts_atom:
                out.append(CONCAT)
        out.append(ch)
        prev = ch

    return "".join(out)


def to_postfix(regex_with_concat: str) -> str:
    precedence = {"|": 1, CONCAT: 2}
    output: List[str] = []
    ops: List[str] = []

    for ch in regex_with_concat:
        if is_literal(ch):
            output.append(ch)
        elif ch == "*":
            # Postfix unary op, can be emitted immediately.
            output.append(ch)
        elif ch in {"|", CONCAT}:
            while ops and ops[-1] != "(" and precedence[ops[-1]] >= precedence[ch]:
                output.append(ops.pop())
            ops.append(ch)
        elif ch == "(":
            ops.append(ch)
        elif ch == ")":
            while ops and ops[-1] != "(":
                output.append(ops.pop())
            if not ops or ops[-1] != "(":
                raise ValueError("Unbalanced parentheses in regex")
            ops.pop()  # Remove '('.
        else:
            raise ValueError(f"Unsupported token: {ch}")

    while ops:
        top = ops.pop()
        if top == "(":
            raise ValueError("Unbalanced parentheses in regex")
        output.append(top)

    return "".join(output)


class NFA:
    def __init__(self) -> None:
        self.transitions: Dict[int, Dict[Optional[str], Set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.start: Optional[int] = None
        self.accept: Optional[int] = None
        self._next_state = 0

    def _new_state(self) -> int:
        s = self._next_state
        self._next_state += 1
        return s

    def _add_transition(self, src: int, symbol: Optional[str], dst: int) -> None:
        self.transitions[src][symbol].add(dst)

    @classmethod
    def from_regex(cls, regex: str) -> tuple[NFA, str, str]:
        nfa = cls()

        if regex == "":
            start = nfa._new_state()
            accept = nfa._new_state()
            nfa._add_transition(start, None, accept)
            nfa.start = start
            nfa.accept = accept
            return nfa, "", ""

        regex_concat = insert_concat_ops(regex)
        postfix = to_postfix(regex_concat)
        nfa._build_from_postfix(postfix)
        return nfa, regex_concat, postfix

    def _build_from_postfix(self, postfix: str) -> None:
        stack: List[Fragment] = []

        for token in postfix:
            if is_literal(token):
                stack.append(self._literal_fragment(token))
            elif token == CONCAT:
                if len(stack) < 2:
                    raise ValueError("Invalid regex: concat missing operand")
                right = stack.pop()
                left = stack.pop()
                stack.append(self._concat(left, right))
            elif token == "|":
                if len(stack) < 2:
                    raise ValueError("Invalid regex: union missing operand")
                right = stack.pop()
                left = stack.pop()
                stack.append(self._union(left, right))
            elif token == "*":
                if not stack:
                    raise ValueError("Invalid regex: star missing operand")
                frag = stack.pop()
                stack.append(self._star(frag))
            else:
                raise ValueError(f"Unsupported postfix token: {token}")

        if len(stack) != 1:
            raise ValueError("Invalid regex: leftover fragments after parsing")

        result = stack.pop()
        self.start = result.start
        self.accept = result.accept

    def _literal_fragment(self, token: str) -> Fragment:
        start = self._new_state()
        accept = self._new_state()
        if token == EPSILON:
            self._add_transition(start, None, accept)
        else:
            self._add_transition(start, token, accept)
        return Fragment(start, accept)

    def _concat(self, left: Fragment, right: Fragment) -> Fragment:
        self._add_transition(left.accept, None, right.start)
        return Fragment(left.start, right.accept)

    def _union(self, left: Fragment, right: Fragment) -> Fragment:
        start = self._new_state()
        accept = self._new_state()
        self._add_transition(start, None, left.start)
        self._add_transition(start, None, right.start)
        self._add_transition(left.accept, None, accept)
        self._add_transition(right.accept, None, accept)
        return Fragment(start, accept)

    def _star(self, frag: Fragment) -> Fragment:
        start = self._new_state()
        accept = self._new_state()
        self._add_transition(start, None, frag.start)
        self._add_transition(start, None, accept)
        self._add_transition(frag.accept, None, frag.start)
        self._add_transition(frag.accept, None, accept)
        return Fragment(start, accept)

    def epsilon_closure(self, states: Set[int]) -> Set[int]:
        closure = set(states)
        stack = list(states)

        while stack:
            state = stack.pop()
            for nxt in self.transitions[state].get(None, set()):
                if nxt not in closure:
                    closure.add(nxt)
                    stack.append(nxt)

        return closure

    def move(self, states: Set[int], symbol: str) -> Set[int]:
        out: Set[int] = set()
        for state in states:
            out.update(self.transitions[state].get(symbol, set()))
        return out

    def matches(self, text: str) -> bool:
        if self.start is None or self.accept is None:
            raise ValueError("NFA not initialized")

        current = self.epsilon_closure({self.start})
        for ch in text:
            current = self.epsilon_closure(self.move(current, ch))
            if not current:
                return False

        return self.accept in current

    def transitions_pretty(self) -> List[str]:
        lines: List[str] = []
        for src in sorted(self.transitions):
            for symbol in sorted(
                self.transitions[src], key=lambda s: "" if s is None else s
            ):
                for dst in sorted(self.transitions[src][symbol]):
                    label = EPSILON if symbol is None else symbol
                    lines.append(f"q{src} --{label}--> q{dst}")
        return lines

    @property
    def state_count(self) -> int:
        return self._next_state


def run_case(regex: str, samples: List[str]) -> None:
    nfa, regex_concat, postfix = NFA.from_regex(regex)

    print("=" * 68)
    print(f"regex: {regex!r}")
    print(f"infix_with_concat: {regex_concat!r}")
    print(f"postfix: {postfix!r}")
    print(f"start=q{nfa.start}, accept=q{nfa.accept}, states={nfa.state_count}")
    print("transitions:")
    for line in nfa.transitions_pretty():
        print(f"  {line}")

    print("matches:")
    for s in samples:
        print(f"  {s!r:10} -> {nfa.matches(s)}")


def main() -> None:
    cases = [
        ("a(b|c)*d", ["ad", "abd", "acbd", "abcbcd", "ab", "aecd"]),
        ("(ab|c)*", ["", "ab", "c", "abcab", "ac", "ccab"]),
        ("", ["", "a"]),
        ("ε", ["", "a"]),
    ]

    for regex, samples in cases:
        run_case(regex, samples)


if __name__ == "__main__":
    main()
