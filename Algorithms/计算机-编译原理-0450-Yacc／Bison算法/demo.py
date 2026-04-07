"""Yacc/Bison core algorithm MVP: hand-written SLR(1) shift-reduce parser.

This demo shows the source-level mechanics behind parser generators:
1. Build LR(0) item sets (closure/goto).
2. Use FOLLOW sets to fill SLR ACTION/GOTO tables.
3. Drive parsing with state stack + shift/reduce actions.
4. Attach semantic actions to reductions for expression evaluation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple

import numpy as np

EPSILON = "ε"


@dataclass(frozen=True)
class Production:
    head: str
    body: Tuple[str, ...]


@dataclass(frozen=True)
class Token:
    kind: str
    lexeme: str
    value: float | None


@dataclass
class ParserArtifacts:
    productions: List[Production]
    nonterminals: set[str]
    terminals: set[str]
    states: List[FrozenSet[Tuple[int, int]]]
    action_table: Dict[Tuple[int, str], Tuple[str, int]]
    goto_table: Dict[Tuple[int, str], int]


def build_expression_grammar() -> List[Production]:
    """Augmented grammar.

    0) S' -> E
    1) E  -> E + T
    2) E  -> T
    3) T  -> T * F
    4) T  -> F
    5) F  -> ( E )
    6) F  -> id
    """
    return [
        Production("S'", ("E",)),
        Production("E", ("E", "+", "T")),
        Production("E", ("T",)),
        Production("T", ("T", "*", "F")),
        Production("T", ("F",)),
        Production("F", ("(", "E", ")")),
        Production("F", ("id",)),
    ]


def compute_symbols(productions: Sequence[Production]) -> tuple[set[str], set[str]]:
    nonterminals = {p.head for p in productions}
    terminals: set[str] = set()
    for prod in productions:
        for sym in prod.body:
            if sym not in nonterminals and sym != EPSILON:
                terminals.add(sym)
    terminals.add("$")
    return nonterminals, terminals


def compute_first_sets(
    productions: Sequence[Production], nonterminals: set[str], terminals: set[str]
) -> Dict[str, set[str]]:
    first: Dict[str, set[str]] = {nt: set() for nt in nonterminals}
    for t in terminals:
        first[t] = {t}

    changed = True
    while changed:
        changed = False
        for prod in productions:
            head = prod.head
            body = prod.body
            before = len(first[head])
            if not body:
                first[head].add(EPSILON)
            else:
                can_derive_epsilon = True
                for sym in body:
                    sym_first = first.get(sym, {sym})
                    first[head] |= (sym_first - {EPSILON})
                    if EPSILON not in sym_first:
                        can_derive_epsilon = False
                        break
                if can_derive_epsilon:
                    first[head].add(EPSILON)
            if len(first[head]) > before:
                changed = True
    return first


def first_of_sequence(
    seq: Sequence[str], first_sets: Dict[str, set[str]], nonterminals: set[str]
) -> set[str]:
    if not seq:
        return {EPSILON}

    out: set[str] = set()
    all_epsilon = True
    for sym in seq:
        sym_first = first_sets[sym] if sym in nonterminals else {sym}
        out |= (sym_first - {EPSILON})
        if EPSILON not in sym_first:
            all_epsilon = False
            break

    if all_epsilon:
        out.add(EPSILON)
    return out


def compute_follow_sets(
    productions: Sequence[Production],
    nonterminals: set[str],
    start_symbol: str,
    first_sets: Dict[str, set[str]],
) -> Dict[str, set[str]]:
    follow: Dict[str, set[str]] = {nt: set() for nt in nonterminals}
    follow[start_symbol].add("$")

    changed = True
    while changed:
        changed = False
        for prod in productions:
            head = prod.head
            body = prod.body
            for i, sym in enumerate(body):
                if sym not in nonterminals:
                    continue

                beta = body[i + 1 :]
                beta_first = first_of_sequence(beta, first_sets, nonterminals)
                add_from_beta = beta_first - {EPSILON}

                if not add_from_beta.issubset(follow[sym]):
                    follow[sym] |= add_from_beta
                    changed = True

                if not beta or EPSILON in beta_first:
                    if not follow[head].issubset(follow[sym]):
                        follow[sym] |= follow[head]
                        changed = True
    return follow


def index_productions_by_head(productions: Sequence[Production]) -> Dict[str, List[int]]:
    table: Dict[str, List[int]] = {}
    for i, prod in enumerate(productions):
        table.setdefault(prod.head, []).append(i)
    return table


def closure(
    items: Iterable[Tuple[int, int]],
    productions: Sequence[Production],
    prod_by_head: Dict[str, List[int]],
) -> FrozenSet[Tuple[int, int]]:
    out = set(items)
    queue = list(items)

    while queue:
        prod_id, dot = queue.pop()
        body = productions[prod_id].body
        if dot >= len(body):
            continue

        sym = body[dot]
        if sym not in prod_by_head:
            continue

        for next_prod_id in prod_by_head[sym]:
            item = (next_prod_id, 0)
            if item not in out:
                out.add(item)
                queue.append(item)

    return frozenset(out)


def goto(
    state: FrozenSet[Tuple[int, int]],
    symbol: str,
    productions: Sequence[Production],
    prod_by_head: Dict[str, List[int]],
) -> FrozenSet[Tuple[int, int]]:
    advanced = {
        (prod_id, dot + 1)
        for prod_id, dot in state
        if dot < len(productions[prod_id].body) and productions[prod_id].body[dot] == symbol
    }
    if not advanced:
        return frozenset()
    return closure(advanced, productions, prod_by_head)


def build_canonical_lr0_collection(
    productions: Sequence[Production], nonterminals: set[str], terminals: set[str]
) -> tuple[List[FrozenSet[Tuple[int, int]]], Dict[Tuple[int, str], int]]:
    prod_by_head = index_productions_by_head(productions)

    start_state = closure({(0, 0)}, productions, prod_by_head)
    states: List[FrozenSet[Tuple[int, int]]] = [start_state]
    state_id: Dict[FrozenSet[Tuple[int, int]], int] = {start_state: 0}
    transitions: Dict[Tuple[int, str], int] = {}

    symbols = sorted((terminals - {"$"}) | nonterminals)

    queue: deque[FrozenSet[Tuple[int, int]]] = deque([start_state])
    while queue:
        current = queue.popleft()
        sid = state_id[current]

        for sym in symbols:
            nxt = goto(current, sym, productions, prod_by_head)
            if not nxt:
                continue
            if nxt not in state_id:
                state_id[nxt] = len(states)
                states.append(nxt)
                queue.append(nxt)
            transitions[(sid, sym)] = state_id[nxt]

    return states, transitions


def set_action(
    action_table: Dict[Tuple[int, str], Tuple[str, int]],
    key: Tuple[int, str],
    value: Tuple[str, int],
) -> None:
    old = action_table.get(key)
    if old is not None and old != value:
        state, lookahead = key
        raise ValueError(
            f"SLR conflict at state={state}, lookahead={lookahead}: "
            f"existing={old}, new={value}"
        )
    action_table[key] = value


def build_slr_parser() -> ParserArtifacts:
    productions = build_expression_grammar()
    nonterminals, terminals = compute_symbols(productions)

    first_sets = compute_first_sets(productions, nonterminals, terminals)
    # The real grammar start is E (not augmented S').
    follow_sets = compute_follow_sets(productions, nonterminals, start_symbol="E", first_sets=first_sets)

    states, transitions = build_canonical_lr0_collection(productions, nonterminals, terminals)

    action_table: Dict[Tuple[int, str], Tuple[str, int]] = {}
    goto_table: Dict[Tuple[int, str], int] = {}

    for sid, state in enumerate(states):
        for prod_id, dot in state:
            prod = productions[prod_id]
            if dot < len(prod.body):
                sym = prod.body[dot]
                if sym in terminals and sym != "$":
                    next_state = transitions[(sid, sym)]
                    set_action(action_table, (sid, sym), ("shift", next_state))
            else:
                if prod.head == "S'":
                    set_action(action_table, (sid, "$"), ("accept", 0))
                else:
                    for lookahead in sorted(follow_sets[prod.head]):
                        set_action(action_table, (sid, lookahead), ("reduce", prod_id))

        for nt in nonterminals:
            if nt == "S'":
                continue
            nxt = transitions.get((sid, nt))
            if nxt is not None:
                goto_table[(sid, nt)] = nxt

    return ParserArtifacts(
        productions=productions,
        nonterminals=nonterminals,
        terminals=terminals,
        states=states,
        action_table=action_table,
        goto_table=goto_table,
    )


def tokenize(expression: str, variables: Dict[str, float] | None = None) -> List[Token]:
    vars_map = variables or {}
    tokens: List[Token] = []
    i = 0

    while i < len(expression):
        ch = expression[i]
        if ch.isspace():
            i += 1
            continue

        if ch in {"+", "*", "(", ")"}:
            tokens.append(Token(kind=ch, lexeme=ch, value=None))
            i += 1
            continue

        if ch.isdigit() or ch == ".":
            j = i
            dot_count = 0
            while j < len(expression) and (expression[j].isdigit() or expression[j] == "."):
                if expression[j] == ".":
                    dot_count += 1
                j += 1
            lexeme = expression[i:j]
            if dot_count > 1 or lexeme == ".":
                raise ValueError(f"Invalid numeric literal: {lexeme}")
            tokens.append(Token(kind="id", lexeme=lexeme, value=float(lexeme)))
            i = j
            continue

        if ch.isalpha() or ch == "_":
            j = i
            while j < len(expression) and (expression[j].isalnum() or expression[j] == "_"):
                j += 1
            name = expression[i:j]
            if name not in vars_map:
                raise ValueError(f"Unknown identifier '{name}'. Provide it via variables dict.")
            tokens.append(Token(kind="id", lexeme=name, value=float(vars_map[name])))
            i = j
            continue

        raise ValueError(f"Unsupported character at position {i}: {ch!r}")

    tokens.append(Token(kind="$", lexeme="$", value=None))
    return tokens


def apply_semantic_action(prod_id: int, rhs_values: Sequence[float | str]) -> float:
    if prod_id == 1:  # E -> E + T
        return float(rhs_values[0]) + float(rhs_values[2])
    if prod_id == 2:  # E -> T
        return float(rhs_values[0])
    if prod_id == 3:  # T -> T * F
        return float(rhs_values[0]) * float(rhs_values[2])
    if prod_id == 4:  # T -> F
        return float(rhs_values[0])
    if prod_id == 5:  # F -> ( E )
        return float(rhs_values[1])
    if prod_id == 6:  # F -> id
        return float(rhs_values[0])
    if prod_id == 0:  # S' -> E (normally not reduced in ACTION table)
        return float(rhs_values[0])
    raise ValueError(f"Unexpected production id: {prod_id}")


def parse_and_evaluate(
    expression: str,
    parser: ParserArtifacts,
    variables: Dict[str, float] | None = None,
) -> tuple[float, List[str]]:
    tokens = tokenize(expression, variables)

    state_stack: List[int] = [0]
    symbol_stack: List[str] = []
    value_stack: List[float | str] = []
    trace: List[str] = []

    cursor = 0
    while True:
        state = state_stack[-1]
        lookahead = tokens[cursor].kind
        action = parser.action_table.get((state, lookahead))
        if action is None:
            raise SyntaxError(
                f"Parse error at token index {cursor}, state={state}, lookahead={lookahead}"
            )

        act, arg = action
        if act == "shift":
            tok = tokens[cursor]
            shifted_value: float | str = tok.value if tok.kind == "id" else tok.lexeme
            state_stack.append(arg)
            symbol_stack.append(tok.kind)
            value_stack.append(shifted_value)
            trace.append(f"shift {tok.kind} -> state {arg}")
            cursor += 1
            continue

        if act == "reduce":
            prod = parser.productions[arg]
            rhs_len = len(prod.body)

            if rhs_len > 0:
                rhs_values = value_stack[-rhs_len:]
                del value_stack[-rhs_len:]
                del symbol_stack[-rhs_len:]
                del state_stack[-rhs_len:]
            else:
                rhs_values = []

            reduced_value = apply_semantic_action(arg, rhs_values)
            goto_state = parser.goto_table.get((state_stack[-1], prod.head))
            if goto_state is None:
                raise SyntaxError(
                    f"Missing goto entry for state={state_stack[-1]}, nonterminal={prod.head}"
                )

            state_stack.append(goto_state)
            symbol_stack.append(prod.head)
            value_stack.append(reduced_value)

            rhs_text = " ".join(prod.body) if prod.body else EPSILON
            trace.append(f"reduce {prod.head} -> {rhs_text}; goto {goto_state}")
            continue

        if act == "accept":
            if len(value_stack) != 1:
                raise RuntimeError(f"Unexpected value stack at accept: {value_stack}")
            trace.append("accept")
            return float(value_stack[-1]), trace

        raise RuntimeError(f"Unknown action type: {act}")


def run_fixed_cases(parser: ParserArtifacts) -> None:
    cases = [
        ("2+3*4", None, 14.0),
        ("(2+3)*4", None, 20.0),
        ("7*(8+1)+5", None, 68.0),
        ("12+3*(4+5)", None, 39.0),
        ("x*(y+2)", {"x": 3.0, "y": 10.0}, 36.0),
    ]

    preds: List[float] = []
    golds: List[float] = []

    print("[Case Results]")
    first_trace: List[str] | None = None
    for expr, env, expected in cases:
        value, trace = parse_and_evaluate(expr, parser, env)
        if first_trace is None:
            first_trace = trace
        preds.append(value)
        golds.append(expected)
        print(f"  {expr:>14s} => {value:.6f}")

    pred_arr = np.array(preds, dtype=np.float64)
    gold_arr = np.array(golds, dtype=np.float64)
    if not np.allclose(pred_arr, gold_arr):
        raise AssertionError(f"Value mismatch. pred={pred_arr}, gold={gold_arr}")

    print("\n[Trace of first expression]")
    assert first_trace is not None
    for step in first_trace:
        print(f"  {step}")


def run_negative_case(parser: ParserArtifacts) -> None:
    bad_expr = "2+*3"
    try:
        parse_and_evaluate(bad_expr, parser)
    except SyntaxError as exc:
        print("\n[Negative Case]")
        print(f"  {bad_expr!r} correctly failed with SyntaxError: {exc}")
        return
    raise AssertionError("Negative case should fail but unexpectedly succeeded.")


def main() -> None:
    parser = build_slr_parser()

    print("SLR(1) parser table is built successfully.")
    print(f"  states        : {len(parser.states)}")
    print(f"  action entries: {len(parser.action_table)}")
    print(f"  goto entries  : {len(parser.goto_table)}")

    run_fixed_cases(parser)
    run_negative_case(parser)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
