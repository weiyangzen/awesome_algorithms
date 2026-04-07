"""Canonical LR(1) parser MVP for CS-0288.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = "ε"
END = "$"

Production = tuple[str, ...]
Grammar = dict[str, list[Production]]


@dataclass(frozen=True)
class LR1Item:
    lhs: str
    rhs: Production
    dot: int
    lookahead: str


@dataclass(frozen=True)
class ParseStep:
    step_id: int
    state_stack: str
    symbol_stack: str
    remaining_input: str
    action: str


@dataclass
class ParseResult:
    accepted: bool
    steps: list[ParseStep]
    error: str | None = None


def build_expression_grammar() -> tuple[Grammar, str]:
    """Build a classic expression grammar that is LR(1)."""
    grammar: Grammar = {
        "S": [("E",)],
        "E": [("E", "+", "T"), ("T",)],
        "T": [("T", "*", "F"), ("F",)],
        "F": [("(", "E", ")"), ("id",)],
    }
    return grammar, "S"


def collect_terminals(grammar: Grammar) -> set[str]:
    nonterminals = set(grammar.keys())
    terminals: set[str] = set()
    for productions in grammar.values():
        for rhs in productions:
            for symbol in rhs:
                if symbol != EPS and symbol not in nonterminals:
                    terminals.add(symbol)
    return terminals


def augment_grammar(grammar: Grammar, start_symbol: str) -> tuple[Grammar, str]:
    aug_start = f"{start_symbol}'"
    while aug_start in grammar:
        aug_start += "'"

    augmented: Grammar = {aug_start: [(start_symbol,)]}
    for lhs, productions in grammar.items():
        augmented[lhs] = list(productions)
    return augmented, aug_start


def enumerate_productions(grammar: Grammar) -> list[tuple[str, Production]]:
    ordered: list[tuple[str, Production]] = []
    for lhs, productions in grammar.items():
        for rhs in productions:
            ordered.append((lhs, rhs))
    return ordered


def first_of_sequence(
    symbols: Production,
    first_sets: dict[str, set[str]],
    nonterminals: set[str],
) -> set[str]:
    if not symbols or symbols == (EPS,):
        return {EPS}

    out: set[str] = set()
    nullable_prefix = True

    for symbol in symbols:
        if symbol == EPS:
            out.add(EPS)
            break

        if symbol in nonterminals:
            out |= first_sets[symbol] - {EPS}
            if EPS in first_sets[symbol]:
                continue
            nullable_prefix = False
            break

        out.add(symbol)
        nullable_prefix = False
        break

    if nullable_prefix:
        out.add(EPS)

    return out


def compute_first_sets(grammar: Grammar) -> dict[str, set[str]]:
    nonterminals = set(grammar.keys())
    first_sets: dict[str, set[str]] = {nt: set() for nt in nonterminals}

    changed = True
    while changed:
        changed = False
        for lhs, productions in grammar.items():
            for rhs in productions:
                rhs_first = first_of_sequence(rhs, first_sets, nonterminals)
                old_size = len(first_sets[lhs])
                first_sets[lhs] |= rhs_first
                if len(first_sets[lhs]) != old_size:
                    changed = True

    return first_sets


def closure_lr1(
    items: frozenset[LR1Item],
    grammar: Grammar,
    first_sets: dict[str, set[str]],
) -> frozenset[LR1Item]:
    nonterminals = set(grammar.keys())
    closure: set[LR1Item] = set(items)

    changed = True
    while changed:
        changed = False
        to_add: set[LR1Item] = set()

        for item in closure:
            if item.dot >= len(item.rhs):
                continue

            symbol = item.rhs[item.dot]
            if symbol not in nonterminals:
                continue

            beta = item.rhs[item.dot + 1 :]
            lookahead_seed = beta + (item.lookahead,)
            lookaheads = first_of_sequence(lookahead_seed, first_sets, nonterminals) - {EPS}

            for prod_rhs in grammar[symbol]:
                for lookahead in lookaheads:
                    candidate = LR1Item(lhs=symbol, rhs=prod_rhs, dot=0, lookahead=lookahead)
                    if candidate not in closure:
                        to_add.add(candidate)

        if to_add:
            closure |= to_add
            changed = True

    return frozenset(closure)


def goto_lr1(
    items: frozenset[LR1Item],
    symbol: str,
    grammar: Grammar,
    first_sets: dict[str, set[str]],
) -> frozenset[LR1Item]:
    moved: set[LR1Item] = set()

    for item in items:
        if item.dot < len(item.rhs) and item.rhs[item.dot] == symbol:
            moved.add(
                LR1Item(lhs=item.lhs, rhs=item.rhs, dot=item.dot + 1, lookahead=item.lookahead)
            )

    if not moved:
        return frozenset()
    return closure_lr1(frozenset(moved), grammar, first_sets)


def canonical_lr1_collection(
    grammar: Grammar,
    aug_start: str,
    terminals: set[str],
    nonterminals: set[str],
    first_sets: dict[str, set[str]],
) -> tuple[list[frozenset[LR1Item]], dict[tuple[int, str], int]]:
    start_item = LR1Item(lhs=aug_start, rhs=grammar[aug_start][0], dot=0, lookahead=END)
    start_state = closure_lr1(frozenset({start_item}), grammar, first_sets)

    states: list[frozenset[LR1Item]] = [start_state]
    state_id = {start_state: 0}
    transitions: dict[tuple[int, str], int] = {}

    symbols = sorted(terminals | nonterminals)
    cursor = 0
    while cursor < len(states):
        state = states[cursor]
        for symbol in symbols:
            nxt = goto_lr1(state, symbol, grammar, first_sets)
            if not nxt:
                continue

            sid = state_id.get(nxt)
            if sid is None:
                sid = len(states)
                states.append(nxt)
                state_id[nxt] = sid

            transitions[(cursor, symbol)] = sid

        cursor += 1

    return states, transitions


def production_to_str(lhs: str, rhs: Production) -> str:
    body = EPS if rhs == (EPS,) else " ".join(rhs)
    return f"{lhs} -> {body}"


def action_to_str(action: tuple[str, int | None]) -> str:
    kind, value = action
    if kind == "shift":
        return f"s{value}"
    if kind == "reduce":
        return f"r{value}"
    return "acc"


def build_lr1_tables(
    states: list[frozenset[LR1Item]],
    transitions: dict[tuple[int, str], int],
    productions: list[tuple[str, Production]],
    terminals: set[str],
    nonterminals: set[str],
    aug_start: str,
) -> tuple[
    dict[tuple[int, str], tuple[str, int | None]],
    dict[tuple[int, str], int],
    list[str],
]:
    production_to_idx = {prod: idx for idx, prod in enumerate(productions)}

    action_table: dict[tuple[int, str], tuple[str, int | None]] = {}
    goto_table: dict[tuple[int, str], int] = {}
    conflicts: list[str] = []

    def set_action(state: int, terminal: str, action: tuple[str, int | None]) -> None:
        key = (state, terminal)
        old = action_table.get(key)
        if old is not None and old != action:
            conflicts.append(
                f"ACTION[{state}, {terminal}] conflict: {action_to_str(old)} vs {action_to_str(action)}"
            )
            return
        action_table[key] = action

    def set_goto(state: int, nt: str, target: int) -> None:
        key = (state, nt)
        old = goto_table.get(key)
        if old is not None and old != target:
            conflicts.append(f"GOTO[{state}, {nt}] conflict: {old} vs {target}")
            return
        goto_table[key] = target

    for sid, state in enumerate(states):
        for item in state:
            if item.dot < len(item.rhs):
                symbol = item.rhs[item.dot]
                target = transitions.get((sid, symbol))
                if target is None:
                    continue

                if symbol in terminals:
                    set_action(sid, symbol, ("shift", target))
                elif symbol in nonterminals:
                    set_goto(sid, symbol, target)
                continue

            if item.lhs == aug_start and item.lookahead == END:
                set_action(sid, END, ("accept", None))
            else:
                prod = (item.lhs, item.rhs)
                prod_idx = production_to_idx[prod]
                set_action(sid, item.lookahead, ("reduce", prod_idx))

    return action_table, goto_table, conflicts


def parse_tokens(
    tokens: list[str],
    productions: list[tuple[str, Production]],
    action_table: dict[tuple[int, str], tuple[str, int | None]],
    goto_table: dict[tuple[int, str], int],
) -> ParseResult:
    stream = tokens + [END]
    cursor = 0

    state_stack: list[int] = [0]
    symbol_stack: list[str] = [END]

    steps: list[ParseStep] = []

    for step_id in range(1, 10_000):
        state = state_stack[-1]
        lookahead = stream[cursor]
        action = action_table.get((state, lookahead))

        action_text = ""
        if action is None:
            expected = sorted(t for (s, t) in action_table.keys() if s == state)
            error = (
                f"no ACTION entry for state={state}, lookahead='{lookahead}', "
                f"expected one of {expected}"
            )
            action_text = f"error: {error}"
            steps.append(
                ParseStep(
                    step_id=step_id,
                    state_stack=" ".join(map(str, state_stack)),
                    symbol_stack=" ".join(symbol_stack),
                    remaining_input=" ".join(stream[cursor:]),
                    action=action_text,
                )
            )
            return ParseResult(accepted=False, steps=steps, error=error)

        kind, value = action

        if kind == "shift":
            assert value is not None
            symbol_stack.append(lookahead)
            state_stack.append(value)
            cursor += 1
            action_text = f"shift to state {value} on '{lookahead}'"

        elif kind == "reduce":
            assert value is not None
            lhs, rhs = productions[value]
            pop_count = 0 if rhs == (EPS,) else len(rhs)

            for _ in range(pop_count):
                symbol_stack.pop()
                state_stack.pop()

            goto_state = goto_table.get((state_stack[-1], lhs))
            if goto_state is None:
                error = f"missing GOTO for state={state_stack[-1]}, nonterminal='{lhs}'"
                action_text = f"error: {error}"
                steps.append(
                    ParseStep(
                        step_id=step_id,
                        state_stack=" ".join(map(str, state_stack)),
                        symbol_stack=" ".join(symbol_stack),
                        remaining_input=" ".join(stream[cursor:]),
                        action=action_text,
                    )
                )
                return ParseResult(accepted=False, steps=steps, error=error)

            symbol_stack.append(lhs)
            state_stack.append(goto_state)
            action_text = (
                f"reduce by r{value}: {production_to_str(lhs, rhs)}; goto state {goto_state}"
            )

        elif kind == "accept":
            action_text = "accept"
            steps.append(
                ParseStep(
                    step_id=step_id,
                    state_stack=" ".join(map(str, state_stack)),
                    symbol_stack=" ".join(symbol_stack),
                    remaining_input=" ".join(stream[cursor:]),
                    action=action_text,
                )
            )
            return ParseResult(accepted=True, steps=steps)

        else:
            error = f"unknown action kind: {kind}"
            action_text = f"error: {error}"
            steps.append(
                ParseStep(
                    step_id=step_id,
                    state_stack=" ".join(map(str, state_stack)),
                    symbol_stack=" ".join(symbol_stack),
                    remaining_input=" ".join(stream[cursor:]),
                    action=action_text,
                )
            )
            return ParseResult(accepted=False, steps=steps, error=error)

        steps.append(
            ParseStep(
                step_id=step_id,
                state_stack=" ".join(map(str, state_stack)),
                symbol_stack=" ".join(symbol_stack),
                remaining_input=" ".join(stream[cursor:]),
                action=action_text,
            )
        )

    return ParseResult(
        accepted=False,
        steps=steps,
        error="exceeded max parser steps (possible infinite loop)",
    )


def print_productions(productions: list[tuple[str, Production]]) -> None:
    print("Productions (index used by reduce action rN):")
    for idx, (lhs, rhs) in enumerate(productions):
        print(f"  r{idx}: {production_to_str(lhs, rhs)}")


def print_states(states: list[frozenset[LR1Item]], title: str, max_states: int = 10) -> None:
    print(f"\n{title} (showing up to {max_states} states):")
    for sid, state in enumerate(states[:max_states]):
        print(f"State {sid}:")
        for item in sorted(
            state,
            key=lambda x: (x.lhs, x.rhs, x.dot, x.lookahead),
        ):
            rhs = list(item.rhs)
            rhs.insert(item.dot, "·")
            rhs_text = " ".join(rhs)
            print(f"  {item.lhs} -> {rhs_text}, {item.lookahead}")


def print_action_goto_tables(
    action_table: dict[tuple[int, str], tuple[str, int | None]],
    goto_table: dict[tuple[int, str], int],
    terminals: list[str],
    nonterminals: list[str],
    num_states: int,
) -> None:
    print("\nLR(1) ACTION table:")
    header = ["state"] + terminals + [END]
    widths = [max(5, len(h)) for h in header]

    for sid in range(num_states):
        widths[0] = max(widths[0], len(str(sid)))
        for idx, t in enumerate(terminals + [END], start=1):
            val = action_table.get((sid, t))
            text = "." if val is None else action_to_str(val)
            widths[idx] = max(widths[idx], len(text))

    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("-+-".join("-" * w for w in widths))

    for sid in range(num_states):
        row = [str(sid)]
        for t in terminals + [END]:
            val = action_table.get((sid, t))
            row.append("." if val is None else action_to_str(val))
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))

    print("\nLR(1) GOTO table:")
    header = ["state"] + nonterminals
    widths = [max(5, len(h)) for h in header]

    for sid in range(num_states):
        widths[0] = max(widths[0], len(str(sid)))
        for idx, nt in enumerate(nonterminals, start=1):
            val = goto_table.get((sid, nt))
            text = "." if val is None else str(val)
            widths[idx] = max(widths[idx], len(text))

    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("-+-".join("-" * w for w in widths))

    for sid in range(num_states):
        row = [str(sid)]
        for nt in nonterminals:
            val = goto_table.get((sid, nt))
            row.append("." if val is None else str(val))
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def print_parse_trace(case_name: str, tokens: list[str], result: ParseResult) -> None:
    print(f"\nCase: {case_name}")
    print(f"Input tokens: {' '.join(tokens)}")
    print("Step | State Stack           | Symbol Stack          | Remaining Input       | Action")
    print("-----+-----------------------+-----------------------+-----------------------+------------------------------")
    for step in result.steps:
        print(
            f"{step.step_id:>4} | {step.state_stack[:21]:<21} | "
            f"{step.symbol_stack[:21]:<21} | {step.remaining_input[:21]:<21} | {step.action}"
        )

    if result.accepted:
        print("Result: ACCEPT")
    else:
        print(f"Result: REJECT ({result.error})")


def run_demo() -> None:
    grammar, start_symbol = build_expression_grammar()
    terminals = collect_terminals(grammar)

    augmented_grammar, aug_start = augment_grammar(grammar, start_symbol)
    productions = enumerate_productions(augmented_grammar)

    nonterminals = set(augmented_grammar.keys())
    first_sets = compute_first_sets(augmented_grammar)

    lr1_states, lr1_transitions = canonical_lr1_collection(
        grammar=augmented_grammar,
        aug_start=aug_start,
        terminals=terminals,
        nonterminals=nonterminals,
        first_sets=first_sets,
    )

    action_table, goto_table, conflicts = build_lr1_tables(
        states=lr1_states,
        transitions=lr1_transitions,
        productions=productions,
        terminals=terminals,
        nonterminals=nonterminals,
        aug_start=aug_start,
    )

    print_productions(productions)
    print_states(lr1_states, "Canonical LR(1) items")

    ordered_terminals = sorted(terminals)
    ordered_nonterminals = sorted(nt for nt in nonterminals if nt != aug_start)
    print_action_goto_tables(
        action_table,
        goto_table,
        ordered_terminals,
        ordered_nonterminals,
        len(lr1_states),
    )

    if conflicts:
        print("\nConflicts:")
        for conflict in conflicts:
            print(f"  - {conflict}")
        raise RuntimeError("LR(1) table has conflicts")

    cases: list[tuple[str, list[str], bool]] = [
        ("valid_1", ["id", "+", "id", "*", "id"], True),
        ("valid_2", ["(", "id", "+", "id", ")", "*", "id"], True),
        ("valid_3", ["id", "*", "(", "id", "+", "id", ")"], True),
        ("invalid_1", ["id", "+", "*", "id"], False),
        ("invalid_2", ["(", "id", "+", "id", "*", "id"], False),
    ]

    results: list[ParseResult] = []
    for case_name, tokens, expected in cases:
        result = parse_tokens(tokens, productions, action_table, goto_table)
        print_parse_trace(case_name, tokens, result)
        assert result.accepted == expected, (
            f"case {case_name} failed expectation: expected {expected}, got {result.accepted}"
        )
        results.append(result)

    state_count = len(lr1_states)
    action_slots = state_count * (len(terminals) + 1)
    action_used = len(action_table)
    action_fill = float(np.array([action_used / action_slots], dtype=np.float64)[0])
    step_counts = np.array([len(result.steps) for result in results], dtype=np.int64)

    print("\nSummary:")
    print(f"  Canonical LR(1) states: {state_count}")
    print(f"  ACTION entries: {action_used}/{action_slots} ({action_fill:.2%})")
    print(
        "  Parse steps: "
        f"min={int(step_counts.min())}, max={int(step_counts.max())}, mean={float(step_counts.mean()):.2f}"
    )
    print("\nAll LR(1) checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
