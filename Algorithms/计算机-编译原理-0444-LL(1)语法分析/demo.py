"""LL(1) parser MVP for CS-0284.

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
ParseTable = dict[tuple[str, str], Production]


@dataclass(frozen=True)
class ParseStep:
    step_id: int
    stack: str
    remaining_input: str
    action: str


@dataclass
class ParseResult:
    accepted: bool
    steps: list[ParseStep]
    error: str | None = None


def build_expression_grammar() -> Grammar:
    """Classic LL(1) expression grammar.

    E  -> T E'
    E' -> + T E' | ε
    T  -> F T'
    T' -> * F T' | ε
    F  -> ( E ) | id
    """
    return {
        "E": [("T", "E'")],
        "E'": [('+', 'T', "E'"), (EPS,)],
        "T": [("F", "T'")],
        "T'": [('*', 'F', "T'"), (EPS,)],
        "F": [('(', 'E', ')'), ('id',)],
    }


def collect_terminals(grammar: Grammar) -> set[str]:
    nonterminals = set(grammar.keys())
    terminals: set[str] = set()
    for productions in grammar.values():
        for rhs in productions:
            for symbol in rhs:
                if symbol != EPS and symbol not in nonterminals:
                    terminals.add(symbol)
    return terminals


def first_of_sequence(
    symbols: Production,
    first_sets: dict[str, set[str]],
    nonterminals: set[str],
) -> set[str]:
    if not symbols or symbols == (EPS,):
        return {EPS}

    result: set[str] = set()
    nullable_prefix = True

    for symbol in symbols:
        if symbol == EPS:
            result.add(EPS)
            break

        if symbol in nonterminals:
            result |= first_sets[symbol] - {EPS}
            if EPS in first_sets[symbol]:
                continue
            nullable_prefix = False
            break

        result.add(symbol)
        nullable_prefix = False
        break

    if nullable_prefix:
        result.add(EPS)

    return result


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


def compute_follow_sets(
    grammar: Grammar,
    start_symbol: str,
    first_sets: dict[str, set[str]],
) -> dict[str, set[str]]:
    nonterminals = set(grammar.keys())
    follow_sets: dict[str, set[str]] = {nt: set() for nt in nonterminals}
    follow_sets[start_symbol].add(END)

    changed = True
    while changed:
        changed = False

        for lhs, productions in grammar.items():
            for rhs in productions:
                for idx, symbol in enumerate(rhs):
                    if symbol not in nonterminals:
                        continue

                    beta = rhs[idx + 1 :]
                    first_beta = first_of_sequence(beta, first_sets, nonterminals)

                    old_size = len(follow_sets[symbol])
                    follow_sets[symbol] |= first_beta - {EPS}

                    if not beta or EPS in first_beta:
                        follow_sets[symbol] |= follow_sets[lhs]

                    if len(follow_sets[symbol]) != old_size:
                        changed = True

    return follow_sets


def build_ll1_table(
    grammar: Grammar,
    first_sets: dict[str, set[str]],
    follow_sets: dict[str, set[str]],
) -> tuple[ParseTable, list[str]]:
    nonterminals = set(grammar.keys())
    table: ParseTable = {}
    conflicts: list[str] = []

    for lhs, productions in grammar.items():
        for rhs in productions:
            rhs_first = first_of_sequence(rhs, first_sets, nonterminals)

            for terminal in sorted(rhs_first - {EPS}):
                key = (lhs, terminal)
                old = table.get(key)
                if old is not None and old != rhs:
                    conflicts.append(
                        f"M[{lhs}, {terminal}] conflict: {production_to_str(old)} vs {production_to_str(rhs)}"
                    )
                else:
                    table[key] = rhs

            if EPS in rhs_first:
                for terminal in sorted(follow_sets[lhs]):
                    key = (lhs, terminal)
                    old = table.get(key)
                    if old is not None and old != rhs:
                        conflicts.append(
                            f"M[{lhs}, {terminal}] conflict: {production_to_str(old)} vs {production_to_str(rhs)}"
                        )
                    else:
                        table[key] = rhs

    return table, conflicts


def parse_tokens(
    tokens: list[str],
    start_symbol: str,
    grammar: Grammar,
    table: ParseTable,
) -> ParseResult:
    terminals = collect_terminals(grammar)

    stack: list[str] = [END, start_symbol]
    stream = tokens + [END]
    cursor = 0
    steps: list[ParseStep] = []

    for step_id in range(1, 10_000):
        stack_view = " ".join(stack)
        input_view = " ".join(stream[cursor:])

        top = stack[-1]
        lookahead = stream[cursor]

        if top == END and lookahead == END:
            steps.append(
                ParseStep(
                    step_id=step_id,
                    stack=stack_view,
                    remaining_input=input_view,
                    action="accept",
                )
            )
            return ParseResult(accepted=True, steps=steps)

        if top in terminals or top == END:
            if top == lookahead:
                stack.pop()
                cursor += 1
                action = f"match '{lookahead}'"
                steps.append(
                    ParseStep(
                        step_id=step_id,
                        stack=stack_view,
                        remaining_input=input_view,
                        action=action,
                    )
                )
                continue

            error = f"terminal mismatch: top='{top}', lookahead='{lookahead}'"
            steps.append(
                ParseStep(
                    step_id=step_id,
                    stack=stack_view,
                    remaining_input=input_view,
                    action=f"error: {error}",
                )
            )
            return ParseResult(accepted=False, steps=steps, error=error)

        key = (top, lookahead)
        production = table.get(key)
        if production is None:
            expected = sorted({t for (nt, t) in table.keys() if nt == top})
            error = f"no table entry for ({top}, {lookahead}), expected one of {expected}"
            steps.append(
                ParseStep(
                    step_id=step_id,
                    stack=stack_view,
                    remaining_input=input_view,
                    action=f"error: {error}",
                )
            )
            return ParseResult(accepted=False, steps=steps, error=error)

        stack.pop()
        if production != (EPS,):
            for symbol in reversed(production):
                stack.append(symbol)

        action = f"expand {top} -> {production_to_str(production)}"
        steps.append(
            ParseStep(
                step_id=step_id,
                stack=stack_view,
                remaining_input=input_view,
                action=action,
            )
        )

    return ParseResult(
        accepted=False,
        steps=steps,
        error="exceeded max parser steps (possible infinite loop)",
    )


def production_to_str(rhs: Production) -> str:
    if rhs == (EPS,):
        return EPS
    return " ".join(rhs)


def format_set(items: set[str]) -> str:
    ordered = sorted(items, key=lambda x: (x == EPS, x))
    return "{" + ", ".join(ordered) + "}"


def print_first_follow(
    first_sets: dict[str, set[str]],
    follow_sets: dict[str, set[str]],
    nonterminals: list[str],
) -> None:
    print("\nFIRST sets:")
    for nt in nonterminals:
        print(f"  FIRST({nt}) = {format_set(first_sets[nt])}")

    print("\nFOLLOW sets:")
    for nt in nonterminals:
        print(f"  FOLLOW({nt}) = {format_set(follow_sets[nt])}")


def print_parse_table(
    table: ParseTable,
    nonterminals: list[str],
    terminals: list[str],
) -> None:
    columns = terminals + [END]
    widths = [max(6, len(col)) for col in ["NT"] + columns]

    def cell_text(nt: str, terminal: str) -> str:
        rhs = table.get((nt, terminal))
        if rhs is None:
            return "."
        return production_to_str(rhs)

    for idx, nt in enumerate(nonterminals):
        widths[0] = max(widths[0], len(nt))
        for col_idx, terminal in enumerate(columns, start=1):
            widths[col_idx] = max(widths[col_idx], len(cell_text(nt, terminal)))

    header_cells = ["NT"] + columns
    header = " | ".join(text.ljust(widths[i]) for i, text in enumerate(header_cells))
    sep = "-+-".join("-" * widths[i] for i in range(len(widths)))

    print("\nLL(1) parse table:")
    print(header)
    print(sep)
    for nt in nonterminals:
        row = [nt] + [cell_text(nt, terminal) for terminal in columns]
        print(" | ".join(text.ljust(widths[i]) for i, text in enumerate(row)))


def print_parse_trace(case_name: str, tokens: list[str], result: ParseResult) -> None:
    print(f"\nCase: {case_name}")
    print(f"Input tokens: {' '.join(tokens)}")
    print("Step | Stack                | Input                | Action")
    print("-----+----------------------+----------------------+-------------------------------")
    for step in result.steps:
        print(
            f"{step.step_id:>4} | {step.stack[:20]:<20} | {step.remaining_input[:20]:<20} | {step.action}"
        )
    if result.accepted:
        print("Result: ACCEPT")
    else:
        print(f"Result: REJECT ({result.error})")


def run_demo() -> None:
    grammar = build_expression_grammar()
    start_symbol = "E"
    nonterminals = sorted(grammar.keys())
    terminals = sorted(collect_terminals(grammar))

    print("Grammar productions:")
    for lhs in nonterminals:
        rhs_text = " | ".join(production_to_str(rhs) for rhs in grammar[lhs])
        print(f"  {lhs} -> {rhs_text}")

    first_sets = compute_first_sets(grammar)
    follow_sets = compute_follow_sets(grammar, start_symbol, first_sets)
    print_first_follow(first_sets, follow_sets, nonterminals)

    table, conflicts = build_ll1_table(grammar, first_sets, follow_sets)
    print_parse_table(table, nonterminals, terminals)

    if conflicts:
        print("\nConflicts detected:")
        for conflict in conflicts:
            print(f"  - {conflict}")
        raise RuntimeError("Grammar is not LL(1): parse table contains conflicts")

    cases: list[tuple[str, list[str], bool]] = [
        ("valid_1", ["id", "+", "id", "*", "id"], True),
        ("valid_2", ["(", "id", "+", "id", ")", "*", "id"], True),
        ("invalid_1", ["id", "+", "*", "id"], False),
    ]

    results: list[ParseResult] = []
    for case_name, tokens, expected in cases:
        result = parse_tokens(tokens, start_symbol, grammar, table)
        print_parse_trace(case_name, tokens, result)
        assert result.accepted == expected, (
            f"case {case_name} failed expectation: expected {expected}, got {result.accepted}"
        )
        results.append(result)

    table_cells = len(nonterminals) * (len(terminals) + 1)
    used_cells = len(table)
    fill_ratio = float(np.array([used_cells / table_cells], dtype=float)[0])

    step_counts = np.array([len(r.steps) for r in results], dtype=np.int64)
    print("\nSummary:")
    print(f"  Nonterminals: {len(nonterminals)}")
    print(f"  Terminals(+END): {len(terminals) + 1}")
    print(f"  Parse table used cells: {used_cells}/{table_cells} ({fill_ratio:.2%})")
    print(
        "  Trace steps stats: "
        f"min={int(step_counts.min())}, max={int(step_counts.max())}, mean={float(step_counts.mean()):.2f}"
    )
    print("\nAll LL(1) checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
