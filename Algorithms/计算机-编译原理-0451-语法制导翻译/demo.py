"""Minimal runnable MVP for syntax-directed translation (SDT).

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class Token:
    kind: str
    lexeme: str
    position: int


@dataclass(frozen=True)
class Quad:
    op: str
    arg1: str
    arg2: str
    result: str


@dataclass(frozen=True)
class TranslationResult:
    expr: str
    result_place: str
    quads: List[Quad]


TOKEN_PATTERN = re.compile(r"[A-Za-z_]\w*|\d+|[+*()]")


def tokenize(source: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    n = len(source)

    while pos < n:
        if source[pos].isspace():
            pos += 1
            continue

        match = TOKEN_PATTERN.match(source, pos)
        if match is None:
            raise SyntaxError(f"Unexpected character at position {pos}: {source[pos]!r}")

        lexeme = match.group(0)
        if lexeme == "+":
            kind = "PLUS"
        elif lexeme == "*":
            kind = "MUL"
        elif lexeme == "(":
            kind = "LPAREN"
        elif lexeme == ")":
            kind = "RPAREN"
        elif lexeme.isdigit():
            kind = "NUM"
        else:
            kind = "ID"

        tokens.append(Token(kind=kind, lexeme=lexeme, position=pos))
        pos = match.end()

    tokens.append(Token(kind="EOF", lexeme="$", position=n))
    return tokens


class Emitter:
    def __init__(self) -> None:
        self._temp_count = 0
        self.quads: List[Quad] = []

    def new_temp(self) -> str:
        self._temp_count += 1
        return f"t{self._temp_count}"

    def emit_binary(self, op: str, left: str, right: str) -> str:
        temp = self.new_temp()
        self.quads.append(Quad(op=op, arg1=left, arg2=right, result=temp))
        return temp


class Parser:
    """Recursive-descent translator with SDT semantic actions.

    Grammar (left recursion removed):
        E  -> T E'
        E' -> + T E' | ε
        T  -> F T'
        T' -> * F T' | ε
        F  -> ( E ) | id | num

    Semantic sketch:
        E.place  = E'.syn, with E'.inh = T.place
        E'.syn   = temp(E'.inh + T.place) or E'.inh on ε
        T.place  = T'.syn, with T'.inh = F.place
        T'.syn   = temp(T'.inh * F.place) or T'.inh on ε
        F.place  = id.lexeme | num.lexeme | E.place
    """

    def __init__(self, tokens: Sequence[Token]) -> None:
        self.tokens = list(tokens)
        self.pos = 0
        self.emitter = Emitter()

    def current(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.current()
        self.pos += 1
        return tok

    def expect(self, kind: str) -> Token:
        tok = self.current()
        if tok.kind != kind:
            raise SyntaxError(
                f"Expected {kind}, got {tok.kind}({tok.lexeme!r}) at position {tok.position}"
            )
        return self.advance()

    def translate(self, expr: str) -> TranslationResult:
        result_place = self.parse_E()
        self.expect("EOF")
        return TranslationResult(expr=expr, result_place=result_place, quads=list(self.emitter.quads))

    def parse_E(self) -> str:
        # E -> T E'
        t_place = self.parse_T()
        return self.parse_E_prime(inh=t_place)

    def parse_E_prime(self, inh: str) -> str:
        # E' -> + T E' | ε
        if self.current().kind == "PLUS":
            self.advance()
            t_place = self.parse_T()
            temp = self.emitter.emit_binary("+", inh, t_place)
            return self.parse_E_prime(inh=temp)
        return inh

    def parse_T(self) -> str:
        # T -> F T'
        f_place = self.parse_F()
        return self.parse_T_prime(inh=f_place)

    def parse_T_prime(self, inh: str) -> str:
        # T' -> * F T' | ε
        if self.current().kind == "MUL":
            self.advance()
            f_place = self.parse_F()
            temp = self.emitter.emit_binary("*", inh, f_place)
            return self.parse_T_prime(inh=temp)
        return inh

    def parse_F(self) -> str:
        # F -> (E) | id | num
        tok = self.current()
        if tok.kind in {"ID", "NUM"}:
            self.advance()
            return tok.lexeme

        if tok.kind == "LPAREN":
            self.advance()
            e_place = self.parse_E()
            self.expect("RPAREN")
            return e_place

        raise SyntaxError(f"Unexpected token {tok.kind}({tok.lexeme!r}) at position {tok.position}")


def eval_operand(operand: str, table: Dict[str, int]) -> int:
    if operand in table:
        return table[operand]
    if operand.isdigit():
        return int(operand)
    raise KeyError(f"Unknown operand {operand!r}; missing symbol-table binding")


def execute_three_address_code(quads: Sequence[Quad], env: Dict[str, int], final_place: str) -> int:
    table: Dict[str, int] = {name: int(value) for name, value in env.items()}

    for quad in quads:
        left = eval_operand(quad.arg1, table)
        right = eval_operand(quad.arg2, table)
        if quad.op == "+":
            table[quad.result] = left + right
        elif quad.op == "*":
            table[quad.result] = left * right
        else:
            raise ValueError(f"Unsupported op: {quad.op}")

    return eval_operand(final_place, table)


def pretty_tokens(tokens: Sequence[Token]) -> str:
    return " ".join(token.lexeme for token in tokens if token.kind != "EOF")


def pretty_quads(quads: Sequence[Quad]) -> List[str]:
    if not quads:
        return ["(no temporary code emitted; single atom expression)"]
    return [f"{idx:02d}. {q.result} = {q.arg1} {q.op} {q.arg2}" for idx, q in enumerate(quads, start=1)]


def run_positive_cases() -> None:
    cases = [
        ("a + b * c", {"a": 2, "b": 3, "c": 4}),
        ("(a + b) * (c + 2)", {"a": 2, "b": 3, "c": 4}),
        ("x * y + z * 3 + 1", {"x": 5, "y": 6, "z": 7}),
        ("42", {}),
        ("m * (n + 8) * p", {"m": 2, "n": 1, "p": 4}),
    ]

    instruction_counts: List[int] = []

    for idx, (expr, env) in enumerate(cases, start=1):
        tokens = tokenize(expr)
        parser = Parser(tokens)
        translation = parser.translate(expr)

        actual = execute_three_address_code(translation.quads, env, translation.result_place)
        expected = int(eval(expr, {"__builtins__": {}}, env))
        assert actual == expected, f"Mismatch for {expr!r}: actual={actual}, expected={expected}"

        instruction_counts.append(len(translation.quads))

        print(f"\nCase {idx}: {expr}")
        print("  tokens:", pretty_tokens(tokens))
        print("  result place:", translation.result_place)
        print("  three-address code:")
        for line in pretty_quads(translation.quads):
            print("   ", line)
        print(f"  execution: {actual} (expected {expected})")

    avg_count = sum(instruction_counts) / len(instruction_counts)
    print("\nCode size stats:")
    print("  instruction counts:", instruction_counts)
    print(f"  average instructions: {avg_count:.2f}")


def run_negative_case() -> None:
    bad_expr = "a + * b"
    try:
        parser = Parser(tokenize(bad_expr))
        parser.translate(bad_expr)
    except SyntaxError as exc:
        print("\nNegative case:", bad_expr)
        print("  parser rejected as expected:", exc)
        return

    raise AssertionError("Negative case should have failed but succeeded")


def main() -> None:
    print("Syntax-Directed Translation MVP (expression -> three-address code)")
    run_positive_cases()
    run_negative_case()
    print("\nAll SDT checks passed.")


if __name__ == "__main__":
    main()
