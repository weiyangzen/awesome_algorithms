"""Recursive-descent parsing MVP for CS-0285.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

TokenKind = Literal["NUMBER", "PLUS", "MINUS", "STAR", "SLASH", "LPAREN", "RPAREN", "EOF"]


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    value: str
    pos: int


@dataclass(frozen=True)
class Number:
    value: float


@dataclass(frozen=True)
class UnaryOp:
    op: str
    operand: "ASTNode"


@dataclass(frozen=True)
class BinaryOp:
    op: str
    left: "ASTNode"
    right: "ASTNode"


ASTNode = Number | UnaryOp | BinaryOp


def tokenize(text: str) -> list[Token]:
    """Convert source text into a flat token stream."""
    tokens: list[Token] = []
    i = 0

    single_char = {
        "+": "PLUS",
        "-": "MINUS",
        "*": "STAR",
        "/": "SLASH",
        "(": "LPAREN",
        ")": "RPAREN",
    }

    while i < len(text):
        ch = text[i]

        if ch.isspace():
            i += 1
            continue

        if ch.isdigit() or ch == ".":
            start = i
            dot_count = 1 if ch == "." else 0
            i += 1
            while i < len(text) and (text[i].isdigit() or text[i] == "."):
                if text[i] == ".":
                    dot_count += 1
                i += 1

            literal = text[start:i]
            if literal == "." or dot_count > 1:
                raise SyntaxError(f"Invalid numeric literal '{literal}' at position {start}")

            tokens.append(Token("NUMBER", literal, start))
            continue

        kind = single_char.get(ch)
        if kind is not None:
            tokens.append(Token(kind, ch, i))
            i += 1
            continue

        raise SyntaxError(f"Unexpected character '{ch}' at position {i}")

    tokens.append(Token("EOF", "", len(text)))
    return tokens


class Parser:
    """LL(1) recursive-descent parser for arithmetic expressions.

    Grammar (left recursion removed):
        Expr   -> Term ((PLUS | MINUS) Term)*
        Term   -> Factor ((STAR | SLASH) Factor)*
        Factor -> NUMBER | LPAREN Expr RPAREN | MINUS Factor
    """

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.index = 0
        self.trace: list[str] = []

    def current(self) -> Token:
        return self.tokens[self.index]

    def advance(self) -> Token:
        token = self.current()
        self.index += 1
        return token

    def expect(self, kind: TokenKind) -> Token:
        token = self.current()
        if token.kind != kind:
            raise SyntaxError(
                f"Expected {kind}, got {token.kind} ('{token.value}') at position {token.pos}"
            )
        return self.advance()

    def parse(self) -> ASTNode:
        self.trace.append("parse -> Expr")
        node = self.parse_expr()
        self.expect("EOF")
        self.trace.append("parse <- Expr EOF")
        return node

    def parse_expr(self) -> ASTNode:
        self.trace.append("Expr -> Term Expr'")
        node = self.parse_term()

        while self.current().kind in ("PLUS", "MINUS"):
            op = self.advance().value
            self.trace.append(f"Expr' -> {op} Term Expr'")
            rhs = self.parse_term()
            node = BinaryOp(op=op, left=node, right=rhs)

        self.trace.append("Expr' -> eps")
        return node

    def parse_term(self) -> ASTNode:
        self.trace.append("Term -> Factor Term'")
        node = self.parse_factor()

        while self.current().kind in ("STAR", "SLASH"):
            op = self.advance().value
            self.trace.append(f"Term' -> {op} Factor Term'")
            rhs = self.parse_factor()
            node = BinaryOp(op=op, left=node, right=rhs)

        self.trace.append("Term' -> eps")
        return node

    def parse_factor(self) -> ASTNode:
        token = self.current()

        if token.kind == "NUMBER":
            self.advance()
            self.trace.append(f"Factor -> NUMBER({token.value})")
            return Number(float(token.value))

        if token.kind == "LPAREN":
            self.advance()
            self.trace.append("Factor -> LPAREN Expr RPAREN")
            node = self.parse_expr()
            self.expect("RPAREN")
            return node

        if token.kind == "MINUS":
            self.advance()
            self.trace.append("Factor -> MINUS Factor")
            return UnaryOp(op="-", operand=self.parse_factor())

        raise SyntaxError(
            f"Expected NUMBER/LPAREN/MINUS, got {token.kind} ('{token.value}') at position {token.pos}"
        )


def evaluate(node: ASTNode) -> float:
    """Recursively evaluate AST."""
    if isinstance(node, Number):
        return node.value

    if isinstance(node, UnaryOp):
        if node.op == "-":
            return -evaluate(node.operand)
        raise ValueError(f"Unsupported unary operator: {node.op}")

    left_val = evaluate(node.left)
    right_val = evaluate(node.right)

    if node.op == "+":
        return left_val + right_val
    if node.op == "-":
        return left_val - right_val
    if node.op == "*":
        return left_val * right_val
    if node.op == "/":
        if right_val == 0:
            raise ZeroDivisionError("Division by zero")
        return left_val / right_val

    raise ValueError(f"Unsupported binary operator: {node.op}")


def ast_depth(node: ASTNode) -> int:
    if isinstance(node, Number):
        return 1
    if isinstance(node, UnaryOp):
        return 1 + ast_depth(node.operand)
    return 1 + max(ast_depth(node.left), ast_depth(node.right))


def ast_size(node: ASTNode) -> int:
    if isinstance(node, Number):
        return 1
    if isinstance(node, UnaryOp):
        return 1 + ast_size(node.operand)
    return 1 + ast_size(node.left) + ast_size(node.right)


def to_infix(node: ASTNode) -> str:
    if isinstance(node, Number):
        if node.value.is_integer():
            return str(int(node.value))
        return str(node.value)
    if isinstance(node, UnaryOp):
        return f"(-{to_infix(node.operand)})"
    return f"({to_infix(node.left)} {node.op} {to_infix(node.right)})"


def parse_expression(text: str) -> tuple[ASTNode, list[str], list[Token]]:
    tokens = tokenize(text)
    parser = Parser(tokens)
    ast = parser.parse()
    return ast, parser.trace, tokens


def run_demo() -> None:
    valid_cases: list[tuple[str, float]] = [
        ("1 + 2 * 3", 7.0),
        ("-(4 + 5) * (6 - 2)", -36.0),
        ("10 / (2 + 3) + 7 * 2", 16.0),
        ("3.5 * (2 - 0.5)", 5.25),
    ]

    invalid_cases = [
        "3 + * 4",
        "1 + (2 * 3",
        "7 / )2(",
    ]

    print("=== Recursive Descent Parser Demo ===")
    print("Grammar: Expr -> Term ((+|-) Term)*; Term -> Factor ((*|/) Factor)*; Factor -> NUMBER | (Expr) | -Factor")

    stats_rows: list[list[float]] = []

    for idx, (expr, expected) in enumerate(valid_cases, start=1):
        ast, trace, tokens = parse_expression(expr)
        value = evaluate(ast)

        if abs(value - expected) > 1e-9:
            raise AssertionError(f"Unexpected result for {expr}: expected={expected}, got={value}")

        token_count = len(tokens) - 1  # exclude EOF from user-visible count
        node_count = ast_size(ast)
        depth = ast_depth(ast)
        stats_rows.append([token_count, node_count, depth])

        print(f"\n[Valid #{idx}] {expr}")
        print(f"  Tokens: {token_count}, AST nodes: {node_count}, AST depth: {depth}")
        print(f"  AST(infix): {to_infix(ast)}")
        print(f"  Value: {value}")
        print("  Trace (first 8 steps):")
        for step in trace[:8]:
            print(f"    - {step}")

    print("\n=== Error Handling Cases ===")
    for expr in invalid_cases:
        try:
            parse_expression(expr)
            raise AssertionError(f"Expected parse failure but succeeded: {expr}")
        except SyntaxError as exc:
            print(f"  {expr!r} -> SyntaxError: {exc}")

    stats = np.array(stats_rows, dtype=float)
    mean_token, mean_nodes, mean_depth = stats.mean(axis=0)
    max_depth = int(stats[:, 2].max())

    print("\n=== Aggregate Stats (numpy) ===")
    print(f"  Mean token count: {mean_token:.2f}")
    print(f"  Mean AST node count: {mean_nodes:.2f}")
    print(f"  Mean AST depth: {mean_depth:.2f}")
    print(f"  Max AST depth: {max_depth}")

    print("\nAll demo cases passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
