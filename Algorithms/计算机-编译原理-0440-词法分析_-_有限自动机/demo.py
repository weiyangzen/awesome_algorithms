"""词法分析（有限自动机）最小可运行示例。

MVP 目标：
1. 用显式 DFA（状态 + 字符类别 + 转移表）实现一个小型词法分析器；
2. 支持常见 token：标识符/关键字/整数字面量/运算符/分隔符；
3. 不依赖交互输入，运行脚本即可看到分词结果与自检样例。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# ------------------------------
# 字符类别定义（列）
# ------------------------------
C_LETTER = 0
C_DIGIT = 1
C_UNDERSCORE = 2
C_WS = 3
C_EQ = 4
C_EXCL = 5
C_LT = 6
C_GT = 7
C_PLUS = 8
C_MINUS = 9
C_STAR = 10
C_SLASH = 11
C_LPAREN = 12
C_RPAREN = 13
C_LBRACE = 14
C_RBRACE = 15
C_SEMI = 16
C_COMMA = 17
C_OTHER = 18

N_CLASSES = 19

# ------------------------------
# 状态定义（行）
# ------------------------------
S_START = 0
S_IDENT = 1
S_INT = 2
S_EQ = 3
S_EQEQ = 4
S_BANG = 5
S_NEQ = 6
S_LT = 7
S_LE = 8
S_GT = 9
S_GE = 10
S_PLUS = 11
S_MINUS = 12
S_STAR = 13
S_SLASH = 14
S_LPAREN = 15
S_RPAREN = 16
S_LBRACE = 17
S_RBRACE = 18
S_SEMI = 19
S_COMMA = 20
S_WS = 21

N_STATES = 22

# ------------------------------
# Token 类型定义
# ------------------------------
TK_IDENT = 0
TK_INT = 1
TK_ASSIGN = 2
TK_EQ = 3
TK_BANG = 4
TK_NEQ = 5
TK_LT = 6
TK_LE = 7
TK_GT = 8
TK_GE = 9
TK_PLUS = 10
TK_MINUS = 11
TK_STAR = 12
TK_SLASH = 13
TK_LPAREN = 14
TK_RPAREN = 15
TK_LBRACE = 16
TK_RBRACE = 17
TK_SEMI = 18
TK_COMMA = 19
TK_WS = 20

TOKEN_NAMES: Dict[int, str] = {
    TK_IDENT: "IDENT",
    TK_INT: "INT",
    TK_ASSIGN: "ASSIGN",
    TK_EQ: "EQ",
    TK_BANG: "BANG",
    TK_NEQ: "NEQ",
    TK_LT: "LT",
    TK_LE: "LE",
    TK_GT: "GT",
    TK_GE: "GE",
    TK_PLUS: "PLUS",
    TK_MINUS: "MINUS",
    TK_STAR: "STAR",
    TK_SLASH: "SLASH",
    TK_LPAREN: "LPAREN",
    TK_RPAREN: "RPAREN",
    TK_LBRACE: "LBRACE",
    TK_RBRACE: "RBRACE",
    TK_SEMI: "SEMI",
    TK_COMMA: "COMMA",
    TK_WS: "WS",
}

KEYWORDS = {"if", "else", "while", "return", "int", "float", "for"}


@dataclass(frozen=True)
class Token:
    token_type: str
    lexeme: str
    start: int
    end: int


def classify_char(ch: str) -> int:
    if ch.isalpha():
        return C_LETTER
    if ch.isdigit():
        return C_DIGIT
    if ch == "_":
        return C_UNDERSCORE
    if ch.isspace():
        return C_WS
    if ch == "=":
        return C_EQ
    if ch == "!":
        return C_EXCL
    if ch == "<":
        return C_LT
    if ch == ">":
        return C_GT
    if ch == "+":
        return C_PLUS
    if ch == "-":
        return C_MINUS
    if ch == "*":
        return C_STAR
    if ch == "/":
        return C_SLASH
    if ch == "(":
        return C_LPAREN
    if ch == ")":
        return C_RPAREN
    if ch == "{":
        return C_LBRACE
    if ch == "}":
        return C_RBRACE
    if ch == ";":
        return C_SEMI
    if ch == ",":
        return C_COMMA
    return C_OTHER


def build_dfa_tables() -> tuple[np.ndarray, np.ndarray]:
    """构建 DFA 转移表与接受态映射表。"""
    transitions = np.full((N_STATES, N_CLASSES), -1, dtype=np.int16)

    # Start state transitions.
    transitions[S_START, C_LETTER] = S_IDENT
    transitions[S_START, C_UNDERSCORE] = S_IDENT
    transitions[S_START, C_DIGIT] = S_INT
    transitions[S_START, C_WS] = S_WS
    transitions[S_START, C_EQ] = S_EQ
    transitions[S_START, C_EXCL] = S_BANG
    transitions[S_START, C_LT] = S_LT
    transitions[S_START, C_GT] = S_GT
    transitions[S_START, C_PLUS] = S_PLUS
    transitions[S_START, C_MINUS] = S_MINUS
    transitions[S_START, C_STAR] = S_STAR
    transitions[S_START, C_SLASH] = S_SLASH
    transitions[S_START, C_LPAREN] = S_LPAREN
    transitions[S_START, C_RPAREN] = S_RPAREN
    transitions[S_START, C_LBRACE] = S_LBRACE
    transitions[S_START, C_RBRACE] = S_RBRACE
    transitions[S_START, C_SEMI] = S_SEMI
    transitions[S_START, C_COMMA] = S_COMMA

    # IDENT: [A-Za-z_][A-Za-z0-9_]*
    transitions[S_IDENT, C_LETTER] = S_IDENT
    transitions[S_IDENT, C_DIGIT] = S_IDENT
    transitions[S_IDENT, C_UNDERSCORE] = S_IDENT

    # INT: [0-9]+
    transitions[S_INT, C_DIGIT] = S_INT

    # Multi-char operator refinement.
    transitions[S_EQ, C_EQ] = S_EQEQ
    transitions[S_BANG, C_EQ] = S_NEQ
    transitions[S_LT, C_EQ] = S_LE
    transitions[S_GT, C_EQ] = S_GE

    # Whitespace loop.
    transitions[S_WS, C_WS] = S_WS

    accept_token = np.full(N_STATES, -1, dtype=np.int16)
    accept_token[S_IDENT] = TK_IDENT
    accept_token[S_INT] = TK_INT
    accept_token[S_EQ] = TK_ASSIGN
    accept_token[S_EQEQ] = TK_EQ
    accept_token[S_BANG] = TK_BANG
    accept_token[S_NEQ] = TK_NEQ
    accept_token[S_LT] = TK_LT
    accept_token[S_LE] = TK_LE
    accept_token[S_GT] = TK_GT
    accept_token[S_GE] = TK_GE
    accept_token[S_PLUS] = TK_PLUS
    accept_token[S_MINUS] = TK_MINUS
    accept_token[S_STAR] = TK_STAR
    accept_token[S_SLASH] = TK_SLASH
    accept_token[S_LPAREN] = TK_LPAREN
    accept_token[S_RPAREN] = TK_RPAREN
    accept_token[S_LBRACE] = TK_LBRACE
    accept_token[S_RBRACE] = TK_RBRACE
    accept_token[S_SEMI] = TK_SEMI
    accept_token[S_COMMA] = TK_COMMA
    accept_token[S_WS] = TK_WS

    return transitions, accept_token


def normalize_token_type(base_type: str, lexeme: str) -> str:
    if base_type == "IDENT" and lexeme in KEYWORDS:
        return f"KW_{lexeme.upper()}"
    return base_type


def tokenize(text: str, transitions: np.ndarray, accept_token: np.ndarray) -> List[Token]:
    """用 DFA 做 maximal munch 分词。"""
    tokens: List[Token] = []
    pos = 0
    n = len(text)

    while pos < n:
        state = S_START
        i = pos
        last_accept_state = -1
        last_accept_pos = -1

        while i < n:
            char_class = classify_char(text[i])
            next_state = int(transitions[state, char_class])
            if next_state < 0:
                break
            state = next_state
            i += 1
            if int(accept_token[state]) >= 0:
                last_accept_state = state
                last_accept_pos = i

        if last_accept_state < 0:
            snippet = text[pos : min(pos + 12, n)].replace("\n", "\\n")
            raise ValueError(
                f"Lexical error at index {pos}: unexpected char {text[pos]!r}, "
                f"context={snippet!r}"
            )

        lexeme = text[pos:last_accept_pos]
        token_id = int(accept_token[last_accept_state])
        token_type = normalize_token_type(TOKEN_NAMES[token_id], lexeme)

        if token_type != "WS":
            tokens.append(Token(token_type=token_type, lexeme=lexeme, start=pos, end=last_accept_pos))

        pos = last_accept_pos

    tokens.append(Token(token_type="EOF", lexeme="", start=n, end=n))
    return tokens


def run_demo_cases(transitions: np.ndarray, accept_token: np.ndarray) -> None:
    cases = [
        (
            "int x=10;",
            ["KW_INT", "IDENT", "ASSIGN", "INT", "SEMI", "EOF"],
        ),
        (
            "if(a!=0)return a+1;",
            [
                "KW_IF",
                "LPAREN",
                "IDENT",
                "NEQ",
                "INT",
                "RPAREN",
                "KW_RETURN",
                "IDENT",
                "PLUS",
                "INT",
                "SEMI",
                "EOF",
            ],
        ),
        (
            "while (count<=10) count = count + 1;",
            [
                "KW_WHILE",
                "LPAREN",
                "IDENT",
                "LE",
                "INT",
                "RPAREN",
                "IDENT",
                "ASSIGN",
                "IDENT",
                "PLUS",
                "INT",
                "SEMI",
                "EOF",
            ],
        ),
    ]

    all_passed = True
    for idx, (source, expected_types) in enumerate(cases, start=1):
        got_tokens = tokenize(source, transitions, accept_token)
        got_types = [t.token_type for t in got_tokens]
        ok = got_types == expected_types
        all_passed = all_passed and ok
        print(f"[CASE {idx}] pass={ok}")
        print(f"  source: {source!r}")
        print(f"  got   : {got_types}")
        print(f"  expect: {expected_types}")

    if not all_passed:
        raise AssertionError("One or more lexer demo cases failed.")


def pretty_print_tokens(tokens: List[Token]) -> None:
    print("idx  token_type      lexeme         span")
    for i, tok in enumerate(tokens, start=1):
        print(f"{i:>3}  {tok.token_type:<14} {tok.lexeme!r:<13} [{tok.start}, {tok.end})")


def main() -> None:
    transitions, accept_token = build_dfa_tables()

    print("=== Lexer DFA MVP ===")
    print(f"states={N_STATES}, classes={N_CLASSES}, accepting_states={int(np.sum(accept_token >= 0))}")

    source = (
        "int sum = a1 + 24;\n"
        "if (sum >= 30) {\n"
        "  return sum - 1;\n"
        "}\n"
    )

    print("\n[Tokenization sample]")
    print(source)
    tokens = tokenize(source, transitions, accept_token)
    pretty_print_tokens(tokens)

    print("\n[Deterministic test cases]")
    run_demo_cases(transitions, accept_token)
    print("\nAll lexer DFA demo cases passed.")


if __name__ == "__main__":
    main()
