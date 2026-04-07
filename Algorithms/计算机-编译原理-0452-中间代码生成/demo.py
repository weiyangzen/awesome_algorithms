"""Minimal runnable MVP for intermediate code generation (IR/TAC)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


# ===== AST model =====
@dataclass(frozen=True)
class Number:
    value: int


@dataclass(frozen=True)
class Variable:
    name: str


@dataclass(frozen=True)
class Binary:
    op: str
    left: "Expr"
    right: "Expr"


Expr = Union[Number, Variable, Binary]


@dataclass(frozen=True)
class Assign:
    target: str
    expr: Expr


@dataclass(frozen=True)
class IfElse:
    condition: Expr
    then_body: Tuple["Stmt", ...]
    else_body: Tuple["Stmt", ...]


@dataclass(frozen=True)
class While:
    condition: Expr
    body: Tuple["Stmt", ...]


@dataclass(frozen=True)
class Return:
    expr: Expr


Stmt = Union[Assign, IfElse, While, Return]


# ===== TAC model =====
@dataclass(frozen=True)
class TACInstruction:
    opcode: str
    dest: Optional[str] = None
    arg1: Optional[str] = None
    arg2: Optional[str] = None
    operator: Optional[str] = None
    label: Optional[str] = None
    true_label: Optional[str] = None
    false_label: Optional[str] = None
    target: Optional[str] = None


def format_instruction(inst: TACInstruction) -> str:
    if inst.opcode == "label":
        return f"{inst.label}:"
    if inst.opcode == "assign":
        return f"{inst.dest} = {inst.arg1}"
    if inst.opcode == "binop":
        return f"{inst.dest} = {inst.arg1} {inst.operator} {inst.arg2}"
    if inst.opcode == "cjump":
        return f"if {inst.arg1} != 0 goto {inst.true_label} else {inst.false_label}"
    if inst.opcode == "jump":
        return f"goto {inst.target}"
    if inst.opcode == "return":
        return f"return {inst.arg1}"
    raise ValueError(f"Unsupported opcode: {inst.opcode}")


class TACGenerator:
    """Translate a tiny AST into three-address code."""

    def __init__(self) -> None:
        self.temp_counter = 0
        self.label_counter = 0
        self.instructions: List[TACInstruction] = []

    def new_temp(self) -> str:
        self.temp_counter += 1
        return f"t{self.temp_counter}"

    def new_label(self, prefix: str = "L") -> str:
        self.label_counter += 1
        return f"{prefix}{self.label_counter}"

    def emit(self, inst: TACInstruction) -> None:
        self.instructions.append(inst)

    def gen_expr(self, expr: Expr) -> str:
        if isinstance(expr, Number):
            return str(expr.value)
        if isinstance(expr, Variable):
            return expr.name
        if isinstance(expr, Binary):
            left = self.gen_expr(expr.left)
            right = self.gen_expr(expr.right)
            dest = self.new_temp()
            self.emit(
                TACInstruction(
                    opcode="binop",
                    dest=dest,
                    arg1=left,
                    arg2=right,
                    operator=expr.op,
                )
            )
            return dest
        raise TypeError(f"Unsupported expression: {type(expr)}")

    def gen_stmt(self, stmt: Stmt) -> None:
        if isinstance(stmt, Assign):
            value_place = self.gen_expr(stmt.expr)
            self.emit(TACInstruction(opcode="assign", dest=stmt.target, arg1=value_place))
            return

        if isinstance(stmt, IfElse):
            cond_place = self.gen_expr(stmt.condition)
            label_true = self.new_label("IF_TRUE_")
            label_false = self.new_label("IF_FALSE_")
            label_end = self.new_label("IF_END_")

            self.emit(
                TACInstruction(
                    opcode="cjump",
                    arg1=cond_place,
                    true_label=label_true,
                    false_label=label_false,
                )
            )

            self.emit(TACInstruction(opcode="label", label=label_true))
            for s in stmt.then_body:
                self.gen_stmt(s)
            self.emit(TACInstruction(opcode="jump", target=label_end))

            self.emit(TACInstruction(opcode="label", label=label_false))
            for s in stmt.else_body:
                self.gen_stmt(s)

            self.emit(TACInstruction(opcode="label", label=label_end))
            return

        if isinstance(stmt, While):
            label_head = self.new_label("WHILE_HEAD_")
            label_body = self.new_label("WHILE_BODY_")
            label_end = self.new_label("WHILE_END_")

            self.emit(TACInstruction(opcode="label", label=label_head))
            cond_place = self.gen_expr(stmt.condition)
            self.emit(
                TACInstruction(
                    opcode="cjump",
                    arg1=cond_place,
                    true_label=label_body,
                    false_label=label_end,
                )
            )
            self.emit(TACInstruction(opcode="label", label=label_body))
            for s in stmt.body:
                self.gen_stmt(s)
            self.emit(TACInstruction(opcode="jump", target=label_head))
            self.emit(TACInstruction(opcode="label", label=label_end))
            return

        if isinstance(stmt, Return):
            place = self.gen_expr(stmt.expr)
            self.emit(TACInstruction(opcode="return", arg1=place))
            return

        raise TypeError(f"Unsupported statement: {type(stmt)}")

    def generate(self, program: Sequence[Stmt]) -> List[TACInstruction]:
        for stmt in program:
            self.gen_stmt(stmt)
        return self.instructions


class TACExecutor:
    """A tiny interpreter used to validate generated TAC behavior."""

    def __init__(self, instructions: Sequence[TACInstruction]) -> None:
        self.instructions = list(instructions)
        self.label_to_pc = self._build_label_map(self.instructions)

    @staticmethod
    def _build_label_map(instructions: Sequence[TACInstruction]) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for pc, inst in enumerate(instructions):
            if inst.opcode == "label":
                if inst.label is None:
                    raise ValueError(f"Label opcode without label at pc={pc}")
                if inst.label in mapping:
                    raise ValueError(f"Duplicate label: {inst.label}")
                mapping[inst.label] = pc
        return mapping

    @staticmethod
    def _resolve_operand(token: str, env: Dict[str, int]) -> int:
        if token.lstrip("-").isdigit():
            return int(token)
        if token not in env:
            raise ValueError(f"Undefined variable/temp: {token}")
        return env[token]

    @staticmethod
    def _apply_operator(op: str, left: int, right: int) -> int:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "//":
            if right == 0:
                raise ZeroDivisionError("division by zero in TAC execution")
            return left // right
        if op == "<":
            return 1 if left < right else 0
        if op == ">":
            return 1 if left > right else 0
        if op == "<=":
            return 1 if left <= right else 0
        if op == ">=":
            return 1 if left >= right else 0
        if op == "==":
            return 1 if left == right else 0
        if op == "!=":
            return 1 if left != right else 0
        raise ValueError(f"Unsupported binary operator: {op}")

    def run(self, max_steps: int = 10_000) -> Tuple[int, Dict[str, int]]:
        env: Dict[str, int] = {}
        pc = 0
        steps = 0

        while pc < len(self.instructions):
            if steps > max_steps:
                raise RuntimeError("Execution step limit exceeded (possible infinite loop)")
            steps += 1

            inst = self.instructions[pc]

            if inst.opcode == "label":
                pc += 1
                continue

            if inst.opcode == "assign":
                if inst.dest is None or inst.arg1 is None:
                    raise ValueError(f"Malformed assign at pc={pc}")
                env[inst.dest] = self._resolve_operand(inst.arg1, env)
                pc += 1
                continue

            if inst.opcode == "binop":
                if (
                    inst.dest is None
                    or inst.arg1 is None
                    or inst.arg2 is None
                    or inst.operator is None
                ):
                    raise ValueError(f"Malformed binop at pc={pc}")
                left = self._resolve_operand(inst.arg1, env)
                right = self._resolve_operand(inst.arg2, env)
                env[inst.dest] = self._apply_operator(inst.operator, left, right)
                pc += 1
                continue

            if inst.opcode == "cjump":
                if (
                    inst.arg1 is None
                    or inst.true_label is None
                    or inst.false_label is None
                ):
                    raise ValueError(f"Malformed cjump at pc={pc}")
                cond_value = self._resolve_operand(inst.arg1, env)
                next_label = inst.true_label if cond_value != 0 else inst.false_label
                pc = self.label_to_pc[next_label]
                continue

            if inst.opcode == "jump":
                if inst.target is None:
                    raise ValueError(f"Malformed jump at pc={pc}")
                pc = self.label_to_pc[inst.target]
                continue

            if inst.opcode == "return":
                if inst.arg1 is None:
                    raise ValueError(f"Malformed return at pc={pc}")
                return self._resolve_operand(inst.arg1, env), env

            raise ValueError(f"Unsupported opcode in execution: {inst.opcode}")

        raise RuntimeError("Program terminated without return")


def build_sample_program() -> List[Stmt]:
    """Construct a fixed AST program for deterministic verification."""
    return [
        Assign("a", Number(2)),
        Assign("b", Number(4)),
        IfElse(
            condition=Binary("<", Variable("a"), Variable("b")),
            then_body=(
                Assign(
                    "c",
                    Binary(
                        "*",
                        Binary("+", Variable("a"), Variable("b")),
                        Binary("-", Variable("b"), Number(1)),
                    ),
                ),
            ),
            else_body=(Assign("c", Binary("-", Variable("a"), Variable("b"))),),
        ),
        While(
            condition=Binary("<", Variable("c"), Number(20)),
            body=(Assign("c", Binary("+", Variable("c"), Number(3))),),
        ),
        Return(Variable("c")),
    ]


def summarize_opcodes(instructions: Sequence[TACInstruction]) -> str:
    counts: Dict[str, int] = {}
    for inst in instructions:
        counts[inst.opcode] = counts.get(inst.opcode, 0) + 1

    opcodes = sorted(counts.keys())

    if pd is not None:
        frame = pd.DataFrame(
            {
                "opcode": opcodes,
                "count": [counts[k] for k in opcodes],
            }
        )
        return frame.to_string(index=False)

    if np is not None:
        arr = np.array([counts[k] for k in opcodes], dtype=int)
        pairs = [f"{name}: {int(value)}" for name, value in zip(opcodes, arr)]
        return "\n".join(pairs)

    return "\n".join(f"{name}: {counts[name]}" for name in opcodes)


def assert_expected(
    instructions: Sequence[TACInstruction], return_value: int, env: Dict[str, int]
) -> None:
    opcode_counts: Dict[str, int] = {}
    labels: List[str] = []
    for inst in instructions:
        opcode_counts[inst.opcode] = opcode_counts.get(inst.opcode, 0) + 1
        if inst.opcode == "label":
            if inst.label is None:
                raise AssertionError("Label instruction misses label name")
            labels.append(inst.label)

    if len(labels) != len(set(labels)):
        raise AssertionError("Labels are not unique")

    required = {"assign", "binop", "cjump", "jump", "label", "return"}
    missing = sorted(required - set(opcode_counts.keys()))
    if missing:
        raise AssertionError(f"Missing opcode kinds in generated TAC: {missing}")

    if return_value != 21:
        raise AssertionError(f"Return value mismatch: got {return_value}, expected 21")

    if env.get("c") != 21:
        raise AssertionError(f"Variable c mismatch: got {env.get('c')}, expected 21")


def main() -> None:
    program = build_sample_program()
    generator = TACGenerator()
    instructions = generator.generate(program)

    print("=== Generated TAC ===")
    for idx, inst in enumerate(instructions):
        print(f"{idx:02d}: {format_instruction(inst)}")

    print("\n=== Opcode Summary ===")
    print(summarize_opcodes(instructions))

    executor = TACExecutor(instructions)
    return_value, env = executor.run()

    print("\n=== Execution Result ===")
    print(f"return = {return_value}")
    print(f"env(c) = {env.get('c')}")

    assert_expected(instructions, return_value, env)
    print("\nValidation: PASS")


if __name__ == "__main__":
    main()
