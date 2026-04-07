"""Constant propagation MVP for CS-0295.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal

import numpy as np

LatticeKind = Literal["UNDEF", "CONST", "NAC"]
Operand = int | str
Instruction = tuple[str, ...]


@dataclass(frozen=True)
class LatticeValue:
    """Three-point lattice value for constant propagation.

    - UNDEF: no information yet (bottom)
    - CONST(v): known compile-time constant
    - NAC: not-a-constant / conflicting information (top)
    """

    kind: LatticeKind
    value: int | None = None

    @staticmethod
    def undef() -> "LatticeValue":
        return LatticeValue("UNDEF", None)

    @staticmethod
    def nac() -> "LatticeValue":
        return LatticeValue("NAC", None)

    @staticmethod
    def const(v: int) -> "LatticeValue":
        return LatticeValue("CONST", int(v))

    def __str__(self) -> str:
        if self.kind == "CONST":
            return str(self.value)
        return self.kind


UNDEF = LatticeValue.undef()
NAC = LatticeValue.nac()


@dataclass(frozen=True)
class BasicBlock:
    """A minimal basic block for the MVP CFG."""

    name: str
    instructions: list[Instruction]
    successors: list[str]


def meet_value(a: LatticeValue, b: LatticeValue) -> LatticeValue:
    """Lattice meet for classic dense constant propagation."""
    if a.kind == "UNDEF":
        return b
    if b.kind == "UNDEF":
        return a
    if a.kind == "CONST" and b.kind == "CONST" and a.value == b.value:
        return a
    return NAC


def get_value(state: dict[str, LatticeValue], name: str) -> LatticeValue:
    return state.get(name, UNDEF)


def set_value(state: dict[str, LatticeValue], name: str, value: LatticeValue) -> None:
    """Keep state sparse: UNDEF is represented by missing key."""
    if value.kind == "UNDEF":
        state.pop(name, None)
    else:
        state[name] = value


def eval_operand(state: dict[str, LatticeValue], operand: Operand) -> LatticeValue:
    if isinstance(operand, int):
        return LatticeValue.const(operand)
    return get_value(state, operand)


def eval_binop(op: str, lhs: LatticeValue, rhs: LatticeValue) -> LatticeValue:
    if lhs.kind == "NAC" or rhs.kind == "NAC":
        return NAC
    if lhs.kind == "UNDEF" or rhs.kind == "UNDEF":
        return UNDEF

    assert lhs.value is not None
    assert rhs.value is not None

    if op == "+":
        return LatticeValue.const(lhs.value + rhs.value)
    if op == "-":
        return LatticeValue.const(lhs.value - rhs.value)
    if op == "*":
        return LatticeValue.const(lhs.value * rhs.value)
    if op == "//":
        if rhs.value == 0:
            return NAC
        return LatticeValue.const(lhs.value // rhs.value)

    raise ValueError(f"Unsupported binary operator: {op}")


def transfer_instruction(state: dict[str, LatticeValue], inst: Instruction) -> None:
    """Apply a single instruction transfer on the abstract state."""
    opcode = inst[0]

    if opcode == "const":
        # ("const", target, literal)
        target = inst[1]
        literal = int(inst[2])
        set_value(state, target, LatticeValue.const(literal))
        return

    if opcode == "copy":
        # ("copy", target, source_var)
        target = inst[1]
        source = inst[2]
        set_value(state, target, get_value(state, source))
        return

    if opcode == "binop":
        # ("binop", target, op, left_operand, right_operand)
        target = inst[1]
        op = inst[2]
        left_operand: Operand = parse_operand(inst[3])
        right_operand: Operand = parse_operand(inst[4])
        left_value = eval_operand(state, left_operand)
        right_value = eval_operand(state, right_operand)
        set_value(state, target, eval_binop(op, left_value, right_value))
        return

    raise ValueError(f"Unsupported opcode: {opcode}")


def transfer_block(block: BasicBlock, in_state: dict[str, LatticeValue]) -> dict[str, LatticeValue]:
    out_state = dict(in_state)
    for inst in block.instructions:
        transfer_instruction(out_state, inst)
    return out_state


def parse_operand(token: str) -> Operand:
    """Parse a token into int literal or variable name."""
    try:
        return int(token)
    except ValueError:
        return token


def build_predecessors(blocks: dict[str, BasicBlock]) -> dict[str, list[str]]:
    preds: dict[str, list[str]] = {name: [] for name in blocks}
    for src, block in blocks.items():
        for dst in block.successors:
            if dst not in blocks:
                raise ValueError(f"Unknown successor block: {src} -> {dst}")
            preds[dst].append(src)
    return preds


def meet_states(states: list[dict[str, LatticeValue]]) -> dict[str, LatticeValue]:
    if not states:
        return {}

    merged: dict[str, LatticeValue] = {}
    all_vars = sorted({var for state in states for var in state.keys()})

    for var in all_vars:
        acc = UNDEF
        for state in states:
            acc = meet_value(acc, get_value(state, var))
        set_value(merged, var, acc)

    return merged


def constant_propagation(
    blocks: dict[str, BasicBlock], entry: str
) -> tuple[dict[str, dict[str, LatticeValue]], dict[str, dict[str, LatticeValue]]]:
    """Run forward worklist dataflow for constant propagation."""
    preds = build_predecessors(blocks)
    in_states: dict[str, dict[str, LatticeValue]] = {name: {} for name in blocks}
    out_states: dict[str, dict[str, LatticeValue]] = {name: {} for name in blocks}

    worklist: Deque[str] = deque([entry])
    scheduled = {entry}

    # Process any disconnected blocks as well (deterministic full pass).
    for name in blocks:
        if name not in scheduled:
            worklist.append(name)
            scheduled.add(name)

    while worklist:
        block_name = worklist.popleft()
        scheduled.discard(block_name)

        pred_outs = [out_states[p] for p in preds[block_name]]
        if block_name == entry and not preds[block_name]:
            new_in = {}
        else:
            new_in = meet_states(pred_outs)

        new_out = transfer_block(blocks[block_name], new_in)

        if new_in != in_states[block_name] or new_out != out_states[block_name]:
            in_states[block_name] = new_in
            out_states[block_name] = new_out
            for succ in blocks[block_name].successors:
                if succ not in scheduled:
                    worklist.append(succ)
                    scheduled.add(succ)

    return in_states, out_states


def stringify_state(state: dict[str, LatticeValue], variables: list[str]) -> str:
    parts = [f"{var}={get_value(state, var)}" for var in variables]
    return "{" + ", ".join(parts) + "}"


def render_instruction(inst: Instruction, state: dict[str, LatticeValue]) -> str:
    """Render an instruction with currently-known constants substituted."""

    def show_operand(token: str) -> str:
        op = parse_operand(token)
        if isinstance(op, int):
            return str(op)
        value = get_value(state, op)
        if value.kind == "CONST" and value.value is not None:
            return str(value.value)
        return op

    opcode = inst[0]
    if opcode == "const":
        return f"{inst[1]} = {inst[2]}"
    if opcode == "copy":
        return f"{inst[1]} = {show_operand(inst[2])}"
    if opcode == "binop":
        return f"{inst[1]} = {show_operand(inst[3])} {inst[2]} {show_operand(inst[4])}"
    raise ValueError(f"Unsupported opcode while rendering: {opcode}")


def print_analysis(
    blocks: dict[str, BasicBlock],
    in_states: dict[str, dict[str, LatticeValue]],
    out_states: dict[str, dict[str, LatticeValue]],
) -> None:
    variables = sorted({var for state in out_states.values() for var in state.keys()})

    print("=== Constant Propagation States ===")
    for name, block in blocks.items():
        print(f"\n[{name}]")
        print(f"IN : {stringify_state(in_states[name], variables)}")

        cursor = dict(in_states[name])
        for inst in block.instructions:
            before = render_instruction(inst, cursor)
            transfer_instruction(cursor, inst)
            print(f"  {before}")

        print(f"OUT: {stringify_state(out_states[name], variables)}")


def summarize_state_counts(
    blocks: dict[str, BasicBlock],
    out_states: dict[str, dict[str, LatticeValue]],
    variables: list[str],
) -> None:
    """Use numpy only for compact MVP statistics output."""
    print("\n=== Out-State Summary (CONST / NAC / UNDEF) ===")
    for name in blocks:
        states = [get_value(out_states[name], var).kind for var in variables]
        const_count = sum(kind == "CONST" for kind in states)
        nac_count = sum(kind == "NAC" for kind in states)
        undef_count = sum(kind == "UNDEF" for kind in states)
        counts = np.array([const_count, nac_count, undef_count], dtype=int)
        print(f"{name:>8}: {counts.tolist()}")


def run_demo() -> None:
    # A hand-written CFG to demonstrate:
    # 1) stable constants through a diamond (then/else both produce p=24)
    # 2) NAC at merge when branch assignments disagree (mix=1 vs mix=2)
    blocks = {
        "entry": BasicBlock(
            name="entry",
            instructions=[
                ("const", "a", "4"),
                ("const", "b", "8"),
                ("binop", "t", "+", "a", "b"),
            ],
            successors=["then", "else"],
        ),
        "then": BasicBlock(
            name="then",
            instructions=[
                ("copy", "x", "t"),
                ("binop", "p", "*", "x", "2"),
            ],
            successors=["join1"],
        ),
        "else": BasicBlock(
            name="else",
            instructions=[
                ("copy", "x", "t"),
                ("const", "p", "24"),
            ],
            successors=["join1"],
        ),
        "join1": BasicBlock(
            name="join1",
            instructions=[
                ("binop", "q", "-", "p", "a"),
                ("const", "r", "1"),
                ("binop", "s", "+", "q", "r"),
            ],
            successors=["case_a", "case_b"],
        ),
        "case_a": BasicBlock(
            name="case_a",
            instructions=[
                ("const", "mix", "1"),
            ],
            successors=["merge2"],
        ),
        "case_b": BasicBlock(
            name="case_b",
            instructions=[
                ("const", "mix", "2"),
            ],
            successors=["merge2"],
        ),
        "merge2": BasicBlock(
            name="merge2",
            instructions=[
                ("binop", "n", "+", "mix", "3"),
                ("binop", "final", "+", "s", "n"),
            ],
            successors=[],
        ),
    }

    in_states, out_states = constant_propagation(blocks, entry="entry")
    print_analysis(blocks, in_states, out_states)

    # Deterministic correctness checks for this CFG.
    assert get_value(out_states["entry"], "t") == LatticeValue.const(12)
    assert get_value(out_states["join1"], "s") == LatticeValue.const(21)
    assert get_value(in_states["merge2"], "mix") == NAC
    assert get_value(out_states["merge2"], "n") == NAC
    assert get_value(out_states["merge2"], "final") == NAC

    tracked_vars = sorted({v for state in out_states.values() for v in state.keys()} | {"mix", "n"})
    summarize_state_counts(blocks, out_states, tracked_vars)

    print("\nAll checks passed for CS-0295 (常量传播).")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
