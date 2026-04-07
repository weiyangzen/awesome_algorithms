"""死代码消除（Dead Code Elimination, DCE）最小可运行示例。

实现目标：
1. 用基本块 CFG 表示简化三地址代码；
2. 通过活跃变量数据流分析识别无用赋值；
3. 在保留副作用指令的前提下删除死代码；
4. 验证优化前后可观察语义一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Instruction:
    """简化 IR 指令。"""

    op: str
    dst: Optional[str] = None
    args: Tuple[str, ...] = ()
    target: Optional[str] = None
    true_target: Optional[str] = None
    false_target: Optional[str] = None

    def uses(self) -> set[str]:
        if self.op in {"const", "jump"}:
            tokens: Iterable[str] = ()
        elif self.op == "call":
            # args[0] 视为函数名，其余是实参。
            tokens = self.args[1:]
        else:
            tokens = self.args
        return {token for token in tokens if is_variable(token)}

    def defines(self) -> set[str]:
        if self.dst is None:
            return set()
        if self.op in {"print", "jump", "cjump", "return"}:
            return set()
        return {self.dst}

    def has_side_effect(self) -> bool:
        return self.op in {"print", "return", "call"}

    def to_ir(self) -> str:
        if self.op == "const":
            return f"{self.dst} = {self.args[0]}"
        if self.op == "copy":
            return f"{self.dst} = {self.args[0]}"
        if self.op in {"add", "sub", "mul", "div"}:
            symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[self.op]
            return f"{self.dst} = {self.args[0]} {symbol} {self.args[1]}"
        if self.op == "call":
            fn = self.args[0]
            params = ", ".join(self.args[1:])
            return f"{self.dst} = call {fn}({params})"
        if self.op == "print":
            return f"print {self.args[0]}"
        if self.op == "cjump":
            return f"if {self.args[0]} goto {self.true_target} else {self.false_target}"
        if self.op == "jump":
            return f"goto {self.target}"
        if self.op == "return":
            return f"return {self.args[0]}"
        raise ValueError(f"Unsupported op: {self.op}")


@dataclass
class BasicBlock:
    name: str
    instructions: List[Instruction]
    successors: List[str]


@dataclass
class DceReport:
    passes: int
    removed_count: int
    removed_by_pass: List[List[str]]


@dataclass
class RunResult:
    return_value: float
    printed_values: List[float]
    trace_log: List[str]


def is_variable(token: str) -> bool:
    if not token:
        return False
    return token[0].isalpha() or token[0] == "_"


def parse_number(token: str) -> float:
    try:
        return float(token)
    except ValueError as exc:
        raise ValueError(f"Token '{token}' is not a numeric literal") from exc


def value_of(token: str, env: Dict[str, float]) -> float:
    if is_variable(token):
        if token not in env:
            raise KeyError(f"Variable '{token}' is used before definition")
        return env[token]
    return parse_number(token)


def build_sample_cfg() -> Dict[str, BasicBlock]:
    """构造一个含分支与副作用的示例 CFG。"""
    return {
        "entry": BasicBlock(
            name="entry",
            instructions=[
                Instruction(op="add", dst="t0", args=("a", "b")),
                Instruction(op="mul", dst="t1", args=("t0", "2")),
                Instruction(op="sub", dst="dead_entry", args=("a", "b")),
                Instruction(op="cjump", args=("flag",), true_target="then", false_target="else"),
            ],
            successors=["then", "else"],
        ),
        "then": BasicBlock(
            name="then",
            instructions=[
                Instruction(op="add", dst="u", args=("t1", "1")),
                Instruction(op="mul", dst="dead_then", args=("u", "99")),
                Instruction(op="jump", target="join"),
            ],
            successors=["join"],
        ),
        "else": BasicBlock(
            name="else",
            instructions=[
                Instruction(op="sub", dst="u", args=("t1", "1")),
                Instruction(op="div", dst="dead_else", args=("u", "3")),
                Instruction(op="jump", target="join"),
            ],
            successors=["join"],
        ),
        "join": BasicBlock(
            name="join",
            instructions=[
                Instruction(op="add", dst="v", args=("u", "0")),
                Instruction(op="mul", dst="tmp", args=("v", "2")),
                Instruction(op="call", dst="dbg", args=("trace", "v")),
                Instruction(op="print", args=("v",)),
                Instruction(op="copy", dst="ret", args=("v",)),
                Instruction(op="return", args=("ret",)),
                Instruction(op="add", dst="dead_after_return", args=("ret", "42")),
            ],
            successors=[],
        ),
    }


def clone_cfg(cfg: Dict[str, BasicBlock]) -> Dict[str, BasicBlock]:
    new_cfg: Dict[str, BasicBlock] = {}
    for block_name, block in cfg.items():
        new_cfg[block_name] = BasicBlock(
            name=block.name,
            instructions=list(block.instructions),
            successors=list(block.successors),
        )
    return new_cfg


def compute_use_def(block: BasicBlock) -> tuple[set[str], set[str]]:
    use: set[str] = set()
    defined: set[str] = set()
    for inst in block.instructions:
        for var in inst.uses():
            if var not in defined:
                use.add(var)
        defined |= inst.defines()
    return use, defined


def liveness_analysis(cfg: Dict[str, BasicBlock]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    block_order = list(cfg.keys())
    use_map: dict[str, set[str]] = {}
    def_map: dict[str, set[str]] = {}

    for block_name in block_order:
        use_map[block_name], def_map[block_name] = compute_use_def(cfg[block_name])

    live_in = {name: set() for name in block_order}
    live_out = {name: set() for name in block_order}

    changed = True
    while changed:
        changed = False
        for block_name in reversed(block_order):
            block = cfg[block_name]
            out_set: set[str] = set()
            for succ in block.successors:
                out_set |= live_in[succ]

            in_set = use_map[block_name] | (out_set - def_map[block_name])

            if out_set != live_out[block_name] or in_set != live_in[block_name]:
                live_out[block_name] = out_set
                live_in[block_name] = in_set
                changed = True

    return live_in, live_out


def dce_one_pass(cfg: Dict[str, BasicBlock]) -> tuple[int, List[str]]:
    _, live_out = liveness_analysis(cfg)

    removed_count = 0
    removed_ir: List[str] = []

    for block_name in cfg.keys():
        block = cfg[block_name]
        live = set(live_out[block_name])
        kept_rev: List[Instruction] = []

        for inst in reversed(block.instructions):
            defs = inst.defines()
            if defs and not inst.has_side_effect():
                defined_var = next(iter(defs))
                if defined_var not in live:
                    removed_count += 1
                    removed_ir.append(f"{block_name}: {inst.to_ir()}")
                    continue

            live = (live - defs) | inst.uses()
            kept_rev.append(inst)

        block.instructions = list(reversed(kept_rev))

    return removed_count, removed_ir


def eliminate_dead_code(cfg: Dict[str, BasicBlock], max_passes: int = 10) -> DceReport:
    if max_passes <= 0:
        raise ValueError("max_passes must be positive")

    removed_total = 0
    removed_by_pass: List[List[str]] = []

    for pass_id in range(1, max_passes + 1):
        removed_count, removed_ir = dce_one_pass(cfg)
        if removed_count == 0:
            return DceReport(
                passes=pass_id,
                removed_count=removed_total,
                removed_by_pass=removed_by_pass,
            )
        removed_total += removed_count
        removed_by_pass.append(removed_ir)

    return DceReport(passes=max_passes, removed_count=removed_total, removed_by_pass=removed_by_pass)


def execute_program(cfg: Dict[str, BasicBlock], inputs: Dict[str, float]) -> RunResult:
    env = dict(inputs)
    printed_values: List[float] = []
    trace_log: List[str] = []

    current_block = "entry"
    pc = 0
    steps = 0
    max_steps = 10_000

    while True:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("Execution exceeded max_steps; possible infinite loop")

        block = cfg[current_block]
        if pc >= len(block.instructions):
            raise RuntimeError(f"Block '{current_block}' reached end without terminator")

        inst = block.instructions[pc]

        if inst.op == "copy":
            env[inst.dst] = value_of(inst.args[0], env)
            pc += 1
            continue

        if inst.op == "const":
            env[inst.dst] = parse_number(inst.args[0])
            pc += 1
            continue

        if inst.op in {"add", "sub", "mul", "div"}:
            lhs = value_of(inst.args[0], env)
            rhs = value_of(inst.args[1], env)
            if inst.op == "add":
                env[inst.dst] = lhs + rhs
            elif inst.op == "sub":
                env[inst.dst] = lhs - rhs
            elif inst.op == "mul":
                env[inst.dst] = lhs * rhs
            else:
                env[inst.dst] = lhs / rhs
            pc += 1
            continue

        if inst.op == "call":
            func_name = inst.args[0]
            arg_vals = [value_of(arg, env) for arg in inst.args[1:]]
            if func_name == "trace":
                trace_log.append(f"trace({', '.join(f'{v:.2f}' for v in arg_vals)})")
                env[inst.dst] = arg_vals[0] if arg_vals else 0.0
            else:
                raise ValueError(f"Unsupported function call: {func_name}")
            pc += 1
            continue

        if inst.op == "print":
            printed_values.append(value_of(inst.args[0], env))
            pc += 1
            continue

        if inst.op == "cjump":
            cond_value = value_of(inst.args[0], env)
            current_block = inst.true_target if cond_value != 0 else inst.false_target
            pc = 0
            continue

        if inst.op == "jump":
            current_block = inst.target
            pc = 0
            continue

        if inst.op == "return":
            return RunResult(
                return_value=value_of(inst.args[0], env),
                printed_values=printed_values,
                trace_log=trace_log,
            )

        raise ValueError(f"Unsupported op during execution: {inst.op}")


def count_instructions(cfg: Dict[str, BasicBlock]) -> int:
    return sum(len(block.instructions) for block in cfg.values())


def format_cfg(cfg: Dict[str, BasicBlock]) -> str:
    lines: List[str] = []
    for block in cfg.values():
        lines.append(f"[{block.name}]")
        for idx, inst in enumerate(block.instructions, start=1):
            lines.append(f"  {idx:02d}. {inst.to_ir()}")
        succ = ", ".join(block.successors) if block.successors else "<exit>"
        lines.append(f"  successors: {succ}")
    return "\n".join(lines)


def ensure_no_pure_dead_assignments(cfg: Dict[str, BasicBlock]) -> None:
    _, live_out = liveness_analysis(cfg)
    for block_name, block in cfg.items():
        live = set(live_out[block_name])
        for inst in reversed(block.instructions):
            defs = inst.defines()
            if defs and not inst.has_side_effect():
                defined_var = next(iter(defs))
                assert defined_var in live, (
                    f"Found remaining dead assignment in block '{block_name}': {inst.to_ir()}"
                )
            live = (live - defs) | inst.uses()


def run_equivalence_tests(
    original_cfg: Dict[str, BasicBlock],
    optimized_cfg: Dict[str, BasicBlock],
) -> None:
    test_inputs = [
        {"a": 4.0, "b": 1.0, "flag": 1.0},
        {"a": 4.0, "b": 1.0, "flag": 0.0},
        {"a": -3.0, "b": 2.0, "flag": 1.0},
        {"a": -3.0, "b": 2.0, "flag": 0.0},
    ]

    for idx, inp in enumerate(test_inputs, start=1):
        before = execute_program(original_cfg, inp)
        after = execute_program(optimized_cfg, inp)

        assert np.isclose(before.return_value, after.return_value), (
            f"Return mismatch on case {idx}: {before.return_value} vs {after.return_value}"
        )
        assert np.allclose(before.printed_values, after.printed_values), (
            f"Printed outputs mismatch on case {idx}: {before.printed_values} vs {after.printed_values}"
        )
        assert before.trace_log == after.trace_log, (
            f"Trace log mismatch on case {idx}: {before.trace_log} vs {after.trace_log}"
        )


def main() -> None:
    original_cfg = build_sample_cfg()
    optimized_cfg = clone_cfg(original_cfg)

    before_count = count_instructions(optimized_cfg)
    report = eliminate_dead_code(optimized_cfg)
    after_count = count_instructions(optimized_cfg)

    run_equivalence_tests(original_cfg, optimized_cfg)
    ensure_no_pure_dead_assignments(optimized_cfg)

    removed_ratio = (before_count - after_count) / max(before_count, 1)

    print("=== Dead Code Elimination (Liveness-based) ===")
    print(f"Instruction count: before={before_count}, after={after_count}")
    print(f"Removed: {before_count - after_count} ({removed_ratio:.2%})")
    print(f"Optimization passes: {report.passes}")
    print()

    for pass_idx, removed_lines in enumerate(report.removed_by_pass, start=1):
        print(f"Pass {pass_idx} removed {len(removed_lines)} instructions:")
        for line in removed_lines:
            print(f"  - {line}")
        print()

    print("--- Original CFG ---")
    print(format_cfg(original_cfg))
    print()
    print("--- Optimized CFG ---")
    print(format_cfg(optimized_cfg))
    print()
    print("Semantic equivalence checks passed.")


if __name__ == "__main__":
    main()
