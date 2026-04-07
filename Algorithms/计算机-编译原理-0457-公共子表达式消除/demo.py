"""公共子表达式消除（CSE）最小可运行 MVP。

实现策略：
- 在每个基本块内执行局部公共子表达式消除；
- 使用局部值编号（Local Value Numbering, LVN）追踪表达式等价类；
- 对可复用表达式改写为复制语句；
- 通过随机测试校验优化前后语义一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, List, Set, Tuple, Union

import numpy as np

COMMUTATIVE_OPS: Set[str] = {"+", "*"}


@dataclass(frozen=True)
class BinOp:
    target: str
    op: str
    left: str
    right: str


@dataclass(frozen=True)
class Assign:
    target: str
    source: str


@dataclass(frozen=True)
class Const:
    target: str
    value: int


Instruction = Union[BinOp, Assign, Const]


@dataclass
class BasicBlock:
    name: str
    instructions: List[Instruction]


@dataclass
class LVNState:
    next_vn: int = 1
    var_vn: Dict[str, int] = field(default_factory=dict)
    const_vn: Dict[int, int] = field(default_factory=dict)
    expr_vn: Dict[Tuple[str, int, int], int] = field(default_factory=dict)
    vn_holders: Dict[int, Set[str]] = field(default_factory=dict)


def format_instruction(inst: Instruction) -> str:
    if isinstance(inst, BinOp):
        return f"{inst.target} = {inst.left} {inst.op} {inst.right}"
    if isinstance(inst, Assign):
        return f"{inst.target} = {inst.source}"
    if isinstance(inst, Const):
        return f"{inst.target} = {inst.value}"
    raise TypeError(f"Unsupported instruction type: {type(inst)}")


def print_program(blocks: List[BasicBlock], title: str) -> None:
    print(f"\n=== {title} ===")
    for block in blocks:
        print(f"[{block.name}]")
        for idx, inst in enumerate(block.instructions, start=1):
            print(f"  {idx:02d}. {format_instruction(inst)}")


def new_value_number(state: LVNState) -> int:
    vn = state.next_vn
    state.next_vn += 1
    state.vn_holders.setdefault(vn, set())
    return vn


def detach_var_holder(state: LVNState, var: str) -> None:
    old_vn = state.var_vn.get(var)
    if old_vn is None:
        return
    holders = state.vn_holders.get(old_vn)
    if holders is not None:
        holders.discard(var)


def attach_var_to_vn(state: LVNState, var: str, vn: int) -> None:
    detach_var_holder(state, var)
    state.var_vn[var] = vn
    state.vn_holders.setdefault(vn, set()).add(var)


def get_or_create_var_vn(state: LVNState, var: str) -> int:
    if var in state.var_vn:
        return state.var_vn[var]
    vn = new_value_number(state)
    state.var_vn[var] = vn
    state.vn_holders[vn].add(var)
    return vn


def pick_holder(state: LVNState, vn: int) -> str | None:
    holders = sorted(state.vn_holders.get(vn, set()))
    if not holders:
        return None
    return holders[0]


def canonical_expr_key(op: str, vn_left: int, vn_right: int) -> Tuple[str, int, int]:
    if op in COMMUTATIVE_OPS and vn_left > vn_right:
        vn_left, vn_right = vn_right, vn_left
    return (op, vn_left, vn_right)


def local_cse_block(block: BasicBlock) -> Tuple[BasicBlock, int, List[str]]:
    state = LVNState()
    optimized: List[Instruction] = []
    eliminated = 0
    trace: List[str] = []

    for idx, inst in enumerate(block.instructions, start=1):
        if isinstance(inst, Const):
            existed_vn = state.const_vn.get(inst.value)
            if existed_vn is None:
                existed_vn = new_value_number(state)
                state.const_vn[inst.value] = existed_vn
                optimized.append(inst)
                trace.append(
                    f"{block.name}:{idx} 常量 {inst.value} 首次出现，保留定义 {inst.target} -> VN{existed_vn}"
                )
            else:
                holder = pick_holder(state, existed_vn)
                if holder is not None and holder != inst.target:
                    optimized.append(Assign(inst.target, holder))
                    eliminated += 1
                    trace.append(
                        f"{block.name}:{idx} 常量 {inst.value} 已有等价变量 {holder}，改写为复制"
                    )
                else:
                    optimized.append(inst)
                    trace.append(
                        f"{block.name}:{idx} 常量 {inst.value} 已知但无可复用变量，保留原语句"
                    )
            attach_var_to_vn(state, inst.target, existed_vn)
            continue

        if isinstance(inst, Assign):
            src_vn = get_or_create_var_vn(state, inst.source)
            holder = pick_holder(state, src_vn) or inst.source
            optimized.append(Assign(inst.target, holder))
            attach_var_to_vn(state, inst.target, src_vn)
            trace.append(
                f"{block.name}:{idx} 复制传播 {inst.target} <- {holder} (VN{src_vn})"
            )
            continue

        if isinstance(inst, BinOp):
            left_vn = get_or_create_var_vn(state, inst.left)
            right_vn = get_or_create_var_vn(state, inst.right)
            key = canonical_expr_key(inst.op, left_vn, right_vn)
            existed_vn = state.expr_vn.get(key)

            if existed_vn is not None:
                holder = pick_holder(state, existed_vn)
                if holder is not None:
                    if holder != inst.target:
                        optimized.append(Assign(inst.target, holder))
                    eliminated += 1
                    attach_var_to_vn(state, inst.target, existed_vn)
                    trace.append(
                        f"{block.name}:{idx} 命中公共子表达式 {inst.left} {inst.op} {inst.right}，"
                        f"复用 {holder} (VN{existed_vn})"
                    )
                    continue

            if existed_vn is None:
                result_vn = new_value_number(state)
                state.expr_vn[key] = result_vn
            else:
                result_vn = existed_vn

            optimized.append(inst)
            attach_var_to_vn(state, inst.target, result_vn)
            trace.append(
                f"{block.name}:{idx} 新建表达式类 {inst.left} {inst.op} {inst.right} -> VN{result_vn}"
            )
            continue

        raise TypeError(f"Unsupported instruction: {inst}")

    return BasicBlock(name=block.name, instructions=optimized), eliminated, trace


def optimize_program(blocks: List[BasicBlock]) -> Tuple[List[BasicBlock], int, Dict[str, List[str]]]:
    optimized_blocks: List[BasicBlock] = []
    total_eliminated = 0
    traces: Dict[str, List[str]] = {}

    for block in blocks:
        new_block, eliminated, trace = local_cse_block(block)
        optimized_blocks.append(new_block)
        total_eliminated += eliminated
        traces[block.name] = trace

    return optimized_blocks, total_eliminated, traces


def eval_binop(op: str, left: int, right: int) -> int:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    raise ValueError(f"Unsupported op: {op}")


def execute_program(blocks: List[BasicBlock], env: Dict[str, int]) -> Dict[str, int]:
    memory = dict(env)
    for block in blocks:
        for inst in block.instructions:
            if isinstance(inst, Const):
                memory[inst.target] = inst.value
            elif isinstance(inst, Assign):
                if inst.source not in memory:
                    raise KeyError(f"Undefined source variable: {inst.source}")
                memory[inst.target] = memory[inst.source]
            elif isinstance(inst, BinOp):
                if inst.left not in memory or inst.right not in memory:
                    raise KeyError(f"Undefined operand in instruction: {format_instruction(inst)}")
                memory[inst.target] = eval_binop(inst.op, memory[inst.left], memory[inst.right])
            else:
                raise TypeError(f"Unsupported instruction type: {type(inst)}")
    return memory


def collect_program_inputs(blocks: List[BasicBlock]) -> List[str]:
    defined: Set[str] = set()
    required_inputs: Set[str] = set()

    for block in blocks:
        for inst in block.instructions:
            if isinstance(inst, BinOp):
                for operand in (inst.left, inst.right):
                    if operand not in defined:
                        required_inputs.add(operand)
                defined.add(inst.target)
            elif isinstance(inst, Assign):
                if inst.source not in defined:
                    required_inputs.add(inst.source)
                defined.add(inst.target)
            elif isinstance(inst, Const):
                defined.add(inst.target)
            else:
                raise TypeError(f"Unsupported instruction type: {type(inst)}")

    return sorted(required_inputs)


def verify_semantics(
    original: List[BasicBlock],
    optimized: List[BasicBlock],
    rounds: int = 200,
    seed: int = 2026,
) -> None:
    input_vars = collect_program_inputs(original)
    rng = random.Random(seed)

    for case_id in range(1, rounds + 1):
        env = {name: rng.randint(-9, 9) for name in input_vars}
        out_original = execute_program(original, env)
        out_optimized = execute_program(optimized, env)

        if out_original != out_optimized:
            raise AssertionError(
                "语义校验失败\n"
                f"case={case_id}\n"
                f"input={env}\n"
                f"original={out_original}\n"
                f"optimized={out_optimized}"
            )


def count_binops(block: BasicBlock) -> int:
    return sum(1 for inst in block.instructions if isinstance(inst, BinOp))


def build_sample_program() -> List[BasicBlock]:
    return [
        BasicBlock(
            name="B1",
            instructions=[
                BinOp("t1", "+", "a", "b"),
                BinOp("t2", "+", "a", "b"),
                BinOp("t3", "*", "t2", "c"),
                BinOp("t4", "+", "b", "a"),
                BinOp("a", "+", "a", "d"),
                BinOp("t5", "+", "a", "b"),
                BinOp("t6", "+", "a", "b"),
            ],
        ),
        BasicBlock(
            name="B2",
            instructions=[
                BinOp("u1", "*", "x", "y"),
                BinOp("u2", "*", "y", "x"),
                Assign("z", "u2"),
                BinOp("u3", "+", "z", "x"),
                BinOp("u4", "+", "u1", "x"),
            ],
        ),
    ]


def main() -> None:
    original = build_sample_program()
    optimized, eliminated, traces = optimize_program(original)

    print_program(original, "优化前 IR")

    print("\n=== 块内 CSE 追踪 ===")
    for block in original:
        print(f"[{block.name}]")
        for line in traces[block.name]:
            print(f"  - {line}")

    print_program(optimized, "优化后 IR")

    before_counts = np.array([count_binops(block) for block in original], dtype=np.int64)
    after_counts = np.array([count_binops(block) for block in optimized], dtype=np.int64)
    reduction_counts = before_counts - after_counts

    print("\n=== 统计信息 ===")
    for i, block in enumerate(original):
        print(
            f"{block.name}: 二元表达式 {before_counts[i]} -> {after_counts[i]} "
            f"(减少 {reduction_counts[i]})"
        )

    total_before = int(before_counts.sum())
    total_after = int(after_counts.sum())
    ratio = (total_before - total_after) / total_before if total_before > 0 else 0.0
    print(f"总减少的二元表达式条数: {total_before - total_after}")
    print(f"总 CSE 命中次数: {eliminated}")
    print(f"二元表达式削减比例: {ratio:.2%}")

    verify_semantics(original, optimized, rounds=200, seed=2026)
    print("\n语义一致性校验: 200/200 随机用例通过")


if __name__ == "__main__":
    main()
