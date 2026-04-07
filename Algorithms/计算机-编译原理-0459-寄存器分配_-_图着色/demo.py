"""寄存器分配（图着色）最小可运行示例。

目标：
1. 在简化 CFG 上执行活跃变量分析；
2. 由 live-out 信息构建干涉图（Interference Graph）；
3. 使用简化版 Chaitin/Briggs 图着色完成寄存器分配；
4. 在寄存器不足时给出溢出（spill）变量与栈槽映射。

运行：
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Instruction:
    """简化三地址 IR 指令。"""

    op: str
    dst: Optional[str] = None
    args: Tuple[str, ...] = ()
    true_target: Optional[str] = None
    false_target: Optional[str] = None
    target: Optional[str] = None

    def uses(self) -> set[str]:
        if self.op == "const":
            tokens: Iterable[str] = ()
        elif self.op in {"jump"}:
            tokens = ()
        elif self.op in {"ret", "cjump"}:
            tokens = self.args
        else:
            tokens = self.args
        return {tok for tok in tokens if is_virtual_reg(tok)}

    def defines(self) -> Optional[str]:
        if self.op in {"const", "add", "sub", "mul", "mov"}:
            return self.dst
        return None

    def is_move(self) -> bool:
        return self.op == "mov"

    def to_ir(self) -> str:
        if self.op == "const":
            return f"{self.dst} = {self.args[0]}"
        if self.op == "mov":
            return f"{self.dst} = {self.args[0]}"
        if self.op in {"add", "sub", "mul"}:
            symbol = {"add": "+", "sub": "-", "mul": "*"}[self.op]
            return f"{self.dst} = {self.args[0]} {symbol} {self.args[1]}"
        if self.op == "cjump":
            return f"if {self.args[0]} != 0 goto {self.true_target} else {self.false_target}"
        if self.op == "jump":
            return f"goto {self.target}"
        if self.op == "ret":
            return f"return {self.args[0]}"
        raise ValueError(f"Unsupported instruction op: {self.op}")


@dataclass
class BasicBlock:
    name: str
    instructions: List[Instruction]
    successors: List[str]


@dataclass
class AllocationResult:
    physical_registers: List[str]
    color_of: Dict[str, str]
    spilled: List[str]
    spill_slots: Dict[str, int]


def is_virtual_reg(token: str) -> bool:
    if not token:
        return False
    return token[0].isalpha() or token[0] == "_"


def build_sample_cfg() -> Dict[str, BasicBlock]:
    """构造一个寄存器压力较高的示例 CFG。"""
    return {
        "entry": BasicBlock(
            name="entry",
            instructions=[
                Instruction("const", dst="v1", args=("1",)),
                Instruction("const", dst="v2", args=("2",)),
                Instruction("const", dst="v3", args=("3",)),
                Instruction("add", dst="v4", args=("v1", "v2")),
                Instruction("add", dst="cond", args=("v3", "v1")),
                Instruction("cjump", args=("cond",), true_target="then", false_target="else"),
            ],
            successors=["then", "else"],
        ),
        "then": BasicBlock(
            name="then",
            instructions=[
                Instruction("add", dst="v5", args=("v4", "v2")),
                Instruction("add", dst="v6", args=("v5", "v3")),
                Instruction("jump", target="merge"),
            ],
            successors=["merge"],
        ),
        "else": BasicBlock(
            name="else",
            instructions=[
                Instruction("sub", dst="v5", args=("v4", "v1")),
                Instruction("add", dst="v6", args=("v5", "v2")),
                Instruction("jump", target="merge"),
            ],
            successors=["merge"],
        ),
        "merge": BasicBlock(
            name="merge",
            instructions=[
                Instruction("add", dst="v7", args=("v6", "v4")),
                Instruction("add", dst="v8", args=("v7", "v2")),
                Instruction("mul", dst="v9", args=("v8", "v3")),
                Instruction("mov", dst="retv", args=("v9",)),
                Instruction("ret", args=("retv",)),
            ],
            successors=[],
        ),
    }


def compute_predecessors(cfg: Dict[str, BasicBlock]) -> Dict[str, List[str]]:
    preds: Dict[str, List[str]] = {name: [] for name in cfg}
    for block_name, block in cfg.items():
        for succ in block.successors:
            if succ not in cfg:
                raise ValueError(f"Unknown successor: {block_name} -> {succ}")
            preds[succ].append(block_name)
    return preds


def compute_use_def(block: BasicBlock) -> tuple[set[str], set[str]]:
    use: set[str] = set()
    defined: set[str] = set()
    for inst in block.instructions:
        for reg in inst.uses():
            if reg not in defined:
                use.add(reg)
        reg_def = inst.defines()
        if reg_def is not None:
            defined.add(reg_def)
    return use, defined


def liveness_analysis(cfg: Dict[str, BasicBlock]) -> tuple[Dict[str, set[str]], Dict[str, set[str]]]:
    """经典逆向活跃变量分析（基本块粒度）。"""
    block_order = list(cfg.keys())
    live_in: Dict[str, set[str]] = {name: set() for name in block_order}
    live_out: Dict[str, set[str]] = {name: set() for name in block_order}
    use_map: Dict[str, set[str]] = {}
    def_map: Dict[str, set[str]] = {}

    for name in block_order:
        use_map[name], def_map[name] = compute_use_def(cfg[name])

    changed = True
    while changed:
        changed = False
        for name in reversed(block_order):
            block = cfg[name]
            out_new: set[str] = set()
            for succ in block.successors:
                out_new |= live_in[succ]
            in_new = use_map[name] | (out_new - def_map[name])
            if in_new != live_in[name] or out_new != live_out[name]:
                live_in[name] = in_new
                live_out[name] = out_new
                changed = True

    return live_in, live_out


def instruction_live_out_sets(
    cfg: Dict[str, BasicBlock],
    live_out_by_block: Dict[str, set[str]],
) -> Dict[tuple[str, int], set[str]]:
    """从块级 live-out 回推到指令级 live-out。"""
    result: Dict[tuple[str, int], set[str]] = {}
    for block_name, block in cfg.items():
        live = set(live_out_by_block[block_name])
        for idx in range(len(block.instructions) - 1, -1, -1):
            inst = block.instructions[idx]
            result[(block_name, idx)] = set(live)
            reg_def = inst.defines()
            if reg_def is not None and reg_def in live:
                live.remove(reg_def)
            live |= inst.uses()
    return result


def collect_virtual_registers(cfg: Dict[str, BasicBlock]) -> List[str]:
    regs: set[str] = set()
    for block in cfg.values():
        for inst in block.instructions:
            reg_def = inst.defines()
            if reg_def is not None:
                regs.add(reg_def)
            regs |= inst.uses()
    return sorted(regs)


def add_undirected_edge(graph: Dict[str, set[str]], a: str, b: str) -> None:
    if a == b:
        return
    graph[a].add(b)
    graph[b].add(a)


def build_interference_graph(
    cfg: Dict[str, BasicBlock],
    live_out_at_inst: Dict[tuple[str, int], set[str]],
) -> tuple[Dict[str, set[str]], List[tuple[str, str]]]:
    """依据 '定义变量 与 指令后活跃变量' 关系构建干涉图。

    对 move 指令，按经典做法忽略与 move 源操作数的干涉边，便于后续可能的合并。
    """
    regs = collect_virtual_registers(cfg)
    graph: Dict[str, set[str]] = {reg: set() for reg in regs}
    move_pairs: List[tuple[str, str]] = []

    for block_name, block in cfg.items():
        for idx, inst in enumerate(block.instructions):
            reg_def = inst.defines()
            if reg_def is None:
                continue

            live_after = set(live_out_at_inst[(block_name, idx)])
            move_src: Optional[str] = None
            if inst.is_move():
                src = inst.args[0]
                if is_virtual_reg(src):
                    move_src = src
                    move_pairs.append((reg_def, src))

            for live_reg in sorted(live_after):
                if move_src is not None and live_reg == move_src:
                    continue
                add_undirected_edge(graph, reg_def, live_reg)

    return graph, move_pairs


def compute_spill_costs(cfg: Dict[str, BasicBlock]) -> Dict[str, float]:
    """简单 spill cost: 每个虚拟寄存器的引用次数。"""
    regs = collect_virtual_registers(cfg)
    counts = {reg: 0.0 for reg in regs}
    for block in cfg.values():
        for inst in block.instructions:
            reg_def = inst.defines()
            if reg_def is not None:
                counts[reg_def] += 1.0
            for reg in inst.uses():
                counts[reg] += 1.0
    return counts


def choose_spill_candidate(
    mutable_graph: Dict[str, set[str]],
    spill_cost: Dict[str, float],
) -> str:
    """选择溢出候选：最小 cost/degree。"""
    candidates = sorted(mutable_graph.keys())
    best = candidates[0]
    best_score = spill_cost.get(best, 1.0) / max(1, len(mutable_graph[best]))
    for node in candidates[1:]:
        score = spill_cost.get(node, 1.0) / max(1, len(mutable_graph[node]))
        if score < best_score:
            best = node
            best_score = score
    return best


def graph_coloring_allocate(
    graph: Dict[str, set[str]],
    k_registers: Sequence[str],
    spill_cost: Dict[str, float],
) -> AllocationResult:
    k = len(k_registers)
    if k == 0:
        raise ValueError("At least one physical register is required")

    mutable: Dict[str, set[str]] = {n: set(nei) for n, nei in graph.items()}
    stack: List[tuple[str, bool]] = []

    # Simplify / Spill selection.
    while mutable:
        low_degree_nodes = sorted([n for n, nei in mutable.items() if len(nei) < k])
        if low_degree_nodes:
            node = low_degree_nodes[0]
            marked_spill = False
        else:
            node = choose_spill_candidate(mutable, spill_cost)
            marked_spill = True

        neighbors = list(mutable[node])
        for nei in neighbors:
            mutable[nei].remove(node)
        del mutable[node]
        stack.append((node, marked_spill))

    # Select / Color.
    color_of: Dict[str, str] = {}
    spilled: set[str] = set()
    for node, _ in reversed(stack):
        used_colors = {color_of[nei] for nei in graph[node] if nei in color_of}
        available = [r for r in k_registers if r not in used_colors]
        if available:
            color_of[node] = available[0]
        else:
            spilled.add(node)

    spill_slots = {reg: idx for idx, reg in enumerate(sorted(spilled))}
    return AllocationResult(
        physical_registers=list(k_registers),
        color_of=color_of,
        spilled=sorted(spilled),
        spill_slots=spill_slots,
    )


def assert_valid_coloring(graph: Dict[str, set[str]], color_of: Dict[str, str]) -> None:
    for u, neighbors in graph.items():
        if u not in color_of:
            continue
        for v in neighbors:
            if v in color_of and color_of[u] == color_of[v]:
                raise AssertionError(f"Invalid coloring: {u} and {v} share {color_of[u]}")


def print_cfg(cfg: Dict[str, BasicBlock]) -> None:
    print("=== Input CFG ===")
    for block_name, block in cfg.items():
        print(f"[{block_name}]")
        for idx, inst in enumerate(block.instructions):
            print(f"  {idx:02d}: {inst.to_ir()}")
        print(f"  succ: {block.successors}")
    print()


def print_liveness(live_in: Dict[str, set[str]], live_out: Dict[str, set[str]]) -> None:
    print("=== Block Liveness ===")
    for block_name in live_in.keys():
        in_regs = ", ".join(sorted(live_in[block_name]))
        out_regs = ", ".join(sorted(live_out[block_name]))
        print(f"{block_name:>6} | IN={{ {in_regs} }} | OUT={{ {out_regs} }}")
    print()


def print_graph(graph: Dict[str, set[str]]) -> None:
    print("=== Interference Graph ===")
    for node in sorted(graph.keys()):
        neighbors = ", ".join(sorted(graph[node]))
        print(f"{node:>6} : [{neighbors}]")

    degrees = np.array([len(graph[n]) for n in sorted(graph.keys())], dtype=np.int64)
    print()
    print(
        "Degree stats (numpy): "
        f"min={int(degrees.min())}, max={int(degrees.max())}, "
        f"mean={float(degrees.mean()):.2f}"
    )
    print()


def print_allocation(result: AllocationResult) -> None:
    print("=== Allocation Result ===")
    print(f"Physical registers: {result.physical_registers}")
    print("Assigned colors:")
    for reg in sorted(result.color_of.keys()):
        print(f"  {reg:>6} -> {result.color_of[reg]}")

    if result.spilled:
        print("Spilled virtual registers:")
        for reg in result.spilled:
            print(f"  {reg:>6} -> stack[{result.spill_slots[reg]}]")
    else:
        print("Spilled virtual registers: (none)")
    print()


def rewrite_instruction(inst: Instruction, alloc: AllocationResult) -> str:
    def locate(token: str) -> str:
        if not is_virtual_reg(token):
            return token
        if token in alloc.color_of:
            return alloc.color_of[token]
        return f"stack[{alloc.spill_slots[token]}]"

    if inst.op == "const":
        if inst.dst is None:
            raise ValueError("const without dst")
        return f"{locate(inst.dst)} = {inst.args[0]}"
    if inst.op == "mov":
        if inst.dst is None:
            raise ValueError("mov without dst")
        return f"{locate(inst.dst)} = {locate(inst.args[0])}"
    if inst.op in {"add", "sub", "mul"}:
        if inst.dst is None:
            raise ValueError(f"{inst.op} without dst")
        symbol = {"add": "+", "sub": "-", "mul": "*"}[inst.op]
        return f"{locate(inst.dst)} = {locate(inst.args[0])} {symbol} {locate(inst.args[1])}"
    if inst.op == "cjump":
        return f"if {locate(inst.args[0])} != 0 goto {inst.true_target} else {inst.false_target}"
    if inst.op == "jump":
        return f"goto {inst.target}"
    if inst.op == "ret":
        return f"return {locate(inst.args[0])}"
    raise ValueError(f"Unsupported op while rewriting: {inst.op}")


def print_rewritten_program(cfg: Dict[str, BasicBlock], alloc: AllocationResult) -> None:
    print("=== Rewritten Program (register/memory locations) ===")
    for block_name, block in cfg.items():
        print(f"[{block_name}]")
        for inst in block.instructions:
            print(f"  {rewrite_instruction(inst, alloc)}")
    print()


def run_demo() -> None:
    cfg = build_sample_cfg()
    print_cfg(cfg)

    live_in, live_out = liveness_analysis(cfg)
    print_liveness(live_in, live_out)

    live_out_inst = instruction_live_out_sets(cfg, live_out)
    graph, move_pairs = build_interference_graph(cfg, live_out_inst)
    print_graph(graph)

    if move_pairs:
        print(f"Move-related pairs (for possible coalescing): {sorted(move_pairs)}")
        print()

    spill_cost = compute_spill_costs(cfg)
    physical_regs = ["r0", "r1", "r2"]
    alloc = graph_coloring_allocate(graph, physical_regs, spill_cost)
    assert_valid_coloring(graph, alloc.color_of)
    print_allocation(alloc)
    print_rewritten_program(cfg, alloc)

    # 演示断言：确保关键变量参与了分配流程。
    assert "v4" in graph
    assert "retv" in graph
    assert set(alloc.color_of.keys()) | set(alloc.spilled) == set(graph.keys())
    print("All checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
