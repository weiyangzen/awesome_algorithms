"""Minimal runnable MVP for basic block partition in compiler theory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is optional for this MVP
    np = None


@dataclass(frozen=True)
class TACInstruction:
    """A tiny three-address-code instruction model for block partition demo."""

    op: str
    text: str
    label: Optional[str] = None
    target: Optional[str] = None
    true_target: Optional[str] = None
    false_target: Optional[str] = None

    def is_jump(self) -> bool:
        return self.op == "jump"

    def is_cjump(self) -> bool:
        return self.op == "cjump"

    def is_return(self) -> bool:
        return self.op == "return"


@dataclass(frozen=True)
class BasicBlock:
    name: str
    start_idx: int
    end_idx: int
    instructions: Tuple[TACInstruction, ...]


def build_sample_program() -> List[TACInstruction]:
    """A fixed TAC program with branch/merge pattern."""
    return [
        TACInstruction(op="label", label="L0", text="L0:"),
        TACInstruction(op="assign", text="t1 = a + b"),
        TACInstruction(op="cjump", text="if t1 > 0 goto L1 else L2", true_target="L1", false_target="L2"),
        TACInstruction(op="label", label="L1", text="L1:"),
        TACInstruction(op="assign", text="x = t1 * 2"),
        TACInstruction(op="jump", text="goto L3", target="L3"),
        TACInstruction(op="label", label="L2", text="L2:"),
        TACInstruction(op="assign", text="x = t1 - 1"),
        TACInstruction(op="label", label="L3", text="L3:"),
        TACInstruction(op="assign", text="y = x + 4"),
        TACInstruction(op="return", text="return y"),
    ]


def build_label_index(program: Sequence[TACInstruction]) -> Dict[str, int]:
    label_to_index: Dict[str, int] = {}
    for idx, inst in enumerate(program):
        if inst.op == "label":
            if inst.label is None:
                raise ValueError(f"Label instruction at index {idx} has no label name")
            if inst.label in label_to_index:
                raise ValueError(f"Duplicate label detected: {inst.label}")
            label_to_index[inst.label] = idx
    return label_to_index


def resolve_label(label: Optional[str], label_to_index: Mapping[str, int]) -> int:
    if label is None:
        raise ValueError("Jump-like instruction misses target label")
    if label not in label_to_index:
        raise ValueError(f"Unknown label target: {label}")
    return label_to_index[label]


def find_leader_indices(program: Sequence[TACInstruction], label_to_index: Mapping[str, int]) -> List[int]:
    """Classical leader rules:
    1) first instruction;
    2) every jump target;
    3) instruction following jump/cjump.
    """
    if not program:
        return []

    leaders = {0}
    n = len(program)

    for idx, inst in enumerate(program):
        if inst.is_jump():
            leaders.add(resolve_label(inst.target, label_to_index))
            if idx + 1 < n:
                leaders.add(idx + 1)
        elif inst.is_cjump():
            leaders.add(resolve_label(inst.true_target, label_to_index))
            if inst.false_target is not None:
                leaders.add(resolve_label(inst.false_target, label_to_index))
            if idx + 1 < n:
                leaders.add(idx + 1)

    return sorted(leaders)


def partition_basic_blocks(program: Sequence[TACInstruction], leaders: Sequence[int]) -> List[BasicBlock]:
    if not program:
        return []
    if not leaders:
        raise ValueError("Leaders cannot be empty when program is non-empty")
    if leaders[0] != 0:
        raise ValueError("Leader list must start with 0")

    blocks: List[BasicBlock] = []
    for block_id, start in enumerate(leaders):
        end = leaders[block_id + 1] - 1 if block_id + 1 < len(leaders) else len(program) - 1
        block = BasicBlock(
            name=f"B{block_id + 1}",
            start_idx=start,
            end_idx=end,
            instructions=tuple(program[start : end + 1]),
        )
        blocks.append(block)

    return blocks


def build_index_to_block(blocks: Sequence[BasicBlock]) -> Dict[int, str]:
    index_to_block: Dict[int, str] = {}
    for block in blocks:
        for idx in range(block.start_idx, block.end_idx + 1):
            index_to_block[idx] = block.name
    return index_to_block


def deduplicate_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def build_block_cfg(
    blocks: Sequence[BasicBlock],
    index_to_block: Mapping[int, str],
    label_to_index: Mapping[str, int],
) -> Dict[str, List[str]]:
    cfg: Dict[str, List[str]] = {block.name: [] for block in blocks}

    for i, block in enumerate(blocks):
        last_inst = block.instructions[-1]
        successors: List[str] = []

        if last_inst.is_jump():
            target_idx = resolve_label(last_inst.target, label_to_index)
            successors.append(index_to_block[target_idx])
        elif last_inst.is_cjump():
            true_idx = resolve_label(last_inst.true_target, label_to_index)
            successors.append(index_to_block[true_idx])
            if last_inst.false_target is not None:
                false_idx = resolve_label(last_inst.false_target, label_to_index)
                successors.append(index_to_block[false_idx])
            elif i + 1 < len(blocks):
                successors.append(blocks[i + 1].name)
        elif last_inst.is_return():
            successors = []
        elif i + 1 < len(blocks):
            successors.append(blocks[i + 1].name)

        cfg[block.name] = deduplicate_preserve_order(successors)

    return cfg


def summarize_block_sizes(blocks: Sequence[BasicBlock]) -> Dict[str, float]:
    sizes = [len(block.instructions) for block in blocks]
    if not sizes:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

    if np is not None:
        arr = np.array(sizes, dtype=float)
        return {
            "count": float(arr.size),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(arr.std()),
        }

    mean = sum(sizes) / len(sizes)
    var = sum((x - mean) ** 2 for x in sizes) / len(sizes)
    return {
        "count": float(len(sizes)),
        "mean": float(mean),
        "min": float(min(sizes)),
        "max": float(max(sizes)),
        "std": float(var**0.5),
    }


def print_program(program: Sequence[TACInstruction]) -> None:
    print("=== TAC Program ===")
    for idx, inst in enumerate(program):
        print(f"{idx:02d}: {inst.text}")


def print_blocks(blocks: Sequence[BasicBlock]) -> None:
    print("\n=== Basic Blocks ===")
    for block in blocks:
        print(f"{block.name} [{block.start_idx}, {block.end_idx}]")
        for idx, inst in enumerate(block.instructions, start=block.start_idx):
            print(f"  {idx:02d}: {inst.text}")


def print_cfg(cfg: Mapping[str, Sequence[str]], block_order: Sequence[str]) -> None:
    print("\n=== Block CFG ===")
    for block in block_order:
        succs = cfg[block]
        succ_text = ", ".join(succs) if succs else "(exit)"
        print(f"{block} -> {succ_text}")


def assert_expected(
    leaders: Sequence[int],
    blocks: Sequence[BasicBlock],
    cfg: Mapping[str, Sequence[str]],
) -> None:
    expected_leaders = [0, 3, 6, 8]
    if list(leaders) != expected_leaders:
        raise AssertionError(f"Leader mismatch: got {leaders}, expected {expected_leaders}")

    expected_ranges = {
        "B1": (0, 2),
        "B2": (3, 5),
        "B3": (6, 7),
        "B4": (8, 10),
    }
    for block in blocks:
        got_range = (block.start_idx, block.end_idx)
        if expected_ranges[block.name] != got_range:
            raise AssertionError(
                f"Range mismatch on {block.name}: got {got_range}, expected {expected_ranges[block.name]}"
            )

    expected_cfg = {
        "B1": ["B2", "B3"],
        "B2": ["B4"],
        "B3": ["B4"],
        "B4": [],
    }
    for block_name, expected_succs in expected_cfg.items():
        got_succs = list(cfg[block_name])
        if got_succs != expected_succs:
            raise AssertionError(
                f"CFG mismatch at {block_name}: got {got_succs}, expected {expected_succs}"
            )


def main() -> None:
    program = build_sample_program()
    label_to_index = build_label_index(program)
    leaders = find_leader_indices(program, label_to_index)
    blocks = partition_basic_blocks(program, leaders)
    index_to_block = build_index_to_block(blocks)
    cfg = build_block_cfg(blocks, index_to_block, label_to_index)
    summary = summarize_block_sizes(blocks)

    print_program(program)
    print("\nLeaders:", leaders)
    print_blocks(blocks)
    print_cfg(cfg, [block.name for block in blocks])

    print("\n=== Block Size Summary ===")
    print(
        "count={count:.0f}, mean={mean:.2f}, min={min:.0f}, max={max:.0f}, std={std:.2f}".format(
            **summary
        )
    )

    assert_expected(leaders, blocks, cfg)
    print("\nValidation: PASS")


if __name__ == "__main__":
    main()
