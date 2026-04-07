"""Minimal runnable MVP for live interval analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple


@dataclass(frozen=True)
class Instruction:
    op: str
    defs: Tuple[str, ...]
    uses: Tuple[str, ...]
    text: str


@dataclass
class BasicBlock:
    name: str
    instructions: List[Instruction]
    successors: List[str]


@dataclass(frozen=True)
class LineRecord:
    pos: int
    block: str
    index_in_block: int
    instr: Instruction
    live_before: Set[str]
    live_after: Set[str]


@dataclass(frozen=True)
class Interval:
    var: str
    start: int
    end: int
    segments: Tuple[Tuple[int, int], ...]

    @property
    def span(self) -> int:
        return self.end - self.start + 1

    @property
    def holes(self) -> int:
        return max(0, len(self.segments) - 1)


def format_set(values: Iterable[str]) -> str:
    ordered = sorted(set(values))
    return "{" + ", ".join(ordered) + "}" if ordered else "{}"


def build_sample_cfg() -> Tuple[Dict[str, BasicBlock], List[str]]:
    blocks: Dict[str, BasicBlock] = {
        "entry": BasicBlock(
            name="entry",
            instructions=[
                Instruction("const", ("p",), (), "p = const 7"),
                Instruction("const", ("q",), (), "q = const 3"),
                Instruction("add", ("x",), ("p", "q"), "x = add p, q"),
                Instruction("gt", ("cond",), ("x", "q"), "cond = gt x, q"),
                Instruction("cjump", (), ("cond",), "cjump cond then else"),
            ],
            successors=["then", "else"],
        ),
        "then": BasicBlock(
            name="then",
            instructions=[
                Instruction("mul", ("t",), ("x", "p"), "t = mul x, p"),
                Instruction("add", ("y",), ("t", "q"), "y = add t, q"),
                Instruction("jump", (), (), "jump merge"),
            ],
            successors=["merge"],
        ),
        "else": BasicBlock(
            name="else",
            instructions=[
                Instruction("sub", ("u",), ("x", "q"), "u = sub x, q"),
                Instruction("add", ("y",), ("u", "p"), "y = add u, p"),
                Instruction("jump", (), (), "jump merge"),
            ],
            successors=["merge"],
        ),
        "merge": BasicBlock(
            name="merge",
            instructions=[
                Instruction("add", ("z",), ("y", "x"), "z = add y, x"),
                Instruction("ret", (), ("z",), "ret z"),
            ],
            successors=[],
        ),
    }
    order = ["entry", "then", "else", "merge"]
    return blocks, order


def block_use_def(block: BasicBlock) -> Tuple[Set[str], Set[str]]:
    use: Set[str] = set()
    defs: Set[str] = set()
    for instr in block.instructions:
        for var in instr.uses:
            if var not in defs:
                use.add(var)
        defs.update(instr.defs)
    return use, defs


def liveness_analysis(
    blocks: Dict[str, BasicBlock], order: Sequence[str]
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]], int]:
    use_map: Dict[str, Set[str]] = {}
    def_map: Dict[str, Set[str]] = {}
    for bname in order:
        use_map[bname], def_map[bname] = block_use_def(blocks[bname])

    live_in: Dict[str, Set[str]] = {bname: set() for bname in order}
    live_out: Dict[str, Set[str]] = {bname: set() for bname in order}

    changed = True
    iterations = 0
    while changed:
        changed = False
        iterations += 1
        for bname in reversed(order):
            block = blocks[bname]
            new_out: Set[str] = set()
            for succ in block.successors:
                new_out |= live_in[succ]
            new_in = use_map[bname] | (new_out - def_map[bname])

            if new_out != live_out[bname] or new_in != live_in[bname]:
                live_out[bname] = new_out
                live_in[bname] = new_in
                changed = True

    return use_map, def_map, live_in, live_out, iterations


def instruction_liveness(
    block: BasicBlock, live_out_block: Set[str]
) -> Tuple[List[Set[str]], List[Set[str]]]:
    n = len(block.instructions)
    live_before: List[Set[str]] = [set() for _ in range(n)]
    live_after: List[Set[str]] = [set() for _ in range(n)]

    current = set(live_out_block)
    for i in range(n - 1, -1, -1):
        instr = block.instructions[i]
        after_i = set(current)
        before_i = set(instr.uses) | (after_i - set(instr.defs))
        live_after[i] = after_i
        live_before[i] = before_i
        current = before_i

    return live_before, live_after


def linearize_records(
    blocks: Dict[str, BasicBlock], order: Sequence[str], live_out: Dict[str, Set[str]]
) -> List[LineRecord]:
    records: List[LineRecord] = []
    pos = 1
    for bname in order:
        block = blocks[bname]
        before, after = instruction_liveness(block, live_out[bname])
        for idx, instr in enumerate(block.instructions, start=1):
            records.append(
                LineRecord(
                    pos=pos,
                    block=bname,
                    index_in_block=idx,
                    instr=instr,
                    live_before=before[idx - 1],
                    live_after=after[idx - 1],
                )
            )
            pos += 1
    return records


def build_segments(points: List[int]) -> Tuple[Tuple[int, int], ...]:
    if not points:
        return tuple()
    segs: List[Tuple[int, int]] = []
    start = points[0]
    prev = points[0]
    for p in points[1:]:
        if p == prev + 1:
            prev = p
            continue
        segs.append((start, prev))
        start = p
        prev = p
    segs.append((start, prev))
    return tuple(segs)


def live_intervals(records: Sequence[LineRecord]) -> List[Interval]:
    points_by_var: Dict[str, Set[int]] = {}
    for rec in records:
        related = set(rec.instr.defs) | set(rec.instr.uses) | rec.live_before | rec.live_after
        for var in related:
            points_by_var.setdefault(var, set()).add(rec.pos)

    intervals: List[Interval] = []
    for var, points in points_by_var.items():
        ordered = sorted(points)
        segs = build_segments(ordered)
        intervals.append(
            Interval(
                var=var,
                start=ordered[0],
                end=ordered[-1],
                segments=segs,
            )
        )
    intervals.sort(key=lambda it: (it.start, it.end, it.var))
    return intervals


def peak_register_pressure(records: Sequence[LineRecord]) -> Tuple[int, List[int]]:
    max_pressure = -1
    max_positions: List[int] = []
    for rec in records:
        pressure = len(rec.live_before)
        if pressure > max_pressure:
            max_pressure = pressure
            max_positions = [rec.pos]
        elif pressure == max_pressure:
            max_positions.append(rec.pos)
    return max_pressure, max_positions


def print_block_liveness(
    order: Sequence[str],
    use_map: Dict[str, Set[str]],
    def_map: Dict[str, Set[str]],
    live_in: Dict[str, Set[str]],
    live_out: Dict[str, Set[str]],
    iterations: int,
) -> None:
    print(f"=== Block Liveness (fixed point in {iterations} iterations) ===")
    for bname in order:
        print(
            f"[{bname}] use={format_set(use_map[bname])} "
            f"def={format_set(def_map[bname])} "
            f"in={format_set(live_in[bname])} "
            f"out={format_set(live_out[bname])}"
        )
    print()


def print_instruction_records(records: Sequence[LineRecord]) -> None:
    print("=== Instruction-Level Liveness ===")
    for rec in records:
        instr = rec.instr
        print(
            f"[{rec.pos:02d}] {rec.block}.{rec.index_in_block} {instr.text:<24} "
            f"defs={format_set(instr.defs):<18} uses={format_set(instr.uses):<18}"
        )
        print(
            f"     live_before={format_set(rec.live_before):<24} "
            f"live_after={format_set(rec.live_after)}"
        )
    print()


def print_intervals(intervals: Sequence[Interval]) -> None:
    print("=== Live Intervals (linearized) ===")
    print("var      start  end  span  holes  segments")
    for it in intervals:
        segs = ", ".join(f"[{l},{r}]" for l, r in it.segments)
        print(
            f"{it.var:<8} {it.start:>5} {it.end:>4} {it.span:>5} "
            f"{it.holes:>6}  {segs}"
        )
    print()


def sanity_checks(intervals: Sequence[Interval], records: Sequence[LineRecord]) -> None:
    if not intervals:
        raise AssertionError("No intervals were produced.")
    vars_from_records = set()
    for rec in records:
        vars_from_records |= set(rec.instr.defs) | set(rec.instr.uses)
    vars_from_intervals = {it.var for it in intervals}
    if vars_from_records != vars_from_intervals:
        missing = vars_from_records - vars_from_intervals
        extra = vars_from_intervals - vars_from_records
        raise AssertionError(f"Interval set mismatch, missing={missing}, extra={extra}")
    for it in intervals:
        if it.start > it.end:
            raise AssertionError(f"Invalid interval for {it.var}: start > end")


def main() -> None:
    blocks, order = build_sample_cfg()
    use_map, def_map, live_in, live_out, iterations = liveness_analysis(blocks, order)
    records = linearize_records(blocks, order, live_out)
    intervals = live_intervals(records)
    pressure, positions = peak_register_pressure(records)

    print_block_liveness(order, use_map, def_map, live_in, live_out, iterations)
    print_instruction_records(records)
    print_intervals(intervals)
    print(f"Peak register pressure = {pressure}, positions = {positions}")

    sanity_checks(intervals, records)
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
