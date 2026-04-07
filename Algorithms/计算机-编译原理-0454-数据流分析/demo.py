"""Minimal runnable MVP for data flow analysis in compiler theory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Set, Tuple


SetMap = Dict[str, Set[str]]
MeetFunc = Callable[[Iterable[Set[str]]], Set[str]]
TransferFunc = Callable[[str, Set[str]], Set[str]]


@dataclass(frozen=True)
class CFG:
    blocks: List[str]
    preds: Dict[str, List[str]]
    succs: Dict[str, List[str]]
    entry: str
    exit: str


@dataclass(frozen=True)
class DataFlowProblem:
    name: str
    direction: str  # "forward" or "backward"
    cfg: CFG
    meet: MeetFunc
    transfer: TransferFunc
    boundary_value: Set[str]
    initial_value: Set[str]


def union_meet(values: Iterable[Set[str]]) -> Set[str]:
    result: Set[str] = set()
    for value in values:
        result |= value
    return result


def solve_data_flow(problem: DataFlowProblem) -> Tuple[SetMap, SetMap, int]:
    if problem.direction not in {"forward", "backward"}:
        raise ValueError(f"Unsupported direction: {problem.direction}")

    in_map: SetMap = {block: set(problem.initial_value) for block in problem.cfg.blocks}
    out_map: SetMap = {block: set(problem.initial_value) for block in problem.cfg.blocks}

    if problem.direction == "forward":
        in_map[problem.cfg.entry] = set(problem.boundary_value)
    else:
        out_map[problem.cfg.exit] = set(problem.boundary_value)

    iterations = 0
    changed = True
    while changed:
        changed = False
        iterations += 1

        if problem.direction == "forward":
            order = problem.cfg.blocks
            for block in order:
                if block == problem.cfg.entry:
                    new_in = set(problem.boundary_value)
                else:
                    pred_out_sets = [out_map[pred] for pred in problem.cfg.preds[block]]
                    new_in = problem.meet(pred_out_sets) if pred_out_sets else set(problem.initial_value)

                new_out = problem.transfer(block, new_in)
                if new_in != in_map[block] or new_out != out_map[block]:
                    in_map[block] = new_in
                    out_map[block] = new_out
                    changed = True
        else:
            order = list(reversed(problem.cfg.blocks))
            for block in order:
                if block == problem.cfg.exit:
                    new_out = set(problem.boundary_value)
                else:
                    succ_in_sets = [in_map[succ] for succ in problem.cfg.succs[block]]
                    new_out = problem.meet(succ_in_sets) if succ_in_sets else set(problem.initial_value)

                new_in = problem.transfer(block, new_out)
                if new_in != in_map[block] or new_out != out_map[block]:
                    in_map[block] = new_in
                    out_map[block] = new_out
                    changed = True

    return in_map, out_map, iterations


def build_cfg() -> CFG:
    blocks = ["B1", "B2", "B3", "B4", "B5", "B6"]
    succs = {
        "B1": ["B2"],
        "B2": ["B3", "B4"],
        "B3": ["B5"],
        "B4": ["B5"],
        "B5": ["B6"],
        "B6": [],
    }
    preds = {block: [] for block in blocks}
    for src, dsts in succs.items():
        for dst in dsts:
            preds[dst].append(src)

    return CFG(blocks=blocks, preds=preds, succs=succs, entry="B1", exit="B6")


def build_reaching_definitions_problem(cfg: CFG) -> DataFlowProblem:
    gen: Mapping[str, Set[str]] = {
        "B1": {"d1:a@B1", "d2:b@B1"},
        "B2": set(),
        "B3": {"d3:b@B3"},
        "B4": {"d4:a@B4"},
        "B5": {"d5:c@B5"},
        "B6": set(),
    }
    kill: Mapping[str, Set[str]] = {
        "B1": {"d3:b@B3", "d4:a@B4"},
        "B2": set(),
        "B3": {"d2:b@B1"},
        "B4": {"d1:a@B1"},
        "B5": set(),
        "B6": set(),
    }

    def transfer(block: str, in_set: Set[str]) -> Set[str]:
        return gen[block] | (in_set - kill[block])

    return DataFlowProblem(
        name="Reaching Definitions",
        direction="forward",
        cfg=cfg,
        meet=union_meet,
        transfer=transfer,
        boundary_value=set(),
        initial_value=set(),
    )


def build_live_variables_problem(cfg: CFG) -> DataFlowProblem:
    use: Mapping[str, Set[str]] = {
        "B1": set(),
        "B2": {"a", "b"},
        "B3": {"a"},
        "B4": {"b"},
        "B5": {"a", "b"},
        "B6": {"c"},
    }
    defs: Mapping[str, Set[str]] = {
        "B1": {"a", "b"},
        "B2": set(),
        "B3": {"b"},
        "B4": {"a"},
        "B5": {"c"},
        "B6": set(),
    }

    def transfer(block: str, out_set: Set[str]) -> Set[str]:
        return use[block] | (out_set - defs[block])

    return DataFlowProblem(
        name="Live Variables",
        direction="backward",
        cfg=cfg,
        meet=union_meet,
        transfer=transfer,
        boundary_value=set(),
        initial_value=set(),
    )


def format_set(values: Set[str]) -> str:
    if not values:
        return "{}"
    return "{" + ", ".join(sorted(values)) + "}"


def print_result(title: str, cfg: CFG, in_map: SetMap, out_map: SetMap, iterations: int) -> None:
    print(f"\n=== {title} ===")
    print(f"Converged in {iterations} iterations")
    for block in cfg.blocks:
        print(
            f"{block}: IN={format_set(in_map[block])}  "
            f"OUT={format_set(out_map[block])}"
        )


def assert_expected(
    name: str,
    cfg: CFG,
    in_map: Mapping[str, Set[str]],
    out_map: Mapping[str, Set[str]],
    expected_in: Mapping[str, Set[str]],
    expected_out: Mapping[str, Set[str]],
) -> None:
    for block in cfg.blocks:
        if in_map[block] != expected_in[block]:
            raise AssertionError(
                f"{name} IN mismatch at {block}: got {in_map[block]}, expected {expected_in[block]}"
            )
        if out_map[block] != expected_out[block]:
            raise AssertionError(
                f"{name} OUT mismatch at {block}: got {out_map[block]}, expected {expected_out[block]}"
            )


def main() -> None:
    cfg = build_cfg()
    print("CFG blocks:", cfg.blocks)
    print("CFG edges:")
    for src in cfg.blocks:
        for dst in cfg.succs[src]:
            print(f"  {src} -> {dst}")

    rd_problem = build_reaching_definitions_problem(cfg)
    rd_in, rd_out, rd_iterations = solve_data_flow(rd_problem)
    print_result(rd_problem.name, cfg, rd_in, rd_out, rd_iterations)

    lv_problem = build_live_variables_problem(cfg)
    lv_in, lv_out, lv_iterations = solve_data_flow(lv_problem)
    print_result(lv_problem.name, cfg, lv_in, lv_out, lv_iterations)

    expected_rd_in: SetMap = {
        "B1": set(),
        "B2": {"d1:a@B1", "d2:b@B1"},
        "B3": {"d1:a@B1", "d2:b@B1"},
        "B4": {"d1:a@B1", "d2:b@B1"},
        "B5": {"d1:a@B1", "d2:b@B1", "d3:b@B3", "d4:a@B4"},
        "B6": {"d1:a@B1", "d2:b@B1", "d3:b@B3", "d4:a@B4", "d5:c@B5"},
    }
    expected_rd_out: SetMap = {
        "B1": {"d1:a@B1", "d2:b@B1"},
        "B2": {"d1:a@B1", "d2:b@B1"},
        "B3": {"d1:a@B1", "d3:b@B3"},
        "B4": {"d2:b@B1", "d4:a@B4"},
        "B5": {"d1:a@B1", "d2:b@B1", "d3:b@B3", "d4:a@B4", "d5:c@B5"},
        "B6": {"d1:a@B1", "d2:b@B1", "d3:b@B3", "d4:a@B4", "d5:c@B5"},
    }

    expected_lv_in: SetMap = {
        "B1": set(),
        "B2": {"a", "b"},
        "B3": {"a"},
        "B4": {"b"},
        "B5": {"a", "b"},
        "B6": {"c"},
    }
    expected_lv_out: SetMap = {
        "B1": {"a", "b"},
        "B2": {"a", "b"},
        "B3": {"a", "b"},
        "B4": {"a", "b"},
        "B5": {"c"},
        "B6": set(),
    }

    assert_expected("Reaching Definitions", cfg, rd_in, rd_out, expected_rd_in, expected_rd_out)
    assert_expected("Live Variables", cfg, lv_in, lv_out, expected_lv_in, expected_lv_out)

    print("\nValidation: PASS")


if __name__ == "__main__":
    main()
