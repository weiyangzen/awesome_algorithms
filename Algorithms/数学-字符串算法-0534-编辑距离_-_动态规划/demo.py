"""编辑距离（Levenshtein distance）动态规划最小可运行示例。

运行方式：
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class EditOperation:
    """一次对齐编辑操作。"""

    action: str
    src_char: str
    dst_char: str


def build_distance_table(source: str, target: str) -> tuple[list[list[int]], dict[str, int]]:
    """构建 Levenshtein 距离 DP 表。

    状态定义：
    - dp[i][j] 表示 source 前 i 个字符转换为 target 前 j 个字符的最小编辑代价。
    """
    n, m = len(source), len(target)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # 边界：空串到前缀只可能通过全插入或全删除完成。
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            replace_cost = 0 if source[i - 1] == target[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # 删除
                dp[i][j - 1] + 1,  # 插入
                dp[i - 1][j - 1] + replace_cost,  # 匹配/替换
            )

    stats = {
        "rows": n + 1,
        "cols": m + 1,
        "cells_computed": n * m,
    }
    return dp, stats


def backtrack_operations(source: str, target: str, dp: list[list[int]]) -> list[EditOperation]:
    """从 DP 表尾部回溯一条最优编辑路径。"""
    i, j = len(source), len(target)
    reversed_ops: list[EditOperation] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            replace_cost = 0 if source[i - 1] == target[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + replace_cost:
                if replace_cost == 0:
                    reversed_ops.append(
                        EditOperation("match", source[i - 1], target[j - 1])
                    )
                else:
                    reversed_ops.append(
                        EditOperation("replace", source[i - 1], target[j - 1])
                    )
                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            reversed_ops.append(EditOperation("delete", source[i - 1], ""))
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            reversed_ops.append(EditOperation("insert", "", target[j - 1]))
            j -= 1
            continue

        raise RuntimeError("Invalid DP backtracking state.")

    reversed_ops.reverse()
    return reversed_ops


def apply_operations(source: str, operations: list[EditOperation]) -> str:
    """将对齐编辑操作应用于 source，得到转换结果。"""
    out_chars: list[str] = []
    src_idx = 0

    for op in operations:
        if op.action == "match":
            if src_idx >= len(source) or source[src_idx] != op.src_char:
                raise AssertionError("match operation mismatch while replaying operations")
            out_chars.append(op.src_char)
            src_idx += 1
        elif op.action == "replace":
            if src_idx >= len(source) or source[src_idx] != op.src_char:
                raise AssertionError("replace operation mismatch while replaying operations")
            out_chars.append(op.dst_char)
            src_idx += 1
        elif op.action == "delete":
            if src_idx >= len(source) or source[src_idx] != op.src_char:
                raise AssertionError("delete operation mismatch while replaying operations")
            src_idx += 1
        elif op.action == "insert":
            out_chars.append(op.dst_char)
        else:
            raise ValueError(f"Unknown operation: {op.action}")

    if src_idx != len(source):
        raise AssertionError("Not all source characters were consumed during replay")

    return "".join(out_chars)


def brute_force_distance(source: str, target: str) -> int:
    """递归+记忆化真值函数，用于小样例对拍。"""

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> int:
        if i == len(source):
            return len(target) - j
        if j == len(target):
            return len(source) - i
        if source[i] == target[j]:
            return solve(i + 1, j + 1)
        return 1 + min(
            solve(i + 1, j),  # delete
            solve(i, j + 1),  # insert
            solve(i + 1, j + 1),  # replace
        )

    return solve(0, 0)


def summarize_operations(operations: list[EditOperation]) -> str:
    """将编辑路径压缩为便于阅读的一行文本。"""
    chunks: list[str] = []
    for op in operations:
        if op.action == "match":
            chunks.append(f"M({op.src_char})")
        elif op.action == "replace":
            chunks.append(f"R({op.src_char}->{op.dst_char})")
        elif op.action == "delete":
            chunks.append(f"D({op.src_char})")
        elif op.action == "insert":
            chunks.append(f"I({op.dst_char})")
        else:
            chunks.append(f"?({op.action})")
    return " ".join(chunks)


def validate_case(source: str, target: str) -> dict[str, object]:
    """执行单样例验证，失败时抛异常。"""
    dp, stats = build_distance_table(source, target)
    distance = dp[len(source)][len(target)]

    operations = backtrack_operations(source, target, dp)
    transformed = apply_operations(source, operations)

    expected = brute_force_distance(source, target)
    edit_steps = sum(1 for op in operations if op.action != "match")

    if transformed != target:
        raise AssertionError(
            f"Replay mismatch: source={source!r}, target={target!r}, transformed={transformed!r}"
        )
    if distance != expected:
        raise AssertionError(
            f"Distance mismatch: source={source!r}, target={target!r}, dp={distance}, brute={expected}"
        )
    if edit_steps != distance:
        raise AssertionError(
            f"Operation-cost mismatch: source={source!r}, target={target!r}, "
            f"distance={distance}, edit_steps={edit_steps}"
        )

    return {
        "source": source,
        "target": target,
        "distance": distance,
        "expected": expected,
        "transformed": transformed,
        "operations": operations,
        "operation_trace": summarize_operations(operations),
        "edit_steps": edit_steps,
        **stats,
    }


def run_demo_samples() -> None:
    """运行内置样例并打印结构化报告。"""
    samples: list[tuple[str, str]] = [
        ("", ""),
        ("kitten", "sitting"),
        ("flaw", "lawn"),
        ("intention", "execution"),
        ("星期三", "星期四"),
        ("算法", "算术"),
        ("abc", ""),
        ("", "abc"),
        ("aaaa", "aaab"),
    ]

    for idx, (source, target) in enumerate(samples, start=1):
        result = validate_case(source, target)
        print("=" * 76)
        print(f"sample #{idx}")
        print(f"source: {result['source']!r}")
        print(f"target: {result['target']!r}")
        print(f"distance: {result['distance']}")
        print(f"expected: {result['expected']}")
        print(f"transformed: {result['transformed']!r}")
        print(f"edit_steps: {result['edit_steps']}")
        print(
            "dp_stats: "
            f"rows={result['rows']}, cols={result['cols']}, "
            f"cells_computed={result['cells_computed']}"
        )
        print(f"operation_trace: {result['operation_trace']}")

    print("=" * 76)
    print("All demo samples passed DP distance, brute-force distance, and replay checks.")


def main() -> None:
    run_demo_samples()


if __name__ == "__main__":
    main()
