"""Nested Loop Join (NLJ) minimal runnable MVP.

This demo implements the join logic directly in Python source code
without using a database engine as a black box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

Row = Mapping[str, Any]
KeySpec = Union[str, Sequence[str]]


@dataclass
class JoinResult:
    rows: List[Dict[str, Any]]
    comparisons: int
    matches: int


def normalize_key_spec(spec: KeySpec) -> Tuple[str, ...]:
    """Normalize key specification into a tuple of column names."""
    if isinstance(spec, str):
        return (spec,)
    cols = tuple(spec)
    if not cols:
        raise ValueError("Key specification cannot be empty.")
    return cols


def build_key(row: Row, cols: Tuple[str, ...]) -> Any:
    """Build join key from one row. Returns scalar for 1-col key, tuple otherwise."""
    if len(cols) == 1:
        return row.get(cols[0])
    return tuple(row.get(c) for c in cols)


def prefix_row(row: Row, prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in row.items()}


def collect_columns(rows: Iterable[Row]) -> Tuple[str, ...]:
    cols = set()
    for row in rows:
        cols.update(row.keys())
    return tuple(sorted(cols))


def nested_loop_join(
    left_rows: Sequence[Row],
    right_rows: Sequence[Row],
    left_on: KeySpec,
    right_on: KeySpec,
    join_type: str = "inner",
) -> JoinResult:
    """Perform classic nested loop join.

    Supported join types: inner, left, right, full.
    """
    allowed = {"inner", "left", "right", "full"}
    if join_type not in allowed:
        raise ValueError(f"Unsupported join_type={join_type!r}. Choose from {sorted(allowed)}")

    left_keys = normalize_key_spec(left_on)
    right_keys = normalize_key_spec(right_on)
    if len(left_keys) != len(right_keys):
        raise ValueError("left_on and right_on must have the same key length.")

    left_cols = collect_columns(left_rows)
    right_cols = collect_columns(right_rows)
    left_null = {f"l_{c}": None for c in left_cols}
    right_null = {f"r_{c}": None for c in right_cols}

    result_rows: List[Dict[str, Any]] = []
    comparisons = 0
    matches = 0
    right_matched = [False] * len(right_rows)

    for left_row in left_rows:
        left_key = build_key(left_row, left_keys)
        left_prefixed = prefix_row(left_row, "l_")
        hit = False

        for idx, right_row in enumerate(right_rows):
            comparisons += 1
            right_key = build_key(right_row, right_keys)
            if left_key == right_key:
                hit = True
                matches += 1
                right_matched[idx] = True
                result_rows.append({**left_prefixed, **prefix_row(right_row, "r_")})

        if not hit and join_type in {"left", "full"}:
            result_rows.append({**left_prefixed, **right_null})

    if join_type in {"right", "full"}:
        for idx, right_row in enumerate(right_rows):
            if not right_matched[idx]:
                result_rows.append({**left_null, **prefix_row(right_row, "r_")})

    return JoinResult(rows=result_rows, comparisons=comparisons, matches=matches)


def print_result(title: str, result: JoinResult) -> None:
    print(f"\n== {title} ==")
    print(
        f"rows={len(result.rows)}, comparisons={result.comparisons}, "
        f"matches={result.matches}"
    )
    for row in result.rows:
        print(row)


def main() -> None:
    left_table = [
        {"order_id": 1, "customer_id": 101, "amount": 120.0},
        {"order_id": 2, "customer_id": 102, "amount": 50.0},
        {"order_id": 3, "customer_id": 999, "amount": 10.0},
        {"order_id": 4, "customer_id": 101, "amount": 220.0},
        {"order_id": 5, "customer_id": 104, "amount": 75.0},
    ]

    right_table = [
        {"customer_id": 101, "name": "Alice", "tier": "gold"},
        {"customer_id": 102, "name": "Bob", "tier": "silver"},
        {"customer_id": 104, "name": "Diana", "tier": "silver"},
        {"customer_id": 105, "name": "Eve", "tier": "bronze"},
    ]

    inner = nested_loop_join(left_table, right_table, "customer_id", "customer_id", "inner")
    left = nested_loop_join(left_table, right_table, "customer_id", "customer_id", "left")
    right = nested_loop_join(left_table, right_table, "customer_id", "customer_id", "right")
    full = nested_loop_join(left_table, right_table, "customer_id", "customer_id", "full")

    # Each left row compares with each right row in classic NLJ.
    expected_comparisons = len(left_table) * len(right_table)
    assert inner.comparisons == expected_comparisons
    assert left.comparisons == expected_comparisons
    assert right.comparisons == expected_comparisons
    assert full.comparisons == expected_comparisons

    # Match counts: customer_id 101 has 2 orders + 102 has 1 + 104 has 1 = 4
    assert inner.matches == 4
    assert len(inner.rows) == 4

    # Left join includes the unmatched left row (customer_id=999)
    unmatched_left = [r for r in left.rows if r["r_customer_id"] is None]
    assert len(unmatched_left) == 1

    # Right join includes the unmatched right row (customer_id=105)
    unmatched_right = [r for r in right.rows if r["l_order_id"] is None]
    assert len(unmatched_right) == 1

    # Full join = inner matches + unmatched left + unmatched right
    assert len(full.rows) == len(inner.rows) + len(unmatched_left) + len(unmatched_right)

    print_result("INNER JOIN", inner)
    print_result("LEFT JOIN", left)
    print_result("RIGHT JOIN", right)
    print_result("FULL JOIN", full)

    print("\nAll assertions passed. Nested Loop Join MVP is working.")


if __name__ == "__main__":
    main()
