"""Sort-Merge Join minimal runnable MVP.

This demo implements sort-merge join directly in Python source code
instead of delegating the core algorithm to a database engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

Row = Mapping[str, Any]
KeySpec = Union[str, Sequence[str]]
CanonicalAtom = Tuple[int, Any]
CanonicalKey = Tuple[CanonicalAtom, ...]


@dataclass
class JoinResult:
    rows: List[Dict[str, Any]]
    key_comparisons: int
    matched_pairs: int


def normalize_key_spec(spec: KeySpec) -> Tuple[str, ...]:
    """Normalize key specification into a non-empty tuple."""
    if isinstance(spec, str):
        return (spec,)
    cols = tuple(spec)
    if not cols:
        raise ValueError("Key specification cannot be empty.")
    return cols


def extract_key(row: Row, cols: Tuple[str, ...]) -> Tuple[Any, ...]:
    """Extract raw key values from one row as a tuple."""
    return tuple(row.get(c) for c in cols)


def canonicalize_value(value: Any) -> CanonicalAtom:
    """Map arbitrary values to an orderable representation.

    The rank component ensures cross-type comparison safety.
    """
    if value is None:
        return (0, 0)
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, (int, float)):
        return (2, float(value))
    if isinstance(value, str):
        return (3, value)
    return (4, repr(value))


def canonicalize_key(raw_key: Tuple[Any, ...]) -> CanonicalKey:
    return tuple(canonicalize_value(v) for v in raw_key)


def collect_columns(rows: Iterable[Row]) -> Tuple[str, ...]:
    cols = set()
    for row in rows:
        cols.update(row.keys())
    return tuple(sorted(cols))


def prefix_row(row: Row, prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in row.items()}


def prepare_sorted_rows(rows: Sequence[Row], key_cols: Tuple[str, ...]) -> List[Tuple[Row, CanonicalKey]]:
    prepared = [(row, canonicalize_key(extract_key(row, key_cols))) for row in rows]
    prepared.sort(key=lambda item: item[1])
    return prepared


def sort_merge_join(
    left_rows: Sequence[Row],
    right_rows: Sequence[Row],
    left_on: KeySpec,
    right_on: KeySpec,
    join_type: str = "inner",
) -> JoinResult:
    """Perform sort-merge join.

    Supported join types: inner, left, right, full.
    """
    allowed = {"inner", "left", "right", "full"}
    if join_type not in allowed:
        raise ValueError(f"Unsupported join_type={join_type!r}. Choose from {sorted(allowed)}")

    left_cols = normalize_key_spec(left_on)
    right_cols = normalize_key_spec(right_on)
    if len(left_cols) != len(right_cols):
        raise ValueError("left_on and right_on must have the same key length.")

    left_prepared = prepare_sorted_rows(left_rows, left_cols)
    right_prepared = prepare_sorted_rows(right_rows, right_cols)

    left_all_cols = collect_columns(left_rows)
    right_all_cols = collect_columns(right_rows)
    left_null = {f"l_{c}": None for c in left_all_cols}
    right_null = {f"r_{c}": None for c in right_all_cols}

    result_rows: List[Dict[str, Any]] = []
    key_comparisons = 0
    matched_pairs = 0

    i = 0
    j = 0
    left_n = len(left_prepared)
    right_n = len(right_prepared)

    while i < left_n and j < right_n:
        left_key = left_prepared[i][1]
        right_key = right_prepared[j][1]
        key_comparisons += 1

        if left_key < right_key:
            if join_type in {"left", "full"}:
                result_rows.append({**prefix_row(left_prepared[i][0], "l_"), **right_null})
            i += 1
            continue

        if left_key > right_key:
            if join_type in {"right", "full"}:
                result_rows.append({**left_null, **prefix_row(right_prepared[j][0], "r_")})
            j += 1
            continue

        # Equal key: consume one key-group from each side and emit Cartesian product.
        i_end = i
        while i_end < left_n and left_prepared[i_end][1] == left_key:
            i_end += 1

        j_end = j
        while j_end < right_n and right_prepared[j_end][1] == right_key:
            j_end += 1

        for li in range(i, i_end):
            left_prefixed = prefix_row(left_prepared[li][0], "l_")
            for rj in range(j, j_end):
                result_rows.append({**left_prefixed, **prefix_row(right_prepared[rj][0], "r_")})
                matched_pairs += 1

        i = i_end
        j = j_end

    if join_type in {"left", "full"}:
        while i < left_n:
            result_rows.append({**prefix_row(left_prepared[i][0], "l_"), **right_null})
            i += 1

    if join_type in {"right", "full"}:
        while j < right_n:
            result_rows.append({**left_null, **prefix_row(right_prepared[j][0], "r_")})
            j += 1

    return JoinResult(
        rows=result_rows,
        key_comparisons=key_comparisons,
        matched_pairs=matched_pairs,
    )


def print_result(title: str, result: JoinResult) -> None:
    print(f"\n== {title} ==")
    print(
        f"rows={len(result.rows)}, key_comparisons={result.key_comparisons}, "
        f"matched_pairs={result.matched_pairs}"
    )
    for row in result.rows:
        print(row)


def main() -> None:
    # Intentionally unsorted input with duplicates and unmatched keys.
    left_table = [
        {"order_id": 10, "customer_id": 102, "amount": 50.0},
        {"order_id": 11, "customer_id": 101, "amount": 120.0},
        {"order_id": 12, "customer_id": 999, "amount": 10.0},
        {"order_id": 13, "customer_id": 101, "amount": 220.0},
        {"order_id": 14, "customer_id": 103, "amount": 88.0},
    ]

    right_table = [
        {"customer_id": 101, "name": "Alice", "tier": "gold"},
        {"customer_id": 102, "name": "Bob-A", "tier": "silver"},
        {"customer_id": 102, "name": "Bob-B", "tier": "silver"},
        {"customer_id": 104, "name": "Diana", "tier": "bronze"},
    ]

    inner = sort_merge_join(left_table, right_table, "customer_id", "customer_id", "inner")
    left = sort_merge_join(left_table, right_table, "customer_id", "customer_id", "left")
    right = sort_merge_join(left_table, right_table, "customer_id", "customer_id", "right")
    full = sort_merge_join(left_table, right_table, "customer_id", "customer_id", "full")

    # Expected match multiplicities:
    # customer_id=101 -> 2 x 1 = 2
    # customer_id=102 -> 1 x 2 = 2
    assert inner.matched_pairs == 4
    assert len(inner.rows) == 4

    # Left unmatched keys: 103 and 999.
    left_unmatched = [r for r in left.rows if r["r_customer_id"] is None]
    assert len(left_unmatched) == 2

    # Right unmatched key: 104.
    right_unmatched = [r for r in right.rows if r["l_order_id"] is None]
    assert len(right_unmatched) == 1

    assert len(left.rows) == 6
    assert len(right.rows) == 5
    assert len(full.rows) == 7

    # Sort-merge join should compare key-groups, not all row pairs.
    assert inner.key_comparisons == 4
    assert left.key_comparisons == 4
    assert right.key_comparisons == 4
    assert full.key_comparisons == 4

    print_result("INNER JOIN", inner)
    print_result("LEFT JOIN", left)
    print_result("RIGHT JOIN", right)
    print_result("FULL JOIN", full)

    print("\nAll assertions passed. Sort-Merge Join MVP is working.")


if __name__ == "__main__":
    main()
