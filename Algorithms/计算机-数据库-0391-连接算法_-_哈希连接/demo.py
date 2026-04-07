"""Hash Join minimal runnable MVP.

This demo implements an in-memory equi-hash join directly in Python source code
without relying on a database engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

Row = Mapping[str, Any]
KeySpec = Union[str, Sequence[str]]


@dataclass
class JoinStats:
    build_side: str
    build_rows: int
    probe_rows: int
    hash_buckets: int
    probe_lookups: int
    matched_pairs: int


@dataclass
class JoinResult:
    rows: List[Dict[str, Any]]
    stats: JoinStats


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


def collect_columns(rows: Sequence[Row]) -> Tuple[str, ...]:
    cols = set()
    for row in rows:
        cols.update(row.keys())
    return tuple(sorted(cols))


def choose_build_side(left_rows: Sequence[Row], right_rows: Sequence[Row], build_side: str) -> bool:
    """Return True if build side is left, False if build side is right."""
    allowed = {"auto", "left", "right"}
    if build_side not in allowed:
        raise ValueError(f"Unsupported build_side={build_side!r}. Choose from {sorted(allowed)}")
    if build_side == "left":
        return True
    if build_side == "right":
        return False
    # Auto mode chooses smaller side; on ties build right for deterministic output.
    return len(left_rows) < len(right_rows)


def hash_join(
    left_rows: Sequence[Row],
    right_rows: Sequence[Row],
    left_on: KeySpec,
    right_on: KeySpec,
    join_type: str = "inner",
    build_side: str = "auto",
) -> JoinResult:
    """Perform in-memory equi-hash join.

    Supported join types: inner, left, right, full.
    Supported build_side: auto, left, right.
    """
    allowed_joins = {"inner", "left", "right", "full"}
    if join_type not in allowed_joins:
        raise ValueError(f"Unsupported join_type={join_type!r}. Choose from {sorted(allowed_joins)}")

    left_keys = normalize_key_spec(left_on)
    right_keys = normalize_key_spec(right_on)
    if len(left_keys) != len(right_keys):
        raise ValueError("left_on and right_on must have the same key length.")

    build_is_left = choose_build_side(left_rows, right_rows, build_side)

    if build_is_left:
        build_rows = left_rows
        build_keys = left_keys
        probe_rows = right_rows
        probe_keys = right_keys
        build_side_name = "left"
    else:
        build_rows = right_rows
        build_keys = right_keys
        probe_rows = left_rows
        probe_keys = left_keys
        build_side_name = "right"

    hash_table: Dict[Any, List[int]] = {}
    for build_idx, build_row in enumerate(build_rows):
        key = build_key(build_row, build_keys)
        hash_table.setdefault(key, []).append(build_idx)

    build_matched = [False] * len(build_rows)
    probe_matched = [False] * len(probe_rows)
    result_rows: List[Dict[str, Any]] = []
    probe_lookups = 0
    matched_pairs = 0

    for probe_idx, probe_row in enumerate(probe_rows):
        probe_lookups += 1
        key = build_key(probe_row, probe_keys)
        candidate_ids = hash_table.get(key)
        if not candidate_ids:
            continue

        probe_matched[probe_idx] = True
        for build_idx in candidate_ids:
            build_matched[build_idx] = True
            matched_pairs += 1

            if build_is_left:
                left_row = build_rows[build_idx]
                right_row = probe_row
            else:
                left_row = probe_row
                right_row = build_rows[build_idx]
            result_rows.append({**prefix_row(left_row, "l_"), **prefix_row(right_row, "r_")})

    if build_is_left:
        left_matched = build_matched
        right_matched = probe_matched
    else:
        left_matched = probe_matched
        right_matched = build_matched

    left_cols = collect_columns(left_rows)
    right_cols = collect_columns(right_rows)
    left_null = {f"l_{c}": None for c in left_cols}
    right_null = {f"r_{c}": None for c in right_cols}

    if join_type in {"left", "full"}:
        for idx, left_row in enumerate(left_rows):
            if not left_matched[idx]:
                result_rows.append({**prefix_row(left_row, "l_"), **right_null})

    if join_type in {"right", "full"}:
        for idx, right_row in enumerate(right_rows):
            if not right_matched[idx]:
                result_rows.append({**left_null, **prefix_row(right_row, "r_")})

    stats = JoinStats(
        build_side=build_side_name,
        build_rows=len(build_rows),
        probe_rows=len(probe_rows),
        hash_buckets=len(hash_table),
        probe_lookups=probe_lookups,
        matched_pairs=matched_pairs,
    )
    return JoinResult(rows=result_rows, stats=stats)


def print_result(title: str, result: JoinResult) -> None:
    print(f"\n== {title} ==")
    print(
        f"rows={len(result.rows)}, matched_pairs={result.stats.matched_pairs}, "
        f"build_side={result.stats.build_side}, hash_buckets={result.stats.hash_buckets}, "
        f"probe_lookups={result.stats.probe_lookups}"
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
        {"customer_id": 101, "name": "Alice_backup", "tier": "gold"},
        {"customer_id": 102, "name": "Bob", "tier": "silver"},
        {"customer_id": 104, "name": "Diana", "tier": "silver"},
        {"customer_id": 105, "name": "Eve", "tier": "bronze"},
    ]

    inner = hash_join(left_table, right_table, "customer_id", "customer_id", join_type="inner")
    left = hash_join(left_table, right_table, "customer_id", "customer_id", join_type="left")
    right = hash_join(left_table, right_table, "customer_id", "customer_id", join_type="right")
    full = hash_join(left_table, right_table, "customer_id", "customer_id", join_type="full")

    # customer_id=101: 2 orders x 2 customers = 4
    # customer_id=102: 1 x 1 = 1
    # customer_id=104: 1 x 1 = 1
    expected_inner_pairs = 6
    assert inner.stats.matched_pairs == expected_inner_pairs
    assert len(inner.rows) == expected_inner_pairs

    # Build side is deterministic in auto mode when both sides equal length.
    assert inner.stats.build_side == "right"
    assert inner.stats.probe_lookups == len(left_table)
    assert inner.stats.hash_buckets == 4  # keys: 101, 102, 104, 105

    unmatched_left = [r for r in left.rows if r["r_customer_id"] is None]
    assert len(unmatched_left) == 1
    assert unmatched_left[0]["l_customer_id"] == 999

    unmatched_right = [r for r in right.rows if r["l_order_id"] is None]
    assert len(unmatched_right) == 1
    assert unmatched_right[0]["r_customer_id"] == 105

    assert len(full.rows) == len(inner.rows) + len(unmatched_left) + len(unmatched_right)

    print_result("INNER HASH JOIN", inner)
    print_result("LEFT HASH JOIN", left)
    print_result("RIGHT HASH JOIN", right)
    print_result("FULL HASH JOIN", full)
    print("\nAll assertions passed. Hash Join MVP is working.")


if __name__ == "__main__":
    main()
