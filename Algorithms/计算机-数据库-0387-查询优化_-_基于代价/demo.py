"""Cost-based query optimization (CBO) minimal runnable MVP.

The demo includes:
1) table statistics collection,
2) access path costing (SeqScan vs IndexScan),
3) Selinger-style DP join-order search,
4) physical join method choice (NestedLoop/HashJoin/MergeJoin),
5) semantic equivalence check against a naive execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd

Row = Dict[str, Any]


@dataclass(frozen=True)
class Predicate:
    """Single-table filter predicate."""

    name: str
    table: str
    refs: FrozenSet[str]

    def eval(self, row: Mapping[str, Any]) -> bool:
        if self.name == "o.status == paid":
            return row["o.status"] == "paid"
        if self.name == "o.amount > 120":
            return float(row["o.amount"]) > 120.0
        if self.name == "c.tier == gold":
            return row["c.tier"] == "gold"
        if self.name == "l.qty >= 2":
            return int(row["l.qty"]) >= 2
        raise ValueError(f"Unknown predicate: {self.name}")


@dataclass(frozen=True)
class JoinCondition:
    left_table: str
    left_col: str
    right_table: str
    right_col: str
    name: str


@dataclass(frozen=True)
class TableData:
    name: str
    rows: Tuple[Row, ...]
    indexed_cols: FrozenSet[str]


@dataclass(frozen=True)
class TableStats:
    row_count: int
    ndv: Dict[str, int]
    predicate_selectivity: Dict[str, float]


@dataclass(frozen=True)
class AccessPlan:
    table: str
    access_method: str
    predicates: Tuple[Predicate, ...]
    est_rows: float
    cost: float
    ndv_est: Dict[str, float]

    @property
    def tables(self) -> FrozenSet[str]:
        return frozenset({self.table})


@dataclass(frozen=True)
class JoinPlan:
    left: "PlanNode"
    right: "PlanNode"
    condition: JoinCondition
    join_method: str
    est_rows: float
    cost: float
    ndv_est: Dict[str, float]

    @property
    def tables(self) -> FrozenSet[str]:
        return self.left.tables | self.right.tables


PlanNode = Union[AccessPlan, JoinPlan]


@dataclass(frozen=True)
class QuerySpec:
    table_predicates: Dict[str, Tuple[Predicate, ...]]
    joins: Tuple[JoinCondition, ...]
    output_columns: Tuple[str, ...]


@dataclass
class CBOResult:
    best_plan: PlanNode
    dp_table: Dict[FrozenSet[str], PlanNode]
    candidate_count: int


def build_demo_data() -> Dict[str, TableData]:
    tables: Dict[str, TableData] = {
        "o": TableData(
            name="orders",
            indexed_cols=frozenset({"o.status", "o.customer_id"}),
            rows=(
                {"o.order_id": 1, "o.customer_id": 101, "o.amount": 180.0, "o.status": "paid"},
                {"o.order_id": 2, "o.customer_id": 102, "o.amount": 95.0, "o.status": "paid"},
                {"o.order_id": 3, "o.customer_id": 103, "o.amount": 260.0, "o.status": "paid"},
                {"o.order_id": 4, "o.customer_id": 101, "o.amount": 75.0, "o.status": "cancelled"},
                {"o.order_id": 5, "o.customer_id": 104, "o.amount": 410.0, "o.status": "paid"},
                {"o.order_id": 6, "o.customer_id": 105, "o.amount": 150.0, "o.status": "paid"},
                {"o.order_id": 7, "o.customer_id": 106, "o.amount": 85.0, "o.status": "paid"},
                {"o.order_id": 8, "o.customer_id": 103, "o.amount": 133.0, "o.status": "paid"},
            ),
        ),
        "c": TableData(
            name="customers",
            indexed_cols=frozenset({"c.customer_id", "c.tier"}),
            rows=(
                {"c.customer_id": 101, "c.name": "Alice", "c.tier": "gold"},
                {"c.customer_id": 102, "c.name": "Bob", "c.tier": "silver"},
                {"c.customer_id": 103, "c.name": "Cathy", "c.tier": "gold"},
                {"c.customer_id": 104, "c.name": "Dan", "c.tier": "gold"},
                {"c.customer_id": 105, "c.name": "Eve", "c.tier": "bronze"},
                {"c.customer_id": 106, "c.name": "Frank", "c.tier": "gold"},
            ),
        ),
        "l": TableData(
            name="lineitems",
            indexed_cols=frozenset({"l.order_id"}),
            rows=(
                {"l.order_id": 1, "l.sku": "A1", "l.qty": 2, "l.price": 35.0},
                {"l.order_id": 1, "l.sku": "B2", "l.qty": 1, "l.price": 20.0},
                {"l.order_id": 2, "l.sku": "A1", "l.qty": 3, "l.price": 35.0},
                {"l.order_id": 3, "l.sku": "C3", "l.qty": 2, "l.price": 70.0},
                {"l.order_id": 3, "l.sku": "D4", "l.qty": 4, "l.price": 25.0},
                {"l.order_id": 5, "l.sku": "E5", "l.qty": 2, "l.price": 60.0},
                {"l.order_id": 6, "l.sku": "F6", "l.qty": 1, "l.price": 90.0},
                {"l.order_id": 8, "l.sku": "G7", "l.qty": 2, "l.price": 50.0},
                {"l.order_id": 8, "l.sku": "H8", "l.qty": 1, "l.price": 45.0},
            ),
        ),
    }
    return tables


def build_query_spec() -> QuerySpec:
    p_order_paid = Predicate(
        name="o.status == paid",
        table="o",
        refs=frozenset({"o.status"}),
    )
    p_order_amount = Predicate(
        name="o.amount > 120",
        table="o",
        refs=frozenset({"o.amount"}),
    )
    p_customer_gold = Predicate(
        name="c.tier == gold",
        table="c",
        refs=frozenset({"c.tier"}),
    )
    p_line_qty = Predicate(
        name="l.qty >= 2",
        table="l",
        refs=frozenset({"l.qty"}),
    )

    joins = (
        JoinCondition(
            left_table="o",
            left_col="o.customer_id",
            right_table="c",
            right_col="c.customer_id",
            name="o.customer_id = c.customer_id",
        ),
        JoinCondition(
            left_table="o",
            left_col="o.order_id",
            right_table="l",
            right_col="l.order_id",
            name="o.order_id = l.order_id",
        ),
    )

    return QuerySpec(
        table_predicates={
            "o": (p_order_paid, p_order_amount),
            "c": (p_customer_gold,),
            "l": (p_line_qty,),
        },
        joins=joins,
        output_columns=("o.order_id", "c.name", "l.sku", "l.qty", "l.price"),
    )


def gather_table_stats(tables: Dict[str, TableData], spec: QuerySpec) -> Dict[str, TableStats]:
    out: Dict[str, TableStats] = {}
    for alias, table in tables.items():
        df = pd.DataFrame(list(table.rows))
        row_count = int(df.shape[0])
        ndv = {col: int(df[col].nunique(dropna=False)) for col in df.columns}
        selectivity: Dict[str, float] = {}
        for pred in spec.table_predicates.get(alias, tuple()):
            matched = sum(1 for r in table.rows if pred.eval(r))
            selectivity[pred.name] = matched / row_count if row_count else 1.0
        out[alias] = TableStats(
            row_count=row_count,
            ndv=ndv,
            predicate_selectivity=selectivity,
        )
    return out


def estimate_base_rows(stats: TableStats, predicates: Sequence[Predicate]) -> float:
    if stats.row_count == 0:
        return 0.0
    sel = 1.0
    for pred in predicates:
        sel *= stats.predicate_selectivity.get(pred.name, 1.0)
    est = float(stats.row_count) * sel
    return max(1.0, est)


def choose_access_plan(
    alias: str,
    tables: Dict[str, TableData],
    stats: Dict[str, TableStats],
    spec: QuerySpec,
) -> AccessPlan:
    table = tables[alias]
    table_stats = stats[alias]
    predicates = spec.table_predicates.get(alias, tuple())
    est_rows = estimate_base_rows(table_stats, predicates)

    # SeqScan cost: linear scan + predicate evaluation.
    seq_cost = float(table_stats.row_count) + 0.3 * est_rows

    # IndexScan cost: index probe startup + random fetch by estimated rows.
    indexed_pred = any(
        any(ref in table.indexed_cols for ref in pred.refs)
        for pred in predicates
    )
    if indexed_pred:
        index_cost = float(np.log2(table_stats.row_count + 1.0) * 2.5 + est_rows * 1.6)
    else:
        index_cost = float("inf")

    if index_cost < seq_cost:
        method = "IndexScan"
        final_cost = index_cost
    else:
        method = "SeqScan"
        final_cost = seq_cost

    ndv_est = {
        col: float(min(v, max(1.0, np.floor(est_rows))))
        for col, v in table_stats.ndv.items()
    }
    return AccessPlan(
        table=alias,
        access_method=method,
        predicates=tuple(predicates),
        est_rows=est_rows,
        cost=final_cost,
        ndv_est=ndv_est,
    )


def find_join_condition(left_tables: FrozenSet[str], right_tables: FrozenSet[str], joins: Sequence[JoinCondition]) -> JoinCondition | None:
    for cond in joins:
        if cond.left_table in left_tables and cond.right_table in right_tables:
            return cond
        if cond.left_table in right_tables and cond.right_table in left_tables:
            return JoinCondition(
                left_table=cond.right_table,
                left_col=cond.right_col,
                right_table=cond.left_table,
                right_col=cond.left_col,
                name=cond.name,
            )
    return None


def estimate_join_cardinality(left: PlanNode, right: PlanNode, cond: JoinCondition) -> float:
    left_ndv = max(1.0, left.ndv_est.get(cond.left_col, left.est_rows))
    right_ndv = max(1.0, right.ndv_est.get(cond.right_col, right.est_rows))
    denom = max(left_ndv, right_ndv)
    est = (left.est_rows * right.est_rows) / denom
    return max(1.0, est)


def merge_ndv_maps(left: PlanNode, right: PlanNode, cond: JoinCondition, est_rows: float) -> Dict[str, float]:
    merged = dict(left.ndv_est)
    merged.update(right.ndv_est)
    join_ndv = min(
        left.ndv_est.get(cond.left_col, left.est_rows),
        right.ndv_est.get(cond.right_col, right.est_rows),
    )
    join_ndv = max(1.0, min(join_ndv, est_rows))
    merged[cond.left_col] = join_ndv
    merged[cond.right_col] = join_ndv
    for key in list(merged):
        merged[key] = max(1.0, min(merged[key], est_rows))
    return merged


def choose_join_plan(left: PlanNode, right: PlanNode, cond: JoinCondition) -> JoinPlan:
    est_rows = estimate_join_cardinality(left, right, cond)

    nested_loop_cost = left.cost + right.cost + left.est_rows * right.est_rows
    hash_join_cost = left.cost + right.cost + 1.2 * (left.est_rows + right.est_rows)
    merge_join_cost = (
        left.cost
        + right.cost
        + left.est_rows * np.log2(left.est_rows + 1.0)
        + right.est_rows * np.log2(right.est_rows + 1.0)
        + 0.8 * (left.est_rows + right.est_rows)
    )

    candidates = {
        "NestedLoop": float(nested_loop_cost),
        "HashJoin": float(hash_join_cost),
        "MergeJoin": float(merge_join_cost),
    }
    method = min(candidates, key=candidates.get)
    cost = candidates[method]
    ndv_est = merge_ndv_maps(left, right, cond, est_rows)

    return JoinPlan(
        left=left,
        right=right,
        condition=cond,
        join_method=method,
        est_rows=est_rows,
        cost=cost,
        ndv_est=ndv_est,
    )


def optimize_cost_based(tables: Dict[str, TableData], spec: QuerySpec, stats: Dict[str, TableStats]) -> CBOResult:
    aliases = sorted(tables.keys())
    dp: Dict[FrozenSet[str], PlanNode] = {}
    candidate_count = 0

    # 1-table subsets: choose best access path.
    for a in aliases:
        key = frozenset({a})
        dp[key] = choose_access_plan(a, tables, stats, spec)
        candidate_count += 1

    # Larger subsets: DP on subset size.
    for size in range(2, len(aliases) + 1):
        for subset_tuple in combinations(aliases, size):
            subset = frozenset(subset_tuple)
            best_plan: PlanNode | None = None

            for left_size in range(1, size):
                for left_tuple in combinations(subset_tuple, left_size):
                    left_set = frozenset(left_tuple)
                    right_set = subset - left_set
                    if not right_set:
                        continue
                    if left_set not in dp or right_set not in dp:
                        continue

                    cond = find_join_condition(left_set, right_set, spec.joins)
                    if cond is None:
                        continue

                    candidate = choose_join_plan(dp[left_set], dp[right_set], cond)
                    candidate_count += 1
                    if best_plan is None or candidate.cost < best_plan.cost:
                        best_plan = candidate

            if best_plan is not None:
                dp[subset] = best_plan

    full_set = frozenset(aliases)
    if full_set not in dp:
        raise ValueError(f"No valid full join plan for subset={full_set}")
    return CBOResult(best_plan=dp[full_set], dp_table=dp, candidate_count=candidate_count)


def execute_access(plan: AccessPlan, tables: Dict[str, TableData]) -> List[Row]:
    rows = [dict(r) for r in tables[plan.table].rows]
    for pred in plan.predicates:
        rows = [r for r in rows if pred.eval(r)]
    return rows


def join_nested_loop(left_rows: Sequence[Row], right_rows: Sequence[Row], cond: JoinCondition) -> List[Row]:
    out: List[Row] = []
    for lrow in left_rows:
        lv = lrow[cond.left_col]
        for rrow in right_rows:
            if lv == rrow[cond.right_col]:
                out.append({**lrow, **rrow})
    return out


def join_hash(left_rows: Sequence[Row], right_rows: Sequence[Row], cond: JoinCondition) -> List[Row]:
    if len(left_rows) <= len(right_rows):
        build_rows = left_rows
        probe_rows = right_rows
        build_key = cond.left_col
        probe_key = cond.right_col
        left_build = True
    else:
        build_rows = right_rows
        probe_rows = left_rows
        build_key = cond.right_col
        probe_key = cond.left_col
        left_build = False

    buckets: Dict[Any, List[Row]] = {}
    for row in build_rows:
        buckets.setdefault(row[build_key], []).append(row)

    out: List[Row] = []
    for prow in probe_rows:
        matches = buckets.get(prow[probe_key], [])
        for brow in matches:
            out.append({**brow, **prow} if left_build else {**prow, **brow})
    return out


def join_merge(left_rows: Sequence[Row], right_rows: Sequence[Row], cond: JoinCondition) -> List[Row]:
    left_sorted = sorted(left_rows, key=lambda r: r[cond.left_col])
    right_sorted = sorted(right_rows, key=lambda r: r[cond.right_col])
    i, j = 0, 0
    out: List[Row] = []

    while i < len(left_sorted) and j < len(right_sorted):
        left_key = left_sorted[i][cond.left_col]
        right_key = right_sorted[j][cond.right_col]
        if left_key < right_key:
            i += 1
            continue
        if left_key > right_key:
            j += 1
            continue

        li = i
        rj = j
        left_group: List[Row] = []
        right_group: List[Row] = []
        while li < len(left_sorted) and left_sorted[li][cond.left_col] == left_key:
            left_group.append(left_sorted[li])
            li += 1
        while rj < len(right_sorted) and right_sorted[rj][cond.right_col] == right_key:
            right_group.append(right_sorted[rj])
            rj += 1

        for lrow in left_group:
            for rrow in right_group:
                out.append({**lrow, **rrow})

        i = li
        j = rj

    return out


def execute_plan(node: PlanNode, tables: Dict[str, TableData]) -> List[Row]:
    if isinstance(node, AccessPlan):
        return execute_access(node, tables)

    left_rows = execute_plan(node.left, tables)
    right_rows = execute_plan(node.right, tables)
    if node.join_method == "NestedLoop":
        return join_nested_loop(left_rows, right_rows, node.condition)
    if node.join_method == "HashJoin":
        return join_hash(left_rows, right_rows, node.condition)
    if node.join_method == "MergeJoin":
        return join_merge(left_rows, right_rows, node.condition)
    raise ValueError(f"Unsupported join method: {node.join_method}")


def execute_naive(tables: Dict[str, TableData], spec: QuerySpec) -> List[Row]:
    """Reference execution in fixed order: ((orders join customers) join lineitems)."""
    o_rows = [dict(r) for r in tables["o"].rows]
    c_rows = [dict(r) for r in tables["c"].rows]
    l_rows = [dict(r) for r in tables["l"].rows]

    for pred in spec.table_predicates["o"]:
        o_rows = [r for r in o_rows if pred.eval(r)]
    for pred in spec.table_predicates["c"]:
        c_rows = [r for r in c_rows if pred.eval(r)]
    for pred in spec.table_predicates["l"]:
        l_rows = [r for r in l_rows if pred.eval(r)]

    oc = []
    for o in o_rows:
        for c in c_rows:
            if o["o.customer_id"] == c["c.customer_id"]:
                oc.append({**o, **c})

    out = []
    for row in oc:
        for l in l_rows:
            if row["o.order_id"] == l["l.order_id"]:
                out.append({**row, **l})
    return out


def project_rows(rows: Iterable[Row], columns: Sequence[str]) -> List[Row]:
    return [{c: row[c] for c in columns} for row in rows]


def canonical_rows(rows: Iterable[Row]) -> List[Tuple[Tuple[str, Any], ...]]:
    return sorted(tuple(sorted(r.items())) for r in rows)


def format_plan(node: PlanNode, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(node, AccessPlan):
        pred_names = [p.name for p in node.predicates]
        return (
            f"{pad}Access(table={node.table}, method={node.access_method}, "
            f"est_rows={node.est_rows:.2f}, cost={node.cost:.2f}, preds={pred_names})"
        )

    header = (
        f"{pad}{node.join_method}(on={node.condition.name}, "
        f"est_rows={node.est_rows:.2f}, cost={node.cost:.2f})"
    )
    left = format_plan(node.left, indent + 1)
    right = format_plan(node.right, indent + 1)
    return f"{header}\n{left}\n{right}"


def summarize_stats(stats: Dict[str, TableStats]) -> str:
    lines = []
    for alias in sorted(stats):
        ts = stats[alias]
        lines.append(f"[{alias}] rows={ts.row_count}, ndv={ts.ndv}")
        if ts.predicate_selectivity:
            sel_txt = ", ".join(
                f"{k}: {v:.3f}" for k, v in sorted(ts.predicate_selectivity.items())
            )
            lines.append(f"  selectivity -> {sel_txt}")
    return "\n".join(lines)


def main() -> None:
    tables = build_demo_data()
    spec = build_query_spec()
    stats = gather_table_stats(tables, spec)

    print("=== Table Stats ===")
    print(summarize_stats(stats))

    cbo_result = optimize_cost_based(tables, spec, stats)
    best_plan = cbo_result.best_plan

    print("\n=== Chosen Physical Plan (CBO) ===")
    print(format_plan(best_plan))
    print(f"\nDP entries: {len(cbo_result.dp_table)}, candidates evaluated: {cbo_result.candidate_count}")

    naive_rows = project_rows(execute_naive(tables, spec), spec.output_columns)
    cbo_rows = project_rows(execute_plan(best_plan, tables), spec.output_columns)

    # Semantic equivalence check.
    assert canonical_rows(naive_rows) == canonical_rows(cbo_rows)

    # Stronger sanity checks: optimizer should pick at least one non-NLJ decision.
    plan_str = format_plan(best_plan)
    assert "HashJoin" in plan_str or "MergeJoin" in plan_str

    print("\n=== Query Result Rows ===")
    for row in sorted(cbo_rows, key=lambda r: (r["o.order_id"], r["l.sku"])):
        print(row)

    print("\nAll assertions passed. Cost-based optimizer MVP is working.")


if __name__ == "__main__":
    main()
