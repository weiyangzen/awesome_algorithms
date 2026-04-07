"""Rule-based query optimization minimal runnable MVP.

This demo shows how a tiny rule optimizer rewrites a logical query plan
without using any database optimizer black box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

Row = Dict[str, Any]
Schema = Tuple[str, ...]


@dataclass(frozen=True)
class Predicate:
    """Atomic boolean predicate with explicit column references."""

    name: str
    refs: frozenset[str]

    def eval(self, row: Mapping[str, Any]) -> bool:
        if self.name == "o.amount > 100":
            return float(row["o.amount"]) > 100.0
        if self.name == "o.status == paid":
            return row["o.status"] == "paid"
        if self.name == "c.tier == gold":
            return row["c.tier"] == "gold"
        if self.name == "o.customer_id == c.customer_id":
            return row["o.customer_id"] == row["c.customer_id"]
        raise ValueError(f"Unknown predicate: {self.name}")


P_AMOUNT_GT_100 = Predicate("o.amount > 100", frozenset({"o.amount"}))
P_STATUS_PAID = Predicate("o.status == paid", frozenset({"o.status"}))
P_TIER_GOLD = Predicate("c.tier == gold", frozenset({"c.tier"}))
P_JOIN_KEY = Predicate(
    "o.customer_id == c.customer_id",
    frozenset({"o.customer_id", "c.customer_id"}),
)


@dataclass(frozen=True)
class PlanNode:
    def schema(self) -> Schema:
        raise NotImplementedError


@dataclass(frozen=True)
class Scan(PlanNode):
    table_name: str
    rows: Tuple[Row, ...]

    def schema(self) -> Schema:
        if not self.rows:
            return tuple()
        return tuple(self.rows[0].keys())


@dataclass(frozen=True)
class Filter(PlanNode):
    predicates: Tuple[Predicate, ...]
    child: PlanNode

    def schema(self) -> Schema:
        return self.child.schema()


@dataclass(frozen=True)
class Project(PlanNode):
    columns: Tuple[str, ...]
    child: PlanNode

    def schema(self) -> Schema:
        return self.columns


@dataclass(frozen=True)
class Join(PlanNode):
    predicates: Tuple[Predicate, ...]
    left: PlanNode
    right: PlanNode

    def schema(self) -> Schema:
        return self.left.schema() + self.right.schema()


@dataclass
class RuleStats:
    fired: Dict[str, int]

    def hit(self, rule_name: str, count: int = 1) -> None:
        self.fired[rule_name] = self.fired.get(rule_name, 0) + count


@dataclass
class OptimizationResult:
    plan: PlanNode
    iterations: int
    stats: RuleStats


def dedup_predicates(predicates: Iterable[Predicate]) -> Tuple[Predicate, ...]:
    seen: Set[str] = set()
    result: List[Predicate] = []
    for p in predicates:
        if p.name not in seen:
            result.append(p)
            seen.add(p.name)
    return tuple(result)


def canonicalize(node: PlanNode, stats: RuleStats) -> Tuple[PlanNode, bool]:
    """Canonicalize plan shape and remove obvious redundancy."""
    changed = False

    if isinstance(node, Scan):
        return node, False

    if isinstance(node, Filter):
        new_child, child_changed = canonicalize(node.child, stats)
        changed |= child_changed
        new_preds = dedup_predicates(node.predicates)
        if len(new_preds) != len(node.predicates):
            stats.hit("R-CANON-DEDUP-FILTER")
            changed = True

        # Merge stacked filters.
        if isinstance(new_child, Filter):
            merged = dedup_predicates(new_preds + new_child.predicates)
            stats.hit("R-CANON-MERGE-FILTER")
            return Filter(predicates=merged, child=new_child.child), True

        return Filter(predicates=new_preds, child=new_child), changed

    if isinstance(node, Project):
        new_child, child_changed = canonicalize(node.child, stats)
        changed |= child_changed
        cols = tuple(dict.fromkeys(node.columns))
        if cols != node.columns:
            stats.hit("R-CANON-DEDUP-PROJECT")
            changed = True

        # Remove identity projection.
        if cols == new_child.schema():
            stats.hit("R-CANON-REMOVE-IDENTITY-PROJECT")
            return new_child, True

        # Collapse stacked projects: keep top projection columns.
        if isinstance(new_child, Project):
            stats.hit("R-CANON-COLLAPSE-PROJECT")
            return Project(columns=cols, child=new_child.child), True

        return Project(columns=cols, child=new_child), changed

    if isinstance(node, Join):
        new_left, left_changed = canonicalize(node.left, stats)
        new_right, right_changed = canonicalize(node.right, stats)
        changed |= left_changed or right_changed
        new_preds = dedup_predicates(node.predicates)
        if len(new_preds) != len(node.predicates):
            stats.hit("R-CANON-DEDUP-JOIN-PRED")
            changed = True
        return Join(predicates=new_preds, left=new_left, right=new_right), changed

    raise TypeError(f"Unsupported node type: {type(node)}")


def pushdown_filters(node: PlanNode, stats: RuleStats) -> Tuple[PlanNode, bool]:
    """Apply selection pushdown rules recursively."""
    changed = False

    if isinstance(node, Scan):
        return node, False

    if isinstance(node, Project):
        new_child, child_changed = pushdown_filters(node.child, stats)
        changed |= child_changed
        return Project(columns=node.columns, child=new_child), changed

    if isinstance(node, Join):
        new_left, left_changed = pushdown_filters(node.left, stats)
        new_right, right_changed = pushdown_filters(node.right, stats)
        changed |= left_changed or right_changed
        return Join(predicates=node.predicates, left=new_left, right=new_right), changed

    if isinstance(node, Filter):
        new_child, child_changed = pushdown_filters(node.child, stats)
        changed |= child_changed
        current = Filter(predicates=node.predicates, child=new_child)

        # Rule: Filter over Project can be swapped if predicate only references
        # projected columns.
        if isinstance(current.child, Project):
            project_cols = set(current.child.columns)
            refs = set().union(*(p.refs for p in current.predicates))
            if refs.issubset(project_cols):
                stats.hit("R-PUSHDOWN-FILTER-THROUGH-PROJECT")
                swapped = Project(
                    columns=current.child.columns,
                    child=Filter(predicates=current.predicates, child=current.child.child),
                )
                return swapped, True

        # Rule: Filter over Join gets split into left / right / residual predicates.
        if isinstance(current.child, Join):
            left_schema = set(current.child.left.schema())
            right_schema = set(current.child.right.schema())
            left_preds: List[Predicate] = []
            right_preds: List[Predicate] = []
            residual: List[Predicate] = []

            for pred in current.predicates:
                if pred.refs.issubset(left_schema):
                    left_preds.append(pred)
                elif pred.refs.issubset(right_schema):
                    right_preds.append(pred)
                else:
                    residual.append(pred)

            new_left = current.child.left
            new_right = current.child.right
            did_push = False

            if left_preds:
                stats.hit("R-PUSHDOWN-FILTER-TO-LEFT", len(left_preds))
                new_left = Filter(predicates=tuple(left_preds), child=new_left)
                did_push = True
            if right_preds:
                stats.hit("R-PUSHDOWN-FILTER-TO-RIGHT", len(right_preds))
                new_right = Filter(predicates=tuple(right_preds), child=new_right)
                did_push = True

            if did_push:
                rebuilt_join = Join(
                    predicates=current.child.predicates,
                    left=new_left,
                    right=new_right,
                )
                if residual:
                    return Filter(predicates=tuple(residual), child=rebuilt_join), True
                return rebuilt_join, True

        return current, changed

    raise TypeError(f"Unsupported node type: {type(node)}")


def prune_columns(node: PlanNode, required: Set[str], stats: RuleStats) -> PlanNode:
    """Top-down projection pruning based on required columns."""
    if isinstance(node, Scan):
        table_schema = set(node.schema())
        projected = tuple(col for col in node.schema() if col in required)
        if set(projected) == table_schema:
            return node
        stats.hit("R-PRUNE-SCAN-COLUMNS")
        return Project(columns=projected, child=node)

    if isinstance(node, Filter):
        local_required = set(required)
        for pred in node.predicates:
            local_required.update(pred.refs)
        new_child = prune_columns(node.child, local_required, stats)
        return Filter(predicates=node.predicates, child=new_child)

    if isinstance(node, Project):
        kept = tuple(col for col in node.columns if col in required)
        if not kept:
            kept = node.columns
        new_child = prune_columns(node.child, set(kept), stats)
        if kept != node.columns:
            stats.hit("R-PRUNE-PROJECT-COLUMNS")
        return Project(columns=kept, child=new_child)

    if isinstance(node, Join):
        left_schema = set(node.left.schema())
        right_schema = set(node.right.schema())
        left_required = {c for c in required if c in left_schema}
        right_required = {c for c in required if c in right_schema}
        for pred in node.predicates:
            for col in pred.refs:
                if col in left_schema:
                    left_required.add(col)
                elif col in right_schema:
                    right_required.add(col)
        new_left = prune_columns(node.left, left_required, stats)
        new_right = prune_columns(node.right, right_required, stats)
        return Join(predicates=node.predicates, left=new_left, right=new_right)

    raise TypeError(f"Unsupported node type: {type(node)}")


def optimize_rule_based(plan: PlanNode, output_columns: Sequence[str]) -> OptimizationResult:
    """Run rule-based optimization to a fixpoint."""
    stats = RuleStats(fired={})
    current = plan
    max_iters = 10
    iteration = 0

    while iteration < max_iters:
        iteration += 1
        changed = False

        current, c1 = canonicalize(current, stats)
        changed |= c1

        current, c2 = pushdown_filters(current, stats)
        changed |= c2

        current, c3 = canonicalize(current, stats)
        changed |= c3

        if not changed:
            break

    # One top-down projection pruning pass after filter pushdown converges.
    current = prune_columns(current, set(output_columns), stats)
    current, _ = canonicalize(current, stats)

    return OptimizationResult(plan=current, iterations=iteration, stats=stats)


def execute_plan(node: PlanNode) -> List[Row]:
    if isinstance(node, Scan):
        return [dict(r) for r in node.rows]

    if isinstance(node, Filter):
        child_rows = execute_plan(node.child)
        return [r for r in child_rows if all(pred.eval(r) for pred in node.predicates)]

    if isinstance(node, Project):
        child_rows = execute_plan(node.child)
        cols = node.columns
        return [{c: row.get(c) for c in cols} for row in child_rows]

    if isinstance(node, Join):
        left_rows = execute_plan(node.left)
        right_rows = execute_plan(node.right)
        out: List[Row] = []
        for lrow in left_rows:
            for rrow in right_rows:
                merged = {**lrow, **rrow}
                if all(pred.eval(merged) for pred in node.predicates):
                    out.append(merged)
        return out

    raise TypeError(f"Unsupported node type: {type(node)}")


def format_plan(node: PlanNode, indent: int = 0) -> str:
    pad = "  " * indent

    if isinstance(node, Scan):
        return f"{pad}Scan({node.table_name}, cols={list(node.schema())})"

    if isinstance(node, Filter):
        pred_names = [p.name for p in node.predicates]
        header = f"{pad}Filter({pred_names})"
        return header + "\n" + format_plan(node.child, indent + 1)

    if isinstance(node, Project):
        header = f"{pad}Project({list(node.columns)})"
        return header + "\n" + format_plan(node.child, indent + 1)

    if isinstance(node, Join):
        pred_names = [p.name for p in node.predicates]
        header = f"{pad}Join({pred_names})"
        left_str = format_plan(node.left, indent + 1)
        right_str = format_plan(node.right, indent + 1)
        return f"{header}\n{left_str}\n{right_str}"

    raise TypeError(f"Unsupported node type: {type(node)}")


def summarize_stats(stats: RuleStats) -> str:
    if not stats.fired:
        return "<no rule fired>"
    lines = []
    for name in sorted(stats.fired):
        lines.append(f"{name}: {stats.fired[name]}")
    return "\n".join(lines)


def build_demo_plan() -> Tuple[PlanNode, Tuple[str, ...]]:
    """Construct an intentionally naive logical plan."""
    orders_rows: Tuple[Row, ...] = (
        {
            "o.order_id": 1,
            "o.customer_id": 101,
            "o.amount": 120.0,
            "o.status": "paid",
            "o.extra": "x1",
        },
        {
            "o.order_id": 2,
            "o.customer_id": 102,
            "o.amount": 80.0,
            "o.status": "paid",
            "o.extra": "x2",
        },
        {
            "o.order_id": 3,
            "o.customer_id": 103,
            "o.amount": 300.0,
            "o.status": "cancelled",
            "o.extra": "x3",
        },
        {
            "o.order_id": 4,
            "o.customer_id": 101,
            "o.amount": 230.0,
            "o.status": "paid",
            "o.extra": "x4",
        },
    )

    customers_rows: Tuple[Row, ...] = (
        {
            "c.customer_id": 101,
            "c.name": "Alice",
            "c.tier": "gold",
            "c.region": "north",
        },
        {
            "c.customer_id": 102,
            "c.name": "Bob",
            "c.tier": "silver",
            "c.region": "north",
        },
        {
            "c.customer_id": 103,
            "c.name": "Cathy",
            "c.tier": "gold",
            "c.region": "south",
        },
    )

    output_cols = ("o.order_id", "c.name")

    naive_plan: PlanNode = Project(
        columns=output_cols,
        child=Filter(
            predicates=(P_AMOUNT_GT_100, P_STATUS_PAID, P_TIER_GOLD),
            child=Join(
                predicates=(P_JOIN_KEY,),
                left=Scan(table_name="orders", rows=orders_rows),
                right=Scan(table_name="customers", rows=customers_rows),
            ),
        ),
    )
    return naive_plan, output_cols


def main() -> None:
    naive_plan, output_cols = build_demo_plan()

    print("=== Original Plan ===")
    print(format_plan(naive_plan))

    original_rows = execute_plan(naive_plan)

    result = optimize_rule_based(naive_plan, output_cols)

    print("\n=== Optimized Plan ===")
    print(format_plan(result.plan))

    optimized_rows = execute_plan(result.plan)

    # Semantic equivalence check.
    assert original_rows == optimized_rows

    # Stronger shape assertions: filters should be pushed below join.
    optimized_str = format_plan(result.plan)
    assert "Filter(['o.amount > 100', 'o.status == paid'])" in optimized_str
    assert "Filter(['c.tier == gold'])" in optimized_str

    # Projection pruning should add narrow projects above scans.
    assert "Project(['o.order_id', 'o.customer_id', 'o.amount', 'o.status'])" in optimized_str
    assert "Project(['c.customer_id', 'c.name', 'c.tier'])" in optimized_str

    print("\n=== Rule Fire Stats ===")
    print(summarize_stats(result.stats))

    print("\n=== Query Result Rows ===")
    for row in optimized_rows:
        print(row)

    print("\nAll assertions passed. Rule-based optimizer MVP is working.")


if __name__ == "__main__":
    main()
