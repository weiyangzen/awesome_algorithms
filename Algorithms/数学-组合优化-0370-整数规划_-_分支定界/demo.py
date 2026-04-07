"""Integer programming branch-and-bound MVP.

This script solves a binary integer linear program (0-1 ILP):

    max c^T x
    s.t. A x <= b
         x_i in {0, 1}

The branch-and-bound search logic is handwritten.
LP relaxation is used as a node upper bound when SciPy is available.
If SciPy is unavailable, the solver falls back to a conservative (looser)
upper bound so the script remains runnable without interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import product
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linprog
except Exception:  # pragma: no cover - runtime fallback
    linprog = None

EPS = 1e-12
INT_TOL = 1e-9


@dataclass
class Node:
    """State of one branch-and-bound node."""

    node_id: int
    depth: int
    lb: np.ndarray
    ub: np.ndarray
    upper_bound: float
    relax_x: Optional[np.ndarray]
    bound_source: str


def validate_binary_ilp(c: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
    """Validate model shape and MVP assumptions."""
    if c.ndim != 1:
        raise ValueError("c must be a 1D vector")
    if a.ndim != 2:
        raise ValueError("A must be a 2D matrix")
    if b.ndim != 1:
        raise ValueError("b must be a 1D vector")
    if a.shape[1] != c.shape[0]:
        raise ValueError("A.shape[1] must equal len(c)")
    if a.shape[0] != b.shape[0]:
        raise ValueError("A.shape[0] must equal len(b)")

    if not (np.isfinite(c).all() and np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("c/A/b must contain only finite numbers")

    # MVP assumption for fast partial-feasibility checks in fallback mode.
    if np.any(a < -EPS):
        raise ValueError("This MVP requires A >= 0 for all constraints")


def is_integral_vector(x: np.ndarray, tol: float = INT_TOL) -> bool:
    return bool(np.all(np.abs(x - np.rint(x)) <= tol))


def partial_infeasible(a: np.ndarray, b: np.ndarray, lb: np.ndarray, tol: float = EPS) -> bool:
    """For A>=0 and x>=lb, A@lb is a lower bound of all feasible lhs values."""
    min_lhs = a @ lb
    return bool(np.any(min_lhs - b > tol))


def solve_lp_relaxation(
    c: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> Optional[Tuple[float, np.ndarray]]:
    """Solve LP relaxation for one node and return (upper_bound, x_relax)."""
    if linprog is None:
        return None

    bounds = [(float(lb[i]), float(ub[i])) for i in range(c.shape[0])]
    result = linprog(
        c=-c,  # convert max c^T x to min -c^T x
        A_ub=a,
        b_ub=b,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        # infeasible node in branch-and-bound terms
        if result.status in (2, 4):
            return None
        # unbounded should not happen with 0<=x<=1 bounds
        if result.status == 3:
            raise RuntimeError("LP relaxation is unexpectedly unbounded")
        return None

    x_relax = np.asarray(result.x, dtype=float)
    upper_bound = float(c @ x_relax)
    return upper_bound, x_relax


def build_node(
    node_id: int,
    depth: int,
    lb: np.ndarray,
    ub: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    use_lp_bounds: bool,
) -> Optional[Node]:
    """Create a node, computing its upper bound; return None if infeasible."""
    if np.any(lb - ub > EPS):
        return None

    if partial_infeasible(a=a, b=b, lb=lb):
        return None

    if use_lp_bounds:
        lp = solve_lp_relaxation(c=c, a=a, b=b, lb=lb, ub=ub)
        if lp is None:
            return None
        upper_bound, relax_x = lp
        source = "lp"
    else:
        # Conservative bound when LP solver is unavailable:
        # ignore A x <= b and use sign-based optimistic assignment.
        relax_x = np.where(c >= 0.0, ub, lb).astype(float)
        upper_bound = float(c @ relax_x)
        source = "loose"

    return Node(
        node_id=node_id,
        depth=depth,
        lb=lb,
        ub=ub,
        upper_bound=upper_bound,
        relax_x=relax_x,
        bound_source=source,
    )


def choose_branch_variable(node: Node, c: np.ndarray) -> Optional[int]:
    free_idx = np.where(node.lb + EPS < node.ub)[0]
    if free_idx.size == 0:
        return None

    # Prefer the most fractional variable from LP relaxation.
    if node.relax_x is not None:
        frac = np.abs(node.relax_x - np.rint(node.relax_x))
        fractional = [int(i) for i in free_idx if frac[i] > INT_TOL]
        if fractional:
            return max(fractional, key=lambda i: float(frac[i]))

    # Fallback: branch on largest absolute objective coefficient.
    return int(max(free_idx, key=lambda i: abs(float(c[i]))))


def branch_and_bound_binary_ilp(
    c: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    max_nodes: int = 5000,
    tol: float = 1e-9,
) -> Dict[str, object]:
    """Best-bound branch-and-bound for binary ILP."""
    validate_binary_ilp(c=c, a=a, b=b)

    n = c.shape[0]
    use_lp_bounds = linprog is not None

    node_counter = 0
    root_lb = np.zeros(n, dtype=float)
    root_ub = np.ones(n, dtype=float)
    root = build_node(
        node_id=node_counter,
        depth=0,
        lb=root_lb,
        ub=root_ub,
        c=c,
        a=a,
        b=b,
        use_lp_bounds=use_lp_bounds,
    )

    if root is None:
        return {
            "status": "infeasible",
            "best_obj": float("-inf"),
            "best_x": None,
            "used_lp_relaxation": use_lp_bounds,
            "stats": {
                "visited_nodes": 0,
                "generated_nodes": 0,
                "pruned_by_bound": 0,
                "pruned_infeasible": 1,
                "pruned_integral_lp": 0,
                "incumbent_updates": 0,
            },
        }

    heap: list[Tuple[float, int, Node]] = [(-root.upper_bound, root.node_id, root)]
    best_obj = float("-inf")
    best_x: Optional[np.ndarray] = None

    stats = {
        "visited_nodes": 0,
        "generated_nodes": 1,
        "pruned_by_bound": 0,
        "pruned_infeasible": 0,
        "pruned_integral_lp": 0,
        "incumbent_updates": 0,
    }

    while heap and stats["visited_nodes"] < max_nodes:
        _, _, node = heapq.heappop(heap)
        stats["visited_nodes"] += 1

        if node.upper_bound <= best_obj + tol:
            stats["pruned_by_bound"] += 1
            continue

        # Leaf node: all variables fixed.
        if np.all(np.abs(node.lb - node.ub) <= EPS):
            x_leaf = node.lb.copy()
            if np.all(a @ x_leaf <= b + tol):
                obj_leaf = float(c @ x_leaf)
                if obj_leaf > best_obj + tol:
                    best_obj = obj_leaf
                    best_x = np.rint(x_leaf).astype(int)
                    stats["incumbent_updates"] += 1
            else:
                stats["pruned_infeasible"] += 1
            continue

        # LP node already integral -> direct incumbent candidate.
        if node.bound_source == "lp" and node.relax_x is not None and is_integral_vector(node.relax_x):
            x_int = np.rint(node.relax_x)
            if np.all(a @ x_int <= b + tol):
                obj_int = float(c @ x_int)
                if obj_int > best_obj + tol:
                    best_obj = obj_int
                    best_x = x_int.astype(int)
                    stats["incumbent_updates"] += 1
                stats["pruned_integral_lp"] += 1
                continue

        branch_idx = choose_branch_variable(node=node, c=c)
        if branch_idx is None:
            continue

        for value in (0.0, 1.0):
            child_lb = node.lb.copy()
            child_ub = node.ub.copy()
            child_lb[branch_idx] = value
            child_ub[branch_idx] = value

            node_counter += 1
            child = build_node(
                node_id=node_counter,
                depth=node.depth + 1,
                lb=child_lb,
                ub=child_ub,
                c=c,
                a=a,
                b=b,
                use_lp_bounds=use_lp_bounds,
            )
            if child is None:
                stats["pruned_infeasible"] += 1
                continue

            if child.upper_bound <= best_obj + tol:
                stats["pruned_by_bound"] += 1
                continue

            heapq.heappush(heap, (-child.upper_bound, child.node_id, child))
            stats["generated_nodes"] += 1

    status = "optimal" if not heap else "max_nodes_reached"
    return {
        "status": status,
        "best_obj": best_obj,
        "best_x": best_x,
        "used_lp_relaxation": use_lp_bounds,
        "stats": stats,
    }


def brute_force_binary_ilp(c: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    """Exact oracle for tiny problems; used only for demo verification."""
    n = c.shape[0]
    best_obj = float("-inf")
    best_x: Optional[np.ndarray] = None

    for bits in product((0, 1), repeat=n):
        x = np.asarray(bits, dtype=float)
        if np.all(a @ x <= b + 1e-9):
            obj = float(c @ x)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_x = x.astype(int)

    return best_obj, best_x


def main() -> None:
    # A fixed 0-1 ILP instance (project selection under multiple budgets).
    c = np.array([20.0, 24.0, 15.0, 40.0, 25.0, 30.0, 18.0, 12.0], dtype=float)
    a = np.array(
        [
            [2.0, 3.0, 1.0, 4.0, 2.0, 3.0, 2.0, 1.0],
            [3.0, 1.0, 2.0, 5.0, 3.0, 2.0, 1.0, 2.0],
            [1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0],
        ],
        dtype=float,
    )
    b = np.array([10.0, 11.0, 7.0], dtype=float)

    print("=== Integer Programming - Branch and Bound (MVP) ===")
    print(f"Variables: {c.shape[0]} (binary)")
    print(f"Constraints: {a.shape[0]} (Ax <= b)")
    print(f"Objective coefficients c: {c}")
    print("A matrix:")
    print(a)
    print(f"b vector: {b}")

    result = branch_and_bound_binary_ilp(
        c=c,
        a=a,
        b=b,
        max_nodes=5000,
        tol=1e-9,
    )

    print("\n=== Branch-and-Bound Result ===")
    print(f"Status: {result['status']}")
    print(f"LP relaxation available: {result['used_lp_relaxation']}")

    best_x = result["best_x"]
    best_obj = float(result["best_obj"])
    if best_x is None:
        print("No feasible integer solution found.")
    else:
        print(f"Best objective: {best_obj:.6f}")
        print(f"Best x: {best_x}")
        print(f"Resource usage A@x: {a @ best_x}")
        print(f"Budget limits b: {b}")

    print("\nNode statistics:")
    stats: Dict[str, int] = result["stats"]  # type: ignore[assignment]
    for key in (
        "visited_nodes",
        "generated_nodes",
        "pruned_by_bound",
        "pruned_infeasible",
        "pruned_integral_lp",
        "incumbent_updates",
    ):
        print(f"- {key}: {stats[key]}")

    brute_obj, brute_x = brute_force_binary_ilp(c=c, a=a, b=b)
    print("\n=== Brute-force Cross-check (tiny instance oracle) ===")
    print(f"Brute-force best objective: {brute_obj:.6f}")
    print(f"Brute-force best x: {brute_x}")
    if best_x is not None and brute_x is not None:
        gap = abs(best_obj - brute_obj)
        print(f"Objective gap |B&B - brute|: {gap:.6e}")


if __name__ == "__main__":
    main()
