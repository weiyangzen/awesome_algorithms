"""Dantzig-Wolfe decomposition MVP on a small block-angular LP.

This demo is intentionally self-contained and uses only numpy.
It includes a tiny active-set LP solver suitable for small teaching examples.

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Block:
    """One independent block subproblem in Dantzig-Wolfe decomposition."""

    name: str
    c: np.ndarray  # local objective coefficients
    a_local: np.ndarray  # A_local x <= b_local
    b_local: np.ndarray
    d_link: np.ndarray  # linking contribution matrix (m_link x n_var)
    vertices: Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class LPResult:
    """Result container of the tiny LP solver."""

    x: np.ndarray
    objective: float
    active_set: Tuple[int, ...]


@dataclass(frozen=True)
class RMPPrimal:
    """Primal RMP solution."""

    lambdas_by_block: Tuple[np.ndarray, ...]
    artificial: np.ndarray
    objective: float


@dataclass(frozen=True)
class RMPDual:
    """Dual RMP solution, used for pricing."""

    pi: np.ndarray
    mu: np.ndarray
    objective: float


def _to_2d_or_empty(a: np.ndarray | None, n: int) -> np.ndarray:
    if a is None:
        return np.zeros((0, n), dtype=float)
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != n:
        raise ValueError("Constraint matrix has invalid shape")
    return arr


def _to_1d_or_empty(b: np.ndarray | None) -> np.ndarray:
    if b is None:
        return np.zeros(0, dtype=float)
    arr = np.asarray(b, dtype=float).reshape(-1)
    return arr


def solve_lp_small(
    c: np.ndarray,
    a_le: np.ndarray | None = None,
    b_le: np.ndarray | None = None,
    a_ge: np.ndarray | None = None,
    b_ge: np.ndarray | None = None,
    a_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
    tol: float = 1e-9,
) -> LPResult:
    """Solve tiny LP by enumerating active sets.

    Problem:
        min c^T x
        s.t. A_le x <= b_le
             A_ge x >= b_ge
             A_eq x  = b_eq
             lb <= x <= ub

    Notes:
    - This is exponential in variable count; only suitable for tiny examples.
    - Used here to keep demo self-contained without third-party LP packages.
    """
    c = np.asarray(c, dtype=float).reshape(-1)
    n = c.size
    if n == 0:
        return LPResult(x=np.zeros(0, dtype=float), objective=0.0, active_set=tuple())

    a_le_m = _to_2d_or_empty(a_le, n)
    b_le_v = _to_1d_or_empty(b_le)
    a_ge_m = _to_2d_or_empty(a_ge, n)
    b_ge_v = _to_1d_or_empty(b_ge)
    a_eq_m = _to_2d_or_empty(a_eq, n)
    b_eq_v = _to_1d_or_empty(b_eq)

    if a_le_m.shape[0] != b_le_v.size:
        raise ValueError("A_le and b_le size mismatch")
    if a_ge_m.shape[0] != b_ge_v.size:
        raise ValueError("A_ge and b_ge size mismatch")
    if a_eq_m.shape[0] != b_eq_v.size:
        raise ValueError("A_eq and b_eq size mismatch")

    if lb is None:
        lb_v = np.zeros(n, dtype=float)
    else:
        lb_v = np.asarray(lb, dtype=float).reshape(-1)
        if lb_v.size != n:
            raise ValueError("lb size mismatch")

    if ub is None:
        ub_v = np.full(n, np.inf, dtype=float)
    else:
        ub_v = np.asarray(ub, dtype=float).reshape(-1)
        if ub_v.size != n:
            raise ValueError("ub size mismatch")

    # Convert all constraints into A_all x <= b_all.
    a_parts = [a_le_m, -a_ge_m, a_eq_m, -a_eq_m, -np.eye(n, dtype=float)]
    b_parts = [b_le_v, -b_ge_v, b_eq_v, -b_eq_v, -lb_v]

    finite_ub = np.isfinite(ub_v)
    if np.any(finite_ub):
        a_ub_bound = np.zeros((int(np.sum(finite_ub)), n), dtype=float)
        b_ub_bound = ub_v[finite_ub].copy()
        row = 0
        for i, flag in enumerate(finite_ub.tolist()):
            if flag:
                a_ub_bound[row, i] = 1.0
                row += 1
        a_parts.append(a_ub_bound)
        b_parts.append(b_ub_bound)

    a_all = np.vstack(a_parts)
    b_all = np.concatenate(b_parts)

    m = a_all.shape[0]
    if m < n:
        raise RuntimeError("Not enough constraints to identify vertices")

    best_x = None
    best_obj = float("inf")
    best_active: Tuple[int, ...] | None = None

    for active in combinations(range(m), n):
        a_act = a_all[list(active), :]
        b_act = b_all[list(active)]

        if np.linalg.matrix_rank(a_act) < n:
            continue

        try:
            x = np.linalg.solve(a_act, b_act)
        except np.linalg.LinAlgError:
            continue

        if np.any(~np.isfinite(x)):
            continue

        if np.all(a_all @ x <= b_all + tol):
            obj = float(c @ x)
            if obj < best_obj - 1e-12:
                best_obj = obj
                best_x = x
                best_active = tuple(active)

    if best_x is None or best_active is None:
        raise RuntimeError("LP infeasible or no vertex found by active-set enumeration")

    return LPResult(x=best_x, objective=best_obj, active_set=best_active)


def enumerate_polytope_vertices(
    a_mat: np.ndarray, b_vec: np.ndarray, tol: float = 1e-9
) -> Tuple[np.ndarray, ...]:
    """Enumerate polytope vertices by active-set combinations."""
    a_mat = np.asarray(a_mat, dtype=float)
    b_vec = np.asarray(b_vec, dtype=float).reshape(-1)
    m, n = a_mat.shape
    if b_vec.size != m:
        raise ValueError("a_mat and b_vec shape mismatch")

    verts: List[np.ndarray] = []
    seen = set()

    for idxs in combinations(range(m), n):
        a_eq = a_mat[list(idxs), :]
        b_eq = b_vec[list(idxs)]

        if np.linalg.matrix_rank(a_eq) < n:
            continue

        try:
            x = np.linalg.solve(a_eq, b_eq)
        except np.linalg.LinAlgError:
            continue

        if np.any(~np.isfinite(x)):
            continue
        if np.all(a_mat @ x <= b_vec + tol):
            key = tuple(np.round(x, 10).tolist())
            if key not in seen:
                seen.add(key)
                verts.append(x)

    if not verts:
        raise RuntimeError("No feasible vertices found; check block constraints")

    return tuple(verts)


def build_demo_blocks() -> List[Block]:
    """Create two blocks with bounded local polytopes."""
    # Block 1: x = [x11, x12]
    a1 = np.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )
    b1 = np.array([4.0, 3.0, 3.0, 0.0, 0.0], dtype=float)
    c1 = np.array([5.0, 4.0], dtype=float)
    d1 = np.array(
        [
            [2.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )

    # Block 2: x = [x21, x22]
    a2 = np.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )
    b2 = np.array([5.0, 4.0, 4.0, 0.0, 0.0], dtype=float)
    c2 = np.array([4.0, 6.0], dtype=float)
    d2 = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )

    v1 = enumerate_polytope_vertices(a1, b1)
    v2 = enumerate_polytope_vertices(a2, b2)

    return [
        Block("Block-1", c=c1, a_local=a1, b_local=b1, d_link=d1, vertices=v1),
        Block("Block-2", c=c2, a_local=a2, b_local=b2, d_link=d2, vertices=v2),
    ]


def _is_duplicate_column(candidate: np.ndarray, cols: Sequence[np.ndarray], tol: float = 1e-10) -> bool:
    for col in cols:
        if np.allclose(candidate, col, atol=tol, rtol=0.0):
            return True
    return False


def _build_lambda_data(
    blocks: Sequence[Block], columns_by_block: Sequence[Sequence[np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Build lambda-side objective and constraint coefficient matrices.

    Returns:
        c_lambda: shape (n_lambda,)
        g_cover: shape (m_link, n_lambda)
        e_convex: shape (n_blocks, n_lambda)
        block_of_col: list mapping lambda index -> block index
    """
    n_blocks = len(blocks)
    m_link = blocks[0].d_link.shape[0]

    c_list: List[float] = []
    g_cols: List[np.ndarray] = []
    block_of_col: List[int] = []

    for b, block in enumerate(blocks):
        for col in columns_by_block[b]:
            c_list.append(float(block.c @ col))
            g_cols.append(block.d_link @ col)
            block_of_col.append(b)

    n_lambda = len(c_list)
    c_lambda = np.array(c_list, dtype=float)
    g_cover = np.column_stack(g_cols).astype(float) if g_cols else np.zeros((m_link, 0), dtype=float)

    e_convex = np.zeros((n_blocks, n_lambda), dtype=float)
    for j, b in enumerate(block_of_col):
        e_convex[b, j] = 1.0

    return c_lambda, g_cover, e_convex, block_of_col


def solve_rmp_primal(
    blocks: Sequence[Block],
    demand: np.ndarray,
    columns_by_block: Sequence[Sequence[np.ndarray]],
    big_m: float,
) -> RMPPrimal:
    """Solve primal RMP with small LP solver."""
    n_blocks = len(blocks)
    m_link = len(demand)

    c_lambda, g_cover, e_convex, _ = _build_lambda_data(blocks, columns_by_block)
    n_lambda = c_lambda.size

    # Variables: [lambda(0..n_lambda-1), s(0..m_link-1)]
    c = np.concatenate([c_lambda, np.full(m_link, big_m, dtype=float)])

    a_ge = np.hstack([g_cover, np.eye(m_link, dtype=float)])
    b_ge = demand.astype(float)

    a_eq = np.hstack([e_convex, np.zeros((n_blocks, m_link), dtype=float)])
    b_eq = np.ones(n_blocks, dtype=float)

    lb = np.zeros(n_lambda + m_link, dtype=float)

    res = solve_lp_small(c=c, a_ge=a_ge, b_ge=b_ge, a_eq=a_eq, b_eq=b_eq, lb=lb)
    x = res.x

    lambdas_list: List[np.ndarray] = []
    cursor = 0
    for b in range(n_blocks):
        cnt = len(columns_by_block[b])
        lambdas_list.append(x[cursor : cursor + cnt].copy())
        cursor += cnt

    artificial = x[n_lambda:].copy()

    return RMPPrimal(
        lambdas_by_block=tuple(lambdas_list),
        artificial=artificial,
        objective=float(res.objective),
    )


def solve_rmp_dual(
    blocks: Sequence[Block],
    demand: np.ndarray,
    columns_by_block: Sequence[Sequence[np.ndarray]],
    big_m: float,
) -> RMPDual:
    """Solve dual RMP with small LP solver.

    Primal RMP:
        min c_lambda^T lambda + M 1^T s
        s.t. G lambda + s >= d
             E lambda = 1
             lambda,s >= 0

    Dual:
        max d^T pi + 1^T mu
        s.t. G^T pi + E^T mu <= c_lambda
             pi <= M
             pi >= 0, mu free

    Free mu is represented as mu = mu_plus - mu_minus, both >= 0.
    """
    n_blocks = len(blocks)
    m_link = len(demand)

    c_lambda, g_cover, e_convex, block_of_col = _build_lambda_data(blocks, columns_by_block)
    n_lambda = c_lambda.size

    # Dual variables y = [pi(m_link), mu_plus(n_blocks), mu_minus(n_blocks)] >= 0
    n_dual = m_link + 2 * n_blocks

    # Max q^T y -> Min -q^T y
    q = np.concatenate(
        [
            demand.astype(float),
            np.ones(n_blocks, dtype=float),
            -np.ones(n_blocks, dtype=float),
        ]
    )
    c_min = -q

    rows: List[np.ndarray] = []
    rhs: List[float] = []

    # Constraints for each lambda column j:
    #   G[:,j]^T pi + mu_b <= c_lambda[j]
    for j in range(n_lambda):
        row = np.zeros(n_dual, dtype=float)
        row[:m_link] = g_cover[:, j]
        b = block_of_col[j]
        row[m_link + b] = 1.0
        row[m_link + n_blocks + b] = -1.0
        rows.append(row)
        rhs.append(float(c_lambda[j]))

    # Constraints from artificial variables: pi_i <= M.
    for i in range(m_link):
        row = np.zeros(n_dual, dtype=float)
        row[i] = 1.0
        rows.append(row)
        rhs.append(float(big_m))

    a_le = np.vstack(rows)
    b_le = np.array(rhs, dtype=float)
    lb = np.zeros(n_dual, dtype=float)

    res = solve_lp_small(c=c_min, a_le=a_le, b_le=b_le, lb=lb)

    y = res.x
    pi = y[:m_link].copy()
    mu_plus = y[m_link : m_link + n_blocks]
    mu_minus = y[m_link + n_blocks :]
    mu = mu_plus - mu_minus

    dual_obj = -float(res.objective)
    return RMPDual(pi=pi, mu=mu, objective=dual_obj)


def solve_pricing_for_block(block: Block, pi: np.ndarray, mu_b: float) -> Tuple[np.ndarray, float, float]:
    """Pricing over pre-enumerated extreme points for one block.

    Reduced cost of a new column v in block b:
        rc(v) = c_b^T v - pi^T(D_b v) - mu_b
    """
    w = block.c - block.d_link.T @ pi

    best_v = None
    best_value = float("inf")
    for v in block.vertices:
        val = float(w @ v)
        if val < best_value:
            best_value = val
            best_v = v

    if best_v is None:
        raise RuntimeError("Pricing failed to find a candidate")

    reduced_cost = best_value - float(mu_b)
    return best_v.copy(), reduced_cost, best_value


def recover_block_decisions(
    columns_by_block: Sequence[Sequence[np.ndarray]], lambdas_by_block: Sequence[np.ndarray]
) -> Tuple[np.ndarray, ...]:
    """Recover x_b = sum_j lambda_bj * v_bj from master variables."""
    xs: List[np.ndarray] = []
    for cols, lam in zip(columns_by_block, lambdas_by_block):
        x_b = np.zeros_like(cols[0], dtype=float)
        for coeff, col in zip(lam, cols):
            x_b += float(coeff) * col
        xs.append(x_b)
    return tuple(xs)


def solve_monolithic_lp(blocks: Sequence[Block], demand: np.ndarray) -> Tuple[float, Tuple[np.ndarray, ...]]:
    """Solve original LP directly (no decomposition) with small LP solver."""
    dims = [len(b.c) for b in blocks]
    offsets = np.cumsum([0] + dims)
    n_total = int(offsets[-1])
    m_link = len(demand)

    c = np.concatenate([b.c for b in blocks])

    # Linking >= constraints.
    a_ge = np.zeros((m_link, n_total), dtype=float)
    for b_idx, b in enumerate(blocks):
        l, r = int(offsets[b_idx]), int(offsets[b_idx + 1])
        a_ge[:, l:r] = b.d_link
    b_ge = demand.astype(float)

    # Local <= constraints.
    local_rows = sum(b.a_local.shape[0] for b in blocks)
    a_le = np.zeros((local_rows, n_total), dtype=float)
    b_le = np.zeros(local_rows, dtype=float)

    row_cursor = 0
    for b_idx, b in enumerate(blocks):
        l, r = int(offsets[b_idx]), int(offsets[b_idx + 1])
        cnt = b.a_local.shape[0]
        a_le[row_cursor : row_cursor + cnt, l:r] = b.a_local
        b_le[row_cursor : row_cursor + cnt] = b.b_local
        row_cursor += cnt

    lb = np.zeros(n_total, dtype=float)
    res = solve_lp_small(c=c, a_le=a_le, b_le=b_le, a_ge=a_ge, b_ge=b_ge, lb=lb)

    x = res.x
    split: List[np.ndarray] = []
    for b_idx in range(len(blocks)):
        l, r = int(offsets[b_idx]), int(offsets[b_idx + 1])
        split.append(x[l:r].copy())

    return float(res.objective), tuple(split)


def run_dantzig_wolfe(
    blocks: Sequence[Block],
    demand: np.ndarray,
    big_m: float = 1_000.0,
    tol: float = 1e-8,
    max_iter: int = 25,
) -> Tuple[RMPPrimal, RMPDual, Tuple[np.ndarray, ...], List[dict]]:
    """Main Dantzig-Wolfe loop with column generation."""
    columns_by_block: List[List[np.ndarray]] = []
    for b in blocks:
        zero_col = np.zeros_like(b.c, dtype=float)
        columns_by_block.append([zero_col])

    logs: List[dict] = []
    final_primal: RMPPrimal | None = None
    final_dual: RMPDual | None = None

    for it in range(1, max_iter + 1):
        primal = solve_rmp_primal(blocks, demand, columns_by_block, big_m)
        dual = solve_rmp_dual(blocks, demand, columns_by_block, big_m)
        final_primal, final_dual = primal, dual

        iter_rcs: List[float] = []
        iter_added: List[bool] = []

        for b_idx, b in enumerate(blocks):
            candidate, rc, _ = solve_pricing_for_block(b, dual.pi, dual.mu[b_idx])
            iter_rcs.append(rc)

            should_add = rc < -tol and (not _is_duplicate_column(candidate, columns_by_block[b_idx]))
            iter_added.append(should_add)
            if should_add:
                columns_by_block[b_idx].append(candidate)

        logs.append(
            {
                "iter": it,
                "primal_obj": primal.objective,
                "dual_obj": dual.objective,
                "pd_gap": abs(primal.objective - dual.objective),
                "art_sum": float(np.sum(primal.artificial)),
                "pi": dual.pi.copy(),
                "mu": dual.mu.copy(),
                "rcs": iter_rcs,
                "added": iter_added,
                "n_cols": [len(c) for c in columns_by_block],
            }
        )

        if not any(iter_added):
            break

    if final_primal is None or final_dual is None:
        raise RuntimeError("Dantzig-Wolfe loop did not execute")

    # Re-solve on final column set.
    final_primal = solve_rmp_primal(blocks, demand, columns_by_block, big_m)
    final_dual = solve_rmp_dual(blocks, demand, columns_by_block, big_m)

    x_blocks = recover_block_decisions(columns_by_block, final_primal.lambdas_by_block)
    return final_primal, final_dual, x_blocks, logs


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    blocks = build_demo_blocks()
    demand = np.array([8.0, 7.0], dtype=float)

    print("Dantzig-Wolfe decomposition MVP (block-angular LP)")
    print(f"Demand RHS: {demand.tolist()}")
    for b in blocks:
        print(f"{b.name}: vars={len(b.c)}, enumerated_vertices={len(b.vertices)}")

    primal, dual, x_blocks, logs = run_dantzig_wolfe(
        blocks, demand, big_m=1_000.0, tol=1e-8, max_iter=25
    )

    print("\nIteration log:")
    for row in logs:
        pi_str = np.array2string(row["pi"], precision=4, suppress_small=True)
        mu_str = np.array2string(row["mu"], precision=4, suppress_small=True)
        print(
            f"  iter={row['iter']:02d}"
            f" | primal={row['primal_obj']:.6f}"
            f" | dual={row['dual_obj']:.6f}"
            f" | gap={row['pd_gap']:.3e}"
            f" | art_sum={row['art_sum']:.6f}"
            f" | rcs={[round(v, 6) for v in row['rcs']]}"
            f" | added={row['added']}"
            f" | n_cols={row['n_cols']}"
            f" | pi={pi_str}"
            f" | mu={mu_str}"
        )

    print("\nRecovered block decisions from D-W master:")
    for b, xb in zip(blocks, x_blocks):
        print(f"  {b.name}: x = {np.array2string(xb, precision=6, suppress_small=True)}")

    total_cost_dw = sum(float(b.c @ xb) for b, xb in zip(blocks, x_blocks))
    total_cover_dw = sum(b.d_link @ xb for b, xb in zip(blocks, x_blocks))

    print(f"  Total primal cost (from recovered x): {total_cost_dw:.6f}")
    print(f"  Linking coverage: {np.array2string(total_cover_dw, precision=6, suppress_small=True)}")
    print(f"  Final RMP objective: {primal.objective:.6f}")
    print(f"  Final RMP dual objective: {dual.objective:.6f}")
    print(f"  Final artificial sum: {float(np.sum(primal.artificial)):.6e}")

    mono_obj, mono_x = solve_monolithic_lp(blocks, demand)
    print("\nMonolithic LP reference:")
    print(f"  Objective: {mono_obj:.6f}")
    for b, xb in zip(blocks, mono_x):
        print(f"  {b.name}: x = {np.array2string(xb, precision=6, suppress_small=True)}")

    gap = abs(total_cost_dw - mono_obj)
    print(f"\nObjective gap |DW - monolithic| = {gap:.6e}")


if __name__ == "__main__":
    main()
