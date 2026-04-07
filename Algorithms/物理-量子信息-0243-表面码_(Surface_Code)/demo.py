"""Surface Code MVP (Z-noise subproblem, perfect syndrome).

This script implements a small but explicit decoder pipeline for a planar
surface code patch:
1) Build a distance-d lattice with data qubits on edges.
2) Inject independent Z errors on data qubits.
3) Extract X-stabilizer syndromes on interior vertices.
4) Decode by minimum-weight pairing (defect-defect / defect-boundary).
5) Apply correction and estimate logical-failure probability.

To keep the MVP auditable, matching is implemented in source via dynamic
programming over defect subsets rather than external black-box decoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd


Edge = tuple[str, int, int]  # ('h'|'v', x, y)
Vertex = tuple[int, int]  # (x, y)


@dataclass(frozen=True)
class SurfaceCodeConfig:
    """Configuration for deterministic MVP runs."""

    distance_exact: int = 3
    distance_mc: int = 5
    p_grid: tuple[float, ...] = (0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15)
    mc_shots: int = 2000
    seed: int = 20260407


def list_data_edges(d: int) -> list[Edge]:
    """List all data-qubit edges for a distance-d planar patch."""
    if d < 3 or d % 2 == 0:
        raise ValueError("Distance d must be an odd integer >= 3.")

    horizontal = [("h", x, y) for y in range(d) for x in range(d - 1)]
    vertical = [("v", x, y) for y in range(d - 1) for x in range(d)]
    return horizontal + vertical


def measured_vertices(d: int) -> list[Vertex]:
    """Interior vertices where X-stabilizers are measured.

    We treat y=0 and y=d-1 as rough boundaries for the Z-error decoding sector,
    so interior rows y=1..d-2 host measured syndromes.
    """
    return [(x, y) for y in range(1, d - 1) for x in range(d)]


def incident_edges(v: Vertex, d: int) -> list[Edge]:
    """Return data edges touching vertex v."""
    x, y = v
    out: list[Edge] = []

    if x > 0:
        out.append(("h", x - 1, y))
    if x < d - 1:
        out.append(("h", x, y))
    if y > 0:
        out.append(("v", x, y - 1))
    if y < d - 1:
        out.append(("v", x, y))

    return out


def toggle_edge(edge_set: set[Edge], edge: Edge) -> None:
    """Toggle edge parity under GF(2)."""
    if edge in edge_set:
        edge_set.remove(edge)
    else:
        edge_set.add(edge)


def syndrome_from_errors(error_edges: set[Edge], d: int) -> list[Vertex]:
    """Compute X-stabilizer syndrome for Z errors."""
    defects: list[Vertex] = []
    for v in measured_vertices(d):
        parity = 0
        for e in incident_edges(v, d):
            if e in error_edges:
                parity ^= 1
        if parity == 1:
            defects.append(v)
    return defects


def manhattan(a: Vertex, b: Vertex) -> int:
    """Manhattan distance on the vertex grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_boundary(v: Vertex, d: int) -> tuple[int, str]:
    """Distance and label of nearest rough boundary (top/bottom)."""
    y = v[1]
    top = y
    bottom = (d - 1) - y
    if top <= bottom:
        return top, "top"
    return bottom, "bottom"


def path_between_vertices(a: Vertex, b: Vertex) -> list[Edge]:
    """Deterministic shortest path edges from a to b (horizontal then vertical)."""
    x1, y1 = a
    x2, y2 = b

    path: list[Edge] = []

    if x2 >= x1:
        for x in range(x1, x2):
            path.append(("h", x, y1))
    else:
        for x in range(x2, x1):
            path.append(("h", x, y1))

    if y2 >= y1:
        for y in range(y1, y2):
            path.append(("v", x2, y))
    else:
        for y in range(y2, y1):
            path.append(("v", x2, y))

    return path


def path_to_boundary(v: Vertex, d: int, boundary: str) -> list[Edge]:
    """Deterministic shortest path from vertex to rough boundary."""
    x, y = v
    if boundary == "top":
        return [("v", x, yy) for yy in range(0, y)]
    if boundary == "bottom":
        return [("v", x, yy) for yy in range(y, d - 1)]
    raise ValueError("boundary must be 'top' or 'bottom'.")


def first_set_bit(mask: int) -> int:
    """Index of least-significant set bit."""
    return (mask & -mask).bit_length() - 1


def min_weight_actions(defects: list[Vertex], d: int) -> list[tuple]:
    """Return minimum-weight correction actions for given syndrome defects.

    Action format:
      - ('B', vertex, 'top'|'bottom')
      - ('P', vertex_a, vertex_b)

    Uses DP over subsets (exact for this defect set).
    """
    m = len(defects)
    if m == 0:
        return []

    pair_dist = [[0 for _ in range(m)] for _ in range(m)]
    bdist = [0 for _ in range(m)]
    bchoice = ["top" for _ in range(m)]

    for i in range(m):
        bdist[i], bchoice[i] = nearest_boundary(defects[i], d)
        for j in range(i + 1, m):
            d_ij = manhattan(defects[i], defects[j])
            pair_dist[i][j] = d_ij
            pair_dist[j][i] = d_ij

    decision: dict[int, tuple] = {}

    @lru_cache(maxsize=None)
    def min_cost(mask: int) -> int:
        if mask == 0:
            return 0

        i = first_set_bit(mask)
        rem = mask ^ (1 << i)

        best_cost = bdist[i] + min_cost(rem)
        best_decision = ("B", i, bchoice[i], rem)

        probe = rem
        while probe:
            j = first_set_bit(probe)
            rem2 = rem ^ (1 << j)
            c = pair_dist[i][j] + min_cost(rem2)
            if c < best_cost:
                best_cost = c
                best_decision = ("P", i, j, rem2)
            probe ^= (1 << j)

        decision[mask] = best_decision
        return best_cost

    full_mask = (1 << m) - 1
    _ = min_cost(full_mask)

    actions: list[tuple] = []
    mask = full_mask
    while mask:
        dec = decision[mask]
        kind = dec[0]

        if kind == "B":
            _, i, boundary, next_mask = dec
            actions.append(("B", defects[i], boundary))
            mask = next_mask
        else:
            _, i, j, next_mask = dec
            actions.append(("P", defects[i], defects[j]))
            mask = next_mask

    return actions


def correction_from_actions(actions: list[tuple], d: int) -> set[Edge]:
    """Convert matching actions into correction edge set."""
    corr: set[Edge] = set()

    for act in actions:
        if act[0] == "B":
            _, v, boundary = act
            path = path_to_boundary(v, d, boundary)
        else:
            _, a, b = act
            path = path_between_vertices(a, b)

        for e in path:
            toggle_edge(corr, e)

    return corr


def decode_z_errors(error_edges: set[Edge], d: int) -> tuple[set[Edge], list[Vertex], list[tuple]]:
    """Decode one shot: syndrome extraction + minimum-weight correction."""
    defects = syndrome_from_errors(error_edges, d)
    actions = min_weight_actions(defects, d)
    correction = correction_from_actions(actions, d)
    return correction, defects, actions


def residual_chain(error_edges: set[Edge], correction_edges: set[Edge]) -> set[Edge]:
    """Residual chain after applying correction (XOR in GF(2))."""
    out = set(error_edges)
    for e in correction_edges:
        toggle_edge(out, e)
    return out


def logical_failure_from_residual(residual: set[Edge], d: int) -> bool:
    """Detect non-trivial top-bottom class by parity on a reference cut."""
    cut_y = (d - 1) // 2
    parity = 0
    for x in range(d):
        if ("v", x, cut_y) in residual:
            parity ^= 1
    return parity == 1


def sample_z_errors(edges: list[Edge], p: float, rng: np.random.Generator) -> set[Edge]:
    """Sample i.i.d. Z errors over data edges."""
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1].")

    flags = rng.random(len(edges)) < p
    return {e for e, f in zip(edges, flags) if f}


def enumerate_failure_by_weight(d: int) -> tuple[np.ndarray, int, int]:
    """Exact failure statistics grouped by error weight for distance d.

    Returns:
      fail_count_by_weight[w] = number of failing patterns with Hamming weight w
      n_edges             = number of data edges
      checked_patterns    = total enumerated patterns
    """
    edges = list_data_edges(d)
    n_edges = len(edges)
    total_patterns = 1 << n_edges

    fail_count = np.zeros(n_edges + 1, dtype=np.int64)

    for mask in range(total_patterns):
        err: set[Edge] = set()
        for i, e in enumerate(edges):
            if (mask >> i) & 1:
                err.add(e)

        corr, _, _ = decode_z_errors(err, d)
        res = residual_chain(err, corr)

        # Decoder should clear all measured-vertex syndromes.
        assert len(syndrome_from_errors(res, d)) == 0

        if logical_failure_from_residual(res, d):
            fail_count[mask.bit_count()] += 1

    return fail_count, n_edges, total_patterns


def failure_probability_from_weight_stats(
    p: float,
    fail_count_by_weight: np.ndarray,
    n_edges: int,
) -> float:
    """Compute exact logical-failure probability from grouped combinatorics."""
    p_fail = 0.0
    for w in range(n_edges + 1):
        count = float(fail_count_by_weight[w])
        if count == 0.0:
            continue
        p_fail += count * (p**w) * ((1.0 - p) ** (n_edges - w))
    return p_fail


def run_monte_carlo(d: int, p: float, shots: int, seed: int) -> tuple[float, float, float]:
    """Monte Carlo estimate for a larger distance where full enumeration is costly."""
    rng = np.random.default_rng(seed)
    edges = list_data_edges(d)

    logical_fail = 0
    mean_syndrome_weight = 0.0
    mean_error_weight = 0.0

    for _ in range(shots):
        err = sample_z_errors(edges, p, rng)
        corr, defects, _ = decode_z_errors(err, d)
        res = residual_chain(err, corr)

        logical_fail += int(logical_failure_from_residual(res, d))
        mean_syndrome_weight += len(defects)
        mean_error_weight += len(err)

    inv = 1.0 / float(shots)
    return logical_fail * inv, mean_syndrome_weight * inv, mean_error_weight * inv


def run_mvp(config: SurfaceCodeConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run exact d=3 table and MC d=5 table."""
    d_exact = config.distance_exact
    d_mc = config.distance_mc

    fail_count, n_edges_exact, total_patterns = enumerate_failure_by_weight(d_exact)

    exact_rows: list[dict[str, float]] = []
    mc_rows: list[dict[str, float]] = []

    for idx, p in enumerate(config.p_grid):
        exact_fail = failure_probability_from_weight_stats(p, fail_count, n_edges_exact)
        mc_fail, mean_syn, mean_err = run_monte_carlo(
            d=d_mc,
            p=p,
            shots=config.mc_shots,
            seed=config.seed + idx,
        )

        exact_rows.append(
            {
                "p": p,
                f"logical_fail_d{d_exact}_exact": exact_fail,
            }
        )
        mc_rows.append(
            {
                "p": p,
                f"logical_fail_d{d_mc}_mc": mc_fail,
                "mean_syndrome_weight": mean_syn,
                "mean_error_weight": mean_err,
                "shots": float(config.mc_shots),
            }
        )

    exact_df = pd.DataFrame(exact_rows)
    mc_df = pd.DataFrame(mc_rows)

    merge_df = exact_df.merge(mc_df, on="p", how="inner")
    merge_df["distance_gain(d3-d5)"] = (
        merge_df[f"logical_fail_d{d_exact}_exact"] - merge_df[f"logical_fail_d{d_mc}_mc"]
    )

    diagnostics = {
        "distance_exact": float(d_exact),
        "distance_mc": float(d_mc),
        "n_edges_exact": float(n_edges_exact),
        "enumerated_patterns_exact": float(total_patterns),
        "exact_p0_fail": float(failure_probability_from_weight_stats(0.0, fail_count, n_edges_exact)),
    }

    assert diagnostics["exact_p0_fail"] == 0.0

    return merge_df, pd.DataFrame([diagnostics]), diagnostics


def main() -> None:
    config = SurfaceCodeConfig()
    result_df, diag_df, diagnostics = run_mvp(config)

    print("Surface Code MVP (Z-noise sector, perfect syndrome)")
    print("=" * 78)
    print("Pipeline: sample errors -> syndrome -> MW pairing decode -> residual homology")
    print()
    print("Logical-failure table:")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.10f}"))
    print()
    print("Diagnostics:")
    print(diag_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(
        "Summary: d=3 exact enumeration and d=5 Monte Carlo are both complete; "
        "all hard assertions passed."
    )
    print(f"exact_p0_fail = {diagnostics['exact_p0_fail']:.1f}")


if __name__ == "__main__":
    main()
