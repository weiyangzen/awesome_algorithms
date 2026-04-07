"""SimRank minimal runnable MVP for CS-0321.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class DirectedGraph:
    nodes: list[str]
    edges: list[tuple[str, str]]


def build_demo_graph() -> DirectedGraph:
    """Build a deterministic directed graph for SimRank demonstration."""
    nodes = ["A", "B", "C", "D", "E", "F", "G", "H"]
    edges = [
        ("A", "C"),
        ("B", "C"),
        ("A", "D"),
        ("B", "D"),
        ("F", "C"),
        ("G", "D"),
        ("C", "E"),
        ("D", "E"),
        ("C", "H"),
        ("D", "H"),
        ("F", "H"),
        ("G", "H"),
    ]
    return DirectedGraph(nodes=nodes, edges=edges)


def build_inlink_transition_matrix(graph: DirectedGraph) -> tuple[sparse.csr_matrix, dict[str, int]]:
    """Return column-stochastic matrix P where P[i, j] = 1/|In(j)| if i -> j else 0."""
    n = len(graph.nodes)
    if n == 0:
        raise ValueError("graph must have at least one node")

    node_to_idx = {name: i for i, name in enumerate(graph.nodes)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for src, dst in graph.edges:
        if src not in node_to_idx or dst not in node_to_idx:
            raise ValueError(f"edge contains unknown node: {(src, dst)}")
        rows.append(node_to_idx[src])
        cols.append(node_to_idx[dst])
        data.append(1.0)

    adjacency = sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=float).tocsr()
    # Column L1 normalization turns adjacency into in-link transition probabilities.
    p = normalize(adjacency, norm="l1", axis=0, copy=True)
    return p.tocsr(), node_to_idx


def split_ranges(n: int, blocks: int) -> list[tuple[int, int]]:
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    blocks = min(blocks, n)
    step = (n + blocks - 1) // blocks
    return [(start, min(n, start + step)) for start in range(0, n, step)]


def simrank_step_dense(s: np.ndarray, p_dense: np.ndarray, c: float) -> np.ndarray:
    s_next = c * (p_dense.T @ s @ p_dense)
    np.fill_diagonal(s_next, 1.0)
    return s_next


def simrank_step_blockwise(
    s: np.ndarray,
    p_dense: np.ndarray,
    c: float,
    blocks: int,
    workers: int,
) -> np.ndarray:
    """Compute one SimRank iteration using row blocks (parallel/distributed-friendly)."""
    n = s.shape[0]
    row_ranges = split_ranges(n, blocks)
    right = s @ p_dense
    out = np.zeros_like(s)

    def compute_block(start: int, end: int) -> tuple[int, int, np.ndarray]:
        p_cols = p_dense[:, start:end]
        block = c * (p_cols.T @ right)
        return start, end, block

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(compute_block, start, end) for start, end in row_ranges]
            for fut in futures:
                start, end, block = fut.result()
                out[start:end, :] = block
    else:
        for start, end in row_ranges:
            _, _, block = compute_block(start, end)
            out[start:end, :] = block

    np.fill_diagonal(out, 1.0)
    return out


def simrank_step_torch(s: np.ndarray, p_dense: np.ndarray, c: float) -> np.ndarray:
    ts = torch.from_numpy(s).to(dtype=torch.float32)
    tp = torch.from_numpy(p_dense).to(dtype=torch.float32)
    tnext = c * (tp.T @ ts @ tp)
    tnext.fill_diagonal_(1.0)
    return tnext.cpu().numpy().astype(np.float64)


def run_simrank(
    p: sparse.csr_matrix,
    c: float = 0.8,
    max_iter: int = 12,
    tol: float = 1e-4,
    blocks: int = 4,
    workers: int = 4,
) -> tuple[np.ndarray, pd.DataFrame]:
    if not (0.0 < c < 1.0):
        raise ValueError("decay factor c must be in (0, 1)")

    n = p.shape[0]
    p_dense = p.toarray().astype(np.float64)
    s = np.eye(n, dtype=np.float64)

    history_rows: list[dict[str, float]] = []

    for it in range(1, max_iter + 1):
        s_next_block = simrank_step_blockwise(s, p_dense, c, blocks=blocks, workers=workers)
        s_next_dense = simrank_step_dense(s, p_dense, c)
        s_next_torch = simrank_step_torch(s, p_dense, c)

        if not np.allclose(s_next_block, s_next_dense, atol=1e-10):
            raise RuntimeError("blockwise step mismatch with dense reference")
        if not np.allclose(s_next_block, s_next_torch, atol=1e-6):
            raise RuntimeError("PyTorch step mismatch with NumPy reference")

        delta_max = float(np.max(np.abs(s_next_block - s)))
        mse = float(mean_squared_error(s.ravel(), s_next_block.ravel()))

        history_rows.append(
            {
                "iter": float(it),
                "delta_max": delta_max,
                "mse": mse,
            }
        )

        s = s_next_block

        if delta_max < tol:
            break

    history = pd.DataFrame(history_rows)
    history["iter"] = history["iter"].astype(int)
    return s, history


def top_similar_pairs(sim: np.ndarray, nodes: list[str], top_k: int = 8) -> pd.DataFrame:
    n = len(nodes)
    rows: list[dict[str, object]] = []

    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "node_i": nodes[i],
                    "node_j": nodes[j],
                    "simrank": float(sim[i, j]),
                }
            )

    df = pd.DataFrame(rows).sort_values("simrank", ascending=False).head(top_k)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df.reset_index(drop=True)


def run_self_check() -> None:
    graph = build_demo_graph()
    p, _ = build_inlink_transition_matrix(graph)
    s, history = run_simrank(p, c=0.8, max_iter=8, tol=1e-5, blocks=3, workers=2)

    if s.shape[0] != len(graph.nodes):
        raise RuntimeError("shape mismatch in similarity matrix")
    if not np.allclose(np.diag(s), 1.0, atol=1e-9):
        raise RuntimeError("diagonal must remain 1")
    if not np.allclose(s, s.T, atol=1e-9):
        raise RuntimeError("SimRank matrix should be symmetric")
    if history.empty:
        raise RuntimeError("iteration history should not be empty")


def main() -> None:
    run_self_check()

    graph = build_demo_graph()
    p, node_to_idx = build_inlink_transition_matrix(graph)

    workers = min(4, max(1, len(graph.nodes) // 2))
    sim, history = run_simrank(p, c=0.8, max_iter=12, tol=1e-4, blocks=4, workers=workers)

    pairs = top_similar_pairs(sim, graph.nodes, top_k=8)

    print("=== SimRank Config ===")
    print(f"nodes={len(graph.nodes)}, edges={len(graph.edges)}, c=0.8, max_iter=12, tol=1e-4")
    print(f"blocks=4, workers={workers}")

    print("\n=== Iteration History ===")
    print(history.to_string(index=False, formatters={"delta_max": lambda x: f"{x:.6f}", "mse": lambda x: f"{x:.8f}"}))

    print("\n=== Top Similar Node Pairs ===")
    print(pairs.to_string(index=False, formatters={"simrank": lambda x: f"{x:.6f}"}))

    focus_pairs = [("C", "D"), ("E", "H"), ("A", "B")]
    print("\n=== Focus Pairs ===")
    for u, v in focus_pairs:
        i, j = node_to_idx[u], node_to_idx[v]
        print(f"S({u},{v}) = {sim[i, j]:.6f}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
