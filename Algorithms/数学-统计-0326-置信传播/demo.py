"""Minimal runnable MVP for Belief Propagation (MATH-0326).

This script implements sum-product Belief Propagation on a tree-structured
binary pairwise Markov Random Field (MRF) using NumPy only.

It verifies correctness by comparing BP marginals against exact marginals
computed by exhaustive enumeration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

StateVec = np.ndarray
MessageMap = Dict[Tuple[int, int], StateVec]
UndirectedPsi = Dict[Tuple[int, int], np.ndarray]


@dataclass
class PairwiseTreeMRF:
    """Binary pairwise MRF on a tree."""

    n_nodes: int
    edges: List[Tuple[int, int]]
    neighbors: List[List[int]]
    phi: np.ndarray  # shape (n_nodes, 2), positive unary potentials
    psi: UndirectedPsi  # keyed by (u, v) with u < v, shape (2, 2)


def _normalize(vec: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    total = float(np.sum(vec))
    if total <= eps:
        raise ValueError("Cannot normalize vector with non-positive sum.")
    return vec / total


def _get_psi(i: int, j: int, psi: UndirectedPsi) -> np.ndarray:
    """Return pairwise potential matrix ordered as (x_i, x_j)."""
    if i < j:
        return psi[(i, j)]
    return psi[(j, i)].T


def build_random_tree_mrf(n_nodes: int, seed: int) -> PairwiseTreeMRF:
    """Create a reproducible random tree-structured binary MRF."""
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2.")

    rng = np.random.default_rng(seed)

    # Random recursive tree: each node i attaches to one previous node [0, i-1].
    edges: List[Tuple[int, int]] = []
    for i in range(1, n_nodes):
        parent = int(rng.integers(0, i))
        edges.append((parent, i))

    neighbors: List[List[int]] = [[] for _ in range(n_nodes)]
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)

    # Positive unary potentials.
    phi = rng.uniform(0.2, 1.4, size=(n_nodes, 2)).astype(np.float64)

    # Positive pairwise potentials, mildly favoring equal states (attractive bias).
    psi: UndirectedPsi = {}
    for u, v in edges:
        coupling = float(rng.uniform(0.0, 1.0))
        base = np.array(
            [
                [1.0 + 0.8 * coupling, 1.0 - 0.4 * coupling],
                [1.0 - 0.4 * coupling, 1.0 + 0.8 * coupling],
            ],
            dtype=np.float64,
        )
        # Keep strictly positive entries.
        base = np.maximum(base, 1e-6)
        key = (u, v) if u < v else (v, u)
        psi[key] = base

    return PairwiseTreeMRF(
        n_nodes=n_nodes,
        edges=edges,
        neighbors=neighbors,
        phi=phi,
        psi=psi,
    )


def build_parent_order(neighbors: List[List[int]], root: int) -> Tuple[List[int], List[int]]:
    """BFS parent array and traversal order from root."""
    n = len(neighbors)
    parent = [-1] * n
    order: List[int] = []

    parent[root] = root
    q: deque[int] = deque([root])

    while q:
        node = q.popleft()
        order.append(node)
        for nxt in neighbors[node]:
            if parent[nxt] != -1:
                continue
            parent[nxt] = node
            q.append(nxt)

    if len(order) != n:
        raise ValueError("Graph is disconnected; expected a tree/connected component.")

    parent[root] = -1
    return parent, order


def compute_message(
    src: int,
    dst: int,
    model: PairwiseTreeMRF,
    messages: MessageMap,
    eps: float = 1e-15,
) -> np.ndarray:
    """Compute one sum-product message m_{src->dst}."""
    local = model.phi[src].copy()
    for nb in model.neighbors[src]:
        if nb == dst:
            continue
        incoming_key = (nb, src)
        if incoming_key not in messages:
            raise KeyError(f"Missing incoming message {incoming_key} before computing {(src, dst)}")
        local *= messages[incoming_key]

    psi_src_dst = _get_psi(src, dst, model.psi)  # shape (2,2), rows x_src cols x_dst
    msg = local @ psi_src_dst
    return _normalize(msg, eps=eps)


def sum_product_bp_tree(model: PairwiseTreeMRF, root: int = 0) -> Tuple[MessageMap, np.ndarray]:
    """Run exact two-pass sum-product BP on a tree and return node marginals."""
    parent, order = build_parent_order(model.neighbors, root)
    messages: MessageMap = {}

    # Upward pass: leaves -> root
    for node in reversed(order):
        p = parent[node]
        if p == -1:
            continue
        messages[(node, p)] = compute_message(node, p, model, messages)

    # Downward pass: root -> leaves
    for node in order:
        for nb in model.neighbors[node]:
            if nb == parent[node]:
                continue
            messages[(node, nb)] = compute_message(node, nb, model, messages)

    n = model.n_nodes
    marginals = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        belief = model.phi[i].copy()
        for nb in model.neighbors[i]:
            belief *= messages[(nb, i)]
        marginals[i] = _normalize(belief)

    return messages, marginals


def enumerate_exact_marginals(model: PairwiseTreeMRF) -> np.ndarray:
    """Exact marginals by exhaustive enumeration over all binary assignments."""
    n = model.n_nodes
    n_states = 1 << n

    idx = np.arange(n_states, dtype=np.int64)
    states = ((idx[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.int64)

    weights = np.ones(n_states, dtype=np.float64)

    for i in range(n):
        weights *= model.phi[i, states[:, i]]

    for (u, v), mat in model.psi.items():
        weights *= mat[states[:, u], states[:, v]]

    z = float(np.sum(weights))
    if z <= 0.0:
        raise ValueError("Partition function is non-positive.")

    probs = weights / z
    marginals = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        marginals[i, 0] = float(np.sum(probs[states[:, i] == 0]))
        marginals[i, 1] = float(np.sum(probs[states[:, i] == 1]))

    return marginals


def validate_messages(messages: MessageMap, atol: float = 1e-12) -> None:
    for key, msg in messages.items():
        if np.any(msg < -atol):
            raise AssertionError(f"Message {key} has negative entries: {msg}")
        s = float(np.sum(msg))
        if not np.isclose(s, 1.0, atol=atol):
            raise AssertionError(f"Message {key} not normalized, sum={s}")


def main() -> None:
    print("Belief Propagation MVP - MATH-0326")
    print("=" * 80)

    model = build_random_tree_mrf(n_nodes=9, seed=20260326)
    messages, bp_marginals = sum_product_bp_tree(model, root=0)
    exact_marginals = enumerate_exact_marginals(model)

    validate_messages(messages)

    if not np.allclose(bp_marginals.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("BP node marginals are not normalized.")

    abs_err = np.abs(bp_marginals - exact_marginals)
    max_err = float(np.max(abs_err))
    mean_err = float(np.mean(abs_err))

    print(f"nodes: {model.n_nodes}, undirected edges: {len(model.edges)}")
    print(f"directed messages: {len(messages)} (expected {2 * len(model.edges)})")
    print("-" * 80)
    print("Per-node marginals [P(x_i=0), P(x_i=1)]")
    for i in range(model.n_nodes):
        bp_row = np.round(bp_marginals[i], 8)
        ex_row = np.round(exact_marginals[i], 8)
        print(f"node {i:02d} | BP={bp_row} | Exact={ex_row}")

    print("-" * 80)
    print(f"max abs error:  {max_err:.3e}")
    print(f"mean abs error: {mean_err:.3e}")

    # Tree BP should match exact enumeration up to numerical tolerance.
    assert len(messages) == 2 * len(model.edges)
    assert max_err < 1e-10

    print("All checks passed.")


if __name__ == "__main__":
    main()
