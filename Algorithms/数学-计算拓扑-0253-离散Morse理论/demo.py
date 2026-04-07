"""离散 Morse 理论的最小可运行示例。

实现内容：
1) 用有限单纯复形构造 Hasse 图。
2) 通过贪心 + 有向无环约束生成离散向量场（匹配）。
3) 统计临界胞元。
4) 用 Z2 边界矩阵计算 Betti 数，校验弱 Morse 不等式与 Euler 恒等式。
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

Simplex = Tuple[int, ...]
Pair = Tuple[Simplex, Simplex]  # (p-cell, (p+1)-cell)


def simplex_dim(s: Simplex) -> int:
    return len(s) - 1


def faces_of(simplex: Simplex) -> List[Simplex]:
    """返回一个单纯形的所有余维 1 面。"""
    if len(simplex) <= 1:
        return []
    return [tuple(simplex[:i] + simplex[i + 1 :]) for i in range(len(simplex))]


def closure_of_maximal(maximal_simplices: Sequence[Simplex]) -> Set[Simplex]:
    """从极大单纯形生成闭包（包含所有非空子面）。"""
    all_cells: Set[Simplex] = set()
    for simp in maximal_simplices:
        s = tuple(sorted(simp))
        for r in range(1, len(s) + 1):
            for face in combinations(s, r):
                all_cells.add(face)
    return all_cells


def group_cells_by_dim(cells: Iterable[Simplex]) -> Dict[int, List[Simplex]]:
    grouped: Dict[int, List[Simplex]] = {}
    for c in cells:
        d = simplex_dim(c)
        grouped.setdefault(d, []).append(c)
    for d in grouped:
        grouped[d].sort()
    return grouped


def incidence_pairs(cells_by_dim: Dict[int, List[Simplex]]) -> List[Pair]:
    """列出所有 (alpha^p, beta^(p+1))，其中 alpha 是 beta 的面。"""
    pairs: List[Pair] = []
    max_dim = max(cells_by_dim)
    for p in range(max_dim):
        low_cells = cells_by_dim.get(p, [])
        high_cells = cells_by_dim.get(p + 1, [])
        low_set = set(low_cells)
        for beta in high_cells:
            for alpha in faces_of(beta):
                if alpha in low_set:
                    pairs.append((alpha, beta))
    # 固定顺序，保证复现
    pairs.sort(key=lambda x: (simplex_dim(x[0]), x[0], x[1]))
    return pairs


def build_orientation(
    pairs_all: Sequence[Pair],
    matched: Set[Pair],
    cells: Sequence[Simplex],
) -> Dict[Simplex, List[Simplex]]:
    """按 Forman 方向规则构建有向图。

    - 未匹配关联边：高维 -> 低维
    - 匹配关联边：低维 -> 高维
    """
    adj: Dict[Simplex, List[Simplex]] = {c: [] for c in cells}
    matched_set = set(matched)
    for alpha, beta in pairs_all:
        if (alpha, beta) in matched_set:
            adj[alpha].append(beta)
        else:
            adj[beta].append(alpha)
    return adj


def has_cycle(adj: Dict[Simplex, List[Simplex]]) -> bool:
    """DFS 检测有向环。"""
    color: Dict[Simplex, int] = {node: 0 for node in adj}  # 0未访问,1访问中,2完成

    def dfs(u: Simplex) -> bool:
        color[u] = 1
        for v in adj[u]:
            if color[v] == 1:
                return True
            if color[v] == 0 and dfs(v):
                return True
        color[u] = 2
        return False

    for node in adj:
        if color[node] == 0 and dfs(node):
            return True
    return False


def greedy_acyclic_matching(cells_by_dim: Dict[int, List[Simplex]]) -> Set[Pair]:
    """贪心构建离散向量场：只接受不引入有向环的匹配。"""
    all_cells = [c for d in sorted(cells_by_dim) for c in cells_by_dim[d]]
    all_pairs = incidence_pairs(cells_by_dim)
    matched: Set[Pair] = set()
    used_cells: Set[Simplex] = set()

    for alpha, beta in all_pairs:
        if alpha in used_cells or beta in used_cells:
            continue

        tentative = set(matched)
        tentative.add((alpha, beta))
        adj = build_orientation(all_pairs, tentative, all_cells)

        if not has_cycle(adj):
            matched.add((alpha, beta))
            used_cells.add(alpha)
            used_cells.add(beta)

    return matched


def critical_cells(cells: Iterable[Simplex], matched: Set[Pair]) -> Dict[int, List[Simplex]]:
    matched_cells: Set[Simplex] = set()
    for a, b in matched:
        matched_cells.add(a)
        matched_cells.add(b)

    crit: Dict[int, List[Simplex]] = {}
    for c in sorted(cells):
        if c not in matched_cells:
            d = simplex_dim(c)
            crit.setdefault(d, []).append(c)
    return crit


def boundary_matrix_mod2(
    high_cells: Sequence[Simplex],
    low_cells: Sequence[Simplex],
) -> np.ndarray:
    """构造 Z2 下的边界矩阵 B: C_k -> C_{k-1}。"""
    low_index = {c: i for i, c in enumerate(low_cells)}
    mat = np.zeros((len(low_cells), len(high_cells)), dtype=np.uint8)
    for j, beta in enumerate(high_cells):
        for alpha in faces_of(beta):
            i = low_index.get(alpha)
            if i is not None:
                mat[i, j] ^= 1
    return mat


def rank_mod2(mat: np.ndarray) -> int:
    """高斯消元计算 GF(2) 秩。"""
    a = mat.copy() % 2
    rows, cols = a.shape
    r = 0
    c = 0
    while r < rows and c < cols:
        pivot = None
        for i in range(r, rows):
            if a[i, c] == 1:
                pivot = i
                break
        if pivot is None:
            c += 1
            continue
        if pivot != r:
            a[[r, pivot]] = a[[pivot, r]]
        for i in range(rows):
            if i != r and a[i, c] == 1:
                a[i, :] ^= a[r, :]
        r += 1
        c += 1
    return r


def betti_numbers(cells_by_dim: Dict[int, List[Simplex]]) -> List[int]:
    max_dim = max(cells_by_dim)
    n = [len(cells_by_dim.get(k, [])) for k in range(max_dim + 1)]

    # rank_B[k] 表示 B_k: C_k -> C_{k-1} 的秩。B_0 视作 0。
    rank_B = [0] * (max_dim + 2)
    for k in range(1, max_dim + 1):
        high = cells_by_dim.get(k, [])
        low = cells_by_dim.get(k - 1, [])
        bm = boundary_matrix_mod2(high, low)
        rank_B[k] = rank_mod2(bm)

    betti = []
    for k in range(max_dim + 1):
        beta_k = n[k] - rank_B[k] - rank_B[k + 1]
        betti.append(beta_k)
    return betti


def euler_characteristic_from_counts(counts: Sequence[int]) -> int:
    return sum(((-1) ** k) * v for k, v in enumerate(counts))


def analyze_complex(name: str, maximal_simplices: Sequence[Simplex]) -> None:
    cells = closure_of_maximal(maximal_simplices)
    cells_by_dim = group_cells_by_dim(cells)
    max_dim = max(cells_by_dim)

    matched = greedy_acyclic_matching(cells_by_dim)
    critical = critical_cells(cells, matched)
    betti = betti_numbers(cells_by_dim)

    c_counts = [len(critical.get(k, [])) for k in range(max_dim + 1)]
    n_counts = [len(cells_by_dim.get(k, [])) for k in range(max_dim + 1)]

    weak_ok = [c_counts[k] >= betti[k] for k in range(max_dim + 1)]
    euler_cells = euler_characteristic_from_counts(n_counts)
    euler_critical = euler_characteristic_from_counts(c_counts)

    print(f"\n=== {name} ===")
    print(f"最大维度: {max_dim}")
    print(f"各维胞元数 n_k: {n_counts}")
    print(f"匹配对数量: {len(matched)}")
    print(f"各维临界胞元 c_k: {c_counts}")
    print(f"Betti 数 beta_k: {betti}")
    print(f"弱 Morse 不等式 c_k >= beta_k: {weak_ok}")
    print(f"Euler(胞元计数) = {euler_cells}, Euler(临界计数) = {euler_critical}")


def main() -> None:
    examples = [
        (
            "S^1 的三角剖分（仅边界环）",
            [(0, 1), (1, 2), (0, 2)],
        ),
        (
            "2-单形（可收缩圆盘）",
            [(0, 1, 2)],
        ),
        (
            "四面体边界（S^2）",
            [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
        ),
    ]

    for name, maximal in examples:
        analyze_complex(name, maximal)


if __name__ == "__main__":
    main()
