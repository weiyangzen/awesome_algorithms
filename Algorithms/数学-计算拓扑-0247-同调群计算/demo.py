"""同调群计算：最小可运行 MVP（模 2 系数）。

实现范围：
1) 从极大单纯形构造有限单纯复形闭包；
2) 构造边界矩阵 d_k: C_k -> C_{k-1}（GF(2)）；
3) 通过 GF(2) 高斯消元计算秩并得到 Betti 数；
4) 以 H_k ≅ (Z/2Z)^beta_k 的形式输出同调群摘要。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

Simplex = Tuple[int, ...]
ComplexByDim = Dict[int, List[Simplex]]
BoundaryMap = Dict[int, np.ndarray]


@dataclass
class HomologyResult:
    name: str
    simplex_counts: List[int]
    boundary_ranks: List[int]
    betti_numbers: List[int]
    homology_groups: List[str]
    boundary_consistent: bool


def simplex_dim(simplex: Simplex) -> int:
    return len(simplex) - 1


def closure_of_maximal(maximal_simplices: Sequence[Simplex]) -> List[Simplex]:
    """从极大单纯形生成闭包，返回排序后的所有非空单纯形。"""
    all_simplices = set()
    for simplex in maximal_simplices:
        ordered = tuple(sorted(simplex))
        if len(ordered) == 0:
            raise ValueError("simplex must be non-empty")
        for r in range(1, len(ordered) + 1):
            for face in combinations(ordered, r):
                all_simplices.add(face)
    return sorted(all_simplices, key=lambda s: (len(s), s))


def group_simplices_by_dim(simplices: Iterable[Simplex]) -> ComplexByDim:
    grouped: ComplexByDim = {}
    for simplex in simplices:
        dim = simplex_dim(simplex)
        grouped.setdefault(dim, []).append(simplex)
    for dim in grouped:
        grouped[dim].sort()
    return grouped


def faces_of(simplex: Simplex) -> List[Simplex]:
    """返回 simplex 的所有余维 1 面。"""
    if len(simplex) <= 1:
        return []
    return [tuple(simplex[:i] + simplex[i + 1 :]) for i in range(len(simplex))]


def boundary_matrix_mod2(higher: Sequence[Simplex], lower: Sequence[Simplex]) -> np.ndarray:
    """构造边界矩阵 d_k: C_k -> C_{k-1}（模 2）。"""
    matrix = np.zeros((len(lower), len(higher)), dtype=np.uint8)
    if not higher or not lower:
        return matrix

    lower_index = {simplex: i for i, simplex in enumerate(lower)}
    for col, simplex in enumerate(higher):
        for face in faces_of(simplex):
            row = lower_index.get(face)
            if row is not None:
                matrix[row, col] ^= 1
    return matrix


def gf2_rank(matrix: np.ndarray) -> int:
    """GF(2) 高斯消元求秩。"""
    a = np.asarray(matrix, dtype=np.uint8).copy()
    n_rows, n_cols = a.shape
    rank = 0
    pivot_row = 0

    for col in range(n_cols):
        pivot = None
        for row in range(pivot_row, n_rows):
            if a[row, col] == 1:
                pivot = row
                break

        if pivot is None:
            continue

        if pivot != pivot_row:
            a[[pivot_row, pivot], :] = a[[pivot, pivot_row], :]

        for row in range(n_rows):
            if row != pivot_row and a[row, col] == 1:
                a[row, :] ^= a[pivot_row, :]

        rank += 1
        pivot_row += 1
        if pivot_row == n_rows:
            break

    return rank


def build_boundary_maps(simplices_by_dim: ComplexByDim) -> BoundaryMap:
    """构造全部边界矩阵 d_k（k>=1）。"""
    max_dim = max(simplices_by_dim)
    boundaries: BoundaryMap = {}
    for k in range(1, max_dim + 1):
        higher = simplices_by_dim.get(k, [])
        lower = simplices_by_dim.get(k - 1, [])
        boundaries[k] = boundary_matrix_mod2(higher, lower)
    return boundaries


def compute_betti_numbers(simplices_by_dim: ComplexByDim, boundaries: BoundaryMap) -> Tuple[List[int], List[int]]:
    """返回 (betti, boundary_ranks)。"""
    max_dim = max(simplices_by_dim)
    counts = [len(simplices_by_dim.get(k, [])) for k in range(max_dim + 1)]

    ranks = [0] * (max_dim + 2)
    for k in range(1, max_dim + 1):
        ranks[k] = gf2_rank(boundaries[k])

    betti: List[int] = []
    for k in range(max_dim + 1):
        beta_k = counts[k] - ranks[k] - ranks[k + 1]
        betti.append(int(beta_k))

    return betti, ranks[:-1]


def verify_boundary_square_zero(boundaries: BoundaryMap) -> bool:
    """检查 d_{k-1} * d_k == 0（模 2）。"""
    if not boundaries:
        return True

    max_k = max(boundaries)
    for k in range(2, max_k + 1):
        left = boundaries[k - 1]
        right = boundaries[k]
        if left.shape[1] != right.shape[0]:
            return False
        product = (left @ right) % 2
        if np.any(product):
            return False
    return True


def format_homology_groups(betti: Sequence[int]) -> List[str]:
    """把 Betti 数格式化为同调群摘要。"""
    groups: List[str] = []
    for k, beta in enumerate(betti):
        if beta == 0:
            groups.append(f"H_{k} = 0")
        elif beta == 1:
            groups.append(f"H_{k} ≅ Z/2Z")
        else:
            groups.append(f"H_{k} ≅ (Z/2Z)^{beta}")
    return groups


def analyze_complex(name: str, maximal_simplices: Sequence[Simplex]) -> HomologyResult:
    simplices = closure_of_maximal(maximal_simplices)
    simplices_by_dim = group_simplices_by_dim(simplices)
    max_dim = max(simplices_by_dim)

    boundaries = build_boundary_maps(simplices_by_dim)
    betti, ranks = compute_betti_numbers(simplices_by_dim, boundaries)
    counts = [len(simplices_by_dim.get(k, [])) for k in range(max_dim + 1)]
    homology = format_homology_groups(betti)
    boundary_ok = verify_boundary_square_zero(boundaries)

    return HomologyResult(
        name=name,
        simplex_counts=counts,
        boundary_ranks=ranks,
        betti_numbers=betti,
        homology_groups=homology,
        boundary_consistent=boundary_ok,
    )


def print_result(result: HomologyResult) -> None:
    print(f"\n=== {result.name} ===")
    print(f"simplex counts n_k: {result.simplex_counts}")

    rank_desc = []
    for k, rank in enumerate(result.boundary_ranks):
        if k == 0:
            continue
        rank_desc.append(f"rank(d_{k})={rank}")
    print("boundary ranks: " + (", ".join(rank_desc) if rank_desc else "(no non-trivial boundaries)"))

    print(f"betti beta_k: {result.betti_numbers}")
    print("homology summary:")
    for line in result.homology_groups:
        print(f"  - {line}")
    print(f"boundary consistency d_(k-1)d_k=0: {result.boundary_consistent}")


def main() -> None:
    examples: List[Tuple[str, List[Simplex]]] = [
        (
            "S^1 (triangle boundary)",
            [(0, 1), (1, 2), (0, 2)],
        ),
        (
            "Filled triangle (contractible 2-simplex)",
            [(0, 1, 2)],
        ),
        (
            "S^2 (tetrahedron boundary)",
            [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
        ),
        (
            "Disjoint union S^1 ⊔ S^1",
            [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)],
        ),
    ]

    for name, maximal in examples:
        result = analyze_complex(name, maximal)
        print_result(result)


if __name__ == "__main__":
    main()
