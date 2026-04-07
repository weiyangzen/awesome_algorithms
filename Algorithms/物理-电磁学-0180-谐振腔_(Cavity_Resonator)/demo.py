"""矩形谐振腔（Cavity Resonator）最小可运行示例。

问题：二维矩形理想导体谐振腔中的 TM_z 模式。
连续模型（归约到横向平面）：
    -∇_t^2 E_z = k_c^2 E_z,
边界条件（PEC）：
    E_z = 0 on boundary.

离散方法：二维五点差分 + 稀疏特征值求解，得到最低若干本征频率，
并与解析 TM_mn 频率进行对照。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class CavityConfig:
    """矩形谐振腔配置参数。"""

    a: float = 0.22  # x 方向长度（米）
    b: float = 0.10  # y 方向长度（米）
    c: float = 299_792_458.0  # 真空光速（m/s）
    nx: int = 80  # x 方向内部网格点数（不含边界）
    ny: int = 40  # y 方向内部网格点数（不含边界）
    num_modes: int = 6  # 计算前几个本征模
    max_mode_index: int = 8  # 解析模态枚举上限


@dataclass(frozen=True)
class AnalyticMode:
    """解析 TM 模式信息。"""

    m: int
    n: int
    lambda_mn: float
    frequency_hz: float


def _laplacian_1d_dirichlet(n: int, h: float) -> sp.csr_matrix:
    """构造 1D Dirichlet Laplacian（三对角）。"""
    main = -2.0 * np.ones(n) / (h * h)
    off = np.ones(n - 1) / (h * h)
    return sp.diags(diagonals=[off, main, off], offsets=[-1, 0, 1], format="csr")


def build_negative_laplacian_2d(config: CavityConfig) -> Tuple[sp.csr_matrix, float, float]:
    """构造二维 -Laplace 稀疏矩阵（Dirichlet 边界）。"""
    if config.nx < 2 or config.ny < 2:
        raise ValueError("nx and ny must be >= 2")

    hx = config.a / (config.nx + 1)
    hy = config.b / (config.ny + 1)

    lx = _laplacian_1d_dirichlet(config.nx, hx)
    ly = _laplacian_1d_dirichlet(config.ny, hy)

    ix = sp.eye(config.nx, format="csr")
    iy = sp.eye(config.ny, format="csr")

    laplace = sp.kron(iy, lx, format="csr") + sp.kron(ly, ix, format="csr")
    a_matrix = -laplace
    return a_matrix, hx, hy


def analytic_tm_modes(config: CavityConfig, count: int) -> List[AnalyticMode]:
    """枚举矩形腔 TM_mn 解析模式并按频率升序返回前 count 项。"""
    modes: List[AnalyticMode] = []
    for m in range(1, config.max_mode_index + 1):
        for n in range(1, config.max_mode_index + 1):
            lambda_mn = (m * math.pi / config.a) ** 2 + (n * math.pi / config.b) ** 2
            frequency_hz = (config.c / (2.0 * math.pi)) * math.sqrt(lambda_mn)
            modes.append(AnalyticMode(m=m, n=n, lambda_mn=lambda_mn, frequency_hz=frequency_hz))

    modes.sort(key=lambda mode: mode.lambda_mn)
    return modes[:count]


def solve_cavity_modes(config: CavityConfig) -> Dict[str, object]:
    """求解离散本征问题并与解析解比较。"""
    if config.a <= 0.0 or config.b <= 0.0:
        raise ValueError("a and b must be positive")
    if config.c <= 0.0:
        raise ValueError("c must be positive")
    if config.num_modes < 1:
        raise ValueError("num_modes must be >= 1")

    a_matrix, hx, hy = build_negative_laplacian_2d(config)
    system_size = a_matrix.shape[0]

    # eigsh 要求 k < N。
    k = min(config.num_modes, system_size - 1)
    if k < 1:
        raise ValueError("system too small for eigen decomposition")

    eigenvalues, eigenvectors = spla.eigsh(a_matrix, k=k, which="SM", tol=1e-10)
    order = np.argsort(eigenvalues)
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    numerical_freq_hz = (config.c / (2.0 * math.pi)) * np.sqrt(eigenvalues)

    analytic = analytic_tm_modes(config, count=k)
    analytic_freq_hz = np.array([mode.frequency_hz for mode in analytic], dtype=float)

    relative_errors = np.abs(numerical_freq_hz - analytic_freq_hz) / analytic_freq_hz

    gram = eigenvectors.T @ eigenvectors
    diag_deviation = float(np.max(np.abs(np.diag(gram) - 1.0)))
    off_diag = gram - np.diag(np.diag(gram))
    orthogonality_offdiag_max = float(np.max(np.abs(off_diag)))

    first_mode = eigenvectors[:, 0].reshape((config.ny, config.nx), order="C")
    first_mode_with_boundary = np.zeros((config.ny + 2, config.nx + 2), dtype=float)
    first_mode_with_boundary[1:-1, 1:-1] = first_mode
    boundary_values = np.concatenate(
        [
            first_mode_with_boundary[0, :],
            first_mode_with_boundary[-1, :],
            first_mode_with_boundary[:, 0],
            first_mode_with_boundary[:, -1],
        ]
    )
    boundary_max_abs = float(np.max(np.abs(boundary_values)))

    return {
        "hx": hx,
        "hy": hy,
        "system_size": system_size,
        "k": k,
        "eigenvalues": eigenvalues,
        "numerical_freq_hz": numerical_freq_hz,
        "analytic_modes": analytic,
        "analytic_freq_hz": analytic_freq_hz,
        "relative_errors": relative_errors,
        "diag_deviation": diag_deviation,
        "orthogonality_offdiag_max": orthogonality_offdiag_max,
        "first_mode_l2": float(np.linalg.norm(first_mode)),
        "boundary_max_abs": boundary_max_abs,
    }


def main() -> None:
    config = CavityConfig()
    result = solve_cavity_modes(config)

    print("=== Cavity Resonator | 2D Rectangular TM Modes (FDM Eigenproblem) ===")
    print(f"a, b (m)                 : {config.a:.6f}, {config.b:.6f}")
    print(f"nx, ny (interior)        : {config.nx}, {config.ny}")
    print(f"hx, hy (m)               : {result['hx']:.6e}, {result['hy']:.6e}")
    print(f"sparse system size       : {result['system_size']}")
    print(f"modes solved             : {result['k']}")
    print("--- Frequency comparison ---")
    print("idx  mode(m,n)   numerical(GHz)   analytic(GHz)   rel_error")

    numerical_ghz = np.asarray(result["numerical_freq_hz"]) / 1e9
    analytic_ghz = np.asarray(result["analytic_freq_hz"]) / 1e9
    rel_err = np.asarray(result["relative_errors"])
    analytic_modes: List[AnalyticMode] = result["analytic_modes"]  # type: ignore[assignment]

    for i in range(int(result["k"])):
        mode = analytic_modes[i]
        print(
            f"{i + 1:>3d}  ({mode.m:>1d},{mode.n:>1d})"
            f"{numerical_ghz[i]:>15.6f}{analytic_ghz[i]:>15.6f}{rel_err[i]:>12.3e}"
        )

    print("--- Eigenvector quality checks ---")
    print(f"max|diag(V^T V)-1|        : {result['diag_deviation']:.3e}")
    print(f"max|offdiag(V^T V)|       : {result['orthogonality_offdiag_max']:.3e}")
    print(f"first mode interior L2    : {result['first_mode_l2']:.6e}")
    print(f"first mode boundary max   : {result['boundary_max_abs']:.3e} (Dirichlet should be 0)")


if __name__ == "__main__":
    main()
