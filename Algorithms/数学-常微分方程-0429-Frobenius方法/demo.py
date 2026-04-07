"""Frobenius方法最小可运行示例。

示例方程选用 Bessel 方程：
    x^2 y'' + x y' + (x^2 - nu^2) y = 0
该方程在 x=0 为正则奇点，适合 Frobenius 级数法。

这里实现：
1) 指标方程与根；
2) 递推系数求解（手写，不依赖黑盒 ODE 求解器）；
3) 截断级数求值；
4) 与 scipy.special.jv 的参考解对比和残差验证。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import special


Array = np.ndarray


@dataclass
class FrobeniusResult:
    nu: float
    x: Array
    n_terms: int
    indicial_roots: tuple[float, float]
    r_used: float
    coefficients: Array
    y_series: Array
    y_reference: Array
    abs_error: Array
    residual: Array


def _check_grid(x: Array) -> None:
    if x.ndim != 1:
        raise ValueError("x 必须是一维网格")
    if x.size < 5:
        raise ValueError("x 至少需要 5 个点")
    if not np.all(np.isfinite(x)):
        raise ValueError("x 含有非有限值")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("x 必须严格递增")
    if np.min(x) <= 0.0:
        raise ValueError("Bessel Frobenius 示例要求 x > 0（避免奇点数值问题）")


def indicial_roots_bessel(nu: float) -> tuple[float, float]:
    """Bessel 方程的指标方程为 r^2 - nu^2 = 0。"""
    return (nu, -nu)


def frobenius_coefficients_bessel_j(
    nu: float,
    n_terms: int,
    a0: float | None = None,
) -> tuple[tuple[float, float], float, Array]:
    """构造与 J_nu 对应的一组 Frobenius 系数。

    对 Bessel 方程有递推：
        a_k * ((k+r)^2 - nu^2) + a_{k-2} = 0,  k >= 2
    选择 r = nu，对应第一类 Bessel 函数 J_nu 的 Frobenius 支。
    """
    if nu < 0.0:
        raise ValueError("本 MVP 约定 nu >= 0")
    if n_terms < 2:
        raise ValueError("n_terms 至少为 2")

    roots = indicial_roots_bessel(nu)
    r = roots[0]

    if a0 is None:
        # 归一化选择，使级数与 J_nu 在 x -> 0 的主导项一致。
        a0 = 1.0 / (2.0**nu * special.gamma(nu + 1.0))

    coeffs = np.zeros(n_terms + 1, dtype=float)
    coeffs[0] = float(a0)

    denom_k1 = (1.0 + r) * (1.0 + r) - nu * nu
    if abs(denom_k1) < 1e-14:
        raise ValueError("k=1 项发生退化，当前简化实现无法处理")
    coeffs[1] = 0.0

    for k in range(2, n_terms + 1):
        denom = (k + r) * (k + r) - nu * nu
        if abs(denom) < 1e-14:
            raise ValueError(f"k={k} 的递推分母过小，无法稳定计算")
        coeffs[k] = -coeffs[k - 2] / denom

    return roots, r, coeffs


def evaluate_frobenius_series(x: Array, r: float, coeffs: Array) -> Array:
    """计算 y(x)=sum_{k=0}^N a_k x^{k+r}。"""
    k = np.arange(coeffs.size, dtype=float)
    powers = np.power(x[:, None], k[None, :] + r)
    return powers @ coeffs


def solve_bessel_with_frobenius(nu: float, x: Array, n_terms: int) -> FrobeniusResult:
    x = np.asarray(x, dtype=float)
    _check_grid(x)

    roots, r, coeffs = frobenius_coefficients_bessel_j(nu=nu, n_terms=n_terms)
    y_series = evaluate_frobenius_series(x, r=r, coeffs=coeffs)
    y_reference = special.jv(nu, x)
    abs_error = np.abs(y_series - y_reference)

    dy = np.gradient(y_series, x, edge_order=2)
    ddy = np.gradient(dy, x, edge_order=2)
    residual = x * x * ddy + x * dy + (x * x - nu * nu) * y_series

    return FrobeniusResult(
        nu=nu,
        x=x,
        n_terms=n_terms,
        indicial_roots=roots,
        r_used=r,
        coefficients=coeffs,
        y_series=y_series,
        y_reference=y_reference,
        abs_error=abs_error,
        residual=residual,
    )


def make_report_table(result: FrobeniusResult, sample_points: int = 8) -> pd.DataFrame:
    if sample_points < 2:
        raise ValueError("sample_points 至少为 2")
    idx = np.linspace(0, result.x.size - 1, sample_points, dtype=int)
    return pd.DataFrame(
        {
            "x": result.x[idx],
            "y_series": result.y_series[idx],
            "y_reference": result.y_reference[idx],
            "abs_error": result.abs_error[idx],
            "residual": result.residual[idx],
        }
    )


def main() -> None:
    nu = 0.5
    n_terms = 32
    x = np.linspace(1e-4, 8.0, 600)

    result = solve_bessel_with_frobenius(nu=nu, x=x, n_terms=n_terms)

    rel_l2_error = np.linalg.norm(result.y_series - result.y_reference) / (
        np.linalg.norm(result.y_reference) + 1e-12
    )
    max_abs_error = float(np.max(result.abs_error))

    # 数值微分在两端误差较大，残差评估忽略两端。
    interior = slice(30, -30)
    max_abs_residual = float(np.max(np.abs(result.residual[interior])))

    nonzero_coeff_idx = np.where(np.abs(result.coefficients) > 1e-14)[0]
    preview_coeff_idx = nonzero_coeff_idx[:8]

    print("=== Frobenius Method MVP ===")
    print("ODE: x^2 y'' + x y' + (x^2 - nu^2) y = 0 (Bessel)")
    print(f"nu = {result.nu:.3f}, truncation N = {result.n_terms}")
    print(
        "indicial roots = "
        f"({result.indicial_roots[0]:.3f}, {result.indicial_roots[1]:.3f}), "
        f"selected r = {result.r_used:.3f}"
    )
    print()
    print("first non-zero coefficients (k, a_k):")
    for k in preview_coeff_idx:
        print(f"  k={int(k):2d}, a_k={result.coefficients[k]: .8e}")

    print()
    df = make_report_table(result)
    print(df.to_string(index=False, float_format=lambda v: f"{v: .6e}"))

    print()
    print(f"relative L2 error vs scipy.special.jv: {rel_l2_error:.6e}")
    print(f"max absolute error on grid:           {max_abs_error:.6e}")
    print(f"max |ODE residual| on interior:       {max_abs_residual:.6e}")

    ok = rel_l2_error < 1e-6 and max_abs_error < 1e-6 and max_abs_residual < 2e-3
    print(f"PASS: {ok}")
    if not ok:
        raise RuntimeError("Frobenius MVP validation failed")


if __name__ == "__main__":
    main()
