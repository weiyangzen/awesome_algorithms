"""降阶法（Reduction of Order）最小可运行示例。

目标方程采用二阶线性齐次 ODE:
    y'' + p(x) y' + q(x) y = 0
在已知一个非零解 y1(x) 的前提下，使用降阶公式构造第二个线性无关解：
    y2(x) = y1(x) * ∫ [exp(-∫p)/y1^2] dx
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid


Array = np.ndarray


@dataclass
class ReductionResult:
    x: Array
    p: Array
    q: Array
    y1: Array
    y2: Array
    integral_p: Array
    integrand: Array
    integral_integrand: Array
    wronskian: Array
    residual_y1: Array
    residual_y2: Array


def _check_grid(x: Array) -> None:
    if x.ndim != 1:
        raise ValueError("x_grid 必须是一维数组")
    if x.size < 3:
        raise ValueError("x_grid 至少需要 3 个点")
    if not np.all(np.isfinite(x)):
        raise ValueError("x_grid 含有非有限值")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("x_grid 必须严格递增")


def _first_second_derivative(x: Array, y: Array) -> tuple[Array, Array]:
    dy = np.gradient(y, x, edge_order=2)
    ddy = np.gradient(dy, x, edge_order=2)
    return dy, ddy


def reduction_of_order_second_solution(
    x_grid: Array,
    p_fn: Callable[[Array], Array],
    q_fn: Callable[[Array], Array],
    y1_fn: Callable[[Array], Array],
    y1_floor: float = 1e-10,
) -> ReductionResult:
    """在给定 y1 的条件下，使用降阶法构造第二解 y2。"""
    if y1_floor <= 0.0:
        raise ValueError("y1_floor 必须为正数")

    x = np.asarray(x_grid, dtype=float)
    _check_grid(x)

    p = np.asarray(p_fn(x), dtype=float)
    q = np.asarray(q_fn(x), dtype=float)
    y1 = np.asarray(y1_fn(x), dtype=float)

    for name, arr in (("p", p), ("q", q), ("y1", y1)):
        if arr.shape != x.shape:
            raise ValueError(f"{name} 的形状必须与 x_grid 一致")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} 含有非有限值")

    if np.min(np.abs(y1)) <= y1_floor:
        raise ValueError("y1 在网格上过于接近 0，无法稳定构造积分核")

    integral_p = cumulative_trapezoid(p, x, initial=0.0)
    integrand = np.exp(-integral_p) / (y1 * y1)
    integral_integrand = cumulative_trapezoid(integrand, x, initial=0.0)
    y2 = y1 * integral_integrand

    y1_prime, y1_second = _first_second_derivative(x, y1)
    y2_prime, y2_second = _first_second_derivative(x, y2)

    residual_y1 = y1_second + p * y1_prime + q * y1
    residual_y2 = y2_second + p * y2_prime + q * y2
    wronskian = y1 * y2_prime - y1_prime * y2

    return ReductionResult(
        x=x,
        p=p,
        q=q,
        y1=y1,
        y2=y2,
        integral_p=integral_p,
        integrand=integrand,
        integral_integrand=integral_integrand,
        wronskian=wronskian,
        residual_y1=residual_y1,
        residual_y2=residual_y2,
    )


def make_report_table(result: ReductionResult, sample_points: int = 7) -> pd.DataFrame:
    if sample_points < 2:
        raise ValueError("sample_points 必须至少为 2")
    idx = np.linspace(0, result.x.size - 1, sample_points, dtype=int)
    return pd.DataFrame(
        {
            "x": result.x[idx],
            "y1": result.y1[idx],
            "y2": result.y2[idx],
            "Wronskian": result.wronskian[idx],
            "residual_y2": result.residual_y2[idx],
        }
    )


def main() -> None:
    # 示例方程：y'' - y = 0
    # 已知解：y1 = e^x
    # 理论上降阶可得 y2 = sinh(x)（与 e^x 线性无关）
    x = np.linspace(0.0, 2.0, 401)

    p_fn = lambda t: np.zeros_like(t)
    q_fn = lambda t: -np.ones_like(t)
    y1_fn = np.exp

    result = reduction_of_order_second_solution(x, p_fn=p_fn, q_fn=q_fn, y1_fn=y1_fn)

    y2_expected = np.sinh(x)
    rel_l2_error = np.linalg.norm(result.y2 - y2_expected) / (
        np.linalg.norm(y2_expected) + 1e-12
    )

    # 数值微分在边界精度较差，评估指标时忽略两端少量点
    interior = slice(20, -20)
    max_residual_y1 = float(np.max(np.abs(result.residual_y1[interior])))
    max_residual_y2 = float(np.max(np.abs(result.residual_y2[interior])))
    min_wronskian_abs = float(np.min(np.abs(result.wronskian[interior])))

    df = make_report_table(result)
    print("=== Reduction of Order MVP ===")
    print("ODE: y'' - y = 0")
    print("Known solution y1(x): exp(x)")
    print("Constructed second solution y2(x) = y1(x) * integral(exp(-integral(p))/y1^2)")
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v: .6e}"))
    print()
    print(f"relative L2 error to sinh(x): {rel_l2_error:.6e}")
    print(f"max |residual(y1)| on interior: {max_residual_y1:.6e}")
    print(f"max |residual(y2)| on interior: {max_residual_y2:.6e}")
    print(f"min |Wronskian| on interior:    {min_wronskian_abs:.6e}")

    ok = (
        rel_l2_error < 5e-4
        and max_residual_y1 < 5e-3
        and max_residual_y2 < 5e-3
        and min_wronskian_abs > 1e-2
    )
    print(f"PASS: {ok}")
    if not ok:
        raise RuntimeError("降阶法验证未通过，请检查离散精度或实现")


if __name__ == "__main__":
    main()
