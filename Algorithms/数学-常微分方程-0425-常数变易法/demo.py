"""常数变易法（Variation of Parameters）最小可运行示例。

目标问题：二阶线性非齐次常微分方程（标准型）
    y'' + p(x) y' + q(x) y = g(x)

已知齐次方程一组基解 y1, y2（以及导数 y1', y2'），
通过常数变易法构造特解并叠加齐次解，再与 SciPy 参考解对照。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from scipy.integrate import cumulative_trapezoid, solve_ivp
from sklearn.metrics import mean_squared_error


Array = np.ndarray
Func1D = Callable[[Array], Array]


@dataclass
class VariationCase:
    name: str
    p: Func1D
    q: Func1D
    g: Func1D
    y1: Func1D
    y1_prime: Func1D
    y2: Func1D
    y2_prime: Func1D
    y0: float
    dy0: float
    x_grid: Array
    exact_solution: Func1D


@dataclass
class VariationResult:
    name: str
    x: Array
    y: Array
    y_scipy: Array
    y_exact: Array
    y_h: Array
    y_p: Array
    wronskian: Array
    c1: float
    c2: float
    abs_error_to_scipy: Array
    abs_error_to_exact: Array
    residual_numeric: Array
    rel_l2_to_scipy: float
    rel_l2_to_exact: float
    max_abs_residual: float
    mse_to_scipy: float
    torch_max_abs_to_scipy: float


def _check_grid(x: Array) -> None:
    if x.ndim != 1:
        raise ValueError("x_grid 必须是一维数组")
    if x.size < 3:
        raise ValueError("x_grid 至少需要 3 个点")
    if not np.all(np.isfinite(x)):
        raise ValueError("x_grid 含有非有限值")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("x_grid 必须严格递增")


def _eval_on_grid(func: Func1D, x: Array, name: str) -> Array:
    values = np.asarray(func(x), dtype=float)
    if values.shape != x.shape:
        raise ValueError(f"{name}(x) 输出形状必须与 x_grid 一致")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}(x) 含有非有限值")
    return values


def _first_second_derivative(x: Array, y: Array) -> tuple[Array, Array]:
    dy = np.gradient(y, x, edge_order=2)
    ddy = np.gradient(dy, x, edge_order=2)
    return dy, ddy


def _solve_reference_with_scipy(
    *,
    p: Func1D,
    q: Func1D,
    g: Func1D,
    y0: float,
    dy0: float,
    x_grid: Array,
) -> Array:
    def rhs(xi: float, state: Array) -> Array:
        yv, vv = state
        xi_arr = np.array([xi], dtype=float)
        px = float(p(xi_arr)[0])
        qx = float(q(xi_arr)[0])
        gx = float(g(xi_arr)[0])
        return np.array([vv, gx - px * vv - qx * yv], dtype=float)

    ivp = solve_ivp(
        rhs,
        (float(x_grid[0]), float(x_grid[-1])),
        np.array([y0, dy0], dtype=float),
        t_eval=x_grid,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not ivp.success:
        raise RuntimeError(f"SciPy 参考解失败: {ivp.message}")
    return np.asarray(ivp.y[0], dtype=float)


def solve_by_variation_of_parameters(
    *,
    name: str,
    p: Func1D,
    q: Func1D,
    g: Func1D,
    y1: Func1D,
    y1_prime: Func1D,
    y2: Func1D,
    y2_prime: Func1D,
    y0: float,
    dy0: float,
    x_grid: Array,
    exact_solution: Func1D,
    wronskian_floor: float = 1e-10,
) -> VariationResult:
    if wronskian_floor <= 0.0:
        raise ValueError("wronskian_floor 必须为正数")

    x = np.asarray(x_grid, dtype=float)
    _check_grid(x)

    p_values = _eval_on_grid(p, x, "p")
    q_values = _eval_on_grid(q, x, "q")
    g_values = _eval_on_grid(g, x, "g")

    y1_values = _eval_on_grid(y1, x, "y1")
    y1p_values = _eval_on_grid(y1_prime, x, "y1_prime")
    y2_values = _eval_on_grid(y2, x, "y2")
    y2p_values = _eval_on_grid(y2_prime, x, "y2_prime")

    wronskian = y1_values * y2p_values - y1p_values * y2_values
    if np.any(np.abs(wronskian) < wronskian_floor):
        raise ValueError("Wronskian 过小，基解可能线性相关或数值不稳定")

    u1_prime = -(y2_values * g_values) / wronskian
    u2_prime = (y1_values * g_values) / wronskian

    # 从 x0 累积积分，令 u1(x0)=u2(x0)=0，得到一个固定特解。
    u1 = cumulative_trapezoid(u1_prime, x, initial=0.0)
    u2 = cumulative_trapezoid(u2_prime, x, initial=0.0)
    y_p = u1 * y1_values + u2 * y2_values

    # 用初值解齐次部分系数 c1, c2。
    a = np.array(
        [[y1_values[0], y2_values[0]], [y1p_values[0], y2p_values[0]]],
        dtype=float,
    )
    b = np.array([y0, dy0], dtype=float)
    if abs(np.linalg.det(a)) < wronskian_floor:
        raise ValueError("初值点基解矩阵不可逆，无法确定 c1/c2")
    c1, c2 = np.linalg.solve(a, b)

    y_h = c1 * y1_values + c2 * y2_values
    y_total = y_h + y_p

    y_prime_num, y_second_num = _first_second_derivative(x, y_total)
    residual = y_second_num + p_values * y_prime_num + q_values * y_total - g_values

    y_scipy = _solve_reference_with_scipy(
        p=p,
        q=q,
        g=g,
        y0=y0,
        dy0=dy0,
        x_grid=x,
    )

    y_exact = _eval_on_grid(exact_solution, x, "exact_solution")

    abs_err_scipy = np.abs(y_total - y_scipy)
    abs_err_exact = np.abs(y_total - y_exact)

    rel_l2_scipy = float(
        np.linalg.norm(y_total - y_scipy) / (np.linalg.norm(y_scipy) + 1e-12)
    )
    rel_l2_exact = float(
        np.linalg.norm(y_total - y_exact) / (np.linalg.norm(y_exact) + 1e-12)
    )

    interior = slice(8, -8) if x.size > 17 else slice(0, x.size)
    max_abs_residual = float(np.max(np.abs(residual[interior])))

    mse_val = float(mean_squared_error(y_scipy, y_total))
    torch_max_abs = float(
        torch.max(
            torch.abs(
                torch.tensor(y_total, dtype=torch.float64)
                - torch.tensor(y_scipy, dtype=torch.float64)
            )
        ).item()
    )

    return VariationResult(
        name=name,
        x=x,
        y=y_total,
        y_scipy=y_scipy,
        y_exact=y_exact,
        y_h=y_h,
        y_p=y_p,
        wronskian=wronskian,
        c1=float(c1),
        c2=float(c2),
        abs_error_to_scipy=abs_err_scipy,
        abs_error_to_exact=abs_err_exact,
        residual_numeric=residual,
        rel_l2_to_scipy=rel_l2_scipy,
        rel_l2_to_exact=rel_l2_exact,
        max_abs_residual=max_abs_residual,
        mse_to_scipy=mse_val,
        torch_max_abs_to_scipy=torch_max_abs,
    )


def make_report_table(result: VariationResult, sample_points: int = 7) -> pd.DataFrame:
    if sample_points < 2:
        raise ValueError("sample_points 至少为 2")
    idx = np.linspace(0, result.x.size - 1, sample_points, dtype=int)
    return pd.DataFrame(
        {
            "x": result.x[idx],
            "y_var": result.y[idx],
            "y_scipy": result.y_scipy[idx],
            "y_exact": result.y_exact[idx],
            "abs_err_scipy": result.abs_error_to_scipy[idx],
            "abs_err_exact": result.abs_error_to_exact[idx],
            "residual": result.residual_numeric[idx],
        }
    )


def build_cases() -> list[VariationCase]:
    case1_x = np.linspace(0.0, 2.0, 801)
    case2_x = np.linspace(0.0, 2.0, 801)
    case3_x = np.linspace(0.0, 2.2, 901)

    return [
        VariationCase(
            name="Exponential resonance",
            p=lambda x: np.zeros_like(x),
            q=lambda x: -np.ones_like(x),
            g=lambda x: np.exp(x),
            y1=lambda x: np.exp(x),
            y1_prime=lambda x: np.exp(x),
            y2=lambda x: np.exp(-x),
            y2_prime=lambda x: -np.exp(-x),
            y0=1.0,
            dy0=0.0,
            x_grid=case1_x,
            exact_solution=lambda x: 0.25 * np.exp(x)
            + 0.75 * np.exp(-x)
            + 0.5 * x * np.exp(x),
        ),
        VariationCase(
            name="Sine forcing with resonance",
            p=lambda x: np.zeros_like(x),
            q=lambda x: np.ones_like(x),
            g=lambda x: np.sin(x),
            y1=lambda x: np.cos(x),
            y1_prime=lambda x: -np.sin(x),
            y2=lambda x: np.sin(x),
            y2_prime=lambda x: np.cos(x),
            y0=0.0,
            dy0=1.0,
            x_grid=case2_x,
            exact_solution=lambda x: 1.5 * np.sin(x) - 0.5 * x * np.cos(x),
        ),
        VariationCase(
            name="Repeated-root homogeneous basis",
            p=lambda x: 2.0 * np.ones_like(x),
            q=lambda x: np.ones_like(x),
            g=lambda x: x,
            y1=lambda x: np.exp(-x),
            y1_prime=lambda x: -np.exp(-x),
            y2=lambda x: x * np.exp(-x),
            y2_prime=lambda x: (1.0 - x) * np.exp(-x),
            y0=1.0,
            dy0=-1.0,
            x_grid=case3_x,
            exact_solution=lambda x: (3.0 + x) * np.exp(-x) + x - 2.0,
        ),
    ]


def main() -> None:
    print("=== Variation of Parameters MVP ===")

    results: list[VariationResult] = []
    for case in build_cases():
        result = solve_by_variation_of_parameters(
            name=case.name,
            p=case.p,
            q=case.q,
            g=case.g,
            y1=case.y1,
            y1_prime=case.y1_prime,
            y2=case.y2,
            y2_prime=case.y2_prime,
            y0=case.y0,
            dy0=case.dy0,
            x_grid=case.x_grid,
            exact_solution=case.exact_solution,
        )
        results.append(result)

        print()
        print(f"[{result.name}]")
        print(f"c1={result.c1:.6f}, c2={result.c2:.6f}")
        print(
            "Wronskian range: "
            f"[{np.min(result.wronskian):.6e}, {np.max(result.wronskian):.6e}]"
        )
        table = make_report_table(result, sample_points=7)
        print(table.to_string(index=False, float_format=lambda v: f"{v: .6e}"))
        print(f"relative L2 error to SciPy: {result.rel_l2_to_scipy:.6e}")
        print(f"relative L2 error to exact: {result.rel_l2_to_exact:.6e}")
        print(f"max abs residual (interior): {result.max_abs_residual:.6e}")
        print(f"MSE to SciPy (sklearn): {result.mse_to_scipy:.6e}")
        print(f"torch max abs to SciPy: {result.torch_max_abs_to_scipy:.6e}")

    summary = pd.DataFrame(
        {
            "case": [r.name for r in results],
            "rel_l2_scipy": [r.rel_l2_to_scipy for r in results],
            "rel_l2_exact": [r.rel_l2_to_exact for r in results],
            "max_abs_residual": [r.max_abs_residual for r in results],
            "mse_scipy": [r.mse_to_scipy for r in results],
            "torch_max_abs": [r.torch_max_abs_to_scipy for r in results],
        }
    )

    pass_all = bool(
        (summary["rel_l2_scipy"] < 5e-5).all()
        and (summary["rel_l2_exact"] < 8e-5).all()
        and (summary["max_abs_residual"] < 6e-2).all()
    )

    print()
    print("=== Summary ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v: .6e}"))
    print(f"PASS: {pass_all}")

    if not pass_all:
        raise RuntimeError("Validation failed: metrics exceeded thresholds")


if __name__ == "__main__":
    main()
