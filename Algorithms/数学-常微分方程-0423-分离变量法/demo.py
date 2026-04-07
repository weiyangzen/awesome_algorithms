"""分离变量法（Separation of Variables）最小可运行示例。

目标问题：
    dy/dx = f(x) * g(y),  y(x0)=y0

通过变量分离得到
    ∫_{y0}^{y(x)} 1/g(s) ds = ∫_{x0}^{x} f(t) dt
本实现使用数值积分 + 单调反函数插值恢复 y(x)，并与 SciPy 参考解对照。
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
class SeparableCase:
    name: str
    f: Func1D
    g: Func1D
    y0: float
    x_grid: Array
    y_bounds: tuple[float, float]
    exact_solution: Func1D


@dataclass
class SeparationResult:
    name: str
    x: Array
    y_sep: Array
    y_scipy: Array
    y_exact: Array
    abs_error_to_scipy: Array
    abs_error_to_exact: Array
    residual_numeric: Array
    relative_l2_error_to_scipy: float
    relative_l2_error_to_exact: float
    max_abs_residual: float
    mse_to_scipy: float
    torch_max_abs_error_to_scipy: float
    y_range: tuple[float, float]


def _check_grid(x: Array) -> None:
    if x.ndim != 1:
        raise ValueError("x_grid 必须是一维数组")
    if x.size < 3:
        raise ValueError("x_grid 至少包含 3 个点")
    if not np.all(np.isfinite(x)):
        raise ValueError("x_grid 含有非有限值")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("x_grid 必须严格递增")


def _first_derivative(x: Array, y: Array) -> Array:
    return np.gradient(y, x, edge_order=2)


def _build_monotone_integral_map(
    g: Func1D,
    y_bounds: tuple[float, float],
    n_points: int,
    g_floor: float,
) -> tuple[Array, Array]:
    y_min, y_max = y_bounds
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min >= y_max:
        raise ValueError("y_bounds 非法，要求有限且 y_min < y_max")
    if n_points < 32:
        raise ValueError("n_points 太小，至少为 32")
    if g_floor <= 0.0:
        raise ValueError("g_floor 必须为正数")

    y_grid = np.linspace(y_min, y_max, n_points, dtype=float)
    g_values = np.asarray(g(y_grid), dtype=float)
    if g_values.shape != y_grid.shape:
        raise ValueError("g(y) 输出形状必须与 y_grid 一致")
    if not np.all(np.isfinite(g_values)):
        raise ValueError("g(y) 含有非有限值")
    if np.any(np.abs(g_values) < g_floor):
        raise ValueError("给定 y_bounds 内 g(y) 过小或变号，无法稳定分离")

    inv_g = 1.0 / g_values
    h_grid = cumulative_trapezoid(inv_g, y_grid, initial=0.0)

    if h_grid[-1] < h_grid[0]:
        h_grid = h_grid[::-1]
        y_grid = y_grid[::-1]

    if not np.all(np.diff(h_grid) > 0.0):
        raise ValueError("H(y) 非严格单调，无法做反函数插值")
    return y_grid, h_grid


def solve_by_separation(
    *,
    name: str,
    f: Func1D,
    g: Func1D,
    y0: float,
    x_grid: Array,
    y_bounds: tuple[float, float],
    exact_solution: Func1D,
    y_map_points: int = 50_001,
    g_floor: float = 1e-10,
) -> SeparationResult:
    x = np.asarray(x_grid, dtype=float)
    _check_grid(x)
    if not (y_bounds[0] < y0 < y_bounds[1]):
        raise ValueError("初值 y0 必须位于 y_bounds 内部")

    f_values = np.asarray(f(x), dtype=float)
    if f_values.shape != x.shape:
        raise ValueError("f(x) 输出形状必须与 x_grid 一致")
    if not np.all(np.isfinite(f_values)):
        raise ValueError("f(x) 含有非有限值")

    y_grid, h_grid = _build_monotone_integral_map(
        g=g,
        y_bounds=y_bounds,
        n_points=y_map_points,
        g_floor=g_floor,
    )
    h_y0 = float(np.interp(y0, y_grid, h_grid))

    rhs_integral = cumulative_trapezoid(f_values, x, initial=0.0)
    targets = h_y0 + rhs_integral

    h_min = float(h_grid[0])
    h_max = float(h_grid[-1])
    if np.any(targets < h_min) or np.any(targets > h_max):
        raise ValueError(
            "目标积分值超出 H(y) 映射范围。请扩大 y_bounds 或缩小 x_grid 区间。"
        )

    y_sep = np.interp(targets, h_grid, y_grid)
    y_prime_num = _first_derivative(x, y_sep)
    residual = y_prime_num - f_values * np.asarray(g(y_sep), dtype=float)

    def rhs(xi: float, yi: Array) -> Array:
        x_arr = np.array([xi], dtype=float)
        y_arr = np.array([yi[0]], dtype=float)
        return np.array([f(x_arr)[0] * g(y_arr)[0]], dtype=float)

    ivp = solve_ivp(
        rhs,
        (float(x[0]), float(x[-1])),
        np.array([y0], dtype=float),
        t_eval=x,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not ivp.success:
        raise RuntimeError(f"SciPy 参考解失败: {ivp.message}")
    y_scipy = np.asarray(ivp.y[0], dtype=float)

    y_exact = np.asarray(exact_solution(x), dtype=float)
    if y_exact.shape != x.shape or not np.all(np.isfinite(y_exact)):
        raise ValueError("exact_solution(x) 输出非法")

    abs_err_scipy = np.abs(y_sep - y_scipy)
    abs_err_exact = np.abs(y_sep - y_exact)

    rel_l2_scipy = float(
        np.linalg.norm(y_sep - y_scipy) / (np.linalg.norm(y_scipy) + 1e-12)
    )
    rel_l2_exact = float(
        np.linalg.norm(y_sep - y_exact) / (np.linalg.norm(y_exact) + 1e-12)
    )

    interior = slice(8, -8) if x.size > 17 else slice(0, x.size)
    max_abs_residual = float(np.max(np.abs(residual[interior])))

    mse_val = float(mean_squared_error(y_scipy, y_sep))
    torch_max_abs = float(
        torch.max(
            torch.abs(
                torch.tensor(y_sep, dtype=torch.float64)
                - torch.tensor(y_scipy, dtype=torch.float64)
            )
        ).item()
    )

    return SeparationResult(
        name=name,
        x=x,
        y_sep=y_sep,
        y_scipy=y_scipy,
        y_exact=y_exact,
        abs_error_to_scipy=abs_err_scipy,
        abs_error_to_exact=abs_err_exact,
        residual_numeric=residual,
        relative_l2_error_to_scipy=rel_l2_scipy,
        relative_l2_error_to_exact=rel_l2_exact,
        max_abs_residual=max_abs_residual,
        mse_to_scipy=mse_val,
        torch_max_abs_error_to_scipy=torch_max_abs,
        y_range=(float(np.min(y_sep)), float(np.max(y_sep))),
    )


def make_report_table(result: SeparationResult, sample_points: int = 7) -> pd.DataFrame:
    if sample_points < 2:
        raise ValueError("sample_points 必须至少为 2")
    idx = np.linspace(0, result.x.size - 1, sample_points, dtype=int)
    return pd.DataFrame(
        {
            "x": result.x[idx],
            "y_sep": result.y_sep[idx],
            "y_scipy": result.y_scipy[idx],
            "y_exact": result.y_exact[idx],
            "abs_err_scipy": result.abs_error_to_scipy[idx],
            "abs_err_exact": result.abs_error_to_exact[idx],
            "residual": result.residual_numeric[idx],
        }
    )


def build_cases() -> list[SeparableCase]:
    case1_x = np.linspace(0.0, 1.2, 601)
    case2_x = np.linspace(0.0, 4.0, 801)
    case3_x = np.linspace(0.0, 0.7, 501)

    return [
        SeparableCase(
            name="Variable-rate exponential",
            f=lambda x: x,
            g=lambda y: y,
            y0=1.0,
            x_grid=case1_x,
            y_bounds=(0.05, 3.5),
            exact_solution=lambda x: np.exp(0.5 * x**2),
        ),
        SeparableCase(
            name="Logistic growth",
            f=lambda x: np.ones_like(x),
            g=lambda y: y * (1.0 - y),
            y0=0.2,
            x_grid=case2_x,
            y_bounds=(1e-4, 1.0 - 1e-4),
            exact_solution=lambda x: 1.0 / (1.0 + 4.0 * np.exp(-x)),
        ),
        SeparableCase(
            name="Arctan/tangent pair",
            f=lambda x: 1.0 + x**2,
            g=lambda y: 1.0 + y**2,
            y0=0.0,
            x_grid=case3_x,
            y_bounds=(-4.0, 4.0),
            exact_solution=lambda x: np.tan(x + x**3 / 3.0),
        ),
    ]


def main() -> None:
    print("=== Separation of Variables MVP ===")
    summary_rows: list[dict[str, float | str]] = []
    for case in build_cases():
        result = solve_by_separation(
            name=case.name,
            f=case.f,
            g=case.g,
            y0=case.y0,
            x_grid=case.x_grid,
            y_bounds=case.y_bounds,
            exact_solution=case.exact_solution,
        )
        table = make_report_table(result, sample_points=7)

        print()
        print(f"[{result.name}]")
        print(f"y range (separated): [{result.y_range[0]:.6f}, {result.y_range[1]:.6f}]")
        print(table.to_string(index=False, float_format=lambda v: f"{v: .6e}"))
        print(f"relative L2 error to SciPy: {result.relative_l2_error_to_scipy:.6e}")
        print(f"relative L2 error to exact: {result.relative_l2_error_to_exact:.6e}")
        print(f"max |residual| (interior):  {result.max_abs_residual:.6e}")
        print(f"MSE to SciPy (sklearn):     {result.mse_to_scipy:.6e}")
        print(f"max |error| (torch):        {result.torch_max_abs_error_to_scipy:.6e}")

        summary_rows.append(
            {
                "name": result.name,
                "rel_l2_scipy": result.relative_l2_error_to_scipy,
                "rel_l2_exact": result.relative_l2_error_to_exact,
                "max_abs_residual": result.max_abs_residual,
                "mse_scipy": result.mse_to_scipy,
                "torch_max_abs": result.torch_max_abs_error_to_scipy,
            }
        )

    summary = pd.DataFrame(summary_rows)
    print()
    print("=== Summary ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v: .6e}"))

    ok = bool(
        (summary["rel_l2_scipy"] < 2e-4).all()
        and (summary["rel_l2_exact"] < 3e-4).all()
        and (summary["max_abs_residual"] < 3e-3).all()
    )
    print(f"PASS: {ok}")
    if not ok:
        raise RuntimeError("分离变量法数值验证未通过，请检查参数或实现")


if __name__ == "__main__":
    main()
