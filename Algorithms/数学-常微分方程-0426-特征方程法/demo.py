"""特征方程法（Characteristic Equation Method）最小可运行示例。

目标问题：二阶线性常系数齐次常微分方程
    a y'' + b y' + c y = 0,  a != 0

通过特征方程
    a r^2 + b r + c = 0
判断根型并构造解析解，再与 SciPy 数值解做一致性校验。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


Array = np.ndarray


@dataclass
class CharacteristicResult:
    name: str
    a: float
    b: float
    c: float
    x: Array
    y: Array
    y_prime: Array
    y_second_numeric: Array
    residual_numeric: Array
    y_scipy: Array
    abs_error_to_scipy: Array
    relative_l2_error: float
    max_abs_residual: float
    root_case: str
    roots: tuple[complex, complex]
    c1: float
    c2: float


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


def _solve_distinct_real(
    x: Array, r1: float, r2: float, y0: float, dy0: float
) -> tuple[Array, Array, float, float]:
    # y = c1*e^(r1 x) + c2*e^(r2 x)
    c1 = (dy0 - r2 * y0) / (r1 - r2)
    c2 = y0 - c1
    exp1 = np.exp(r1 * x)
    exp2 = np.exp(r2 * x)
    y = c1 * exp1 + c2 * exp2
    y_prime = c1 * r1 * exp1 + c2 * r2 * exp2
    return y, y_prime, float(c1), float(c2)


def _solve_repeated_real(
    x: Array, r: float, y0: float, dy0: float
) -> tuple[Array, Array, float, float]:
    # y = (c1 + c2*x) * e^(r x)
    c1 = y0
    c2 = dy0 - r * y0
    expv = np.exp(r * x)
    y = (c1 + c2 * x) * expv
    y_prime = (c2 + r * (c1 + c2 * x)) * expv
    return y, y_prime, float(c1), float(c2)


def _solve_complex_conjugate(
    x: Array, alpha: float, beta: float, y0: float, dy0: float
) -> tuple[Array, Array, float, float]:
    # y = e^(alpha x) * [c1 cos(beta x) + c2 sin(beta x)]
    c1 = y0
    c2 = (dy0 - alpha * c1) / beta
    expv = np.exp(alpha * x)
    cosv = np.cos(beta * x)
    sinv = np.sin(beta * x)

    base = c1 * cosv + c2 * sinv
    dbase = -c1 * beta * sinv + c2 * beta * cosv
    y = expv * base
    y_prime = expv * (alpha * base + dbase)
    return y, y_prime, float(c1), float(c2)


def _solve_reference_with_scipy(
    a: float,
    b: float,
    c: float,
    y0: float,
    dy0: float,
    x: Array,
) -> Array:
    def rhs(_: float, state: Array) -> Array:
        y_val, v_val = state
        return np.array([v_val, -(b * v_val + c * y_val) / a], dtype=float)

    sol = solve_ivp(
        rhs,
        (float(x[0]), float(x[-1])),
        np.array([y0, dy0], dtype=float),
        t_eval=x,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"SciPy 参考解失败: {sol.message}")
    return np.asarray(sol.y[0], dtype=float)


def solve_by_characteristic_equation(
    *,
    name: str,
    a: float,
    b: float,
    c: float,
    y0: float,
    dy0: float,
    x_grid: Array,
    root_tol: float = 1e-12,
) -> CharacteristicResult:
    """对二阶常系数齐次 ODE 使用特征方程法求解。"""
    if abs(a) < 1e-15:
        raise ValueError("系数 a 不能为 0")
    if root_tol <= 0.0:
        raise ValueError("root_tol 必须为正数")

    x = np.asarray(x_grid, dtype=float)
    _check_grid(x)

    disc = b * b - 4.0 * a * c
    if disc > root_tol:
        sqrt_disc = float(np.sqrt(disc))
        r1 = (-b + sqrt_disc) / (2.0 * a)
        r2 = (-b - sqrt_disc) / (2.0 * a)
        y, y_prime, c1, c2 = _solve_distinct_real(x, r1, r2, y0, dy0)
        root_case = "distinct_real"
        roots = (complex(r1, 0.0), complex(r2, 0.0))
    elif abs(disc) <= root_tol:
        r = (-b) / (2.0 * a)
        y, y_prime, c1, c2 = _solve_repeated_real(x, r, y0, dy0)
        root_case = "repeated_real"
        roots = (complex(r, 0.0), complex(r, 0.0))
    else:
        alpha = (-b) / (2.0 * a)
        beta = np.sqrt(-disc) / (2.0 * abs(a))
        y, y_prime, c1, c2 = _solve_complex_conjugate(x, alpha, beta, y0, dy0)
        root_case = "complex_conjugate"
        roots = (complex(alpha, beta), complex(alpha, -beta))

    y_first_num, y_second_num = _first_second_derivative(x, y)
    residual = a * y_second_num + b * y_first_num + c * y

    y_scipy = _solve_reference_with_scipy(a=a, b=b, c=c, y0=y0, dy0=dy0, x=x)
    abs_error = np.abs(y - y_scipy)
    relative_l2_error = float(
        np.linalg.norm(y - y_scipy) / (np.linalg.norm(y_scipy) + 1e-12)
    )
    interior = slice(10, -10) if x.size > 21 else slice(0, x.size)
    max_abs_residual = float(np.max(np.abs(residual[interior])))

    return CharacteristicResult(
        name=name,
        a=float(a),
        b=float(b),
        c=float(c),
        x=x,
        y=y,
        y_prime=y_prime,
        y_second_numeric=y_second_num,
        residual_numeric=residual,
        y_scipy=y_scipy,
        abs_error_to_scipy=abs_error,
        relative_l2_error=relative_l2_error,
        max_abs_residual=max_abs_residual,
        root_case=root_case,
        roots=roots,
        c1=c1,
        c2=c2,
    )


def make_report_table(result: CharacteristicResult, sample_points: int = 6) -> pd.DataFrame:
    if sample_points < 2:
        raise ValueError("sample_points 必须至少为 2")
    idx = np.linspace(0, result.x.size - 1, sample_points, dtype=int)
    return pd.DataFrame(
        {
            "x": result.x[idx],
            "y_char": result.y[idx],
            "y_scipy": result.y_scipy[idx],
            "abs_err": result.abs_error_to_scipy[idx],
            "residual": result.residual_numeric[idx],
        }
    )


def main() -> None:
    x = np.linspace(0.0, 3.0, 601)
    scenarios = [
        {
            "name": "Distinct roots",
            "a": 1.0,
            "b": -3.0,
            "c": 2.0,  # r=1,2
            "y0": 1.0,
            "dy0": 0.0,
        },
        {
            "name": "Repeated root",
            "a": 1.0,
            "b": -2.0,
            "c": 1.0,  # r=1 (double)
            "y0": 1.0,
            "dy0": 2.0,
        },
        {
            "name": "Complex roots",
            "a": 1.0,
            "b": 2.0,
            "c": 5.0,  # r=-1 ± 2i
            "y0": 1.0,
            "dy0": -1.0,
        },
    ]

    summary_rows: list[dict[str, float | str]] = []
    print("=== Characteristic Equation Method MVP ===")
    for case in scenarios:
        result = solve_by_characteristic_equation(x_grid=x, **case)
        table = make_report_table(result, sample_points=6)

        print()
        print(f"[{result.name}]  {result.a} y'' + {result.b} y' + {result.c} y = 0")
        print(f"root_case: {result.root_case}")
        print(
            f"roots: {result.roots[0].real:.6f}{result.roots[0].imag:+.6f}j, "
            f"{result.roots[1].real:.6f}{result.roots[1].imag:+.6f}j"
        )
        print(f"constants: c1={result.c1:.6f}, c2={result.c2:.6f}")
        print(table.to_string(index=False, float_format=lambda v: f"{v: .6e}"))
        print(f"relative L2 error to SciPy: {result.relative_l2_error:.6e}")
        print(f"max |residual| (interior):  {result.max_abs_residual:.6e}")

        summary_rows.append(
            {
                "name": result.name,
                "root_case": result.root_case,
                "relative_l2_error": result.relative_l2_error,
                "max_abs_residual": result.max_abs_residual,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print()
    print("=== Summary ===")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v: .6e}"))

    ok = bool(
        (summary_df["relative_l2_error"] < 1e-5).all()
        and (summary_df["max_abs_residual"] < 2e-2).all()
    )
    print(f"PASS: {ok}")
    if not ok:
        raise RuntimeError("特征方程法验证未通过，请检查实现或离散精度")


if __name__ == "__main__":
    main()
