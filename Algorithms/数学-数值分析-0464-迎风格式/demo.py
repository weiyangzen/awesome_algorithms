"""迎风格式（Upwind Scheme）最小可运行示例。

求解一维线性对流方程：u_t + a u_x = 0（周期边界）。
运行后会输出：
1) 光滑初值的网格收敛结果（L1/Linf + 估计阶）
2) 间断初值的极值变化（观察单调性与数值耗散）
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np


Array = np.ndarray


def initial_smooth(x: Array) -> Array:
    """光滑初值：用于误差和收敛阶评估。"""
    return np.sin(2.0 * np.pi * x) + 0.5 * np.sin(4.0 * np.pi * x)


def initial_step(x: Array) -> Array:
    """间断初值：用于观察迎风格式的单调性与耗散。"""
    return np.where((x >= 0.2) & (x <= 0.5), 1.0, 0.0)


def exact_shift_periodic(x: Array, t: float, a: float, init_fn: Callable[[Array], Array]) -> Array:
    """解析解：周期域上的平移 u(x,t)=u0((x-a t) mod 1)。"""
    x0 = np.mod(x - a * t, 1.0)
    return init_fn(x0)


def upwind_periodic(u0: Array, a: float, dx: float, dt: float, n_steps: int) -> Array:
    """一阶迎风格式（统一写法）+ 周期边界。"""
    nu = a * dt / dx
    if abs(nu) > 1.0 + 1e-12:
        raise ValueError(f"CFL violated: |nu|={abs(nu):.6f} > 1")

    nu_plus = max(nu, 0.0)
    nu_minus = min(nu, 0.0)

    u = u0.astype(float).copy()
    for _ in range(n_steps):
        u_left = np.roll(u, 1)
        u_right = np.roll(u, -1)
        u = u - nu_plus * (u - u_left) - nu_minus * (u_right - u)
    return u


def run_case(
    nx: int,
    final_time: float,
    a: float,
    cfl_target: float,
    init_fn: Callable[[Array], Array],
) -> Dict[str, float]:
    """运行单个网格规模的计算并返回误差信息。"""
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    dx = 1.0 / nx

    dt_guess = cfl_target * dx / abs(a)
    n_steps = max(1, int(math.ceil(final_time / dt_guess)))
    dt = final_time / n_steps
    cfl_effective = abs(a) * dt / dx

    u0 = init_fn(x)
    u_num = upwind_periodic(u0=u0, a=a, dx=dx, dt=dt, n_steps=n_steps)
    u_ex = exact_shift_periodic(x=x, t=final_time, a=a, init_fn=init_fn)

    err = np.abs(u_num - u_ex)
    return {
        "nx": float(nx),
        "dx": dx,
        "dt": dt,
        "cfl": cfl_effective,
        "l1": float(np.mean(err)),
        "linf": float(np.max(err)),
        "u_min": float(np.min(u_num)),
        "u_max": float(np.max(u_num)),
    }


def estimate_orders(errors: List[float]) -> List[float]:
    """用相邻网格误差估计收敛阶 p = log2(e_h/e_{h/2})。"""
    orders: List[float] = [float("nan")]
    for i in range(1, len(errors)):
        prev_e = errors[i - 1]
        curr_e = errors[i]
        if prev_e <= 0.0 or curr_e <= 0.0:
            orders.append(float("nan"))
        else:
            orders.append(float(np.log2(prev_e / curr_e)))
    return orders


def print_convergence_table(results: List[Dict[str, float]]) -> None:
    """打印收敛结果表。"""
    l1_orders = estimate_orders([r["l1"] for r in results])
    linf_orders = estimate_orders([r["linf"] for r in results])

    print("=== Smooth Initial Condition: Convergence of Upwind Scheme ===")
    header = "{:>6} {:>10} {:>10} {:>10} {:>12} {:>10} {:>12} {:>10}".format(
        "Nx", "dx", "dt", "CFL", "L1", "ord(L1)", "Linf", "ord(Linf)"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results):
        print(
            "{:6d} {:10.4e} {:10.4e} {:10.4f} {:12.4e} {:10.4f} {:12.4e} {:10.4f}".format(
                int(r["nx"]),
                r["dx"],
                r["dt"],
                r["cfl"],
                r["l1"],
                l1_orders[i],
                r["linf"],
                linf_orders[i],
            )
        )


def run_step_demo(nx: int, final_time: float, a: float, cfl_target: float) -> None:
    """间断初值演示：观察极值是否保持在合理范围。"""
    r = run_case(nx=nx, final_time=final_time, a=a, cfl_target=cfl_target, init_fn=initial_step)

    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    u0 = initial_step(x)
    print("\n=== Discontinuous Initial Condition: Monotonicity Check ===")
    print(f"Nx={nx}, CFL={r['cfl']:.4f}, T={final_time:.3f}")
    print(f"Initial min/max: {u0.min():.6f} / {u0.max():.6f}")
    print(f"Final   min/max: {r['u_min']:.6f} / {r['u_max']:.6f}")
    print("Note: Upwind is typically non-oscillatory under CFL<=1, but diffusive near discontinuities.")


def main() -> None:
    a = 1.0
    final_time = 0.5
    cfl_target = 0.8
    grid_list = [50, 100, 200, 400]

    results = [
        run_case(nx=nx, final_time=final_time, a=a, cfl_target=cfl_target, init_fn=initial_smooth)
        for nx in grid_list
    ]
    print_convergence_table(results)

    run_step_demo(nx=400, final_time=0.3, a=a, cfl_target=cfl_target)


if __name__ == "__main__":
    main()
