"""有限差分法（抛物型）最小可运行示例。

求解一维热方程：
    u_t = alpha * u_xx, x in (0, 1), t > 0
边界条件：
    u(0, t) = u(1, t) = 0
初值：
    u(x, 0) = sin(pi x)

使用显式 FTCS 差分格式，并和解析解对比误差。
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def initial_condition(x: np.ndarray) -> np.ndarray:
    """初值 u(x,0) = sin(pi x)。"""
    return np.sin(np.pi * x)


def exact_solution(x: np.ndarray, t: float, alpha: float) -> np.ndarray:
    """对应初边值问题的解析解。"""
    return np.exp(-alpha * (np.pi ** 2) * t) * np.sin(np.pi * x)


def solve_heat_ftcs(
    alpha: float,
    nx: int,
    t_end: float,
    target_r: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    """显式 FTCS 解热方程。

    r = alpha * dt / dx^2 需要满足 r <= 0.5 才稳定。
    返回：x, u_num(t_end), r_actual, dt, n_steps
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if nx < 3:
        raise ValueError("nx must be >= 3")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if target_r <= 0.0:
        raise ValueError("target_r must be positive")

    dx = 1.0 / nx
    x = np.linspace(0.0, 1.0, nx + 1)

    dt_guess = target_r * dx * dx / alpha
    n_steps = max(1, math.ceil(t_end / dt_guess))
    dt = t_end / n_steps
    r_actual = alpha * dt / (dx * dx)

    if r_actual > 0.5 + 1e-12:
        raise ValueError(
            f"unstable FTCS setup: r={r_actual:.6f} > 0.5, "
            "decrease target_r or increase n_steps"
        )

    u = initial_condition(x)
    u[0] = 0.0
    u[-1] = 0.0

    for _ in range(n_steps):
        u_next = u.copy()
        u_next[1:-1] = u[1:-1] + r_actual * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        u = u_next

    return x, u, r_actual, dt, n_steps


def compute_errors(u_num: np.ndarray, u_ref: np.ndarray, dx: float) -> Dict[str, float]:
    """计算 L1 / L2 / Linf 误差。"""
    err = u_num - u_ref
    return {
        "l1": float(np.sum(np.abs(err)) * dx),
        "l2": float(np.sqrt(np.sum(err * err) * dx)),
        "linf": float(np.max(np.abs(err))),
    }


def main() -> None:
    alpha = 1.0
    nx = 80
    t_end = 0.05
    target_r = 0.45

    x, u_num, r_actual, dt, n_steps = solve_heat_ftcs(
        alpha=alpha,
        nx=nx,
        t_end=t_end,
        target_r=target_r,
    )

    u_ref = exact_solution(x, t_end, alpha)
    dx = 1.0 / nx
    errors = compute_errors(u_num, u_ref, dx)

    energy_num = float(np.sum(u_num * u_num) * dx)
    energy_ref = float(np.sum(u_ref * u_ref) * dx)

    center_idx = nx // 2

    print("=== Finite Difference (Parabolic / Heat Equation) ===")
    print(f"alpha       : {alpha}")
    print(f"nx          : {nx}")
    print(f"t_end       : {t_end}")
    print(f"n_steps     : {n_steps}")
    print(f"dx          : {dx:.8f}")
    print(f"dt          : {dt:.8f}")
    print(f"r_actual    : {r_actual:.8f} (must <= 0.5)")
    print("--- Errors against exact solution ---")
    print(f"L1          : {errors['l1']:.8e}")
    print(f"L2          : {errors['l2']:.8e}")
    print(f"Linf        : {errors['linf']:.8e}")
    print("--- Sanity checks ---")
    print(f"u_num(0.5)  : {u_num[center_idx]:.8f}")
    print(f"u_ref(0.5)  : {u_ref[center_idx]:.8f}")
    print(f"energy_num  : {energy_num:.8e}")
    print(f"energy_ref  : {energy_ref:.8e}")


if __name__ == "__main__":
    main()
