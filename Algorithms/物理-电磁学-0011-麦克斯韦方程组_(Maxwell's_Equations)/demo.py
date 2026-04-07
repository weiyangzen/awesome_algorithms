"""麦克斯韦方程组（2D TMz, Yee-FDTD）最小可运行示例。

该示例在无源、均匀介质 + 周期边界下，数值推进 Maxwell 方程：
    dEx/dt =  (1/epsilon) dHz/dy
    dEy/dt = -(1/epsilon) dHz/dx
    dHz/dt = -(1/mu) (dEy/dx - dEx/dy)

并使用解析平面波解做误差与守恒检查。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class MaxwellConfig:
    """仿真参数。"""

    epsilon: float = 1.0
    mu: float = 1.0
    lx: float = 1.0
    ly: float = 1.0
    nx: int = 80
    ny: int = 64
    mode_x: int = 2
    mode_y: int = 1
    hz_amplitude: float = 1.0
    t_end: float = 0.4
    courant_factor: float = 0.7


def derive_constants(config: MaxwellConfig) -> Dict[str, float]:
    """推导波速、阻抗、波数和角频率。"""
    c = 1.0 / math.sqrt(config.mu * config.epsilon)
    impedance = math.sqrt(config.mu / config.epsilon)
    kx = 2.0 * math.pi * config.mode_x / config.lx
    ky = 2.0 * math.pi * config.mode_y / config.ly
    k_norm = math.hypot(kx, ky)
    omega = c * k_norm
    return {
        "c": c,
        "impedance": impedance,
        "kx": kx,
        "ky": ky,
        "k_norm": k_norm,
        "omega": omega,
    }


def build_staggered_grids(config: MaxwellConfig) -> Dict[str, Array]:
    """构造 TMz Yee 交错网格坐标。"""
    dx = config.lx / config.nx
    dy = config.ly / config.ny

    x_idx = np.arange(config.nx, dtype=np.float64)
    y_idx = np.arange(config.ny, dtype=np.float64)

    x_ex = x_idx * dx
    y_ex = (y_idx + 0.5) * dy
    ex_x, ex_y = np.meshgrid(x_ex, y_ex, indexing="ij")

    x_ey = (x_idx + 0.5) * dx
    y_ey = y_idx * dy
    ey_x, ey_y = np.meshgrid(x_ey, y_ey, indexing="ij")

    x_hz = (x_idx + 0.5) * dx
    y_hz = (y_idx + 0.5) * dy
    hz_x, hz_y = np.meshgrid(x_hz, y_hz, indexing="ij")

    return {
        "dx": np.array(dx),
        "dy": np.array(dy),
        "ex_x": ex_x,
        "ex_y": ex_y,
        "ey_x": ey_x,
        "ey_y": ey_y,
        "hz_x": hz_x,
        "hz_y": hz_y,
    }


def exact_tmz_fields(
    x: Array,
    y: Array,
    t: float,
    *,
    epsilon: float,
    omega: float,
    kx: float,
    ky: float,
    hz_amplitude: float,
    field: str,
) -> Array:
    """给出满足无源 Maxwell 方程的 TMz 解析平面波。"""
    phase = kx * x + ky * y - omega * t
    s = np.sin(phase)

    if field == "Hz":
        return hz_amplitude * s
    if field == "Ex":
        return -(ky * hz_amplitude / (epsilon * omega)) * s
    if field == "Ey":
        return (kx * hz_amplitude / (epsilon * omega)) * s
    raise ValueError("field must be one of: Ex, Ey, Hz")


def compute_l2_relative_error(num: Array, ref: Array) -> float:
    """计算相对 L2 误差。"""
    diff = num - ref
    denom = max(1e-14, float(np.linalg.norm(ref.ravel(), ord=2)))
    return float(np.linalg.norm(diff.ravel(), ord=2) / denom)


def electromagnetic_energy(ex: Array, ey: Array, hz: Array, epsilon: float, mu: float, dx: float, dy: float) -> float:
    """计算离散总电磁能。"""
    density = epsilon * (ex * ex + ey * ey) + mu * (hz * hz)
    return 0.5 * float(np.sum(density) * dx * dy)


def divergence_electric(ex: Array, ey: Array, dx: float, dy: float) -> Array:
    """在 Hz 位置计算离散 div(E)。"""
    d_ex_dx = (np.roll(ex, -1, axis=0) - ex) / dx
    d_ey_dy = (np.roll(ey, -1, axis=1) - ey) / dy
    return d_ex_dx + d_ey_dy


def run_tmx_fdtd(config: MaxwellConfig) -> Dict[str, float]:
    """执行 2D TMz Yee-FDTD 并返回验证指标。"""
    if config.epsilon <= 0.0 or config.mu <= 0.0:
        raise ValueError("epsilon and mu must be positive")
    if config.lx <= 0.0 or config.ly <= 0.0:
        raise ValueError("lx and ly must be positive")
    if config.nx < 8 or config.ny < 8:
        raise ValueError("nx and ny must be >= 8")
    if config.mode_x == 0 and config.mode_y == 0:
        raise ValueError("mode_x and mode_y cannot both be zero")
    if config.t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if config.courant_factor <= 0.0:
        raise ValueError("courant_factor must be positive")

    constants = derive_constants(config)
    grids = build_staggered_grids(config)

    dx = float(grids["dx"])
    dy = float(grids["dy"])
    c = constants["c"]

    dt_limit = 1.0 / (c * math.sqrt((1.0 / dx**2) + (1.0 / dy**2)))
    dt_guess = config.courant_factor * dt_limit
    n_steps = max(1, math.ceil(config.t_end / dt_guess))
    dt = config.t_end / n_steps

    courant_2d = c * dt * math.sqrt((1.0 / dx**2) + (1.0 / dy**2))
    if courant_2d > 1.0 + 1e-12:
        raise ValueError(f"unstable setup: 2D Courant={courant_2d:.6f} > 1.0")

    kx = constants["kx"]
    ky = constants["ky"]
    omega = constants["omega"]

    ex = exact_tmz_fields(
        grids["ex_x"],
        grids["ex_y"],
        t=0.0,
        epsilon=config.epsilon,
        omega=omega,
        kx=kx,
        ky=ky,
        hz_amplitude=config.hz_amplitude,
        field="Ex",
    )
    ey = exact_tmz_fields(
        grids["ey_x"],
        grids["ey_y"],
        t=0.0,
        epsilon=config.epsilon,
        omega=omega,
        kx=kx,
        ky=ky,
        hz_amplitude=config.hz_amplitude,
        field="Ey",
    )
    # Leapfrog: Hz 位于半步时间层，初始化到 t = -dt/2。
    hz = exact_tmz_fields(
        grids["hz_x"],
        grids["hz_y"],
        t=-0.5 * dt,
        epsilon=config.epsilon,
        omega=omega,
        kx=kx,
        ky=ky,
        hz_amplitude=config.hz_amplitude,
        field="Hz",
    )

    energy_initial = electromagnetic_energy(ex, ey, hz, config.epsilon, config.mu, dx, dy)

    coef_h = dt / config.mu
    coef_e = dt / config.epsilon

    for _ in range(n_steps):
        d_ey_dx = (np.roll(ey, -1, axis=0) - ey) / dx
        d_ex_dy = (np.roll(ex, -1, axis=1) - ex) / dy
        hz = hz - coef_h * (d_ey_dx - d_ex_dy)

        d_hz_dy = (hz - np.roll(hz, 1, axis=1)) / dy
        d_hz_dx = (hz - np.roll(hz, 1, axis=0)) / dx
        ex = ex + coef_e * d_hz_dy
        ey = ey - coef_e * d_hz_dx

    t_e = n_steps * dt
    t_h = t_e - 0.5 * dt

    ex_ref = exact_tmz_fields(
        grids["ex_x"],
        grids["ex_y"],
        t=t_e,
        epsilon=config.epsilon,
        omega=omega,
        kx=kx,
        ky=ky,
        hz_amplitude=config.hz_amplitude,
        field="Ex",
    )
    ey_ref = exact_tmz_fields(
        grids["ey_x"],
        grids["ey_y"],
        t=t_e,
        epsilon=config.epsilon,
        omega=omega,
        kx=kx,
        ky=ky,
        hz_amplitude=config.hz_amplitude,
        field="Ey",
    )
    hz_ref = exact_tmz_fields(
        grids["hz_x"],
        grids["hz_y"],
        t=t_h,
        epsilon=config.epsilon,
        omega=omega,
        kx=kx,
        ky=ky,
        hz_amplitude=config.hz_amplitude,
        field="Hz",
    )

    ex_rel_l2 = compute_l2_relative_error(ex, ex_ref)
    ey_rel_l2 = compute_l2_relative_error(ey, ey_ref)
    hz_rel_l2 = compute_l2_relative_error(hz, hz_ref)

    div_e = divergence_electric(ex, ey, dx, dy)
    max_abs_div_e = float(np.max(np.abs(div_e)))

    energy_final = electromagnetic_energy(ex, ey, hz, config.epsilon, config.mu, dx, dy)
    energy_drift = energy_final - energy_initial

    return {
        "nx": float(config.nx),
        "ny": float(config.ny),
        "dx": dx,
        "dy": dy,
        "n_steps": float(n_steps),
        "dt": dt,
        "courant_2d": courant_2d,
        "c": c,
        "impedance": constants["impedance"],
        "kx": kx,
        "ky": ky,
        "omega": omega,
        "rel_l2_ex": ex_rel_l2,
        "rel_l2_ey": ey_rel_l2,
        "rel_l2_hz": hz_rel_l2,
        "max_abs_div_e": max_abs_div_e,
        "max_abs_div_b": 0.0,
        "energy_initial": energy_initial,
        "energy_final": energy_final,
        "energy_drift": energy_drift,
    }


def main() -> None:
    config = MaxwellConfig()
    result = run_tmx_fdtd(config)

    print("=== Maxwell's Equations | 2D TMz Yee-FDTD ===")
    print(f"grid (nx, ny)         : ({int(result['nx'])}, {int(result['ny'])})")
    print(f"dx, dy                : {result['dx']:.8e}, {result['dy']:.8e}")
    print(f"n_steps, dt           : {int(result['n_steps'])}, {result['dt']:.8e}")
    print(f"courant_2d            : {result['courant_2d']:.8f} (must <= 1.0)")
    print("--- Wave constants ---")
    print(f"c                     : {result['c']:.8e}")
    print(f"impedance             : {result['impedance']:.8e}")
    print(f"kx, ky                : {result['kx']:.8e}, {result['ky']:.8e}")
    print(f"omega                 : {result['omega']:.8e}")
    print("--- Relative L2 errors ---")
    print(f"RelL2(Ex)             : {result['rel_l2_ex']:.8e}")
    print(f"RelL2(Ey)             : {result['rel_l2_ey']:.8e}")
    print(f"RelL2(Hz)             : {result['rel_l2_hz']:.8e}")
    print("--- Gauss law checks ---")
    print(f"max|div(E)|           : {result['max_abs_div_e']:.8e}")
    print(f"max|div(B)|           : {result['max_abs_div_b']:.8e} (TMz with Bx=By=0)")
    print("--- Energy check ---")
    print(f"energy_initial        : {result['energy_initial']:.8e}")
    print(f"energy_final          : {result['energy_final']:.8e}")
    print(f"energy_drift          : {result['energy_drift']:.8e}")


if __name__ == "__main__":
    main()
