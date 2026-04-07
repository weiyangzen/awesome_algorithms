"""电磁感应定律微分形式（Faraday 微分形式）最小可运行示例。

目标方程：
    curl(E) = - dB/dt

在二维周期网格上构造一组解析场 Ex, Ey, Bz（仅 z 向磁场），
并用中心差分 + 显式时间积分验证：
1) 点态残差 curl(E) + dB/dt 的离散误差
2) 根据 Faraday 更新得到的 Bz 与解析 Bz 的终态误差
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List

import numpy as np


@dataclass(frozen=True)
class FaradayConfig:
    """配置参数。"""

    length_x: float = 1.0
    length_y: float = 1.0
    nx: int = 96
    ny: int = 96
    mode_x: int = 2
    mode_y: int = 3
    amplitude: float = 1.0
    omega: float = 7.0
    t_end: float = 0.8
    cfl_like: float = 0.06


@dataclass(frozen=True)
class WaveNumbers:
    """波数与系数缓存。"""

    kx: float
    ky: float
    coef_b: float


def build_grid(config: FaradayConfig) -> Dict[str, np.ndarray | float]:
    """构建二维周期网格。"""
    dx = config.length_x / config.nx
    dy = config.length_y / config.ny

    x = np.arange(config.nx, dtype=float) * dx
    y = np.arange(config.ny, dtype=float) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")

    return {
        "x": x,
        "y": y,
        "xx": xx,
        "yy": yy,
        "dx": dx,
        "dy": dy,
    }


def derive_wave_numbers(config: FaradayConfig) -> WaveNumbers:
    """计算波数和 Bz 系数。"""
    kx = 2.0 * math.pi * config.mode_x / config.length_x
    ky = 2.0 * math.pi * config.mode_y / config.length_y

    coef_b = -config.amplitude * (kx * kx + ky * ky) / (ky * config.omega)
    return WaveNumbers(kx=kx, ky=ky, coef_b=coef_b)


def electric_field(xx: np.ndarray, yy: np.ndarray, t: float, config: FaradayConfig, wave: WaveNumbers) -> tuple[np.ndarray, np.ndarray]:
    """解析电场 Ex, Ey。"""
    phase_t = math.cos(config.omega * t)

    ex = config.amplitude * np.sin(wave.kx * xx) * np.cos(wave.ky * yy) * phase_t
    ey = -config.amplitude * (wave.kx / wave.ky) * np.cos(wave.kx * xx) * np.sin(wave.ky * yy) * phase_t
    return ex, ey


def magnetic_field_exact(xx: np.ndarray, yy: np.ndarray, t: float, config: FaradayConfig, wave: WaveNumbers) -> np.ndarray:
    """解析磁场 Bz。"""
    return wave.coef_b * np.sin(wave.kx * xx) * np.sin(wave.ky * yy) * math.sin(config.omega * t)


def magnetic_dt_exact(xx: np.ndarray, yy: np.ndarray, t: float, config: FaradayConfig, wave: WaveNumbers) -> np.ndarray:
    """解析时间导数 dBz/dt。"""
    return wave.coef_b * config.omega * np.sin(wave.kx * xx) * np.sin(wave.ky * yy) * math.cos(config.omega * t)


def curl_z_central(ex: np.ndarray, ey: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """中心差分计算 (curl E)_z = dEy/dx - dEx/dy（周期边界）。"""
    d_ey_dx = (np.roll(ey, -1, axis=1) - np.roll(ey, 1, axis=1)) / (2.0 * dx)
    d_ex_dy = (np.roll(ex, -1, axis=0) - np.roll(ex, 1, axis=0)) / (2.0 * dy)
    return d_ey_dx - d_ex_dy


def compute_time_grid(config: FaradayConfig, dx: float, dy: float) -> Dict[str, float | int]:
    """根据 cfl_like 估计时间步并与终止时刻对齐。"""
    dt_guess = config.cfl_like * min(dx, dy)
    n_steps = max(1, math.ceil(config.t_end / dt_guess))
    dt = config.t_end / n_steps
    return {
        "dt": dt,
        "n_steps": n_steps,
    }


def compute_error_metrics(num: np.ndarray, ref: np.ndarray, dx: float, dy: float) -> Dict[str, float]:
    """L2/Linf/相对L2 误差。"""
    err = num - ref
    cell_area = dx * dy

    l2 = float(np.sqrt(np.sum(err * err) * cell_area))
    linf = float(np.max(np.abs(err)))

    ref_l2 = float(np.sqrt(np.sum(ref * ref) * cell_area))
    rel_l2 = l2 / max(ref_l2, 1e-14)

    return {
        "l2": l2,
        "linf": linf,
        "rel_l2": rel_l2,
    }


def simulate_faraday(config: FaradayConfig) -> Dict[str, float | int]:
    """运行 Faraday 微分形式的离散验证。"""
    if config.length_x <= 0.0 or config.length_y <= 0.0:
        raise ValueError("length_x and length_y must be positive")
    if config.nx < 8 or config.ny < 8:
        raise ValueError("nx and ny must be >= 8")
    if config.mode_x < 1 or config.mode_y < 1:
        raise ValueError("mode_x and mode_y must be >= 1")
    if config.omega <= 0.0:
        raise ValueError("omega must be positive")
    if config.t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if config.cfl_like <= 0.0:
        raise ValueError("cfl_like must be positive")

    grid = build_grid(config)
    dx = float(grid["dx"])
    dy = float(grid["dy"])
    xx = grid["xx"]
    yy = grid["yy"]

    wave = derive_wave_numbers(config)

    time_grid = compute_time_grid(config, dx, dy)
    dt = float(time_grid["dt"])
    n_steps = int(time_grid["n_steps"])

    bz = magnetic_field_exact(xx, yy, t=0.0, config=config, wave=wave)

    for step in range(n_steps):
        t_n = step * dt
        ex, ey = electric_field(xx, yy, t=t_n, config=config, wave=wave)
        curl_e_z = curl_z_central(ex, ey, dx, dy)
        bz = bz - dt * curl_e_z

    t_final = n_steps * dt
    bz_ref = magnetic_field_exact(xx, yy, t=t_final, config=config, wave=wave)
    b_metrics = compute_error_metrics(bz, bz_ref, dx, dy)

    # 在一个探测时刻评估方程残差：curl(E) + dB/dt。
    t_probe = 0.37 * t_final
    ex_probe, ey_probe = electric_field(xx, yy, t=t_probe, config=config, wave=wave)
    curl_probe = curl_z_central(ex_probe, ey_probe, dx, dy)
    dbdt_probe = magnetic_dt_exact(xx, yy, t=t_probe, config=config, wave=wave)

    residual = curl_probe + dbdt_probe
    residual_rms = float(np.sqrt(np.mean(residual * residual)))
    residual_linf = float(np.max(np.abs(residual)))

    curl_rms = float(np.sqrt(np.mean(curl_probe * curl_probe)))
    residual_rel_rms = residual_rms / max(curl_rms, 1e-14)

    return {
        "nx": config.nx,
        "ny": config.ny,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "n_steps": n_steps,
        "t_final": t_final,
        "b_l2": b_metrics["l2"],
        "b_linf": b_metrics["linf"],
        "b_rel_l2": b_metrics["rel_l2"],
        "residual_rms": residual_rms,
        "residual_linf": residual_linf,
        "residual_rel_rms": residual_rel_rms,
    }


def run_resolution_study(base_config: FaradayConfig, resolutions: Iterable[int]) -> List[Dict[str, float | int]]:
    """不同网格下的误差对比。"""
    records: List[Dict[str, float | int]] = []
    for n in resolutions:
        cfg = replace(base_config, nx=n, ny=n)
        records.append(simulate_faraday(cfg))
    return records


def main() -> None:
    base_config = FaradayConfig()
    result = simulate_faraday(base_config)

    print("=== Differential Form of Faraday's Law | 2D Periodic MVP ===")
    print(f"grid             : {result['nx']} x {result['ny']}")
    print(f"dx, dy           : {result['dx']:.6e}, {result['dy']:.6e}")
    print(f"dt               : {result['dt']:.6e}")
    print(f"n_steps          : {result['n_steps']}")
    print(f"t_final          : {result['t_final']:.6f}")
    print("--- Bz terminal error (integrate -curl(E) over time) ---")
    print(f"L2(Bz)           : {result['b_l2']:.6e}")
    print(f"Linf(Bz)         : {result['b_linf']:.6e}")
    print(f"RelL2(Bz)        : {result['b_rel_l2']:.6e}")
    print("--- Differential-law residual at probe time ---")
    print(f"RMS(curlE+dB/dt) : {result['residual_rms']:.6e}")
    print(f"Linf residual    : {result['residual_linf']:.6e}")
    print(f"RelRMS residual  : {result['residual_rel_rms']:.6e}")

    print("--- Resolution study (nx=ny) ---")
    study = run_resolution_study(base_config, resolutions=[24, 48, 96])
    print(" n    dt        B_rel_l2      residual_rel_rms")
    for row in study:
        print(
            f"{int(row['nx']):>3d}  {float(row['dt']):.2e}  "
            f"{float(row['b_rel_l2']):.3e}      {float(row['residual_rel_rms']):.3e}"
        )


if __name__ == "__main__":
    main()
