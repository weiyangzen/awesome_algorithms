"""电磁波动方程（1D Yee-FDTD）最小可运行示例。

模型：无源、均匀介质中的 1D Maxwell 方程
    dE/dt = -(1/epsilon) dH/dx
    dH/dt = -(1/mu) dE/dx

在周期边界条件下，使用 Yee 交错网格进行显式时间推进，
并与解析平面波解做误差对比。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class FDTDConfig:
    """参数配置。"""

    epsilon: float = 1.0
    mu: float = 1.0
    length: float = 1.0
    nx: int = 400
    mode: int = 3
    amplitude: float = 1.0
    t_end: float = 0.5
    courant_target: float = 0.95


def derive_wave_constants(epsilon: float, mu: float, length: float, mode: int) -> Dict[str, float]:
    """计算波速、阻抗与色散参数。"""
    c = 1.0 / math.sqrt(mu * epsilon)
    impedance = math.sqrt(mu / epsilon)
    k = 2.0 * math.pi * mode / length
    omega = c * k
    return {
        "c": c,
        "impedance": impedance,
        "k": k,
        "omega": omega,
    }


def exact_plane_wave(
    x: np.ndarray,
    t: float,
    amplitude: float,
    k: float,
    omega: float,
    impedance: float,
    field: str,
) -> np.ndarray:
    """解析行波解。"""
    phase = k * x - omega * t
    if field == "E":
        return amplitude * np.sin(phase)
    if field == "H":
        return (amplitude / impedance) * np.sin(phase)
    raise ValueError("field must be 'E' or 'H'")


def discrete_energy(e_field: np.ndarray, h_field: np.ndarray, epsilon: float, mu: float, dx: float) -> float:
    """离散电磁能量。"""
    return 0.5 * dx * (epsilon * float(np.sum(e_field * e_field)) + mu * float(np.sum(h_field * h_field)))


def compute_errors(num: np.ndarray, ref: np.ndarray, dx: float) -> Dict[str, float]:
    """计算 L1/L2/Linf 与相对 L2 误差。"""
    err = num - ref
    l1 = float(np.sum(np.abs(err)) * dx)
    l2 = float(np.sqrt(np.sum(err * err) * dx))
    linf = float(np.max(np.abs(err)))

    ref_norm = float(np.sqrt(np.sum(ref * ref) * dx))
    rel_l2 = l2 / max(ref_norm, 1e-14)
    return {
        "l1": l1,
        "l2": l2,
        "linf": linf,
        "rel_l2": rel_l2,
    }


def run_fdtd_1d_periodic(config: FDTDConfig) -> Dict[str, object]:
    """执行 1D 周期边界 Yee-FDTD。"""
    if config.epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if config.mu <= 0.0:
        raise ValueError("mu must be positive")
    if config.length <= 0.0:
        raise ValueError("length must be positive")
    if config.nx < 8:
        raise ValueError("nx must be >= 8")
    if config.mode < 1:
        raise ValueError("mode must be >= 1")
    if config.t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if config.courant_target <= 0.0:
        raise ValueError("courant_target must be positive")

    constants = derive_wave_constants(
        epsilon=config.epsilon,
        mu=config.mu,
        length=config.length,
        mode=config.mode,
    )

    c = constants["c"]
    impedance = constants["impedance"]
    k = constants["k"]
    omega = constants["omega"]

    dx = config.length / config.nx
    x_e = np.arange(config.nx) * dx
    x_h = (np.arange(config.nx) + 0.5) * dx

    dt_guess = config.courant_target * dx / c
    n_steps = max(1, math.ceil(config.t_end / dt_guess))
    dt = config.t_end / n_steps
    courant_actual = c * dt / dx

    if courant_actual > 1.0 + 1e-12:
        raise ValueError(
            f"unstable setup: Courant={courant_actual:.6f} > 1.0; "
            "decrease courant_target or increase nx"
        )

    e_field = exact_plane_wave(
        x=x_e,
        t=0.0,
        amplitude=config.amplitude,
        k=k,
        omega=omega,
        impedance=impedance,
        field="E",
    )
    # Leapfrog: H 存在半步时间层，因此初始化到 t = -dt/2。
    h_field = exact_plane_wave(
        x=x_h,
        t=-0.5 * dt,
        amplitude=config.amplitude,
        k=k,
        omega=omega,
        impedance=impedance,
        field="H",
    )

    energy_initial = discrete_energy(e_field, h_field, config.epsilon, config.mu, dx)

    coef_h = dt / (config.mu * dx)
    coef_e = dt / (config.epsilon * dx)

    for _ in range(n_steps):
        h_field = h_field - coef_h * (np.roll(e_field, -1) - e_field)
        e_field = e_field - coef_e * (h_field - np.roll(h_field, 1))

    t_e = n_steps * dt
    t_h = t_e - 0.5 * dt

    e_ref = exact_plane_wave(
        x=x_e,
        t=t_e,
        amplitude=config.amplitude,
        k=k,
        omega=omega,
        impedance=impedance,
        field="E",
    )
    h_ref = exact_plane_wave(
        x=x_h,
        t=t_h,
        amplitude=config.amplitude,
        k=k,
        omega=omega,
        impedance=impedance,
        field="H",
    )

    errors_e = compute_errors(e_field, e_ref, dx)
    errors_h = compute_errors(h_field, h_ref, dx)

    energy_final = discrete_energy(e_field, h_field, config.epsilon, config.mu, dx)

    return {
        "constants": constants,
        "dx": dx,
        "dt": dt,
        "n_steps": n_steps,
        "courant_actual": courant_actual,
        "E": e_field,
        "H": h_field,
        "E_ref": e_ref,
        "H_ref": h_ref,
        "errors_E": errors_e,
        "errors_H": errors_h,
        "energy_initial": energy_initial,
        "energy_final": energy_final,
        "energy_drift": energy_final - energy_initial,
    }


def main() -> None:
    config = FDTDConfig()
    result = run_fdtd_1d_periodic(config)

    constants = result["constants"]
    errors_e = result["errors_E"]
    errors_h = result["errors_H"]

    print("=== Electromagnetic Wave Equation | 1D Yee-FDTD ===")
    print(f"epsilon           : {config.epsilon}")
    print(f"mu                : {config.mu}")
    print(f"length            : {config.length}")
    print(f"nx                : {config.nx}")
    print(f"mode              : {config.mode}")
    print(f"t_end             : {config.t_end}")
    print(f"n_steps           : {result['n_steps']}")
    print(f"dx                : {result['dx']:.8e}")
    print(f"dt                : {result['dt']:.8e}")
    print(f"courant_actual    : {result['courant_actual']:.8f} (must <= 1.0)")
    print("--- Wave constants ---")
    print(f"c                 : {constants['c']:.8e}")
    print(f"impedance(Z)      : {constants['impedance']:.8e}")
    print(f"k                 : {constants['k']:.8e}")
    print(f"omega             : {constants['omega']:.8e}")
    print("--- Errors (E field) ---")
    print(f"L1(E)             : {errors_e['l1']:.8e}")
    print(f"L2(E)             : {errors_e['l2']:.8e}")
    print(f"Linf(E)           : {errors_e['linf']:.8e}")
    print(f"RelL2(E)          : {errors_e['rel_l2']:.8e}")
    print("--- Errors (H field) ---")
    print(f"L1(H)             : {errors_h['l1']:.8e}")
    print(f"L2(H)             : {errors_h['l2']:.8e}")
    print(f"Linf(H)           : {errors_h['linf']:.8e}")
    print(f"RelL2(H)          : {errors_h['rel_l2']:.8e}")
    print("--- Energy check ---")
    print(f"energy_initial    : {result['energy_initial']:.8e}")
    print(f"energy_final      : {result['energy_final']:.8e}")
    print(f"energy_drift      : {result['energy_drift']:.8e}")


if __name__ == "__main__":
    main()
