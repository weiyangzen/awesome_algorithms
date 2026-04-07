"""位移电流（Displacement Current）最小可运行示例。

场景：真空平行板电容器串联电阻，施加阶跃电压。
我们用数值方式计算电通量变化率，从而得到位移电流，验证：
1) I_d = epsilon0 * d(Phi_E)/dt 与导线电流 I_c 一致；
2) 在 r=a 的安培回路上，基于 I_c 与 I_d 得到的磁场一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy import constants as const


@dataclass(frozen=True)
class DisplacementConfig:
    """模型配置。"""

    v_supply: float = 12.0
    resistance: float = 50.0
    plate_radius: float = 0.05
    plate_gap: float = 1.0e-3
    t_end_factor: float = 6.0
    n_time: int = 4000


def validate_config(config: DisplacementConfig) -> None:
    """参数合法性检查。"""
    if config.v_supply <= 0.0:
        raise ValueError("v_supply must be positive")
    if config.resistance <= 0.0:
        raise ValueError("resistance must be positive")
    if config.plate_radius <= 0.0:
        raise ValueError("plate_radius must be positive")
    if config.plate_gap <= 0.0:
        raise ValueError("plate_gap must be positive")
    if config.t_end_factor <= 0.0:
        raise ValueError("t_end_factor must be positive")
    if config.n_time < 16:
        raise ValueError("n_time must be >= 16")


def compute_geometry(config: DisplacementConfig) -> Dict[str, float]:
    """由几何参数计算面积、电容、时间常数。"""
    area = math.pi * config.plate_radius * config.plate_radius
    capacitance = const.epsilon_0 * area / config.plate_gap
    tau = config.resistance * capacitance

    return {
        "area": area,
        "capacitance": capacitance,
        "tau": tau,
    }


def simulate_rc_step(config: DisplacementConfig) -> Dict[str, np.ndarray | float]:
    """模拟 RC 阶跃充电并给出导线电流与电通量。"""
    geom = compute_geometry(config)
    tau = float(geom["tau"])
    t_end = config.t_end_factor * tau

    t = np.linspace(0.0, t_end, config.n_time)
    exp_term = np.exp(-t / tau)

    vc = config.v_supply * (1.0 - exp_term)
    ic = (config.v_supply / config.resistance) * exp_term

    e_field = vc / config.plate_gap
    flux_e = e_field * float(geom["area"])

    return {
        "t": t,
        "vc": vc,
        "ic": ic,
        "e_field": e_field,
        "flux_e": flux_e,
        "area": float(geom["area"]),
        "capacitance": float(geom["capacitance"]),
        "tau": tau,
        "t_end": t_end,
    }


def displacement_current_from_flux(t: np.ndarray, flux_e: np.ndarray) -> np.ndarray:
    """数值计算 I_d = epsilon0 * dPhi_E/dt。"""
    dflux_dt = np.gradient(flux_e, t, edge_order=2)
    return const.epsilon_0 * dflux_dt


def compute_error_metrics(reference: np.ndarray, estimate: np.ndarray, trim: int = 4) -> Dict[str, float]:
    """计算 L2/Linf/相对 L2 误差。"""
    if reference.size <= 2 * trim:
        trim = 0

    ref = reference[trim : reference.size - trim] if trim > 0 else reference
    est = estimate[trim : estimate.size - trim] if trim > 0 else estimate

    diff = est - ref
    l2 = float(np.sqrt(np.mean(diff * diff)))
    linf = float(np.max(np.abs(diff)))

    ref_l2 = float(np.sqrt(np.mean(ref * ref)))
    rel_l2 = l2 / max(ref_l2, 1e-14)

    return {
        "l2": l2,
        "linf": linf,
        "rel_l2": rel_l2,
    }


def ampere_maxwell_fields(i_conduction: np.ndarray, i_displacement: np.ndarray, loop_radius: float) -> Dict[str, np.ndarray]:
    """在 r=loop_radius 回路上比较磁场。"""
    factor = const.mu_0 / (2.0 * math.pi * loop_radius)

    b_from_conduction = factor * i_conduction
    b_from_displacement = factor * i_displacement

    return {
        "b_conduction": b_from_conduction,
        "b_displacement": b_from_displacement,
    }


def build_snapshot_table(
    t: np.ndarray,
    vc: np.ndarray,
    ic: np.ndarray,
    id_current: np.ndarray,
    b_conduction: np.ndarray,
    b_displacement: np.ndarray,
    rows: int = 8,
) -> pd.DataFrame:
    """构造输出快照表。"""
    idx = np.linspace(0, t.size - 1, rows, dtype=int)

    df = pd.DataFrame(
        {
            "t_s": t[idx],
            "Vc_V": vc[idx],
            "Ic_A": ic[idx],
            "Id_A": id_current[idx],
            "B_from_Ic_T": b_conduction[idx],
            "B_from_Id_T": b_displacement[idx],
        }
    )

    return df


def main() -> None:
    config = DisplacementConfig()
    validate_config(config)

    sim = simulate_rc_step(config)

    t = sim["t"]
    vc = sim["vc"]
    ic = sim["ic"]
    flux_e = sim["flux_e"]

    id_current = displacement_current_from_flux(t=t, flux_e=flux_e)
    current_metrics = compute_error_metrics(reference=ic, estimate=id_current, trim=4)

    loop_radius = config.plate_radius
    b_fields = ampere_maxwell_fields(
        i_conduction=ic,
        i_displacement=id_current,
        loop_radius=loop_radius,
    )
    b_metrics = compute_error_metrics(
        reference=b_fields["b_conduction"],
        estimate=b_fields["b_displacement"],
        trim=4,
    )

    df = build_snapshot_table(
        t=t,
        vc=vc,
        ic=ic,
        id_current=id_current,
        b_conduction=b_fields["b_conduction"],
        b_displacement=b_fields["b_displacement"],
        rows=8,
    )

    print("=== Displacement Current MVP | RC Charging Capacitor ===")
    print(f"epsilon0 [F/m]                 : {const.epsilon_0:.9e}")
    print(f"mu0 [H/m]                      : {const.mu_0:.9e}")
    print(f"plate radius a [m]             : {config.plate_radius:.6e}")
    print(f"plate gap d [m]                : {config.plate_gap:.6e}")
    print(f"area A [m^2]                   : {float(sim['area']):.6e}")
    print(f"capacitance C [F]              : {float(sim['capacitance']):.6e}")
    print(f"time constant tau=RC [s]       : {float(sim['tau']):.6e}")
    print(f"time grid points               : {config.n_time}")
    print(f"t_end [s]                      : {float(sim['t_end']):.6e}")

    print("--- Current consistency: Id vs Ic ---")
    print(f"L2 error [A]                   : {current_metrics['l2']:.6e}")
    print(f"Linf error [A]                 : {current_metrics['linf']:.6e}")
    print(f"Relative L2 error              : {current_metrics['rel_l2']:.6e}")

    print("--- Ampere-Maxwell consistency at r=a ---")
    print(f"L2 error in B [T]              : {b_metrics['l2']:.6e}")
    print(f"Linf error in B [T]            : {b_metrics['linf']:.6e}")
    print(f"Relative L2 error in B         : {b_metrics['rel_l2']:.6e}")

    print("--- Time snapshots ---")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.6e}"))


if __name__ == "__main__":
    main()
