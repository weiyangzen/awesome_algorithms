"""阻抗匹配最小可运行示例：四分之一波长变换器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatchCase:
    """单个阻抗匹配任务配置。"""

    z0_system: float  # 参考系统阻抗(ohm)
    z_load: float  # 负载电阻(ohm)，MVP中假设为纯电阻
    f0_hz: float  # 设计中心频率(Hz)
    phase_velocity: float = 2.0e8  # 传输线相速度(m/s)


def quarter_wave_impedance(z0_system: float, z_load: float) -> float:
    """四分之一波长变换器特性阻抗 Zt = sqrt(Z0 * ZL)。"""
    if z0_system <= 0 or z_load <= 0:
        raise ValueError("z0_system and z_load must be positive for this MVP")
    return float(np.sqrt(z0_system * z_load))


def input_impedance_quarter_wave(
    zt: float,
    z_load: float,
    frequencies_hz: np.ndarray,
    f0_hz: float,
    phase_velocity: float,
) -> np.ndarray:
    """计算频率扫频下四分之一波长线段的输入阻抗。"""
    line_length = phase_velocity / (4.0 * f0_hz)
    beta = 2.0 * np.pi * frequencies_hz / phase_velocity
    tan_term = np.tan(beta * line_length)
    numerator = z_load + 1j * zt * tan_term
    denominator = zt + 1j * z_load * tan_term
    return zt * numerator / denominator


def reflection_coefficient(z_in: np.ndarray | complex, z_ref: float) -> np.ndarray:
    """反射系数 Γ = (Zin - Z0)/(Zin + Z0)。"""
    z_in_array = np.asarray(z_in, dtype=np.complex128)
    return (z_in_array - z_ref) / (z_in_array + z_ref)


def return_loss_db(gamma: np.ndarray) -> np.ndarray:
    """回波损耗(dB) = -20log10(|Γ|)。"""
    mag = np.abs(gamma)
    return -20.0 * np.log10(np.maximum(mag, 1e-15))


def mismatch_loss_db(gamma: np.ndarray) -> np.ndarray:
    """失配损耗(dB) = -10log10(1 - |Γ|^2)。"""
    power_transfer = np.maximum(1.0 - np.abs(gamma) ** 2, 1e-15)
    return -10.0 * np.log10(power_transfer)


def build_report(case: MatchCase) -> pd.DataFrame:
    """构建匹配前后指标对比表。"""
    zt = quarter_wave_impedance(case.z0_system, case.z_load)

    frequencies_hz = np.linspace(0.5 * case.f0_hz, 1.5 * case.f0_hz, 11)
    z_in_matched = input_impedance_quarter_wave(
        zt=zt,
        z_load=case.z_load,
        frequencies_hz=frequencies_hz,
        f0_hz=case.f0_hz,
        phase_velocity=case.phase_velocity,
    )

    gamma_direct = reflection_coefficient(case.z_load + 0j, case.z0_system)
    gamma_before = np.full_like(z_in_matched, gamma_direct, dtype=np.complex128)
    gamma_after = reflection_coefficient(z_in_matched, case.z0_system)

    df = pd.DataFrame(
        {
            "freq_MHz": frequencies_hz / 1e6,
            "|Gamma|_before": np.abs(gamma_before),
            "|Gamma|_after": np.abs(gamma_after),
            "RL_before_dB": return_loss_db(gamma_before),
            "RL_after_dB": return_loss_db(gamma_after),
            "ML_before_dB": mismatch_loss_db(gamma_before),
            "ML_after_dB": mismatch_loss_db(gamma_after),
            "Zin_real_ohm": np.real(z_in_matched),
            "Zin_imag_ohm": np.imag(z_in_matched),
        }
    )
    return df


def main() -> None:
    case = MatchCase(z0_system=50.0, z_load=200.0, f0_hz=1.0e9)
    zt = quarter_wave_impedance(case.z0_system, case.z_load)
    line_length_cm = case.phase_velocity / (4.0 * case.f0_hz) * 100.0

    print("=== Impedance Matching MVP (Quarter-Wave Transformer) ===")
    print(f"System Z0: {case.z0_system:.2f} ohm")
    print(f"Load ZL:   {case.z_load:.2f} ohm")
    print(f"Center f0: {case.f0_hz/1e9:.3f} GHz")
    print(f"Designed Zt: {zt:.2f} ohm")
    print(f"Line length at f0: {line_length_cm:.2f} cm")

    report = build_report(case)

    center_row = report.iloc[(report["freq_MHz"] - case.f0_hz / 1e6).abs().argmin()]
    print("\n--- Sweep Report ---")
    print(report.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))

    print("\n--- Key Point Near f0 ---")
    print(f"freq_MHz: {center_row['freq_MHz']:.2f}")
    print(f"|Gamma| before: {center_row['|Gamma|_before']:.4f}")
    print(f"|Gamma| after:  {center_row['|Gamma|_after']:.6f}")
    print(f"RL before: {center_row['RL_before_dB']:.2f} dB")
    print(f"RL after:  {center_row['RL_after_dB']:.2f} dB")


if __name__ == "__main__":
    main()
