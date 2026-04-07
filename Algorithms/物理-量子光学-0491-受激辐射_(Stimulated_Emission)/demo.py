"""受激辐射最小可运行验证脚本。

该 MVP 聚焦两件可计算事实：
1) 受激/自发速率比随光子通量的交叉点；
2) 反转粒子数穿越阈值时，介质从净吸收转为净放大。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class StimulatedEmissionConfig:
    # 物理参数（采用教学用量级，强调机制而非特定材料标定）
    sigma: float = 3.0e-20  # m^2, 受激辐射截面
    tau_sp: float = 10.0e-9  # s, 自发辐射寿命
    alpha: float = 25.0  # 1/m, 有效损耗系数

    # 传播与扫描参数
    medium_length: float = 0.03  # m
    i_in: float = 1.0e-3  # 输入光强（归一化单位）
    delta_n_min: float = 0.0  # m^-3
    delta_n_max: float = 1.8e21  # m^-3
    gain_points: int = 31

    # 速率比扫描
    n2_ref: float = 8.0e20  # m^-3
    rate_points: int = 25


def spontaneous_rate(n2: float, tau_sp: float) -> float:
    """自发辐射速率（每单位体积）。"""
    return n2 / tau_sp


def stimulated_rate(n2: float, photon_flux: float, sigma: float) -> float:
    """受激辐射速率（每单位体积）。"""
    return sigma * photon_flux * n2


def photon_flux_crossover(sigma: float, tau_sp: float) -> float:
    """R_stim = R_sp 时的通量阈值 Phi*。"""
    return 1.0 / (sigma * tau_sp)


def net_gain(delta_n: float, sigma: float, alpha: float) -> float:
    """净增益系数 g_net = sigma * DeltaN - alpha。"""
    return sigma * delta_n - alpha


def propagate_intensity_numeric(i0: float, g_net: float, length: float) -> float:
    """数值积分 dI/dz = g_net * I，返回 I(L)。"""

    def rhs(_z: float, y: np.ndarray) -> np.ndarray:
        return np.array([g_net * y[0]], dtype=float)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, length),
        y0=np.array([i0], dtype=float),
        t_eval=np.array([length], dtype=float),
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return float(sol.y[0, -1])


def propagate_intensity_closed_form(i0: float, g_net: float, length: float) -> float:
    """闭式解 I(L) = I0 * exp(g_net * L)。"""
    return float(i0 * np.exp(g_net * length))


def build_gain_table(cfg: StimulatedEmissionConfig) -> pd.DataFrame:
    delta_n_grid = np.linspace(cfg.delta_n_min, cfg.delta_n_max, cfg.gain_points)

    rows: list[dict[str, float]] = []
    for delta_n in delta_n_grid:
        g = net_gain(delta_n, cfg.sigma, cfg.alpha)
        i_num = propagate_intensity_numeric(cfg.i_in, g, cfg.medium_length)
        i_exact = propagate_intensity_closed_form(cfg.i_in, g, cfg.medium_length)
        rows.append(
            {
                "delta_n": float(delta_n),
                "g_net": float(g),
                "i_out_numeric": i_num,
                "i_out_exact": i_exact,
                "gain_ratio": i_num / cfg.i_in,
                "ode_vs_exact_relerr": abs(i_num - i_exact) / max(i_exact, 1e-30),
            }
        )

    return pd.DataFrame(rows)


def build_rate_ratio_table(cfg: StimulatedEmissionConfig) -> pd.DataFrame:
    phi_star = photon_flux_crossover(cfg.sigma, cfg.tau_sp)
    phi_grid = np.logspace(np.log10(phi_star) - 2.0, np.log10(phi_star) + 2.0, cfg.rate_points)

    rows: list[dict[str, float]] = []
    for phi in phi_grid:
        r_sp = spontaneous_rate(cfg.n2_ref, cfg.tau_sp)
        r_st = stimulated_rate(cfg.n2_ref, phi, cfg.sigma)
        rows.append(
            {
                "photon_flux": float(phi),
                "r_sp": float(r_sp),
                "r_st": float(r_st),
                "stim_over_sp": float(r_st / r_sp),
            }
        )

    return pd.DataFrame(rows)


def linear_crossing(x: np.ndarray, y: np.ndarray, y_target: float) -> float:
    """在线性插值下估计 y 穿越 y_target 的 x。"""
    shifted = y - y_target
    crossing_idx = np.where((shifted[:-1] <= 0.0) & (shifted[1:] >= 0.0))[0]
    if len(crossing_idx) == 0:
        return float("nan")

    i = int(crossing_idx[0])
    x0, x1 = x[i], x[i + 1]
    y0, y1 = shifted[i], shifted[i + 1]
    if abs(y1 - y0) < 1e-18:
        return float(x0)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def nearest_row(df: pd.DataFrame, col: str, value: float) -> pd.Series:
    idx = (df[col] - value).abs().idxmin()
    return df.loc[idx]


def make_validation_report(
    cfg: StimulatedEmissionConfig,
    gain_df: pd.DataFrame,
    rate_df: pd.DataFrame,
) -> dict[str, float | bool]:
    delta_n_theory = cfg.alpha / cfg.sigma
    delta_n_est = linear_crossing(
        gain_df["delta_n"].to_numpy(),
        gain_df["gain_ratio"].to_numpy(),
        y_target=1.0,
    )
    threshold_rel_err = abs(delta_n_est - delta_n_theory) / delta_n_theory

    below = nearest_row(gain_df, "delta_n", 0.75 * delta_n_theory)
    above = nearest_row(gain_df, "delta_n", 1.25 * delta_n_theory)

    ln_gain = np.log(gain_df["gain_ratio"].to_numpy())
    slope_est, intercept_est = np.polyfit(gain_df["delta_n"].to_numpy(), ln_gain, deg=1)
    slope_theory = cfg.sigma * cfg.medium_length
    slope_rel_err = abs(slope_est - slope_theory) / slope_theory

    phi_theory = photon_flux_crossover(cfg.sigma, cfg.tau_sp)
    phi_est = linear_crossing(
        rate_df["photon_flux"].to_numpy(),
        rate_df["stim_over_sp"].to_numpy(),
        y_target=1.0,
    )
    phi_rel_err = abs(phi_est - phi_theory) / phi_theory

    pass_threshold = threshold_rel_err < 0.03
    pass_gain_side = bool((below["gain_ratio"] < 1.0) and (above["gain_ratio"] > 1.0))
    pass_slope = slope_rel_err < 0.02
    pass_phi = phi_rel_err < 0.03

    return {
        "delta_n_theory": float(delta_n_theory),
        "delta_n_est": float(delta_n_est),
        "threshold_rel_err": float(threshold_rel_err),
        "below_gain_ratio": float(below["gain_ratio"]),
        "above_gain_ratio": float(above["gain_ratio"]),
        "slope_theory": float(slope_theory),
        "slope_est": float(slope_est),
        "intercept_est": float(intercept_est),
        "slope_rel_err": float(slope_rel_err),
        "phi_theory": float(phi_theory),
        "phi_est": float(phi_est),
        "phi_rel_err": float(phi_rel_err),
        "max_ode_exact_relerr": float(gain_df["ode_vs_exact_relerr"].max()),
        "pass_threshold": pass_threshold,
        "pass_gain_side": pass_gain_side,
        "pass_slope": pass_slope,
        "pass_phi": pass_phi,
        "pass_all": bool(pass_threshold and pass_gain_side and pass_slope and pass_phi),
    }


def main() -> None:
    cfg = StimulatedEmissionConfig()

    gain_df = build_gain_table(cfg)
    rate_df = build_rate_ratio_table(cfg)
    report = make_validation_report(cfg, gain_df, rate_df)

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)

    print("=== Stimulated Emission MVP ===")
    print(f"sigma={cfg.sigma:.3e} m^2, tau_sp={cfg.tau_sp:.3e} s, alpha={cfg.alpha:.3e} 1/m, L={cfg.medium_length:.3e} m")

    print("\n[1] 受激/自发速率交叉验证")
    print(f"theory Phi* = {report['phi_theory']:.6e}")
    print(f"estimated Phi* = {report['phi_est']:.6e}")
    print(f"relative error = {report['phi_rel_err']:.3e}")

    print("\n[2] 净增益阈值验证 (gain_ratio = I_out / I_in)")
    print(f"theory DeltaN_th = {report['delta_n_theory']:.6e}")
    print(f"estimated DeltaN_th = {report['delta_n_est']:.6e}")
    print(f"relative error = {report['threshold_rel_err']:.3e}")
    print(f"below-th gain ratio = {report['below_gain_ratio']:.6f}")
    print(f"above-th gain ratio = {report['above_gain_ratio']:.6f}")

    print("\n[3] ln(gain_ratio)-DeltaN 线性检验")
    print(f"theory slope = {report['slope_theory']:.6e}")
    print(f"estimated slope = {report['slope_est']:.6e}")
    print(f"slope relative error = {report['slope_rel_err']:.3e}")
    print(f"max ODE-vs-exact relerr = {report['max_ode_exact_relerr']:.3e}")

    print("\n[4] Sample rows: rate table (head 5)")
    print(rate_df.head(5).to_string(index=False))

    print("\n[5] Sample rows: gain table (around threshold)")
    th = report["delta_n_theory"]
    around = gain_df[(gain_df["delta_n"] > 0.8 * th) & (gain_df["delta_n"] < 1.2 * th)]
    print(around.to_string(index=False))

    print(f"\nValidation: {'PASS' if report['pass_all'] else 'FAIL'}")


if __name__ == "__main__":
    main()
