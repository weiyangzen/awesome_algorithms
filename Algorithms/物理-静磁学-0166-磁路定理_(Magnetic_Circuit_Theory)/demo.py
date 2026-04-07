"""磁路定理 (Magnetic Circuit Theory) 最小可运行 MVP。

目标：
1) 用磁路定理(霍普金森定律)求解两回路磁路中的分支磁通；
2) 用带噪“测量数据”估计等效磁阻与气隙长度；
3) 给出 NumPy / SciPy / scikit-learn / PyTorch 的可复现对照。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

try:
    import torch
except Exception:  # pragma: no cover - torch 不可用时仍保证脚本可运行
    torch = None


MU0 = 4.0e-7 * np.pi  # 真空磁导率, H/m


@dataclass(frozen=True)
class MagneticCircuitConfig:
    """两回路磁路 + 参数估计配置。"""

    random_seed: int = 20260407
    turns: int = 250
    n_samples: int = 25
    current_min_a: float = 0.2
    current_max_a: float = 2.0
    true_gap_m: float = 1.2e-3
    flux_noise_std_wb: float = 3.0e-7

    # shared limb
    l_shared_m: float = 0.18
    area_shared_m2: float = 2.6e-4
    mur_shared: float = 1200.0

    # branch 1 (contains air gap)
    l_branch1_core_m: float = 0.22
    area_branch1_m2: float = 2.1e-4
    mur_branch1: float = 1200.0
    gap_fringing_factor: float = 1.08

    # branch 2 (no air gap)
    l_branch2_m: float = 0.24
    area_branch2_m2: float = 2.3e-4
    mur_branch2: float = 900.0

    # optimization settings
    gap_lower_bound_m: float = 2.0e-4
    gap_upper_bound_m: float = 3.0e-3
    gap_init_m: float = 9.0e-4
    torch_steps: int = 2000
    torch_lr: float = 0.05


def reluctance(length_m: float, area_m2: float, mu_r: float) -> float:
    """计算磁阻 R = l / (mu0 * mu_r * A), 单位 At/Wb。"""
    if length_m <= 0.0 or area_m2 <= 0.0 or mu_r <= 0.0:
        raise ValueError("length_m, area_m2, mu_r must all be positive")
    return float(length_m / (MU0 * mu_r * area_m2))


def build_reluctances(config: MagneticCircuitConfig, gap_m: float) -> Dict[str, float]:
    """根据几何参数构造共享支路、分支1、分支2磁阻。"""
    if not (config.gap_lower_bound_m / 5.0 <= gap_m <= config.gap_upper_bound_m * 2.0):
        raise ValueError("gap_m is outside a physically reasonable range")

    r_shared = reluctance(config.l_shared_m, config.area_shared_m2, config.mur_shared)
    r_b1_core = reluctance(config.l_branch1_core_m, config.area_branch1_m2, config.mur_branch1)
    effective_gap_area = config.gap_fringing_factor * config.area_branch1_m2
    r_gap = float(gap_m / (MU0 * effective_gap_area))
    r_b1_total = r_b1_core + r_gap
    r_b2 = reluctance(config.l_branch2_m, config.area_branch2_m2, config.mur_branch2)

    return {
        "R_shared": r_shared,
        "R_branch1_core": r_b1_core,
        "R_gap": r_gap,
        "R_branch1_total": r_b1_total,
        "R_branch2": r_b2,
    }


def total_flux_closed_form(ni_at: np.ndarray, r_shared: float, r_b1: float, r_b2: float) -> np.ndarray:
    """闭式计算总磁通 Phi_total。

    对方程组
        [Rsh+R1  Rsh] [phi1] = [NI]
        [Rsh    Rsh+R2] [phi2]   [NI]
    解得
        phi_total = NI * (R1 + R2) / (R1*R2 + Rsh*(R1+R2)).
    """
    det = r_b1 * r_b2 + r_shared * (r_b1 + r_b2)
    return ni_at * (r_b1 + r_b2) / det


def solve_two_loop_flux(
    currents_a: np.ndarray,
    config: MagneticCircuitConfig,
    gap_m: float,
) -> pd.DataFrame:
    """用 2x2 线性方程组求解两回路分支磁通。"""
    currents = np.asarray(currents_a, dtype=float)
    ni = config.turns * currents

    rel = build_reluctances(config, gap_m)
    r_shared = rel["R_shared"]
    r_b1 = rel["R_branch1_total"]
    r_b2 = rel["R_branch2"]

    mat = np.array([[r_shared + r_b1, r_shared], [r_shared, r_shared + r_b2]], dtype=float)
    rhs = np.vstack([ni, ni])
    phi = np.linalg.solve(mat, rhs)
    phi1 = phi[0]
    phi2 = phi[1]
    phi_total = phi1 + phi2

    loop1_drop = r_shared * phi_total + r_b1 * phi1
    loop2_drop = r_shared * phi_total + r_b2 * phi2

    return pd.DataFrame(
        {
            "I_A": currents,
            "NI_At": ni,
            "phi1_Wb": phi1,
            "phi2_Wb": phi2,
            "phi_total_Wb": phi_total,
            "B1_T": phi1 / config.area_branch1_m2,
            "B2_T": phi2 / config.area_branch2_m2,
            "loop1_drop_At": loop1_drop,
            "loop2_drop_At": loop2_drop,
            "loop1_residual_At": ni - loop1_drop,
            "loop2_residual_At": ni - loop2_drop,
        }
    )


def simulate_flux_measurements(config: MagneticCircuitConfig) -> pd.DataFrame:
    """生成带噪总磁通测量数据。"""
    rng = np.random.default_rng(config.random_seed)
    currents = np.linspace(config.current_min_a, config.current_max_a, config.n_samples)

    solved = solve_two_loop_flux(currents, config, config.true_gap_m)
    phi_true = solved["phi_total_Wb"].to_numpy(dtype=float)
    phi_measured = phi_true + rng.normal(0.0, config.flux_noise_std_wb, size=phi_true.size)

    return pd.DataFrame(
        {
            "I_A": currents,
            "NI_At": config.turns * currents,
            "phi_total_true_Wb": phi_true,
            "phi_total_measured_Wb": phi_measured,
            "B2_true_T": solved["B2_T"].to_numpy(dtype=float),
        }
    )


def estimate_equivalent_reluctance(
    ni_at: np.ndarray,
    phi_measured_wb: np.ndarray,
) -> Dict[str, float]:
    """用 sklearn 线性回归拟合 Phi = k * NI，得到 R_eq = 1/k。"""
    model = LinearRegression(fit_intercept=False)
    x = np.asarray(ni_at, dtype=float).reshape(-1, 1)
    y = np.asarray(phi_measured_wb, dtype=float)
    model.fit(x, y)

    slope = float(model.coef_[0])
    if slope <= 0.0:
        raise RuntimeError("estimated slope is non-positive, invalid for reluctance")

    pred = model.predict(x)
    return {
        "slope_phi_per_at": slope,
        "r_eq_at_per_wb": 1.0 / slope,
        "rmse_wb": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def model_total_flux_from_gap(
    currents_a: np.ndarray,
    gap_m: float,
    config: MagneticCircuitConfig,
) -> np.ndarray:
    """给定气隙，返回总磁通模型预测。"""
    currents = np.asarray(currents_a, dtype=float)
    ni = config.turns * currents
    rel = build_reluctances(config, gap_m)
    return total_flux_closed_form(ni, rel["R_shared"], rel["R_branch1_total"], rel["R_branch2"])


def estimate_gap_with_scipy(
    currents_a: np.ndarray,
    phi_measured_wb: np.ndarray,
    config: MagneticCircuitConfig,
) -> Dict[str, float]:
    """用 scipy.least_squares 反演气隙长度。"""

    y = np.asarray(phi_measured_wb, dtype=float)

    def residual(g_vec: np.ndarray) -> np.ndarray:
        g = float(g_vec[0])
        pred = model_total_flux_from_gap(currents_a, g, config)
        # 放大残差量级以改善数值停止条件
        return (pred - y) * 1.0e6

    result = least_squares(
        residual,
        x0=np.array([config.gap_init_m], dtype=float),
        bounds=(
            np.array([config.gap_lower_bound_m], dtype=float),
            np.array([config.gap_upper_bound_m], dtype=float),
        ),
        x_scale=np.array([1.0e-3], dtype=float),
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=10000,
    )

    gap_hat = float(result.x[0])
    pred = model_total_flux_from_gap(currents_a, gap_hat, config)

    return {
        "gap_m": gap_hat,
        "rmse_wb": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "nfev": float(result.nfev),
        "cost": float(result.cost),
    }


def estimate_gap_with_torch(
    currents_a: np.ndarray,
    phi_measured_wb: np.ndarray,
    config: MagneticCircuitConfig,
) -> Optional[Dict[str, float]]:
    """用 PyTorch 梯度下降反演气隙长度。torch 不可用时返回 None。"""
    if torch is None:
        return None

    import torch.nn.functional as f

    torch.manual_seed(config.random_seed)

    i_t = torch.tensor(np.asarray(currents_a, dtype=float), dtype=torch.float64)
    y_t = torch.tensor(np.asarray(phi_measured_wb, dtype=float), dtype=torch.float64)
    ni_t = config.turns * i_t

    rel = build_reluctances(config, config.true_gap_m)
    r_shared = rel["R_shared"]
    r_b1_core = rel["R_branch1_core"]
    r_b2 = rel["R_branch2"]
    c_gap = 1.0 / (MU0 * config.gap_fringing_factor * config.area_branch1_m2)

    raw = torch.tensor(-7.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=config.torch_lr)

    for _ in range(config.torch_steps):
        optimizer.zero_grad()
        gap = f.softplus(raw) + 1.0e-6
        r_b1 = r_b1_core + c_gap * gap
        det = r_b1 * r_b2 + r_shared * (r_b1 + r_b2)
        pred = ni_t * (r_b1 + r_b2) / det
        loss = torch.mean((pred - y_t) ** 2) * 1.0e12
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        gap_hat_t = f.softplus(raw) + 1.0e-6
        gap_hat = float(gap_hat_t.detach().item())
        r_b1 = r_b1_core + c_gap * gap_hat_t
        det = r_b1 * r_b2 + r_shared * (r_b1 + r_b2)
        pred = (ni_t * (r_b1 + r_b2) / det).cpu().numpy()

    y_np = np.asarray(phi_measured_wb, dtype=float)
    return {
        "gap_m": gap_hat,
        "rmse_wb": float(np.sqrt(mean_squared_error(y_np, pred))),
        "r2": float(r2_score(y_np, pred)),
    }


def equivalent_reluctance_from_gap(gap_m: float, config: MagneticCircuitConfig) -> float:
    """由气隙参数计算等效磁阻 R_eq = NI / Phi_total。"""
    rel = build_reluctances(config, gap_m)
    r_shared = rel["R_shared"]
    r_b1 = rel["R_branch1_total"]
    r_b2 = rel["R_branch2"]
    det = r_b1 * r_b2 + r_shared * (r_b1 + r_b2)
    return float(det / (r_b1 + r_b2))


def main() -> None:
    config = MagneticCircuitConfig()

    rel_true = build_reluctances(config, config.true_gap_m)
    measurement = simulate_flux_measurements(config)

    ni = measurement["NI_At"].to_numpy(dtype=float)
    phi_meas = measurement["phi_total_measured_Wb"].to_numpy(dtype=float)
    currents = measurement["I_A"].to_numpy(dtype=float)

    eq_fit = estimate_equivalent_reluctance(ni, phi_meas)
    scipy_gap_fit = estimate_gap_with_scipy(currents, phi_meas, config)
    torch_gap_fit = estimate_gap_with_torch(currents, phi_meas, config)

    true_r_eq = equivalent_reluctance_from_gap(config.true_gap_m, config)
    scipy_r_eq = equivalent_reluctance_from_gap(scipy_gap_fit["gap_m"], config)

    summary_rows = [
        {
            "method": "truth",
            "gap_m": config.true_gap_m,
            "R_eq_At_per_Wb": true_r_eq,
            "gap_abs_err_m": 0.0,
            "gap_rel_err_pct": 0.0,
        },
        {
            "method": "scipy_least_squares",
            "gap_m": scipy_gap_fit["gap_m"],
            "R_eq_At_per_Wb": scipy_r_eq,
            "gap_abs_err_m": abs(scipy_gap_fit["gap_m"] - config.true_gap_m),
            "gap_rel_err_pct": abs(scipy_gap_fit["gap_m"] - config.true_gap_m)
            / config.true_gap_m
            * 100.0,
        },
    ]

    if torch_gap_fit is not None:
        torch_r_eq = equivalent_reluctance_from_gap(torch_gap_fit["gap_m"], config)
        summary_rows.append(
            {
                "method": "torch_adam",
                "gap_m": torch_gap_fit["gap_m"],
                "R_eq_At_per_Wb": torch_r_eq,
                "gap_abs_err_m": abs(torch_gap_fit["gap_m"] - config.true_gap_m),
                "gap_rel_err_pct": abs(torch_gap_fit["gap_m"] - config.true_gap_m)
                / config.true_gap_m
                * 100.0,
            }
        )

    summary = pd.DataFrame(summary_rows)

    solved_preview = solve_two_loop_flux(np.array([0.8, 1.2, 1.6]), config, config.true_gap_m)

    print("=== Magnetic Circuit Theory MVP (Two-Loop Reluctance Network) ===")
    print(
        f"turns={config.turns}, current_range=[{config.current_min_a:.3f}, {config.current_max_a:.3f}] A, "
        f"samples={config.n_samples}"
    )
    print(
        f"true_gap={config.true_gap_m:.6e} m, flux_noise_std={config.flux_noise_std_wb:.2e} Wb, "
        f"mu0={MU0:.6e} H/m"
    )

    print("--- reluctance components at true gap (At/Wb) ---")
    print(
        pd.DataFrame(
            [
                {"component": k, "value_At_per_Wb": v}
                for k, v in rel_true.items()
            ]
        ).to_string(index=False, float_format=lambda v: f"{v:.6e}")
    )

    print("--- two-loop solver preview (noise-free) ---")
    print(solved_preview.to_string(index=False, float_format=lambda v: f"{v:.6e}"))

    print("--- first 5 synthetic measurements ---")
    print(measurement.head(5).to_string(index=False, float_format=lambda v: f"{v:.6e}"))

    print("--- equivalent reluctance from sklearn ---")
    print(
        f"R_eq_fit={eq_fit['r_eq_at_per_wb']:.6e} At/Wb, "
        f"R_eq_true={true_r_eq:.6e} At/Wb, "
        f"R2={eq_fit['r2']:.6f}, RMSE={eq_fit['rmse_wb']:.3e} Wb"
    )

    print("--- gap estimation summary ---")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6e}"))
    print(
        f"scipy fit metrics: R2={scipy_gap_fit['r2']:.6f}, "
        f"RMSE={scipy_gap_fit['rmse_wb']:.3e} Wb, nfev={int(scipy_gap_fit['nfev'])}"
    )
    if torch_gap_fit is not None:
        print(
            f"torch fit metrics: R2={torch_gap_fit['r2']:.6f}, "
            f"RMSE={torch_gap_fit['rmse_wb']:.3e} Wb"
        )

    max_loop_residual = float(
        np.max(
            np.abs(
                solved_preview[["loop1_residual_At", "loop2_residual_At"]].to_numpy(dtype=float)
            )
        )
    )
    gap_rel_err = abs(scipy_gap_fit["gap_m"] - config.true_gap_m) / config.true_gap_m
    eq_rel_err = abs(eq_fit["r_eq_at_per_wb"] - true_r_eq) / true_r_eq

    checks = {
        "ampere_loop_residual < 1e-9 At": max_loop_residual < 1.0e-9,
        "scipy_gap_relative_error < 2%": gap_rel_err < 0.02,
        "equivalent_reluctance_relative_error < 1%": eq_rel_err < 0.01,
        "sklearn_flux_fit_r2 > 0.999": eq_fit["r2"] > 0.999,
    }

    if torch_gap_fit is not None:
        torch_gap_diff = abs(torch_gap_fit["gap_m"] - scipy_gap_fit["gap_m"])
        checks["torch_vs_scipy_gap_diff < 2e-6 m"] = torch_gap_diff < 2.0e-6

    print("--- threshold checks ---")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("Validation: PASS")
    else:
        print("Validation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
