"""磁化率 (Magnetic Susceptibility) 最小可运行 MVP。

目标：在线性磁化区间中，根据测得的磁场强度 H 与磁化强度 M，
用回归估计材料磁化率 chi（χ）。

模型：
    M = χ H + M0
其中 M0 代表小的偏置项（仪器零点/背景项）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

try:
    import torch
except Exception:  # pragma: no cover - torch 不可用时仍保证 MVP 可运行
    torch = None


MU0 = 4.0e-7 * np.pi  # 真空磁导率 (H/m)


@dataclass(frozen=True)
class SusceptibilityConfig:
    """MVP 参数配置。"""

    random_seed: int = 7
    n_samples: int = 80
    h_min: float = -2.0e5  # A/m
    h_max: float = 2.0e5  # A/m
    true_chi: float = 3.2e-3
    true_m0: float = 12.0  # A/m
    noise_std: float = 8.0  # A/m
    torch_steps: int = 1200
    torch_lr: float = 0.08


def generate_synthetic_dataset(config: SusceptibilityConfig) -> pd.DataFrame:
    """生成线性磁化区间下的合成测量数据。"""
    rng = np.random.default_rng(config.random_seed)
    h = np.linspace(config.h_min, config.h_max, config.n_samples)
    m_clean = config.true_chi * h + config.true_m0
    m_measured = m_clean + rng.normal(0.0, config.noise_std, size=config.n_samples)
    b_measured = MU0 * (h + m_measured)
    return pd.DataFrame(
        {
            "H_A_per_m": h,
            "M_A_per_m": m_measured,
            "B_T": b_measured,
        }
    )


def estimate_with_numpy(h: np.ndarray, m: np.ndarray) -> Dict[str, float]:
    """使用 numpy 最小二乘估计 χ 与 M0。"""
    design = np.column_stack([h, np.ones_like(h)])
    chi, m0 = np.linalg.lstsq(design, m, rcond=None)[0]
    pred = chi * h + m0
    return {
        "chi": float(chi),
        "m0": float(m0),
        "rmse": float(np.sqrt(mean_squared_error(m, pred))),
        "r2": float(r2_score(m, pred)),
    }


def estimate_with_scipy(h: np.ndarray, m: np.ndarray) -> Dict[str, float]:
    """使用 scipy.stats.linregress 估计 χ 与 M0。"""
    fit = stats.linregress(h, m)
    pred = fit.slope * h + fit.intercept
    return {
        "chi": float(fit.slope),
        "m0": float(fit.intercept),
        "rmse": float(np.sqrt(mean_squared_error(m, pred))),
        "r2": float(r2_score(m, pred)),
    }


def estimate_with_sklearn(h: np.ndarray, m: np.ndarray) -> Dict[str, float]:
    """使用 scikit-learn 线性回归估计 χ 与 M0。"""
    model = LinearRegression(fit_intercept=True)
    model.fit(h.reshape(-1, 1), m)
    pred = model.predict(h.reshape(-1, 1))
    return {
        "chi": float(model.coef_[0]),
        "m0": float(model.intercept_),
        "rmse": float(np.sqrt(mean_squared_error(m, pred))),
        "r2": float(r2_score(m, pred)),
    }


def estimate_with_torch(
    h: np.ndarray,
    m: np.ndarray,
    config: SusceptibilityConfig,
) -> Optional[Dict[str, float]]:
    """使用 PyTorch 梯度下降估计 χ 与 M0。torch 不可用时返回 None。"""
    if torch is None:
        return None

    torch.manual_seed(config.random_seed)
    x = torch.tensor(h, dtype=torch.float64).view(-1, 1)
    y = torch.tensor(m, dtype=torch.float64).view(-1, 1)

    layer = torch.nn.Linear(1, 1, bias=True, dtype=torch.float64)
    optimizer = torch.optim.Adam(layer.parameters(), lr=config.torch_lr)

    for _ in range(config.torch_steps):
        optimizer.zero_grad()
        pred = layer(x)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = layer(x).cpu().numpy().reshape(-1)
        chi = float(layer.weight.item())
        m0 = float(layer.bias.item())

    return {
        "chi": chi,
        "m0": m0,
        "rmse": float(np.sqrt(mean_squared_error(m, pred))),
        "r2": float(r2_score(m, pred)),
    }


def classify_material(chi: float) -> str:
    """按 χ 的符号与量级做简化分类。"""
    if chi > 1.0e-2:
        return "ferromagnetic-like (large effective χ; linear range may be narrow)"
    if chi > 0.0:
        return "paramagnetic"
    if chi < 0.0:
        return "diamagnetic"
    return "approximately non-magnetic"


def format_summary_table(estimates: Dict[str, Dict[str, float]], true_chi: float) -> pd.DataFrame:
    """整理估计结果表。"""
    rows = []
    for method, metrics in estimates.items():
        chi = metrics["chi"]
        rel_pct = abs(chi - true_chi) / max(abs(true_chi), 1e-15) * 100.0
        rows.append(
            {
                "method": method,
                "chi_est": chi,
                "m0_est(A/m)": metrics["m0"],
                "abs_err_chi": abs(chi - true_chi),
                "rel_err_chi(%)": rel_pct,
                "rmse(A/m)": metrics["rmse"],
                "r2": metrics["r2"],
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values(by="abs_err_chi", ascending=True).reset_index(drop=True)


def main() -> None:
    config = SusceptibilityConfig()
    data = generate_synthetic_dataset(config)

    h = data["H_A_per_m"].to_numpy(dtype=float)
    m = data["M_A_per_m"].to_numpy(dtype=float)
    b = data["B_T"].to_numpy(dtype=float)

    estimates: Dict[str, Dict[str, float]] = {
        "numpy_lstsq": estimate_with_numpy(h, m),
        "scipy_linregress": estimate_with_scipy(h, m),
        "sklearn_linear_regression": estimate_with_sklearn(h, m),
    }

    torch_est = estimate_with_torch(h, m, config)
    if torch_est is not None:
        estimates["torch_adam_linear"] = torch_est

    summary = format_summary_table(estimates, true_chi=config.true_chi)
    best_chi = float(summary.iloc[0]["chi_est"])

    mask = np.abs(h) > 1.0
    chi_from_b = float(np.mean(b[mask] / (MU0 * h[mask]) - 1.0))

    print("=== Magnetic Susceptibility MVP (Linear Magnetization Regime) ===")
    print(f"samples: {config.n_samples}, H range: [{config.h_min:.1f}, {config.h_max:.1f}] A/m")
    print(
        f"ground truth (synthetic): chi={config.true_chi:.6e}, "
        f"M0={config.true_m0:.6f} A/m, noise_std={config.noise_std:.3f} A/m"
    )
    print("--- first 5 synthetic measurements ---")
    print(data.head(5).to_string(index=False, float_format=lambda v: f"{v:.6e}"))
    print("--- susceptibility estimation summary ---")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6e}"))
    print("--- derived physical quantities ---")
    print(f"best chi estimate           : {best_chi:.6e}")
    print(f"material class (simplified) : {classify_material(best_chi)}")
    print(f"relative permeability mu_r  : {1.0 + best_chi:.6e}")
    print(f"chi from B-H direct formula : {chi_from_b:.6e} (uses mean(B/(mu0*H)-1))")


if __name__ == "__main__":
    main()
