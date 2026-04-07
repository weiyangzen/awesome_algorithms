"""倾向得分（Propensity Score）最小可运行示例。

本脚本演示：
1) 在观测数据中用逻辑回归估计倾向得分 e(x)=P(T=1|X)；
2) 使用逆概率加权（IPTW）估计平均处理效应 ATE；
3) 对比 naive 均值差与 IPTW 估计，并查看协变量平衡性改善。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_observational_data(n: int = 6000, seed: int = 7) -> tuple[pd.DataFrame, float]:
    """构造带混杂的观测数据，返回数据表与真值 ATE。"""
    rng = np.random.default_rng(seed)

    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)

    # 治疗分配机制依赖协变量，制造混杂。
    logit_p = -0.2 + 1.0 * x1 - 0.9 * x2 + 0.5 * x1 * x2
    true_ps = sigmoid(logit_p)
    t = rng.binomial(1, true_ps, size=n)

    # 常数处理效应，便于评估估计偏差。
    true_ate = 2.0
    y0 = 1.5 + 1.3 * x1 - 1.1 * x2 + 0.7 * x1 * x2 + rng.normal(0.0, 1.0, size=n)
    y = y0 + true_ate * t

    df = pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y, "true_ps": true_ps})
    return df, true_ate


def fit_propensity_score(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """用逻辑回归拟合倾向得分。"""
    model = LogisticRegression(solver="lbfgs", max_iter=2000)
    model.fit(x, t)
    ps_hat = model.predict_proba(x)[:, 1]
    return ps_hat


def estimate_ate_iptw(y: np.ndarray, t: np.ndarray, ps: np.ndarray, eps: float = 1e-3) -> float:
    """使用归一化 IPTW 估计 ATE。"""
    ps = np.clip(ps, eps, 1.0 - eps)
    w_treated = t / ps
    w_control = (1 - t) / (1 - ps)
    mu1_hat = np.sum(w_treated * y) / np.sum(w_treated)
    mu0_hat = np.sum(w_control * y) / np.sum(w_control)
    return float(mu1_hat - mu0_hat)


def standardized_mean_difference(x: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> float:
    """计算处理组和对照组的标准化均值差（SMD）。"""
    treated = t == 1
    control = ~treated

    if w is None:
        w = np.ones_like(x, dtype=float)

    wt = w[treated]
    wc = w[control]
    xt = x[treated]
    xc = x[control]

    mt = np.average(xt, weights=wt)
    mc = np.average(xc, weights=wc)
    vt = np.average((xt - mt) ** 2, weights=wt)
    vc = np.average((xc - mc) ** 2, weights=wc)

    pooled_sd = np.sqrt((vt + vc) / 2.0)
    if pooled_sd <= 0:
        return 0.0
    return float((mt - mc) / pooled_sd)


def main() -> None:
    df, true_ate = generate_observational_data(n=6000, seed=7)
    x = df[["x1", "x2"]].to_numpy()
    t = df["t"].to_numpy()
    y = df["y"].to_numpy()

    ps_hat = fit_propensity_score(x, t)
    naive_ate = float(y[t == 1].mean() - y[t == 0].mean())
    iptw_ate = estimate_ate_iptw(y, t, ps_hat, eps=1e-3)

    ps_clipped = np.clip(ps_hat, 1e-3, 1.0 - 1e-3)
    iptw_weight = t / ps_clipped + (1 - t) / (1 - ps_clipped)

    balance = pd.DataFrame(
        {
            "covariate": ["x1", "x2"],
            "SMD_before_weighting": [
                standardized_mean_difference(df["x1"].to_numpy(), t),
                standardized_mean_difference(df["x2"].to_numpy(), t),
            ],
            "SMD_after_IPTW": [
                standardized_mean_difference(df["x1"].to_numpy(), t, iptw_weight),
                standardized_mean_difference(df["x2"].to_numpy(), t, iptw_weight),
            ],
        }
    )

    print("=== Propensity Score / IPTW Demo ===")
    print(f"Sample size: {len(df)}")
    print(f"True ATE: {true_ate:.4f}")
    print(f"Naive difference in means: {naive_ate:.4f}")
    print(f"IPTW ATE estimate: {iptw_ate:.4f}")
    print(f"Absolute error (Naive): {abs(naive_ate - true_ate):.4f}")
    print(f"Absolute error (IPTW): {abs(iptw_ate - true_ate):.4f}")
    print()
    print("Covariate balance (absolute SMD < 0.1 is commonly desired):")
    print(balance.to_string(index=False, float_format=lambda v: f"{v: .4f}"))


if __name__ == "__main__":
    main()
