"""时间序列分析 - 状态空间模型 (本地水平模型) 的最小可运行 MVP.

该脚本演示:
1) 生成本地水平状态空间模型的合成数据
2) 使用卡尔曼滤波计算对数似然
3) 通过 MLE 估计过程噪声/观测噪声方差
4) 运行 RTS 平滑并给出简单评估与多步预测
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class FilterResult:
    predicted_mean: np.ndarray
    predicted_var: np.ndarray
    filtered_mean: np.ndarray
    filtered_var: np.ndarray
    innovation: np.ndarray
    innovation_var: np.ndarray
    loglik: float


@dataclass
class SmootherResult:
    smoothed_mean: np.ndarray
    smoothed_var: np.ndarray


def simulate_local_level(
    n: int,
    q: float,
    r: float,
    x0: float = 0.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """模拟本地水平模型.

    状态方程: x_t = x_{t-1} + w_t, w_t ~ N(0, q)
    观测方程: y_t = x_t + v_t, v_t ~ N(0, r)
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)
    x_prev = x0

    for t in range(n):
        x_t = x_prev + rng.normal(0.0, np.sqrt(q))
        y_t = x_t + rng.normal(0.0, np.sqrt(r))
        x[t] = x_t
        y[t] = y_t
        x_prev = x_t

    return x, y


def kalman_filter_local_level(
    y: np.ndarray,
    q: float,
    r: float,
    m0: float = 0.0,
    c0: float = 10.0,
) -> FilterResult:
    """对本地水平模型执行卡尔曼滤波并返回高斯对数似然."""
    n = y.shape[0]

    a = np.zeros(n, dtype=float)  # predicted mean: E[x_t | y_1..y_{t-1}]
    p = np.zeros(n, dtype=float)  # predicted var
    m = np.zeros(n, dtype=float)  # filtered mean: E[x_t | y_1..y_t]
    c = np.zeros(n, dtype=float)  # filtered var
    v = np.zeros(n, dtype=float)  # innovation
    f = np.zeros(n, dtype=float)  # innovation var

    loglik = 0.0
    m_prev = m0
    c_prev = c0

    for t in range(n):
        # Prediction step
        a_t = m_prev
        p_t = c_prev + q

        if np.isnan(y[t]):
            # 缺失观测: 仅执行状态传播
            m_t = a_t
            c_t = p_t
            v_t = 0.0
            f_t = np.nan
        else:
            # Update step
            v_t = y[t] - a_t
            f_t = p_t + r
            k_t = p_t / f_t
            m_t = a_t + k_t * v_t
            c_t = (1.0 - k_t) * p_t

            loglik += -0.5 * (
                np.log(2.0 * np.pi) + np.log(f_t) + (v_t * v_t) / f_t
            )

        a[t] = a_t
        p[t] = p_t
        m[t] = m_t
        c[t] = c_t
        v[t] = v_t
        f[t] = f_t

        m_prev = m_t
        c_prev = c_t

    return FilterResult(
        predicted_mean=a,
        predicted_var=p,
        filtered_mean=m,
        filtered_var=c,
        innovation=v,
        innovation_var=f,
        loglik=loglik,
    )


def rts_smoother_local_level(
    filtered: FilterResult,
    q: float,
) -> SmootherResult:
    """对本地水平模型执行 Rauch-Tung-Striebel 平滑."""
    m = filtered.filtered_mean
    c = filtered.filtered_var
    a = filtered.predicted_mean
    p = filtered.predicted_var
    n = m.shape[0]

    s = np.zeros(n, dtype=float)
    s_var = np.zeros(n, dtype=float)

    s[-1] = m[-1]
    s_var[-1] = c[-1]

    for t in range(n - 2, -1, -1):
        denom = c[t] + q  # 等于 p[t+1]，在本地水平模型下更直观
        j_t = c[t] / denom
        s[t] = m[t] + j_t * (s[t + 1] - a[t + 1])
        s_var[t] = c[t] + (j_t * j_t) * (s_var[t + 1] - p[t + 1])

    return SmootherResult(smoothed_mean=s, smoothed_var=s_var)


def fit_local_level_mle(y: np.ndarray) -> tuple[float, float, float]:
    """用极大似然估计 q, r (通过 log 参数保证正值)."""

    obs_var = float(np.nanvar(y))
    theta0 = np.log(
        np.array(
            [
                max(obs_var * 0.20, 1e-6),  # log(q)
                max(obs_var * 0.80, 1e-6),  # log(r)
            ],
            dtype=float,
        )
    )

    def objective(theta: np.ndarray) -> float:
        q_hat = float(np.exp(theta[0]))
        r_hat = float(np.exp(theta[1]))
        result = kalman_filter_local_level(y=y, q=q_hat, r=r_hat)
        return -result.loglik

    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0), (-20.0, 20.0)],
    )

    if not res.success:
        raise RuntimeError(f"MLE 优化失败: {res.message}")

    q_mle = float(np.exp(res.x[0]))
    r_mle = float(np.exp(res.x[1]))
    nll = float(res.fun)
    return q_mle, r_mle, nll


def forecast_local_level(
    last_mean: float,
    last_var: float,
    q: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """本地水平模型的 h 步预测."""
    means = np.full(horizon, last_mean, dtype=float)
    vars_ = np.array([last_var + (h + 1) * q for h in range(horizon)], dtype=float)
    return means, vars_


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    # 1) 生成数据
    n = 220
    q_true = 0.08
    r_true = 0.45
    x_true, y_full = simulate_local_level(n=n, q=q_true, r=r_true, seed=2026)

    # 2) 构造部分缺失观测, 展示状态空间模型的鲁棒性
    y_obs = y_full.copy()
    y_obs[::37] = np.nan

    # 3) MLE 估计参数
    q_hat, r_hat, nll = fit_local_level_mle(y_obs)

    # 4) 滤波 + 平滑
    filt = kalman_filter_local_level(y=y_obs, q=q_hat, r=r_hat)
    smooth = rts_smoother_local_level(filtered=filt, q=q_hat)

    # 5) 评估
    state_rmse_filter = rmse(x_true, filt.filtered_mean)
    state_rmse_smooth = rmse(x_true, smooth.smoothed_mean)

    observed_mask = ~np.isnan(y_obs)
    one_step_rmse = rmse(y_obs[observed_mask], filt.predicted_mean[observed_mask])

    # 6) 多步预测
    horizon = 5
    pred_mean, pred_var = forecast_local_level(
        last_mean=float(filt.filtered_mean[-1]),
        last_var=float(filt.filtered_var[-1]),
        q=q_hat,
        horizon=horizon,
    )

    # 7) 输出摘要
    print("=== 状态空间模型 MVP: 本地水平模型 ===")
    print(f"样本数: {n}, 缺失观测数: {np.isnan(y_obs).sum()}")
    print(f"真实参数: q={q_true:.4f}, r={r_true:.4f}")
    print(f"MLE参数: q={q_hat:.4f}, r={r_hat:.4f}, NLL={nll:.4f}")
    print(f"状态RMSE(滤波): {state_rmse_filter:.4f}")
    print(f"状态RMSE(平滑): {state_rmse_smooth:.4f}")
    print(f"一步预测RMSE(观测): {one_step_rmse:.4f}")

    print("\n未来5步预测 (mean, std):")
    for i in range(horizon):
        print(f"t+{i+1}: mean={pred_mean[i]:.4f}, std={np.sqrt(pred_var[i]):.4f}")


if __name__ == "__main__":
    main()
