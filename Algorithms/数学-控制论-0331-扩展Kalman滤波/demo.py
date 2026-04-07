"""扩展 Kalman 滤波（EKF）最小可运行示例.

场景:
- 状态: [px, py, vx, vy]
- 状态转移: 近似匀速模型（线性）
- 观测: 雷达量测 [range, bearing]（非线性）
"""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def f_state(x: np.ndarray, dt: float) -> np.ndarray:
    """状态转移函数 x_k = f(x_{k-1})."""
    px, py, vx, vy = x
    return np.array([px + vx * dt, py + vy * dt, vx, vy], dtype=float)


def jacobian_f(dt: float) -> np.ndarray:
    """状态转移对状态的 Jacobian."""
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def h_measurement(x: np.ndarray) -> np.ndarray:
    """观测函数 z = h(x) = [range, bearing]."""
    px, py = x[0], x[1]
    rng = np.sqrt(px * px + py * py)
    bearing = np.arctan2(py, px)
    return np.array([rng, bearing], dtype=float)


def jacobian_h(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """观测函数对状态的 Jacobian."""
    px, py = x[0], x[1]
    r2 = px * px + py * py
    r2 = max(r2, eps)
    r = np.sqrt(r2)
    return np.array(
        [
            [px / r, py / r, 0.0, 0.0],
            [-py / r2, px / r2, 0.0, 0.0],
        ],
        dtype=float,
    )


def ekf_predict(x: np.ndarray, p: np.ndarray, q: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """EKF 预测步骤."""
    f = jacobian_f(dt)
    x_pred = f_state(x, dt)
    p_pred = f @ p @ f.T + q
    return x_pred, p_pred


def ekf_update(
    x_pred: np.ndarray,
    p_pred: np.ndarray,
    z: np.ndarray,
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """EKF 更新步骤."""
    h = jacobian_h(x_pred)
    z_pred = h_measurement(x_pred)
    innovation = z - z_pred
    innovation[1] = wrap_angle(innovation[1])

    s = h @ p_pred @ h.T + r
    k = p_pred @ h.T @ np.linalg.inv(s)

    x_upd = x_pred + k @ innovation

    # Joseph 形式，数值稳定性更好
    i = np.eye(p_pred.shape[0], dtype=float)
    p_upd = (i - k @ h) @ p_pred @ (i - k @ h).T + k @ r @ k.T
    return x_upd, p_upd


def simulate_truth_and_measurements(
    n_steps: int,
    dt: float,
    q_acc: float,
    r_range: float,
    r_bearing: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """生成真实轨迹与观测数据."""
    rng = np.random.default_rng(seed)

    x_true = np.array([1.0, 1.0, 1.2, 0.4], dtype=float)
    f = jacobian_f(dt)
    g = np.array(
        [
            [0.5 * dt * dt, 0.0],
            [0.0, 0.5 * dt * dt],
            [dt, 0.0],
            [0.0, dt],
        ],
        dtype=float,
    )
    cov_acc = np.diag([q_acc, q_acc])

    truths = []
    measurements = []
    for _ in range(n_steps):
        a_noise = rng.multivariate_normal(mean=np.zeros(2), cov=cov_acc)
        x_true = f @ x_true + g @ a_noise
        truths.append(x_true.copy())

        z_clean = h_measurement(x_true)
        z_noisy = z_clean + np.array(
            [
                rng.normal(0.0, r_range),
                rng.normal(0.0, r_bearing),
            ],
            dtype=float,
        )
        z_noisy[1] = wrap_angle(z_noisy[1])
        measurements.append(z_noisy)

    return np.array(truths), np.array(measurements)


def run_ekf_demo() -> None:
    """运行 EKF MVP 并打印结果."""
    # 配置
    n_steps = 120
    dt = 0.1

    q_acc = 0.35
    r_range = 0.25
    r_bearing = 0.03

    # 真实系统与量测
    truths, zs = simulate_truth_and_measurements(
        n_steps=n_steps,
        dt=dt,
        q_acc=q_acc,
        r_range=r_range,
        r_bearing=r_bearing,
    )

    # EKF 噪声协方差建模
    g = np.array(
        [
            [0.5 * dt * dt, 0.0],
            [0.0, 0.5 * dt * dt],
            [dt, 0.0],
            [0.0, dt],
        ],
        dtype=float,
    )
    q = g @ np.diag([q_acc, q_acc]) @ g.T
    r = np.diag([r_range * r_range, r_bearing * r_bearing])

    # 初值故意与真实值有偏差
    x_est = np.array([0.3, -0.5, 0.0, 0.0], dtype=float)
    p_est = np.diag([2.0, 2.0, 1.0, 1.0])

    estimates = []
    for z in zs:
        x_pred, p_pred = ekf_predict(x_est, p_est, q, dt)
        x_est, p_est = ekf_update(x_pred, p_pred, z, r)
        estimates.append(x_est.copy())

    est = np.array(estimates)

    pos_rmse = np.sqrt(np.mean(np.sum((est[:, :2] - truths[:, :2]) ** 2, axis=1)))
    vel_rmse = np.sqrt(np.mean(np.sum((est[:, 2:] - truths[:, 2:]) ** 2, axis=1)))

    print("EKF demo completed")
    print(f"steps={n_steps}, dt={dt}")
    print(f"position RMSE={pos_rmse:.4f}")
    print(f"velocity RMSE={vel_rmse:.4f}")
    print("final true state:", np.array2string(truths[-1], precision=4))
    print("final estimated state:", np.array2string(est[-1], precision=4))


def main() -> None:
    run_ekf_demo()


if __name__ == "__main__":
    main()
