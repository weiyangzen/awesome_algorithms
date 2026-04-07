"""Kalman 滤波最小可运行示例（线性高斯状态估计）.

场景:
- 状态: [px, py, vx, vy]
- 状态转移: 匀速模型（线性）
- 观测: 位置直接量测 [px, py]（线性）
"""

from __future__ import annotations

import numpy as np


def check_matrix(name: str, arr: np.ndarray, shape: tuple[int, int] | None = None) -> None:
    """检查矩阵维度和数值合法性."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got ndim={arr.ndim}")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")


def check_vector(name: str, arr: np.ndarray, dim: int | None = None) -> None:
    """检查向量维度和数值合法性."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got ndim={arr.ndim}")
    if dim is not None and arr.shape[0] != dim:
        raise ValueError(f"{name} size mismatch: expected {dim}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")


def build_cv_model(dt: float, q_acc: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造匀速模型矩阵 F/H/Q（Q 由加速度噪声离散化得到）与 G."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if q_acc < 0.0:
        raise ValueError("q_acc must be non-negative")

    f = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    h = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
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
    return f, h, q, g


def kf_predict(
    x_prev: np.ndarray,
    p_prev: np.ndarray,
    f: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Kalman 预测步骤."""
    n = x_prev.shape[0]
    check_vector("x_prev", x_prev)
    check_matrix("p_prev", p_prev, shape=(n, n))
    check_matrix("f", f, shape=(n, n))
    check_matrix("q", q, shape=(n, n))

    x_pred = f @ x_prev
    p_pred = f @ p_prev @ f.T + q
    return x_pred, p_pred


def kf_update(
    x_pred: np.ndarray,
    p_pred: np.ndarray,
    z: np.ndarray,
    h: np.ndarray,
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kalman 更新步骤，返回更新后的状态、协方差、创新和增益."""
    n = x_pred.shape[0]
    m = z.shape[0]

    check_vector("x_pred", x_pred, dim=n)
    check_matrix("p_pred", p_pred, shape=(n, n))
    check_vector("z", z, dim=m)
    check_matrix("h", h, shape=(m, n))
    check_matrix("r", r, shape=(m, m))

    innovation = z - h @ x_pred
    s = h @ p_pred @ h.T + r
    # 用 solve 替代显式逆提升数值稳定性。
    k = np.linalg.solve(s.T, (p_pred @ h.T).T).T

    x_upd = x_pred + k @ innovation

    # Joseph 形式保证协方差数值上更稳定。
    i = np.eye(n, dtype=float)
    p_upd = (i - k @ h) @ p_pred @ (i - k @ h).T + k @ r @ k.T

    if not np.all(np.isfinite(x_upd)) or not np.all(np.isfinite(p_upd)):
        raise RuntimeError("non-finite values detected in Kalman update")

    return x_upd, p_upd, innovation, k


def simulate_truth_and_measurements(
    n_steps: int,
    dt: float,
    q_acc: float,
    r_pos: float,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """生成线性系统真值轨迹与带噪位置观测."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if r_pos <= 0.0:
        raise ValueError("r_pos must be positive")

    f, h, _, g = build_cv_model(dt=dt, q_acc=q_acc)
    cov_acc = np.diag([q_acc, q_acc])

    rng = np.random.default_rng(seed)
    x_true = np.array([0.0, 0.0, 1.0, -0.2], dtype=float)

    truths = []
    measurements = []
    for _ in range(n_steps):
        acc_noise = rng.multivariate_normal(mean=np.zeros(2), cov=cov_acc)
        x_true = f @ x_true + g @ acc_noise
        z = h @ x_true + rng.normal(0.0, r_pos, size=2)
        truths.append(x_true.copy())
        measurements.append(z)

    return np.asarray(truths), np.asarray(measurements)


def run_kalman_demo() -> None:
    """运行 Kalman 滤波演示并打印核心指标."""
    n_steps = 100
    dt = 0.1
    q_acc = 0.25
    r_pos = 0.45

    truths, zs = simulate_truth_and_measurements(
        n_steps=n_steps,
        dt=dt,
        q_acc=q_acc,
        r_pos=r_pos,
        seed=2026,
    )

    f, h, q, _ = build_cv_model(dt=dt, q_acc=q_acc)
    r = np.diag([r_pos * r_pos, r_pos * r_pos])

    x_est = np.array([1.5, -1.2, 0.0, 0.0], dtype=float)
    p_est = np.diag([8.0, 8.0, 2.0, 2.0])

    estimates = []
    innovation_norms = []
    gain_trace = []

    for z in zs:
        x_pred, p_pred = kf_predict(x_est, p_est, f, q)
        x_est, p_est, innovation, k = kf_update(x_pred, p_pred, z, h, r)

        estimates.append(x_est.copy())
        innovation_norms.append(float(np.linalg.norm(innovation, ord=2)))
        gain_trace.append(float(np.trace(k @ h)))

    estimates_arr = np.asarray(estimates)

    pos_rmse = float(np.sqrt(np.mean(np.sum((estimates_arr[:, :2] - truths[:, :2]) ** 2, axis=1))))
    vel_rmse = float(np.sqrt(np.mean(np.sum((estimates_arr[:, 2:] - truths[:, 2:]) ** 2, axis=1))))
    meas_rmse = float(np.sqrt(np.mean(np.sum((zs - truths[:, :2]) ** 2, axis=1))))

    print("Kalman filter demo completed")
    print(f"steps={n_steps}, dt={dt}")
    print(f"measurement position RMSE={meas_rmse:.4f}")
    print(f"filtered position RMSE={pos_rmse:.4f}")
    print(f"filtered velocity RMSE={vel_rmse:.4f}")
    print(f"mean innovation norm={np.mean(innovation_norms):.4f}")
    print(f"mean trace(KH)={np.mean(gain_trace):.4f}")
    print("final true state:", np.array2string(truths[-1], precision=4))
    print("final estimated state:", np.array2string(estimates_arr[-1], precision=4))


def main() -> None:
    run_kalman_demo()


if __name__ == "__main__":
    main()
