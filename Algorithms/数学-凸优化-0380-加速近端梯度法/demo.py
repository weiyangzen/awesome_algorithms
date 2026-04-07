"""加速近端梯度法（FISTA）最小可运行示例。

目标问题（LASSO）：
    min_x 0.5/n * ||Ax - b||_2^2 + lam * ||x||_1

实现内容：
- ISTA（非加速近端梯度）
- FISTA（Nesterov 加速近端梯度，含简单重启）
- 与高精度参考解对照的收敛速度比较
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SolverResult:
    x: np.ndarray
    history: np.ndarray
    n_iter: int


def soft_threshold(v: np.ndarray, tau: float) -> np.ndarray:
    """prox_{tau * ||.||_1}(v) 的闭式解。"""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def objective(A: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> float:
    n = A.shape[0]
    r = A @ x - b
    return 0.5 * float(r @ r) / n + lam * float(np.abs(x).sum())


def grad_f(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """平滑项 f(x)=0.5/n||Ax-b||^2 的梯度。"""
    n = A.shape[0]
    return (A.T @ (A @ x - b)) / n


def lipschitz_constant(A: np.ndarray) -> float:
    """L = ||A||_2^2 / n，使用 SVD 精确计算（中小规模示例足够）。"""
    n = A.shape[0]
    sigma_max = float(np.linalg.svd(A, compute_uv=False)[0])
    return (sigma_max * sigma_max) / n


def ista(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    L: float,
    max_iter: int,
    tol: float,
    x0: Optional[np.ndarray] = None,
) -> SolverResult:
    d = A.shape[1]
    x = np.zeros(d, dtype=float) if x0 is None else x0.astype(float, copy=True)

    history: list[float] = []
    for k in range(max_iter):
        x_next = soft_threshold(x - grad_f(A, b, x) / L, lam / L)
        obj_next = objective(A, b, x_next, lam)
        history.append(obj_next)

        if k > 0:
            prev = history[-2]
            if abs(prev - obj_next) <= tol * max(1.0, abs(prev)):
                x = x_next
                break

        x = x_next

    hist = np.asarray(history, dtype=float)
    return SolverResult(x=x, history=hist, n_iter=hist.size)


def fista(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    L: float,
    max_iter: int,
    tol: float,
    x0: Optional[np.ndarray] = None,
    restart: bool = True,
) -> SolverResult:
    d = A.shape[1]
    x = np.zeros(d, dtype=float) if x0 is None else x0.astype(float, copy=True)
    y = x.copy()
    t = 1.0

    history: list[float] = []
    for k in range(max_iter):
        x_next = soft_threshold(y - grad_f(A, b, y) / L, lam / L)
        obj_next = objective(A, b, x_next, lam)
        history.append(obj_next)

        if k > 0:
            prev = history[-2]
            if abs(prev - obj_next) <= tol * max(1.0, abs(prev)):
                x = x_next
                break

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y_next = x_next + ((t - 1.0) / t_next) * (x_next - x)

        # 简单重启：若外推点目标值不降，则取消动量。
        if restart:
            if objective(A, b, y_next, lam) > obj_next:
                y_next = x_next.copy()
                t_next = 1.0

        x, y, t = x_next, y_next, t_next

    hist = np.asarray(history, dtype=float)
    return SolverResult(x=x, history=hist, n_iter=hist.size)


def make_problem(
    seed: int = 7,
    n_samples: int = 220,
    n_features: int = 500,
    sparsity: int = 28,
    noise_std: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    A = rng.normal(size=(n_samples, n_features)) / np.sqrt(n_samples)

    x_true = np.zeros(n_features, dtype=float)
    support = rng.choice(n_features, size=sparsity, replace=False)
    x_true[support] = rng.normal(loc=0.0, scale=2.0, size=sparsity)

    b = A @ x_true + noise_std * rng.normal(size=n_samples)
    return A, b, x_true


def support_f1_score(
    x_est: np.ndarray,
    x_true: np.ndarray,
    threshold: float = 1e-3,
) -> float:
    est = np.abs(x_est) > threshold
    tru = np.abs(x_true) > 0.0

    tp = float(np.logical_and(est, tru).sum())
    fp = float(np.logical_and(est, np.logical_not(tru)).sum())
    fn = float(np.logical_and(np.logical_not(est), tru).sum())

    if tp == 0.0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2.0 * precision * recall / (precision + recall)


def first_reach_iteration(values: np.ndarray, target: float) -> Optional[int]:
    idx = np.where(values <= target)[0]
    if idx.size == 0:
        return None
    return int(idx[0] + 1)


def report_solver(
    name: str,
    result: SolverResult,
    f_ref: float,
    x_true: np.ndarray,
    target_obj: float,
) -> None:
    x = result.x
    final_obj = float(result.history[-1])
    gap = final_obj - f_ref
    l2_err = float(np.linalg.norm(x - x_true))
    f1 = support_f1_score(x, x_true)
    nnz = int((np.abs(x) > 1e-3).sum())
    hit_iter = first_reach_iteration(result.history, target_obj)

    print(f"[{name}]")
    print(f"  iterations              : {result.n_iter}")
    print(f"  final objective         : {final_obj:.8f}")
    print(f"  objective gap to ref    : {gap:.3e}")
    print(f"  ||x - x_true||_2        : {l2_err:.6f}")
    print(f"  support F1              : {f1:.4f}")
    print(f"  estimated nonzeros      : {nnz}")
    print(
        "  first iter <= target obj: "
        + (str(hit_iter) if hit_iter is not None else "not reached")
    )


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    lam = 0.01
    max_iter_compare = 400
    tol = 1e-10

    A, b, x_true = make_problem()
    n, d = A.shape
    L = lipschitz_constant(A)

    # 高精度 FISTA 作为近似参考最优值（用于比较收敛速度）。
    ref = fista(A, b, lam=lam, L=L, max_iter=5000, tol=1e-13, restart=True)
    f_ref = float(ref.history[-1])

    ista_res = ista(A, b, lam=lam, L=L, max_iter=max_iter_compare, tol=tol)
    fista_res = fista(A, b, lam=lam, L=L, max_iter=max_iter_compare, tol=tol, restart=True)

    init_obj = objective(A, b, np.zeros(d, dtype=float), lam)
    target_obj = f_ref + (init_obj - f_ref) * 1e-3

    print("=== Accelerated Proximal Gradient (FISTA) Demo ===")
    print(f"problem size     : n={n}, d={d}")
    print(f"true sparsity    : {(np.abs(x_true) > 0).sum()}")
    print(f"lambda           : {lam}")
    print(f"Lipschitz L      : {L:.6f}")
    print(f"init objective   : {init_obj:.8f}")
    print(f"reference obj    : {f_ref:.8f}")
    print(f"target objective : {target_obj:.8f}")
    print()

    report_solver("ISTA", ista_res, f_ref=f_ref, x_true=x_true, target_obj=target_obj)
    print()
    report_solver("FISTA", fista_res, f_ref=f_ref, x_true=x_true, target_obj=target_obj)

    speedup = None
    i_ista = first_reach_iteration(ista_res.history, target_obj)
    i_fista = first_reach_iteration(fista_res.history, target_obj)
    if i_ista is not None and i_fista is not None and i_fista > 0:
        speedup = i_ista / i_fista

    print()
    if speedup is None:
        print("speedup (to target objective): unavailable")
    else:
        print(f"speedup (to target objective): {speedup:.2f}x")

    print("history tail (last 5 objective values):")
    print(f"  ISTA : {ista_res.history[-5:]}")
    print(f"  FISTA: {fista_res.history[-5:]}")


if __name__ == "__main__":
    main()
