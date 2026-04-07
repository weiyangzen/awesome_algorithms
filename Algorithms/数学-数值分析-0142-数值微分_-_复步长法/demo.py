"""数值微分 - 复步长法（Complex-Step Differentiation）最小可运行示例。"""

from __future__ import annotations

import numpy as np


def complex_step_derivative(f, x: float, h: float = 1e-20) -> float:
    """Use Im(f(x + i*h))/h to estimate f'(x) for analytic f."""
    z = x + 1j * h
    return float(np.imag(f(z)) / h)


def forward_difference(f, x: float, h: float) -> float:
    """Classic forward finite difference."""
    return float((f(x + h) - f(x)) / h)


def central_difference(f, x: float, h: float) -> float:
    """Classic central finite difference."""
    return float((f(x + h) - f(x - h)) / (2.0 * h))


def complex_step_gradient(f, x: np.ndarray, h: float = 1e-20) -> np.ndarray:
    """Compute gradient via per-dimension complex-step perturbation."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x, dtype=float)
    for k in range(x.size):
        z = x.astype(np.complex128, copy=True)
        z[k] += 1j * h
        grad[k] = np.imag(f(z)) / h
    return grad


def f_scalar(x):
    """Analytic scalar function that supports complex input."""
    return np.exp(x) * np.sin(x) + x**3


def f_scalar_prime_exact(x: float) -> float:
    return float(np.exp(x) * (np.sin(x) + np.cos(x)) + 3.0 * x**2)


def f_vector(x: np.ndarray):
    """2D scalar-valued objective for gradient demo."""
    return np.exp(x[0]) * np.sin(x[1]) + x[0] ** 2 * x[1]


def grad_vector_exact(x: np.ndarray) -> np.ndarray:
    g0 = np.exp(x[0]) * np.sin(x[1]) + 2.0 * x[0] * x[1]
    g1 = np.exp(x[0]) * np.cos(x[1]) + x[0] ** 2
    return np.array([g0, g1], dtype=float)


def run_scalar_demo() -> None:
    x0 = 1.0
    exact = f_scalar_prime_exact(x0)
    h_list = [1e-2, 1e-6, 1e-12, 1e-20]

    print("=== 标量导数演示: f(x) = exp(x)*sin(x) + x^3, x=1 ===")
    print(f"解析导数: {exact:.16e}")
    print(
        f"{'h':>10} | {'forward_err':>14} | {'central_err':>14} | {'complex_err':>14}"
    )
    print("-" * 64)

    for h in h_list:
        fd = forward_difference(f_scalar, x0, h)
        cd = central_difference(f_scalar, x0, h)
        cs = complex_step_derivative(f_scalar, x0, h)
        fd_err = abs(fd - exact)
        cd_err = abs(cd - exact)
        cs_err = abs(cs - exact)
        print(f"{h:10.1e} | {fd_err:14.6e} | {cd_err:14.6e} | {cs_err:14.6e}")

    print()


def run_vector_demo() -> None:
    x = np.array([0.7, -0.3], dtype=float)
    exact = grad_vector_exact(x)
    est = complex_step_gradient(f_vector, x, h=1e-20)
    l2_err = np.linalg.norm(est - exact)

    print("=== 多变量梯度演示: f(x0,x1)=exp(x0)*sin(x1)+x0^2*x1 ===")
    print(f"x = {x}")
    print(f"复步长梯度估计: {est}")
    print(f"解析梯度:       {exact}")
    print(f"L2 误差: {l2_err:.6e}")


def main() -> None:
    np.set_printoptions(precision=12, suppress=False)
    run_scalar_demo()
    run_vector_demo()


if __name__ == "__main__":
    main()
