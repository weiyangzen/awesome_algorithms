"""Minimal runnable MVP for Implicit Runge-Kutta (Gauss-Legendre 2-stage)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ScalarFunc = Callable[[float, float], float]


@dataclass
class IRKTableau:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray


@dataclass
class StepDiagnostic:
    step_index: int
    newton_iters: int
    residual_inf_norm: float


@dataclass
class IRKResult:
    t_values: np.ndarray
    y_values: np.ndarray
    diagnostics: list[StepDiagnostic]


def gauss_legendre_2stage_tableau() -> IRKTableau:
    """Return 2-stage Gauss-Legendre tableau (order 4, A-stable)."""
    sqrt3 = float(np.sqrt(3.0))
    a = np.array(
        [
            [0.25, 0.25 - sqrt3 / 6.0],
            [0.25 + sqrt3 / 6.0, 0.25],
        ],
        dtype=float,
    )
    b = np.array([0.5, 0.5], dtype=float)
    c = np.array([0.5 - sqrt3 / 6.0, 0.5 + sqrt3 / 6.0], dtype=float)
    return IRKTableau(a=a, b=b, c=c)


def check_inputs(t0: float, t_end: float, y0: float, n_steps: int, tol: float, max_iter: int) -> None:
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if t_end <= t0:
        raise ValueError("t_end must be greater than t0")
    if not np.isfinite(t0) or not np.isfinite(t_end) or not np.isfinite(y0):
        raise ValueError("t0, t_end, y0 must be finite")
    if tol <= 0.0 or not np.isfinite(tol):
        raise ValueError("tol must be positive and finite")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")


def stage_residual(
    func: ScalarFunc,
    t_n: float,
    y_n: float,
    h: float,
    tableau: IRKTableau,
    k_vec: np.ndarray,
) -> np.ndarray:
    """Residual G(k) for implicit stage equations: G_i = k_i - f(t_i, y_i)."""
    s = int(k_vec.size)
    if s != int(tableau.b.size):
        raise ValueError("stage dimension mismatch")

    g = np.empty(s, dtype=float)
    a_dot_k = tableau.a @ k_vec
    for i in range(s):
        t_stage = t_n + tableau.c[i] * h
        y_stage = y_n + h * a_dot_k[i]
        f_val = func(float(t_stage), float(y_stage))
        g[i] = k_vec[i] - f_val

    return g


def numerical_jacobian(
    residual_func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Finite-difference Jacobian: J_ij = d(residual_i)/d(x_j)."""
    if eps <= 0.0 or not np.isfinite(eps):
        raise ValueError("eps must be positive and finite")

    base = residual_func(x)
    n = int(x.size)
    jac = np.empty((n, n), dtype=float)

    for j in range(n):
        dx = np.zeros_like(x)
        perturb = eps * max(1.0, abs(float(x[j])))
        dx[j] = perturb
        col = (residual_func(x + dx) - base) / perturb
        jac[:, j] = col

    return jac


def newton_solve_stage_equations(
    func: ScalarFunc,
    t_n: float,
    y_n: float,
    h: float,
    tableau: IRKTableau,
    tol: float,
    max_iter: int,
    fd_eps: float = 1e-8,
) -> tuple[np.ndarray, int, float]:
    """Solve stage equations G(k)=0 using Newton iteration."""
    s = int(tableau.b.size)
    k = np.full(s, float(func(t_n, y_n)), dtype=float)

    def residual_local(v: np.ndarray) -> np.ndarray:
        return stage_residual(
            func=func,
            t_n=t_n,
            y_n=y_n,
            h=h,
            tableau=tableau,
            k_vec=v,
        )

    for it in range(1, max_iter + 1):
        g = residual_local(k)
        res_norm = float(np.linalg.norm(g, ord=np.inf))
        if not np.isfinite(res_norm):
            raise RuntimeError(f"non-finite residual at step t={t_n:.6f}")
        if res_norm <= tol:
            return k, it - 1, res_norm

        jac = numerical_jacobian(residual_local, k, eps=fd_eps)
        if not np.all(np.isfinite(jac)):
            raise RuntimeError(f"non-finite Jacobian at step t={t_n:.6f}")

        try:
            delta = np.linalg.solve(jac, -g)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"singular Newton system at step t={t_n:.6f}") from exc

        if not np.all(np.isfinite(delta)):
            raise RuntimeError(f"non-finite Newton increment at step t={t_n:.6f}")

        k = k + delta
        delta_norm = float(np.linalg.norm(delta, ord=np.inf))
        if delta_norm <= tol * max(1.0, float(np.linalg.norm(k, ord=np.inf))):
            g_new = residual_local(k)
            res_new = float(np.linalg.norm(g_new, ord=np.inf))
            return k, it, res_new

    final_res = float(np.linalg.norm(residual_local(k), ord=np.inf))
    raise RuntimeError(
        f"Newton did not converge at step t={t_n:.6f}; residual={final_res:.3e}"
    )


def implicit_runge_kutta_solve(
    func: ScalarFunc,
    t0: float,
    y0: float,
    t_end: float,
    n_steps: int,
    tol: float = 1e-12,
    max_iter: int = 20,
) -> IRKResult:
    """Solve scalar ODE by Gauss-Legendre 2-stage implicit RK."""
    check_inputs(t0=t0, t_end=t_end, y0=y0, n_steps=n_steps, tol=tol, max_iter=max_iter)

    tableau = gauss_legendre_2stage_tableau()
    h = (t_end - t0) / n_steps

    t_values = np.linspace(t0, t_end, n_steps + 1, dtype=float)
    y_values = np.empty(n_steps + 1, dtype=float)
    y_values[0] = y0
    diagnostics: list[StepDiagnostic] = []

    for n in range(n_steps):
        t_n = float(t_values[n])
        y_n = float(y_values[n])

        k_vec, n_iters, res_norm = newton_solve_stage_equations(
            func=func,
            t_n=t_n,
            y_n=y_n,
            h=h,
            tableau=tableau,
            tol=tol,
            max_iter=max_iter,
        )

        y_next = y_n + h * float(np.dot(tableau.b, k_vec))
        if not np.isfinite(y_next):
            raise RuntimeError(f"non-finite state at step {n}")

        y_values[n + 1] = y_next
        diagnostics.append(
            StepDiagnostic(
                step_index=n,
                newton_iters=n_iters,
                residual_inf_norm=res_norm,
            )
        )

    return IRKResult(t_values=t_values, y_values=y_values, diagnostics=diagnostics)


def explicit_euler_solve(
    func: ScalarFunc,
    t0: float,
    y0: float,
    t_end: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline explicit method for stiff-case contrast."""
    h = (t_end - t0) / n_steps
    t_values = np.linspace(t0, t_end, n_steps + 1, dtype=float)
    y_values = np.empty(n_steps + 1, dtype=float)
    y_values[0] = y0

    for i in range(n_steps):
        y_values[i + 1] = y_values[i] + h * func(float(t_values[i]), float(y_values[i]))

    return t_values, y_values


def exact_nonstiff(t: np.ndarray) -> np.ndarray:
    """Exact solution of y' = y - t^2 + 1, y(0)=0.5."""
    return (t + 1.0) ** 2 - 0.5 * np.exp(t)


def rhs_nonstiff(t: float, y: float) -> float:
    return y - t * t + 1.0


def rhs_stiff(_t: float, y: float) -> float:
    return -15.0 * y


def exact_stiff(t: np.ndarray) -> np.ndarray:
    return np.exp(-15.0 * t)


def estimate_order(h_values: np.ndarray, err_values: np.ndarray) -> float:
    """Least-squares slope in log-log plane."""
    mask = (h_values > 0.0) & (err_values > 0.0) & np.isfinite(err_values)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")

    x = np.log(h_values[mask])
    y = np.log(err_values[mask])
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def run_convergence_demo() -> None:
    print("=" * 92)
    print("Convergence demo: y' = y - t^2 + 1, y(0)=0.5, t in [0, 2]")
    print("Method: implicit RK (Gauss-Legendre 2-stage)")
    print("=" * 92)

    t0 = 0.0
    t_end = 2.0
    y0 = 0.5
    step_list = [10, 20, 40, 80, 160]

    h_values: list[float] = []
    errors: list[float] = []
    mean_iters: list[float] = []
    max_residuals: list[float] = []

    print(f"{'N':>6} {'h':>12} {'IRK_max_err':>18} {'mean_newton_iters':>20} {'max_stage_res':>18}")

    for n_steps in step_list:
        result = implicit_runge_kutta_solve(
            func=rhs_nonstiff,
            t0=t0,
            y0=y0,
            t_end=t_end,
            n_steps=n_steps,
            tol=1e-12,
            max_iter=20,
        )

        y_exact = exact_nonstiff(result.t_values)
        err = float(np.max(np.abs(result.y_values - y_exact)))

        h = (t_end - t0) / n_steps
        h_values.append(h)
        errors.append(err)

        iter_arr = np.array([d.newton_iters for d in result.diagnostics], dtype=float)
        res_arr = np.array([d.residual_inf_norm for d in result.diagnostics], dtype=float)
        mean_it = float(np.mean(iter_arr))
        max_res = float(np.max(res_arr))
        mean_iters.append(mean_it)
        max_residuals.append(max_res)

        print(f"{n_steps:6d} {h:12.6f} {err:18.10e} {mean_it:20.4f} {max_res:18.10e}")

    order = estimate_order(np.array(h_values), np.array(errors))
    print("-" * 92)
    print(f"Estimated order (expect near 4): {order:.4f}")

    idx = 1
    n_show = step_list[idx]
    result_show = implicit_runge_kutta_solve(
        func=rhs_nonstiff,
        t0=t0,
        y0=y0,
        t_end=t_end,
        n_steps=n_show,
        tol=1e-12,
        max_iter=20,
    )
    y_exact_show = exact_nonstiff(result_show.t_values)

    print("\nTrajectory preview (N=20, first 5 points):")
    print(f"{'t':>10} {'y_num':>16} {'y_exact':>16} {'abs_err':>16}")
    for i in range(5):
        t_i = float(result_show.t_values[i])
        y_num = float(result_show.y_values[i])
        y_ex = float(y_exact_show[i])
        print(f"{t_i:10.4f} {y_num:16.8f} {y_ex:16.8f} {abs(y_num - y_ex):16.8e}")


def run_stiff_demo() -> None:
    print("\n" + "=" * 92)
    print("Stiff contrast: y' = -15 y, y(0)=1, t in [0, 1], h=0.2")
    print("Compare implicit RK vs explicit Euler")
    print("=" * 92)

    t0 = 0.0
    t_end = 1.0
    y0 = 1.0
    n_steps = 5

    irk_res = implicit_runge_kutta_solve(
        func=rhs_stiff,
        t0=t0,
        y0=y0,
        t_end=t_end,
        n_steps=n_steps,
        tol=1e-12,
        max_iter=20,
    )
    t_euler, y_euler = explicit_euler_solve(
        func=rhs_stiff,
        t0=t0,
        y0=y0,
        t_end=t_end,
        n_steps=n_steps,
    )
    y_exact = exact_stiff(irk_res.t_values)

    irk_err = float(abs(irk_res.y_values[-1] - y_exact[-1]))
    euler_err = float(abs(y_euler[-1] - y_exact[-1]))

    print(f"Final y(T) exact : {float(y_exact[-1]):.10e}")
    print(f"Final y(T) IRK   : {float(irk_res.y_values[-1]):.10e}  |err|={irk_err:.3e}")
    print(f"Final y(T) Euler : {float(y_euler[-1]):.10e}  |err|={euler_err:.3e}")
    print("Preview first 4 points:")
    print(f"{'t':>8} {'IRK':>16} {'Euler':>16} {'Exact':>16}")
    for i in range(4):
        print(
            f"{float(irk_res.t_values[i]):8.4f} "
            f"{float(irk_res.y_values[i]):16.8e} "
            f"{float(y_euler[i]):16.8e} "
            f"{float(y_exact[i]):16.8e}"
        )


def main() -> None:
    run_convergence_demo()
    run_stiff_demo()


if __name__ == "__main__":
    main()
