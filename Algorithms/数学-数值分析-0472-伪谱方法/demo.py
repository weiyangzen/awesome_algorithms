"""Fourier pseudo-spectral MVP for 1D forced viscous Burgers equation.

Problem:
    u_t + u * u_x = nu * u_xx + s(x, t), x in [0, L), periodic boundary.

Manufactured exact solution:
    u(x, t) = exp(-t) * sin(x)

The script solves the PDE with:
    - spectral differentiation via FFT/IFFT,
    - physical-space nonlinear product,
    - 2/3-rule de-aliasing for nonlinear term,
    - explicit RK4 time integration.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np

LENGTH = 2.0 * math.pi
NU = 0.01
T_FINAL = 1.0


def periodic_grid(n: int, length: float = LENGTH) -> np.ndarray:
    """Create N periodic nodes on [0, length)."""
    if n < 8:
        raise ValueError(f"n must be >= 8, got {n}")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"length must be positive finite, got {length}")
    return np.linspace(0.0, length, n, endpoint=False, dtype=float)


def angular_wavenumbers(n: int, length: float = LENGTH) -> np.ndarray:
    """Return angular wavenumbers aligned with numpy FFT ordering."""
    return 2.0 * math.pi * np.fft.fftfreq(n, d=length / n)


def dealias_mask_23(n: int) -> np.ndarray:
    """Build 2/3-rule mask in FFT index space."""
    freq_index = np.fft.fftfreq(n) * n
    cutoff = n // 3
    return np.abs(freq_index) <= cutoff


def exact_solution(x: np.ndarray, t: float) -> np.ndarray:
    """Manufactured analytic solution u(x, t) = exp(-t) * sin(x)."""
    return math.exp(-t) * np.sin(x)


def forcing_term(x: np.ndarray, t: float, nu: float = NU) -> np.ndarray:
    """Forcing that makes exact_solution satisfy Burgers equation."""
    exp_t = math.exp(-t)
    return (nu - 1.0) * exp_t * np.sin(x) + 0.5 * (exp_t * exp_t) * np.sin(2.0 * x)


def burgers_rhs_pseudospectral(
    u: np.ndarray,
    t: float,
    x: np.ndarray,
    k: np.ndarray,
    dealias_mask: np.ndarray,
    nu: float = NU,
) -> np.ndarray:
    """Compute du/dt using pseudo-spectral spatial discretization."""
    if u.ndim != 1:
        raise ValueError("u must be 1D")
    if u.shape != x.shape or u.shape != k.shape or u.shape != dealias_mask.shape:
        raise ValueError("u, x, k, and dealias_mask must share the same shape")
    if not np.all(np.isfinite(u)):
        raise ValueError("u contains non-finite values")

    u_hat = np.fft.fft(u)
    ux = np.fft.ifft(1j * k * u_hat).real
    uxx = np.fft.ifft(-(k * k) * u_hat).real

    nonlinear = u * ux
    nonlinear_hat = np.fft.fft(nonlinear)
    nonlinear_hat *= dealias_mask
    nonlinear_filtered = np.fft.ifft(nonlinear_hat).real

    rhs = -nonlinear_filtered + nu * uxx + forcing_term(x, t, nu)
    return rhs


def rk4_step(
    u: np.ndarray,
    t: float,
    dt: float,
    rhs_func: Callable[[np.ndarray, float], np.ndarray],
) -> np.ndarray:
    """One explicit RK4 step."""
    k1 = rhs_func(u, t)
    k2 = rhs_func(u + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs_func(u + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs_func(u + dt * k3, t + dt)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_burgers_pseudospectral(
    n: int,
    t_final: float = T_FINAL,
    nu: float = NU,
    cfl: float = 0.20,
    length: float = LENGTH,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Integrate from t=0 to t_final and return (x, u_final, steps, dt)."""
    if not np.isfinite(t_final) or t_final <= 0.0:
        raise ValueError(f"t_final must be positive finite, got {t_final}")
    if not np.isfinite(nu) or nu < 0.0:
        raise ValueError(f"nu must be non-negative finite, got {nu}")
    if not np.isfinite(cfl) or cfl <= 0.0:
        raise ValueError(f"cfl must be positive finite, got {cfl}")

    x = periodic_grid(n, length)
    k = angular_wavenumbers(n, length)
    mask = dealias_mask_23(n)

    u = exact_solution(x, t=0.0)

    dx = length / n
    dt_adv = cfl * dx
    dt_diff = 0.4 * dx * dx / max(nu, 1.0e-14)
    dt = min(dt_adv, dt_diff, t_final)
    n_steps = max(1, int(math.ceil(t_final / dt)))
    dt = t_final / n_steps

    def rhs(local_u: np.ndarray, local_t: float) -> np.ndarray:
        return burgers_rhs_pseudospectral(local_u, local_t, x, k, mask, nu)

    for step in range(n_steps):
        t_now = step * dt
        u = rk4_step(u, t_now, dt, rhs)

    return x, u, n_steps, dt


def error_metrics(u_num: np.ndarray, u_ref: np.ndarray) -> tuple[float, float]:
    """Return L2 and Linf errors."""
    if u_num.shape != u_ref.shape:
        raise ValueError("u_num and u_ref must have same shape")
    diff = u_num - u_ref
    l2_err = float(np.sqrt(np.mean(diff * diff)))
    linf_err = float(np.max(np.abs(diff)))
    return l2_err, linf_err


def safe_log2_rate(prev_err: float, curr_err: float) -> float:
    """Compute log2(prev/curr) safely near machine precision."""
    if prev_err <= 1.0e-14 or curr_err <= 1.0e-14:
        return float("nan")
    return float(np.log2(prev_err / curr_err))


def run_convergence_study(
    grid_sizes: Iterable[int],
    t_final: float = T_FINAL,
    nu: float = NU,
) -> list[dict[str, float]]:
    """Run solver on multiple grids and collect error/rate statistics."""
    records: list[dict[str, float]] = []
    prev_l2 = float("nan")
    prev_linf = float("nan")

    for n in grid_sizes:
        x, u_num, steps, dt = integrate_burgers_pseudospectral(n=n, t_final=t_final, nu=nu)
        u_ref = exact_solution(x, t_final)
        l2_err, linf_err = error_metrics(u_num, u_ref)

        rec = {
            "N": float(n),
            "steps": float(steps),
            "dt": float(dt),
            "L2_error": l2_err,
            "Linf_error": linf_err,
            "rate_L2": safe_log2_rate(prev_l2, l2_err),
            "rate_Linf": safe_log2_rate(prev_linf, linf_err),
        }
        records.append(rec)
        prev_l2 = l2_err
        prev_linf = linf_err

    return records


def print_results_table(records: list[dict[str, float]]) -> None:
    """Print convergence table."""
    header = (
        f"{'N':>6} | {'steps':>7} | {'dt':>10} | {'L2_error':>12} | {'Linf_error':>12} | "
        f"{'rate(L2)':>9} | {'rate(Linf)':>10}"
    )
    print(header)
    print("-" * len(header))

    for rec in records:
        n = int(rec["N"])
        steps = int(rec["steps"])
        dt = rec["dt"]
        l2e = rec["L2_error"]
        lie = rec["Linf_error"]
        r2 = rec["rate_L2"]
        ri = rec["rate_Linf"]

        r2_str = f"{r2:9.3f}" if np.isfinite(r2) else f"{'-':>9}"
        ri_str = f"{ri:10.3f}" if np.isfinite(ri) else f"{'-':>10}"

        print(
            f"{n:6d} | {steps:7d} | {dt:10.3e} | {l2e:12.3e} | {lie:12.3e} | "
            f"{r2_str} | {ri_str}"
        )


def main() -> None:
    grid_sizes = [32, 64, 128, 256]
    pass_threshold = 2.0e-5

    records = run_convergence_study(grid_sizes=grid_sizes)

    print("Fourier Pseudo-Spectral Method for Forced Viscous Burgers Equation")
    print(f"Exact solution: u(x,t)=exp(-t)sin(x), nu={NU}, T={T_FINAL}")
    print_results_table(records)

    finest = records[-1]
    max_error_all = max(rec["Linf_error"] for rec in records)
    passed = finest["Linf_error"] < pass_threshold

    print("\nSummary")
    print(f"  max_error_all = {max_error_all:.3e}")
    print(f"  finest_grid_Linf_error = {finest['Linf_error']:.3e}")
    print(f"  pass_threshold = {pass_threshold:.1e}")
    print(f"  pass = {passed}")


if __name__ == "__main__":
    main()
