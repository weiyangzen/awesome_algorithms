"""Minimal runnable MVP for flux limiter schemes (1D linear advection).

Model:
    u_t + a u_x = 0, x in [0,1), periodic BC.

Implemented schemes:
1. First-order upwind (baseline).
2. TVD flux-limiter scheme with selectable limiter
   (minmod, vanleer, superbee, mc).

The script runs deterministic experiments and prints diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayFunc = Callable[[np.ndarray], np.ndarray]


@dataclass
class ExperimentResult:
    scheme: str
    nx: int
    n_steps: int
    cfl: float
    l1: float
    l2: float
    linf: float
    mass_error: float
    tv0: float
    tvt: float
    overshoot: float


def ensure_finite(name: str, arr: np.ndarray) -> np.ndarray:
    """Raise a clear error if numerical values diverge."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def limiter_phi(r: np.ndarray, kind: str) -> np.ndarray:
    """Evaluate common TVD limiter functions."""
    if kind == "minmod":
        return np.maximum(0.0, np.minimum(1.0, r))
    if kind == "vanleer":
        return (r + np.abs(r)) / (1.0 + np.abs(r))
    if kind == "superbee":
        return np.maximum(
            0.0,
            np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)),
        )
    if kind == "mc":
        return np.maximum(
            0.0,
            np.minimum(np.minimum((1.0 + r) * 0.5, 2.0), 2.0 * r),
        )
    raise ValueError(f"Unknown limiter kind: {kind}")


def initial_condition_smooth(x: np.ndarray) -> np.ndarray:
    """Smooth periodic profile for convergence tests."""
    return np.sin(2.0 * np.pi * x) + 0.2 * np.cos(4.0 * np.pi * x)


def initial_condition_square(x: np.ndarray) -> np.ndarray:
    """Discontinuous profile for shock-capturing diagnostics."""
    return np.where((x >= 0.2) & (x <= 0.5), 1.0, 0.0)


def exact_solution_periodic(x: np.ndarray, t: float, a: float, u0_func: ArrayFunc) -> np.ndarray:
    """Exact periodic advection solution u(x,t)=u0((x-a*t) mod 1)."""
    x_back = (x - a * t) % 1.0
    return ensure_finite("u_exact", u0_func(x_back).astype(float))


def upwind_step(u: np.ndarray, cfl: float) -> np.ndarray:
    """One conservative first-order upwind step for positive a (0 < cfl <= 1)."""
    if not (0.0 < cfl <= 1.0 + 1e-12):
        raise ValueError(f"Invalid CFL for upwind: {cfl}")

    # Conservative form with flux F_{j+1/2}=a*u_j for a>0.
    interface_flux = u
    return ensure_finite(
        "u_next",
        u - cfl * (interface_flux - np.roll(interface_flux, 1)),
    )


def flux_limiter_step(u: np.ndarray, cfl: float, limiter: str, eps: float = 1e-12) -> np.ndarray:
    """One TVD flux-limiter step for positive advection speed.

    For a > 0:
        F_{j+1/2} = a*u_j + 0.5*a*(1-cfl)*phi(r_j)*(u_j-u_{j-1}),
    where r_j = (u_j-u_{j-1}) / (u_{j+1}-u_j).

    Since update uses cfl=a*dt/dx, factor a is absorbed in cfl-scaled update.
    """
    if not (0.0 < cfl <= 1.0 + 1e-12):
        raise ValueError(f"Invalid CFL for flux limiter: {cfl}")

    du_minus = u - np.roll(u, 1)
    du_plus = np.roll(u, -1) - u

    # Safe ratio r = du_minus / du_plus without triggering divide-by-zero warnings.
    r = np.zeros_like(u)
    valid = np.abs(du_plus) > eps
    np.divide(du_minus, du_plus, out=r, where=valid)
    phi = limiter_phi(r, limiter)

    # Flux-corrected transport:
    # F = F_upwind + phi(r) * (F_LW - F_upwind),
    # where F_LW - F_upwind = 0.5*(1-cfl)*(u_{j+1}-u_j) in normalized form.
    interface_flux = u + 0.5 * (1.0 - cfl) * phi * du_plus
    u_next = u - cfl * (interface_flux - np.roll(interface_flux, 1))
    return ensure_finite("u_next", u_next)


def solve_advection(
    nx: int,
    a: float,
    t_end: float,
    cfl_target: float,
    u0_func: ArrayFunc,
    scheme: str,
    limiter: str = "vanleer",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Solve 1D periodic advection using a selected explicit scheme."""
    if nx < 16:
        raise ValueError("nx must be >= 16")
    if a <= 0.0:
        raise ValueError("This MVP currently supports a > 0 only")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if cfl_target <= 0.0:
        raise ValueError("cfl_target must be positive")

    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    u0 = ensure_finite("u0", u0_func(x).astype(float))

    dx = 1.0 / nx
    dt0 = cfl_target * dx / a
    n_steps = max(1, int(np.ceil(t_end / dt0)))
    dt = t_end / n_steps
    cfl = a * dt / dx

    if cfl > 1.0 + 1e-12:
        raise ValueError(f"Unstable actual CFL={cfl:.6f}; need CFL <= 1")

    u = u0.copy()
    for _ in range(n_steps):
        if scheme == "upwind":
            u = upwind_step(u, cfl)
        elif scheme == "flux_limiter":
            u = flux_limiter_step(u, cfl=cfl, limiter=limiter)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    return x, u0, u, n_steps, cfl


def error_norms(u_num: np.ndarray, u_ref: np.ndarray, dx: float) -> tuple[float, float, float]:
    """Compute L1/L2/Linf of error vector."""
    err = u_num - u_ref
    l1 = float(np.sum(np.abs(err)) * dx)
    l2 = float(np.sqrt(np.sum(err**2) * dx))
    linf = float(np.max(np.abs(err)))
    return l1, l2, linf


def total_variation(u: np.ndarray) -> float:
    """Periodic discrete total variation."""
    return float(np.sum(np.abs(np.roll(u, -1) - u)))


def overshoot_amount(u: np.ndarray, lo: float, hi: float) -> float:
    """Amount violating expected range [lo, hi]."""
    over_hi = max(0.0, float(np.max(u) - hi))
    under_lo = max(0.0, float(lo - np.min(u)))
    return over_hi + under_lo


def run_case(
    nx: int,
    a: float,
    t_end: float,
    cfl_target: float,
    u0_func: ArrayFunc,
    scheme: str,
    limiter: str = "vanleer",
) -> ExperimentResult:
    """Run one simulation case and summarize diagnostics."""
    x, u0, u_num, n_steps, cfl = solve_advection(
        nx=nx,
        a=a,
        t_end=t_end,
        cfl_target=cfl_target,
        u0_func=u0_func,
        scheme=scheme,
        limiter=limiter,
    )

    dx = 1.0 / nx
    u_exact = exact_solution_periodic(x, t=t_end, a=a, u0_func=u0_func)
    l1, l2, linf = error_norms(u_num, u_exact, dx)

    mass_error = float(np.sum(u_num - u0) * dx)
    tv0 = total_variation(u0)
    tvt = total_variation(u_num)

    lo = float(np.min(u0))
    hi = float(np.max(u0))
    osc = overshoot_amount(u_num, lo=lo, hi=hi)

    return ExperimentResult(
        scheme=scheme,
        nx=nx,
        n_steps=n_steps,
        cfl=cfl,
        l1=l1,
        l2=l2,
        linf=linf,
        mass_error=mass_error,
        tv0=tv0,
        tvt=tvt,
        overshoot=osc,
    )


def convergence_order(err_coarse: float, err_fine: float, h_ratio: float) -> float:
    """Estimate p from err ~ h^p."""
    if err_coarse <= 0.0 or err_fine <= 0.0 or h_ratio <= 1.0:
        raise ValueError("Invalid inputs for convergence_order")
    return float(np.log(err_coarse / err_fine) / np.log(h_ratio))


def print_table(results: list[ExperimentResult], title: str) -> None:
    print(f"\n{title}")
    print("scheme       | nx  | n_steps | cfl      | L1         | L2         | Linf       | mass_err")
    print("-------------+-----+---------+----------+------------+------------+------------+---------")
    for r in results:
        print(
            f"{r.scheme:11s} | {r.nx:3d} | {r.n_steps:7d} | {r.cfl:8.5f} |"
            f" {r.l1:10.3e} | {r.l2:10.3e} | {r.linf:10.3e} | {r.mass_error:7.1e}"
        )


def main() -> None:
    a = 1.0
    t_end = 0.4
    cfl_target = 0.85
    limiter = "vanleer"

    print("=== Flux Limiter MVP: 1D Linear Advection ===")
    print(f"a={a}, t_end={t_end}, cfl_target={cfl_target}, limiter={limiter}")

    # 1) Smooth-case convergence study.
    resolutions = [100, 200, 400]
    smooth_results: list[ExperimentResult] = []
    for nx in resolutions:
        smooth_results.append(
            run_case(
                nx=nx,
                a=a,
                t_end=t_end,
                cfl_target=cfl_target,
                u0_func=initial_condition_smooth,
                scheme="upwind",
                limiter=limiter,
            )
        )
        smooth_results.append(
            run_case(
                nx=nx,
                a=a,
                t_end=t_end,
                cfl_target=cfl_target,
                u0_func=initial_condition_smooth,
                scheme="flux_limiter",
                limiter=limiter,
            )
        )

    print_table(smooth_results, "Smooth initial condition")

    upwind_only = [r for r in smooth_results if r.scheme == "upwind"]
    limiter_only = [r for r in smooth_results if r.scheme == "flux_limiter"]

    p_up_1 = convergence_order(upwind_only[0].l2, upwind_only[1].l2, h_ratio=2.0)
    p_up_2 = convergence_order(upwind_only[1].l2, upwind_only[2].l2, h_ratio=2.0)
    p_fl_1 = convergence_order(limiter_only[0].l2, limiter_only[1].l2, h_ratio=2.0)
    p_fl_2 = convergence_order(limiter_only[1].l2, limiter_only[2].l2, h_ratio=2.0)

    print(
        "\nEmpirical order (L2): "
        f"upwind: p(100->200)={p_up_1:.3f}, p(200->400)={p_up_2:.3f}; "
        f"flux_limiter: p(100->200)={p_fl_1:.3f}, p(200->400)={p_fl_2:.3f}"
    )

    # 2) Discontinuous-case diagnostics.
    nx_disc = 400
    up_disc = run_case(
        nx=nx_disc,
        a=a,
        t_end=t_end,
        cfl_target=cfl_target,
        u0_func=initial_condition_square,
        scheme="upwind",
        limiter=limiter,
    )
    fl_disc = run_case(
        nx=nx_disc,
        a=a,
        t_end=t_end,
        cfl_target=cfl_target,
        u0_func=initial_condition_square,
        scheme="flux_limiter",
        limiter=limiter,
    )

    print("\nDiscontinuous initial condition (square wave)")
    print("scheme       | TV0       | TVT       | TV_drop    | overshoot  | mass_err")
    print("-------------+-----------+-----------+------------+------------+---------")
    for r in [up_disc, fl_disc]:
        print(
            f"{r.scheme:11s} | {r.tv0:9.5f} | {r.tvt:9.5f} |"
            f" {r.tv0 - r.tvt:10.5f} | {r.overshoot:10.3e} | {r.mass_error:7.1e}"
        )

    # Lightweight self-checks for regression protection.
    if p_up_1 < 0.70 or p_up_2 < 0.70:
        raise AssertionError("Upwind order is unexpectedly low; check implementation")
    if p_fl_1 < 1.10 or p_fl_2 < 1.10:
        raise AssertionError("Flux limiter order is lower than expected on smooth data")
    if fl_disc.overshoot > 2e-2:
        raise AssertionError("Flux limiter produced too much overshoot on square wave")

    print("\nAll flux-limiter MVP checks passed.")


if __name__ == "__main__":
    main()
