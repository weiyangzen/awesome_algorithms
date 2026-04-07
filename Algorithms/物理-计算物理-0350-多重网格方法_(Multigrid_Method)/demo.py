"""Minimal runnable MVP for Multigrid Method (PHYS-0339).

Model problem:
    -u''(x) = f(x), x in (0, 1)
    u(0) = u(1) = 0
with manufactured exact solution u(x) = sin(pi x),
so f(x) = pi^2 sin(pi x).

We solve the finite-difference linear system on a 1D hierarchy using
an explicit V-cycle multigrid implementation:
- weighted Jacobi smoothing
- full-weighting restriction
- linear interpolation prolongation
- direct solve on the coarsest level

The script prints convergence history and validation status.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class MGConfig:
    """Configuration for the 1D V-cycle multigrid solver."""

    n_finest: int = 255
    pre_smooth: int = 3
    post_smooth: int = 3
    omega: float = 2.0 / 3.0
    max_cycles: int = 12
    residual_tol: float = 1e-10

    def validate(self) -> None:
        if self.n_finest < 3:
            raise ValueError("n_finest must be >= 3")
        # Require n = 2^k - 1 so each coarsening step n_c = (n_f - 1)/2 is integer.
        value = self.n_finest + 1
        if value & (value - 1):
            raise ValueError("n_finest must satisfy n_finest = 2^k - 1")
        if self.pre_smooth < 0 or self.post_smooth < 0:
            raise ValueError("smoothing steps must be non-negative")
        if not (0.0 < self.omega < 1.5):
            raise ValueError("omega should be in (0, 1.5) for this Jacobi setup")
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")
        if self.residual_tol <= 0.0:
            raise ValueError("residual_tol must be positive")


def build_levels(n_finest: int) -> list[tuple[int, float]]:
    """Build grid hierarchy [(n_level, h_level), ...] from fine to coarse."""
    levels: list[tuple[int, float]] = []
    n = n_finest
    while True:
        h = 1.0 / (n + 1)
        levels.append((n, h))
        if n <= 3:
            break
        if (n - 1) % 2 != 0:
            raise ValueError("Cannot coarsen: n does not satisfy n = 2*n_c + 1")
        n = (n - 1) // 2
    return levels


def apply_operator(u: np.ndarray, h: float) -> np.ndarray:
    """Apply 1D Poisson FD operator A u = (2u_i - u_{i-1} - u_{i+1}) / h^2."""
    left = np.empty_like(u)
    right = np.empty_like(u)
    left[0] = 0.0
    left[1:] = u[:-1]
    right[-1] = 0.0
    right[:-1] = u[1:]
    return (2.0 * u - left - right) / (h * h)


def weighted_jacobi(
    u: np.ndarray,
    b: np.ndarray,
    h: float,
    omega: float,
    steps: int,
) -> np.ndarray:
    """Perform weighted Jacobi smoothing for A u = b."""
    out = u.copy()
    hh = h * h
    for _ in range(steps):
        left = np.empty_like(out)
        right = np.empty_like(out)
        left[0] = 0.0
        left[1:] = out[:-1]
        right[-1] = 0.0
        right[:-1] = out[1:]
        jacobi_target = 0.5 * (left + right + hh * b)
        out = (1.0 - omega) * out + omega * jacobi_target
    return out


def restrict_full_weighting(r_fine: np.ndarray) -> np.ndarray:
    """Restrict fine-grid residual to coarse grid using full weighting."""
    n_f = r_fine.size
    n_c = (n_f - 1) // 2
    return (
        0.25 * r_fine[0 : 2 * n_c : 2]
        + 0.5 * r_fine[1 : 2 * n_c + 1 : 2]
        + 0.25 * r_fine[2 : 2 * n_c + 2 : 2]
    )


def prolong_linear(e_coarse: np.ndarray) -> np.ndarray:
    """Prolong coarse-grid correction to fine grid with linear interpolation."""
    n_c = e_coarse.size
    n_f = 2 * n_c + 1
    e_f = np.zeros(n_f, dtype=np.float64)

    # Injection at coincident points.
    e_f[1 : 2 * n_c + 1 : 2] = e_coarse

    # Midpoints between adjacent coarse points.
    if n_c > 1:
        e_f[2 : 2 * n_c : 2] = 0.5 * (e_coarse[:-1] + e_coarse[1:])

    # Midpoints adjacent to boundaries (boundary values are zero).
    e_f[0] = 0.5 * e_coarse[0]
    e_f[-1] = 0.5 * e_coarse[-1]
    return e_f


def direct_solve_coarsest(b: np.ndarray, h: float) -> np.ndarray:
    """Direct solve on the coarsest level."""
    n = b.size
    diag = np.full(n, 2.0 / (h * h), dtype=np.float64)
    off = np.full(n - 1, -1.0 / (h * h), dtype=np.float64)
    mat = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return np.linalg.solve(mat, b)


def v_cycle(
    level_idx: int,
    u: np.ndarray,
    b: np.ndarray,
    levels: list[tuple[int, float]],
    cfg: MGConfig,
) -> np.ndarray:
    """Recursive 1D V-cycle multigrid."""
    _, h = levels[level_idx]

    if level_idx == len(levels) - 1:
        return direct_solve_coarsest(b=b, h=h)

    # Pre-smoothing.
    u = weighted_jacobi(u=u, b=b, h=h, omega=cfg.omega, steps=cfg.pre_smooth)

    # Residual equation on coarse grid.
    residual = b - apply_operator(u=u, h=h)
    b_coarse = restrict_full_weighting(residual)

    # Coarse-grid correction.
    e_coarse0 = np.zeros_like(b_coarse)
    e_coarse = v_cycle(
        level_idx=level_idx + 1,
        u=e_coarse0,
        b=b_coarse,
        levels=levels,
        cfg=cfg,
    )
    u += prolong_linear(e_coarse)

    # Post-smoothing.
    u = weighted_jacobi(u=u, b=b, h=h, omega=cfg.omega, steps=cfg.post_smooth)
    return u


def l2_norm(values: np.ndarray, h: float) -> float:
    """Discrete L2 norm approximating integral norm."""
    return float(np.sqrt(h * np.dot(values, values)))


def scipy_reference_solution(b: np.ndarray, h: float) -> np.ndarray:
    """Reference discrete solution from SciPy sparse direct solver."""
    n = b.size
    main = np.full(n, 2.0 / (h * h), dtype=np.float64)
    off = np.full(n - 1, -1.0 / (h * h), dtype=np.float64)
    a = diags(diagonals=[off, main, off], offsets=[-1, 0, 1], format="csc")
    return spsolve(a, b)


def main() -> None:
    cfg = MGConfig()
    cfg.validate()

    levels = build_levels(cfg.n_finest)
    n0, h0 = levels[0]

    x = np.linspace(h0, 1.0 - h0, n0, dtype=np.float64)
    u_exact = np.sin(np.pi * x)
    rhs = (np.pi**2) * np.sin(np.pi * x)

    u = np.zeros_like(rhs)
    residual0 = rhs - apply_operator(u=u, h=h0)
    res0_norm = l2_norm(residual0, h0)

    rows: list[dict[str, float]] = []
    rows.append(
        {
            "cycle": 0.0,
            "res_l2": res0_norm,
            "res_reduction": 1.0,
            "err_l2": l2_norm(u - u_exact, h0),
            "max_abs_err": float(np.max(np.abs(u - u_exact))),
        }
    )

    for cycle in range(1, cfg.max_cycles + 1):
        u = v_cycle(level_idx=0, u=u, b=rhs, levels=levels, cfg=cfg)
        residual = rhs - apply_operator(u=u, h=h0)
        res_norm = l2_norm(residual, h0)
        err = u - u_exact
        rows.append(
            {
                "cycle": float(cycle),
                "res_l2": res_norm,
                "res_reduction": res_norm / res0_norm,
                "err_l2": l2_norm(err, h0),
                "max_abs_err": float(np.max(np.abs(err))),
            }
        )
        if res_norm < cfg.residual_tol:
            break

    df = pd.DataFrame(rows)

    u_ref = scipy_reference_solution(rhs, h0)
    ref_gap_inf = float(np.max(np.abs(u - u_ref)))
    final_res = float(df["res_l2"].iloc[-1])
    final_reduction = float(df["res_reduction"].iloc[-1])
    final_err_l2 = float(df["err_l2"].iloc[-1])
    final_max_err = float(df["max_abs_err"].iloc[-1])

    print("=== Multigrid Method MVP (1D Poisson) ===")
    print(
        f"n_finest={cfg.n_finest}, h_finest={h0:.6e}, "
        f"levels={[n for n, _ in levels]}, pre/post=({cfg.pre_smooth},{cfg.post_smooth}), "
        f"omega={cfg.omega:.3f}, max_cycles={cfg.max_cycles}"
    )
    print(df.to_string(index=False, float_format=lambda v: f"{v:.8e}"))
    print(
        "Final metrics: "
        f"res_l2={final_res:.3e}, reduction={final_reduction:.3e}, "
        f"err_l2={final_err_l2:.3e}, max_abs_err={final_max_err:.3e}, "
        f"|u_mg-u_scipy|_inf={ref_gap_inf:.3e}"
    )

    # Validation gates for this MVP configuration.
    assert final_reduction < 1e-8, f"Residual reduction too weak: {final_reduction:.3e}"
    assert final_res < 1e-7, f"Residual L2 too large: {final_res:.3e}"
    assert final_err_l2 < 2e-5, f"L2 error too large: {final_err_l2:.3e}"
    assert final_max_err < 4e-5, f"Max abs error too large: {final_max_err:.3e}"
    assert ref_gap_inf < 1e-10, f"MG and SciPy discrete solutions differ too much: {ref_gap_inf:.3e}"

    print("Validation: PASS")


if __name__ == "__main__":
    main()
