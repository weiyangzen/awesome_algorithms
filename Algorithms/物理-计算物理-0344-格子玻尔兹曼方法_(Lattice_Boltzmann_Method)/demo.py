"""Minimal runnable MVP for Lattice Boltzmann Method (PHYS-0337).

This script implements a D2Q9 BGK LBM solver with Guo forcing for a
2D channel Poiseuille flow:
- periodic boundary in x-direction
- bounce-back walls at y=0 and y=H
- constant body force gx driving the flow

The numerical steady profile is compared against the analytic parabolic profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LBMConfig:
    nx: int = 64
    ny: int = 24
    tau: float = 0.80
    force_x: float = 3.0e-6
    force_y: float = 0.0
    steps: int = 5000
    report_every: int = 500

    def validate(self) -> None:
        if self.nx < 8 or self.ny < 8:
            raise ValueError("nx and ny must both be >= 8")
        if self.tau <= 0.5:
            raise ValueError("tau must be > 0.5 for positive viscosity")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.report_every <= 0:
            raise ValueError("report_every must be positive")


def d2q9_lattice() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return D2Q9 discrete velocities, weights, and opposite-index map."""
    cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int8)
    cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int8)
    w = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=np.float64)
    opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int8)
    return cx, cy, w, opp


def macroscopic(
    f: np.ndarray,
    force_x: float,
    force_y: float,
    cx: np.ndarray,
    cy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rho, ux, uy from populations with half-force velocity correction."""
    rho = np.sum(f, axis=0)
    mom_x = np.sum(f * cx[:, None, None], axis=0)
    mom_y = np.sum(f * cy[:, None, None], axis=0)

    ux = (mom_x + 0.5 * force_x) / rho
    uy = (mom_y + 0.5 * force_y) / rho
    return rho, ux, uy


def equilibrium(
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Build D2Q9 equilibrium distribution."""
    cu = cx[:, None, None] * ux[None, :, :] + cy[:, None, None] * uy[None, :, :]
    u2 = ux**2 + uy**2
    feq = w[:, None, None] * rho[None, :, :] * (
        1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2[None, :, :]
    )
    return feq


def guo_forcing_term(
    ux: np.ndarray,
    uy: np.ndarray,
    force_x: float,
    force_y: float,
    tau: float,
    cx: np.ndarray,
    cy: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Guo forcing source term for BGK collision."""
    cs2 = 1.0 / 3.0
    cs4 = cs2 * cs2

    ci_dot_u = cx[:, None, None] * ux[None, :, :] + cy[:, None, None] * uy[None, :, :]

    # (c_i - u)/cs2 term
    term_x = (cx[:, None, None] - ux[None, :, :]) / cs2
    term_y = (cy[:, None, None] - uy[None, :, :]) / cs2

    # ((c_i · u)/cs4) * c_i term
    term_x += (ci_dot_u * cx[:, None, None]) / cs4
    term_y += (ci_dot_u * cy[:, None, None]) / cs4

    pre = (1.0 - 0.5 / tau) * w[:, None, None]
    source = pre * (term_x * force_x + term_y * force_y)
    return source


def stream(f_post: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Streaming via periodic rolls on each discrete direction."""
    f_stream = np.empty_like(f_post)
    for i in range(9):
        f_stream[i] = np.roll(f_post[i], shift=(cy[i], cx[i]), axis=(0, 1))
    return f_stream


def apply_bounce_back_walls(f: np.ndarray) -> None:
    """On-node bounce-back at bottom (y=0) and top (y=ny-1) walls."""
    # Bottom wall
    f[2, 0, :] = f[4, 0, :]
    f[5, 0, :] = f[7, 0, :]
    f[6, 0, :] = f[8, 0, :]

    # Top wall
    f[4, -1, :] = f[2, -1, :]
    f[7, -1, :] = f[5, -1, :]
    f[8, -1, :] = f[6, -1, :]


def run_lbm(config: LBMConfig) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Run the D2Q9 BGK simulation and return fields plus convergence history."""
    config.validate()
    cx, cy, w, _ = d2q9_lattice()

    rho = np.ones((config.ny, config.nx), dtype=np.float64)
    ux = np.zeros((config.ny, config.nx), dtype=np.float64)
    uy = np.zeros((config.ny, config.nx), dtype=np.float64)
    f = equilibrium(rho, ux, uy, cx, cy, w)

    omega = 1.0 / config.tau
    history: list[dict[str, float | int]] = []
    prev_profile = np.zeros(config.ny, dtype=np.float64)

    for step in range(1, config.steps + 1):
        rho, ux, uy = macroscopic(f, config.force_x, config.force_y, cx, cy)

        # Enforce no-slip wall velocity in collision equilibrium.
        ux[0, :] = 0.0
        ux[-1, :] = 0.0
        uy[0, :] = 0.0
        uy[-1, :] = 0.0

        feq = equilibrium(rho, ux, uy, cx, cy, w)
        force_term = guo_forcing_term(
            ux=ux,
            uy=uy,
            force_x=config.force_x,
            force_y=config.force_y,
            tau=config.tau,
            cx=cx,
            cy=cy,
            w=w,
        )

        f_post = f - omega * (f - feq) + force_term
        f = stream(f_post, cx, cy)
        apply_bounce_back_walls(f)

        if step % config.report_every == 0 or step == config.steps:
            rho_m, ux_m, _ = macroscopic(f, config.force_x, config.force_y, cx, cy)
            mean_profile = np.mean(ux_m, axis=1)
            residual = float(np.max(np.abs(mean_profile - prev_profile)))
            prev_profile = mean_profile
            history.append(
                {
                    "step": step,
                    "mean_u": float(np.mean(mean_profile[1:-1])),
                    "u_max": float(np.max(mean_profile[1:-1])),
                    "rho_min": float(np.min(rho_m)),
                    "rho_max": float(np.max(rho_m)),
                    "profile_residual": residual,
                }
            )

    rho, ux, _ = macroscopic(f, config.force_x, config.force_y, cx, cy)
    return rho, ux, pd.DataFrame(history)


def analytic_poiseuille_profile(ny: int, force_x: float, tau: float) -> np.ndarray:
    """Analytic channel Poiseuille profile for walls at y=0 and y=H."""
    nu = (tau - 0.5) / 3.0
    y = np.arange(ny, dtype=np.float64)
    h = float(ny - 1)
    profile = (force_x / (2.0 * nu)) * y * (h - y)
    return profile


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 norm with epsilon guard."""
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b) + 1.0e-15)
    return num / den


def main() -> None:
    config = LBMConfig()
    rho, ux, history = run_lbm(config)

    ux_mean_y = np.mean(ux, axis=1)
    ux_ref = analytic_poiseuille_profile(config.ny, config.force_x, config.tau)

    interior = slice(1, config.ny - 1)
    rel_err_l2 = relative_l2(ux_mean_y[interior], ux_ref[interior])
    u_max_num = float(np.max(ux_mean_y[interior]))
    u_max_ref = float(np.max(ux_ref[interior]))
    max_mach = np.sqrt(3.0) * u_max_num

    mass_drift = float(np.max(np.abs(rho - 1.0)))
    final_residual = float(history.iloc[-1]["profile_residual"])

    summary = pd.DataFrame(
        {
            "metric": [
                "grid (nx, ny)",
                "steps",
                "tau",
                "kinematic_nu",
                "force_x",
                "u_max_numeric",
                "u_max_analytic",
                "relative_L2_error_profile",
                "max_mach",
                "mass_drift_inf",
                "final_profile_residual",
            ],
            "value": [
                f"({config.nx}, {config.ny})",
                config.steps,
                config.tau,
                (config.tau - 0.5) / 3.0,
                config.force_x,
                u_max_num,
                u_max_ref,
                rel_err_l2,
                max_mach,
                mass_drift,
                final_residual,
            ],
        }
    )

    print("LBM D2Q9 Poiseuille MVP")
    print(history.to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))

    assert rel_err_l2 < 0.12, f"velocity profile L2 error too large: {rel_err_l2:.3e}"
    assert max_mach < 0.1, f"Mach number too large for weakly compressible LBM: {max_mach:.3e}"
    assert mass_drift < 8.0e-3, f"density drift too large: {mass_drift:.3e}"
    assert final_residual < 2.0e-4, f"flow not converged enough: {final_residual:.3e}"

    print("Validation: PASS")


if __name__ == "__main__":
    main()
