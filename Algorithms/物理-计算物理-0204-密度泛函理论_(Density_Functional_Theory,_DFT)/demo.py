"""Minimal runnable MVP for Density Functional Theory (DFT), PHYS-0203.

This demo implements a pedagogical 1D orbital-free DFT SCF loop with:
- Thomas-Fermi local kinetic energy term
- Weizsaecker gradient kinetic correction
- Soft-Coulomb Hartree interaction
- LDA exchange term

Goal: provide a transparent, source-level algorithm prototype that runs end-to-end
without relying on a high-level electronic-structure black box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal


@dataclass(frozen=True)
class Grid1D:
    x_min: float
    x_max: float
    n_points: int

    @property
    def x(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.n_points)

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.n_points - 1)


@dataclass(frozen=True)
class OFDFTConfig:
    n_electrons: int = 4
    omega: float = 0.30
    soft_coulomb: float = 0.80
    lambda_w: float = 1.50
    mix: float = 0.03
    max_iter: int = 260
    tol_density: float = 8e-5
    tol_energy: float = 1e-6


def normalize_orbital(phi: np.ndarray, dx: float, n_electrons: int) -> np.ndarray:
    norm = float(np.sum(phi**2) * dx)
    if norm <= 0.0:
        raise ValueError("orbital norm must be positive")
    return phi * np.sqrt(n_electrons / norm)


def build_laplacian_tridiagonal(n_points: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Second derivative operator d^2/dx^2 in tridiagonal form."""

    diag = np.full(n_points, -2.0 / dx**2, dtype=float)
    offdiag = np.full(n_points - 1, 1.0 / dx**2, dtype=float)
    return diag, offdiag


def apply_tridiagonal(diag: np.ndarray, offdiag: np.ndarray, vector: np.ndarray) -> np.ndarray:
    out = diag * vector
    out[:-1] += offdiag * vector[1:]
    out[1:] += offdiag * vector[:-1]
    return out


def build_soft_coulomb_kernel(x: np.ndarray, softness: float) -> np.ndarray:
    delta = x[:, None] - x[None, :]
    return 1.0 / np.sqrt(delta**2 + softness**2)


def external_potential_harmonic(x: np.ndarray, omega: float) -> np.ndarray:
    return 0.5 * (omega**2) * x**2


def hartree_potential(kernel: np.ndarray, density: np.ndarray, dx: float) -> np.ndarray:
    return dx * (kernel @ density)


def lda_local_terms(density: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return local OF-DFT ingredients.

    Uses pedagogical 3D-like local exponents/constants in a 1D toy model:
    - T_TF[n] = C_TF * integral n^(5/3)
    - E_x[n]  = C_x  * integral n^(4/3)

    Returns:
    (t_tf_density, e_x_density, v_tf, v_x)
    """

    n_safe = np.clip(density, 1e-14, None)

    c_tf = (3.0 / 10.0) * (3.0 * np.pi**2) ** (2.0 / 3.0)
    c_x = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)

    t_tf_density = c_tf * n_safe ** (5.0 / 3.0)
    e_x_density = c_x * n_safe ** (4.0 / 3.0)

    v_tf = (5.0 / 3.0) * c_tf * n_safe ** (2.0 / 3.0)
    v_x = (4.0 / 3.0) * c_x * n_safe ** (1.0 / 3.0)

    return t_tf_density, e_x_density, v_tf, v_x


def total_energy(
    phi: np.ndarray,
    lap_diag: np.ndarray,
    lap_offdiag: np.ndarray,
    v_ext: np.ndarray,
    kernel: np.ndarray,
    dx: float,
    lambda_w: float,
) -> tuple[float, float, float, float, float, float]:
    """Compute OF-DFT energy decomposition.

    Returns:
    (E_total, T_w, T_tf, E_ext, E_H, E_x)
    """

    density = np.clip(phi**2, 1e-14, None)
    v_h = hartree_potential(kernel, density, dx)
    t_tf_density, e_x_density, _, _ = lda_local_terms(density)

    grad_phi = np.gradient(phi, dx, edge_order=2)
    t_w = 0.5 * lambda_w * float(np.sum(grad_phi**2) * dx)

    t_tf = float(np.sum(t_tf_density) * dx)
    e_ext = float(np.sum(v_ext * density) * dx)
    e_h = 0.5 * float(np.sum(v_h * density) * dx)
    e_x = float(np.sum(e_x_density) * dx)

    e_total = t_w + t_tf + e_ext + e_h + e_x
    return e_total, t_w, t_tf, e_ext, e_h, e_x


def solve_ground_state(
    lap_diag: np.ndarray,
    lap_offdiag: np.ndarray,
    v_eff: np.ndarray,
    lambda_w: float,
) -> tuple[float, np.ndarray]:
    """Solve [-lambda_w/2 * Laplacian + v_eff] phi = mu * phi for the lowest state."""

    h_diag = -0.5 * lambda_w * lap_diag + v_eff
    h_offdiag = -0.5 * lambda_w * lap_offdiag

    evals, evecs = eigh_tridiagonal(
        h_diag,
        h_offdiag,
        select="i",
        select_range=(0, 0),
        check_finite=False,
    )
    return float(evals[0]), evecs[:, 0]


def run_ofdft_scf(grid: Grid1D, cfg: OFDFTConfig) -> dict[str, object]:
    if cfg.n_electrons <= 0:
        raise ValueError("n_electrons must be positive")

    x = grid.x
    dx = grid.dx

    v_ext = external_potential_harmonic(x, cfg.omega)
    kernel = build_soft_coulomb_kernel(x, cfg.soft_coulomb)
    lap_diag, lap_offdiag = build_laplacian_tridiagonal(grid.n_points, dx)

    phi = np.exp(-0.45 * x**2)
    phi = normalize_orbital(phi, dx, cfg.n_electrons)

    history: list[dict[str, float]] = []
    converged = False
    e_prev = np.inf

    for iteration in range(1, cfg.max_iter + 1):
        density = np.clip(phi**2, 1e-14, None)
        v_h = hartree_potential(kernel, density, dx)
        _, _, v_tf, v_x = lda_local_terms(density)

        v_eff = v_ext + v_h + v_tf + v_x
        chemical_potential, phi_out = solve_ground_state(lap_diag, lap_offdiag, v_eff, cfg.lambda_w)

        # Align eigenvector phase with previous iterate to avoid artificial sign flips.
        if float(np.sum(phi_out * phi) * dx) < 0.0:
            phi_out = -phi_out
        phi_out = normalize_orbital(phi_out, dx, cfg.n_electrons)

        density_old = np.clip(phi**2, 1e-14, None)
        density_out = np.clip(phi_out**2, 1e-14, None)
        density_new = (1.0 - cfg.mix) * density_old + cfg.mix * density_out
        density_new = np.clip(density_new, 1e-14, None)

        phi_new = np.sqrt(density_new)
        phi_new = normalize_orbital(phi_new, dx, cfg.n_electrons)
        density_new = np.clip(phi_new**2, 1e-14, None)
        drho = float(np.sqrt(np.sum((density_new - density_old) ** 2) * dx))

        e_total, t_w, t_tf, e_ext, e_h, e_x = total_energy(
            phi_new,
            lap_diag,
            lap_offdiag,
            v_ext,
            kernel,
            dx,
            cfg.lambda_w,
        )

        denergy = float(abs(e_total - e_prev))

        history.append(
            {
                "iter": float(iteration),
                "E_total": e_total,
                "dE": denergy,
                "drho_L2": drho,
                "mu": chemical_potential,
                "T_w": t_w,
                "T_tf": t_tf,
                "E_ext": e_ext,
                "E_H": e_h,
                "E_x": e_x,
                "N_integral": float(np.sum(density_new) * dx),
                "density_min": float(np.min(density_new)),
            }
        )

        if drho < cfg.tol_density and denergy < cfg.tol_energy:
            converged = True
            phi = phi_new
            break

        phi = phi_new
        e_prev = e_total

    if not history:
        raise RuntimeError("SCF failed to produce iteration history")

    return {
        "converged": converged,
        "iterations": len(history),
        "history": pd.DataFrame(history),
        "phi": phi,
        "density": phi**2,
        "x": x,
        "dx": dx,
        "cfg": cfg,
        "final_metrics": history[-1],
    }


def main() -> None:
    grid = Grid1D(x_min=-9.0, x_max=9.0, n_points=280)
    cfg = OFDFTConfig(
        n_electrons=4,
        omega=0.30,
        soft_coulomb=0.80,
        lambda_w=1.50,
        mix=0.03,
        max_iter=260,
        tol_density=8e-5,
        tol_energy=1e-6,
    )

    result = run_ofdft_scf(grid, cfg)
    history = result["history"]
    final = result["final_metrics"]

    n_err = abs(final["N_integral"] - cfg.n_electrons)

    checks = {
        "SCF converged": bool(result["converged"]),
        "density residual < 2e-4": final["drho_L2"] < 2e-4,
        "electron-number error < 1e-6": n_err < 1e-6,
        "Weizsaecker kinetic > 0": final["T_w"] > 0.0,
        "Thomas-Fermi kinetic > 0": final["T_tf"] > 0.0,
        "Hartree energy > 0": final["E_H"] > 0.0,
        "exchange energy < 0": final["E_x"] < 0.0,
        "density non-negative": final["density_min"] >= 0.0,
    }

    pd.set_option("display.float_format", lambda value: f"{value:.8f}")

    print("=== Density Functional Theory MVP | PHYS-0203 ===")
    print("Model: 1D orbital-free DFT (TF + Weizsaecker + Hartree + LDA exchange)")
    print(f"Grid: N={grid.n_points}, x in [{grid.x_min}, {grid.x_max}], dx={grid.dx:.6f}")
    print()

    print("Recent SCF iterations (tail 8):")
    print(
        history[["iter", "E_total", "dE", "drho_L2", "mu", "N_integral"]]
        .tail(8)
        .to_string(index=False)
    )

    summary = pd.DataFrame(
        [
            {
                "iterations": result["iterations"],
                "converged": bool(result["converged"]),
                "E_total": final["E_total"],
                "T_w": final["T_w"],
                "T_tf": final["T_tf"],
                "E_ext": final["E_ext"],
                "E_H": final["E_H"],
                "E_x": final["E_x"],
                "mu": final["mu"],
                "drho_L2": final["drho_L2"],
                "N_error": n_err,
            }
        ]
    )

    print()
    print("Final summary:")
    print(summary.to_string(index=False))

    print()
    print("Checks:")
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"- {name}: {status}")

    all_passed = all(checks.values())
    print()
    print(f"Validation: {'PASS' if all_passed else 'FAIL'}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
