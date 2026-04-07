"""Minimal runnable MVP for Kohn-Sham Equations (PHYS-0204).

This script implements a 1D real-space self-consistent-field (SCF) Kohn-Sham solver:
- External potential: harmonic trap
- Electron-electron interaction: soft-Coulomb Hartree term
- Exchange-correlation: exchange-only LDA (toy choice for closure)

The goal is algorithm transparency, not production-grade materials accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal


@dataclass(frozen=True)
class Grid1D:
    """Uniform 1D grid."""

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
class KSConfig:
    """Configuration for a toy Kohn-Sham SCF run."""

    n_electrons: int = 4
    omega: float = 0.32
    soft_coulomb: float = 0.75
    xc_scale: float = 1.0
    mix: float = 0.30
    max_iter: int = 150
    tol_density: float = 5e-5
    tol_energy: float = 8e-7


def require_closed_shell(n_electrons: int) -> None:
    if n_electrons <= 0:
        raise ValueError("n_electrons must be positive")
    if n_electrons % 2 != 0:
        raise ValueError("This MVP assumes spin-unpolarized closed shell (even n_electrons)")


def build_kinetic_tridiagonal(n_points: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference kinetic operator T = -0.5 * d^2/dx^2 (tridiagonal form)."""

    diag = np.full(n_points, 1.0 / dx**2, dtype=float)
    offdiag = np.full(n_points - 1, -0.5 / dx**2, dtype=float)
    return diag, offdiag


def apply_tridiagonal(diag: np.ndarray, offdiag: np.ndarray, vector: np.ndarray) -> np.ndarray:
    out = diag * vector
    out[:-1] += offdiag * vector[1:]
    out[1:] += offdiag * vector[:-1]
    return out


def normalize_columns(vectors: np.ndarray, dx: float) -> np.ndarray:
    normalized = np.array(vectors, dtype=float, copy=True)
    norms = np.sqrt(np.sum(normalized**2, axis=0) * dx)
    if np.any(norms <= 0.0):
        raise ValueError("Orbital normalization encountered non-positive norm")
    normalized /= norms
    return normalized


def normalize_density(density: np.ndarray, dx: float, target_electrons: float) -> np.ndarray:
    integral = float(np.sum(density) * dx)
    if integral <= 0.0:
        raise ValueError("Density integral must be positive")
    return density * (target_electrons / integral)


def density_from_orbitals(orbitals: np.ndarray, n_electrons: int, dx: float) -> np.ndarray:
    n_occ = n_electrons // 2
    density = 2.0 * np.sum(orbitals[:, :n_occ] ** 2, axis=1)
    return normalize_density(density, dx, float(n_electrons))


def external_potential_harmonic(x: np.ndarray, omega: float) -> np.ndarray:
    return 0.5 * (omega**2) * x**2


def build_soft_coulomb_kernel(x: np.ndarray, softness: float) -> np.ndarray:
    delta = x[:, None] - x[None, :]
    return 1.0 / np.sqrt(delta**2 + softness**2)


def hartree_potential(kernel: np.ndarray, density: np.ndarray, dx: float) -> np.ndarray:
    return dx * (kernel @ density)


def lda_exchange_energy_density_and_potential(
    density: np.ndarray,
    xc_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exchange energy density and exchange potential.

    Uses 3D homogeneous electron gas exchange formula:
    e_x_density = Cx * n^(4/3), v_x = dE_x/dn = -(3/pi)^(1/3) * n^(1/3).
    """

    n_safe = np.clip(density, 1e-14, None)
    c_x = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    pref = (3.0 / np.pi) ** (1.0 / 3.0)

    e_x_density = xc_scale * c_x * n_safe ** (4.0 / 3.0)
    v_x = -xc_scale * pref * n_safe ** (1.0 / 3.0)
    return e_x_density, v_x


def solve_kohn_sham(
    kinetic_diag: np.ndarray,
    kinetic_offdiag: np.ndarray,
    v_eff: np.ndarray,
    n_occ: int,
) -> tuple[np.ndarray, np.ndarray]:
    h_diag = kinetic_diag + v_eff
    evals, evecs = eigh_tridiagonal(
        h_diag,
        kinetic_offdiag,
        select="i",
        select_range=(0, n_occ - 1),
        check_finite=False,
    )
    return evals, evecs


def kinetic_energy(
    orbitals: np.ndarray,
    kinetic_diag: np.ndarray,
    kinetic_offdiag: np.ndarray,
    dx: float,
    n_electrons: int,
) -> float:
    n_occ = n_electrons // 2
    total = 0.0
    for i in range(n_occ):
        psi = orbitals[:, i]
        t_psi = apply_tridiagonal(kinetic_diag, kinetic_offdiag, psi)
        total += 2.0 * float(np.sum(psi * t_psi) * dx)
    return total


def total_energy(
    density: np.ndarray,
    orbitals: np.ndarray,
    v_ext: np.ndarray,
    v_h: np.ndarray,
    e_x_density: np.ndarray,
    kinetic_diag: np.ndarray,
    kinetic_offdiag: np.ndarray,
    dx: float,
    n_electrons: int,
) -> tuple[float, float, float, float, float]:
    e_kin = kinetic_energy(orbitals, kinetic_diag, kinetic_offdiag, dx, n_electrons)
    e_ext = float(np.sum(v_ext * density) * dx)
    e_h = 0.5 * float(np.sum(v_h * density) * dx)
    e_x = float(np.sum(e_x_density) * dx)
    e_total = e_kin + e_ext + e_h + e_x
    return e_total, e_kin, e_ext, e_h, e_x


def run_scf_kohn_sham(grid: Grid1D, cfg: KSConfig) -> dict[str, object]:
    require_closed_shell(cfg.n_electrons)

    x = grid.x
    dx = grid.dx
    n_occ = cfg.n_electrons // 2

    v_ext = external_potential_harmonic(x, cfg.omega)
    kernel = build_soft_coulomb_kernel(x, cfg.soft_coulomb)
    kinetic_diag, kinetic_offdiag = build_kinetic_tridiagonal(grid.n_points, dx)

    density = np.exp(-0.7 * x**2)
    density = normalize_density(density, dx, float(cfg.n_electrons))

    history: list[dict[str, float]] = []
    converged = False
    e_prev = np.inf

    for iteration in range(1, cfg.max_iter + 1):
        v_h = hartree_potential(kernel, density, dx)
        _, v_x = lda_exchange_energy_density_and_potential(density, cfg.xc_scale)
        v_eff = v_ext + v_h + v_x

        evals, evecs = solve_kohn_sham(kinetic_diag, kinetic_offdiag, v_eff, n_occ)
        orbitals = normalize_columns(evecs, dx)
        density_out = density_from_orbitals(orbitals, cfg.n_electrons, dx)

        density_new = (1.0 - cfg.mix) * density + cfg.mix * density_out
        density_new = normalize_density(density_new, dx, float(cfg.n_electrons))

        v_h_new = hartree_potential(kernel, density_new, dx)
        e_x_density_new, v_x_new = lda_exchange_energy_density_and_potential(density_new, cfg.xc_scale)

        e_total, e_kin, e_ext, e_h, e_x = total_energy(
            density_new,
            orbitals,
            v_ext,
            v_h_new,
            e_x_density_new,
            kinetic_diag,
            kinetic_offdiag,
            dx,
            cfg.n_electrons,
        )

        sum_occ_eigs = 2.0 * float(np.sum(evals))
        double_count = float(np.sum((v_h_new + v_x_new) * density_new) * dx)
        e_ks_identity = sum_occ_eigs - double_count + e_h + e_x

        drho = float(np.sqrt(np.sum((density_new - density) ** 2) * dx))
        denergy = float(abs(e_total - e_prev))

        history.append(
            {
                "iter": float(iteration),
                "E_total": e_total,
                "dE": denergy,
                "drho_L2": drho,
                "eps_min": float(evals[0]),
                "eps_max_occ": float(evals[n_occ - 1]),
                "E_kin": e_kin,
                "E_ext": e_ext,
                "E_H": e_h,
                "E_x": e_x,
                "E_sum_eigs": sum_occ_eigs,
                "E_ks_identity": e_ks_identity,
                "N_integral": float(np.sum(density_new) * dx),
            }
        )

        if drho < cfg.tol_density and denergy < cfg.tol_energy:
            converged = True
            density = density_new
            break

        density = density_new
        e_prev = e_total

    if not history:
        raise RuntimeError("SCF did not generate iteration records")

    return {
        "converged": converged,
        "iterations": len(history),
        "history": pd.DataFrame(history),
        "density": density,
        "x": x,
        "dx": dx,
        "cfg": cfg,
        "final_metrics": history[-1],
    }


def main() -> None:
    grid = Grid1D(x_min=-8.0, x_max=8.0, n_points=260)
    cfg = KSConfig(
        n_electrons=4,
        omega=0.32,
        soft_coulomb=0.75,
        xc_scale=1.0,
        mix=0.30,
        max_iter=150,
        tol_density=5e-5,
        tol_energy=8e-7,
    )

    result = run_scf_kohn_sham(grid, cfg)
    history = result["history"]
    final = result["final_metrics"]

    n_err = abs(final["N_integral"] - cfg.n_electrons)
    ks_identity_gap = abs(final["E_total"] - final["E_ks_identity"])

    checks = {
        "SCF converged": bool(result["converged"]),
        "density residual < 1e-4": final["drho_L2"] < 1e-4,
        "electron-number error < 1e-6": n_err < 1e-6,
        "Hartree energy positive": final["E_H"] > 0.0,
        "exchange energy negative": final["E_x"] < 0.0,
        "KS energy identity gap < 2e-2": ks_identity_gap < 2e-2,
    }

    pd.set_option("display.float_format", lambda value: f"{value:.8f}")

    print("=== Kohn-Sham Equations MVP | PHYS-0204 ===")
    print("Model: 1D real-space SCF with harmonic v_ext + soft-Coulomb Hartree + LDA exchange")
    print(f"Grid: N={grid.n_points}, x in [{grid.x_min}, {grid.x_max}], dx={grid.dx:.6f}")
    print()
    print("Recent SCF iterations (tail 8):")
    print(
        history[["iter", "E_total", "dE", "drho_L2", "eps_min", "eps_max_occ", "N_integral"]]
        .tail(8)
        .to_string(index=False)
    )

    summary = pd.DataFrame(
        [
            {
                "iterations": result["iterations"],
                "converged": bool(result["converged"]),
                "E_total": final["E_total"],
                "E_kin": final["E_kin"],
                "E_ext": final["E_ext"],
                "E_H": final["E_H"],
                "E_x": final["E_x"],
                "E_sum_eigs": final["E_sum_eigs"],
                "E_ks_identity": final["E_ks_identity"],
                "KS_identity_gap": ks_identity_gap,
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
