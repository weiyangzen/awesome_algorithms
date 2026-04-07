"""Minimal runnable MVP for Exchange-Correlation Energy (PHYS-0205).

This demo implements a transparent toy Kohn-Sham SCF loop with:
- LDA exchange (Dirac exchange)
- LDA correlation (Perdew-Zunger 1981 parameterization, unpolarized)

The purpose is educational algorithm tracing, not production-grade materials prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal


@dataclass(frozen=True)
class Grid1D:
    """Uniform real-space 1D grid."""

    x_min: float
    x_max: float
    n_points: int

    @property
    def x(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.n_points, dtype=float)

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.n_points - 1)


@dataclass(frozen=True)
class XCConfig:
    """Configuration for the toy LDA-XC SCF."""

    n_electrons: int = 4
    omega: float = 0.33
    soft_coulomb: float = 0.85
    mix: float = 0.26
    max_iter: int = 140
    tol_density: float = 6.0e-5
    tol_energy: float = 1.0e-6


def require_even_electrons(n_electrons: int) -> None:
    if n_electrons <= 0:
        raise ValueError("n_electrons must be positive")
    if n_electrons % 2 != 0:
        raise ValueError("This MVP assumes spin-unpolarized closed shell, so n_electrons must be even")


def build_kinetic_tridiagonal(n_points: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Return diagonal/off-diagonal of T = -0.5 d^2/dx^2 with central differences."""
    diag = np.full(n_points, 1.0 / dx**2, dtype=float)
    offdiag = np.full(n_points - 1, -0.5 / dx**2, dtype=float)
    return diag, offdiag


def apply_tridiagonal(diag: np.ndarray, offdiag: np.ndarray, vec: np.ndarray) -> np.ndarray:
    out = diag * vec
    out[:-1] += offdiag * vec[1:]
    out[1:] += offdiag * vec[:-1]
    return out


def normalize_orbitals(orbitals: np.ndarray, dx: float) -> np.ndarray:
    """Normalize each orbital with continuous normalization int |psi|^2 dx = 1."""
    psi = np.array(orbitals, dtype=float, copy=True)
    norms = np.sqrt(np.sum(psi**2, axis=0) * dx)
    psi /= norms
    return psi


def normalize_density(density: np.ndarray, dx: float, n_electrons: float) -> np.ndarray:
    integral = float(np.sum(density) * dx)
    if integral <= 0.0:
        raise ValueError("Density integral must stay positive")
    return density * (n_electrons / integral)


def density_from_orbitals(orbitals: np.ndarray, dx: float, n_electrons: int) -> np.ndarray:
    """Construct spin-unpolarized density n(x)=2*sum_occ |psi_i|^2."""
    n_occ = n_electrons // 2
    psi_occ = orbitals[:, :n_occ]
    density = 2.0 * np.sum(psi_occ**2, axis=1)
    return normalize_density(density, dx, float(n_electrons))


def external_potential_harmonic(x: np.ndarray, omega: float) -> np.ndarray:
    return 0.5 * omega**2 * x**2


def build_hartree_kernel(x: np.ndarray, soft_coulomb: float) -> np.ndarray:
    diff = x[:, None] - x[None, :]
    return 1.0 / np.sqrt(diff**2 + soft_coulomb**2)


def hartree_potential(kernel: np.ndarray, density: np.ndarray, dx: float) -> np.ndarray:
    return dx * (kernel @ density)


def density_to_rs(density: np.ndarray) -> np.ndarray:
    """Convert density n to Wigner-Seitz radius rs=(3/(4*pi*n))^(1/3)."""
    n_safe = np.clip(density, 1.0e-14, None)
    return (3.0 / (4.0 * np.pi * n_safe)) ** (1.0 / 3.0)


def lda_exchange_per_particle(density: np.ndarray) -> np.ndarray:
    """Dirac exchange per particle for unpolarized electron gas."""
    n_safe = np.clip(density, 1.0e-14, None)
    c_x = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    return -c_x * np.cbrt(n_safe)


def lda_exchange_energy_density(density: np.ndarray) -> np.ndarray:
    return np.asarray(density, dtype=float) * lda_exchange_per_particle(density)


def lda_exchange_potential(density: np.ndarray) -> np.ndarray:
    """v_x = -(3/pi)^(1/3) * n^(1/3)."""
    pref = (3.0 / np.pi) ** (1.0 / 3.0)
    n_safe = np.clip(density, 1.0e-14, None)
    return -pref * np.cbrt(n_safe)


def pz81_correlation_per_particle_from_rs(rs: np.ndarray) -> np.ndarray:
    """Perdew-Zunger 1981 correlation epsilon_c(rs), spin-unpolarized."""
    rs_safe = np.clip(rs, 1.0e-14, None)
    eps = np.empty_like(rs_safe)

    mask_low = rs_safe < 1.0
    mask_high = ~mask_low

    # rs < 1: epsilon_c = A ln(rs) + B + C rs ln(rs) + D rs
    a, b, c, d = 0.0311, -0.048, 0.0020, -0.0116
    rs_l = rs_safe[mask_low]
    eps[mask_low] = a * np.log(rs_l) + b + c * rs_l * np.log(rs_l) + d * rs_l

    # rs >= 1: epsilon_c = gamma / (1 + beta1 sqrt(rs) + beta2 rs)
    gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
    rs_h = rs_safe[mask_high]
    denom = 1.0 + beta1 * np.sqrt(rs_h) + beta2 * rs_h
    eps[mask_high] = gamma / denom

    return eps


def pz81_correlation_derivative_from_rs(rs: np.ndarray) -> np.ndarray:
    """d epsilon_c / d rs for PZ81, spin-unpolarized."""
    rs_safe = np.clip(rs, 1.0e-14, None)
    deps = np.empty_like(rs_safe)

    mask_low = rs_safe < 1.0
    mask_high = ~mask_low

    a, _b, c, d = 0.0311, -0.048, 0.0020, -0.0116
    rs_l = rs_safe[mask_low]
    deps[mask_low] = a / rs_l + c * (np.log(rs_l) + 1.0) + d

    gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
    rs_h = rs_safe[mask_high]
    denom = 1.0 + beta1 * np.sqrt(rs_h) + beta2 * rs_h
    ddenom_drs = beta1 / (2.0 * np.sqrt(rs_h)) + beta2
    deps[mask_high] = -gamma * ddenom_drs / (denom**2)

    return deps


def lda_correlation_per_particle(density: np.ndarray) -> np.ndarray:
    rs = density_to_rs(density)
    return pz81_correlation_per_particle_from_rs(rs)


def lda_correlation_energy_density(density: np.ndarray) -> np.ndarray:
    return np.asarray(density, dtype=float) * lda_correlation_per_particle(density)


def lda_correlation_potential(density: np.ndarray) -> np.ndarray:
    """v_c = epsilon_c - (rs/3) * d epsilon_c / d rs."""
    rs = density_to_rs(density)
    eps_c = pz81_correlation_per_particle_from_rs(rs)
    deps_drs = pz81_correlation_derivative_from_rs(rs)
    return eps_c - (rs / 3.0) * deps_drs


def solve_kohn_sham_orbitals(
    kinetic_diag: np.ndarray,
    kinetic_offdiag: np.ndarray,
    v_eff: np.ndarray,
    n_occ: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve H psi = eps psi for the lowest occupied states."""
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
    psi_occ = orbitals[:, :n_occ]

    e_kin = 0.0
    for i in range(n_occ):
        psi = psi_occ[:, i]
        t_psi = apply_tridiagonal(kinetic_diag, kinetic_offdiag, psi)
        e_kin += 2.0 * float(np.sum(psi * t_psi) * dx)
    return e_kin


def total_energy(
    density: np.ndarray,
    orbitals: np.ndarray,
    v_ext: np.ndarray,
    v_h: np.ndarray,
    kinetic_diag: np.ndarray,
    kinetic_offdiag: np.ndarray,
    dx: float,
    n_electrons: int,
) -> dict[str, float]:
    e_kin = kinetic_energy(orbitals, kinetic_diag, kinetic_offdiag, dx, n_electrons)
    e_ext = float(np.sum(v_ext * density) * dx)
    e_h = 0.5 * float(np.sum(v_h * density) * dx)
    e_x = float(np.sum(lda_exchange_energy_density(density)) * dx)
    e_c = float(np.sum(lda_correlation_energy_density(density)) * dx)
    e_xc = e_x + e_c
    e_total = e_kin + e_ext + e_h + e_xc

    return {
        "E_total": e_total,
        "E_kin": e_kin,
        "E_ext": e_ext,
        "E_H": e_h,
        "E_x": e_x,
        "E_c": e_c,
        "E_xc": e_xc,
    }


def run_scf_xc(grid: Grid1D, cfg: XCConfig) -> dict[str, object]:
    require_even_electrons(cfg.n_electrons)

    x = grid.x
    dx = grid.dx
    n_occ = cfg.n_electrons // 2

    v_ext = external_potential_harmonic(x, cfg.omega)
    kernel = build_hartree_kernel(x, cfg.soft_coulomb)
    kinetic_diag, kinetic_offdiag = build_kinetic_tridiagonal(grid.n_points, dx)

    density = np.exp(-0.75 * x**2)
    density = normalize_density(density, dx, float(cfg.n_electrons))

    history: list[dict[str, float]] = []
    e_prev = np.inf
    converged = False

    for it in range(1, cfg.max_iter + 1):
        v_h = hartree_potential(kernel, density, dx)
        v_x = lda_exchange_potential(density)
        v_c = lda_correlation_potential(density)
        v_eff = v_ext + v_h + v_x + v_c

        evals, evecs = solve_kohn_sham_orbitals(kinetic_diag, kinetic_offdiag, v_eff, n_occ)
        orbitals = normalize_orbitals(evecs, dx)
        density_out = density_from_orbitals(orbitals, dx, cfg.n_electrons)

        density_new = (1.0 - cfg.mix) * density + cfg.mix * density_out
        density_new = normalize_density(density_new, dx, float(cfg.n_electrons))

        v_h_new = hartree_potential(kernel, density_new, dx)
        energies = total_energy(
            density=density_new,
            orbitals=orbitals,
            v_ext=v_ext,
            v_h=v_h_new,
            kinetic_diag=kinetic_diag,
            kinetic_offdiag=kinetic_offdiag,
            dx=dx,
            n_electrons=cfg.n_electrons,
        )

        drho = float(np.sqrt(np.sum((density_new - density) ** 2) * dx))
        denergy = float(abs(energies["E_total"] - e_prev))

        history.append(
            {
                "iter": float(it),
                "E_total": energies["E_total"],
                "dE": denergy,
                "drho_L2": drho,
                "eps_min": float(evals[0]),
                "eps_max_occ": float(evals[n_occ - 1]),
                "N_integral": float(np.sum(density_new) * dx),
                "E_kin": energies["E_kin"],
                "E_ext": energies["E_ext"],
                "E_H": energies["E_H"],
                "E_x": energies["E_x"],
                "E_c": energies["E_c"],
                "E_xc": energies["E_xc"],
            }
        )

        if it > 1 and drho < cfg.tol_density and denergy < cfg.tol_energy:
            converged = True
            density = density_new
            break

        density = density_new
        e_prev = energies["E_total"]

    if not history:
        raise RuntimeError("SCF produced no iteration record")

    final = history[-1]
    return {
        "converged": converged,
        "iterations": len(history),
        "history": pd.DataFrame(history),
        "density": density,
        "x": x,
        "dx": dx,
        "final_metrics": final,
        "cfg": cfg,
    }


def main() -> None:
    grid = Grid1D(x_min=-8.0, x_max=8.0, n_points=260)
    cfg = XCConfig(
        n_electrons=4,
        omega=0.33,
        soft_coulomb=0.85,
        mix=0.26,
        max_iter=140,
        tol_density=6.0e-5,
        tol_energy=1.0e-6,
    )

    result = run_scf_xc(grid, cfg)
    hist = result["history"]
    final = result["final_metrics"]

    n_err = abs(final["N_integral"] - cfg.n_electrons)
    checks = {
        "SCF converged": bool(result["converged"]),
        "density residual < 1e-4": final["drho_L2"] < 1.0e-4,
        "electron-number error < 1e-6": n_err < 1.0e-6,
        "occupied-bandwidth > 1e-3": (final["eps_max_occ"] - final["eps_min"]) > 1.0e-3,
        "exchange energy is negative": final["E_x"] < 0.0,
        "correlation energy is negative": final["E_c"] < 0.0,
        "|E_xc| > |E_x|": abs(final["E_xc"]) > abs(final["E_x"]),
    }

    pd.set_option("display.float_format", lambda v: f"{v:.8f}")

    print("=== Exchange-Correlation Energy MVP | PHYS-0205 ===")
    print(
        f"grid_points={grid.n_points}, x in [{grid.x_min:.1f}, {grid.x_max:.1f}], "
        f"N_e={cfg.n_electrons}, mix={cfg.mix:.2f}"
    )

    print("\nLast SCF iterations:")
    show_cols = [
        "iter",
        "E_total",
        "dE",
        "drho_L2",
        "eps_min",
        "eps_max_occ",
        "E_x",
        "E_c",
        "E_xc",
    ]
    print(hist[show_cols].tail(8).to_string(index=False))

    print("\nFinal energy decomposition:")
    print(f"- E_total = {final['E_total']:.10f}")
    print(f"- E_kin   = {final['E_kin']:.10f}")
    print(f"- E_ext   = {final['E_ext']:.10f}")
    print(f"- E_H     = {final['E_H']:.10f}")
    print(f"- E_x     = {final['E_x']:.10f}")
    print(f"- E_c     = {final['E_c']:.10f}")
    print(f"- E_xc    = {final['E_xc']:.10f}")
    print(f"- N_int   = {final['N_integral']:.10f} (error={n_err:.3e})")

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
