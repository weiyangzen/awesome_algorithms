"""Minimal runnable MVP for Hybrid Functionals (PHYS-0208).

This script implements a toy 1D closed-shell Kohn-Sham SCF loop with
hybrid exchange:

    E_x^hyb = alpha * E_x^HF + (1-alpha) * E_x^LDA

The goal is algorithm transparency instead of production-grade DFT accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh


@dataclass(frozen=True)
class Grid1D:
    """Uniform 1D grid with Dirichlet boundaries at both ends."""

    x_min: float
    x_max: float
    n_points: int

    @property
    def x_full(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.n_points)

    @property
    def x_interior(self) -> np.ndarray:
        return self.x_full[1:-1]

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.n_points - 1)


@dataclass(frozen=True)
class HybridConfig:
    """Configuration for toy hybrid-functional SCF."""

    n_electrons: int = 4
    omega: float = 0.32
    soft_coulomb: float = 0.75
    alpha_hf: float = 0.25
    mix: float = 0.22
    max_iter: int = 140
    tol_density: float = 6e-5
    tol_energy: float = 8e-7


def require_even_electrons(n_electrons: int) -> None:
    if n_electrons <= 0:
        raise ValueError("n_electrons must be positive")
    if n_electrons % 2 != 0:
        raise ValueError("This MVP assumes closed-shell spin-unpolarized states, so n_electrons must be even")


def build_kinetic_matrix(n_points: int, dx: float) -> np.ndarray:
    """Finite-difference kinetic matrix for T = -0.5 * d^2/dx^2 on interior points."""
    diag = np.full(n_points, 1.0 / dx**2, dtype=float)
    off = np.full(n_points - 1, -0.5 / dx**2, dtype=float)
    return np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)


def normalize_orbitals(orbitals: np.ndarray, dx: float) -> np.ndarray:
    """Normalize each orbital with continuous norm int |psi|^2 dx = 1."""
    psi = np.array(orbitals, dtype=float, copy=True)
    norms = np.sqrt(np.sum(psi**2, axis=0) * dx)
    if np.any(norms <= 0.0) or np.any(~np.isfinite(norms)):
        raise ValueError("Orbital norm is non-positive or non-finite")
    psi /= norms

    # Deterministic phase convention for reproducible reporting.
    peak_rows = np.argmax(np.abs(psi), axis=0)
    signs = np.sign(psi[peak_rows, np.arange(psi.shape[1])])
    signs[signs == 0.0] = 1.0
    psi *= signs
    return psi


def normalize_density(density: np.ndarray, dx: float, n_electrons: float) -> np.ndarray:
    integral = float(np.sum(density) * dx)
    if integral <= 0.0 or not np.isfinite(integral):
        raise ValueError("Density integral must stay positive and finite")
    return density * (n_electrons / integral)


def density_from_orbitals(psi_occ: np.ndarray, dx: float, n_electrons: int) -> np.ndarray:
    density = 2.0 * np.sum(psi_occ**2, axis=1)
    return normalize_density(density, dx, float(n_electrons))


def external_potential_harmonic(x: np.ndarray, omega: float) -> np.ndarray:
    return 0.5 * (omega**2) * x**2


def build_soft_coulomb_kernel(x: np.ndarray, soft_coulomb: float) -> np.ndarray:
    dx = x[:, None] - x[None, :]
    return 1.0 / np.sqrt(dx**2 + soft_coulomb**2)


def hartree_potential(kernel: np.ndarray, density: np.ndarray, dx: float) -> np.ndarray:
    return dx * (kernel @ density)


def lda_exchange_energy_density(density: np.ndarray) -> np.ndarray:
    """3D-LDA exchange energy density term e_x(n)*n = Cx*n^(4/3)."""
    c_x = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    n_safe = np.clip(density, 1e-14, None)
    return c_x * n_safe ** (4.0 / 3.0)


def lda_exchange_potential(density: np.ndarray) -> np.ndarray:
    """v_x^LDA(n) = -(3/pi)^(1/3) * n^(1/3)."""
    pref = (3.0 / np.pi) ** (1.0 / 3.0)
    n_safe = np.clip(density, 1e-14, None)
    return -pref * n_safe ** (1.0 / 3.0)


def exact_exchange_matrix(kernel: np.ndarray, psi_occ: np.ndarray, dx: float) -> np.ndarray:
    """Build discrete nonlocal exact-exchange operator matrix.

    F_x[p,q] = -dx * sum_i psi_i[p] psi_i[q] * kernel[p,q]
    """
    density_matrix = psi_occ @ psi_occ.T
    return -dx * kernel * density_matrix


def kinetic_energy(psi_occ: np.ndarray, kinetic: np.ndarray, dx: float) -> float:
    t_psi = kinetic @ psi_occ
    return 2.0 * float(np.sum(psi_occ * t_psi) * dx)


def exact_exchange_energy(psi_occ: np.ndarray, fock_x: np.ndarray, dx: float) -> float:
    fx_psi = fock_x @ psi_occ
    return float(np.sum(psi_occ * fx_psi) * dx)


def energy_components(
    density: np.ndarray,
    psi_occ: np.ndarray,
    v_ext: np.ndarray,
    v_h: np.ndarray,
    kinetic: np.ndarray,
    fock_x: np.ndarray,
    alpha_hf: float,
    dx: float,
) -> dict[str, float]:
    e_kin = kinetic_energy(psi_occ, kinetic, dx)
    e_ext = float(np.sum(v_ext * density) * dx)
    e_h = 0.5 * float(np.sum(v_h * density) * dx)
    e_x_lda = float(np.sum(lda_exchange_energy_density(density)) * dx)
    e_x_hf = exact_exchange_energy(psi_occ, fock_x, dx)
    e_x_hyb = alpha_hf * e_x_hf + (1.0 - alpha_hf) * e_x_lda
    e_total = e_kin + e_ext + e_h + e_x_hyb
    return {
        "E_total": e_total,
        "E_kin": e_kin,
        "E_ext": e_ext,
        "E_H": e_h,
        "E_x_hf": e_x_hf,
        "E_x_lda": e_x_lda,
        "E_x_hyb": e_x_hyb,
    }


def solve_lowest_orbitals(hamiltonian: np.ndarray, n_occ: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = eigh(hamiltonian, check_finite=False, driver="evd")
    return eigvals[:n_occ], normalize_orbitals(eigvecs[:, :n_occ], dx)


def run_scf_hybrid(grid: Grid1D, cfg: HybridConfig) -> dict[str, object]:
    require_even_electrons(cfg.n_electrons)
    if not (0.0 <= cfg.alpha_hf <= 1.0):
        raise ValueError("alpha_hf must be in [0, 1]")

    x = grid.x_interior
    dx = grid.dx
    n_grid = x.size
    n_occ = cfg.n_electrons // 2

    kinetic = build_kinetic_matrix(n_grid, dx)
    kernel = build_soft_coulomb_kernel(x, cfg.soft_coulomb)
    v_ext = external_potential_harmonic(x, cfg.omega)

    density = np.exp(-0.75 * x**2)
    density = normalize_density(density, dx, float(cfg.n_electrons))

    v_h0 = hartree_potential(kernel, density, dx)
    v_x0 = lda_exchange_potential(density)
    h0 = kinetic + np.diag(v_ext + v_h0 + (1.0 - cfg.alpha_hf) * v_x0)
    _, psi_occ_prev = solve_lowest_orbitals(h0, n_occ, dx)

    e_prev = np.inf
    converged = False
    history: list[dict[str, float]] = []

    for it in range(1, cfg.max_iter + 1):
        v_h = hartree_potential(kernel, density, dx)
        v_x_lda = lda_exchange_potential(density)
        fock_x = exact_exchange_matrix(kernel, psi_occ_prev, dx)

        h_mat = kinetic + np.diag(v_ext + v_h + (1.0 - cfg.alpha_hf) * v_x_lda) + cfg.alpha_hf * fock_x
        evals_occ, psi_occ = solve_lowest_orbitals(h_mat, n_occ, dx)

        density_out = density_from_orbitals(psi_occ, dx, cfg.n_electrons)
        density_new = (1.0 - cfg.mix) * density + cfg.mix * density_out
        density_new = normalize_density(density_new, dx, float(cfg.n_electrons))

        v_h_new = hartree_potential(kernel, density_new, dx)
        fock_x_new = exact_exchange_matrix(kernel, psi_occ, dx)
        e_comp = energy_components(
            density=density_new,
            psi_occ=psi_occ,
            v_ext=v_ext,
            v_h=v_h_new,
            kinetic=kinetic,
            fock_x=fock_x_new,
            alpha_hf=cfg.alpha_hf,
            dx=dx,
        )

        drho = float(np.sqrt(np.sum((density_new - density) ** 2) * dx))
        d_energy = float(abs(e_comp["E_total"] - e_prev))

        history.append(
            {
                "iter": float(it),
                "E_total": e_comp["E_total"],
                "dE": d_energy,
                "drho_L2": drho,
                "eps_min": float(evals_occ[0]),
                "eps_max_occ": float(evals_occ[-1]),
                "N_integral": float(np.sum(density_new) * dx),
                "E_kin": e_comp["E_kin"],
                "E_ext": e_comp["E_ext"],
                "E_H": e_comp["E_H"],
                "E_x_hf": e_comp["E_x_hf"],
                "E_x_lda": e_comp["E_x_lda"],
                "E_x_hyb": e_comp["E_x_hyb"],
            }
        )

        if drho < cfg.tol_density and d_energy < cfg.tol_energy:
            converged = True
            density = density_new
            psi_occ_prev = psi_occ
            break

        density = density_new
        psi_occ_prev = psi_occ
        e_prev = e_comp["E_total"]

    if not history:
        raise RuntimeError("SCF failed before producing iteration records")

    final_metrics = history[-1]
    return {
        "converged": converged,
        "iterations": len(history),
        "history": pd.DataFrame(history),
        "final_metrics": final_metrics,
        "density": density,
        "x": x,
        "dx": dx,
        "cfg": cfg,
    }


def main() -> None:
    grid = Grid1D(x_min=-7.0, x_max=7.0, n_points=190)
    cfg = HybridConfig(
        n_electrons=4,
        omega=0.32,
        soft_coulomb=0.75,
        alpha_hf=0.25,
        mix=0.22,
        max_iter=140,
        tol_density=6e-5,
        tol_energy=8e-7,
    )

    result = run_scf_hybrid(grid, cfg)
    history = result["history"]
    final = result["final_metrics"]

    n_error = abs(final["N_integral"] - cfg.n_electrons)
    checks = {
        "SCF converged": bool(result["converged"]),
        "density residual < 1e-4": final["drho_L2"] < 1e-4,
        "energy change < 1e-5": final["dE"] < 1e-5,
        "electron-number error < 1e-6": n_error < 1e-6,
        "occupied-bandwidth > 1e-3": (final["eps_max_occ"] - final["eps_min"]) > 1e-3,
        "hybrid exchange is negative": final["E_x_hyb"] < 0.0,
    }

    pd.set_option("display.float_format", lambda v: f"{v:.8f}")

    print("=== Hybrid Functionals MVP | PHYS-0208 ===")
    print(
        "grid: "
        f"[{grid.x_min}, {grid.x_max}], n_points={grid.n_points}, interior={grid.n_points - 2}, dx={grid.dx:.6f}; "
        f"N_e={cfg.n_electrons}"
    )
    print(
        "model: "
        f"v_ext=harmonic(omega={cfg.omega}), soft_coulomb={cfg.soft_coulomb}, alpha_hf={cfg.alpha_hf}"
    )
    print(
        "SCF: "
        f"mix={cfg.mix}, max_iter={cfg.max_iter}, "
        f"tol_density={cfg.tol_density:.1e}, tol_energy={cfg.tol_energy:.1e}"
    )

    print("\nIteration tail (last 8):")
    print(history.tail(8).to_string(index=False))

    summary = pd.DataFrame(
        {
            "quantity": [
                "iterations",
                "converged (0/1)",
                "E_total",
                "E_kin",
                "E_ext",
                "E_H",
                "E_x_hf",
                "E_x_lda",
                "E_x_hyb",
                "drho_L2",
                "dE",
                "eps_min",
                "eps_max_occ",
                "N_integral",
                "N_error",
            ],
            "value": [
                float(result["iterations"]),
                float(1 if result["converged"] else 0),
                final["E_total"],
                final["E_kin"],
                final["E_ext"],
                final["E_H"],
                final["E_x_hf"],
                final["E_x_lda"],
                final["E_x_hyb"],
                final["drho_L2"],
                final["dE"],
                final["eps_min"],
                final["eps_max_occ"],
                final["N_integral"],
                n_error,
            ],
        }
    )

    print("\nFinal summary:")
    print(summary.to_string(index=False))

    print("\nThreshold checks:")
    for key, ok in checks.items():
        print(f"- {key}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
