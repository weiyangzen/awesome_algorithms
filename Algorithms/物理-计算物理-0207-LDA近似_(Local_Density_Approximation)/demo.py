"""Minimal runnable MVP for LDA (Local Density Approximation), PHYS-0206."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal


@dataclass(frozen=True)
class Grid1D:
    """Uniform 1D real-space grid."""

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
class LDAConfig:
    """Configuration for a toy Kohn-Sham LDA calculation."""

    n_electrons: int = 4
    omega: float = 0.35
    soft_coulomb: float = 0.8
    mix: float = 0.28
    max_iter: int = 120
    tol_density: float = 5e-5
    tol_energy: float = 5e-7


def require_even_electrons(n_electrons: int) -> None:
    if n_electrons <= 0:
        raise ValueError("n_electrons must be positive")
    if n_electrons % 2 != 0:
        raise ValueError("This MVP assumes spin-unpolarized closed shell, so n_electrons must be even")


def build_kinetic_tridiagonal(n_points: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Return diagonal/off-diagonal of kinetic operator T = -0.5 d^2/dx^2."""
    d = np.full(n_points, 1.0 / dx**2, dtype=float)
    e = np.full(n_points - 1, -0.5 / dx**2, dtype=float)
    return d, e


def apply_tridiagonal(diag: np.ndarray, offdiag: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply a symmetric tridiagonal operator to a vector."""
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


def density_from_orbitals(orbitals: np.ndarray, dx: float, n_electrons: int) -> np.ndarray:
    """Construct spin-unpolarized density n(x) = 2 * sum_{occ} |psi_i|^2."""
    n_occ = n_electrons // 2
    psi_occ = orbitals[:, :n_occ]
    density = 2.0 * np.sum(psi_occ**2, axis=1)
    return normalize_density(density, dx, float(n_electrons))


def normalize_density(density: np.ndarray, dx: float, n_electrons: float) -> np.ndarray:
    total = float(np.sum(density) * dx)
    if total <= 0.0:
        raise ValueError("Density integral must stay positive")
    return density * (n_electrons / total)


def external_potential_harmonic(x: np.ndarray, omega: float) -> np.ndarray:
    return 0.5 * (omega**2) * x**2


def build_hartree_kernel(x: np.ndarray, soft_coulomb: float) -> np.ndarray:
    """Soft-Coulomb kernel K(x,x') = 1/sqrt((x-x')^2 + a^2)."""
    dx = x[:, None] - x[None, :]
    return 1.0 / np.sqrt(dx**2 + soft_coulomb**2)


def hartree_potential(kernel: np.ndarray, density: np.ndarray, dx: float) -> np.ndarray:
    return dx * (kernel @ density)


def lda_exchange_energy_density(density: np.ndarray) -> np.ndarray:
    """3D-LDA exchange epsilon_x(n) * n = Cx * n^(4/3)."""
    c_x = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    n_safe = np.clip(density, 1e-14, None)
    return c_x * n_safe ** (4.0 / 3.0)


def lda_exchange_potential(density: np.ndarray) -> np.ndarray:
    """Functional derivative v_x(n) = - (3/pi)^(1/3) * n^(1/3)."""
    pref = (3.0 / np.pi) ** (1.0 / 3.0)
    n_safe = np.clip(density, 1e-14, None)
    return -pref * n_safe ** (1.0 / 3.0)


def solve_kohn_sham_orbitals(
    kinetic_diag: np.ndarray,
    kinetic_offdiag: np.ndarray,
    v_eff: np.ndarray,
    n_occ: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve tridiagonal Kohn-Sham Hamiltonian and return (eigenvalues, eigenvectors)."""
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
    e_t = 0.0
    for i in range(n_occ):
        psi = psi_occ[:, i]
        t_psi = apply_tridiagonal(kinetic_diag, kinetic_offdiag, psi)
        e_t += 2.0 * float(np.sum(psi * t_psi) * dx)
    return e_t


def total_energy(
    density: np.ndarray,
    orbitals: np.ndarray,
    v_ext: np.ndarray,
    v_h: np.ndarray,
    kinetic_diag: np.ndarray,
    kinetic_offdiag: np.ndarray,
    dx: float,
    n_electrons: int,
) -> tuple[float, float, float, float]:
    ts = kinetic_energy(orbitals, kinetic_diag, kinetic_offdiag, dx, n_electrons)
    e_ext = float(np.sum(v_ext * density) * dx)
    e_h = 0.5 * float(np.sum(v_h * density) * dx)
    e_x = float(np.sum(lda_exchange_energy_density(density)) * dx)
    e_tot = ts + e_ext + e_h + e_x
    return e_tot, ts, e_ext + e_h, e_x


def run_scf_lda(grid: Grid1D, cfg: LDAConfig) -> dict[str, object]:
    require_even_electrons(cfg.n_electrons)

    x = grid.x
    dx = grid.dx
    n_occ = cfg.n_electrons // 2

    v_ext = external_potential_harmonic(x, cfg.omega)
    kernel = build_hartree_kernel(x, cfg.soft_coulomb)
    kinetic_diag, kinetic_offdiag = build_kinetic_tridiagonal(grid.n_points, dx)

    density = np.exp(-0.7 * x**2)
    density = normalize_density(density, dx, float(cfg.n_electrons))

    history: list[dict[str, float]] = []
    e_prev = np.inf
    converged = False

    for it in range(1, cfg.max_iter + 1):
        v_h = hartree_potential(kernel, density, dx)
        v_x = lda_exchange_potential(density)
        v_eff = v_ext + v_h + v_x

        evals, evecs = solve_kohn_sham_orbitals(kinetic_diag, kinetic_offdiag, v_eff, n_occ)
        orbitals = normalize_orbitals(evecs, dx)
        density_out = density_from_orbitals(orbitals, dx, cfg.n_electrons)

        density_new = (1.0 - cfg.mix) * density + cfg.mix * density_out
        density_new = normalize_density(density_new, dx, float(cfg.n_electrons))

        v_h_new = hartree_potential(kernel, density_new, dx)
        e_tot, e_kin, e_cls, e_x = total_energy(
            density_new,
            orbitals,
            v_ext,
            v_h_new,
            kinetic_diag,
            kinetic_offdiag,
            dx,
            cfg.n_electrons,
        )

        drho = float(np.sqrt(np.sum((density_new - density) ** 2) * dx))
        denergy = float(abs(e_tot - e_prev))

        history.append(
            {
                "iter": float(it),
                "E_total": e_tot,
                "dE": denergy,
                "drho_L2": drho,
                "eps_min": float(evals[0]),
                "eps_max_occ": float(evals[n_occ - 1]),
                "N_integral": float(np.sum(density_new) * dx),
                "E_kin": e_kin,
                "E_classical": e_cls,
                "E_x": e_x,
            }
        )

        if drho < cfg.tol_density and denergy < cfg.tol_energy:
            converged = True
            density = density_new
            break

        density = density_new
        e_prev = e_tot

    if not history:
        raise RuntimeError("SCF did not produce any iteration record")

    final = history[-1]
    return {
        "converged": converged,
        "iterations": len(history),
        "history": pd.DataFrame(history),
        "density": density,
        "x": x,
        "dx": dx,
        "v_ext": v_ext,
        "final_metrics": final,
        "cfg": cfg,
    }


def main() -> None:
    grid = Grid1D(x_min=-8.0, x_max=8.0, n_points=240)
    cfg = LDAConfig(
        n_electrons=4,
        omega=0.35,
        soft_coulomb=0.8,
        mix=0.28,
        max_iter=120,
        tol_density=5e-5,
        tol_energy=5e-7,
    )

    result = run_scf_lda(grid, cfg)
    hist = result["history"]
    final = result["final_metrics"]

    n_err = abs(final["N_integral"] - cfg.n_electrons)
    checks = {
        "SCF converged": bool(result["converged"]),
        "density residual < 1e-4": final["drho_L2"] < 1e-4,
        "electron-number error < 1e-6": n_err < 1e-6,
        "occupied-bandwidth > 1e-3": (final["eps_max_occ"] - final["eps_min"]) > 1e-3,
        "exchange energy is negative": final["E_x"] < 0.0,
    }

    pd.set_option("display.float_format", lambda v: f"{v:.8f}")

    print("=== LDA (Local Density Approximation) MVP | PHYS-0206 ===")
    print(
        "grid: "
        f"[{grid.x_min}, {grid.x_max}], n_points={grid.n_points}, dx={grid.dx:.6f}; "
        f"N_e={cfg.n_electrons}, omega={cfg.omega}, soft_coulomb={cfg.soft_coulomb}"
    )
    print(
        "SCF: "
        f"mix={cfg.mix}, max_iter={cfg.max_iter}, "
        f"tol_density={cfg.tol_density:.1e}, tol_energy={cfg.tol_energy:.1e}"
    )

    print("\nIteration tail (last 8):")
    print(hist.tail(8).to_string(index=False))

    summary = pd.DataFrame(
        {
            "quantity": [
                "iterations",
                "converged (0/1)",
                "E_total",
                "E_kin",
                "E_classical",
                "E_x",
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
                final["E_classical"],
                final["E_x"],
                final["drho_L2"],
                final["dE"],
                final["eps_min"],
                final["eps_max_occ"],
                final["N_integral"],
                n_err,
            ],
        }
    )

    print("\nFinal summary:")
    print(summary.to_string(index=False))

    print("\nThreshold checks:")
    for k, ok in checks.items():
        print(f"- {k}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
