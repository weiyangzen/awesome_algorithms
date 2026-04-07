"""Minimal runnable MVP for the 1D Quantum Harmonic Oscillator (PHYS-0026)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal
from scipy.special import eval_hermite, factorial


EPS = 1e-12


@dataclass(frozen=True)
class QHOConfig:
    """Configuration for the 1D harmonic oscillator demo."""

    mass: float = 1.0
    omega: float = 1.0
    hbar: float = 1.0
    x_max: float = 8.0
    n_points: int = 1601
    n_states: int = 6


def build_uniform_grid(x_max: float, n_points: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Build symmetric position grid and interior nodes (Dirichlet boundaries)."""
    if x_max <= 0.0:
        raise ValueError("x_max must be positive")
    if n_points < 7:
        raise ValueError("n_points must be >= 7")
    if n_points % 2 == 0:
        raise ValueError("n_points must be odd so x=0 lies on the grid")

    x = np.linspace(-x_max, x_max, n_points)
    dx = float(x[1] - x[0])
    x_inner = x[1:-1]
    return x, x_inner, dx


def build_hamiltonian_tridiagonal(
    x_inner: np.ndarray,
    dx: float,
    mass: float,
    omega: float,
    hbar: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct tridiagonal Hamiltonian for

    H = -(hbar^2/2m) d^2/dx^2 + 0.5 m omega^2 x^2
    with central finite difference and Dirichlet boundaries psi=0 at edges.
    """
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if hbar <= 0.0:
        raise ValueError("hbar must be positive")

    kinetic_main = np.full(x_inner.size, hbar**2 / (mass * dx**2), dtype=float)
    kinetic_off = np.full(x_inner.size - 1, -(hbar**2) / (2.0 * mass * dx**2), dtype=float)
    potential = 0.5 * mass * omega**2 * x_inner**2

    main_diag = kinetic_main + potential
    off_diag = kinetic_off
    return main_diag, off_diag


def solve_lowest_states(
    main_diag: np.ndarray,
    off_diag: np.ndarray,
    n_states: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve lowest n_states eigenpairs of symmetric tridiagonal Hamiltonian."""
    n = main_diag.size
    if n_states <= 0:
        raise ValueError("n_states must be positive")
    if n_states > n:
        raise ValueError("n_states cannot exceed interior grid size")

    eigenvalues, eigenvectors = eigh_tridiagonal(
        d=main_diag,
        e=off_diag,
        select="i",
        select_range=(0, n_states - 1),
        check_finite=True,
    )
    return eigenvalues, eigenvectors


def normalize_on_grid(wavefunctions: np.ndarray, dx: float) -> np.ndarray:
    """Normalize columns so sum |psi|^2 dx = 1."""
    psi = wavefunctions / np.sqrt(dx)
    norms = np.sqrt(np.sum(np.abs(psi) ** 2, axis=0) * dx)
    psi = psi / norms
    return psi


def apply_tridiagonal(
    main_diag: np.ndarray,
    off_diag: np.ndarray,
    vec: np.ndarray,
) -> np.ndarray:
    """Compute y = H vec for tridiagonal H without forming a dense matrix."""
    out = main_diag * vec
    out[:-1] += off_diag * vec[1:]
    out[1:] += off_diag * vec[:-1]
    return out


def residual_norms(
    main_diag: np.ndarray,
    off_diag: np.ndarray,
    energies: np.ndarray,
    psi_inner: np.ndarray,
    dx: float,
) -> np.ndarray:
    """Return ||H psi_n - E_n psi_n||_2 on the discrete grid."""
    residuals = []
    for n in range(psi_inner.shape[1]):
        hpsi = apply_tridiagonal(main_diag, off_diag, psi_inner[:, n])
        r = hpsi - energies[n] * psi_inner[:, n]
        residuals.append(float(np.sqrt(np.sum(np.abs(r) ** 2) * dx)))
    return np.array(residuals)


def extend_with_boundaries(psi_inner: np.ndarray, n_points: int) -> np.ndarray:
    """Pad interior wavefunctions with zero Dirichlet boundaries."""
    psi_full = np.zeros((n_points, psi_inner.shape[1]), dtype=float)
    psi_full[1:-1, :] = psi_inner
    return psi_full


def analytic_energy(n: int, hbar: float, omega: float) -> float:
    """Exact E_n = hbar * omega * (n + 1/2)."""
    return hbar * omega * (n + 0.5)


def analytic_wavefunction(
    n: int,
    x: np.ndarray,
    mass: float,
    omega: float,
    hbar: float,
) -> np.ndarray:
    """Exact normalized 1D harmonic oscillator eigenfunction psi_n(x)."""
    xi = np.sqrt(mass * omega / hbar) * x
    norm = (mass * omega / (np.pi * hbar)) ** 0.25
    norm /= np.sqrt((2.0**n) * factorial(n, exact=False))
    return norm * np.exp(-0.5 * xi**2) * eval_hermite(n, xi)


def align_phase(reference: np.ndarray, target: np.ndarray, dx: float) -> np.ndarray:
    """Flip sign of target if needed so overlap with reference is non-negative."""
    overlap = float(np.sum(reference * target) * dx)
    return target if overlap >= 0.0 else -target


def gram_metrics(psi_inner: np.ndarray, dx: float) -> tuple[np.ndarray, float, float]:
    """Compute Gram matrix and orthogonality errors."""
    gram = psi_inner.T @ psi_inner * dx
    offdiag = gram.copy()
    np.fill_diagonal(offdiag, 0.0)
    max_offdiag = float(np.max(np.abs(offdiag)))
    max_diag_err = float(np.max(np.abs(np.diag(gram) - 1.0)))
    return gram, max_offdiag, max_diag_err


def expectation_x2(psi_full: np.ndarray, x: np.ndarray, dx: float) -> np.ndarray:
    """Compute <x^2> for each state."""
    x2 = x**2
    return np.sum((psi_full**2) * x2[:, None], axis=0) * dx


def main() -> None:
    config = QHOConfig()

    x, x_inner, dx = build_uniform_grid(config.x_max, config.n_points)
    main_diag, off_diag = build_hamiltonian_tridiagonal(
        x_inner=x_inner,
        dx=dx,
        mass=config.mass,
        omega=config.omega,
        hbar=config.hbar,
    )

    energies_num, eigvecs = solve_lowest_states(
        main_diag=main_diag,
        off_diag=off_diag,
        n_states=config.n_states,
    )
    psi_inner = normalize_on_grid(eigvecs, dx)

    residuals = residual_norms(main_diag, off_diag, energies_num, psi_inner, dx)
    _, max_offdiag, max_diag_err = gram_metrics(psi_inner, dx)

    psi_full = extend_with_boundaries(psi_inner, config.n_points)

    rows = []
    psi_l2_errors = []
    for n in range(config.n_states):
        e_exact = analytic_energy(n, config.hbar, config.omega)

        psi_exact = analytic_wavefunction(
            n=n,
            x=x,
            mass=config.mass,
            omega=config.omega,
            hbar=config.hbar,
        )
        # Re-normalize on finite domain to remove tiny truncation mismatch.
        psi_exact /= np.sqrt(np.sum(psi_exact**2) * dx)

        psi_aligned = align_phase(psi_exact, psi_full[:, n], dx)
        psi_full[:, n] = psi_aligned

        l2_err = float(np.sqrt(np.sum((psi_aligned - psi_exact) ** 2) * dx))
        psi_l2_errors.append(l2_err)

        rows.append(
            {
                "n": n,
                "E_num": float(energies_num[n]),
                "E_exact": float(e_exact),
                "abs_err_E": float(abs(energies_num[n] - e_exact)),
                "rel_err_E": float(abs(energies_num[n] - e_exact) / e_exact),
                "residual_L2": float(residuals[n]),
                "psi_L2_error": l2_err,
            }
        )

    x2_num = expectation_x2(psi_full, x, dx)
    x2_exact = np.array(
        [
            (n + 0.5) * config.hbar / (config.mass * config.omega)
            for n in range(config.n_states)
        ],
        dtype=float,
    )

    for i, row in enumerate(rows):
        row["<x^2>_num"] = float(x2_num[i])
        row["<x^2>_exact"] = float(x2_exact[i])
        row["abs_err_<x^2>"] = float(abs(x2_num[i] - x2_exact[i]))

    result_table = pd.DataFrame(rows)

    checks = {
        "max rel energy error < 5e-3": float(result_table["rel_err_E"].max()) < 5e-3,
        "max residual L2 < 1e-9": float(result_table["residual_L2"].max()) < 1e-9,
        "orthogonality offdiag < 1e-10": max_offdiag < 1e-10,
        "orthogonality diag error < 1e-10": max_diag_err < 1e-10,
        "max wavefunction L2 error < 2e-2": max(psi_l2_errors) < 2e-2,
        "max <x^2> abs error < 5e-2": float(result_table["abs_err_<x^2>"].max()) < 5e-2,
    }

    pd.set_option("display.float_format", lambda value: f"{value:.6e}")

    print("=== Quantum Harmonic Oscillator MVP (PHYS-0026) ===")
    print(
        "Config: "
        f"m={config.mass}, omega={config.omega}, hbar={config.hbar}, "
        f"x_max={config.x_max}, n_points={config.n_points}, n_states={config.n_states}"
    )
    print(f"Grid spacing dx = {dx:.6e}")

    print("\nEigenpair summary:")
    print(result_table.to_string(index=False))

    print("\nOrthogonality metrics:")
    print(f"max |offdiag(psi^T psi * dx)| = {max_offdiag:.3e}")
    print(f"max |diag(psi^T psi * dx)-1| = {max_diag_err:.3e}")

    print("\nThreshold checks:")
    for item, ok in checks.items():
        print(f"- {item}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
