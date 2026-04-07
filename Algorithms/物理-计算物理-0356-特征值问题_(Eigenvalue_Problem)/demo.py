"""Minimal runnable MVP for Eigenvalue Problem (PHYS-0340).

Model:
    1D quantum harmonic oscillator (dimensionless units)
    H psi = E psi
    H = -1/2 d^2/dx^2 + 1/2 x^2

We discretize H on [-L, L] with finite differences, obtaining a real
symmetric tridiagonal matrix. The lowest eigenvalues are computed by a
source-visible Lanczos implementation (with full re-orthogonalization),
then compared against:
1) analytic levels E_n = n + 1/2
2) scipy.linalg.eigh_tridiagonal reference on the same discrete model
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal


@dataclass(frozen=True)
class EigenConfig:
    """Configuration for the eigenvalue MVP."""

    n_points: int = 401
    half_domain: float = 8.0
    n_levels: int = 6
    lanczos_steps: int = 300
    seed: int = 7

    def validate(self) -> None:
        if self.n_points < 50:
            raise ValueError("n_points must be >= 50")
        if self.n_points % 2 == 0:
            raise ValueError("n_points must be odd for a symmetric grid around zero")
        if self.half_domain <= 0.0:
            raise ValueError("half_domain must be positive")
        if self.n_levels < 1:
            raise ValueError("n_levels must be >= 1")
        interior_size = self.n_points - 2
        if self.n_levels >= interior_size:
            raise ValueError("n_levels must be smaller than interior matrix size")
        if self.lanczos_steps <= self.n_levels:
            raise ValueError("lanczos_steps must be > n_levels")
        if self.lanczos_steps > interior_size:
            raise ValueError("lanczos_steps cannot exceed interior matrix size")


def build_harmonic_tridiagonal(
    cfg: EigenConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Construct tridiagonal Hamiltonian coefficients for the interior nodes."""
    x = np.linspace(-cfg.half_domain, cfg.half_domain, cfg.n_points, dtype=np.float64)
    dx = float(x[1] - x[0])
    x_inner = x[1:-1]

    diag = (1.0 / dx**2) + 0.5 * (x_inner**2)
    offdiag = np.full(x_inner.size - 1, -0.5 / dx**2, dtype=np.float64)
    return x, x_inner, diag, offdiag, dx


def tridiag_matvec(diag: np.ndarray, offdiag: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Matrix-vector product y = A v for symmetric tridiagonal A."""
    y = diag * vec
    y[:-1] += offdiag * vec[1:]
    y[1:] += offdiag * vec[:-1]
    return y


def lanczos_tridiagonalization(
    diag: np.ndarray,
    offdiag: np.ndarray,
    m_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Lanczos with full re-orthogonalization.

    Returns:
        alpha: diagonal of projected matrix T_m
        beta:  sub-diagonal of projected matrix T_m
        q_mat: Lanczos basis vectors (columns)
    """
    n = diag.size
    rng = np.random.default_rng(seed)

    q = rng.standard_normal(n)
    q /= np.linalg.norm(q)

    q_prev = np.zeros(n, dtype=np.float64)
    beta_prev = 0.0

    q_mat = np.zeros((n, m_steps), dtype=np.float64)
    alpha = np.zeros(m_steps, dtype=np.float64)
    beta = np.zeros(m_steps - 1, dtype=np.float64)

    actual_m = 0
    for j in range(m_steps):
        q_mat[:, j] = q
        z = tridiag_matvec(diag, offdiag, q)

        if j > 0:
            z -= beta_prev * q_prev

        alpha[j] = float(np.dot(q, z))
        z -= alpha[j] * q

        # Full re-orthogonalization to reduce loss of orthogonality.
        for i in range(j + 1):
            coeff = float(np.dot(q_mat[:, i], z))
            z -= coeff * q_mat[:, i]

        b = float(np.linalg.norm(z))
        actual_m = j + 1

        if j < m_steps - 1:
            beta[j] = b

        if b < 1e-14 or j == m_steps - 1:
            break

        q_prev = q
        q = z / b
        beta_prev = b

    alpha = alpha[:actual_m]
    q_mat = q_mat[:, :actual_m]
    beta = beta[: max(actual_m - 1, 0)]
    return alpha, beta, q_mat


def ritz_from_lanczos(
    alpha: np.ndarray,
    beta: np.ndarray,
    q_mat: np.ndarray,
    n_levels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lowest Ritz pairs from Lanczos projection."""
    m = alpha.size
    t_mat = np.diag(alpha)
    if beta.size > 0:
        t_mat += np.diag(beta, 1) + np.diag(beta, -1)

    evals_t, evecs_t = np.linalg.eigh(t_mat)
    idx = np.argsort(evals_t)[:n_levels]

    e_ritz = evals_t[idx]
    v_ritz = q_mat @ evecs_t[:, idx]

    norms = np.linalg.norm(v_ritz, axis=0)
    v_ritz = v_ritz / norms
    return e_ritz, v_ritz


def residual_norms(
    diag: np.ndarray,
    offdiag: np.ndarray,
    evals: np.ndarray,
    evecs: np.ndarray,
) -> np.ndarray:
    """Compute ||A v - lambda v||_2 for each eigenpair."""
    norms = np.zeros(evals.size, dtype=np.float64)
    for i in range(evals.size):
        v = evecs[:, i]
        r = tridiag_matvec(diag, offdiag, v) - evals[i] * v
        norms[i] = np.linalg.norm(r)
    return norms


def reference_with_scipy(diag: np.ndarray, offdiag: np.ndarray, n_levels: int) -> np.ndarray:
    """Reference lowest eigenvalues from SciPy tridiagonal solver."""
    ref, _ = eigh_tridiagonal(
        d=diag,
        e=offdiag,
        select="i",
        select_range=(0, n_levels - 1),
        check_finite=False,
    )
    return ref


def main() -> None:
    cfg = EigenConfig()
    cfg.validate()

    _, _, diag, offdiag, dx = build_harmonic_tridiagonal(cfg)

    alpha, beta, q_mat = lanczos_tridiagonalization(
        diag=diag,
        offdiag=offdiag,
        m_steps=cfg.lanczos_steps,
        seed=cfg.seed,
    )

    evals_lanczos, evecs_lanczos = ritz_from_lanczos(
        alpha=alpha,
        beta=beta,
        q_mat=q_mat,
        n_levels=cfg.n_levels,
    )
    residuals = residual_norms(diag, offdiag, evals_lanczos, evecs_lanczos)

    evals_ref = reference_with_scipy(diag, offdiag, cfg.n_levels)
    n = np.arange(cfg.n_levels, dtype=np.int64)
    evals_exact = n + 0.5

    df = pd.DataFrame(
        {
            "n": n,
            "E_exact": evals_exact,
            "E_lanczos": evals_lanczos,
            "E_scipy_ref": evals_ref,
            "|lanczos-exact|": np.abs(evals_lanczos - evals_exact),
            "|lanczos-scipy|": np.abs(evals_lanczos - evals_ref),
            "residual_2norm": residuals,
        }
    )

    print("=== Eigenvalue Problem MVP (1D Harmonic Oscillator) ===")
    print(
        f"n_points={cfg.n_points}, interior={diag.size}, dx={dx:.6f}, "
        f"half_domain={cfg.half_domain}, lanczos_steps={alpha.size}, n_levels={cfg.n_levels}"
    )
    print(df.to_string(index=False, float_format=lambda x: f"{x:.8e}"))

    max_residual = float(df["residual_2norm"].max())
    max_gap_ref = float(df["|lanczos-scipy|"].max())
    max_err_exact = float(df["|lanczos-exact|"].max())
    strictly_increasing = bool(np.all(np.diff(evals_lanczos) > 0.0))

    assert max_residual < 1e-5, f"Residual too large: {max_residual:.3e}"
    assert max_gap_ref < 1e-6, f"Lanczos vs SciPy mismatch too large: {max_gap_ref:.3e}"
    assert max_err_exact < 5e-2, f"Exact error too large: {max_err_exact:.3e}"
    assert strictly_increasing, "Eigenvalues are not strictly increasing"

    print("Validation: PASS")


if __name__ == "__main__":
    main()
