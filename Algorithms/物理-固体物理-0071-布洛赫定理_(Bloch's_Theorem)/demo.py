"""Minimal runnable MVP for Bloch's Theorem in 1D periodic potential.

Model:
- Potential V(x) = V0 cos(Gx), with G = 2*pi/a
- Plane-wave basis |k + mG>, m in [-M, M]
- Hamiltonian elements:
    H_mn(k) = (k + mG)^2 delta_mn + (V0/2)(delta_{m,n+1}+delta_{m,n-1})

The script demonstrates Bloch's theorem computationally by:
1) solving band energies E_n(k) over the first Brillouin zone,
2) reconstructing u_{n,k}(x) and checking u(x+a)=u(x),
3) checking E_n(k)=E_n(-k) symmetry,
4) cross-validating band-bottom curvature with sklearn/scipy/torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class BlochConfig:
    a: float = 1.0
    v0: float = 1.2
    m_max: int = 5
    n_k: int = 161  # keep odd so k=0 is included exactly
    num_bands: int = 4
    n_x_periodicity: int = 257
    fit_k_window: float = 0.35
    symmetry_tol: float = 1.0e-10
    periodicity_tol: float = 1.0e-10
    min_gap_tol: float = 1.0e-3
    curvature_rel_tol: float = 6.0e-2
    torch_steps: int = 1600
    torch_lr: float = 0.05
    random_seed: int = 17


def reciprocal_G(cfg: BlochConfig) -> float:
    return 2.0 * pi / cfg.a


def plane_wave_indices(cfg: BlochConfig) -> np.ndarray:
    return np.arange(-cfg.m_max, cfg.m_max + 1, dtype=int)


def build_hamiltonian(k: float, cfg: BlochConfig, m_vals: np.ndarray) -> np.ndarray:
    """Build real-symmetric plane-wave Hamiltonian at given k."""
    g = reciprocal_G(cfg)
    d = m_vals.size
    h = np.zeros((d, d), dtype=float)

    # Diagonal kinetic term
    diag = (k + g * m_vals) ** 2
    h[np.diag_indices(d)] = diag

    # Cosine potential couples neighboring m by +/-1 in index order
    coupling = 0.5 * cfg.v0
    i = np.arange(d - 1)
    h[i, i + 1] = coupling
    h[i + 1, i] = coupling
    return h


def solve_bands(cfg: BlochConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve low-energy bands and corresponding eigenvectors on k-grid."""
    m_vals = plane_wave_indices(cfg)
    k_max = pi / cfg.a
    k_grid = np.linspace(-k_max, k_max, cfg.n_k)

    bands = np.empty((cfg.n_k, cfg.num_bands), dtype=float)
    eigvecs = np.empty((cfg.n_k, cfg.num_bands, m_vals.size), dtype=complex)

    for i, k in enumerate(k_grid):
        h = build_hamiltonian(float(k), cfg, m_vals)
        w, v = eigh(h, check_finite=True)
        bands[i, :] = w[: cfg.num_bands]
        eigvecs[i, :, :] = v[:, : cfg.num_bands].T

    return k_grid, bands, eigvecs


def compute_band_gaps(bands: np.ndarray) -> pd.DataFrame:
    """Compute indirect gap estimates between adjacent bands."""
    rows: list[dict[str, float | int]] = []
    n_bands = bands.shape[1]
    for n in range(n_bands - 1):
        lower_max = float(np.max(bands[:, n]))
        upper_min = float(np.min(bands[:, n + 1]))
        gap = upper_min - lower_max
        rows.append(
            {
                "lower_band": n,
                "upper_band": n + 1,
                "lower_max": lower_max,
                "upper_min": upper_min,
                "gap": float(gap),
            }
        )
    return pd.DataFrame(rows)


def bloch_periodicity_error(
    k: float,
    coeffs: np.ndarray,
    cfg: BlochConfig,
) -> float:
    """Check max |u(x+a)-u(x)| for reconstructed periodic part u_{n,k}(x)."""
    g = reciprocal_G(cfg)
    m_vals = plane_wave_indices(cfg)

    x = np.linspace(0.0, cfg.a, cfg.n_x_periodicity, endpoint=False)
    phase_mx = np.exp(1j * np.outer(m_vals * g, x))
    u_x = coeffs @ phase_mx

    x_shift = x + cfg.a
    phase_mx_shift = np.exp(1j * np.outer(m_vals * g, x_shift))
    u_shift = coeffs @ phase_mx_shift

    err = np.max(np.abs(u_shift - u_x))
    return float(err)


def symmetry_error_k_to_minus_k(k_grid: np.ndarray, bands: np.ndarray) -> float:
    """Compute max |E_n(k)-E_n(-k)| over sampled grid."""
    rev = bands[::-1, :]
    return float(np.max(np.abs(bands - rev)))


def fit_curvature_sklearn(k: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    """Fit E(k)=e0+alpha*k^2 using sklearn LinearRegression."""
    x = (k * k).reshape(-1, 1)
    model = LinearRegression(fit_intercept=True)
    model.fit(x, e)
    alpha = float(model.coef_[0])
    e0 = float(model.intercept_)
    return e0, alpha


def _quad_model(k: np.ndarray, e0: float, alpha: float) -> np.ndarray:
    return e0 + alpha * k * k


def fit_curvature_scipy(k: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    popt, _ = curve_fit(_quad_model, k, e, p0=np.array([float(np.min(e)), 1.0]))
    return float(popt[0]), float(popt[1])


def fit_curvature_torch(k: np.ndarray, e: np.ndarray, cfg: BlochConfig) -> tuple[float, float] | None:
    if torch is None:
        return None

    torch.manual_seed(cfg.random_seed)
    kt = torch.tensor(k, dtype=torch.float64)
    et = torch.tensor(e, dtype=torch.float64)

    e0 = torch.nn.Parameter(torch.tensor(float(np.min(e)), dtype=torch.float64))
    alpha = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    opt = torch.optim.Adam([e0, alpha], lr=cfg.torch_lr)

    for _ in range(cfg.torch_steps):
        opt.zero_grad()
        pred = e0 + alpha * kt * kt
        loss = torch.mean((pred - et) ** 2)
        loss.backward()
        opt.step()

    return float(e0.detach().cpu().item()), float(alpha.detach().cpu().item())


def select_key_k_rows(k_grid: np.ndarray, bands: np.ndarray) -> pd.DataFrame:
    """Provide compact table at representative k points."""
    targets = np.array([-pi, -pi / 2.0, 0.0, pi / 2.0, pi])
    scaled = k_grid * 1.0  # a=1 by default; retain explicit array for clarity
    idx = [int(np.argmin(np.abs(scaled - t))) for t in targets]

    rows = []
    for i in idx:
        row = {
            "k": float(k_grid[i]),
        }
        for b in range(bands.shape[1]):
            row[f"E{b}"] = float(bands[i, b])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = BlochConfig()
    np.random.seed(cfg.random_seed)

    k_grid, bands, eigvecs = solve_bands(cfg)

    # Bloch-periodicity check on a representative interior k point, lowest band.
    i_mid_right = int(3 * (cfg.n_k - 1) // 4)
    k_probe = float(k_grid[i_mid_right])
    coeff_probe = eigvecs[i_mid_right, 0, :]
    periodicity_err = bloch_periodicity_error(k_probe, coeff_probe, cfg)

    # Symmetry and gap checks
    k_sym_err = symmetry_error_k_to_minus_k(k_grid, bands)
    df_gaps = compute_band_gaps(bands)
    first_gap = float(df_gaps.iloc[0]["gap"])

    # Curvature fit around k=0 for lowest band
    mask = np.abs(k_grid) <= cfg.fit_k_window
    k_fit = k_grid[mask]
    e_fit = bands[mask, 0]

    e0_sk, alpha_sk = fit_curvature_sklearn(k_fit, e_fit)
    e0_sp, alpha_sp = fit_curvature_scipy(k_fit, e_fit)
    torch_fit = fit_curvature_torch(k_fit, e_fit, cfg)

    alpha_ref = alpha_sk
    rel_sp = abs(alpha_sp - alpha_ref) / max(abs(alpha_ref), 1e-12)

    rel_torch = None
    e0_torch = None
    alpha_torch = None
    if torch_fit is not None:
        e0_torch, alpha_torch = torch_fit
        rel_torch = abs(alpha_torch - alpha_ref) / max(abs(alpha_ref), 1e-12)

    # Summaries
    df_key = select_key_k_rows(k_grid, bands)

    print("=== Bloch Theorem MVP (1D cosine periodic potential) ===")
    print(f"a={cfg.a:.3f}, V0={cfg.v0:.3f}, m_max={cfg.m_max}, n_k={cfg.n_k}, num_bands={cfg.num_bands}")
    print("\nRepresentative band energies:")
    with pd.option_context("display.max_rows", 20, "display.width", 140):
        print(df_key.to_string(index=False, float_format=lambda x: f"{x:10.6f}"))

    print("\nAdjacent band gaps:")
    with pd.option_context("display.max_rows", 20, "display.width", 140):
        print(df_gaps.to_string(index=False, float_format=lambda x: f"{x:10.6f}"))

    print("\nChecks:")
    print(f"- max |u(x+a)-u(x)| at probe k={k_probe:.6f}: {periodicity_err:.3e}")
    print(f"- max |E(k)-E(-k)| over grid: {k_sym_err:.3e}")
    print(f"- first gap (band 0->1): {first_gap:.6f}")
    print("- curvature fits for lowest band near k=0:")
    print(f"  sklearn: e0={e0_sk:.6f}, alpha={alpha_sk:.6f}")
    print(f"  scipy  : e0={e0_sp:.6f}, alpha={alpha_sp:.6f}, rel_diff={rel_sp:.3e}")
    if torch_fit is not None:
        print(
            "  torch  : "
            f"e0={e0_torch:.6f}, alpha={alpha_torch:.6f}, rel_diff={rel_torch:.3e}"
        )
    else:
        print("  torch  : not available, skipped")

    # Assertions for automated validation
    assert periodicity_err < cfg.periodicity_tol, (
        f"Bloch periodicity check failed: {periodicity_err:.3e} >= {cfg.periodicity_tol:.3e}"
    )
    assert k_sym_err < cfg.symmetry_tol, f"E(k)=E(-k) symmetry failed: {k_sym_err:.3e}"
    assert first_gap > cfg.min_gap_tol, (
        f"First band gap too small: {first_gap:.6f} <= {cfg.min_gap_tol:.6f}"
    )
    assert rel_sp < cfg.curvature_rel_tol, (
        f"scipy/sklearn curvature mismatch too large: {rel_sp:.3e}"
    )
    if rel_torch is not None:
        assert rel_torch < cfg.curvature_rel_tol, (
            f"torch/sklearn curvature mismatch too large: {rel_torch:.3e}"
        )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
